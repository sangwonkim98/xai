#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
inference_ax_light.py

- 입력:  jsonl (각 줄: {"id": ..., "text": ..., "label": ...})
- 출력:  raw_inference.jsonl
        (각 줄: {"id", "text", "label", "pred_prob", "pred_label"})

A.X-4.0-Light을 "생성형 분류기"처럼 써서
문장이 유해/낚시성(1)인지, 비유해(0)인지 확률을 계산하는 스크립트.

아이디어:
    prompt 뒤에 나올 "다음 토큰"이 '0' 또는 '1'일 확률을 비교해서
    P(label=1 | text) 를 추정한다.
"""

import argparse
import json
import math
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# -----------------------------
# 1. 유틸: jsonl 읽기/쓰기
# -----------------------------

def read_jsonl(path):
    """jsonl 파일을 한 줄씩 읽어서 dict 리스트로 반환."""
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def write_jsonl(path, rows):
    """dict 리스트를 jsonl로 저장."""
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


# -----------------------------
# 2. 분류용 프롬프트 템플릿
# -----------------------------

def build_prompt(text: str) -> str:
    """
    A.X-4.0-Light에 전달할 프롬프트.

    핵심 규칙:
      - 출력은 반드시 0 또는 1만 나오도록 강하게 지시
      - 한국어 설명을 섞어줘서 role을 명확히 함
    """
    # 필요하면 여기 표현만 네 과제에 맞게 바꿔도 됨.
    prompt = (
        "당신은 뉴스/댓글/게시글의 유해성·낚시성을 판단하는 분류기입니다.\n"
        "다음 문장이 유해하거나 혐오/낚시성(클릭베이트)이라면 1, 그렇지 않다면 0만 출력하세요.\n"
        "추가 설명이나 다른 문자는 절대 쓰지 마세요.\n\n"
        f"[문장]\n{text}\n\n"
        "[정답] "
    )
    return prompt


# -----------------------------
# 3. A.X-4.0-Light 로딩
# -----------------------------

def load_model(model_name: str = "skt/A.X-4.0-Light", device: str = None):
    """
    HuggingFace에서 토크나이저와 모델을 로딩.
    - device가 None이면 가능하면 cuda, 아니면 cpu.
    """
    print(f"[INFO] Loading tokenizer & model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # dtype은 필요에 따라 bfloat16/float16로 바꿔도 됨
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )

    if device is None:
        device = "cpu"
    model.to(device)
    model.eval()

    print(f"[INFO] Model loaded on device: {device}")

    return tokenizer, model, device


# -----------------------------
# 4. 0/1 토큰 확률 계산 함수
# -----------------------------

def get_prob_1(
    text: str,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    device: str,
) -> float:
    """
    주어진 text에 대해

    P(label=1 | text) = softmax( logit('1'), logit('0') ) 중 '1'쪽

    을 계산해서 반환.

    - 0과 1은 vocab에서 각각 하나의 토큰이라고 가정하고,
      마지막 step의 logits에서 해당 토큰들의 값을 뽑아 비교한다.
    """
    prompt = build_prompt(text)

    # add_special_tokens=True: BOS/CLS 등 모델이 쓰는 스페셜 토큰 포함
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=True,
    ).to(device)

    # forward pass (생성 안 하고 logits만)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits  # shape: [batch, seq_len, vocab_size]

    # 마지막 토큰 위치의 logits 사용
    last_logits = logits[0, -1, :]  # [vocab_size]

    # '0', '1' 토큰 id 가져오기
    token_id_0 = tokenizer("0", add_special_tokens=False)["input_ids"][0]
    token_id_1 = tokenizer("1", add_special_tokens=False)["input_ids"][0]

    # 두 토큰에 대한 logit만 뽑아서 작은 softmax
    selected_logits = torch.stack(
        [last_logits[token_id_0], last_logits[token_id_1]],
        dim=0
    )  # [2]

    probs = torch.softmax(selected_logits, dim=0)  # [2]
    prob_0 = probs[0].item()
    prob_1 = probs[1].item()

    # 디버깅용으로 보고 싶으면 주석 해제
    # print(f"text[:30]={text[:30]}..., P0={prob_0:.3f}, P1={prob_1:.3f}")

    return float(prob_1)


# -----------------------------
# 5. 메인 인퍼런스 루프
# -----------------------------

def run_inference(
    input_path: Path,
    output_path: Path,
    model_name: str = "skt/A.X-4.0-Light",
    device: str = None,
    max_samples: int = None,
):
    """
    - input_s1.jsonl 을 읽어서
    - 각 문장에 대해 P(label=1)를 계산하고
    - raw_inference.jsonl로 저장.
    """
    # 1) 데이터 로드
    print(f"[INFO] Loading input from: {input_path}")
    data = read_jsonl(input_path)
    if max_samples is not None:
        data = data[:max_samples]
        print(f"[INFO] Truncated to {len(data)} samples (max_samples={max_samples})")
    else:
        print(f"[INFO] Total {len(data)} samples loaded.")

    # 2) 모델 로딩
    tokenizer, model, device = load_model(model_name, device)

    results = []
    for idx, row in enumerate(data, start=1):
        text = row["text"]
        label = row.get("label", None)

        # 3) 확률 계산
        try:
            prob_1 = get_prob_1(text, tokenizer, model, device)
        except Exception as e:
            print(f"[WARN] Failed on id={row.get('id')} ({e}), setting prob=0.5")
            prob_1 = 0.5  # 실패 시 중립값

        # 4) 예측 라벨 (threshold=0.5)
        pred_label = 1 if prob_1 >= 0.5 else 0

        result_row = {
            "id": row.get("id"),
            "text": text,
            "label": label,
            "pred_prob": prob_1,      # P(label=1 | text)
            "pred_label": pred_label, # 0 또는 1
        }
        results.append(result_row)

        if idx % 10 == 0 or idx == len(data):
            print(
                f"[INFO] Processed {idx}/{len(data)} "
                f"(id={row.get('id')}, prob_1={prob_1:.3f})"
            )

    # 5) 결과 저장
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_path, results)
    print(f"[INFO] Saved inference results to: {output_path}")


# -----------------------------
# 6. CLI 엔트리포인트
# -----------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default="data/scenario1_oov/input_s1.jsonl",
        help="입력 jsonl 경로",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/scenario1_oov/raw_inference.jsonl",
        help="출력 jsonl 경로",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="skt/A.X-4.0-Light",
        help="HuggingFace 모델 이름",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="cuda, cpu 중 선택 (기본: 자동 선택)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="앞에서부터 일부 샘플만 테스트하고 싶을 때 개수 제한",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_inference(
        input_path=Path(args.input),
        output_path=Path(args.output),
        model_name=args.model_name,
        device=args.device,
        max_samples=args.max_samples,
    )