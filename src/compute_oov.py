#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
compute_oov.py

- 입력:  jsonl (각 줄: {"id": ..., "text": ..., "label": ...})
- 출력:  oov_rate_7d.csv
        (각 행: id, text_len, token_count, unk_count, oov_rate)

역할:
    A.X-4.0-Light 토크나이저 기준으로
    각 문장의 OOV율(UNK 비율)을 계산해서 저장한다.

이 결과는:
    - 산출물4: 시나리오1의 OOV 모니터링 지표(oov_rate_7d.csv)
    - hard_fn_cases 추출 시, FN 문장에 OOV율 붙이는 용도
로 사용된다.
"""

import argparse
import csv
import json
from pathlib import Path

from transformers import AutoTokenizer


# -----------------------------
# 1. 유틸: jsonl 읽기
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


# -----------------------------
# 2. 토크나이저 로딩
# -----------------------------

def load_tokenizer(model_name: str = "skt/A.X-4.0-Light"):
    """
    A.X-4.0-Light 토크나이저 로딩.

    - unk_token_id 를 찾아서, 이 토큰이 몇 번 나오는지로 OOV를 측정한다.
    """
    print(f"[INFO] Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 일부 모델은 unk_token이 없을 수 있으니 방어적으로 처리
    if tokenizer.unk_token_id is None and tokenizer.unk_token is not None:
        unk_id = tokenizer.convert_tokens_to_ids(tokenizer.unk_token)
    else:
        unk_id = tokenizer.unk_token_id

    if unk_id is None:
        print("[WARN] This tokenizer has no unk_token_id. "
              "OOV율이 항상 0으로 나올 수 있음.")
    else:
        print(f"[INFO] unk_token_id = {unk_id} (token={tokenizer.unk_token})")

    return tokenizer, unk_id


# -----------------------------
# 3. 한 문장에 대한 OOV 계산
# -----------------------------

def compute_oov_for_text(text: str, tokenizer, unk_id: int):
    """
    한 문장에 대해:
        - 문자 길이(text_len)
        - 토큰 개수(token_count)
        - UNK 토큰 개수(unk_count)
        - OOV율(unk_count / token_count)

    을 계산해서 dict로 반환.
    """
    # 토크나이저가 사용하는 기본 세팅으로 토큰화
    encoded = tokenizer(
        text,
        add_special_tokens=True,  # BOS/CLS 등 포함(원하면 False로 바꿔도 됨)
        return_attention_mask=False,
        return_token_type_ids=False,
    )

    input_ids = encoded["input_ids"]      # List[int]
    token_count = len(input_ids)

    if token_count == 0:
        # 혹시 모를 edge case 방어
        return {
            "text_len": len(text),
            "token_count": 0,
            "unk_count": 0,
            "oov_rate": 0.0,
        }

    if unk_id is None:
        # unk 토큰 개념이 없으면 OOV율 = 0으로 처리
        unk_count = 0
    else:
        unk_count = sum(1 for tid in input_ids if tid == unk_id)

    oov_rate = float(unk_count) / float(token_count)

    return {
        "text_len": len(text),
        "token_count": token_count,
        "unk_count": unk_count,
        "oov_rate": oov_rate,
    }


# -----------------------------
# 4. 전체 루프 + CSV 저장
# -----------------------------

def run_compute_oov(
    input_path: Path,
    output_path: Path,
    model_name: str = "skt/A.X-4.0-Light",
    max_samples: int = None,
):
    """
    - input_s1.jsonl 을 읽고
    - 각 문장에 대해 OOV 통계를 계산한 뒤
    - oov_rate_7d.csv로 저장한다.
    """
    print(f"[INFO] Loading input from: {input_path}")
    data = read_jsonl(input_path)
    if max_samples is not None:
        data = data[:max_samples]
        print(f"[INFO] Truncated to {len(data)} samples (max_samples={max_samples})")
    else:
        print(f"[INFO] Total {len(data)} samples loaded.")

    tokenizer, unk_id = load_tokenizer(model_name)

    rows_for_csv = []

    for idx, row in enumerate(data, start=1):
        text = row["text"]
        sample_id = row.get("id")

        stats = compute_oov_for_text(text, tokenizer, unk_id)

        csv_row = {
            "id": sample_id,
            "text_len": stats["text_len"],
            "token_count": stats["token_count"],
            "unk_count": stats["unk_count"],
            "oov_rate": stats["oov_rate"],
        }
        rows_for_csv.append(csv_row)

        if idx % 10 == 0 or idx == len(data):
            print(
                f"[INFO] Processed {idx}/{len(data)} "
                f"(id={sample_id}, oov_rate={stats['oov_rate']:.3f})"
            )

    # 출력 디렉토리 만들기
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # CSV로 저장
    fieldnames = ["id", "text_len", "token_count", "unk_count", "oov_rate"]
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows_for_csv:
            writer.writerow(r)

    print(f"[INFO] Saved OOV stats to: {output_path}")


# -----------------------------
# 5. CLI 엔트리포인트
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
        default="outputs/scenario1_oov/oov_rate_7d.csv",
        help="출력 csv 경로",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="skt/A.X-4.0-Light",
        help="토크나이저를 로드할 HF 모델 이름",
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
    run_compute_oov(
        input_path=Path(args.input),
        output_path=Path(args.output),
        model_name=args.model_name,
        max_samples=args.max_samples,
    )