#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
extract_fn_cases.py

- 입력:
    1) raw_inference.jsonl
       (각 줄: {"id", "text", "label", "pred_prob", "pred_label"})
    2) oov_rate_7d.csv
       (각 행: id, text_len, token_count, unk_count, oov_rate)

- 출력:
    hard_fn_cases.csv
    (각 행: id, label, pred_prob, pred_label, text_len, token_count,
            unk_count, oov_rate, text)

역할:
    시나리오1(Hard-OOV 슬라이스)에서
    "정답은 1(유해/낚시)"인데 모델이 확률을 낮게(pred_prob < threshold) 준
    False Negative 케이스들을 추출하는 스크립트.

    → 산출물4에서 말하는 hard_fn_cases.csv 증거 세트에 해당.
"""

import argparse
import csv
import json
from pathlib import Path


# -----------------------------
# 1. 유틸: jsonl / csv 읽기
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


def read_csv_as_dict(path, key_field="id"):
    """
    CSV 파일을 읽어서
        { key_field 값: row_dict }
    형태의 dict로 반환.

    예:
        key_field="id" 이면
        {"s1_001": {"id": "s1_001", "text_len": "...", ...}, ...}
    """
    table = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = row[key_field]
            table[key] = row
    return table


# -----------------------------
# 2. Hard FN 추출 로직
# -----------------------------

def extract_hard_fn(
    inference_path: Path,
    oov_path: Path,
    output_path: Path,
    threshold: float = 0.5,
):
    """
    - raw_inference.jsonl 과 oov_rate_7d.csv 를 로드하고
    - label=1 이면서 pred_prob < threshold 인 행만 골라
    - OOV 통계까지 merge한 뒤 hard_fn_cases.csv 로 저장한다.
    """
    print(f"[INFO] Loading inference from: {inference_path}")
    preds = read_jsonl(inference_path)
    print(f"[INFO] Loaded {len(preds)} inference rows.")

    print(f"[INFO] Loading OOV stats from: {oov_path}")
    oov_table = read_csv_as_dict(oov_path, key_field="id")
    print(f"[INFO] Loaded OOV stats for {len(oov_table)} ids.")

    hard_fn_rows = []

    for row in preds:
        sample_id = row.get("id")
        label = row.get("label")
        pred_prob = row.get("pred_prob", None)

        # label이 1(유해/낚시)인 케이스만 FN 후보
        if label != 1:
            continue

        if pred_prob is None:
            # 예측 확률이 아예 없으면 스킵
            continue

        # threshold 아래이면 FN으로 간주
        if pred_prob >= threshold:
            continue

        # OOV 정보 붙이기
        oov_info = oov_table.get(sample_id, {})
        # csv에서 읽은 값들은 str 이므로, 필요하면 float/int로 변환
        text_len = int(oov_info["text_len"]) if "text_len" in oov_info else None
        token_count = int(oov_info["token_count"]) if "token_count" in oov_info else None
        unk_count = int(oov_info["unk_count"]) if "unk_count" in oov_info else None
        oov_rate = float(oov_info["oov_rate"]) if "oov_rate" in oov_info else None

        hard_fn_rows.append({
            "id": sample_id,
            "label": label,
            "pred_prob": pred_prob,
            "pred_label": row.get("pred_label"),
            "text_len": text_len,
            "token_count": token_count,
            "unk_count": unk_count,
            "oov_rate": oov_rate,
            "text": row.get("text"),
        })

    print(f"[INFO] Found {len(hard_fn_rows)} hard FN cases "
          f"(label=1 & pred_prob < {threshold}).")

    # 출력 디렉토리 생성
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # CSV로 저장
    fieldnames = [
        "id",
        "label",
        "pred_prob",
        "pred_label",
        "text_len",
        "token_count",
        "unk_count",
        "oov_rate",
        "text",
    ]
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in hard_fn_rows:
            writer.writerow(r)

    print(f"[INFO] Saved hard FN cases to: {output_path}")


# -----------------------------
# 3. CLI 엔트리포인트
# -----------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inference",
        type=str,
        default="outputs/scenario1_oov/raw_inference.jsonl",
        help="raw_inference.jsonl 경로",
    )
    parser.add_argument(
        "--oov",
        type=str,
        default="outputs/scenario1_oov/oov_rate_7d.csv",
        help="oov_rate_7d.csv 경로",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/scenario1_oov/hard_fn_cases.csv",
        help="hard_fn_cases.csv 출력 경로",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="label=1인 경우 FN으로 간주할 pred_prob 상한값",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    extract_hard_fn(
        inference_path=Path(args.inference),
        oov_path=Path(args.oov),
        output_path=Path(args.output),
        threshold=args.threshold,
    )