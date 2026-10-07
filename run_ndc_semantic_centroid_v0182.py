#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LLM_SEM v0.18.2 Semantic Vector + NDC centroid + unknown threshold experiment.

Protocol:
  1. Load an existing LLM_SEM checkpoint.
  2. Build one semantic centroid for each NDC main class (0-9).
  3. Evaluate held-out paraphrases with the centroid router.
  4. Evaluate deliberately underspecified / nonsense unknown probes.
  5. Sweep top-1 cosine thresholds and select the threshold with the best
     balanced acceptance score:
       0.5 * known_accept_rate + 0.5 * unknown_reject_rate
  6. Report routing accuracy, threshold behavior, and per-class results.

The experiment does not train model parameters.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch

from model import LanguageModel
from tokenizer import Tokenizer
from ndc import NDC_MAIN
from ndc_semantic_router_v0182 import (
    build_centroids,
    encode_text,
    route_vector,
)


KNOWN_HOLDOUT: Tuple[Tuple[str, str], ...] = (
    ("0", "LLMの仕組みを説明して"),
    ("0", "GPUでプログラムを高速化する方法"),
    ("0", "ソフトウェアとデータベースについて"),
    ("1", "倫理的な判断とは何か"),
    ("1", "人間の認知と心理について"),
    ("1", "論理的に考えるとはどういうこと"),
    ("2", "江戸時代について教えて"),
    ("2", "ある人物の生涯を知りたい"),
    ("2", "世界の地理を説明して"),
    ("3", "景気と市場の関係"),
    ("3", "学校教育の制度について"),
    ("3", "法律は社会で何をするのか"),
    ("4", "ブラックホールはどのような天体か"),
    ("4", "化学反応では何が起きるのか"),
    ("4", "生物の進化について"),
    ("5", "電子回路を設計する"),
    ("5", "機械を設計する工学"),
    ("5", "建築技術について"),
    ("6", "鉄道輸送の仕組み"),
    ("6", "商品の流通について"),
    ("6", "農作物を育てる産業"),
    ("7", "絵画を鑑賞する"),
    ("7", "楽器を演奏する"),
    ("7", "スポーツ競技について"),
    ("8", "英単語の意味を知りたい"),
    ("8", "日本語の文法を説明して"),
    ("8", "翻訳の方法について"),
    ("9", "小説を読む"),
    ("9", "詩の表現について"),
    ("9", "作家と文学作品について"),
)

UNKNOWN_PROBES: Tuple[str, ...] = (
    "それについて",
    "これは何",
    "あれの意味",
    "XYZXYZ",
    "ふにゃらふにゃら",
    "???",
    "123456789",
    "それを詳しく",
    "何か教えて",
    "未定義概念アルファベータ",
    "対象不明の質問",
    "意味のない文字列qzxv",
)


@dataclass
class KnownRow:
    expected: str
    predicted: str
    similarity: float
    margin: float
    correct: bool


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LLM_SEM v0.18.2 NDC semantic centroid threshold experiment"
    )
    p.add_argument(
        "--model",
        default="model/model-sem-internalized-v01575.pt",
    )
    p.add_argument(
        "--tokenizer",
        default="model/tokenizer.json",
    )
    p.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
    )
    p.add_argument(
        "--pooling",
        default="mean",
        choices=["mean", "last", "bos", "max", "attention", "hybrid"],
    )
    p.add_argument("--threshold-min", type=float, default=0.50)
    p.add_argument("--threshold-max", type=float, default=0.99)
    p.add_argument("--threshold-step", type=float, default=0.005)
    p.add_argument("--margin-threshold", type=float, default=0.0)
    return p.parse_args()


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(name)


def threshold_grid(start: float, stop: float, step: float) -> List[float]:
    if step <= 0:
        raise ValueError("threshold step must be positive")
    values: List[float] = []
    value = start
    while value <= stop + 1.0e-12:
        values.append(round(value, 6))
        value += step
    return values


def main() -> int:
    args = parse_args()
    device = choose_device(args.device)

    model_path = Path(args.model)
    tokenizer_path = Path(args.tokenizer)
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    if not tokenizer_path.exists():
        raise FileNotFoundError(tokenizer_path)

    tokenizer = Tokenizer.load(str(tokenizer_path))
    model, checkpoint = LanguageModel.load_checkpoint(
        str(model_path),
        device=device,
    )
    model.eval()

    print("=" * 104)
    print(" LLM_SEM v0.18.2 Semantic Vector + NDC Centroid + Unknown Threshold Experiment")
    print("=" * 104)
    print("Device            :", device)
    if device.type == "cuda":
        print("GPU               :", torch.cuda.get_device_name(device))
    print("Model             :", model_path)
    print("Tokenizer         :", tokenizer_path)
    print("Checkpoint loss   :", checkpoint.get("loss"))
    print("Pooling           :", args.pooling)
    print("Known holdout     :", len(KNOWN_HOLDOUT))
    print("Unknown probes    :", len(UNKNOWN_PROBES))
    print("Margin threshold  :", args.margin_threshold)
    print()

    print("Building NDC 0-9 centroids...")
    centroids = build_centroids(
        model,
        tokenizer,
        pooling=args.pooling,
    )

    known_rows: List[KnownRow] = []
    for expected, text in KNOWN_HOLDOUT:
        vector = encode_text(
            model,
            tokenizer,
            text,
            pooling=args.pooling,
        )
        result = route_vector(
            vector,
            centroids,
            similarity_threshold=-1.0,
            margin_threshold=-1.0,
        )
        known_rows.append(
            KnownRow(
                expected=expected,
                predicted=result.predicted_main,
                similarity=result.top1_similarity,
                margin=result.margin,
                correct=result.predicted_main == expected,
            )
        )

    unknown_raw = []
    for text in UNKNOWN_PROBES:
        vector = encode_text(
            model,
            tokenizer,
            text,
            pooling=args.pooling,
        )
        result = route_vector(
            vector,
            centroids,
            similarity_threshold=-1.0,
            margin_threshold=-1.0,
        )
        unknown_raw.append((text, result))

    raw_correct = sum(row.correct for row in known_rows)
    raw_accuracy = raw_correct / len(known_rows)

    print("Raw NDC routing")
    print("-" * 104)
    print(f"Top-1 accuracy     : {raw_accuracy * 100.0:.2f}% ({raw_correct}/{len(known_rows)})")
    print()

    print("Per-class raw routing")
    print("-" * 104)
    for main in sorted(NDC_MAIN):
        rows = [row for row in known_rows if row.expected == main]
        correct = sum(row.correct for row in rows)
        mean_sim = sum(row.similarity for row in rows) / len(rows)
        mean_margin = sum(row.margin for row in rows) / len(rows)
        print(
            f"NDC {main} {NDC_MAIN[main]:<8} "
            f"accuracy={correct}/{len(rows)} "
            f"mean_sim={mean_sim:.6f} mean_margin={mean_margin:+.6f}"
        )
    print()

    best = None
    for threshold in threshold_grid(
        args.threshold_min,
        args.threshold_max,
        args.threshold_step,
    ):
        known_accept = sum(
            row.correct
            and row.similarity >= threshold
            and row.margin >= args.margin_threshold
            for row in known_rows
        )
        known_accept_rate = known_accept / len(known_rows)

        unknown_reject = sum(
            result.top1_similarity < threshold
            or result.margin < args.margin_threshold
            for _, result in unknown_raw
        )
        unknown_reject_rate = unknown_reject / len(unknown_raw)

        balanced = 0.5 * known_accept_rate + 0.5 * unknown_reject_rate

        candidate = (
            balanced,
            unknown_reject_rate,
            known_accept_rate,
            threshold,
            known_accept,
            unknown_reject,
        )
        if best is None or candidate > best:
            best = candidate

    assert best is not None
    (
        best_balanced,
        best_unknown_reject_rate,
        best_known_accept_rate,
        best_threshold,
        best_known_accept,
        best_unknown_reject,
    ) = best

    print("Threshold calibration")
    print("-" * 104)
    print(f"Selected threshold : {best_threshold:.6f}")
    print(
        f"Known accept       : {best_known_accept}/{len(known_rows)} "
        f"({best_known_accept_rate * 100.0:.2f}%)"
    )
    print(
        f"Unknown reject     : {best_unknown_reject}/{len(unknown_raw)} "
        f"({best_unknown_reject_rate * 100.0:.2f}%)"
    )
    print(f"Balanced score     : {best_balanced * 100.0:.2f}%")
    print()

    print("Known holdout details")
    print("-" * 104)
    accepted_correct = 0
    for (expected, text), row in zip(KNOWN_HOLDOUT, known_rows):
        accepted = (
            row.correct
            and row.similarity >= best_threshold
            and row.margin >= args.margin_threshold
        )
        accepted_correct += int(accepted)
        state = "ACCEPT" if accepted else "REVIEW"
        print(
            f"[{state:<6}] expected={expected} predicted={row.predicted} "
            f"sim={row.similarity:.6f} margin={row.margin:+.6f} "
            f"text={text}"
        )
    print()

    print("Unknown probe details")
    print("-" * 104)
    rejected = 0
    for text, result in unknown_raw:
        is_rejected = (
            result.top1_similarity < best_threshold
            or result.margin < args.margin_threshold
        )
        rejected += int(is_rejected)
        state = "UNKNOWN" if is_rejected else "FALSE_ACCEPT"
        print(
            f"[{state:<12}] predicted={result.predicted_main} "
            f"sim={result.top1_similarity:.6f} margin={result.margin:+.6f} "
            f"text={text}"
        )
    print()

    result_ok = (
        raw_accuracy >= 0.50
        and best_known_accept_rate >= 0.50
        and best_unknown_reject_rate >= 0.50
    )

    print("=" * 104)
    print(
        "RESULT:",
        "PASS" if result_ok else "EXPERIMENTAL_FAIL",
        f"raw={raw_accuracy * 100.0:.2f}%",
        f"known_accept={best_known_accept_rate * 100.0:.2f}%",
        f"unknown_reject={best_unknown_reject_rate * 100.0:.2f}%",
        f"threshold={best_threshold:.6f}",
    )
    print("=" * 104)

    return 0 if result_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
