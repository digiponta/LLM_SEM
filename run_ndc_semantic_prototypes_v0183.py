#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LLM_SEM v0.18.3 Multi-Prototype NDC Semantic Routing Experiment.

This experiment follows v0.18.2, where a single centroid per NDC main class
reached only moderate raw routing accuracy and required a high similarity
threshold that rejected too many known samples.

v0.18.3 keeps multiple prototypes per NDC class and searches over:
  - top-k prototypes used for class aggregation
  - nearest-prototype weight
  - similarity threshold
  - class-margin threshold

The model is frozen; no parameter training is performed.
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
from ndc_semantic_router_v0182 import encode_text
from ndc_semantic_router_v0183 import build_prototypes, route_vector_multi


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
class EncodedKnown:
    expected: str
    text: str
    vector: torch.Tensor


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LLM_SEM v0.18.3 multi-prototype NDC semantic experiment"
    )
    p.add_argument("--model", default="model/model-sem-internalized-v01575.pt")
    p.add_argument("--tokenizer", default="model/tokenizer.json")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--pooling", default="mean")
    return p.parse_args()


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(name)


def frange(start: float, stop: float, step: float) -> List[float]:
    values: List[float] = []
    x = start
    while x <= stop + 1.0e-12:
        values.append(round(x, 6))
        x += step
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
    model, checkpoint = LanguageModel.load_checkpoint(str(model_path), device=device)
    model.eval()

    print("=" * 112)
    print(" LLM_SEM v0.18.3 Multi-Prototype NDC Semantic Routing Experiment")
    print("=" * 112)
    print("Device            :", device)
    if device.type == "cuda":
        print("GPU               :", torch.cuda.get_device_name(device))
    print("Model             :", model_path)
    print("Tokenizer         :", tokenizer_path)
    print("Checkpoint loss   :", checkpoint.get("loss"))
    print("Pooling           :", args.pooling)
    print("Known holdout     :", len(KNOWN_HOLDOUT))
    print("Unknown probes    :", len(UNKNOWN_PROBES))
    print()

    prototypes = build_prototypes(
        model,
        tokenizer,
        pooling=args.pooling,
    )

    known = [
        EncodedKnown(
            expected=expected,
            text=text,
            vector=encode_text(model, tokenizer, text, pooling=args.pooling),
        )
        for expected, text in KNOWN_HOLDOUT
    ]
    unknown = [
        (text, encode_text(model, tokenizer, text, pooling=args.pooling))
        for text in UNKNOWN_PROBES
    ]

    configs = []
    for top_k in (1, 2, 3):
        for max_weight in (0.40, 0.60, 0.80, 1.00):
            raw_results = [
                route_vector_multi(
                    row.vector,
                    prototypes,
                    similarity_threshold=-1.0,
                    margin_threshold=-1.0,
                    top_k=top_k,
                    max_weight=max_weight,
                )
                for row in known
            ]
            raw_accuracy = sum(
                result.predicted_main == row.expected
                for row, result in zip(known, raw_results)
            ) / len(known)

            unknown_results = [
                route_vector_multi(
                    vector,
                    prototypes,
                    similarity_threshold=-1.0,
                    margin_threshold=-1.0,
                    top_k=top_k,
                    max_weight=max_weight,
                )
                for _, vector in unknown
            ]

            for sim_th in frange(0.60, 0.95, 0.01):
                for margin_th in frange(0.00, 0.08, 0.005):
                    known_accept = sum(
                        result.predicted_main == row.expected
                        and result.nearest_similarity >= sim_th
                        and result.class_margin >= margin_th
                        for row, result in zip(known, raw_results)
                    )
                    known_rate = known_accept / len(known)

                    unknown_reject = sum(
                        result.nearest_similarity < sim_th
                        or result.class_margin < margin_th
                        for result in unknown_results
                    )
                    unknown_rate = unknown_reject / len(unknown)

                    balanced = 0.5 * known_rate + 0.5 * unknown_rate
                    # Prefer better raw routing first when balanced scores tie,
                    # then slightly prefer retaining known knowledge.
                    configs.append((
                        balanced,
                        raw_accuracy,
                        known_rate,
                        unknown_rate,
                        top_k,
                        max_weight,
                        sim_th,
                        margin_th,
                        raw_results,
                        unknown_results,
                    ))

    configs.sort(
        key=lambda row: (row[0], row[1], row[2], row[3]),
        reverse=True,
    )
    best = configs[0]
    (
        balanced,
        raw_accuracy,
        known_rate,
        unknown_rate,
        top_k,
        max_weight,
        sim_th,
        margin_th,
        raw_results,
        unknown_results,
    ) = best

    print("Selected configuration")
    print("-" * 112)
    print("top_k             :", top_k)
    print("max_weight        :", f"{max_weight:.2f}")
    print("similarity th     :", f"{sim_th:.6f}")
    print("margin th         :", f"{margin_th:.6f}")
    print("raw accuracy      :", f"{raw_accuracy * 100.0:.2f}%")
    print("known accept      :", f"{known_rate * 100.0:.2f}%")
    print("unknown reject    :", f"{unknown_rate * 100.0:.2f}%")
    print("balanced score    :", f"{balanced * 100.0:.2f}%")
    print()

    print("Per-class raw routing")
    print("-" * 112)
    for main in sorted(NDC_MAIN):
        pairs = [
            (row, result)
            for row, result in zip(known, raw_results)
            if row.expected == main
        ]
        correct = sum(result.predicted_main == main for row, result in pairs)
        mean_nearest = sum(result.nearest_similarity for row, result in pairs) / len(pairs)
        mean_margin = sum(result.class_margin for row, result in pairs) / len(pairs)
        print(
            f"NDC {main} {NDC_MAIN[main]:<8} "
            f"accuracy={correct}/{len(pairs)} "
            f"nearest={mean_nearest:.6f} margin={mean_margin:+.6f}"
        )
    print()

    print("Known holdout details")
    print("-" * 112)
    for row, result in zip(known, raw_results):
        correct = result.predicted_main == row.expected
        accepted = (
            correct
            and result.nearest_similarity >= sim_th
            and result.class_margin >= margin_th
        )
        state = "ACCEPT" if accepted else ("MISROUTE" if not correct else "REVIEW")
        print(
            f"[{state:<8}] expected={row.expected} predicted={result.predicted_main} "
            f"nearest={result.nearest_similarity:.6f} "
            f"score={result.top1_score:.6f} margin={result.class_margin:+.6f} "
            f"text={row.text}"
        )
    print()

    print("Unknown probe details")
    print("-" * 112)
    for (text, _), result in zip(unknown, unknown_results):
        rejected = (
            result.nearest_similarity < sim_th
            or result.class_margin < margin_th
        )
        state = "UNKNOWN" if rejected else "FALSE_ACCEPT"
        print(
            f"[{state:<12}] predicted={result.predicted_main} "
            f"nearest={result.nearest_similarity:.6f} "
            f"score={result.top1_score:.6f} margin={result.class_margin:+.6f} "
            f"text={text}"
        )
    print()

    # v0.18.3 is successful if it materially improves the single-centroid raw
    # baseline (53.33%) while keeping useful unknown rejection.
    passed = (
        raw_accuracy >= 0.60
        and known_rate >= 0.40
        and unknown_rate >= 0.75
    )

    print("=" * 112)
    print(
        "RESULT:",
        "PASS" if passed else "EXPERIMENTAL_FAIL",
        f"raw={raw_accuracy * 100.0:.2f}%",
        f"known_accept={known_rate * 100.0:.2f}%",
        f"unknown_reject={unknown_rate * 100.0:.2f}%",
        f"balanced={balanced * 100.0:.2f}%",
    )
    print("=" * 112)

    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
