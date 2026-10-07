#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LLM_SEM v0.18.7 Confusion-Aware Prototype Augmentation Experiment.

This experiment compares two frozen-base semantic routers on a NEW final holdout:

A) Baseline prototypes:
   first four original NDC seed texts per class (TRAIN ONLY)

B) Augmented prototypes:
   baseline prototypes + confusion-targeted representative phrases

Calibration uses only the two original DEV seed texts per class plus DEV unknowns.
The final-v2 known/unknown examples are not used for prototype construction or
threshold calibration.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch

from model import LanguageModel
from tokenizer import Tokenizer
from ndc import NDC_MAIN
from ndc_semantic_router_v0182 import NDC_MAIN_SEEDS, encode_text
from ndc_semantic_router_v0183 import build_prototypes, route_vector_multi
from ndc_prototypes_v0187 import TRAIN_ONLY_NDC_SEEDS, AUGMENTED_NDC_SEEDS


FINAL_V2: Tuple[Tuple[str, str], ...] = (
    ("0", "生成AIの基礎を知りたい"),
    ("0", "計算機上でデータを処理する仕組み"),
    ("0", "プログラムとソフトウェアの違い"),
    ("1", "道徳的な価値判断について"),
    ("1", "心の働きを研究する分野"),
    ("1", "推論と論理の考え方"),
    ("2", "明治時代の出来事について"),
    ("2", "歴史人物の経歴を調べる"),
    ("2", "地域の成り立ちと地理"),
    ("3", "物価と経済活動の関係"),
    ("3", "学校制度と教育政策"),
    ("3", "社会を支える法制度"),
    ("4", "恒星や銀河の研究分野"),
    ("4", "物質が変化する反応について"),
    ("4", "生命の進化と生物学"),
    ("5", "電子装置を設計する技術"),
    ("5", "機械装置の設計と製造"),
    ("5", "建物を設計する工学"),
    ("6", "物流と輸送のしくみ"),
    ("6", "商品を売買する産業"),
    ("6", "農作物の生産について"),
    ("7", "美術作品を鑑賞する"),
    ("7", "楽曲を演奏する活動"),
    ("7", "競技スポーツのルール"),
    ("8", "外国語の単語の意味"),
    ("8", "文章の文法構造を調べる"),
    ("8", "別の言語に訳す作業"),
    ("9", "物語作品を読む"),
    ("9", "詩歌の表現技法"),
    ("9", "作家の作品について論じる"),
)

UNKNOWN_DEV: Tuple[str, ...] = (
    "これはどういうこと",
    "それの説明",
    "XYZABC",
    "???",
    "未定義対象ガンマ",
    "意味不明な文字列qqq",
)

UNKNOWN_FINAL_V2: Tuple[str, ...] = (
    "あれについて詳しく",
    "何のことですか",
    "314159265",
    "ふわふわ概念",
    "対象未指定の依頼",
    "意味なしzxcvb",
)


def parse_args():
    p = argparse.ArgumentParser(
        description="LLM_SEM v0.18.7 confusion-aware prototype experiment"
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


def frange(start, stop, step):
    values = []
    x = start
    while x <= stop + 1e-12:
        values.append(round(x, 6))
        x += step
    return values


@torch.no_grad()
def encode_labeled(model, tokenizer, rows, pooling):
    return [
        (label, text, encode_text(model, tokenizer, text, pooling=pooling))
        for label, text in rows
    ]


@torch.no_grad()
def encode_unknown(model, tokenizer, rows, pooling):
    return [
        (text, encode_text(model, tokenizer, text, pooling=pooling))
        for text in rows
    ]


@torch.no_grad()
def calibrate(
    prototypes,
    dev_known,
    dev_unknown,
):
    best = None
    for top_k in (1, 2, 3):
        for max_weight in (0.40, 0.60, 0.80, 1.00):
            known_results = [
                route_vector_multi(
                    vector,
                    prototypes,
                    similarity_threshold=-1.0,
                    margin_threshold=-1.0,
                    top_k=top_k,
                    max_weight=max_weight,
                )
                for _, _, vector in dev_known
            ]
            unknown_results = [
                route_vector_multi(
                    vector,
                    prototypes,
                    similarity_threshold=-1.0,
                    margin_threshold=-1.0,
                    top_k=top_k,
                    max_weight=max_weight,
                )
                for _, vector in dev_unknown
            ]

            raw = sum(
                r.predicted_main == label
                for (label, _, _), r in zip(dev_known, known_results)
            ) / len(dev_known)

            for sim_th in frange(0.50, 0.95, 0.01):
                for margin_th in frange(0.00, 0.15, 0.005):
                    known_accept = sum(
                        r.predicted_main == label
                        and r.nearest_similarity >= sim_th
                        and r.class_margin >= margin_th
                        for (label, _, _), r in zip(dev_known, known_results)
                    ) / len(dev_known)

                    unknown_reject = sum(
                        r.nearest_similarity < sim_th
                        or r.class_margin < margin_th
                        for r in unknown_results
                    ) / len(dev_unknown)

                    balanced = 0.5 * known_accept + 0.5 * unknown_reject
                    candidate = (
                        balanced,
                        raw,
                        known_accept,
                        unknown_reject,
                        top_k,
                        max_weight,
                        sim_th,
                        margin_th,
                    )
                    if best is None or candidate > best:
                        best = candidate
    return best


@torch.no_grad()
def evaluate(
    prototypes,
    config,
    final_known,
    final_unknown,
):
    (
        _dev_balanced,
        _dev_raw,
        _dev_known,
        _dev_unknown,
        top_k,
        max_weight,
        sim_th,
        margin_th,
    ) = config

    known_results = []
    for label, text, vector in final_known:
        result = route_vector_multi(
            vector,
            prototypes,
            similarity_threshold=sim_th,
            margin_threshold=margin_th,
            top_k=top_k,
            max_weight=max_weight,
        )
        known_results.append((label, text, result))

    unknown_results = []
    for text, vector in final_unknown:
        result = route_vector_multi(
            vector,
            prototypes,
            similarity_threshold=sim_th,
            margin_threshold=margin_th,
            top_k=top_k,
            max_weight=max_weight,
        )
        unknown_results.append((text, result))

    raw = sum(r.predicted_main == label for label, _, r in known_results) / len(known_results)
    known_accept = sum(
        r.predicted_main == label and r.accepted
        for label, _, r in known_results
    ) / len(known_results)
    unknown_reject = sum(not r.accepted for _, r in unknown_results) / len(unknown_results)
    balanced = 0.5 * known_accept + 0.5 * unknown_reject

    return raw, known_accept, unknown_reject, balanced, known_results, unknown_results


def print_report(name, config, result):
    raw, known_accept, unknown_reject, balanced, known_results, unknown_results = result

    print()
    print("=" * 116)
    print(name)
    print("=" * 116)
    print("DEV selected config")
    print("  top_k          :", config[4])
    print("  max_weight     :", f"{config[5]:.2f}")
    print("  similarity th  :", f"{config[6]:.3f}")
    print("  margin th      :", f"{config[7]:.3f}")
    print("FINAL-V2")
    print("  raw accuracy   :", f"{raw*100.0:.2f}%")
    print("  known accept   :", f"{known_accept*100.0:.2f}%")
    print("  unknown reject :", f"{unknown_reject*100.0:.2f}%")
    print("  balanced       :", f"{balanced*100.0:.2f}%")
    print()

    print("Per-class FINAL-V2")
    print("-" * 116)
    for main in sorted(NDC_MAIN):
        rows = [(label, text, r) for label, text, r in known_results if label == main]
        correct = sum(r.predicted_main == main for _, _, r in rows)
        accepted = sum(r.predicted_main == main and r.accepted for _, _, r in rows)
        mean_nearest = sum(r.nearest_similarity for _, _, r in rows) / len(rows)
        mean_margin = sum(r.class_margin for _, _, r in rows) / len(rows)
        print(
            f"NDC {main} {NDC_MAIN[main]:<8} "
            f"raw={correct}/{len(rows)} accept={accepted}/{len(rows)} "
            f"nearest={mean_nearest:.6f} margin={mean_margin:+.6f}"
        )

    return raw, known_accept, unknown_reject, balanced


def main():
    args = parse_args()
    device = choose_device(args.device)

    tokenizer = Tokenizer.load(args.tokenizer)
    model, checkpoint = LanguageModel.load_checkpoint(args.model, device=device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    dev_rows = []
    for main, texts in NDC_MAIN_SEEDS.items():
        dev_rows.extend((main, text) for text in texts[4:])

    dev_known = encode_labeled(model, tokenizer, dev_rows, args.pooling)
    dev_unknown = encode_unknown(model, tokenizer, UNKNOWN_DEV, args.pooling)
    final_known = encode_labeled(model, tokenizer, FINAL_V2, args.pooling)
    final_unknown = encode_unknown(model, tokenizer, UNKNOWN_FINAL_V2, args.pooling)

    baseline_prototypes = build_prototypes(
        model,
        tokenizer,
        seeds=TRAIN_ONLY_NDC_SEEDS,
        pooling=args.pooling,
    )
    augmented_prototypes = build_prototypes(
        model,
        tokenizer,
        seeds=AUGMENTED_NDC_SEEDS,
        pooling=args.pooling,
    )

    print("=" * 116)
    print(" LLM_SEM v0.18.7 Confusion-Aware Prototype Augmentation Experiment")
    print("=" * 116)
    print("Device             :", device)
    if device.type == "cuda":
        print("GPU                :", torch.cuda.get_device_name(device))
    print("Model              :", args.model)
    print("Checkpoint loss    :", checkpoint.get("loss"))
    print("Base frozen        :", True)
    print("Pooling            :", args.pooling)
    print("Baseline prototypes:", sum(len(v) for v in TRAIN_ONLY_NDC_SEEDS.values()))
    print("Augmented prototypes:", sum(len(v) for v in AUGMENTED_NDC_SEEDS.values()))
    print("DEV known/unknown  :", f"{len(dev_known)}/{len(dev_unknown)}")
    print("FINAL-V2 known/unknown:", f"{len(final_known)}/{len(final_unknown)}")

    baseline_config = calibrate(baseline_prototypes, dev_known, dev_unknown)
    augmented_config = calibrate(augmented_prototypes, dev_known, dev_unknown)

    baseline_result = evaluate(
        baseline_prototypes,
        baseline_config,
        final_known,
        final_unknown,
    )
    augmented_result = evaluate(
        augmented_prototypes,
        augmented_config,
        final_known,
        final_unknown,
    )

    b = print_report("A) TRAIN-ONLY BASELINE", baseline_config, baseline_result)
    a = print_report("B) CONFUSION-AWARE AUGMENTED", augmented_config, augmented_result)

    print()
    print("=" * 116)
    print(" DELTA (AUGMENTED - BASELINE)")
    print("=" * 116)
    print(f"Raw accuracy       : {(a[0]-b[0])*100.0:+.2f} pp")
    print(f"Known accept       : {(a[1]-b[1])*100.0:+.2f} pp")
    print(f"Unknown reject     : {(a[2]-b[2])*100.0:+.2f} pp")
    print(f"Balanced score     : {(a[3]-b[3])*100.0:+.2f} pp")

    passed = (
        a[0] >= b[0]
        and a[1] >= b[1]
        and a[2] >= 0.80
        and (a[0] > b[0] or a[1] > b[1])
    )

    print("=" * 116)
    print(
        "RESULT:",
        "PASS" if passed else "EXPERIMENTAL_FAIL",
        f"aug_raw={a[0]*100.0:.2f}%",
        f"aug_known_accept={a[1]*100.0:.2f}%",
        f"aug_unknown_reject={a[2]*100.0:.2f}%",
        f"aug_balanced={a[3]*100.0:.2f}%",
    )
    print("=" * 116)

    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
