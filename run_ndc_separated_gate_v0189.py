#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LLM_SEM v0.18.9 Separated Classification / Unknown Detection Experiment.

Calibration:
  original DEV known (20) + DEV unknown (6)

Evaluation:
  NEW FINAL-V4 known (30) + NEW FINAL-V4 unknown (6)

Classifier:
  confusion-aware augmented prototype router

Gate:
  independent evidence score based on
    agreement + baseline/augmented similarities + margins

This stage tests whether raw NDC classification strength can be preserved while
recovering known acceptance without sacrificing unknown rejection.
"""

from __future__ import annotations

import argparse
from typing import Tuple

import torch

from model import LanguageModel
from tokenizer import Tokenizer
from ndc import NDC_MAIN
from ndc_semantic_router_v0182 import NDC_MAIN_SEEDS, encode_text
from ndc_semantic_router_v0183 import build_prototypes, route_vector_multi
from ndc_prototypes_v0187 import TRAIN_ONLY_NDC_SEEDS, AUGMENTED_NDC_SEEDS
from ndc_separated_gate_v0189 import route_separated


FINAL_V4: Tuple[Tuple[str, str], ...] = (
    ("0", "情報処理を行う計算機システム"),
    ("0", "AIソフトウェアの仕組み"),
    ("0", "データを扱うプログラム技術"),
    ("1", "倫理について考える哲学分野"),
    ("1", "認知と心の仕組み"),
    ("1", "論理的な推論の原理"),
    ("2", "江戸から明治への歴史的変化"),
    ("2", "歴史上の人物について調査する"),
    ("2", "地域社会の歴史をたどる"),
    ("3", "経済政策と景気の動き"),
    ("3", "教育制度と社会の関係"),
    ("3", "法制度が社会で果たす役割"),
    ("4", "星や銀河を研究する自然科学"),
    ("4", "化学物質の反応を調べる"),
    ("4", "生物の遺伝と進化"),
    ("5", "電子機器を設計する工学"),
    ("5", "機械装置を開発する技術"),
    ("5", "建築構造を設計する分野"),
    ("6", "輸送と物流に関わる産業"),
    ("6", "商業と商品の流通"),
    ("6", "農産物を生産する産業"),
    ("7", "絵画や彫刻を鑑賞する"),
    ("7", "音楽の演奏と芸術表現"),
    ("7", "スポーツ競技を行う"),
    ("8", "英語表現と語彙を学ぶ"),
    ("8", "日本語文法を分析する"),
    ("8", "文章を別の言語へ訳す"),
    ("9", "小説を文学として読む"),
    ("9", "詩の表現を味わう"),
    ("9", "文学作家と作品について"),
)

UNKNOWN_DEV: Tuple[str, ...] = (
    "これはどういうこと",
    "それの説明",
    "XYZABC",
    "???",
    "未定義対象ガンマ",
    "意味不明な文字列qqq",
)

UNKNOWN_FINAL_V4: Tuple[str, ...] = (
    "それを教えてください",
    "何についてですか",
    "1618033988",
    "ぼんやり概念",
    "対象なしの説明依頼",
    "意味なしpoiuy",
)


def parse_args():
    p = argparse.ArgumentParser(
        description="LLM_SEM v0.18.9 separated classification/gate experiment"
    )
    p.add_argument("--model", default="model/model-sem-internalized-v01575.pt")
    p.add_argument("--tokenizer", default="model/tokenizer.json")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--pooling", default="mean")
    return p.parse_args()


def choose_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(name)


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
def raw_augmented_accuracy(augmented_prototypes, rows, top_k, max_weight):
    results = [
        route_vector_multi(
            v,
            augmented_prototypes,
            similarity_threshold=-1.0,
            margin_threshold=-1.0,
            top_k=top_k,
            max_weight=max_weight,
        )
        for _, _, v in rows
    ]
    return sum(
        r.predicted_main == label
        for (label, _, _), r in zip(rows, results)
    ) / len(rows)


@torch.no_grad()
def calibrate_gate(
    baseline_prototypes,
    augmented_prototypes,
    dev_known,
    dev_unknown,
):
    best = None

    # Keep classifier search small; v0.18.7/8 repeatedly preferred top_k=3,
    # max_weight=1.0, but retain a little flexibility.
    for top_k in (1, 2, 3):
        for max_weight in (0.80, 1.00):
            for agreement_bonus in (0.05, 0.10, 0.15, 0.20):
                for aug_sim_w in (0.50, 0.75, 1.00):
                    for base_sim_w in (0.00, 0.25, 0.50):
                        for aug_margin_w in (0.50, 1.00, 1.50):
                            for base_margin_w in (0.00, 0.50, 1.00):
                                known_evidence = [
                                    route_separated(
                                        v,
                                        baseline_prototypes,
                                        augmented_prototypes,
                                        top_k=top_k,
                                        max_weight=max_weight,
                                        agreement_bonus=agreement_bonus,
                                        augmented_similarity_weight=aug_sim_w,
                                        baseline_similarity_weight=base_sim_w,
                                        augmented_margin_weight=aug_margin_w,
                                        baseline_margin_weight=base_margin_w,
                                        evidence_threshold=-999.0,
                                    )
                                    for _, _, v in dev_known
                                ]
                                unknown_evidence = [
                                    route_separated(
                                        v,
                                        baseline_prototypes,
                                        augmented_prototypes,
                                        top_k=top_k,
                                        max_weight=max_weight,
                                        agreement_bonus=agreement_bonus,
                                        augmented_similarity_weight=aug_sim_w,
                                        baseline_similarity_weight=base_sim_w,
                                        augmented_margin_weight=aug_margin_w,
                                        baseline_margin_weight=base_margin_w,
                                        evidence_threshold=-999.0,
                                    )
                                    for _, v in dev_unknown
                                ]

                                values = sorted(
                                    {
                                        round(r.evidence_score, 6)
                                        for r in known_evidence + unknown_evidence
                                    }
                                )
                                if not values:
                                    continue

                                thresholds = [values[0] - 1e-6]
                                thresholds += [
                                    (a + b) / 2.0
                                    for a, b in zip(values, values[1:])
                                ]
                                thresholds.append(values[-1] + 1e-6)

                                raw = sum(
                                    r.predicted_main == label
                                    for (label, _, _), r
                                    in zip(dev_known, known_evidence)
                                ) / len(dev_known)

                                for threshold in thresholds:
                                    known_accept = sum(
                                        r.predicted_main == label
                                        and r.evidence_score >= threshold
                                        for (label, _, _), r
                                        in zip(dev_known, known_evidence)
                                    ) / len(dev_known)

                                    unknown_reject = sum(
                                        r.evidence_score < threshold
                                        for r in unknown_evidence
                                    ) / len(dev_unknown)

                                    balanced = (
                                        0.5 * known_accept
                                        + 0.5 * unknown_reject
                                    )

                                    # Prefer balanced score; then known
                                    # acceptance; then raw classification.
                                    candidate = (
                                        balanced,
                                        known_accept,
                                        unknown_reject,
                                        raw,
                                        top_k,
                                        max_weight,
                                        agreement_bonus,
                                        aug_sim_w,
                                        base_sim_w,
                                        aug_margin_w,
                                        base_margin_w,
                                        threshold,
                                    )
                                    if best is None or candidate > best:
                                        best = candidate
    return best


@torch.no_grad()
def evaluate(
    baseline_prototypes,
    augmented_prototypes,
    config,
    final_known,
    final_unknown,
):
    (
        _dev_balanced,
        _dev_known,
        _dev_unknown,
        _dev_raw,
        top_k,
        max_weight,
        agreement_bonus,
        aug_sim_w,
        base_sim_w,
        aug_margin_w,
        base_margin_w,
        threshold,
    ) = config

    known_results = []
    for label, text, v in final_known:
        r = route_separated(
            v,
            baseline_prototypes,
            augmented_prototypes,
            top_k=top_k,
            max_weight=max_weight,
            agreement_bonus=agreement_bonus,
            augmented_similarity_weight=aug_sim_w,
            baseline_similarity_weight=base_sim_w,
            augmented_margin_weight=aug_margin_w,
            baseline_margin_weight=base_margin_w,
            evidence_threshold=threshold,
        )
        known_results.append((label, text, r))

    unknown_results = []
    for text, v in final_unknown:
        r = route_separated(
            v,
            baseline_prototypes,
            augmented_prototypes,
            top_k=top_k,
            max_weight=max_weight,
            agreement_bonus=agreement_bonus,
            augmented_similarity_weight=aug_sim_w,
            baseline_similarity_weight=base_sim_w,
            augmented_margin_weight=aug_margin_w,
            baseline_margin_weight=base_margin_w,
            evidence_threshold=threshold,
        )
        unknown_results.append((text, r))

    raw = sum(
        r.predicted_main == label
        for label, _, r in known_results
    ) / len(known_results)

    known_accept = sum(
        r.predicted_main == label and r.accepted
        for label, _, r in known_results
    ) / len(known_results)

    unknown_reject = sum(
        not r.accepted
        for _, r in unknown_results
    ) / len(unknown_results)

    balanced = 0.5 * known_accept + 0.5 * unknown_reject
    return raw, known_accept, unknown_reject, balanced, known_results, unknown_results


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
    final_known = encode_labeled(model, tokenizer, FINAL_V4, args.pooling)
    final_unknown = encode_unknown(model, tokenizer, UNKNOWN_FINAL_V4, args.pooling)

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
    print(" LLM_SEM v0.18.9 Separated Classification / Unknown Gate Experiment")
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
    print("FINAL-V4 known/unknown:", f"{len(final_known)}/{len(final_unknown)}")
    print()

    config = calibrate_gate(
        baseline_prototypes,
        augmented_prototypes,
        dev_known,
        dev_unknown,
    )

    print("Selected DEV gate")
    print("-" * 116)
    print(f"dev balanced       : {config[0]*100.0:.2f}%")
    print(f"dev known accept   : {config[1]*100.0:.2f}%")
    print(f"dev unknown reject : {config[2]*100.0:.2f}%")
    print(f"dev raw augmented  : {config[3]*100.0:.2f}%")
    print(f"top_k              : {config[4]}")
    print(f"max_weight         : {config[5]:.2f}")
    print(f"agreement bonus    : {config[6]:.3f}")
    print(f"aug sim weight     : {config[7]:.3f}")
    print(f"base sim weight    : {config[8]:.3f}")
    print(f"aug margin weight  : {config[9]:.3f}")
    print(f"base margin weight : {config[10]:.3f}")
    print(f"evidence threshold : {config[11]:.6f}")

    result = evaluate(
        baseline_prototypes,
        augmented_prototypes,
        config,
        final_known,
        final_unknown,
    )

    raw, known_accept, unknown_reject, balanced, known_results, unknown_results = result

    print()
    print("=" * 116)
    print(" FINAL-V4")
    print("=" * 116)
    print(f"Raw augmented accuracy : {raw*100.0:.2f}%")
    print(f"Known accept           : {known_accept*100.0:.2f}%")
    print(f"Unknown reject         : {unknown_reject*100.0:.2f}%")
    print(f"Balanced score         : {balanced*100.0:.2f}%")
    print()

    print("Known details")
    print("-" * 116)
    for label, text, r in known_results:
        state = "ACCEPT" if (r.accepted and r.predicted_main == label) else (
            "MISROUTE" if r.predicted_main != label else "REVIEW"
        )
        print(
            f"[{state:<8}] expected={label} predicted={r.predicted_main} "
            f"base={r.baseline_main} aug={r.augmented_main} agree={r.agreed} "
            f"evidence={r.evidence_score:.6f} "
            f"sim=({r.baseline_similarity:.3f},{r.augmented_similarity:.3f}) "
            f"margin=({r.baseline_margin:+.3f},{r.augmented_margin:+.3f}) "
            f"text={text}"
        )

    print()
    print("Unknown details")
    print("-" * 116)
    for text, r in unknown_results:
        print(
            f"[{'UNKNOWN' if not r.accepted else 'FALSE_ACCEPT':<12}] "
            f"predicted={r.predicted_main} base={r.baseline_main} aug={r.augmented_main} "
            f"agree={r.agreed} evidence={r.evidence_score:.6f} text={text}"
        )

    passed = (
        raw >= 0.80
        and known_accept >= 0.50
        and unknown_reject >= 0.80
    )

    print()
    print("=" * 116)
    print(
        "RESULT:",
        "PASS" if passed else "EXPERIMENTAL_FAIL",
        f"raw={raw*100.0:.2f}%",
        f"known_accept={known_accept*100.0:.2f}%",
        f"unknown_reject={unknown_reject*100.0:.2f}%",
        f"balanced={balanced*100.0:.2f}%",
    )
    print("=" * 116)

    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
