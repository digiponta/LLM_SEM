#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LLM_SEM v0.18.8 Dual-Router Consensus Gate Experiment.

Calibration:
  original DEV known (2 per NDC class) + 6 DEV unknowns

Evaluation:
  NEW FINAL-V3 known (30) + NEW FINAL-V3 unknown (6)

Compared systems:
  A) TRAIN-only baseline router
  B) confusion-aware augmented router
  C) dual-router consensus gate

The FINAL-V2 set from v0.18.7 is not reused as a fresh final benchmark.
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
from ndc_consensus_v0188 import route_consensus


FINAL_V3: Tuple[Tuple[str, str], ...] = (
    ("0", "人工知能モデルの基本構造"),
    ("0", "コンピュータで情報を管理する方法"),
    ("0", "プログラムを実行する計算環境"),
    ("1", "善悪を判断する考え方"),
    ("1", "人間の心理と認知機能"),
    ("1", "論理的推論の方法"),
    ("2", "戦国時代の歴史を知りたい"),
    ("2", "歴史上の人物の生涯を調べる"),
    ("2", "ある地域の歴史的変化"),
    ("3", "景気変動と経済政策"),
    ("3", "教育制度の社会的役割"),
    ("3", "法律と社会秩序について"),
    ("4", "宇宙に存在する銀河の研究"),
    ("4", "物質の化学的な変化"),
    ("4", "生物の進化と遺伝"),
    ("5", "電気回路を作る工学分野"),
    ("5", "機械製品を設計する技術"),
    ("5", "建築物の構造設計"),
    ("6", "鉄道や物流による輸送"),
    ("6", "商品流通と小売業"),
    ("6", "農業による食料生産"),
    ("7", "絵画や彫刻などの美術"),
    ("7", "音楽を演奏して表現する"),
    ("7", "競技として行うスポーツ"),
    ("8", "英語の語彙と文法"),
    ("8", "日本語の文法構造"),
    ("8", "文章を外国語へ翻訳する"),
    ("9", "小説という文学作品"),
    ("9", "詩の言葉と文学表現"),
    ("9", "作家が書いた文学作品"),
)

UNKNOWN_DEV: Tuple[str, ...] = (
    "これはどういうこと",
    "それの説明",
    "XYZABC",
    "???",
    "未定義対象ガンマ",
    "意味不明な文字列qqq",
)

UNKNOWN_FINAL_V3: Tuple[str, ...] = (
    "それは何ですか",
    "もう少し詳しく",
    "2718281828",
    "もやもや概念",
    "対象が書かれていない質問",
    "無意味文字列asdfgh",
)


def parse_args():
    p = argparse.ArgumentParser(
        description="LLM_SEM v0.18.8 dual-router consensus experiment"
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
def calibrate_single(prototypes, dev_known, dev_unknown):
    best = None
    for top_k in (1, 2, 3):
        for max_weight in (0.40, 0.60, 0.80, 1.00):
            known_raw = [
                route_vector_multi(
                    v, prototypes,
                    similarity_threshold=-1.0,
                    margin_threshold=-1.0,
                    top_k=top_k,
                    max_weight=max_weight,
                )
                for _, _, v in dev_known
            ]
            unknown_raw = [
                route_vector_multi(
                    v, prototypes,
                    similarity_threshold=-1.0,
                    margin_threshold=-1.0,
                    top_k=top_k,
                    max_weight=max_weight,
                )
                for _, v in dev_unknown
            ]
            raw = sum(
                r.predicted_main == label
                for (label, _, _), r in zip(dev_known, known_raw)
            ) / len(dev_known)

            for sim_th in frange(0.50, 0.95, 0.01):
                for margin_th in frange(0.00, 0.15, 0.005):
                    known_accept = sum(
                        r.predicted_main == label
                        and r.nearest_similarity >= sim_th
                        and r.class_margin >= margin_th
                        for (label, _, _), r in zip(dev_known, known_raw)
                    ) / len(dev_known)
                    unknown_reject = sum(
                        r.nearest_similarity < sim_th
                        or r.class_margin < margin_th
                        for r in unknown_raw
                    ) / len(dev_unknown)
                    balanced = 0.5 * known_accept + 0.5 * unknown_reject
                    candidate = (
                        balanced, raw, known_accept, unknown_reject,
                        top_k, max_weight, sim_th, margin_th
                    )
                    if best is None or candidate > best:
                        best = candidate
    return best


@torch.no_grad()
def calibrate_consensus(
    baseline_prototypes,
    augmented_prototypes,
    dev_known,
    dev_unknown,
):
    best = None
    for top_k in (1, 2, 3):
        for max_weight in (0.60, 0.80, 1.00):
            for sim_th in frange(0.65, 0.90, 0.01):
                for margin_th in frange(0.00, 0.08, 0.005):
                    known_results = [
                        route_consensus(
                            v,
                            baseline_prototypes,
                            augmented_prototypes,
                            top_k=top_k,
                            max_weight=max_weight,
                            similarity_threshold=sim_th,
                            margin_threshold=margin_th,
                        )
                        for _, _, v in dev_known
                    ]
                    unknown_results = [
                        route_consensus(
                            v,
                            baseline_prototypes,
                            augmented_prototypes,
                            top_k=top_k,
                            max_weight=max_weight,
                            similarity_threshold=sim_th,
                            margin_threshold=margin_th,
                        )
                        for _, v in dev_unknown
                    ]

                    raw_consensus = sum(
                        r.agreed and r.predicted_main == label
                        for (label, _, _), r in zip(dev_known, known_results)
                    ) / len(dev_known)

                    known_accept = sum(
                        r.accepted and r.predicted_main == label
                        for (label, _, _), r in zip(dev_known, known_results)
                    ) / len(dev_known)

                    unknown_reject = sum(
                        not r.accepted for r in unknown_results
                    ) / len(dev_unknown)

                    balanced = 0.5 * known_accept + 0.5 * unknown_reject

                    # Prefer acceptance if balanced ties; consensus already
                    # protects against disagreement.
                    candidate = (
                        balanced,
                        known_accept,
                        unknown_reject,
                        raw_consensus,
                        top_k,
                        max_weight,
                        sim_th,
                        margin_th,
                    )
                    if best is None or candidate > best:
                        best = candidate
    return best


@torch.no_grad()
def evaluate_single(prototypes, config, final_known, final_unknown):
    _, _, _, _, top_k, max_weight, sim_th, margin_th = config

    known_results = []
    for label, text, v in final_known:
        r = route_vector_multi(
            v, prototypes,
            similarity_threshold=sim_th,
            margin_threshold=margin_th,
            top_k=top_k,
            max_weight=max_weight,
        )
        known_results.append((label, text, r))

    unknown_results = []
    for text, v in final_unknown:
        r = route_vector_multi(
            v, prototypes,
            similarity_threshold=sim_th,
            margin_threshold=margin_th,
            top_k=top_k,
            max_weight=max_weight,
        )
        unknown_results.append((text, r))

    raw = sum(r.predicted_main == label for label, _, r in known_results) / len(known_results)
    known_accept = sum(
        r.predicted_main == label and r.accepted
        for label, _, r in known_results
    ) / len(known_results)
    unknown_reject = sum(not r.accepted for _, r in unknown_results) / len(unknown_results)
    balanced = 0.5 * known_accept + 0.5 * unknown_reject
    return raw, known_accept, unknown_reject, balanced, known_results, unknown_results


@torch.no_grad()
def evaluate_consensus(
    baseline_prototypes,
    augmented_prototypes,
    config,
    final_known,
    final_unknown,
):
    _, _, _, _, top_k, max_weight, sim_th, margin_th = config

    known_results = []
    for label, text, v in final_known:
        r = route_consensus(
            v,
            baseline_prototypes,
            augmented_prototypes,
            top_k=top_k,
            max_weight=max_weight,
            similarity_threshold=sim_th,
            margin_threshold=margin_th,
        )
        known_results.append((label, text, r))

    unknown_results = []
    for text, v in final_unknown:
        r = route_consensus(
            v,
            baseline_prototypes,
            augmented_prototypes,
            top_k=top_k,
            max_weight=max_weight,
            similarity_threshold=sim_th,
            margin_threshold=margin_th,
        )
        unknown_results.append((text, r))

    agreement = sum(r.agreed for _, _, r in known_results) / len(known_results)
    raw = sum(
        r.agreed and r.predicted_main == label
        for label, _, r in known_results
    ) / len(known_results)
    known_accept = sum(
        r.accepted and r.predicted_main == label
        for label, _, r in known_results
    ) / len(known_results)
    unknown_reject = sum(not r.accepted for _, r in unknown_results) / len(unknown_results)
    balanced = 0.5 * known_accept + 0.5 * unknown_reject

    return (
        raw, known_accept, unknown_reject, balanced,
        agreement, known_results, unknown_results
    )


def print_single(name, config, result):
    raw, ka, ur, bal, _, _ = result
    print()
    print("=" * 116)
    print(name)
    print("=" * 116)
    print(f"top_k              : {config[4]}")
    print(f"max_weight         : {config[5]:.2f}")
    print(f"similarity th      : {config[6]:.3f}")
    print(f"margin th          : {config[7]:.3f}")
    print(f"FINAL-V3 raw       : {raw*100.0:.2f}%")
    print(f"FINAL-V3 known     : {ka*100.0:.2f}%")
    print(f"FINAL-V3 unknown   : {ur*100.0:.2f}%")
    print(f"FINAL-V3 balanced  : {bal*100.0:.2f}%")
    return raw, ka, ur, bal


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
    final_known = encode_labeled(model, tokenizer, FINAL_V3, args.pooling)
    final_unknown = encode_unknown(model, tokenizer, UNKNOWN_FINAL_V3, args.pooling)

    baseline_prototypes = build_prototypes(
        model, tokenizer,
        seeds=TRAIN_ONLY_NDC_SEEDS,
        pooling=args.pooling,
    )
    augmented_prototypes = build_prototypes(
        model, tokenizer,
        seeds=AUGMENTED_NDC_SEEDS,
        pooling=args.pooling,
    )

    print("=" * 116)
    print(" LLM_SEM v0.18.8 Dual-Router Consensus Gate Experiment")
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
    print("FINAL-V3 known/unknown:", f"{len(final_known)}/{len(final_unknown)}")

    b_cfg = calibrate_single(baseline_prototypes, dev_known, dev_unknown)
    a_cfg = calibrate_single(augmented_prototypes, dev_known, dev_unknown)
    c_cfg = calibrate_consensus(
        baseline_prototypes,
        augmented_prototypes,
        dev_known,
        dev_unknown,
    )

    b_res = evaluate_single(
        baseline_prototypes, b_cfg, final_known, final_unknown
    )
    a_res = evaluate_single(
        augmented_prototypes, a_cfg, final_known, final_unknown
    )
    c_res = evaluate_consensus(
        baseline_prototypes,
        augmented_prototypes,
        c_cfg,
        final_known,
        final_unknown,
    )

    b = print_single("A) TRAIN-ONLY BASELINE", b_cfg, b_res)
    a = print_single("B) CONFUSION-AWARE AUGMENTED", a_cfg, a_res)

    raw, ka, ur, bal, agreement, known_results, unknown_results = c_res

    print()
    print("=" * 116)
    print("C) DUAL-ROUTER CONSENSUS GATE")
    print("=" * 116)
    print(f"top_k              : {c_cfg[4]}")
    print(f"max_weight         : {c_cfg[5]:.2f}")
    print(f"similarity th      : {c_cfg[6]:.3f}")
    print(f"margin th          : {c_cfg[7]:.3f}")
    print(f"Router agreement   : {agreement*100.0:.2f}%")
    print(f"FINAL-V3 raw       : {raw*100.0:.2f}%")
    print(f"FINAL-V3 known     : {ka*100.0:.2f}%")
    print(f"FINAL-V3 unknown   : {ur*100.0:.2f}%")
    print(f"FINAL-V3 balanced  : {bal*100.0:.2f}%")

    print()
    print("Consensus known details")
    print("-" * 116)
    for label, text, r in known_results:
        state = "ACCEPT" if (r.accepted and r.predicted_main == label) else (
            "MISROUTE" if (r.agreed and r.predicted_main != label) else "UNKNOWN"
        )
        print(
            f"[{state:<8}] expected={label} "
            f"base={r.baseline_main} aug={r.augmented_main} "
            f"agree={r.agreed} "
            f"sim=({r.baseline_similarity:.3f},{r.augmented_similarity:.3f}) "
            f"margin=({r.baseline_margin:+.3f},{r.augmented_margin:+.3f}) "
            f"text={text}"
        )

    print()
    print("=" * 116)
    print(" CONSENSUS DELTA")
    print("=" * 116)
    print(f"vs baseline known  : {(ka-b[1])*100.0:+.2f} pp")
    print(f"vs baseline unknown: {(ur-b[2])*100.0:+.2f} pp")
    print(f"vs baseline balance: {(bal-b[3])*100.0:+.2f} pp")
    print(f"vs augmented known : {(ka-a[1])*100.0:+.2f} pp")
    print(f"vs augmented unknown: {(ur-a[2])*100.0:+.2f} pp")
    print(f"vs augmented balance: {(bal-a[3])*100.0:+.2f} pp")

    passed = (
        bal >= max(b[3], a[3])
        and ur >= 0.80
        and ka >= max(b[1], a[1])
    )

    print("=" * 116)
    print(
        "RESULT:",
        "PASS" if passed else "EXPERIMENTAL_FAIL",
        f"consensus_known={ka*100.0:.2f}%",
        f"consensus_unknown={ur*100.0:.2f}%",
        f"consensus_balanced={bal*100.0:.2f}%",
    )
    print("=" * 116)

    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
