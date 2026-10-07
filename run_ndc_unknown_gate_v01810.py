#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LLM_SEM v0.18.10 Contrastive Unknown Gate Experiment.

Classifier:
  fixed confusion-aware augmented NDC prototype router

Unknown detector:
  contrastive known-vs-unknown prototype gate

Calibration:
  original DEV known (20)
  original DEV unknown (6)

Evaluation:
  NEW FINAL-V5 known (30)
  NEW FINAL-V5 unknown (6)

No projection training is performed. Base LLM remains frozen.
"""

from __future__ import annotations

import argparse
from typing import Tuple

import torch

from model import LanguageModel
from tokenizer import Tokenizer
from ndc_semantic_router_v0182 import NDC_MAIN_SEEDS, encode_text
from ndc_semantic_router_v0183 import build_prototypes
from ndc_prototypes_v0187 import AUGMENTED_NDC_SEEDS
from ndc_unknown_gate_v01810 import route_contrastive


FINAL_V5: Tuple[Tuple[str, str], ...] = (
    ("0", "情報技術でデータを処理する"),
    ("0", "AIモデルをコンピュータ上で動かす"),
    ("0", "ソフトウェア開発の基本"),
    ("1", "倫理と価値観について考える"),
    ("1", "心理学で心の働きを研究する"),
    ("1", "論理学の推論方法"),
    ("2", "近代日本の歴史を学ぶ"),
    ("2", "歴史人物の経歴を調査する"),
    ("2", "地域の歴史的背景"),
    ("3", "経済政策と市場の動向"),
    ("3", "学校教育と社会制度"),
    ("3", "法律と社会のルール"),
    ("4", "宇宙の星や銀河を研究する"),
    ("4", "化学物質の反応を観察する"),
    ("4", "遺伝と生物進化を研究する"),
    ("5", "電子回路の設計技術"),
    ("5", "機械を開発する工学"),
    ("5", "建築構造を設計する"),
    ("6", "物流と輸送を行う産業"),
    ("6", "商品の流通と販売"),
    ("6", "農業で作物を生産する"),
    ("7", "美術作品を鑑賞する"),
    ("7", "音楽を演奏する芸術活動"),
    ("7", "スポーツ競技に参加する"),
    ("8", "英語の文法と語彙"),
    ("8", "日本語の文法を調べる"),
    ("8", "外国語へ文章を翻訳する"),
    ("9", "小説という文学作品"),
    ("9", "詩の文学的な表現"),
    ("9", "文学作品と作家について"),
)

UNKNOWN_DEV: Tuple[str, ...] = (
    "これはどういうこと",
    "それの説明",
    "XYZABC",
    "???",
    "未定義対象ガンマ",
    "意味不明な文字列qqq",
)

UNKNOWN_FINAL_V5: Tuple[str, ...] = (
    "それについてお願いします",
    "何を意味しますか",
    "1414213562",
    "あいまいな何か",
    "対象が不明な説明",
    "無意味列lkjhgf",
)


def parse_args():
    p = argparse.ArgumentParser(
        description="LLM_SEM v0.18.10 contrastive unknown gate experiment"
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
def calibrate(
    augmented_prototypes,
    unknown_prototypes,
    dev_known,
    dev_unknown,
):
    best = None

    for contrast_w in (0.5, 1.0, 1.5, 2.0):
        for margin_w in (0.0, 0.5, 1.0, 1.5):
            for known_w in (0.0, 0.25, 0.5):
                known_results = [
                    route_contrastive(
                        v,
                        augmented_prototypes,
                        unknown_prototypes,
                        top_k=3,
                        max_weight=1.0,
                        contrast_weight=contrast_w,
                        margin_weight=margin_w,
                        known_similarity_weight=known_w,
                        gate_threshold=-999.0,
                    )
                    for _, _, v in dev_known
                ]
                unknown_results = [
                    route_contrastive(
                        v,
                        augmented_prototypes,
                        unknown_prototypes,
                        top_k=3,
                        max_weight=1.0,
                        contrast_weight=contrast_w,
                        margin_weight=margin_w,
                        known_similarity_weight=known_w,
                        gate_threshold=-999.0,
                    )
                    for _, v in dev_unknown
                ]

                raw = sum(
                    r.predicted_main == label
                    for (label, _, _), r in zip(dev_known, known_results)
                ) / len(dev_known)

                values = sorted({
                    round(r.gate_score, 6)
                    for r in known_results + unknown_results
                })

                thresholds = [values[0] - 1e-6]
                thresholds += [
                    (a + b) / 2.0
                    for a, b in zip(values, values[1:])
                ]
                thresholds.append(values[-1] + 1e-6)

                for threshold in thresholds:
                    known_accept = sum(
                        r.predicted_main == label
                        and r.gate_score >= threshold
                        for (label, _, _), r in zip(dev_known, known_results)
                    ) / len(dev_known)

                    unknown_reject = sum(
                        r.gate_score < threshold
                        for r in unknown_results
                    ) / len(dev_unknown)

                    balanced = 0.5 * known_accept + 0.5 * unknown_reject

                    candidate = (
                        balanced,
                        known_accept,
                        unknown_reject,
                        raw,
                        contrast_w,
                        margin_w,
                        known_w,
                        threshold,
                    )
                    if best is None or candidate > best:
                        best = candidate

    return best


@torch.no_grad()
def evaluate(
    augmented_prototypes,
    unknown_prototypes,
    config,
    final_known,
    final_unknown,
):
    (
        _dev_balanced,
        _dev_known,
        _dev_unknown,
        _dev_raw,
        contrast_w,
        margin_w,
        known_w,
        threshold,
    ) = config

    known_results = []
    for label, text, v in final_known:
        r = route_contrastive(
            v,
            augmented_prototypes,
            unknown_prototypes,
            top_k=3,
            max_weight=1.0,
            contrast_weight=contrast_w,
            margin_weight=margin_w,
            known_similarity_weight=known_w,
            gate_threshold=threshold,
        )
        known_results.append((label, text, r))

    unknown_results = []
    for text, v in final_unknown:
        r = route_contrastive(
            v,
            augmented_prototypes,
            unknown_prototypes,
            top_k=3,
            max_weight=1.0,
            contrast_weight=contrast_w,
            margin_weight=margin_w,
            known_similarity_weight=known_w,
            gate_threshold=threshold,
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
        not r.accepted for _, r in unknown_results
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
    final_known = encode_labeled(model, tokenizer, FINAL_V5, args.pooling)
    final_unknown = encode_unknown(model, tokenizer, UNKNOWN_FINAL_V5, args.pooling)

    augmented_prototypes = build_prototypes(
        model,
        tokenizer,
        seeds=AUGMENTED_NDC_SEEDS,
        pooling=args.pooling,
    )

    unknown_prototypes = tuple(v for _, v in dev_unknown)

    print("=" * 116)
    print(" LLM_SEM v0.18.10 Contrastive Unknown Gate Experiment")
    print("=" * 116)
    print("Device             :", device)
    if device.type == "cuda":
        print("GPU                :", torch.cuda.get_device_name(device))
    print("Model              :", args.model)
    print("Checkpoint loss    :", checkpoint.get("loss"))
    print("Base frozen        :", True)
    print("Pooling            :", args.pooling)
    print("Augmented prototypes:", sum(len(v) for v in AUGMENTED_NDC_SEEDS.values()))
    print("Unknown prototypes :", len(unknown_prototypes))
    print("DEV known/unknown  :", f"{len(dev_known)}/{len(dev_unknown)}")
    print("FINAL-V5 known/unknown:", f"{len(final_known)}/{len(final_unknown)}")
    print()

    config = calibrate(
        augmented_prototypes,
        unknown_prototypes,
        dev_known,
        dev_unknown,
    )

    print("Selected DEV gate")
    print("-" * 116)
    print(f"dev balanced       : {config[0]*100.0:.2f}%")
    print(f"dev known accept   : {config[1]*100.0:.2f}%")
    print(f"dev unknown reject : {config[2]*100.0:.2f}%")
    print(f"dev raw augmented  : {config[3]*100.0:.2f}%")
    print(f"contrast weight    : {config[4]:.2f}")
    print(f"margin weight      : {config[5]:.2f}")
    print(f"known sim weight   : {config[6]:.2f}")
    print(f"gate threshold     : {config[7]:.6f}")

    result = evaluate(
        augmented_prototypes,
        unknown_prototypes,
        config,
        final_known,
        final_unknown,
    )

    raw, known_accept, unknown_reject, balanced, known_results, unknown_results = result

    print()
    print("=" * 116)
    print(" FINAL-V5")
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
            f"known_sim={r.known_similarity:.3f} "
            f"unknown_sim={r.unknown_similarity:.3f} "
            f"contrast={r.contrast:+.3f} "
            f"margin={r.class_margin:+.3f} "
            f"score={r.gate_score:.6f} text={text}"
        )

    print()
    print("Unknown details")
    print("-" * 116)
    for text, r in unknown_results:
        print(
            f"[{'UNKNOWN' if not r.accepted else 'FALSE_ACCEPT':<12}] "
            f"predicted={r.predicted_main} known_sim={r.known_similarity:.3f} "
            f"unknown_sim={r.unknown_similarity:.3f} "
            f"contrast={r.contrast:+.3f} score={r.gate_score:.6f} text={text}"
        )

    passed = (
        raw >= 0.80
        and known_accept >= 0.60
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
