#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LLM_SEM v0.18.5 Regularized Residual NDC Projection Experiment.

Changes from v0.18.4:
- Base semantic vector is correctly treated as 64-dimensional.
- Projection is near-identity 64 -> 32 -> 64 residual, not 64 -> 128 -> 64.
- Training/dev/test roles are separated.
- Seed/model selection uses development data only.
- Final known holdout is untouched until the final report.
- Loss includes geometry preservation to prevent collapse/overfit.

Training data:
  4 seed texts per NDC class = 40
Development data:
  2 seed texts per NDC class = 20
Final known holdout:
  30 independent paraphrases
Unknown:
  6 dev probes + 6 final probes
"""

from __future__ import annotations

import argparse
import copy
import math
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F

from model import LanguageModel
from tokenizer import Tokenizer
from ndc import NDC_MAIN
from ndc_semantic_router_v0182 import NDC_MAIN_SEEDS, encode_text
from ndc_semantic_router_v0183 import route_vector_multi
from ndc_projection_v0185 import (
    ResidualNDCProjection,
    build_projected_prototypes,
    project_vector,
    save_projection_checkpoint,
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


def parse_args():
    p = argparse.ArgumentParser(
        description="LLM_SEM v0.18.6 leakage-free residual NDC projection experiment"
    )
    p.add_argument("--model", default="model/model-sem-internalized-v01575.pt")
    p.add_argument("--tokenizer", default="model/tokenizer.json")
    p.add_argument("--output", default="model/ndc-projection-v0186.pt")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--pooling", default="mean")
    p.add_argument("--bottleneck-dim", type=int, default=32)
    p.add_argument("--alpha", type=float, default=0.25)
    p.add_argument("--epochs", type=int, default=250)
    p.add_argument("--lr", type=float, default=3.0e-4)
    p.add_argument("--weight-decay", type=float, default=1.0e-4)
    p.add_argument("--class-weight", type=float, default=1.00)
    p.add_argument("--geometry-weight", type=float, default=1.50)
    p.add_argument("--compact-weight", type=float, default=0.10)
    p.add_argument("--separation-weight", type=float, default=0.05)
    p.add_argument("--separation-margin", type=float, default=0.20)
    p.add_argument("--seeds", default="1,2,3,4,5")
    p.add_argument("--patience", type=int, default=40)
    p.add_argument("--report-every", type=int, default=50)
    return p.parse_args()


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(name)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def encode_rows(model, tokenizer, rows, pooling):
    return [
        (label, text, encode_text(model, tokenizer, text, pooling=pooling))
        for label, text in rows
    ]


def split_seed_texts():
    train_rows = []
    dev_rows = []
    for main in sorted(NDC_MAIN_SEEDS):
        texts = list(NDC_MAIN_SEEDS[main])
        train_rows.extend((main, text) for text in texts[:4])
        dev_rows.extend((main, text) for text in texts[4:])
    return train_rows, dev_rows


def stack_rows(encoded_rows):
    x = torch.stack([row[2] for row in encoded_rows])
    y = torch.tensor(
        [int(row[0]) for row in encoded_rows],
        dtype=torch.long,
        device=x.device,
    )
    return x, y


def class_centers(z, labels):
    centers = []
    for class_id in range(10):
        members = z[labels == class_id]
        centers.append(F.normalize(members.mean(dim=0), p=2, dim=-1))
    return torch.stack(centers)


def losses(head, x, y, base_pairwise, args):
    z = head(x)
    centers = class_centers(z, y)

    logits = z @ centers.t()
    ce = F.cross_entropy(logits / 0.08, y)

    own = centers[y]
    compact = (1.0 - (z * own).sum(dim=-1)).mean()

    center_sim = centers @ centers.t()
    mask = ~torch.eye(10, dtype=torch.bool, device=z.device)
    separation = F.relu(
        center_sim[mask] - args.separation_margin
    ).mean()

    projected_pairwise = z @ z.t()
    geometry = F.mse_loss(projected_pairwise, base_pairwise)

    total = (
        args.class_weight * ce
        + args.geometry_weight * geometry
        + args.compact_weight * compact
        + args.separation_weight * separation
    )
    return total, ce, geometry, compact, separation


@torch.no_grad()
def nearest_center_accuracy(head, x, y):
    z = head(x)
    centers = class_centers(z, y)
    pred = (z @ centers.t()).argmax(dim=-1)
    return float((pred == y).float().mean().item())


@torch.no_grad()
def build_base_prototypes(model, tokenizer, pooling, train_rows):
    """Build runtime prototypes strictly from TRAIN rows.

    v0.18.5 accidentally used all NDC_MAIN_SEEDS, including the two DEV
    examples per class. That made DEV routing artificially perfect and pushed
    threshold calibration toward severe over-rejection on the final test.
    """
    grouped = {main: [] for main in sorted(NDC_MAIN)}
    for main, text in train_rows:
        grouped[main].append(
            encode_text(model, tokenizer, text, pooling=pooling)
        )
    return {
        main: tuple(vectors)
        for main, vectors in grouped.items()
    }


def frange(start, stop, step):
    values = []
    x = start
    while x <= stop + 1e-12:
        values.append(round(x, 6))
        x += step
    return values


@torch.no_grad()
def calibrate_on_dev(
    head,
    base_prototypes,
    dev_known,
    dev_unknown,
):
    projected_prototypes = build_projected_prototypes(head, base_prototypes)
    dev_known_proj = [
        (label, text, project_vector(head, vector))
        for label, text, vector in dev_known
    ]
    dev_unknown_proj = [
        (text, project_vector(head, vector))
        for text, vector in dev_unknown
    ]

    best = None
    for top_k in (1, 2, 3):
        for max_weight in (0.40, 0.60, 0.80, 1.00):
            known_results = [
                route_vector_multi(
                    vector,
                    projected_prototypes,
                    similarity_threshold=-1.0,
                    margin_threshold=-1.0,
                    top_k=top_k,
                    max_weight=max_weight,
                )
                for _, _, vector in dev_known_proj
            ]
            unknown_results = [
                route_vector_multi(
                    vector,
                    projected_prototypes,
                    similarity_threshold=-1.0,
                    margin_threshold=-1.0,
                    top_k=top_k,
                    max_weight=max_weight,
                )
                for _, vector in dev_unknown_proj
            ]

            raw = sum(
                r.predicted_main == label
                for (label, _, _), r in zip(dev_known_proj, known_results)
            ) / len(dev_known_proj)

            for sim_th in frange(0.50, 0.95, 0.01):
                for margin_th in frange(0.00, 0.15, 0.005):
                    known_accept = sum(
                        r.predicted_main == label
                        and r.nearest_similarity >= sim_th
                        and r.class_margin >= margin_th
                        for (label, _, _), r in zip(dev_known_proj, known_results)
                    ) / len(dev_known_proj)

                    unknown_reject = sum(
                        r.nearest_similarity < sim_th
                        or r.class_margin < margin_th
                        for r in unknown_results
                    ) / len(dev_unknown_proj)

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
def final_test(
    head,
    base_prototypes,
    known_test,
    unknown_test,
    config,
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

    projected_prototypes = build_projected_prototypes(head, base_prototypes)

    known_results = []
    for label, text, vector in known_test:
        result = route_vector_multi(
            project_vector(head, vector),
            projected_prototypes,
            similarity_threshold=sim_th,
            margin_threshold=margin_th,
            top_k=top_k,
            max_weight=max_weight,
        )
        known_results.append((label, text, result))

    unknown_results = []
    for text, vector in unknown_test:
        result = route_vector_multi(
            project_vector(head, vector),
            projected_prototypes,
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


def main():
    args = parse_args()
    device = choose_device(args.device)

    tokenizer = Tokenizer.load(args.tokenizer)
    model, checkpoint = LanguageModel.load_checkpoint(args.model, device=device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    train_rows, dev_rows = split_seed_texts()
    train_encoded = encode_rows(model, tokenizer, train_rows, args.pooling)
    dev_encoded = encode_rows(model, tokenizer, dev_rows, args.pooling)
    test_encoded = encode_rows(model, tokenizer, KNOWN_HOLDOUT, args.pooling)

    dev_unknown_texts = UNKNOWN_PROBES[:6]
    test_unknown_texts = UNKNOWN_PROBES[6:]
    dev_unknown = [
        (text, encode_text(model, tokenizer, text, pooling=args.pooling))
        for text in dev_unknown_texts
    ]
    test_unknown = [
        (text, encode_text(model, tokenizer, text, pooling=args.pooling))
        for text in test_unknown_texts
    ]

    x_train, y_train = stack_rows(train_encoded)
    x_dev, y_dev = stack_rows(dev_encoded)
    base_pairwise = F.normalize(x_train, p=2, dim=-1) @ F.normalize(x_train, p=2, dim=-1).t()
    base_prototypes = build_base_prototypes(
        model,
        tokenizer,
        args.pooling,
        train_rows,
    )

    print("=" * 116)
    print(" LLM_SEM v0.18.6 Leakage-Free Residual NDC Projection Experiment")
    print("=" * 116)
    print("Device             :", device)
    if device.type == "cuda":
        print("GPU                :", torch.cuda.get_device_name(device))
    print("Base model         :", args.model)
    print("Checkpoint loss    :", checkpoint.get("loss"))
    print("Base frozen        :", all(not p.requires_grad for p in model.parameters()))
    print("Base vector dim    :", model.d_model)
    print("Projection         :", f"{model.d_model} -> {args.bottleneck_dim} -> {model.d_model} residual")
    print("Residual alpha     :", args.alpha)
    print("Train/dev/test     :", f"{len(train_rows)}/{len(dev_rows)}/{len(KNOWN_HOLDOUT)}")
    print("Prototype source   : TRAIN ONLY")
    print("Unknown dev/test   :", f"{len(dev_unknown)}/{len(test_unknown)}")
    print()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    best = None

    for seed in seeds:
        set_seed(seed)
        head = ResidualNDCProjection(
            input_dim=model.d_model,
            bottleneck_dim=args.bottleneck_dim,
            alpha=args.alpha,
        ).to(device)

        optimizer = torch.optim.AdamW(
            head.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

        best_dev_acc = -1.0
        best_state = copy.deepcopy(head.state_dict())
        stale = 0

        for epoch in range(1, args.epochs + 1):
            head.train()
            optimizer.zero_grad(set_to_none=True)
            total, ce, geometry, compact, separation = losses(
                head, x_train, y_train, base_pairwise, args
            )
            total.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()

            head.eval()
            with torch.no_grad():
                z_train = head(x_train)
                centers = class_centers(z_train, y_train)
                dev_z = head(x_dev)
                dev_pred = (dev_z @ centers.t()).argmax(dim=-1)
                dev_acc = float((dev_pred == y_dev).float().mean().item())

            if dev_acc > best_dev_acc + 1e-9:
                best_dev_acc = dev_acc
                best_state = copy.deepcopy(head.state_dict())
                stale = 0
            else:
                stale += 1

            if epoch == 1 or epoch % args.report_every == 0:
                print(
                    f"seed={seed} epoch={epoch:>3} loss={float(total.item()):.6f} "
                    f"ce={float(ce.item()):.6f} geom={float(geometry.item()):.6f} "
                    f"compact={float(compact.item()):.6f} sep={float(separation.item()):.6f} "
                    f"dev={dev_acc*100.0:5.1f}%"
                )

            if stale >= args.patience:
                break

        head.load_state_dict(best_state)
        head.eval()

        config = calibrate_on_dev(
            head,
            base_prototypes,
            dev_encoded,
            dev_unknown,
        )
        dev_balanced, dev_raw, dev_known, dev_unknown_rate, *_ = config

        print(
            f"seed={seed} dev: raw={dev_raw*100.0:.2f}% "
            f"known_accept={dev_known*100.0:.2f}% "
            f"unknown_reject={dev_unknown_rate*100.0:.2f}% "
            f"balanced={dev_balanced*100.0:.2f}%"
        )

        candidate = (dev_balanced, dev_raw, dev_known, dev_unknown_rate, -seed, head, config)
        if best is None or candidate[:5] > best[:5]:
            best = candidate

    assert best is not None
    _, _, _, _, _, best_head, best_config = best

    (
        raw,
        known_accept,
        unknown_reject,
        balanced,
        known_results,
        unknown_results,
    ) = final_test(
        best_head,
        base_prototypes,
        test_encoded,
        test_unknown,
        best_config,
    )

    print()
    print("=" * 116)
    print(" FINAL UNTOUCHED TEST")
    print("=" * 116)
    print(f"Raw accuracy       : {raw*100.0:.2f}%")
    print(f"Known accept       : {known_accept*100.0:.2f}%")
    print(f"Unknown reject     : {unknown_reject*100.0:.2f}%")
    print(f"Balanced score     : {balanced*100.0:.2f}%")
    print()

    print("Known test details")
    print("-" * 116)
    for label, text, result in known_results:
        state = "ACCEPT" if (result.accepted and result.predicted_main == label) else (
            "MISROUTE" if result.predicted_main != label else "REVIEW"
        )
        print(
            f"[{state:<8}] expected={label} predicted={result.predicted_main} "
            f"nearest={result.nearest_similarity:.6f} margin={result.class_margin:+.6f} "
            f"text={text}"
        )

    print()
    print("Unknown final details")
    print("-" * 116)
    for text, result in unknown_results:
        print(
            f"[{'UNKNOWN' if not result.accepted else 'FALSE_ACCEPT':<12}] "
            f"predicted={result.predicted_main} nearest={result.nearest_similarity:.6f} "
            f"margin={result.class_margin:+.6f} text={text}"
        )

    metadata = {
        "base_model": args.model,
        "base_checkpoint_loss": checkpoint.get("loss"),
        "base_frozen": True,
        "base_dim": model.d_model,
        "architecture": f"{model.d_model}->{args.bottleneck_dim}->{model.d_model} residual",
        "alpha": args.alpha,
        "train_count": len(train_rows),
        "dev_count": len(dev_rows),
        "test_count": len(KNOWN_HOLDOUT),
        "raw_accuracy": raw,
        "known_accept": known_accept,
        "unknown_reject": unknown_reject,
        "balanced_score": balanced,
        "dev_calibration": {
            "top_k": best_config[4],
            "max_weight": best_config[5],
            "similarity_threshold": best_config[6],
            "margin_threshold": best_config[7],
        },
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    save_projection_checkpoint(args.output, best_head, metadata=metadata)

    passed = (
        raw >= 0.70
        and known_accept >= 0.40
        and unknown_reject >= 0.80
    )

    print()
    print("Projection checkpoint:", args.output)
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
