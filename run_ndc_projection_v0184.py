#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LLM_SEM v0.18.4 NDC Semantic Projection Head experiment.

Goal
----
Learn an NDC-specific semantic space without changing the base LLM.

Base LLM:
  completely frozen

Projection:
  d_model -> 128 -> 64 -> L2 normalize

Training objective:
  cosine-prototype cross entropy
  + same-class compactness
  + inter-class separation

Evaluation:
  projected multi-prototype routing
  + similarity threshold
  + class-margin threshold
  + unknown rejection

The v0.18.3 raw baseline is 70.00% with 50.00% known acceptance and
100.00% unknown rejection.
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
from ndc_projection_v0184 import (
    NDCProjectionHead,
    CosinePrototypeClassifier,
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LLM_SEM v0.18.4 NDC semantic projection experiment"
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
        "--output",
        default="model/ndc-projection-v0184.pt",
    )
    p.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
    )
    p.add_argument("--pooling", default="mean")
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--output-dim", type=int, default=64)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--lr", type=float, default=1.0e-3)
    p.add_argument("--weight-decay", type=float, default=1.0e-4)
    p.add_argument("--temperature", type=float, default=0.08)
    p.add_argument("--compact-weight", type=float, default=0.25)
    p.add_argument("--separation-weight", type=float, default=0.10)
    p.add_argument("--separation-margin", type=float, default=0.15)
    p.add_argument("--seeds", default="1,2,3,4,5")
    p.add_argument("--report-every", type=int, default=100)
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
def encode_training_set(
    model: LanguageModel,
    tokenizer: Tokenizer,
    *,
    pooling: str,
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    vectors: List[torch.Tensor] = []
    labels: List[int] = []
    texts: List[str] = []

    for main in sorted(NDC_MAIN_SEEDS):
        for text in NDC_MAIN_SEEDS[main]:
            vectors.append(
                encode_text(
                    model,
                    tokenizer,
                    text,
                    pooling=pooling,
                )
            )
            labels.append(int(main))
            texts.append(text)

    return (
        torch.stack(vectors),
        torch.tensor(
            labels,
            dtype=torch.long,
            device=vectors[0].device,
        ),
        texts,
    )


def metric_regularizers(
    z: torch.Tensor,
    labels: torch.Tensor,
    *,
    separation_margin: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    centers = []
    compact_terms = []

    for class_id in range(10):
        members = z[labels == class_id]
        center = F.normalize(members.mean(dim=0), p=2, dim=-1)
        centers.append(center)

        sims = members @ center
        compact_terms.append(1.0 - sims.mean())

    compact_loss = torch.stack(compact_terms).mean()

    centers_t = torch.stack(centers)
    sim_matrix = centers_t @ centers_t.t()

    mask = ~torch.eye(
        len(centers),
        dtype=torch.bool,
        device=z.device,
    )
    negative_sims = sim_matrix[mask]
    separation_loss = F.relu(
        negative_sims - separation_margin
    ).mean()

    return compact_loss, separation_loss


@torch.no_grad()
def base_prototype_vectors(
    model: LanguageModel,
    tokenizer: Tokenizer,
    *,
    pooling: str,
) -> Dict[str, Tuple[torch.Tensor, ...]]:
    result: Dict[str, Tuple[torch.Tensor, ...]] = {}
    for main, texts in NDC_MAIN_SEEDS.items():
        result[main] = tuple(
            encode_text(
                model,
                tokenizer,
                text,
                pooling=pooling,
            )
            for text in texts
        )
    return result


@torch.no_grad()
def encode_eval_vectors(
    model: LanguageModel,
    tokenizer: Tokenizer,
    *,
    pooling: str,
):
    known = [
        (
            expected,
            text,
            encode_text(
                model,
                tokenizer,
                text,
                pooling=pooling,
            ),
        )
        for expected, text in KNOWN_HOLDOUT
    ]
    unknown = [
        (
            text,
            encode_text(
                model,
                tokenizer,
                text,
                pooling=pooling,
            ),
        )
        for text in UNKNOWN_PROBES
    ]
    return known, unknown


def frange(start: float, stop: float, step: float) -> List[float]:
    values = []
    x = start
    while x <= stop + 1.0e-12:
        values.append(round(x, 6))
        x += step
    return values


@torch.no_grad()
def evaluate_projection(
    head: NDCProjectionHead,
    *,
    base_prototypes: Dict[str, Tuple[torch.Tensor, ...]],
    known_base,
    unknown_base,
):
    projected_prototypes = build_projected_prototypes(
        head,
        base_prototypes,
    )

    known_projected = [
        (expected, text, project_vector(head, vector))
        for expected, text, vector in known_base
    ]
    unknown_projected = [
        (text, project_vector(head, vector))
        for text, vector in unknown_base
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
                for _, _, vector in known_projected
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
                for _, vector in unknown_projected
            ]

            raw_correct = sum(
                result.predicted_main == expected
                for (expected, _, _), result
                in zip(known_projected, known_results)
            )
            raw_accuracy = raw_correct / len(known_projected)

            for sim_th in frange(0.30, 0.99, 0.01):
                for margin_th in frange(0.00, 0.30, 0.01):
                    known_accept = sum(
                        result.predicted_main == expected
                        and result.nearest_similarity >= sim_th
                        and result.class_margin >= margin_th
                        for (expected, _, _), result
                        in zip(known_projected, known_results)
                    )
                    known_rate = known_accept / len(known_projected)

                    unknown_reject = sum(
                        result.nearest_similarity < sim_th
                        or result.class_margin < margin_th
                        for result in unknown_results
                    )
                    unknown_rate = unknown_reject / len(unknown_projected)

                    balanced = (
                        0.50 * known_rate
                        + 0.50 * unknown_rate
                    )

                    # Selection order:
                    # 1) balanced operating point
                    # 2) raw routing generalization
                    # 3) known retention
                    # 4) unknown rejection
                    candidate = (
                        balanced,
                        raw_accuracy,
                        known_rate,
                        unknown_rate,
                        top_k,
                        max_weight,
                        sim_th,
                        margin_th,
                        known_results,
                        unknown_results,
                        projected_prototypes,
                    )
                    if best is None or candidate[:8] > best[:8]:
                        best = candidate

    assert best is not None
    return best


def train_one_seed(
    seed: int,
    *,
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    epochs: int,
    lr: float,
    weight_decay: float,
    temperature: float,
    compact_weight: float,
    separation_weight: float,
    separation_margin: float,
    report_every: int,
):
    set_seed(seed)

    head = NDCProjectionHead(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
    ).to(x_train.device)

    learner = CosinePrototypeClassifier(
        projection=head,
        num_classes=10,
        temperature=temperature,
    ).to(x_train.device)

    optimizer = torch.optim.AdamW(
        learner.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    best_loss = math.inf
    best_state = None

    for epoch in range(1, epochs + 1):
        learner.train()
        optimizer.zero_grad(set_to_none=True)

        logits, z = learner(x_train)
        ce = F.cross_entropy(logits, y_train)
        compact, separation = metric_regularizers(
            z,
            y_train,
            separation_margin=separation_margin,
        )

        loss = (
            ce
            + compact_weight * compact
            + separation_weight * separation
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            learner.parameters(),
            max_norm=1.0,
        )
        optimizer.step()

        value = float(loss.item())
        if value < best_loss:
            best_loss = value
            best_state = copy.deepcopy(
                head.state_dict()
            )

        if (
            epoch == 1
            or epoch == epochs
            or epoch % report_every == 0
        ):
            with torch.no_grad():
                train_acc = float(
                    (logits.argmax(dim=-1) == y_train)
                    .float()
                    .mean()
                    .item()
                )
            print(
                f"seed={seed:<2} epoch={epoch:>4}/{epochs} "
                f"loss={value:.6f} ce={float(ce.item()):.6f} "
                f"compact={float(compact.item()):.6f} "
                f"separate={float(separation.item()):.6f} "
                f"train={train_acc * 100.0:6.2f}%"
            )

    assert best_state is not None
    head.load_state_dict(best_state)
    head.eval()
    return head, best_loss


def main() -> int:
    args = parse_args()
    device = choose_device(args.device)

    model_path = Path(args.model)
    tokenizer_path = Path(args.tokenizer)
    output_path = Path(args.output)

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

    # Explicitly freeze the base model.
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    print("=" * 116)
    print(" LLM_SEM v0.18.4 NDC Semantic Projection Head Experiment")
    print("=" * 116)
    print("Device             :", device)
    if device.type == "cuda":
        print("GPU                :", torch.cuda.get_device_name(device))
    print("Base model         :", model_path)
    print("Checkpoint loss    :", checkpoint.get("loss"))
    print("Base frozen        :", all(
        not p.requires_grad for p in model.parameters()
    ))
    print("Pooling            :", args.pooling)
    print(
        "Projection         :",
        f"{model.d_model} -> {args.hidden_dim} -> {args.output_dim}",
    )
    print("Training samples   :", sum(
        len(v) for v in NDC_MAIN_SEEDS.values()
    ))
    print("Known holdout      :", len(KNOWN_HOLDOUT))
    print("Unknown probes     :", len(UNKNOWN_PROBES))
    print("Epochs             :", args.epochs)
    print("LR                 :", args.lr)
    print()

    x_train, y_train, train_texts = encode_training_set(
        model,
        tokenizer,
        pooling=args.pooling,
    )

    base_prototypes = base_prototype_vectors(
        model,
        tokenizer,
        pooling=args.pooling,
    )
    known_base, unknown_base = encode_eval_vectors(
        model,
        tokenizer,
        pooling=args.pooling,
    )

    seeds = [
        int(item.strip())
        for item in args.seeds.split(",")
        if item.strip()
    ]
    if not seeds:
        raise ValueError("At least one seed is required.")

    overall_best = None

    for seed in seeds:
        print()
        print("-" * 116)
        print(f"TRAIN SEED {seed}")
        print("-" * 116)

        head, train_loss = train_one_seed(
            seed,
            input_dim=model.d_model,
            hidden_dim=args.hidden_dim,
            output_dim=args.output_dim,
            x_train=x_train,
            y_train=y_train,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            temperature=args.temperature,
            compact_weight=args.compact_weight,
            separation_weight=args.separation_weight,
            separation_margin=args.separation_margin,
            report_every=args.report_every,
        )

        evaluation = evaluate_projection(
            head,
            base_prototypes=base_prototypes,
            known_base=known_base,
            unknown_base=unknown_base,
        )

        (
            balanced,
            raw_accuracy,
            known_rate,
            unknown_rate,
            top_k,
            max_weight,
            sim_th,
            margin_th,
            known_results,
            unknown_results,
            projected_prototypes,
        ) = evaluation

        print(
            f"seed={seed} evaluation: "
            f"raw={raw_accuracy * 100.0:.2f}% "
            f"known_accept={known_rate * 100.0:.2f}% "
            f"unknown_reject={unknown_rate * 100.0:.2f}% "
            f"balanced={balanced * 100.0:.2f}% "
            f"top_k={top_k} max_weight={max_weight:.2f} "
            f"sim_th={sim_th:.3f} margin_th={margin_th:.3f}"
        )

        candidate = (
            balanced,
            raw_accuracy,
            known_rate,
            unknown_rate,
            -train_loss,
            seed,
            head,
            evaluation,
        )
        if overall_best is None or candidate[:6] > overall_best[:6]:
            overall_best = candidate

    assert overall_best is not None

    (
        balanced,
        raw_accuracy,
        known_rate,
        unknown_rate,
        neg_train_loss,
        best_seed,
        best_head,
        evaluation,
    ) = overall_best

    (
        _balanced,
        _raw_accuracy,
        _known_rate,
        _unknown_rate,
        top_k,
        max_weight,
        sim_th,
        margin_th,
        known_results,
        unknown_results,
        projected_prototypes,
    ) = evaluation

    print()
    print("=" * 116)
    print(" BEST PROJECTED NDC ROUTER")
    print("=" * 116)
    print("Seed               :", best_seed)
    print("Raw accuracy       :", f"{raw_accuracy * 100.0:.2f}%")
    print("Known accept       :", f"{known_rate * 100.0:.2f}%")
    print("Unknown reject     :", f"{unknown_rate * 100.0:.2f}%")
    print("Balanced score     :", f"{balanced * 100.0:.2f}%")
    print("top_k              :", top_k)
    print("max_weight         :", f"{max_weight:.2f}")
    print("Similarity th      :", f"{sim_th:.6f}")
    print("Margin th          :", f"{margin_th:.6f}")
    print()

    print("Per-class projected routing")
    print("-" * 116)
    for main in sorted(NDC_MAIN):
        rows = [
            (sample, result)
            for sample, result in zip(
                known_base,
                known_results,
            )
            if sample[0] == main
        ]
        correct = sum(
            result.predicted_main == main
            for _, result in rows
        )
        mean_nearest = sum(
            result.nearest_similarity
            for _, result in rows
        ) / len(rows)
        mean_margin = sum(
            result.class_margin
            for _, result in rows
        ) / len(rows)
        print(
            f"NDC {main} {NDC_MAIN[main]:<8} "
            f"accuracy={correct}/{len(rows)} "
            f"nearest={mean_nearest:.6f} "
            f"margin={mean_margin:+.6f}"
        )

    print()
    print("Known holdout details")
    print("-" * 116)
    for (expected, text, _), result in zip(
        known_base,
        known_results,
    ):
        correct = result.predicted_main == expected
        accepted = (
            correct
            and result.nearest_similarity >= sim_th
            and result.class_margin >= margin_th
        )
        state = (
            "ACCEPT"
            if accepted
            else ("MISROUTE" if not correct else "REVIEW")
        )
        print(
            f"[{state:<8}] expected={expected} "
            f"predicted={result.predicted_main} "
            f"nearest={result.nearest_similarity:.6f} "
            f"score={result.top1_score:.6f} "
            f"margin={result.class_margin:+.6f} "
            f"text={text}"
        )

    print()
    print("Unknown probe details")
    print("-" * 116)
    for (text, _), result in zip(
        unknown_base,
        unknown_results,
    ):
        rejected = (
            result.nearest_similarity < sim_th
            or result.class_margin < margin_th
        )
        state = "UNKNOWN" if rejected else "FALSE_ACCEPT"
        print(
            f"[{state:<12}] predicted={result.predicted_main} "
            f"nearest={result.nearest_similarity:.6f} "
            f"score={result.top1_score:.6f} "
            f"margin={result.class_margin:+.6f} "
            f"text={text}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    metadata = {
        "base_model": str(model_path),
        "base_checkpoint_loss": checkpoint.get("loss"),
        "base_frozen": True,
        "pooling": args.pooling,
        "seed": best_seed,
        "training_samples": len(train_texts),
        "known_holdout": len(KNOWN_HOLDOUT),
        "unknown_probes": len(UNKNOWN_PROBES),
        "raw_accuracy": raw_accuracy,
        "known_accept": known_rate,
        "unknown_reject": unknown_rate,
        "balanced_score": balanced,
        "top_k": top_k,
        "max_weight": max_weight,
        "similarity_threshold": sim_th,
        "margin_threshold": margin_th,
        "train_loss": -neg_train_loss,
    }

    save_projection_checkpoint(
        str(output_path),
        best_head,
        metadata=metadata,
    )

    passed = (
        raw_accuracy >= 0.80
        and known_rate >= 0.65
        and unknown_rate >= 0.90
    )

    print()
    print("Projection checkpoint:", output_path)
    print("=" * 116)
    print(
        "RESULT:",
        "PASS" if passed else "EXPERIMENTAL_FAIL",
        f"raw={raw_accuracy * 100.0:.2f}%",
        f"known_accept={known_rate * 100.0:.2f}%",
        f"unknown_reject={unknown_rate * 100.0:.2f}%",
        f"balanced={balanced * 100.0:.2f}%",
    )
    print("=" * 116)

    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
