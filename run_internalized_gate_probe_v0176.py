#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LLM_SEM v0.17.6 Internalized Knowledge Gate Probe

Checks whether the frozen canonical-base semantic space can separate
the intended quantum-sensor query family from false-activation probes.

No training and no Semantic Memory lookup are used.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F

from model import LanguageModel
from tokenizer import Tokenizer


DEFAULT_MODEL = "model/model-sem-canonical-base-v0172.pt"
DEFAULT_TOKENIZER = "model/tokenizer.json"

POSITIVES = [
    "量子センサーとは",
    "量子センサとは",
    "量子センサーとは。",
    "量子センサとは？",
]

NEGATIVES = [
    "量子通信とは",
    "量子コンピュータとは",
    "量子暗号とは",
    "暗号",
    "文学とは",
    "ブラックホールとは",
]

POOLINGS = ["mean", "last", "attention", "hybrid"]


def parse_args():
    p = argparse.ArgumentParser(
        description="LLM_SEM v0.17.6 semantic gate separability probe"
    )
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    return p.parse_args()


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(name)


@torch.no_grad()
def semantic_vector(
    model: LanguageModel,
    tokenizer: Tokenizer,
    text: str,
    pooling: str,
) -> torch.Tensor:
    ids = tokenizer.encode(text, add_bos=True, add_eos=False)
    ids = ids[-model.context_length:]
    x = torch.tensor([ids], dtype=torch.long, device=next(model.parameters()).device)
    if pooling == "hybrid":
        v = model.encode_semantic(
            x,
            pooling="hybrid",
            hybrid_alpha=0.35,
            normalize_hybrid=True,
        )[0]
    else:
        v = model.encode_semantic(x, pooling=pooling)[0]
        v = F.normalize(v, p=2, dim=-1)
    return v


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())


def evaluate_pooling(
    model: LanguageModel,
    tokenizer: Tokenizer,
    pooling: str,
) -> Dict[str, object]:
    positive_vectors = [
        semantic_vector(model, tokenizer, text, pooling)
        for text in POSITIVES
    ]
    negative_vectors = [
        semantic_vector(model, tokenizer, text, pooling)
        for text in NEGATIVES
    ]

    positive_centroid = F.normalize(
        torch.stack(positive_vectors).mean(dim=0),
        p=2,
        dim=-1,
    )
    negative_centroid = F.normalize(
        torch.stack(negative_vectors).mean(dim=0),
        p=2,
        dim=-1,
    )

    rows: List[Dict[str, object]] = []
    for label, texts, vectors in (
        ("POS", POSITIVES, positive_vectors),
        ("NEG", NEGATIVES, negative_vectors),
    ):
        for text, vector in zip(texts, vectors):
            pos_sim = cosine(vector, positive_centroid)
            neg_sim = cosine(vector, negative_centroid)
            rows.append({
                "label": label,
                "text": text,
                "pos_sim": pos_sim,
                "neg_sim": neg_sim,
                "margin": pos_sim - neg_sim,
            })

    positive_margins = [
        float(row["margin"]) for row in rows if row["label"] == "POS"
    ]
    negative_margins = [
        float(row["margin"]) for row in rows if row["label"] == "NEG"
    ]

    min_positive = min(positive_margins)
    max_negative = max(negative_margins)
    gap = min_positive - max_negative

    threshold = 0.5 * (min_positive + max_negative)

    correct = 0
    for row in rows:
        predicted_positive = float(row["margin"]) >= threshold
        expected_positive = row["label"] == "POS"
        correct += int(predicted_positive == expected_positive)

    return {
        "pooling": pooling,
        "rows": rows,
        "min_positive_margin": min_positive,
        "max_negative_margin": max_negative,
        "gap": gap,
        "threshold": threshold,
        "accuracy": correct / len(rows),
        "separable": gap > 0.0,
    }


def main():
    args = parse_args()
    device = choose_device(args.device)

    model_path = Path(args.model)
    tokenizer_path = Path(args.tokenizer)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Canonical base not found: {model_path}\n"
            "Run v0.17.2 /repair first, or pass --model."
        )
    if not tokenizer_path.exists():
        raise FileNotFoundError(tokenizer_path)

    tokenizer = Tokenizer.load(str(tokenizer_path))
    model, _ = LanguageModel.load_checkpoint(str(model_path), device=device)
    model.eval()

    print("=" * 92)
    print(" LLM_SEM v0.17.6 Internalized Knowledge Gate Probe")
    print("=" * 92)
    print("Model       :", model_path)
    print("Tokenizer   :", tokenizer_path)
    print("Device      :", device)
    if device.type == "cuda":
        print("GPU         :", torch.cuda.get_device_name(device))
    print("Positives   :", len(POSITIVES))
    print("Negatives   :", len(NEGATIVES))
    print()

    results = []
    for pooling in POOLINGS:
        result = evaluate_pooling(model, tokenizer, pooling)
        results.append(result)

        print(f"POOLING={pooling}")
        print("-" * 92)
        for row in result["rows"]:
            print(
                f"[{row['label']}] {row['text']:<18s} "
                f"pos={row['pos_sim']:+.6f} "
                f"neg={row['neg_sim']:+.6f} "
                f"margin={row['margin']:+.6f}"
            )
        print(
            f"min_positive_margin : {result['min_positive_margin']:+.6f}"
        )
        print(
            f"max_negative_margin : {result['max_negative_margin']:+.6f}"
        )
        print(f"separation_gap      : {result['gap']:+.6f}")
        print(f"suggested_threshold : {result['threshold']:+.6f}")
        print(f"accuracy            : {result['accuracy']:.1%}")
        print(
            "RESULT              :",
            "SEPARABLE" if result["separable"] else "OVERLAP",
        )
        print()

    best = max(
        results,
        key=lambda item: (
            float(item["gap"]),
            float(item["accuracy"]),
        ),
    )

    print("=" * 92)
    print(" GATE PROBE RESULT")
    print("=" * 92)
    print("Best pooling         :", best["pooling"])
    print("Separation gap       :", f"{best['gap']:+.6f}")
    print("Suggested threshold  :", f"{best['threshold']:+.6f}")
    print("Accuracy             :", f"{best['accuracy']:.1%}")
    print(
        "STATUS               :",
        "GATE_SEPARABLE" if best["separable"] else "GATE_OVERLAP",
    )


if __name__ == "__main__":
    main()
