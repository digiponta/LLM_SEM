#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LLM_SEM v0.17.7 Canonical Unknown Gate Probe

Purpose:
  Determine whether the frozen canonical-base semantic space can separate
  validated KNOWN queries from UNKNOWN queries before generation.

Method:
  - protected 14 prompts are KNOWN anchors
  - known surface variants are held out from the anchor set
  - score(query) = max cosine similarity to any KNOWN anchor
  - compare mean/attention/hybrid pooling
  - report a threshold only if all tested KNOWN queries lie above all UNKNOWNs

No training and no Semantic Memory lookup are used.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F

from model import LanguageModel
from tokenizer import Tokenizer


DEFAULT_MODEL = "model/model-sem-canonical-base-v0172.pt"
DEFAULT_TOKENIZER = "model/tokenizer.json"
DEFAULT_PROTECTED = "data/protected_knowledge_v0167.jsonl"

KNOWN_VARIANTS = [
    "CPUとは、",
    "GPUとは。",
    "Pythonとは？",
    "コンピュータとは。",
    "科学とは？",
    "宇宙とは。",
    "時間とは？",
    "動物とは。",
    "天気とは？",
    "食べ物とは。",
    "交通とは？",
    "なぜGPUは高速？",
    "CPUの役割は？",
    "GPUの役割は？",
]

UNKNOWN_PROBES = [
    "こんにちは",
    "暗号",
    "暗号とは",
    "量子通信とは",
    "量子コンピュータとは",
    "量子暗号とは",
    "ブラックホールとは",
    "相対性理論とは",
    "化学とは",
    "生物学とは",
    "文学とは",
    "音楽とは",
]

POOLINGS = ["mean", "attention", "hybrid"]


def parse_args():
    p = argparse.ArgumentParser(
        description="LLM_SEM v0.17.7 Canonical Unknown Gate Probe"
    )
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    p.add_argument("--protected", default=DEFAULT_PROTECTED)
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    return p.parse_args()


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(name)


def load_prompts(path: Path) -> List[str]:
    import json
    prompts = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, 1):
            raw = raw.strip()
            if not raw:
                continue
            item = json.loads(raw)
            prompt = str(item.get("prompt", "")).strip()
            if not prompt:
                raise ValueError(f"missing prompt at line {line_no}")
            prompts.append(prompt)
    return prompts


@torch.no_grad()
def vector(
    model: LanguageModel,
    tokenizer: Tokenizer,
    text: str,
    pooling: str,
) -> torch.Tensor:
    ids = tokenizer.encode(text, add_bos=True, add_eos=False)
    ids = ids[-model.context_length:]
    x = torch.tensor(
        [ids],
        dtype=torch.long,
        device=next(model.parameters()).device,
    )
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
    return float(
        F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
    )


def nearest_score(
    query: torch.Tensor,
    anchors: List[torch.Tensor],
    anchor_texts: List[str],
) -> Dict[str, object]:
    scores = [cosine(query, anchor) for anchor in anchors]
    index = max(range(len(scores)), key=scores.__getitem__)
    return {
        "score": scores[index],
        "nearest": anchor_texts[index],
    }


def evaluate(
    model: LanguageModel,
    tokenizer: Tokenizer,
    known_anchors: List[str],
    pooling: str,
):
    anchor_vectors = [
        vector(model, tokenizer, text, pooling)
        for text in known_anchors
    ]

    # Exact protected prompts are operational KNOWNs, but variants provide
    # non-trivial holdout coverage for the threshold.
    known_tests = known_anchors + KNOWN_VARIANTS
    rows = []

    for label, texts in (
        ("KNOWN", known_tests),
        ("UNKNOWN", UNKNOWN_PROBES),
    ):
        for text in texts:
            q = vector(model, tokenizer, text, pooling)
            result = nearest_score(
                q,
                anchor_vectors,
                known_anchors,
            )
            rows.append({
                "label": label,
                "text": text,
                "score": float(result["score"]),
                "nearest": result["nearest"],
            })

    known_scores = [
        row["score"] for row in rows if row["label"] == "KNOWN"
    ]
    unknown_scores = [
        row["score"] for row in rows if row["label"] == "UNKNOWN"
    ]

    min_known = min(known_scores)
    max_unknown = max(unknown_scores)
    gap = min_known - max_unknown
    threshold = 0.5 * (min_known + max_unknown)

    correct = 0
    for row in rows:
        predicted_known = row["score"] >= threshold
        expected_known = row["label"] == "KNOWN"
        correct += int(predicted_known == expected_known)

    return {
        "pooling": pooling,
        "rows": rows,
        "min_known": min_known,
        "max_unknown": max_unknown,
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
    protected_path = Path(args.protected)

    tokenizer = Tokenizer.load(str(tokenizer_path))
    known_anchors = load_prompts(protected_path)
    model, _ = LanguageModel.load_checkpoint(
        str(model_path),
        device=device,
    )
    model.eval()

    print("=" * 96)
    print(" LLM_SEM v0.17.7 Canonical Unknown Gate Probe")
    print("=" * 96)
    print("Model          :", model_path)
    print("Known anchors  :", len(known_anchors))
    print("Known variants :", len(KNOWN_VARIANTS))
    print("Unknown probes :", len(UNKNOWN_PROBES))
    print("Device         :", device)
    if device.type == "cuda":
        print("GPU            :", torch.cuda.get_device_name(device))
    print()

    results = []
    for pooling in POOLINGS:
        result = evaluate(
            model,
            tokenizer,
            known_anchors,
            pooling,
        )
        results.append(result)

        print(f"POOLING={pooling}")
        print("-" * 96)
        for row in result["rows"]:
            print(
                f"[{row['label']:<7s}] {row['text']:<22s} "
                f"score={row['score']:.6f} "
                f"nearest={row['nearest']!r}"
            )
        print(f"min_known           : {result['min_known']:.6f}")
        print(f"max_unknown         : {result['max_unknown']:.6f}")
        print(f"separation_gap      : {result['gap']:+.6f}")
        print(f"suggested_threshold : {result['threshold']:.6f}")
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

    print("=" * 96)
    print(" CANONICAL UNKNOWN GATE PROBE RESULT")
    print("=" * 96)
    print("Best pooling         :", best["pooling"])
    print("Min known score      :", f"{best['min_known']:.6f}")
    print("Max unknown score    :", f"{best['max_unknown']:.6f}")
    print("Separation gap       :", f"{best['gap']:+.6f}")
    print("Suggested threshold  :", f"{best['threshold']:.6f}")
    print("Accuracy             :", f"{best['accuracy']:.1%}")
    print(
        "STATUS               :",
        "UNKNOWN_GATE_SEPARABLE"
        if best["separable"]
        else "UNKNOWN_GATE_OVERLAP",
    )


if __name__ == "__main__":
    main()
