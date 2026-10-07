#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LLM_SEM v0.17.7 Three-Way Runtime Validation

Validates:
  1) INTERNALIZED route for the learned quantum-sensor concept
  2) CANONICAL_KNOWN route for protected canonical knowledge
  3) UNKNOWN rejection for unsupported concepts
  4) surface variants
  5) reload reproducibility
  6) no Semantic Memory lookup

Expected terminal status:
  STAGE6_UNKNOWN_GATED_VALIDATED
"""

from __future__ import annotations

import argparse
import json
import unicodedata
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from model import LanguageModel
from tokenizer import Tokenizer


DEFAULT_BASE = "model/model-sem-canonical-base-v0172.pt"
DEFAULT_INTERNAL = "model/model-sem-sleep-v0172.pt"
DEFAULT_TOKENIZER = "model/tokenizer.json"
DEFAULT_PROTECTED = "data/protected_knowledge_v0167.jsonl"
DEFAULT_RESULTS = "results/stage6_unknown_gated_validation_v0177.json"
DEFAULT_GATE_THRESHOLD = 0.069273
DEFAULT_UNKNOWN_THRESHOLD = 0.943319
UNKNOWN_RESPONSE = "その質問については、現在の知識では確実に答えられません。"

TRAILING = "、，,。．.!！?？:：;；"

POSITIVE_SEEDS = [
    "量子センサーとは",
    "量子センサとは",
    "量子センサーとは。",
    "量子センサとは？",
]

NEGATIVE_SEEDS = [
    "量子通信とは",
    "量子コンピュータとは",
    "量子暗号とは",
    "暗号",
    "文学とは",
    "ブラックホールとは",
]

NEW_ANSWER = "原子や電子などの量子的性質を利用する高感度な計測技術である。"

KNOWN_VARIANTS = [
    "CPUとは、",
    "GPUとは。",
    "Pythonとは？",
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


def parse_args():
    p = argparse.ArgumentParser(
        description="LLM_SEM v0.17.7 Three-Way Runtime Validation"
    )
    p.add_argument("--base-model", default=DEFAULT_BASE)
    p.add_argument("--internal-model", default=DEFAULT_INTERNAL)
    p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    p.add_argument("--protected", default=DEFAULT_PROTECTED)
    p.add_argument("--results", default=DEFAULT_RESULTS)
    p.add_argument("--gate-threshold", type=float, default=DEFAULT_GATE_THRESHOLD)
    p.add_argument(
        "--unknown-threshold",
        type=float,
        default=DEFAULT_UNKNOWN_THRESHOLD,
    )
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--max-new-tokens", type=int, default=96)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--repetition-penalty", type=float, default=1.10)
    return p.parse_args()


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(name)


def normalize_prompt(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).strip()
    return text.rstrip(TRAILING).strip()


def load_protected(path: Path) -> List[Dict[str, str]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, 1):
            raw = raw.strip()
            if not raw:
                continue
            item = json.loads(raw)
            prompt = str(item.get("prompt", "")).strip()
            answer = str(item.get("answer", "")).strip()
            if not prompt or not answer:
                raise ValueError(f"invalid protected row at line {line_no}")
            rows.append({"prompt": prompt, "answer": answer})
    return rows


@torch.no_grad()
def semantic_vector(
    model: LanguageModel,
    tokenizer: Tokenizer,
    text: str,
) -> torch.Tensor:
    ids = tokenizer.encode(
        normalize_prompt(text),
        add_bos=True,
        add_eos=False,
    )
    ids = ids[-model.context_length:]
    x = torch.tensor(
        [ids],
        dtype=torch.long,
        device=next(model.parameters()).device,
    )
    v = model.encode_semantic(x, pooling="mean")[0]
    return F.normalize(v, p=2, dim=-1)


@torch.no_grad()
def build_internalized_gate(
    base: LanguageModel,
    tokenizer: Tokenizer,
):
    pos = F.normalize(
        torch.stack([
            semantic_vector(base, tokenizer, p)
            for p in POSITIVE_SEEDS
        ]).mean(dim=0),
        p=2,
        dim=-1,
    )
    neg = F.normalize(
        torch.stack([
            semantic_vector(base, tokenizer, p)
            for p in NEGATIVE_SEEDS
        ]).mean(dim=0),
        p=2,
        dim=-1,
    )
    return pos, neg


@torch.no_grad()
def build_known_anchors(
    base: LanguageModel,
    tokenizer: Tokenizer,
    protected: List[Dict[str, str]],
):
    prompts = [row["prompt"] for row in protected]
    vectors = [semantic_vector(base, tokenizer, p) for p in prompts]
    return prompts, vectors


@torch.no_grad()
def generate(
    model: LanguageModel,
    tokenizer: Tokenizer,
    prompt: str,
    args,
) -> str:
    prompt = normalize_prompt(prompt)
    prefix = tokenizer.encode(prompt, add_bos=True, add_eos=False)
    generated = model.generate(
        prefix,
        max_new_tokens=args.max_new_tokens,
        eos_id=tokenizer.eos_id,
        temperature=args.temperature,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
    )
    return tokenizer.decode(
        generated[len(prefix):],
        skip_special_tokens=True,
    ).strip()


@torch.no_grad()
def route(
    base: LanguageModel,
    tokenizer: Tokenizer,
    prompt: str,
    pos_centroid: torch.Tensor,
    neg_centroid: torch.Tensor,
    anchor_prompts: List[str],
    anchor_vectors: List[torch.Tensor],
    args,
) -> Dict[str, object]:
    v = semantic_vector(base, tokenizer, prompt)

    pos = float(
        F.cosine_similarity(
            v.unsqueeze(0),
            pos_centroid.unsqueeze(0),
        ).item()
    )
    neg = float(
        F.cosine_similarity(
            v.unsqueeze(0),
            neg_centroid.unsqueeze(0),
        ).item()
    )
    margin = pos - neg

    if margin >= args.gate_threshold:
        return {
            "route": "INTERNALIZED",
            "margin": margin,
            "known_score": None,
            "nearest": None,
        }

    scores = [
        float(
            F.cosine_similarity(
                v.unsqueeze(0),
                anchor.unsqueeze(0),
            ).item()
        )
        for anchor in anchor_vectors
    ]
    best = max(range(len(scores)), key=scores.__getitem__)
    score = scores[best]

    if score >= args.unknown_threshold:
        route_name = "CANONICAL_KNOWN"
    else:
        route_name = "UNKNOWN"

    return {
        "route": route_name,
        "margin": margin,
        "known_score": score,
        "nearest": anchor_prompts[best],
    }


def validate_once(
    base: LanguageModel,
    internal: LanguageModel,
    tokenizer: Tokenizer,
    protected: List[Dict[str, str]],
    args,
):
    pos_centroid, neg_centroid = build_internalized_gate(base, tokenizer)
    anchor_prompts, anchor_vectors = build_known_anchors(
        base,
        tokenizer,
        protected,
    )
    protected_map = {x["prompt"]: x["answer"] for x in protected}

    checks = []
    snapshots = {}

    def evaluate(
        category: str,
        prompt: str,
        expected_route: str,
        expected_answer: str,
    ):
        info = route(
            base,
            tokenizer,
            prompt,
            pos_centroid,
            neg_centroid,
            anchor_prompts,
            anchor_vectors,
            args,
        )

        if info["route"] == "INTERNALIZED":
            actual = generate(internal, tokenizer, prompt, args)
        elif info["route"] == "CANONICAL_KNOWN":
            actual = generate(base, tokenizer, prompt, args)
        else:
            actual = UNKNOWN_RESPONSE

        passed = (
            info["route"] == expected_route
            and actual == expected_answer
        )
        row = {
            "category": category,
            "prompt": prompt,
            "expected_route": expected_route,
            "route": info["route"],
            "expected": expected_answer,
            "actual": actual,
            "margin": info["margin"],
            "known_score": info["known_score"],
            "nearest": info["nearest"],
            "pass": passed,
        }
        checks.append(row)
        snapshots[prompt] = {
            "route": info["route"],
            "actual": actual,
            "margin": info["margin"],
            "known_score": info["known_score"],
        }

    for p in ["量子センサーとは", "量子センサとは"]:
        evaluate("internalized", p, "INTERNALIZED", NEW_ANSWER)

    for row in protected:
        evaluate(
            "protected",
            row["prompt"],
            "CANONICAL_KNOWN",
            row["answer"],
        )

    for p in KNOWN_VARIANTS:
        canonical = normalize_prompt(p)
        evaluate(
            "known_variant",
            p,
            "CANONICAL_KNOWN",
            protected_map[canonical],
        )

    for p in UNKNOWN_PROBES:
        evaluate(
            "unknown",
            p,
            "UNKNOWN",
            UNKNOWN_RESPONSE,
        )

    summary = {}
    for row in checks:
        item = summary.setdefault(
            row["category"],
            {"pass": 0, "fail": 0},
        )
        item["pass" if row["pass"] else "fail"] += 1

    return {
        "checks": checks,
        "summary": summary,
        "pass": all(row["pass"] for row in checks),
    }, snapshots


def main():
    args = parse_args()
    device = choose_device(args.device)

    base_path = Path(args.base_model)
    internal_path = Path(args.internal_model)
    tokenizer_path = Path(args.tokenizer)
    protected_path = Path(args.protected)
    results_path = Path(args.results)

    tokenizer = Tokenizer.load(str(tokenizer_path))
    protected = load_protected(protected_path)

    print("=" * 96)
    print(" LLM_SEM v0.17.7 Three-Way Runtime Validation")
    print("=" * 96)
    print("Canonical base         :", base_path)
    print("Internalized model     :", internal_path)
    print("Internalized threshold :", args.gate_threshold)
    print("Unknown threshold      :", args.unknown_threshold)
    print("Protected entries      :", len(protected))
    print("Semantic Memory lookup : DISABLED / NOT USED")
    print("Device                 :", device)
    if device.type == "cuda":
        print("GPU                    :", torch.cuda.get_device_name(device))
    print()

    base1, _ = LanguageModel.load_checkpoint(str(base_path), device=device)
    int1, _ = LanguageModel.load_checkpoint(str(internal_path), device=device)
    base1.eval()
    int1.eval()

    first, snap1 = validate_once(
        base1, int1, tokenizer, protected, args
    )

    print("FIRST LOAD")
    print("-" * 96)
    for row in first["checks"]:
        status = "PASS" if row["pass"] else "FAIL"
        extra = ""
        if row["known_score"] is not None:
            extra = (
                f" known={row['known_score']:.6f} "
                f"nearest={row['nearest']!r}"
            )
        print(
            f"[{status}] {row['category']:14s} {row['prompt']!r} "
            f"route={row['route']} margin={row['margin']:+.6f}{extra}"
        )
        print("       actual:", row["actual"])
    print()

    del base1, int1
    if device.type == "cuda":
        torch.cuda.empty_cache()

    base2, _ = LanguageModel.load_checkpoint(str(base_path), device=device)
    int2, _ = LanguageModel.load_checkpoint(str(internal_path), device=device)
    base2.eval()
    int2.eval()

    second, snap2 = validate_once(
        base2, int2, tokenizer, protected, args
    )

    mismatches = {}
    for prompt in sorted(set(snap1) | set(snap2)):
        a = snap1.get(prompt)
        b = snap2.get(prompt)
        if a != b:
            mismatches[prompt] = {"first": a, "second": b}

    reload_pass = not mismatches
    overall = first["pass"] and second["pass"] and reload_pass

    print("RELOAD REPRODUCIBILITY")
    print("-" * 96)
    print("First validation      :", "PASS" if first["pass"] else "FAIL")
    print("Second validation     :", "PASS" if second["pass"] else "FAIL")
    print("Route/output equality :", "PASS" if reload_pass else "FAIL")
    print()

    result = {
        "version": "v0.17.7",
        "architecture": "three-way-semantic-routing",
        "semantic_memory_lookup": False,
        "gate_threshold": args.gate_threshold,
        "unknown_threshold": args.unknown_threshold,
        "first_load": first,
        "second_load": second,
        "reload": {
            "pass": reload_pass,
            "mismatches": mismatches,
        },
        "overall_pass": overall,
        "status": (
            "STAGE6_UNKNOWN_GATED_VALIDATED"
            if overall
            else "STAGE6_UNKNOWN_GATED_FAIL"
        ),
    }

    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    def cat(name: str) -> bool:
        return first["summary"].get(name, {}).get("fail", 0) == 0

    print("=" * 96)
    print(" STAGE 6 UNKNOWN-GATED RESULT")
    print("=" * 96)
    print("Internalized route     :", "PASS" if cat("internalized") else "FAIL")
    print("Protected 14           :", "PASS" if cat("protected") else "FAIL")
    print("Known variants         :", "PASS" if cat("known_variant") else "FAIL")
    print("Unknown rejection      :", "PASS" if cat("unknown") else "FAIL")
    print("Reload reproducibility :", "PASS" if reload_pass else "FAIL")
    print("Semantic Memory lookup : NONE")
    print("Results                :", results_path)
    print("STATUS                 :", result["status"])

    raise SystemExit(0 if overall else 1)


if __name__ == "__main__":
    main()
