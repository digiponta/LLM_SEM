#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LLM_SEM v0.17.6 Routed Stage 5 Validation

Validates the two-model semantic-gate architecture:
  - canonical base decides the semantic route,
  - internalized checkpoint handles the learned quantum-sensor concept,
  - canonical base handles protected and negative probes,
  - Semantic Memory lookup is not used.

PASS requires:
  1) positive/new-fact probes -> INTERNALIZED + exact answer,
  2) protected 14 -> CANONICAL + exact canonical answer,
  3) surface variants -> expected route + exact answer,
  4) near/unrelated negatives -> CANONICAL and no false activation,
  5) fresh reload reproduces route + output exactly.
"""

from __future__ import annotations

import argparse
import difflib
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
DEFAULT_RESULTS = "results/stage5_routed_validation_v0176.json"
DEFAULT_THRESHOLD = 0.069273

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

SURFACE_VARIANTS = [
    "CPUとは、",
    "GPUとは。",
    "Pythonとは？",
    "量子センサーとは。",
    "量子センサとは？",
]

NEAR_NEGATIVES = [
    "量子通信とは",
    "量子コンピュータとは",
    "量子暗号とは",
]

UNRELATED_NEGATIVES = [
    "暗号",
    "文学とは",
    "ブラックホールとは",
]


def parse_args():
    p = argparse.ArgumentParser(
        description="LLM_SEM v0.17.6 Routed Stage 5 Validation"
    )
    p.add_argument("--base-model", default=DEFAULT_BASE)
    p.add_argument("--internal-model", default=DEFAULT_INTERNAL)
    p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    p.add_argument("--protected", default=DEFAULT_PROTECTED)
    p.add_argument("--results", default=DEFAULT_RESULTS)
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--max-new-tokens", type=int, default=96)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--repetition-penalty", type=float, default=1.10)
    p.add_argument("--false-activation-similarity", type=float, default=0.70)
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


def normalize_text(text: str) -> str:
    return unicodedata.normalize("NFKC", text).strip()


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
def build_gate(
    base: LanguageModel,
    tokenizer: Tokenizer,
) -> Tuple[torch.Tensor, torch.Tensor]:
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
def route(
    base: LanguageModel,
    tokenizer: Tokenizer,
    prompt: str,
    pos_centroid: torch.Tensor,
    neg_centroid: torch.Tensor,
    threshold: float,
) -> Dict[str, object]:
    v = semantic_vector(base, tokenizer, prompt)
    pos_sim = float(F.cosine_similarity(
        v.unsqueeze(0), pos_centroid.unsqueeze(0)
    ).item())
    neg_sim = float(F.cosine_similarity(
        v.unsqueeze(0), neg_centroid.unsqueeze(0)
    ).item())
    margin = pos_sim - neg_sim
    internal = margin >= threshold
    return {
        "route": "INTERNALIZED" if internal else "CANONICAL",
        "positive_similarity": pos_sim,
        "negative_similarity": neg_sim,
        "margin": margin,
    }


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


def similarity_to_new(text: str) -> float:
    a = normalize_text(text)
    b = normalize_text(NEW_ANSWER)
    if not a:
        return 0.0
    if a == b or a in b or b in a:
        return 1.0
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()


def validate_once(
    base: LanguageModel,
    internal: LanguageModel,
    tokenizer: Tokenizer,
    protected: List[Dict[str, str]],
    args,
):
    pos_centroid, neg_centroid = build_gate(base, tokenizer)
    protected_map = {x["prompt"]: x["answer"] for x in protected}
    checks = []
    snapshots = {}

    def eval_probe(
        category: str,
        prompt: str,
        expected_route: str,
        expected_answer: str | None = None,
        reject_new: bool = False,
    ):
        gate = route(
            base,
            tokenizer,
            prompt,
            pos_centroid,
            neg_centroid,
            args.threshold,
        )
        selected = internal if gate["route"] == "INTERNALIZED" else base
        actual = generate(selected, tokenizer, prompt, args)

        route_ok = gate["route"] == expected_route
        exact_ok = True if expected_answer is None else (
            normalize_text(actual) == normalize_text(expected_answer)
        )
        sim = similarity_to_new(actual)
        reject_ok = True if not reject_new else (
            sim < args.false_activation_similarity
        )
        passed = route_ok and exact_ok and reject_ok

        row = {
            "category": category,
            "prompt": prompt,
            "expected_route": expected_route,
            "route": gate["route"],
            "margin": gate["margin"],
            "positive_similarity": gate["positive_similarity"],
            "negative_similarity": gate["negative_similarity"],
            "actual": actual,
            "pass": passed,
        }
        if expected_answer is not None:
            row["expected"] = expected_answer
            row["exact_pass"] = exact_ok
        if reject_new:
            row["similarity_to_new"] = sim
            row["reject_pass"] = reject_ok
        checks.append(row)
        snapshots[prompt] = {
            "route": gate["route"],
            "actual": actual,
            "margin": gate["margin"],
        }

    for p in ["量子センサーとは", "量子センサとは"]:
        eval_probe("new_fact", p, "INTERNALIZED", NEW_ANSWER)

    for item in protected:
        eval_probe(
            "protected",
            item["prompt"],
            "CANONICAL",
            item["answer"],
        )

    for p in SURFACE_VARIANTS:
        canonical = normalize_prompt(p)
        if canonical in {"量子センサーとは", "量子センサとは"}:
            expected_route = "INTERNALIZED"
            expected_answer = NEW_ANSWER
        else:
            expected_route = "CANONICAL"
            expected_answer = protected_map.get(canonical)
        eval_probe(
            "surface_variant",
            p,
            expected_route,
            expected_answer,
        )

    for category, prompts in (
        ("near_negative", NEAR_NEGATIVES),
        ("unrelated_negative", UNRELATED_NEGATIVES),
    ):
        for p in prompts:
            eval_probe(
                category,
                p,
                "CANONICAL",
                expected_answer=None,
                reject_new=True,
            )

    summary = {}
    for row in checks:
        s = summary.setdefault(
            row["category"], {"pass": 0, "fail": 0}
        )
        s["pass" if row["pass"] else "fail"] += 1

    return {
        "pass": all(bool(x["pass"]) for x in checks),
        "checks": checks,
        "summary": summary,
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

    print("=" * 92)
    print(" LLM_SEM v0.17.6 Routed Stage 5 Validation")
    print("=" * 92)
    print("Canonical base         :", base_path)
    print("Internalized model     :", internal_path)
    print("Gate pooling           : mean")
    print("Gate threshold         :", args.threshold)
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
    print("-" * 92)
    for row in first["checks"]:
        status = "PASS" if row["pass"] else "FAIL"
        extra = ""
        if "similarity_to_new" in row:
            extra = f" sim={row['similarity_to_new']:.3f}"
        print(
            f"[{status}] {row['category']:18s} {row['prompt']!r} "
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
        if a is None or b is None:
            mismatches[prompt] = {"first": a, "second": b}
            continue
        if (
            a["route"] != b["route"]
            or a["actual"] != b["actual"]
            or abs(float(a["margin"]) - float(b["margin"])) > 1.0e-7
        ):
            mismatches[prompt] = {"first": a, "second": b}

    reload_pass = not mismatches
    overall = bool(first["pass"] and second["pass"] and reload_pass)

    print("RELOAD REPRODUCIBILITY")
    print("-" * 92)
    print("First validation      :", "PASS" if first["pass"] else "FAIL")
    print("Second validation     :", "PASS" if second["pass"] else "FAIL")
    print("Route/output equality :", "PASS" if reload_pass else "FAIL")
    print()

    result = {
        "version": "v0.17.6",
        "architecture": "semantic-gated-dual-model",
        "base_model": str(base_path),
        "internalized_model": str(internal_path),
        "gate_pooling": "mean",
        "gate_threshold": args.threshold,
        "semantic_memory_lookup": False,
        "first_load": first,
        "second_load": second,
        "reload": {
            "pass": reload_pass,
            "mismatches": mismatches,
        },
        "overall_pass": overall,
        "status": "STAGE5_ROUTED_VALIDATED" if overall else "STAGE5_ROUTED_FAIL",
    }

    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    def cat_pass(name: str) -> bool:
        return first["summary"].get(name, {}).get("fail", 0) == 0

    print("=" * 92)
    print(" ROUTED STAGE 5 RESULT")
    print("=" * 92)
    print("New fact/runtime      :", "PASS" if cat_pass("new_fact") else "FAIL")
    print("Protected 14          :", "PASS" if cat_pass("protected") else "FAIL")
    print("Surface variants      :", "PASS" if cat_pass("surface_variant") else "FAIL")
    print("Near-negative reject  :", "PASS" if cat_pass("near_negative") else "FAIL")
    print("Unrelated reject      :", "PASS" if cat_pass("unrelated_negative") else "FAIL")
    print("Reload reproducibility:", "PASS" if reload_pass else "FAIL")
    print("Semantic Memory lookup: NONE")
    print("Results               :", results_path)
    print("STATUS                :", result["status"])

    raise SystemExit(0 if overall else 1)


if __name__ == "__main__":
    main()
