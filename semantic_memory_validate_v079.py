# semantic_memory_validate_v079.py
#
# LLM_SEM v0.7.9
# Semantic-aware consolidation validator.
#
# This validator promotes VALIDATING -> CONSOLIDATED only when the candidate
# checkpoint:
#   1) routes the taught exact query to the taught semantic label,
#   2) has a positive semantic margin,
#   3) preserves benchmark routing quality within tolerance, and
#   4) keeps Semantic Memory authoritative until validation succeeds.
#
# It replaces LM-head label-token NLL as the primary consolidation criterion.

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from adaptive_semantic_learning import (
    load_semantic_memory_records,
    update_memory_status,
)
from model import LanguageModel
from semantic_eval import load_benchmark
from semantic_router import SemanticRouter, collect_known_loo_scores
from tokenizer import Tokenizer


DEFAULT_MEMORY = "data/semantic_memory.jsonl"
DEFAULT_SOURCE = "model/model-gpu-v0.4.pt"
DEFAULT_CANDIDATE = "model/model-sem-consolidation-v078.pt"
DEFAULT_TOKENIZER = "model/tokenizer.json"
DEFAULT_BENCHMARK = "my_benchmark.csv"


def validating_records(path: Path) -> list[dict]:
    return [
        row for row in load_semantic_memory_records(path)
        if row.get("status") == "VALIDATING"
    ]


def top_margin(ranked) -> float:
    if len(ranked) < 2:
        return 1.0
    return float(ranked[0].similarity - ranked[1].similarity)


def loo_accuracy(router: SemanticRouter, samples) -> float:
    scores = collect_known_loo_scores(router, samples)
    if not scores:
        return 0.0
    return sum(int(row.correct) for row in scores) / len(scores)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LLM_SEM v0.7.9 semantic-aware consolidation validator"
    )
    p.add_argument("--memory", default=DEFAULT_MEMORY)
    p.add_argument("--source", default=DEFAULT_SOURCE)
    p.add_argument("--candidate", default=DEFAULT_CANDIDATE)
    p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    p.add_argument("--benchmark", default=DEFAULT_BENCHMARK)
    p.add_argument("--alpha", type=float, default=0.35)
    p.add_argument("--min-margin", type=float, default=0.02)
    p.add_argument(
        "--max-loo-regression",
        type=float,
        default=0.10,
        help="Maximum allowed absolute LOO accuracy drop.",
    )
    p.add_argument("--allow-cpu", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    memory_path = Path(args.memory)
    records = validating_records(memory_path)
    if not records:
        raise RuntimeError("No VALIDATING Semantic Memory records found.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and not args.allow_cpu:
        raise RuntimeError("CUDA unavailable. Use --allow-cpu only for testing.")

    tokenizer = Tokenizer.load(args.tokenizer)
    benchmark = load_benchmark(args.benchmark)

    source_model, source_ckpt = LanguageModel.load_checkpoint(
        args.source, device=device
    )
    candidate_model, candidate_ckpt = LanguageModel.load_checkpoint(
        args.candidate, device=device
    )
    source_model.eval()
    candidate_model.eval()

    source_router = SemanticRouter(source_model, tokenizer, alpha=args.alpha)
    candidate_router = SemanticRouter(candidate_model, tokenizer, alpha=args.alpha)
    source_router.fit(benchmark)
    candidate_router.fit(benchmark)

    print("=" * 96)
    print(" LLM_SEM v0.7.9 Semantic-Aware Consolidation Validation")
    print("=" * 96)
    print("Device               :", device)
    if device.type == "cuda":
        print("GPU                  :", torch.cuda.get_device_name(0))
    print("Source checkpoint    :", args.source)
    print("Source loss          :", source_ckpt.get("loss"))
    print("Candidate checkpoint :", args.candidate)
    print("Candidate loss       :", candidate_ckpt.get("loss"))
    print("VALIDATING records   :", len(records))
    print("Benchmark samples    :", len(benchmark))
    print("Alpha                :", args.alpha)
    print("Minimum margin       :", args.min_margin)
    print("Max LOO regression   :", args.max_loo_regression)
    print()

    record_passes: list[bool] = []

    for row in records:
        text = str(row["text"])
        expected = str(row["label"])

        source_rank = source_router.route(text)
        candidate_rank = candidate_router.route(text)

        source_top = source_rank[0]
        candidate_top = candidate_rank[0]
        source_margin = top_margin(source_rank)
        candidate_margin = top_margin(candidate_rank)

        source_expected = next(
            item for item in source_rank if item.label == expected
        )
        candidate_expected = next(
            item for item in candidate_rank if item.label == expected
        )
        expected_gain = (
            candidate_expected.similarity - source_expected.similarity
        )

        passed = (
            candidate_top.label == expected
            and candidate_margin >= args.min_margin
        )
        record_passes.append(passed)

        print(
            f"[{'PASS' if passed else 'FAIL'}] {text!r} "
            f"expected={expected}"
        )
        print(
            f"  source_top={source_top.label} "
            f"({source_top.similarity:.6f}) "
            f"margin={source_margin:+.6f}"
        )
        print(
            f"  candidate_top={candidate_top.label} "
            f"({candidate_top.similarity:.6f}) "
            f"margin={candidate_margin:+.6f}"
        )
        print(
            f"  expected_similarity_gain={expected_gain:+.6f}"
        )

    print()
    print("Benchmark preservation")
    print("----------------------")

    source_loo = loo_accuracy(
        SemanticRouter(source_model, tokenizer, alpha=args.alpha),
        benchmark,
    )
    candidate_loo = loo_accuracy(
        SemanticRouter(candidate_model, tokenizer, alpha=args.alpha),
        benchmark,
    )
    loo_regression = source_loo - candidate_loo
    benchmark_ok = loo_regression <= args.max_loo_regression

    print("Source LOO accuracy    :", f"{source_loo * 100:.2f}%")
    print("Candidate LOO accuracy :", f"{candidate_loo * 100:.2f}%")
    print("LOO regression         :", f"{loo_regression * 100:+.2f} pp")
    print(
        "Benchmark check        :",
        "PASS" if benchmark_ok else "FAIL",
    )

    records_ok = all(record_passes)
    passed = records_ok and benchmark_ok

    next_state = "CONSOLIDATED" if passed else "FAILED"
    model_version = Path(args.candidate).name

    for row in records:
        update_memory_status(
            memory_path,
            str(row["text"]),
            next_state,
            model_version=model_version,
            verified=passed,
        )

    print()
    print("Record semantic check :", "PASS" if records_ok else "FAIL")
    print("RESULT                :", "PASS" if passed else "FAIL")
    print("Memory transition     : VALIDATING ->", next_state)

    if passed:
        print(
            "Semantic Memory can now leave the primary routing path; "
            "the candidate checkpoint becomes primary."
        )
    else:
        print(
            "Semantic Memory remains authoritative because semantic "
            "consolidation validation did not fully pass."
        )


if __name__ == "__main__":
    main()
