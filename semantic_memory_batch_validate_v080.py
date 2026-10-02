# semantic_memory_batch_validate_v080.py
#
# LLM_SEM v0.8.0
# Batch semantic consolidation validator.
#
# Validates multiple VALIDATING Semantic Memory records independently while
# enforcing one global benchmark-preservation guard. Passing records become
# CONSOLIDATED; failing records become FAILED.

from __future__ import annotations

import argparse
from collections import defaultdict
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
DEFAULT_CANDIDATE = "model/model-sem-consolidation-v080.pt"
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
        description="LLM_SEM v0.8.0 batch semantic consolidation validator"
    )
    p.add_argument("--memory", default=DEFAULT_MEMORY)
    p.add_argument("--source", default=DEFAULT_SOURCE)
    p.add_argument("--candidate", default=DEFAULT_CANDIDATE)
    p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    p.add_argument("--benchmark", default=DEFAULT_BENCHMARK)
    p.add_argument("--alpha", type=float, default=0.35)
    p.add_argument("--min-margin", type=float, default=0.02)
    p.add_argument("--max-loo-regression", type=float, default=0.10)
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

    print("=" * 100)
    print(" LLM_SEM v0.8.0 Batch Semantic Consolidation Validation")
    print("=" * 100)
    print("Device               :", device)
    if device.type == "cuda":
        print("GPU                  :", torch.cuda.get_device_name(0))
    print("Source checkpoint    :", args.source)
    print("Source loss          :", source_ckpt.get("loss"))
    print("Candidate checkpoint :", args.candidate)
    print("Candidate loss       :", candidate_ckpt.get("loss"))
    print("VALIDATING records   :", len(records))
    print("Benchmark samples    :", len(benchmark))
    print()

    source_loo = loo_accuracy(
        SemanticRouter(source_model, tokenizer, alpha=args.alpha), benchmark
    )
    candidate_loo = loo_accuracy(
        SemanticRouter(candidate_model, tokenizer, alpha=args.alpha), benchmark
    )
    loo_regression = source_loo - candidate_loo
    benchmark_ok = loo_regression <= args.max_loo_regression

    print("Global benchmark preservation")
    print("-----------------------------")
    print("Source LOO accuracy    :", f"{source_loo * 100:.2f}%")
    print("Candidate LOO accuracy :", f"{candidate_loo * 100:.2f}%")
    print("LOO regression         :", f"{loo_regression * 100:+.2f} pp")
    print("Global benchmark check :", "PASS" if benchmark_ok else "FAIL")
    print()

    model_version = Path(args.candidate).name
    results: list[dict] = []
    by_label = defaultdict(lambda: {"pass": 0, "total": 0})

    for index, row in enumerate(records, 1):
        text = str(row["text"])
        expected = str(row["label"])

        source_rank = source_router.route(text)
        candidate_rank = candidate_router.route(text)

        source_top = source_rank[0]
        candidate_top = candidate_rank[0]
        source_margin = top_margin(source_rank)
        candidate_margin = top_margin(candidate_rank)

        source_expected = next(x for x in source_rank if x.label == expected)
        candidate_expected = next(x for x in candidate_rank if x.label == expected)
        expected_gain = (
            candidate_expected.similarity - source_expected.similarity
        )

        record_ok = (
            candidate_top.label == expected
            and candidate_margin >= args.min_margin
        )
        final_ok = record_ok and benchmark_ok
        next_state = "CONSOLIDATED" if final_ok else "FAILED"

        update_memory_status(
            memory_path,
            text,
            next_state,
            model_version=model_version,
            verified=final_ok,
        )

        by_label[expected]["total"] += 1
        by_label[expected]["pass"] += int(final_ok)
        results.append({
            "text": text,
            "expected": expected,
            "record_ok": record_ok,
            "final_ok": final_ok,
            "next_state": next_state,
        })

        print(
            f"{index:02d}. [{'PASS' if final_ok else 'FAIL'}] "
            f"expected={expected:<10} text={text!r}"
        )
        print(
            f"    source_top={source_top.label:<10} "
            f"candidate_top={candidate_top.label:<10}"
        )
        print(
            f"    source_margin={source_margin:+.6f} "
            f"candidate_margin={candidate_margin:+.6f}"
        )
        print(
            f"    expected_sim_gain={expected_gain:+.6f} "
            f"transition=VALIDATING->{next_state}"
        )

    print()
    print("Per-label summary")
    print("-----------------")
    for label in sorted(by_label):
        data = by_label[label]
        print(
            f"{label:<12}: {data['pass']}/{data['total']} "
            f"{'PASS' if data['pass'] == data['total'] else 'PARTIAL/FAIL'}"
        )

    passed = sum(int(row["final_ok"]) for row in results)
    failed = len(results) - passed
    print()
    print("Batch summary")
    print("-------------")
    print("Consolidated :", passed)
    print("Failed       :", failed)
    print("Total        :", len(results))
    print("RESULT       :", "PASS" if failed == 0 else "PARTIAL/FAIL")

    if not benchmark_ok:
        print(
            "Global benchmark preservation failed, so no VALIDATING record "
            "is allowed to become authoritative internal memory."
        )
    elif failed == 0:
        print(
            "All batch records were consolidated successfully while preserving "
            "the existing semantic benchmark."
        )
    else:
        print(
            "Passing records were consolidated individually; failed records "
            "remain available in Semantic Memory for retry."
        )


if __name__ == "__main__":
    main()
