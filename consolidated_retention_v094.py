# consolidated_retention_v094.py
#
# LLM_SEM v0.9.4
# Consolidated Retention / Interference Regression
#
# Pure validation: does not mutate Semantic Memory lifecycle.
# Checks every CONSOLIDATED record against the candidate checkpoint.

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import torch

from adaptive_semantic_learning import load_semantic_memory_records
from model import LanguageModel
from semantic_eval import load_benchmark
from semantic_router import SemanticRouter
from tokenizer import Tokenizer


DEFAULT_SOURCE = "model/model-sem-consolidation-v080.pt"
DEFAULT_CANDIDATE = "model/model-sem-consolidation-v081.pt"
DEFAULT_TOKENIZER = "model/tokenizer.json"
DEFAULT_BENCHMARK = "my_benchmark.csv"
DEFAULT_MEMORY = "data/semantic_memory.jsonl"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LLM_SEM v0.9.4 Consolidated Retention Regression"
    )
    p.add_argument("--source", default=DEFAULT_SOURCE)
    p.add_argument("--candidate", default=DEFAULT_CANDIDATE)
    p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    p.add_argument("--benchmark", default=DEFAULT_BENCHMARK)
    p.add_argument("--memory", default=DEFAULT_MEMORY)
    p.add_argument("--alpha", type=float, default=0.35)
    p.add_argument("--min-margin", type=float, default=0.02)
    p.add_argument("--recovery-min-margin", type=float, default=0.005)
    p.add_argument("--degraded-min-margin", type=float, default=0.01)
    p.add_argument("--recovery-min-gain", type=float, default=0.0)
    p.add_argument("--baseline-max-sim-drop", type=float, default=0.01)
    p.add_argument("--max-loo-drop", type=float, default=0.10)
    p.add_argument("--allow-cpu", action="store_true")
    return p.parse_args()


def loo_accuracy(router: SemanticRouter, samples) -> float:
    total = 0
    correct = 0
    for i, row in enumerate(samples):
        train = [x for j, x in enumerate(samples) if j != i]
        r = SemanticRouter(router.model, router.tokenizer, alpha=router.alpha)
        r.fit(train)
        ranked = r.route(row.text)
        total += 1
        correct += int(bool(ranked) and ranked[0].label == row.label)
    return correct / total if total else 0.0


def margin_for(ranked) -> float:
    if not ranked:
        return -1.0
    second = ranked[1].similarity if len(ranked) > 1 else -1.0
    return float(ranked[0].similarity - second)


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and not args.allow_cpu:
        raise RuntimeError("CUDA unavailable. Use --allow-cpu only for testing.")

    tokenizer = Tokenizer.load(args.tokenizer)
    source_model, source_ckpt = LanguageModel.load_checkpoint(args.source, device=device)
    cand_model, cand_ckpt = LanguageModel.load_checkpoint(args.candidate, device=device)

    benchmark = load_benchmark(args.benchmark)
    records = [
        x for x in load_semantic_memory_records(Path(args.memory))
        if x.get("status") == "CONSOLIDATED"
    ]

    source_router = SemanticRouter(source_model, tokenizer, alpha=args.alpha)
    cand_router = SemanticRouter(cand_model, tokenizer, alpha=args.alpha)
    source_router.fit(benchmark)
    cand_router.fit(benchmark)

    source_loo = loo_accuracy(source_router, benchmark)
    cand_loo = loo_accuracy(cand_router, benchmark)
    loo_drop = source_loo - cand_loo
    global_pass = loo_drop <= args.max_loo_drop

    print("=" * 100)
    print(" LLM_SEM v0.10.10 Retention / Degraded / Recovery Regression")
    print("=" * 100)
    print("Device              :", device)
    if device.type == "cuda":
        print("GPU                 :", torch.cuda.get_device_name(0))
    print("Source checkpoint   :", args.source)
    print("Source loss         :", source_ckpt.get("loss"))
    print("Candidate checkpoint:", args.candidate)
    print("Candidate loss      :", cand_ckpt.get("loss"))
    print("Consolidated records:", len(records))
    print("Retention min margin:", args.min_margin)
    print("Recovery min margin :", args.recovery_min_margin)
    print("Degraded min margin :", args.degraded_min_margin)
    print("Recovery min gain   :", args.recovery_min_gain)
    print("Baseline max sim drop:", args.baseline_max_sim_drop)
    print()
    print("Global benchmark preservation")
    print("-----------------------------")
    print("Source LOO accuracy   :", f"{source_loo*100:.2f}%")
    print("Candidate LOO accuracy:", f"{cand_loo*100:.2f}%")
    print("LOO drop              :", f"{loo_drop*100:+.2f} pp")
    print("Global check          :", "PASS" if global_pass else "FAIL")
    print()

    passed = 0
    failed = 0
    by_label = Counter()
    by_label_pass = Counter()

    for idx, record in enumerate(records, 1):
        text = str(record["text"])
        expected = str(record["label"])
        source_ranked = source_router.route(text)
        cand_ranked = cand_router.route(text)
        source_top = source_ranked[0].label if source_ranked else "(none)"
        cand_top = cand_ranked[0].label if cand_ranked else "(none)"
        source_margin = margin_for(source_ranked)
        cand_margin = margin_for(cand_ranked)
        expected_source = next(
            (float(x.similarity) for x in source_ranked if x.label == expected),
            -1.0,
        )
        expected_candidate = next(
            (float(x.similarity) for x in cand_ranked if x.label == expected),
            -1.0,
        )
        expected_gain = expected_candidate - expected_source

        strict_retention = (
            cand_top == expected
            and cand_margin >= args.min_margin
        )
        degraded_retention = (
            source_top == expected
            and cand_top == expected
            and cand_margin >= args.degraded_min_margin
        )
        recovered = (
            source_top != expected
            and cand_top == expected
            and cand_margin >= args.recovery_min_margin
            and expected_gain > args.recovery_min_gain
        )
        baseline_ambiguous = (
            source_top == expected
            and source_margin < args.degraded_min_margin
            and cand_top == expected
            and cand_margin >= 0.0
            and expected_gain >= -args.baseline_max_sim_drop
        )
        ok = global_pass and (
            strict_retention
            or degraded_retention
            or recovered
            or baseline_ambiguous
        )
        mode = (
            "RETENTION"
            if strict_retention
            else (
                "DEGRADED"
                if degraded_retention
                else ("RECOVERY" if recovered else "FAIL")
            )
        )

        passed += int(ok)
        failed += int(not ok)
        by_label[expected] += 1
        by_label_pass[expected] += int(ok)

        print(
            f"{idx:02d}. [{'PASS' if ok else 'FAIL'}] "
            f"mode={mode:<9} expected={expected:<10} text={text!r}"
        )
        print(
            f"    source_top={source_top:<10} candidate_top={cand_top:<10}"
        )
        print(
            f"    source_margin={source_margin:+.6f} "
            f"candidate_margin={cand_margin:+.6f}"
        )
        print(
            f"    expected_sim_delta={expected_gain:+.6f}"
        )

    print()
    print("Per-label retention")
    print("-------------------")
    for label in sorted(by_label):
        print(f"{label:<12}: {by_label_pass[label]}/{by_label[label]} PASS")

    print()
    print("Retention summary")
    print("-----------------")
    print("Retained :", passed)
    print("Failed   :", failed)
    print("Total    :", len(records))
    rate = passed / len(records) if records else 1.0
    print("Rate     :", f"{rate*100:.2f}%")
    result = global_pass and failed == 0 and len(records) > 0
    print("RESULT   :", "PASS" if result else "FAIL")

    if not records:
        print("No CONSOLIDATED records found; promotion must not proceed.")
    elif result:
        print("All consolidated semantic memories were retained by the candidate.")
    else:
        print("Candidate must not be promoted while retention failures remain.")

    raise SystemExit(0 if result else 1)


if __name__ == "__main__":
    main()
