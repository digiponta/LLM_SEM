# run_post_consolidation_semantic_router_v077.py
#
# LLM_SEM v0.7.7
# Post-consolidation regression using Semantic Vector + centroid routing.
#
# This replaces LM-head label-token likelihood as the primary semantic test.
# Each checkpoint builds its own semantic centroids from the same benchmark.
# Regression cases are then routed in semantic-vector space.

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from adaptive_semantic_learning import exact_memory_label
from model import LanguageModel
from semantic_eval import load_benchmark
from semantic_router import SemanticRouter
from tokenizer import Tokenizer


DEFAULT_MEMORY = "data/semantic_memory.jsonl"
DEFAULT_SOURCE = "model/model-gpu-v0.4.pt"
DEFAULT_CANDIDATE = "model/model-sem-consolidation-v075.pt"
DEFAULT_TOKENIZER = "model/tokenizer.json"
DEFAULT_BENCHMARK = "my_benchmark.csv"
DEFAULT_CASES = "data/post_consolidation_cases_v074.json"


def load_cases(path: Path) -> list[dict]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    required = {"group", "text", "expected_label"}
    if not isinstance(rows, list) or not rows:
        raise ValueError("cases file must contain a non-empty JSON list")
    for i, row in enumerate(rows, 1):
        if not isinstance(row, dict) or not required.issubset(row):
            raise ValueError(f"invalid case at index {i}: {row!r}")
    return rows


def margin(ranked) -> float:
    if len(ranked) < 2:
        return 1.0
    return float(ranked[0].similarity - ranked[1].similarity)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LLM_SEM v0.7.7 semantic-router post-consolidation regression"
    )
    p.add_argument("--memory", default=DEFAULT_MEMORY)
    p.add_argument("--source", default=DEFAULT_SOURCE)
    p.add_argument("--candidate", default=DEFAULT_CANDIDATE)
    p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    p.add_argument("--benchmark", default=DEFAULT_BENCHMARK)
    p.add_argument("--cases", default=DEFAULT_CASES)
    p.add_argument("--alpha", type=float, default=0.35)
    p.add_argument("--min-candidate-margin", type=float, default=0.00)
    p.add_argument(
        "--require-margin-gain",
        action="store_true",
        help="Require candidate expected-label margin to improve over source.",
    )
    p.add_argument("--allow-cpu", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and not args.allow_cpu:
        raise RuntimeError("CUDA unavailable. Use --allow-cpu only for testing.")

    tokenizer = Tokenizer.load(args.tokenizer)
    benchmark = load_benchmark(args.benchmark)
    cases = load_cases(Path(args.cases))
    memory_path = Path(args.memory)

    source_model, source_ckpt = LanguageModel.load_checkpoint(args.source, device=device)
    candidate_model, candidate_ckpt = LanguageModel.load_checkpoint(
        args.candidate, device=device
    )
    source_model.eval()
    candidate_model.eval()

    source_router = SemanticRouter(source_model, tokenizer, alpha=args.alpha)
    candidate_router = SemanticRouter(candidate_model, tokenizer, alpha=args.alpha)
    source_router.fit(benchmark)
    candidate_router.fit(benchmark)

    print("=" * 104)
    print(" LLM_SEM v0.7.7 Semantic-Router Post-Consolidation Regression")
    print("=" * 104)
    print("Device               :", device)
    if device.type == "cuda":
        print("GPU                  :", torch.cuda.get_device_name(0))
    print("Source checkpoint    :", args.source)
    print("Source loss          :", source_ckpt.get("loss"))
    print("Candidate checkpoint :", args.candidate)
    print("Candidate loss       :", candidate_ckpt.get("loss"))
    print("Benchmark samples    :", len(benchmark))
    print("Cases                :", len(cases))
    print("Pooling              : raw hybrid")
    print("Alpha                :", args.alpha)
    print()
    print("PASS is based on semantic-vector centroid routing.")
    print("LM-head label-token NLL is not used for semantic PASS/FAIL.")
    print()

    group_results: dict[str, list[bool]] = {}
    all_pass = True

    for i, row in enumerate(cases, 1):
        group = str(row["group"])
        text = str(row["text"])
        expected = str(row["expected_label"])

        source_rank = source_router.route(text)
        candidate_rank = candidate_router.route(text)

        source_top = source_rank[0]
        candidate_top = candidate_rank[0]
        source_margin = margin(source_rank)
        candidate_margin = margin(candidate_rank)

        source_expected = next(
            (x for x in source_rank if x.label == expected),
            None,
        )
        candidate_expected = next(
            (x for x in candidate_rank if x.label == expected),
            None,
        )
        if source_expected is None or candidate_expected is None:
            raise RuntimeError(f"Expected label missing from router: {expected}")

        expected_similarity_gain = (
            candidate_expected.similarity - source_expected.similarity
        )
        margin_gain = candidate_margin - source_margin

        active_memory = exact_memory_label(memory_path, text)
        memory_independent = active_memory is None

        passed = (
            candidate_top.label == expected
            and candidate_margin >= args.min_candidate_margin
            and memory_independent
        )
        if args.require_margin_gain:
            passed = passed and margin_gain >= 0.0

        all_pass = all_pass and passed
        group_results.setdefault(group, []).append(passed)

        print(
            f"{i:02d}. [{'PASS' if passed else 'FAIL'}] "
            f"{group:<10} text={text!r}"
        )
        print(
            f"    expected={expected:<10} "
            f"source_top={source_top.label:<10} "
            f"candidate_top={candidate_top.label:<10}"
        )
        print(
            f"    source_top_sim={source_top.similarity:.6f} "
            f"candidate_top_sim={candidate_top.similarity:.6f}"
        )
        print(
            f"    source_margin={source_margin:+.6f} "
            f"candidate_margin={candidate_margin:+.6f} "
            f"margin_gain={margin_gain:+.6f}"
        )
        print(
            f"    expected_sim_gain={expected_similarity_gain:+.6f} "
            f"active_memory={active_memory or '(none)'}"
        )
        print("    candidate semantic ranking:")
        for rank_index, item in enumerate(candidate_rank, 1):
            print(
                f"      {rank_index}. {item.label:<10} "
                f"sim={item.similarity:.6f} dist={item.distance:.6f}"
            )

    print()
    print("Group summary")
    print("-------------")
    for group in ("exact", "paraphrase", "related", "unrelated"):
        values = group_results.get(group, [])
        if not values:
            print(f"{group:<12}: MISSING")
            all_pass = False
            continue
        ok = sum(int(v) for v in values)
        print(
            f"{group:<12}: {ok}/{len(values)} "
            f"{'PASS' if all(values) else 'FAIL'}"
        )

    print()
    print("RESULT:", "PASS" if all_pass else "FAIL")
    if all_pass:
        print(
            "Conclusion: consolidated knowledge is usable through the "
            "checkpoint's semantic representation, generalizes to unseen "
            "forms, preserves unrelated semantic routing, and does not depend "
            "on active Semantic Memory."
        )
    else:
        print(
            "Conclusion: inspect semantic rankings and margins. If the "
            "candidate degrades relative semantic structure, the next step is "
            "to constrain incremental training rather than tune LM-head label "
            "likelihood thresholds."
        )


if __name__ == "__main__":
    main()
