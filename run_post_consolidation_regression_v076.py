# run_post_consolidation_regression_v076.py
#
# LLM_SEM v0.7.6
# Calibrated post-consolidation evaluation using relative label competition.
#
# Why this exists:
# Absolute target-label NLL can improve on unrelated inputs simply because the
# model learns the classification prompt format better. Therefore PASS/FAIL is
# based on ranking among all semantic labels, not absolute NLL alone.

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from adaptive_semantic_learning import exact_memory_label
from model import LanguageModel
from tokenizer import Tokenizer
from semantic_memory_validate_v073 import conditional_target_nll


DEFAULT_MEMORY = "data/semantic_memory.jsonl"
DEFAULT_SOURCE = "model/model-gpu-v0.4.pt"
DEFAULT_CANDIDATE = "model/model-sem-consolidation-v075.pt"
DEFAULT_TOKENIZER = "model/tokenizer.json"
DEFAULT_CASES = "data/post_consolidation_cases_v074.json"
DEFAULT_BENCHMARK = "my_benchmark.csv"


def load_labels(path: Path) -> list[str]:
    labels: list[str] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            label = str(row.get("label", "")).strip()
            if label and label not in seen:
                labels.append(label)
                seen.add(label)
    if not labels:
        raise ValueError("No semantic labels found.")
    return labels


def load_cases(path: Path) -> list[dict]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    required = {"group", "text", "target_label", "expected_label"}
    if not isinstance(rows, list) or not rows:
        raise ValueError("cases file must contain a non-empty JSON list")
    for index, row in enumerate(rows, 1):
        if not isinstance(row, dict) or not required.issubset(row):
            raise ValueError(f"invalid case at index {index}: {row!r}")
    return rows


@torch.no_grad()
def score_labels(
    model: LanguageModel,
    tokenizer: Tokenizer,
    text: str,
    labels: list[str],
    device: torch.device,
) -> dict[str, float]:
    prompt = f"入力: {text}\n意味分類: "
    return {
        label: conditional_target_nll(
            model,
            tokenizer,
            prompt,
            label,
            device,
        )
        for label in labels
    }


def ranked(scores: dict[str, float]) -> list[tuple[str, float]]:
    return sorted(scores.items(), key=lambda item: item[1])


def margin_for(
    scores: dict[str, float],
    expected_label: str,
) -> float:
    expected = scores[expected_label]
    others = [value for label, value in scores.items() if label != expected_label]
    if not others:
        return float("inf")
    best_other = min(others)
    # Positive means expected label has lower NLL than its strongest competitor.
    return best_other - expected


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LLM_SEM v0.7.6 calibrated post-consolidation regression"
    )
    p.add_argument("--memory", default=DEFAULT_MEMORY)
    p.add_argument("--source", default=DEFAULT_SOURCE)
    p.add_argument("--candidate", default=DEFAULT_CANDIDATE)
    p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    p.add_argument("--cases", default=DEFAULT_CASES)
    p.add_argument("--benchmark", default=DEFAULT_BENCHMARK)
    p.add_argument("--min-candidate-margin", type=float, default=0.02)
    p.add_argument("--min-margin-gain", type=float, default=0.00)
    p.add_argument("--allow-cpu", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    labels = load_labels(Path(args.benchmark))
    cases = load_cases(Path(args.cases))
    memory_path = Path(args.memory)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and not args.allow_cpu:
        raise RuntimeError("CUDA unavailable. Use --allow-cpu only for testing.")

    tokenizer = Tokenizer.load(args.tokenizer)
    source, _ = LanguageModel.load_checkpoint(args.source, device=device)
    candidate, checkpoint = LanguageModel.load_checkpoint(args.candidate, device=device)
    source.eval()
    candidate.eval()

    print("=" * 100)
    print(" LLM_SEM v0.7.6 Calibrated Post-Consolidation Regression")
    print("=" * 100)
    print("Device               :", device)
    if device.type == "cuda":
        print("GPU                  :", torch.cuda.get_device_name(0))
    print("Source checkpoint    :", args.source)
    print("Candidate checkpoint :", args.candidate)
    print("Candidate loss       :", checkpoint.get("loss"))
    print("Semantic labels      :", ", ".join(labels))
    print("Cases                :", len(cases))
    print()
    print("PASS is based on relative label ranking/margin, not absolute target NLL.")
    print()

    all_pass = True
    group_results: dict[str, list[bool]] = {}

    for index, row in enumerate(cases, 1):
        group = str(row["group"])
        text = str(row["text"])
        target_label = str(row["target_label"])
        expected_label = str(row["expected_label"])

        if expected_label not in labels:
            raise ValueError(f"expected label not in benchmark labels: {expected_label}")

        source_scores = score_labels(source, tokenizer, text, labels, device)
        candidate_scores = score_labels(candidate, tokenizer, text, labels, device)

        source_rank = ranked(source_scores)
        candidate_rank = ranked(candidate_scores)

        source_top = source_rank[0][0]
        candidate_top = candidate_rank[0][0]

        source_margin = margin_for(source_scores, expected_label)
        candidate_margin = margin_for(candidate_scores, expected_label)
        margin_gain = candidate_margin - source_margin

        target_nll_gain = (
            source_scores[target_label] - candidate_scores[target_label]
        )

        active_memory = exact_memory_label(memory_path, text)
        memory_independent = active_memory is None

        passed = (
            candidate_top == expected_label
            and candidate_margin >= args.min_candidate_margin
            and margin_gain >= args.min_margin_gain
            and memory_independent
        )

        all_pass = all_pass and passed
        group_results.setdefault(group, []).append(passed)

        print(
            f"{index:02d}. [{'PASS' if passed else 'FAIL'}] "
            f"{group:<10} text={text!r}"
        )
        print(
            f"    expected={expected_label:<10} "
            f"source_top={source_top:<10} candidate_top={candidate_top:<10}"
        )
        print(
            f"    source_margin={source_margin:+.6f} "
            f"candidate_margin={candidate_margin:+.6f} "
            f"margin_gain={margin_gain:+.6f}"
        )
        print(
            f"    target={target_label:<10} "
            f"target_nll_gain={target_nll_gain:+.6f} "
            f"active_memory={active_memory or '(none)'}"
        )
        print("    candidate ranking:")
        for rank_index, (label, value) in enumerate(candidate_rank, 1):
            print(f"      {rank_index}. {label:<10} nll={value:.6f}")

    print()
    print("Group summary")
    print("-------------")
    for group in ("exact", "paraphrase", "related", "unrelated"):
        rows = group_results.get(group, [])
        if not rows:
            print(f"{group:<12}: MISSING")
            all_pass = False
            continue
        ok = sum(1 for x in rows if x)
        group_pass = all(rows)
        print(
            f"{group:<12}: {ok}/{len(rows)} "
            f"{'PASS' if group_pass else 'FAIL'}"
        )

    print()
    print("RESULT:", "PASS" if all_pass else "FAIL")
    if all_pass:
        print(
            "Conclusion: the candidate checkpoint retains the consolidated "
            "knowledge, generalizes to unseen formulations, preserves correct "
            "relative semantic discrimination, and is independent of active "
            "Semantic Memory."
        )
    else:
        print(
            "Conclusion: inspect the label rankings and margins. Absolute NLL "
            "improvement alone is not treated as evidence of overgeneralization."
        )


if __name__ == "__main__":
    main()
