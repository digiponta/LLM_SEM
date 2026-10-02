# run_post_consolidation_regression_v074.py
#
# LLM_SEM v0.7.4
# Verify that consolidated knowledge remains available from the candidate
# checkpoint without relying on active Semantic Memory.
#
# Four groups are evaluated:
#   exact / paraphrase / related / unrelated
#
# For exact/paraphrase/related, the candidate model should lower the NLL of the
# consolidated target label compared with the pre-consolidation source model.
# For unrelated inputs, the same target label should not become substantially
# easier, guarding against semantic overgeneralization.

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch

from adaptive_semantic_learning import (
    exact_memory_label,
    load_semantic_memory_records,
)
from model import LanguageModel
from tokenizer import Tokenizer
from semantic_memory_validate_v073 import conditional_target_nll


DEFAULT_MEMORY = "data/semantic_memory.jsonl"
DEFAULT_SOURCE = "model/model-gpu-v0.4.pt"
DEFAULT_CANDIDATE = "model/model-sem-consolidation-v073.pt"
DEFAULT_TOKENIZER = "model/tokenizer.json"
DEFAULT_CASES = "data/post_consolidation_cases_v074.json"


def load_cases(path: Path) -> list[dict]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("cases file must contain a non-empty JSON list")
    required = {"group", "text", "target_label", "expect"}
    for index, row in enumerate(rows, 1):
        if not isinstance(row, dict) or not required.issubset(row):
            raise ValueError(f"invalid case at index {index}: {row!r}")
    return rows


def consolidated_records(path: Path) -> list[dict]:
    return [
        row for row in load_semantic_memory_records(path)
        if row.get("status") == "CONSOLIDATED"
    ]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LLM_SEM v0.7.4 post-consolidation regression"
    )
    p.add_argument("--memory", default=DEFAULT_MEMORY)
    p.add_argument("--source", default=DEFAULT_SOURCE)
    p.add_argument("--candidate", default=DEFAULT_CANDIDATE)
    p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    p.add_argument("--cases", default=DEFAULT_CASES)
    p.add_argument(
        "--min-improvement",
        type=float,
        default=0.02,
        help="Minimum source_nll-candidate_nll for improve cases.",
    )
    p.add_argument(
        "--max-unrelated-improvement",
        type=float,
        default=0.20,
        help=(
            "Maximum allowed improvement of the consolidated target label "
            "on unrelated input."
        ),
    )
    p.add_argument("--allow-cpu", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    memory_path = Path(args.memory)
    case_path = Path(args.cases)
    records = consolidated_records(memory_path)
    if not records:
        raise RuntimeError("No CONSOLIDATED Semantic Memory records found.")

    cases = load_cases(case_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and not args.allow_cpu:
        raise RuntimeError("CUDA unavailable. Use --allow-cpu only for testing.")

    tokenizer = Tokenizer.load(args.tokenizer)
    source, _ = LanguageModel.load_checkpoint(args.source, device=device)
    candidate, checkpoint = LanguageModel.load_checkpoint(args.candidate, device=device)
    source.eval()
    candidate.eval()

    consolidated_pairs = {
        (str(row["text"]), str(row["label"]))
        for row in records
    }

    print("=" * 96)
    print(" LLM_SEM v0.7.4 Post-Consolidation Internal-Memory Regression")
    print("=" * 96)
    print("Device               :", device)
    if device.type == "cuda":
        print("GPU                  :", torch.cuda.get_device_name(0))
    print("Source checkpoint    :", args.source)
    print("Candidate checkpoint :", args.candidate)
    print("Candidate loss       :", checkpoint.get("loss"))
    print("Consolidated records :", len(records))
    print("Cases                :", len(cases))
    print()
    print("Important: CONSOLIDATED entries are excluded from active Semantic Memory.")
    print("This test scores the model checkpoints directly.")
    print()

    results: list[dict] = []
    grouped = defaultdict(lambda: {"pass": 0, "total": 0})

    for index, row in enumerate(cases, 1):
        group = str(row["group"])
        text = str(row["text"])
        target = str(row["target_label"])
        expect = str(row["expect"])

        prompt = f"入力: {text}\n意味分類: "
        src = conditional_target_nll(source, tokenizer, prompt, target, device)
        cand = conditional_target_nll(candidate, tokenizer, prompt, target, device)
        improvement = src - cand

        if expect == "improve":
            passed = improvement >= args.min_improvement
            condition = f"gain>={args.min_improvement:+.3f}"
        elif expect == "not_overgeneralize":
            passed = improvement <= args.max_unrelated_improvement
            condition = f"gain<={args.max_unrelated_improvement:+.3f}"
        else:
            raise ValueError(f"unsupported expect mode: {expect}")

        active_memory_label = exact_memory_label(memory_path, text)
        if (text, target) in consolidated_pairs and active_memory_label is not None:
            # This must never happen: consolidated exact entries should not
            # participate in the active-memory path.
            passed = False

        grouped[group]["total"] += 1
        grouped[group]["pass"] += int(passed)

        print(
            f"{index:02d}. [{'PASS' if passed else 'FAIL'}] "
            f"{group:<10} text={text!r}"
        )
        print(
            f"    target={target:<10} source_nll={src:.6f} "
            f"candidate_nll={cand:.6f} gain={improvement:+.6f}"
        )
        print(
            f"    condition={condition} "
            f"active_memory={active_memory_label or '(none)'}"
        )

        results.append(
            {
                "group": group,
                "text": text,
                "target": target,
                "source_nll": src,
                "candidate_nll": cand,
                "improvement": improvement,
                "passed": passed,
            }
        )

    print()
    print("Group summary")
    print("-------------")
    required_groups = ("exact", "paraphrase", "related", "unrelated")
    all_groups_present = True
    all_pass = True
    for group in required_groups:
        data = grouped[group]
        if data["total"] == 0:
            print(f"{group:<12}: MISSING")
            all_groups_present = False
            all_pass = False
            continue
        group_pass = data["pass"] == data["total"]
        all_pass = all_pass and group_pass
        print(
            f"{group:<12}: {data['pass']}/{data['total']} "
            f"{'PASS' if group_pass else 'FAIL'}"
        )

    exact_consolidated_hidden = True
    for row in records:
        label = exact_memory_label(memory_path, str(row["text"]))
        if label is not None:
            exact_consolidated_hidden = False
            print(
                "ERROR: consolidated entry still visible to active memory:",
                row["text"],
            )

    print()
    print(
        "Consolidated hidden from active memory:",
        "PASS" if exact_consolidated_hidden else "FAIL",
    )
    final_pass = all_groups_present and all_pass and exact_consolidated_hidden
    print("RESULT:", "PASS" if final_pass else "FAIL")

    if final_pass:
        print(
            "Conclusion: the consolidated knowledge is retained by the "
            "candidate checkpoint without active Semantic Memory, and no "
            "tested unrelated overgeneralization was detected."
        )
    else:
        print(
            "Conclusion: post-consolidation behavior needs review. "
            "Semantic Memory records remain retained for audit/recovery."
        )


if __name__ == "__main__":
    main()
