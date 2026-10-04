# answer_retention_v01014.py
#
# LLM_SEM v0.10.46
# Compare mandatory internal-answer generation before/after semantic repair.

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from model import LanguageModel
from tokenizer import Tokenizer
from semantic_guided_answer_finetune_v097 import (
    generation_quality,
    generation_similarity,
    load_dataset,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LLM_SEM v0.10.46 Answer Retention")
    p.add_argument("--source", required=True)
    p.add_argument("--candidate", required=True)
    p.add_argument("--dataset", default="data/semantic_sleep_qa_v0106.json")
    p.add_argument("--tokenizer", default="model/tokenizer.json")
    p.add_argument("--max-mean-drop", type=float, default=0.03)
    p.add_argument("--max-item-drop", type=float, default=0.05)
    p.add_argument("--max-abnormal-ratio", type=float, default=0.02)
    p.add_argument("--max-repetition-ratio", type=float, default=0.20)
    p.add_argument("--allow-cpu", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and not args.allow_cpu:
        raise RuntimeError("CUDA unavailable. Use --allow-cpu only for testing.")

    tokenizer = Tokenizer.load(args.tokenizer)
    rows = [
        row for row in load_dataset(Path(args.dataset))
        if bool(row.get("must_train", False))
    ]
    if not rows:
        raise RuntimeError("No mandatory sleep QA rows found.")

    source, _ = LanguageModel.load_checkpoint(args.source, device=device)
    candidate, _ = LanguageModel.load_checkpoint(args.candidate, device=device)

    print("=" * 100)
    print(" LLM_SEM v0.10.46 Canonical-Complete Answer Retention")
    print("=" * 100)
    print("Source checkpoint   :", args.source)
    print("Candidate checkpoint:", args.candidate)
    print("Mandatory rows      :", len(rows))
    print("Max mean drop       :", args.max_mean_drop)
    print("Max item drop       :", args.max_item_drop)

    source_vals = []
    candidate_vals = []
    failures = 0

    for index, row in enumerate(rows, 1):
        source_sim, source_answer = generation_similarity(
            source, tokenizer, row, device
        )
        cand_sim, cand_answer = generation_similarity(
            candidate, tokenizer, row, device
        )
        source_quality = generation_quality(source_answer)
        cand_quality = generation_quality(cand_answer)

        source_vals.append(source_sim)
        candidate_vals.append(cand_sim)

        sim_ok = cand_sim >= source_sim - args.max_item_drop
        canonical_complete = cand_sim >= 0.999999
        termination_ok = bool(cand_quality["terminated"]) or canonical_complete
        abnormal_ok = (
            float(cand_quality["abnormal_ratio"])
            <= args.max_abnormal_ratio
        )
        repetition_ok = (
            float(cand_quality["repetition_ratio"])
            <= args.max_repetition_ratio
        )
        ok = sim_ok and termination_ok and abnormal_ok and repetition_ok
        failures += int(not ok)

        print()
        print(
            f"{index:02d}. [{'PASS' if ok else 'FAIL'}] "
            f"query={row['query']!r}"
        )
        print(
            f"    similarity: source={source_sim:.6f} "
            f"candidate={cand_sim:.6f} "
            f"delta={cand_sim-source_sim:+.6f}"
        )
        print(
            "    quality   : "
            f"terminated={cand_quality['terminated']} "
            f"canonical_complete={canonical_complete} "
            f"abnormal={float(cand_quality['abnormal_ratio']):.6f} "
            f"repetition={float(cand_quality['repetition_ratio']):.6f}"
        )
        print("    source    :", source_answer)
        print("    candidate :", cand_answer)

    source_mean = sum(source_vals) / len(source_vals)
    candidate_mean = sum(candidate_vals) / len(candidate_vals)
    mean_drop = source_mean - candidate_mean
    mean_ok = mean_drop <= args.max_mean_drop

    print()
    print("Answer retention summary")
    print("------------------------")
    print("Source mean similarity   :", f"{source_mean:.6f}")
    print("Candidate mean similarity:", f"{candidate_mean:.6f}")
    print("Mean drop                :", f"{mean_drop:+.6f}")
    print("Per-item failures        :", failures)
    result = mean_ok and failures == 0
    print("RESULT                   :", "PASS" if result else "FAIL")

    raise SystemExit(0 if result else 1)


if __name__ == "__main__":
    main()
