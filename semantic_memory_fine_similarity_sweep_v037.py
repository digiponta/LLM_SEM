# semantic_memory_fine_similarity_sweep_v037.py
#
# LLM_SEM v0.3.7
# Fine similarity sweep with:
#   - Base agreement required
#   - Memory margin ignored for gating
#   - memory similarity swept from 0.76 to 0.85
#
# Reuses the same evaluation protocol/cases as v0.3.6.

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from adaptive_semantic_learning import load_semantic_memory, merge_samples
from model import LanguageModel
from semantic_eval import load_benchmark
from semantic_memory_prototype import (
    build_prototypes,
    prototype_margin,
    rank_prototypes,
)
from semantic_router import SemanticRouter
from tokenizer import Tokenizer

DEFAULT_MODEL = "model/model-gpu-v0.4.pt"
DEFAULT_TOKENIZER = "model/tokenizer.json"
DEFAULT_BENCHMARK = "my_benchmark.csv"
DEFAULT_MEMORY = "data/semantic_memory.jsonl"
DEFAULT_CASES = "semantic_memory_gate_cases_v036.csv"

SIM_THRESHOLDS = tuple(round(0.76 + 0.01 * i, 2) for i in range(10))


def parse_args():
    p = argparse.ArgumentParser(
        description="LLM_SEM v0.3.7 Fine Similarity Sweep."
    )
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    p.add_argument("--benchmark", default=DEFAULT_BENCHMARK)
    p.add_argument("--memory", default=DEFAULT_MEMORY)
    p.add_argument("--cases", default=DEFAULT_CASES)
    p.add_argument("--alpha", type=float, default=0.35)
    p.add_argument(
        "--summary-csv",
        default="semantic_memory_fine_similarity_sweep_v037.csv",
    )
    p.add_argument(
        "--detail-csv",
        default="semantic_memory_fine_similarity_detail_v037.csv",
    )
    return p.parse_args()


def load_cases(path):
    rows = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw = str(row["should_accept"]).strip().lower()
            rows.append(
                {
                    "expected_label": str(row["expected_label"]).strip(),
                    "should_accept": raw in ("1", "true", "yes", "y"),
                    "text": str(row["text"]).strip(),
                }
            )
    return rows


def evaluate(router, prototypes, cases):
    details = []
    for case in cases:
        mem_scores = rank_prototypes(router, case["text"], prototypes)
        if len(mem_scores) < 2:
            raise ValueError("At least two memory labels are required.")

        mem_top1 = mem_scores[0]
        mem_margin = float(prototype_margin(mem_scores))

        base_ranked = router.route(case["text"])
        base_top1 = base_ranked[0]

        details.append(
            {
                "expected_label": case["expected_label"],
                "should_accept": case["should_accept"],
                "text": case["text"],
                "memory_label": mem_top1.label,
                "memory_similarity": mem_top1.similarity,
                "memory_margin": mem_margin,
                "base_label": base_top1.label,
                "base_similarity": base_top1.similarity,
                "agreement": mem_top1.label == base_top1.label,
            }
        )
    return details


def score_threshold(details, sim_th):
    tp = fp = tn = fn = wrong_label_accept = 0
    accepted = correct_accept = 0

    for row in details:
        gate_accept = (
            row["memory_similarity"] >= sim_th
            and row["agreement"]
        )
        if gate_accept:
            accepted += 1

        correct_label = (
            row["expected_label"] != "REJECT"
            and row["memory_label"] == row["expected_label"]
        )

        if row["should_accept"]:
            if gate_accept and correct_label:
                tp += 1
                correct_accept += 1
            else:
                fn += 1
                if gate_accept and not correct_label:
                    wrong_label_accept += 1
        else:
            if gate_accept:
                fp += 1
            else:
                tn += 1

    target_count = sum(int(x["should_accept"]) for x in details)
    reject_count = len(details) - target_count

    precision = tp / max(1, tp + fp + wrong_label_accept)
    recall = tp / max(1, target_count)
    specificity = tn / max(1, reject_count)
    balanced = (recall + specificity) / 2.0
    coverage = accepted / max(1, len(details))
    accepted_accuracy = correct_accept / max(1, accepted)
    false_accept_rate = (fp + wrong_label_accept) / max(1, len(details))

    return {
        "similarity_threshold": sim_th,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "wrong_label_accept": wrong_label_accept,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "balanced_accuracy": balanced,
        "coverage": coverage,
        "accepted_accuracy": accepted_accuracy,
        "false_accept_rate": false_accept_rate,
    }


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer.load(args.tokenizer)
    model, checkpoint = LanguageModel.load_checkpoint(args.model, device=device)

    base = load_benchmark(args.benchmark)
    memory = load_semantic_memory(Path(args.memory))
    if len({row.label for row in memory}) < 2:
        raise ValueError("At least two semantic-memory labels are required.")

    router = SemanticRouter(model, tokenizer, alpha=args.alpha)
    router.fit(merge_samples(base, memory))
    prototypes = build_prototypes(router, memory)
    cases = load_cases(args.cases)

    details = evaluate(router, prototypes, cases)

    with Path(args.detail_csv).open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(details[0].keys()))
        writer.writeheader()
        writer.writerows(details)

    rows = [score_threshold(details, th) for th in SIM_THRESHOLDS]
    rows.sort(
        key=lambda x: (
            x["balanced_accuracy"],
            x["accepted_accuracy"],
            -x["false_accept_rate"],
            x["recall"],
        ),
        reverse=True,
    )

    with Path(args.summary_csv).open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print()
    print("=" * 78)
    print(" LLM_SEM v0.3.7 Fine Similarity Sweep")
    print("=" * 78)
    print("Device           :", device)
    if device.type == "cuda":
        print("GPU              :", torch.cuda.get_device_name(0))
    print("Checkpoint loss  :", checkpoint.get("loss"))
    print("Memory samples   :", len(memory))
    print("Memory labels    :", ", ".join(sorted(prototypes)))
    print("Evaluation cases :", len(cases))
    print("Base agreement   : REQUIRED")
    print("Memory margin    : IGNORED FOR GATING")
    print()

    print("Threshold sweep")
    print("-" * 78)
    print(
        f"{'Sim':>5} {'Prec':>7} {'Recall':>7} {'Spec':>7} "
        f"{'BalAcc':>7} {'Cover':>7} {'AccptAcc':>8} {'False':>7}"
    )
    for row in sorted(rows, key=lambda x: x["similarity_threshold"]):
        print(
            f"{row['similarity_threshold']:>5.2f} "
            f"{row['precision']*100:>6.1f}% "
            f"{row['recall']*100:>6.1f}% "
            f"{row['specificity']*100:>6.1f}% "
            f"{row['balanced_accuracy']*100:>6.1f}% "
            f"{row['coverage']*100:>6.1f}% "
            f"{row['accepted_accuracy']*100:>7.1f}% "
            f"{row['false_accept_rate']*100:>6.1f}%"
        )

    best = rows[0]
    print()
    print("Best configuration")
    print("-" * 78)
    print("Similarity threshold :", f"{best['similarity_threshold']:.2f}")
    print("Precision            :", f"{best['precision']*100:.1f}%")
    print("Recall               :", f"{best['recall']*100:.1f}%")
    print("Specificity          :", f"{best['specificity']*100:.1f}%")
    print("Balanced accuracy    :", f"{best['balanced_accuracy']*100:.1f}%")
    print("Accepted accuracy    :", f"{best['accepted_accuracy']*100:.1f}%")
    print("False accept rate    :", f"{best['false_accept_rate']*100:.1f}%")
    print()
    print("Summary CSV:", args.summary_csv)
    print("Detail CSV :", args.detail_csv)


if __name__ == "__main__":
    main()
