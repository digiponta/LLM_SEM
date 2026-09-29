# semantic_memory_runtime_purity_calibration_v047.py
#
# LLM_SEM v0.4.7 Runtime Purity Calibration
#
# Purpose:
#   Calibrate Local Evidence purity thresholds against the ACTUAL runtime
#   Adaptive Memory file rather than the synthetic scaling teaching pool.
#
# Fixed runtime policy:
#   - Base Router: base benchmark only
#   - Multi-Prototype: 2 / label
#   - Base acceptance: memory_sim >= 0.80 AND memory_label == base_label
#   - Override similarity: memory_sim >= 0.92
#   - Local k: 3
#
# Sweep:
#   local purity threshold = 0.60, 0.67, 0.75, 1.00
#
# Evaluation:
#   Uses semantic_memory_gate_cases_v036.csv by default.
#   Also prints focused results for:
#     量子計算とは何ですか
#     量子力学を説明してください
#     量子暗号について説明して
#     量子状態について説明して

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from adaptive_semantic_learning import load_semantic_memory
from adaptive_semantic_runtime import (
    build_multi_prototypes,
    encode_memory,
    local_evidence,
    rank_multi_prototypes,
    unit,
)
from model import LanguageModel
from semantic_eval import load_benchmark
from semantic_router import SemanticRouter
from tokenizer import Tokenizer

DEFAULT_MODEL = "model/model-gpu-v0.4.pt"
DEFAULT_TOKENIZER = "model/tokenizer.json"
DEFAULT_BENCHMARK = "my_benchmark.csv"
DEFAULT_MEMORY = "data/semantic_memory.jsonl"
DEFAULT_CASES = "semantic_memory_gate_cases_v036.csv"

BASE_SIM_THRESHOLD = 0.80
OVERRIDE_SIM_THRESHOLD = 0.92
LOCAL_K = 3
PROTOTYPES_PER_LABEL = 2
PURITY_THRESHOLDS = (0.60, 0.67, 0.75, 1.00)

FOCUS_TEXTS = (
    "量子計算とは何ですか",
    "量子力学を説明してください",
    "量子暗号について説明して",
    "量子状態について説明して",
)


def parse_args():
    p = argparse.ArgumentParser(
        description="LLM_SEM v0.4.7 Runtime Purity Calibration."
    )
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    p.add_argument("--benchmark", default=DEFAULT_BENCHMARK)
    p.add_argument("--memory", default=DEFAULT_MEMORY)
    p.add_argument("--cases", default=DEFAULT_CASES)
    p.add_argument("--alpha", type=float, default=0.35)
    p.add_argument(
        "--summary-csv",
        default="semantic_memory_runtime_purity_calibration_v047.csv",
    )
    p.add_argument(
        "--detail-csv",
        default="semantic_memory_runtime_purity_calibration_detail_v047.csv",
    )
    return p.parse_args()


def load_cases(path):
    rows = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw = str(row["should_accept"]).strip().lower()
            rows.append({
                "expected_label": str(row["expected_label"]).strip(),
                "should_accept": raw in ("1", "true", "yes", "y"),
                "text": str(row["text"]).strip(),
            })
    return rows


def evaluate_threshold(rows, purity_threshold):
    tp = fp = tn = fn = wrong_label_accept = 0
    accepted = correct_accept = 0

    for row in rows:
        baseline_accept = (
            row["memory_similarity"] >= BASE_SIM_THRESHOLD
            and row["agreement"]
        )

        override_accept = (
            not row["agreement"]
            and row["memory_similarity"] >= OVERRIDE_SIM_THRESHOLD
            and row["local_majority_label"] == row["memory_label"]
            and row["local_purity"] >= purity_threshold
        )

        gate_accept = baseline_accept or override_accept

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

    target_count = sum(int(r["should_accept"]) for r in rows)
    reject_count = len(rows) - target_count

    precision = tp / max(1, tp + fp + wrong_label_accept)
    recall = tp / max(1, target_count)
    specificity = tn / max(1, reject_count)
    balanced = (recall + specificity) / 2.0
    coverage = accepted / max(1, len(rows))
    accepted_accuracy = correct_accept / max(1, accepted)
    false_accept_rate = (fp + wrong_label_accept) / max(1, len(rows))

    return {
        "purity_threshold": purity_threshold,
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
    memory_rows = load_semantic_memory(Path(args.memory))
    cases = load_cases(args.cases)

    router = SemanticRouter(model, tokenizer, alpha=args.alpha)
    router.fit(base)

    encoded_memory = encode_memory(router, memory_rows)
    prototypes = build_multi_prototypes(
        encoded_memory,
        per_label=PROTOTYPES_PER_LABEL,
    )

    detail_rows = []

    for case in cases:
        query = unit(router._encode_tensor(case["text"]))

        ranked_memory = rank_multi_prototypes(query, prototypes)
        mem_sim, mem_label = ranked_memory[0]
        mem_margin = (
            mem_sim - ranked_memory[1][0]
            if len(ranked_memory) > 1
            else None
        )

        base_top1 = router.route(case["text"])[0]
        majority_label, purity, neighbors = local_evidence(
            query,
            encoded_memory,
            k=LOCAL_K,
        )

        detail_rows.append({
            "expected_label": case["expected_label"],
            "should_accept": case["should_accept"],
            "text": case["text"],
            "memory_label": mem_label,
            "memory_similarity": mem_sim,
            "memory_margin": mem_margin,
            "base_label": base_top1.label,
            "base_similarity": base_top1.similarity,
            "agreement": mem_label == base_top1.label,
            "local_majority_label": majority_label,
            "local_purity": purity,
            "neighbors": " | ".join(
                f"{label}:{sim:.6f}:{text}"
                for sim, label, text in neighbors
            ),
        })

    summary = [
        evaluate_threshold(detail_rows, threshold)
        for threshold in PURITY_THRESHOLDS
    ]

    summary.sort(
        key=lambda r: (
            r["balanced_accuracy"],
            r["precision"],
            -r["false_accept_rate"],
            r["recall"],
        ),
        reverse=True,
    )

    with Path(args.summary_csv).open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)

    with Path(args.detail_csv).open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(detail_rows[0].keys()))
        writer.writeheader()
        writer.writerows(detail_rows)

    print()
    print("=" * 106)
    print(" LLM_SEM v0.4.7 Runtime Purity Calibration")
    print("=" * 106)
    print("Device                :", device)
    if device.type == "cuda":
        print("GPU                   :", torch.cuda.get_device_name(0))
    print("Checkpoint loss       :", checkpoint.get("loss"))
    print("Base Router           : FIXED (base benchmark only)")
    print("Adaptive samples      :", len(memory_rows))
    print("Memory labels         :", len(prototypes))
    print("Prototypes / label    :", PROTOTYPES_PER_LABEL)
    print("Override sim          :", f"{OVERRIDE_SIM_THRESHOLD:.2f}")
    print("Local k               :", LOCAL_K)
    print()

    print("Purity calibration")
    print("-" * 106)
    print(
        f"{'Purity':>6} {'Prec':>8} {'Recall':>8} {'Spec':>8} "
        f"{'BalAcc':>8} {'Cover':>8} {'AccptAcc':>9} {'False':>8}"
    )

    for row in sorted(summary, key=lambda r: r["purity_threshold"]):
        print(
            f"{row['purity_threshold']:>6.2f} "
            f"{row['precision']*100:>7.1f}% "
            f"{row['recall']*100:>7.1f}% "
            f"{row['specificity']*100:>7.1f}% "
            f"{row['balanced_accuracy']*100:>7.1f}% "
            f"{row['coverage']*100:>7.1f}% "
            f"{row['accepted_accuracy']*100:>8.1f}% "
            f"{row['false_accept_rate']*100:>7.1f}%"
        )

    print()
    print("Focused runtime cases")
    print("-" * 106)

    lookup = {row["text"]: row for row in detail_rows}
    for text in FOCUS_TEXTS:
        row = lookup.get(text)
        if row is None:
            print(f"{text}: not present in evaluation CSV")
            continue

        print(
            f"{text}\n"
            f"  memory={row['memory_label']} "
            f"sim={row['memory_similarity']:.6f} "
            f"margin={row['memory_margin']:.6f} "
            f"base={row['base_label']} "
            f"local={row['local_majority_label']} "
            f"purity={row['local_purity']:.3f}"
        )

        for threshold in PURITY_THRESHOLDS:
            baseline_accept = (
                row["memory_similarity"] >= BASE_SIM_THRESHOLD
                and row["agreement"]
            )
            override_accept = (
                not row["agreement"]
                and row["memory_similarity"] >= OVERRIDE_SIM_THRESHOLD
                and row["local_majority_label"] == row["memory_label"]
                and row["local_purity"] >= threshold
            )
            action = (
                "ACCEPT"
                if baseline_accept
                else "ADAPTIVE_OVERRIDE"
                if override_accept
                else "GATE_REVIEW"
            )
            print(f"    purity>={threshold:.2f}: {action}")

    best = summary[0]
    print()
    print("Best runtime purity threshold")
    print("-" * 106)
    print("Purity threshold      :", f"{best['purity_threshold']:.2f}")
    print("Precision             :", f"{best['precision']*100:.1f}%")
    print("Recall                :", f"{best['recall']*100:.1f}%")
    print("Specificity           :", f"{best['specificity']*100:.1f}%")
    print("Balanced accuracy     :", f"{best['balanced_accuracy']*100:.1f}%")
    print("False accept rate     :", f"{best['false_accept_rate']*100:.1f}%")
    print()
    print("Summary CSV:", args.summary_csv)
    print("Detail CSV :", args.detail_csv)


if __name__ == "__main__":
    main()
