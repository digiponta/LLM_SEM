# semantic_memory_enrichment_experiment_v048.py
#
# LLM_SEM v0.4.8 Adaptive Memory Enrichment Experiment
#
# Purpose:
#   Test whether adding semantically targeted teaching examples improves the
#   LOCAL STRUCTURE of Adaptive Memory, rather than adding more gate rules.
#
# Starting point:
#   Actual runtime memory: data/semantic_memory.jsonl
#
# Enrichment:
#   A separate controlled pool adds 0 / 2 / 4 / 6 examples per target label.
#   Evaluation texts are not copied verbatim into the enrichment pool.
#
# Fixed runtime conditions:
#   Base Router          : fixed on base benchmark only
#   Multi-Prototype      : 2 / label
#   Base accept sim      : 0.80
#   Override sim         : 0.92
#   Local k              : 3
#   Override purity      : 1.00 (unanimous top-3)
#
# Main question:
#   Does enrichment turn ambiguous/wrong local neighborhoods into stable,
#   label-consistent neighborhoods for the focused quantum queries?

from __future__ import annotations

import argparse
import csv
from collections import Counter
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
from semantic_eval import LabeledSentence, load_benchmark
from semantic_router import SemanticRouter
from tokenizer import Tokenizer

DEFAULT_MODEL = "model/model-gpu-v0.4.pt"
DEFAULT_TOKENIZER = "model/tokenizer.json"
DEFAULT_BENCHMARK = "my_benchmark.csv"
DEFAULT_MEMORY = "data/semantic_memory.jsonl"
DEFAULT_ENRICHMENT = "semantic_memory_enrichment_train_v048.csv"
DEFAULT_CASES = "semantic_memory_gate_cases_v036.csv"

ENRICHMENT_LEVELS = (0, 2, 4, 6)
PROTOTYPES_PER_LABEL = 2
BASE_SIM_THRESHOLD = 0.80
OVERRIDE_SIM_THRESHOLD = 0.92
LOCAL_K = 3
LOCAL_PURITY_THRESHOLD = 1.00

FOCUS_TEXTS = (
    "量子計算とは何ですか",
    "量子力学を説明してください",
    "量子暗号について説明して",
    "量子状態について説明して",
)


def parse_args():
    p = argparse.ArgumentParser(
        description="LLM_SEM v0.4.8 Adaptive Memory Enrichment Experiment."
    )
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    p.add_argument("--benchmark", default=DEFAULT_BENCHMARK)
    p.add_argument("--memory", default=DEFAULT_MEMORY)
    p.add_argument("--enrichment", default=DEFAULT_ENRICHMENT)
    p.add_argument("--cases", default=DEFAULT_CASES)
    p.add_argument("--alpha", type=float, default=0.35)
    p.add_argument(
        "--summary-csv",
        default="semantic_memory_enrichment_experiment_v048.csv",
    )
    p.add_argument(
        "--detail-csv",
        default="semantic_memory_enrichment_experiment_detail_v048.csv",
    )
    return p.parse_args()


def load_csv_labeled(path):
    rows = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = str(row["label"]).strip()
            text = str(row["text"]).strip()
            if label and text:
                rows.append(LabeledSentence(label=label, text=text))
    return rows


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


def select_enrichment(pool, per_label):
    if per_label == 0:
        return []

    grouped = {}
    for row in pool:
        grouped.setdefault(row.label, []).append(row)

    selected = []
    for label in sorted(grouped):
        selected.extend(grouped[label][:per_label])
    return selected


def dedupe(rows):
    out = []
    seen = set()
    for row in rows:
        key = (row.label, row.text.strip())
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def evaluate(rows):
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
            and row["local_purity"] >= LOCAL_PURITY_THRESHOLD
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
    runtime_memory = load_semantic_memory(Path(args.memory))
    enrichment_pool = load_csv_labeled(args.enrichment)
    cases = load_cases(args.cases)

    router = SemanticRouter(model, tokenizer, alpha=args.alpha)
    router.fit(base)

    summary = []
    all_details = []

    for level in ENRICHMENT_LEVELS:
        added = select_enrichment(enrichment_pool, level)
        memory_rows = dedupe(list(runtime_memory) + added)

        encoded_memory = encode_memory(router, memory_rows)
        prototypes = build_multi_prototypes(
            encoded_memory,
            per_label=PROTOTYPES_PER_LABEL,
        )

        details = []
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

            row = {
                "enrichment_per_label": level,
                "memory_samples": len(memory_rows),
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
            }
            details.append(row)
            all_details.append(row)

        metrics = evaluate(details)
        summary.append({
            "enrichment_per_label": level,
            "added_samples": len(added),
            "total_memory_samples": len(memory_rows),
            **metrics,
        })

    with Path(args.summary_csv).open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)

    with Path(args.detail_csv).open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(all_details[0].keys()))
        writer.writeheader()
        writer.writerows(all_details)

    print()
    print("=" * 112)
    print(" LLM_SEM v0.4.8 Adaptive Memory Enrichment Experiment")
    print("=" * 112)
    print("Device                :", device)
    if device.type == "cuda":
        print("GPU                   :", torch.cuda.get_device_name(0))
    print("Checkpoint loss       :", checkpoint.get("loss"))
    print("Base Router           : FIXED (base benchmark only)")
    print("Runtime memory start  :", len(runtime_memory))
    print("Enrichment levels     :", ", ".join(map(str, ENRICHMENT_LEVELS)))
    print("Prototypes / label    :", PROTOTYPES_PER_LABEL)
    print("Override sim          :", f"{OVERRIDE_SIM_THRESHOLD:.2f}")
    print("Local k               :", LOCAL_K)
    print("Required purity       :", f"{LOCAL_PURITY_THRESHOLD:.2f}")
    print()

    print("Enrichment scaling")
    print("-" * 112)
    print(
        f"{'Add/label':>9} {'Total':>6} {'Prec':>8} {'Recall':>8} "
        f"{'Spec':>8} {'BalAcc':>8} {'Cover':>8} {'False':>8}"
    )
    for row in summary:
        print(
            f"{row['enrichment_per_label']:>9d} "
            f"{row['total_memory_samples']:>6d} "
            f"{row['precision']*100:>7.1f}% "
            f"{row['recall']*100:>7.1f}% "
            f"{row['specificity']*100:>7.1f}% "
            f"{row['balanced_accuracy']*100:>7.1f}% "
            f"{row['coverage']*100:>7.1f}% "
            f"{row['false_accept_rate']*100:>7.1f}%"
        )

    print()
    print("Focused local-structure evolution")
    print("-" * 112)

    by_level_text = {
        (r["enrichment_per_label"], r["text"]): r
        for r in all_details
    }

    for text in FOCUS_TEXTS:
        print(text)
        for level in ENRICHMENT_LEVELS:
            row = by_level_text.get((level, text))
            if row is None:
                print(f"  +{level}/label: not in evaluation set")
                continue

            baseline_accept = (
                row["memory_similarity"] >= BASE_SIM_THRESHOLD
                and row["agreement"]
            )
            override_accept = (
                not row["agreement"]
                and row["memory_similarity"] >= OVERRIDE_SIM_THRESHOLD
                and row["local_majority_label"] == row["memory_label"]
                and row["local_purity"] >= LOCAL_PURITY_THRESHOLD
            )
            action = (
                "ACCEPT"
                if baseline_accept
                else "ADAPTIVE_OVERRIDE"
                if override_accept
                else "GATE_REVIEW"
            )

            print(
                f"  +{level}/label: "
                f"memory={row['memory_label']} "
                f"sim={row['memory_similarity']:.3f} "
                f"local={row['local_majority_label']} "
                f"purity={row['local_purity']:.3f} "
                f"base={row['base_label']} "
                f"=> {action}"
            )

    print()
    print("Summary CSV:", args.summary_csv)
    print("Detail CSV :", args.detail_csv)


if __name__ == "__main__":
    main()
