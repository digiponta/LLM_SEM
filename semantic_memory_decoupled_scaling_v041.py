# semantic_memory_decoupled_scaling_v041.py
#
# LLM_SEM v0.4.1 Decoupled Teaching Scaling
#
# Purpose:
#   Measure the effect of teaching volume on Adaptive Multi-Prototype memory
#   while keeping the Base Router fixed.
#
# Sweep:
#   teaching samples per label : 2, 3, 5, 10
#   prototypes per label       : 1, 2, 3
#   subset seed                : 1..5
#
# Important:
#   - Base Router is fit ONLY on the fixed base benchmark.
#   - Adaptive memory vectors are built ONLY from the sampled teaching subset.
#   - Evaluation cases remain fixed.
#   - All semantic vectors are L2-normalized.
#   - Gate: similarity >= 0.80 AND memory label == fixed Base label.

from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

import torch
import torch.nn.functional as F

from model import LanguageModel
from semantic_eval import LabeledSentence, load_benchmark
from semantic_router import SemanticRouter
from tokenizer import Tokenizer

DEFAULT_MODEL = "model/model-gpu-v0.4.pt"
DEFAULT_TOKENIZER = "model/tokenizer.json"
DEFAULT_BENCHMARK = "my_benchmark.csv"
DEFAULT_TRAIN = "semantic_memory_scaling_train_v040.csv"
DEFAULT_CASES = "semantic_memory_gate_cases_v036.csv"
DEFAULT_SIM_THRESHOLD = 0.80

SAMPLE_COUNTS = (2, 3, 5, 10)
PROTOTYPE_COUNTS = (1, 2, 3)
SEEDS = (1, 2, 3, 4, 5)


def unit(v: torch.Tensor) -> torch.Tensor:
    return F.normalize(v, p=2, dim=0)


def parse_args():
    p = argparse.ArgumentParser(
        description="LLM_SEM v0.4.1 Decoupled Teaching Scaling."
    )
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    p.add_argument("--benchmark", default=DEFAULT_BENCHMARK)
    p.add_argument("--train", default=DEFAULT_TRAIN)
    p.add_argument("--cases", default=DEFAULT_CASES)
    p.add_argument("--alpha", type=float, default=0.35)
    p.add_argument("--sim-th", type=float, default=DEFAULT_SIM_THRESHOLD)
    p.add_argument(
        "--summary-csv",
        default="semantic_memory_decoupled_scaling_v041.csv",
    )
    p.add_argument(
        "--detail-csv",
        default="semantic_memory_decoupled_scaling_detail_v041.csv",
    )
    return p.parse_args()


def load_training(path: str) -> Dict[str, List[LabeledSentence]]:
    grouped: Dict[str, List[LabeledSentence]] = defaultdict(list)
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = str(row["label"]).strip()
            text = str(row["text"]).strip()
            if label and text:
                grouped[label].append(LabeledSentence(label=label, text=text))
    return grouped


def load_cases(path: str):
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


def encode_rows(router, rows: Sequence[LabeledSentence]):
    return [
        {
            "label": row.label,
            "text": row.text,
            "vector": unit(router._encode_tensor(row.text)),
        }
        for row in rows
    ]


def spherical_kmeans(vectors, k: int, iterations: int = 20):
    k = max(1, min(k, len(vectors)))
    if k == 1:
        return [unit(torch.stack(vectors, dim=0).mean(dim=0))]
    if len(vectors) <= k:
        return [unit(v.clone()) for v in vectors]

    centers = [vectors[0].clone()]
    while len(centers) < k:
        best_i = 0
        best_d = -1.0
        for i, v in enumerate(vectors):
            nearest = max(float(torch.dot(v, c).item()) for c in centers)
            d = 1.0 - nearest
            if d > best_d:
                best_d = d
                best_i = i
        centers.append(vectors[best_i].clone())

    centers = [unit(c) for c in centers]

    for _ in range(iterations):
        groups = [[] for _ in range(k)]
        for v in vectors:
            sims = [float(torch.dot(v, c).item()) for c in centers]
            groups[max(range(k), key=lambda i: sims[i])].append(v)

        new_centers = []
        for i, group in enumerate(groups):
            if group:
                new_centers.append(unit(torch.stack(group, dim=0).mean(dim=0)))
            else:
                new_centers.append(centers[i])

        if all(
            float(torch.dot(a, b).item()) > 0.999999
            for a, b in zip(centers, new_centers)
        ):
            centers = new_centers
            break
        centers = new_centers

    return centers


def build_prototypes(memory, per_label: int):
    grouped = defaultdict(list)
    for row in memory:
        grouped[row["label"]].append(row["vector"])

    return {
        label: spherical_kmeans(vectors, per_label)
        for label, vectors in grouped.items()
    }


def classify(query, prototypes):
    scored = []
    for label, centers in prototypes.items():
        sim = max(float(torch.dot(query, c).item()) for c in centers)
        scored.append((sim, label))
    scored.sort(reverse=True)
    return scored[0][1], scored[0][0]


def sample_subset(training, sample_count: int, seed: int):
    selected = []
    for label in sorted(training):
        rng = random.Random(seed * 1000 + sum(ord(c) for c in label))
        rows = list(training[label])
        rng.shuffle(rows)
        selected.extend(rows[:sample_count])
    return selected


def evaluate(base_router, cases, prototypes, sim_th):
    tp = fp = tn = fn = wrong_label_accept = 0
    accepted = correct_accept = 0
    detail = []

    for case in cases:
        query = unit(base_router._encode_tensor(case["text"]))
        mem_label, mem_sim = classify(query, prototypes)
        base_top1 = base_router.route(case["text"])[0]
        agree = mem_label == base_top1.label
        gate_accept = mem_sim >= sim_th and agree

        correct_label = (
            case["expected_label"] != "REJECT"
            and mem_label == case["expected_label"]
        )

        if gate_accept:
            accepted += 1

        if case["should_accept"]:
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

        detail.append({
            "expected_label": case["expected_label"],
            "should_accept": case["should_accept"],
            "text": case["text"],
            "memory_label": mem_label,
            "memory_similarity": mem_sim,
            "base_label": base_top1.label,
            "base_similarity": base_top1.similarity,
            "agreement": agree,
            "gate_accept": gate_accept,
            "correct_label": correct_label,
        })

    target_count = sum(int(x["should_accept"]) for x in cases)
    reject_count = len(cases) - target_count

    precision = tp / max(1, tp + fp + wrong_label_accept)
    recall = tp / max(1, target_count)
    specificity = tn / max(1, reject_count)
    balanced = (recall + specificity) / 2.0
    coverage = accepted / max(1, len(cases))
    accepted_accuracy = correct_accept / max(1, accepted)
    false_accept_rate = (fp + wrong_label_accept) / max(1, len(cases))

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
    }, detail


def mean(values):
    return sum(values) / len(values)


def stdev(values):
    if len(values) < 2:
        return 0.0
    m = mean(values)
    return (sum((x - m) ** 2 for x in values) / (len(values) - 1)) ** 0.5


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = Tokenizer.load(args.tokenizer)
    model, checkpoint = LanguageModel.load_checkpoint(args.model, device=device)

    base = load_benchmark(args.benchmark)
    training = load_training(args.train)
    cases = load_cases(args.cases)

    for label, rows in training.items():
        if len(rows) < max(SAMPLE_COUNTS):
            raise ValueError(
                f"{label}: {len(rows)} training rows; "
                f"{max(SAMPLE_COUNTS)} required."
            )

    # Critical v0.4.1 change: Base Router is fixed once, from BASE ONLY.
    base_router = SemanticRouter(model, tokenizer, alpha=args.alpha)
    base_router.fit(base)

    runs = []
    detail_rows = []

    for sample_count in SAMPLE_COUNTS:
        for prototype_count in PROTOTYPE_COUNTS:
            for seed in SEEDS:
                selected = sample_subset(training, sample_count, seed)
                memory = encode_rows(base_router, selected)
                prototypes = build_prototypes(memory, prototype_count)
                metrics, details = evaluate(
                    base_router, cases, prototypes, args.sim_th
                )

                run = {
                    "samples_per_label": sample_count,
                    "prototypes_per_label": prototype_count,
                    "seed": seed,
                    **metrics,
                }
                runs.append(run)

                for d in details:
                    detail_rows.append({
                        "samples_per_label": sample_count,
                        "prototypes_per_label": prototype_count,
                        "seed": seed,
                        **d,
                    })

    grouped = defaultdict(list)
    for row in runs:
        grouped[
            (row["samples_per_label"], row["prototypes_per_label"])
        ].append(row)

    summary = []
    metric_names = [
        "precision",
        "recall",
        "specificity",
        "balanced_accuracy",
        "coverage",
        "accepted_accuracy",
        "false_accept_rate",
    ]

    for (samples, protos), rows in grouped.items():
        out = {
            "samples_per_label": samples,
            "prototypes_per_label": protos,
            "seeds": len(rows),
        }
        for metric in metric_names:
            vals = [r[metric] for r in rows]
            out[f"{metric}_mean"] = mean(vals)
            out[f"{metric}_std"] = stdev(vals)
        summary.append(out)

    summary.sort(
        key=lambda r: (
            r["balanced_accuracy_mean"],
            r["accepted_accuracy_mean"],
            -r["false_accept_rate_mean"],
            r["recall_mean"],
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
    print("=" * 104)
    print(" LLM_SEM v0.4.1 Decoupled Teaching Scaling")
    print("=" * 104)
    print("Device                :", device)
    if device.type == "cuda":
        print("GPU                   :", torch.cuda.get_device_name(0))
    print("Checkpoint loss       :", checkpoint.get("loss"))
    print("Base Router           : FIXED (base benchmark only)")
    print("Similarity threshold  :", f"{args.sim_th:.2f}")
    print("Base agreement        : REQUIRED")
    print("Samples / label       :", ", ".join(map(str, SAMPLE_COUNTS)))
    print("Prototypes / label    :", ", ".join(map(str, PROTOTYPE_COUNTS)))
    print("Subset seeds          :", ", ".join(map(str, SEEDS)))
    print()

    print("Teaching-volume scaling (mean +/- std over 5 subsets)")
    print("-" * 104)
    print(
        f"{'Samples':>7} {'Proto':>5} "
        f"{'Prec':>13} {'Recall':>13} {'Spec':>13} "
        f"{'BalAcc':>13} {'False':>13}"
    )

    for row in sorted(
        summary,
        key=lambda r: (r["samples_per_label"], r["prototypes_per_label"]),
    ):
        print(
            f"{row['samples_per_label']:>7d} "
            f"{row['prototypes_per_label']:>5d} "
            f"{row['precision_mean']*100:>5.1f}+/-{row['precision_std']*100:<5.1f} "
            f"{row['recall_mean']*100:>5.1f}+/-{row['recall_std']*100:<5.1f} "
            f"{row['specificity_mean']*100:>5.1f}+/-{row['specificity_std']*100:<5.1f} "
            f"{row['balanced_accuracy_mean']*100:>5.1f}+/-{row['balanced_accuracy_std']*100:<5.1f} "
            f"{row['false_accept_rate_mean']*100:>5.1f}+/-{row['false_accept_rate_std']*100:<5.1f}"
        )

    best = summary[0]
    print()
    print("Best mean configuration")
    print("-" * 104)
    print("Samples / label    :", best["samples_per_label"])
    print("Prototypes / label :", best["prototypes_per_label"])
    print(
        "Balanced accuracy  :",
        f"{best['balanced_accuracy_mean']*100:.1f}% "
        f"+/- {best['balanced_accuracy_std']*100:.1f}%",
    )
    print(
        "Recall             :",
        f"{best['recall_mean']*100:.1f}% "
        f"+/- {best['recall_std']*100:.1f}%",
    )
    print(
        "False accept rate  :",
        f"{best['false_accept_rate_mean']*100:.1f}% "
        f"+/- {best['false_accept_rate_std']*100:.1f}%",
    )
    print()
    print("Summary CSV:", args.summary_csv)
    print("Detail CSV :", args.detail_csv)


if __name__ == "__main__":
    main()
