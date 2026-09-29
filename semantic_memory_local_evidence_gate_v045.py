# semantic_memory_local_evidence_gate_v045.py
#
# LLM_SEM v0.4.5 Local Evidence Gate
#
# Goal:
#   Reduce false high-confidence overrides by checking local adaptive-memory
#   neighborhood purity when Multi-Prototype disagrees with the Base Router.
#
# Baseline:
#   ACCEPT if memory_sim >= 0.80 AND memory_label == base_label
#
# Conditional override:
#   memory_label != base_label
#   AND memory_sim >= override_sim_threshold
#   AND top-k adaptive-neighbor majority label == memory_label
#   AND local purity >= purity_threshold
#
# Sweep:
#   k                  : 3, 5
#   override sim       : 0.90, 0.92, 0.94, 0.95, 0.96, 0.97
#   local purity       : 0.60, 0.67, 0.75, 0.80, 1.00
#
# Evaluation:
#   samples / label : 2, 3, 5, 10
#   subset seeds    : 1..5
#   prototypes      : 2 / label
#   Base Router     : fixed on base benchmark only

from __future__ import annotations

import argparse
import csv
import random
from collections import Counter, defaultdict
from pathlib import Path

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

BASE_SIM_THRESHOLD = 0.80
PROTOTYPES_PER_LABEL = 2

SAMPLE_COUNTS = (2, 3, 5, 10)
SEEDS = (1, 2, 3, 4, 5)

K_VALUES = (3, 5)
OVERRIDE_SIM_THRESHOLDS = (0.90, 0.92, 0.94, 0.95, 0.96, 0.97)
PURITY_THRESHOLDS = (0.60, 0.67, 0.75, 0.80, 1.00)


def unit(v):
    return F.normalize(v, p=2, dim=0)


def parse_args():
    p = argparse.ArgumentParser(
        description="LLM_SEM v0.4.5 Local Evidence Gate sweep."
    )
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    p.add_argument("--benchmark", default=DEFAULT_BENCHMARK)
    p.add_argument("--train", default=DEFAULT_TRAIN)
    p.add_argument("--cases", default=DEFAULT_CASES)
    p.add_argument("--alpha", type=float, default=0.35)
    p.add_argument(
        "--summary-csv",
        default="semantic_memory_local_evidence_gate_v045.csv",
    )
    p.add_argument(
        "--detail-csv",
        default="semantic_memory_local_evidence_gate_detail_v045.csv",
    )
    return p.parse_args()


def load_training(path):
    grouped = defaultdict(list)
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = str(row["label"]).strip()
            text = str(row["text"]).strip()
            if label and text:
                grouped[label].append(LabeledSentence(label=label, text=text))
    return grouped


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


def sample_subset(training, sample_count, seed):
    selected = []
    for label in sorted(training):
        rng = random.Random(seed * 1000 + sum(ord(c) for c in label))
        rows = list(training[label])
        rng.shuffle(rows)
        selected.extend(rows[:sample_count])
    return selected


def encode_rows(router, rows):
    return [
        {
            "label": row.label,
            "text": row.text,
            "vector": unit(router._encode_tensor(row.text)),
        }
        for row in rows
    ]


def spherical_kmeans(vectors, k, iterations=20):
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


def build_prototypes(memory):
    grouped = defaultdict(list)
    for row in memory:
        grouped[row["label"]].append(row["vector"])

    return {
        label: spherical_kmeans(vectors, PROTOTYPES_PER_LABEL)
        for label, vectors in grouped.items()
    }


def classify_prototype(query, prototypes):
    scored = []
    for label, centers in prototypes.items():
        sim = max(float(torch.dot(query, c).item()) for c in centers)
        scored.append((sim, label))
    scored.sort(reverse=True)
    return scored[0][1], scored[0][0]


def local_evidence(query, memory, k):
    neighbors = sorted(
        [
            (
                float(torch.dot(query, row["vector"]).item()),
                row["label"],
                row["text"],
            )
            for row in memory
        ],
        reverse=True,
    )[:max(1, min(k, len(memory)))]

    labels = [label for _, label, _ in neighbors]
    counts = Counter(labels)
    majority_label, majority_count = counts.most_common(1)[0]
    purity = majority_count / len(neighbors)

    return {
        "majority_label": majority_label,
        "purity": purity,
        "neighbors": neighbors,
    }


def evaluate_config(rows, sim_th, purity_th, k):
    tp = fp = tn = fn = wrong_label_accept = 0
    accepted = correct_accept = 0
    override_accepts = 0

    for row in rows:
        baseline_accept = (
            row["memory_similarity"] >= BASE_SIM_THRESHOLD
            and row["agreement"]
        )

        local = row[f"local_k{k}"]
        override_accept = (
            not row["agreement"]
            and row["memory_similarity"] >= sim_th
            and local["majority_label"] == row["memory_label"]
            and local["purity"] >= purity_th
        )

        gate_accept = baseline_accept or override_accept

        if override_accept:
            override_accepts += 1
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
        "k": k,
        "override_sim_threshold": sim_th,
        "purity_threshold": purity_th,
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
        "override_accepts": override_accepts,
    }


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

    base_router = SemanticRouter(model, tokenizer, alpha=args.alpha)
    base_router.fit(base)

    grouped_rows = defaultdict(list)
    detail_rows = []

    for sample_count in SAMPLE_COUNTS:
        for seed in SEEDS:
            selected = sample_subset(training, sample_count, seed)
            memory = encode_rows(base_router, selected)
            prototypes = build_prototypes(memory)

            for case in cases:
                query = unit(base_router._encode_tensor(case["text"]))
                mem_label, mem_sim = classify_prototype(query, prototypes)
                base_top1 = base_router.route(case["text"])[0]

                row = {
                    "samples_per_label": sample_count,
                    "seed": seed,
                    "expected_label": case["expected_label"],
                    "should_accept": case["should_accept"],
                    "text": case["text"],
                    "memory_label": mem_label,
                    "memory_similarity": mem_sim,
                    "base_label": base_top1.label,
                    "base_similarity": base_top1.similarity,
                    "agreement": mem_label == base_top1.label,
                }

                for k in K_VALUES:
                    local = local_evidence(query, memory, k)
                    row[f"local_k{k}"] = local

                grouped_rows[(sample_count, seed)].append(row)

                flat = dict(row)
                for k in K_VALUES:
                    local = flat.pop(f"local_k{k}")
                    flat[f"k{k}_majority_label"] = local["majority_label"]
                    flat[f"k{k}_purity"] = local["purity"]
                    flat[f"k{k}_neighbors"] = " | ".join(
                        f"{label}:{sim:.4f}:{text}"
                        for sim, label, text in local["neighbors"]
                    )
                detail_rows.append(flat)

    baseline_by_group = {}
    for key, rows in grouped_rows.items():
        baseline_by_group[key] = evaluate_config(
            rows,
            sim_th=999.0,
            purity_th=2.0,
            k=K_VALUES[0],
        )

    run_results = []

    for (samples, seed), rows in grouped_rows.items():
        for k in K_VALUES:
            for sim_th in OVERRIDE_SIM_THRESHOLDS:
                for purity_th in PURITY_THRESHOLDS:
                    result = evaluate_config(rows, sim_th, purity_th, k)
                    result.update({
                        "samples_per_label": samples,
                        "seed": seed,
                    })
                    run_results.append(result)

    grouped_configs = defaultdict(list)
    for row in run_results:
        key = (
            row["samples_per_label"],
            row["k"],
            row["override_sim_threshold"],
            row["purity_threshold"],
        )
        grouped_configs[key].append(row)

    metrics = [
        "precision",
        "recall",
        "specificity",
        "balanced_accuracy",
        "coverage",
        "accepted_accuracy",
        "false_accept_rate",
        "override_accepts",
    ]

    summary = []
    for key, rows in grouped_configs.items():
        samples, k, sim_th, purity_th = key

        baseline_rows = [
            baseline_by_group[(samples, seed)]
            for seed in SEEDS
        ]
        baseline_recall = mean([r["recall"] for r in baseline_rows])
        baseline_false = mean([r["false_accept_rate"] for r in baseline_rows])

        out = {
            "samples_per_label": samples,
            "k": k,
            "override_sim_threshold": sim_th,
            "purity_threshold": purity_th,
            "seeds": len(rows),
        }

        for metric in metrics:
            vals = [r[metric] for r in rows]
            out[f"{metric}_mean"] = mean(vals)
            out[f"{metric}_std"] = stdev(vals)

        out["recall_gain_vs_baseline"] = (
            out["recall_mean"] - baseline_recall
        )
        out["false_gain_vs_baseline"] = (
            out["false_accept_rate_mean"] - baseline_false
        )
        summary.append(out)

    summary.sort(
        key=lambda r: (
            r["balanced_accuracy_mean"],
            r["precision_mean"],
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
    print("=" * 118)
    print(" LLM_SEM v0.4.5 Local Evidence Gate Sweep")
    print("=" * 118)
    print("Device                 :", device)
    if device.type == "cuda":
        print("GPU                    :", torch.cuda.get_device_name(0))
    print("Checkpoint loss        :", checkpoint.get("loss"))
    print("Base Router            : FIXED (base benchmark only)")
    print("Baseline gate          : sim>=0.80 AND Base agreement")
    print("Multi-proto / label    :", PROTOTYPES_PER_LABEL)
    print("Local k values         :", ", ".join(map(str, K_VALUES)))
    print()

    print("Top local-evidence override configurations")
    print("-" * 118)
    print(
        f"{'Samp':>4} {'k':>2} {'Sim':>5} {'Purity':>6} "
        f"{'Prec':>12} {'Recall':>12} {'Spec':>12} "
        f"{'BalAcc':>12} {'False':>12} {'RGain':>8} {'FGain':>8}"
    )

    for row in summary[:20]:
        print(
            f"{row['samples_per_label']:>4d} "
            f"{row['k']:>2d} "
            f"{row['override_sim_threshold']:>5.2f} "
            f"{row['purity_threshold']:>6.2f} "
            f"{row['precision_mean']*100:>5.1f}+/-{row['precision_std']*100:<4.1f} "
            f"{row['recall_mean']*100:>5.1f}+/-{row['recall_std']*100:<4.1f} "
            f"{row['specificity_mean']*100:>5.1f}+/-{row['specificity_std']*100:<4.1f} "
            f"{row['balanced_accuracy_mean']*100:>5.1f}+/-{row['balanced_accuracy_std']*100:<4.1f} "
            f"{row['false_accept_rate_mean']*100:>5.1f}+/-{row['false_accept_rate_std']*100:<4.1f} "
            f"{row['recall_gain_vs_baseline']*100:>+7.1f}% "
            f"{row['false_gain_vs_baseline']*100:>+7.1f}%"
        )

    print()
    print("Best per teaching volume")
    print("-" * 118)
    for samples in SAMPLE_COUNTS:
        best = next(r for r in summary if r["samples_per_label"] == samples)
        print(
            f"samples={samples:>2d}: "
            f"k={best['k']}, "
            f"sim>={best['override_sim_threshold']:.2f}, "
            f"purity>={best['purity_threshold']:.2f}, "
            f"recall={best['recall_mean']*100:.1f}%, "
            f"false={best['false_accept_rate_mean']*100:.1f}%, "
            f"bal={best['balanced_accuracy_mean']*100:.1f}%"
        )

    print()
    print("Summary CSV:", args.summary_csv)
    print("Detail CSV :", args.detail_csv)


if __name__ == "__main__":
    main()
