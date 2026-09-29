# semantic_memory_base_agreement_ablation_v042.py
#
# LLM_SEM v0.4.2 Base Agreement Ablation
#
# Purpose:
#   Determine whether the Base Agreement Gate limits recall.
#
# Compare under identical conditions:
#   A) Memory-only gate:
#        memory_similarity >= threshold
#   B) Memory + Base Agreement gate:
#        memory_similarity >= threshold
#        AND memory_label == base_label
#
# Sweep:
#   teaching samples / label : 2, 3, 5, 10
#   prototypes / label       : 1, 2, 3
#   subset seeds             : 1..5
#
# Base Router is fixed on the base benchmark only.

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


def unit(v):
    return F.normalize(v, p=2, dim=0)


def parse_args():
    p = argparse.ArgumentParser(
        description="LLM_SEM v0.4.2 Base Agreement Ablation."
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
        default="semantic_memory_base_agreement_ablation_v042.csv",
    )
    p.add_argument(
        "--detail-csv",
        default="semantic_memory_base_agreement_ablation_detail_v042.csv",
    )
    return p.parse_args()


def load_training(path: str) -> Dict[str, List[LabeledSentence]]:
    grouped = defaultdict(list)
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


def build_prototypes(memory, per_label):
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


def evaluate_mode(base_router, cases, prototypes, sim_th, require_agreement):
    tp = fp = tn = fn = wrong_label_accept = 0
    accepted = correct_accept = 0
    detail = []

    for case in cases:
        query = unit(base_router._encode_tensor(case["text"]))
        mem_label, mem_sim = classify(query, prototypes)
        base_top1 = base_router.route(case["text"])[0]
        agreement = mem_label == base_top1.label

        gate_accept = (
            mem_sim >= sim_th
            and (agreement or not require_agreement)
        )

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
            "agreement": agreement,
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

    base_router = SemanticRouter(model, tokenizer, alpha=args.alpha)
    base_router.fit(base)

    runs = []
    details_all = []

    for sample_count in SAMPLE_COUNTS:
        for prototype_count in PROTOTYPE_COUNTS:
            for seed in SEEDS:
                selected = sample_subset(training, sample_count, seed)
                memory = encode_rows(base_router, selected)
                prototypes = build_prototypes(memory, prototype_count)

                for require_agreement in (False, True):
                    metrics, details = evaluate_mode(
                        base_router,
                        cases,
                        prototypes,
                        args.sim_th,
                        require_agreement,
                    )

                    mode = "memory_plus_base" if require_agreement else "memory_only"

                    runs.append({
                        "samples_per_label": sample_count,
                        "prototypes_per_label": prototype_count,
                        "seed": seed,
                        "mode": mode,
                        **metrics,
                    })

                    for d in details:
                        details_all.append({
                            "samples_per_label": sample_count,
                            "prototypes_per_label": prototype_count,
                            "seed": seed,
                            "mode": mode,
                            **d,
                        })

    grouped = defaultdict(list)
    for row in runs:
        grouped[
            (
                row["samples_per_label"],
                row["prototypes_per_label"],
                row["mode"],
            )
        ].append(row)

    metric_names = [
        "precision",
        "recall",
        "specificity",
        "balanced_accuracy",
        "coverage",
        "accepted_accuracy",
        "false_accept_rate",
    ]

    summary = []
    for (samples, protos, mode), rows in grouped.items():
        out = {
            "samples_per_label": samples,
            "prototypes_per_label": protos,
            "mode": mode,
            "seeds": len(rows),
        }
        for metric in metric_names:
            vals = [r[metric] for r in rows]
            out[f"{metric}_mean"] = mean(vals)
            out[f"{metric}_std"] = stdev(vals)
        summary.append(out)

    with Path(args.summary_csv).open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)

    with Path(args.detail_csv).open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(details_all[0].keys()))
        writer.writeheader()
        writer.writerows(details_all)

    print()
    print("=" * 114)
    print(" LLM_SEM v0.4.2 Base Agreement Ablation")
    print("=" * 114)
    print("Device                :", device)
    if device.type == "cuda":
        print("GPU                   :", torch.cuda.get_device_name(0))
    print("Checkpoint loss       :", checkpoint.get("loss"))
    print("Base Router           : FIXED (base benchmark only)")
    print("Similarity threshold  :", f"{args.sim_th:.2f}")
    print("Modes                 : memory_only vs memory_plus_base")
    print()

    print("Recall / precision impact of Base Agreement")
    print("-" * 114)
    print(
        f"{'Samples':>7} {'Proto':>5} {'Mode':<17} "
        f"{'Prec':>13} {'Recall':>13} {'Spec':>13} "
        f"{'BalAcc':>13} {'False':>13}"
    )

    for row in sorted(
        summary,
        key=lambda r: (
            r["samples_per_label"],
            r["prototypes_per_label"],
            r["mode"],
        ),
    ):
        print(
            f"{row['samples_per_label']:>7d} "
            f"{row['prototypes_per_label']:>5d} "
            f"{row['mode']:<17} "
            f"{row['precision_mean']*100:>5.1f}+/-{row['precision_std']*100:<5.1f} "
            f"{row['recall_mean']*100:>5.1f}+/-{row['recall_std']*100:<5.1f} "
            f"{row['specificity_mean']*100:>5.1f}+/-{row['specificity_std']*100:<5.1f} "
            f"{row['balanced_accuracy_mean']*100:>5.1f}+/-{row['balanced_accuracy_std']*100:<5.1f} "
            f"{row['false_accept_rate_mean']*100:>5.1f}+/-{row['false_accept_rate_std']*100:<5.1f}"
        )

    print()
    print("Recall delta: memory_only - memory_plus_base")
    print("-" * 114)

    lookup = {
        (r["samples_per_label"], r["prototypes_per_label"], r["mode"]): r
        for r in summary
    }

    print(f"{'Samples':>7} {'Proto':>5} {'Recall gain':>12} {'False gain':>12}")
    for samples in SAMPLE_COUNTS:
        for protos in PROTOTYPE_COUNTS:
            mem = lookup[(samples, protos, "memory_only")]
            baseg = lookup[(samples, protos, "memory_plus_base")]

            recall_gain = (
                mem["recall_mean"] - baseg["recall_mean"]
            ) * 100.0
            false_gain = (
                mem["false_accept_rate_mean"]
                - baseg["false_accept_rate_mean"]
            ) * 100.0

            print(
                f"{samples:>7d} {protos:>5d} "
                f"{recall_gain:>+11.1f}% "
                f"{false_gain:>+11.1f}%"
            )

    print()
    print("Summary CSV:", args.summary_csv)
    print("Detail CSV :", args.detail_csv)


if __name__ == "__main__":
    main()
