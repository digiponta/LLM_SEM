# semantic_memory_multi_prototype_scaling_v040.py
#
# LLM_SEM v0.4.0 Multi-Prototype Scaling
#
# Sweep:
#   teaching samples per label : 2, 3, 5, 10
#   prototypes per label       : 1, 2, 3
#
# Fair conditions:
#   - fixed held-out evaluation cases
#   - L2-normalized memory/query/prototype vectors
#   - cosine similarity
#   - similarity threshold = 0.80
#   - Base agreement required
#   - deterministic teaching prefixes from a fixed scaling training CSV

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
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


@dataclass(frozen=True)
class MemoryVector:
    label: str
    text: str
    vector: torch.Tensor


def unit(vector: torch.Tensor) -> torch.Tensor:
    return F.normalize(vector, p=2, dim=0)


def parse_args():
    p = argparse.ArgumentParser(
        description="LLM_SEM v0.4.0 Multi-Prototype Scaling."
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
        default="semantic_memory_multi_prototype_scaling_v040.csv",
    )
    p.add_argument(
        "--detail-csv",
        default="semantic_memory_multi_prototype_scaling_detail_v040.csv",
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


def encode_memory(router, rows: Sequence[LabeledSentence]) -> List[MemoryVector]:
    return [
        MemoryVector(
            label=row.label,
            text=row.text,
            vector=unit(router._encode_tensor(row.text)),
        )
        for row in rows
    ]


def spherical_kmeans(
    vectors: Sequence[torch.Tensor],
    k: int,
    iterations: int = 20,
) -> List[torch.Tensor]:
    if not vectors:
        return []

    k = max(1, min(k, len(vectors)))

    if k == 1:
        return [unit(torch.stack(list(vectors), dim=0).mean(dim=0))]

    if len(vectors) <= k:
        return [unit(v.clone()) for v in vectors]

    centers = [vectors[0].clone()]

    while len(centers) < k:
        best_index = 0
        best_distance = -1.0
        for i, vector in enumerate(vectors):
            nearest_similarity = max(
                float(torch.dot(vector, center).item())
                for center in centers
            )
            distance = 1.0 - nearest_similarity
            if distance > best_distance:
                best_distance = distance
                best_index = i
        centers.append(vectors[best_index].clone())

    centers = [unit(c) for c in centers]

    for _ in range(iterations):
        groups = [[] for _ in range(k)]

        for vector in vectors:
            sims = [
                float(torch.dot(vector, center).item())
                for center in centers
            ]
            index = max(range(k), key=lambda i: sims[i])
            groups[index].append(vector)

        new_centers = []
        for i, group in enumerate(groups):
            if group:
                new_centers.append(
                    unit(torch.stack(group, dim=0).mean(dim=0))
                )
            else:
                new_centers.append(centers[i])

        stable = all(
            float(torch.dot(a, b).item()) > 0.999999
            for a, b in zip(centers, new_centers)
        )
        centers = new_centers
        if stable:
            break

    return centers


def build_multi_prototypes(
    memory: Sequence[MemoryVector],
    prototype_count: int,
):
    grouped: Dict[str, List[torch.Tensor]] = defaultdict(list)
    for row in memory:
        grouped[row.label].append(row.vector)

    return {
        label: spherical_kmeans(vectors, prototype_count)
        for label, vectors in grouped.items()
    }


def classify(query: torch.Tensor, prototypes):
    scored = []
    for label, centers in prototypes.items():
        similarity = max(
            float(torch.dot(query, center).item())
            for center in centers
        )
        scored.append((similarity, label))
    scored.sort(reverse=True)
    return scored[0][1], scored[0][0]


def evaluate(router, cases, prototypes, sim_th):
    tp = fp = tn = fn = wrong_label_accept = 0
    accepted = correct_accept = 0
    detail_rows = []

    for case in cases:
        query = unit(router._encode_tensor(case["text"]))
        memory_label, memory_similarity = classify(query, prototypes)
        base_top1 = router.route(case["text"])[0]
        agreement = memory_label == base_top1.label
        gate_accept = memory_similarity >= sim_th and agreement

        correct_label = (
            case["expected_label"] != "REJECT"
            and memory_label == case["expected_label"]
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

        detail_rows.append({
            "expected_label": case["expected_label"],
            "should_accept": case["should_accept"],
            "text": case["text"],
            "memory_label": memory_label,
            "memory_similarity": memory_similarity,
            "base_label": base_top1.label,
            "base_similarity": base_top1.similarity,
            "agreement": agreement,
            "gate_accept": gate_accept,
            "correct_label": correct_label,
        })

    target_count = sum(int(row["should_accept"]) for row in cases)
    reject_count = len(cases) - target_count

    precision = tp / max(1, tp + fp + wrong_label_accept)
    recall = tp / max(1, target_count)
    specificity = tn / max(1, reject_count)
    balanced = (recall + specificity) / 2.0
    coverage = accepted / max(1, len(cases))
    accepted_accuracy = correct_accept / max(1, accepted)
    false_accept_rate = (fp + wrong_label_accept) / max(1, len(cases))

    metrics = {
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
    return metrics, detail_rows


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = Tokenizer.load(args.tokenizer)
    model, checkpoint = LanguageModel.load_checkpoint(args.model, device=device)

    base = load_benchmark(args.benchmark)
    training = load_training(args.train)
    cases = load_cases(args.cases)

    labels = sorted(training)
    if len(labels) < 2:
        raise ValueError("Scaling training set needs at least two labels.")

    for label in labels:
        if len(training[label]) < max(SAMPLE_COUNTS):
            raise ValueError(
                f"Label {label!r} has {len(training[label])} samples; "
                f"{max(SAMPLE_COUNTS)} required."
            )

    summary_rows = []
    all_detail_rows = []

    print()
    print("=" * 98)
    print(" LLM_SEM v0.4.0 Multi-Prototype Scaling")
    print("=" * 98)
    print("Device                :", device)
    if device.type == "cuda":
        print("GPU                   :", torch.cuda.get_device_name(0))
    print("Checkpoint loss       :", checkpoint.get("loss"))
    print("Labels                :", ", ".join(labels))
    print("Evaluation cases      :", len(cases))
    print("Similarity threshold  :", f"{args.sim_th:.2f}")
    print("Base agreement        : REQUIRED")
    print("Sample counts / label :", ", ".join(map(str, SAMPLE_COUNTS)))
    print("Prototype counts      :", ", ".join(map(str, PROTOTYPE_COUNTS)))
    print("Normalization         : L2 unit vectors")
    print()

    for sample_count in SAMPLE_COUNTS:
        selected = []
        for label in labels:
            selected.extend(training[label][:sample_count])

        router = SemanticRouter(model, tokenizer, alpha=args.alpha)
        router.fit(list(base) + selected)
        memory = encode_memory(router, selected)

        for prototype_count in PROTOTYPE_COUNTS:
            prototypes = build_multi_prototypes(memory, prototype_count)
            metrics, details = evaluate(
                router,
                cases,
                prototypes,
                args.sim_th,
            )

            row = {
                "samples_per_label": sample_count,
                "prototypes_per_label": prototype_count,
                **metrics,
            }
            summary_rows.append(row)

            for detail in details:
                all_detail_rows.append({
                    "samples_per_label": sample_count,
                    "prototypes_per_label": prototype_count,
                    **detail,
                })

    summary_rows.sort(
        key=lambda row: (
            row["balanced_accuracy"],
            row["accepted_accuracy"],
            -row["false_accept_rate"],
            row["recall"],
            row["coverage"],
        ),
        reverse=True,
    )

    with Path(args.summary_csv).open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    with Path(args.detail_csv).open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(all_detail_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_detail_rows)

    print("Scaling results")
    print("-" * 98)
    print(
        f"{'Samples':>7} {'Proto':>5} {'Prec':>7} {'Recall':>7} "
        f"{'Spec':>7} {'BalAcc':>7} {'Cover':>7} "
        f"{'AccptAcc':>8} {'False':>7}"
    )

    for row in sorted(
        summary_rows,
        key=lambda x: (x["samples_per_label"], x["prototypes_per_label"]),
    ):
        print(
            f"{row['samples_per_label']:>7d} "
            f"{row['prototypes_per_label']:>5d} "
            f"{row['precision']*100:>6.1f}% "
            f"{row['recall']*100:>6.1f}% "
            f"{row['specificity']*100:>6.1f}% "
            f"{row['balanced_accuracy']*100:>6.1f}% "
            f"{row['coverage']*100:>6.1f}% "
            f"{row['accepted_accuracy']*100:>7.1f}% "
            f"{row['false_accept_rate']*100:>6.1f}%"
        )

    best = summary_rows[0]
    print()
    print("Best configuration")
    print("-" * 98)
    print("Samples / label      :", best["samples_per_label"])
    print("Prototypes / label   :", best["prototypes_per_label"])
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
