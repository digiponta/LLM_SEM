# semantic_memory_method_compare_v038.py
#
# LLM_SEM v0.3.8
# Compare Class Prototype vs k-NN vs Multi-Prototype using the same
# semantic memory, evaluation cases, similarity threshold and Base agreement.
#
# Gate:
#   memory similarity >= 0.80
#   AND memory label == Base label
#
# Methods:
#   class_prototype : one mean vector per label
#   knn             : top-k (default 3) similarity-weighted vote
#   multi_prototype : up to 2 spherical k-means centroids per label

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F

from adaptive_semantic_learning import load_semantic_memory, merge_samples
from model import LanguageModel
from semantic_eval import load_benchmark
from semantic_router import SemanticRouter
from tokenizer import Tokenizer

DEFAULT_MODEL = "model/model-gpu-v0.4.pt"
DEFAULT_TOKENIZER = "model/tokenizer.json"
DEFAULT_BENCHMARK = "my_benchmark.csv"
DEFAULT_MEMORY = "data/semantic_memory.jsonl"
DEFAULT_CASES = "semantic_memory_gate_cases_v036.csv"
DEFAULT_SIM_THRESHOLD = 0.80
DEFAULT_KNN_K = 3
DEFAULT_PROTOTYPES_PER_LABEL = 2


@dataclass(frozen=True)
class MemoryVector:
    label: str
    text: str
    vector: torch.Tensor


@dataclass(frozen=True)
class MethodResult:
    label: str
    similarity: float


def parse_args():
    p = argparse.ArgumentParser(
        description="LLM_SEM v0.3.8 Class Prototype vs k-NN vs Multi-Prototype."
    )
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    p.add_argument("--benchmark", default=DEFAULT_BENCHMARK)
    p.add_argument("--memory", default=DEFAULT_MEMORY)
    p.add_argument("--cases", default=DEFAULT_CASES)
    p.add_argument("--alpha", type=float, default=0.35)
    p.add_argument("--sim-th", type=float, default=DEFAULT_SIM_THRESHOLD)
    p.add_argument("--knn-k", type=int, default=DEFAULT_KNN_K)
    p.add_argument(
        "--prototypes-per-label",
        type=int,
        default=DEFAULT_PROTOTYPES_PER_LABEL,
    )
    p.add_argument(
        "--summary-csv",
        default="semantic_memory_method_compare_v038.csv",
    )
    p.add_argument(
        "--detail-csv",
        default="semantic_memory_method_compare_detail_v038.csv",
    )
    return p.parse_args()


def load_cases(path: str):
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


def encode_memory(router, rows) -> List[MemoryVector]:
    return [
        MemoryVector(
            label=row.label,
            text=row.text,
            vector=F.normalize(router._encode_tensor(row.text), dim=0),
        )
        for row in rows
    ]


def class_prototypes(memory: Sequence[MemoryVector]) -> Dict[str, torch.Tensor]:
    grouped: dict[str, list[torch.Tensor]] = defaultdict(list)
    for row in memory:
        grouped[row.label].append(row.vector)

    out = {}
    for label, vectors in grouped.items():
        centroid = torch.stack(vectors, dim=0).mean(dim=0)
        out[label] = F.normalize(centroid, dim=0)
    return out


def classify_class_prototype(
    query: torch.Tensor,
    prototypes: Dict[str, torch.Tensor],
) -> MethodResult:
    scored = [
        (float(torch.dot(query, vector).item()), label)
        for label, vector in prototypes.items()
    ]
    scored.sort(reverse=True)
    return MethodResult(label=scored[0][1], similarity=scored[0][0])


def classify_knn(
    query: torch.Tensor,
    memory: Sequence[MemoryVector],
    k: int,
) -> MethodResult:
    neighbors = sorted(
        [
            (float(torch.dot(query, row.vector).item()), row.label)
            for row in memory
        ],
        reverse=True,
    )[: max(1, min(k, len(memory)))]

    votes: dict[str, float] = defaultdict(float)
    best_sim: dict[str, float] = defaultdict(lambda: -1.0)
    for similarity, label in neighbors:
        votes[label] += max(0.0, similarity)
        best_sim[label] = max(best_sim[label], similarity)

    ranked_labels = sorted(
        votes,
        key=lambda label: (votes[label], best_sim[label]),
        reverse=True,
    )
    label = ranked_labels[0]

    # Gate similarity remains an interpretable cosine score: strongest
    # neighbor supporting the winning label.
    return MethodResult(label=label, similarity=best_sim[label])


def spherical_kmeans(
    vectors: Sequence[torch.Tensor],
    k: int,
    iterations: int = 20,
) -> List[torch.Tensor]:
    if not vectors:
        return []
    if len(vectors) <= k:
        return [F.normalize(v.clone(), dim=0) for v in vectors]

    # Deterministic farthest-point initialization.
    centers = [vectors[0].clone()]
    while len(centers) < k:
        best_index = None
        best_distance = -1.0
        for i, vector in enumerate(vectors):
            nearest_sim = max(float(torch.dot(vector, c).item()) for c in centers)
            distance = 1.0 - nearest_sim
            if distance > best_distance:
                best_distance = distance
                best_index = i
        centers.append(vectors[best_index].clone())

    centers = [F.normalize(c, dim=0) for c in centers]

    for _ in range(iterations):
        groups: list[list[torch.Tensor]] = [[] for _ in range(k)]
        for vector in vectors:
            sims = [float(torch.dot(vector, c).item()) for c in centers]
            index = max(range(k), key=lambda i: sims[i])
            groups[index].append(vector)

        new_centers = []
        for i, group in enumerate(groups):
            if not group:
                new_centers.append(centers[i])
                continue
            centroid = torch.stack(group, dim=0).mean(dim=0)
            new_centers.append(F.normalize(centroid, dim=0))

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
    per_label: int,
) -> Dict[str, List[torch.Tensor]]:
    grouped: dict[str, list[torch.Tensor]] = defaultdict(list)
    for row in memory:
        grouped[row.label].append(row.vector)

    return {
        label: spherical_kmeans(
            vectors,
            max(1, min(per_label, len(vectors))),
        )
        for label, vectors in grouped.items()
    }


def classify_multi_prototype(
    query: torch.Tensor,
    prototypes: Dict[str, List[torch.Tensor]],
) -> MethodResult:
    scored = []
    for label, centers in prototypes.items():
        similarity = max(float(torch.dot(query, c).item()) for c in centers)
        scored.append((similarity, label))
    scored.sort(reverse=True)
    return MethodResult(label=scored[0][1], similarity=scored[0][0])


def evaluate_method(details, method, sim_th):
    tp = fp = tn = fn = wrong_label_accept = 0
    accepted = correct_accept = 0

    for row in details:
        pred = row[method]
        gate_accept = pred["similarity"] >= sim_th and pred["agreement"]

        if gate_accept:
            accepted += 1

        correct_label = (
            row["expected_label"] != "REJECT"
            and pred["label"] == row["expected_label"]
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

    target_count = sum(int(row["should_accept"]) for row in details)
    reject_count = len(details) - target_count

    precision = tp / max(1, tp + fp + wrong_label_accept)
    recall = tp / max(1, target_count)
    specificity = tn / max(1, reject_count)
    balanced = (recall + specificity) / 2.0
    coverage = accepted / max(1, len(details))
    accepted_accuracy = correct_accept / max(1, accepted)
    false_accept_rate = (fp + wrong_label_accept) / max(1, len(details))

    return {
        "method": method,
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
    memory_rows = load_semantic_memory(Path(args.memory))
    if len({row.label for row in memory_rows}) < 2:
        raise ValueError("At least two semantic-memory labels are required.")

    router = SemanticRouter(model, tokenizer, alpha=args.alpha)
    router.fit(merge_samples(base, memory_rows))

    memory = encode_memory(router, memory_rows)
    one_proto = class_prototypes(memory)
    multi_proto = build_multi_prototypes(memory, args.prototypes_per_label)
    cases = load_cases(args.cases)

    details = []
    for case in cases:
        query = F.normalize(router._encode_tensor(case["text"]), dim=0)
        base_top1 = router.route(case["text"])[0]

        results = {
            "class_prototype": classify_class_prototype(query, one_proto),
            "knn": classify_knn(query, memory, args.knn_k),
            "multi_prototype": classify_multi_prototype(query, multi_proto),
        }

        row = {
            "expected_label": case["expected_label"],
            "should_accept": case["should_accept"],
            "text": case["text"],
            "base_label": base_top1.label,
            "base_similarity": base_top1.similarity,
        }

        for method, result in results.items():
            row[method] = {
                "label": result.label,
                "similarity": result.similarity,
                "agreement": result.label == base_top1.label,
            }

        details.append(row)

    flat_details = []
    for row in details:
        flat = {
            "expected_label": row["expected_label"],
            "should_accept": row["should_accept"],
            "text": row["text"],
            "base_label": row["base_label"],
            "base_similarity": row["base_similarity"],
        }
        for method in ("class_prototype", "knn", "multi_prototype"):
            flat[f"{method}_label"] = row[method]["label"]
            flat[f"{method}_similarity"] = row[method]["similarity"]
            flat[f"{method}_agreement"] = row[method]["agreement"]
        flat_details.append(flat)

    with Path(args.detail_csv).open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat_details[0].keys()))
        writer.writeheader()
        writer.writerows(flat_details)

    summary = [
        evaluate_method(details, method, args.sim_th)
        for method in ("class_prototype", "knn", "multi_prototype")
    ]

    with Path(args.summary_csv).open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)

    print()
    print("=" * 92)
    print(" LLM_SEM v0.3.8 Class Prototype vs k-NN vs Multi-Prototype")
    print("=" * 92)
    print("Device                :", device)
    if device.type == "cuda":
        print("GPU                   :", torch.cuda.get_device_name(0))
    print("Checkpoint loss       :", checkpoint.get("loss"))
    print("Memory samples        :", len(memory_rows))
    print("Memory labels         :", ", ".join(sorted(one_proto)))
    print("Evaluation cases      :", len(cases))
    print("Similarity threshold  :", f"{args.sim_th:.2f}")
    print("Base agreement        : REQUIRED")
    print("k-NN k                :", args.knn_k)
    print("Multi-proto / label   :", args.prototypes_per_label)
    print()

    print("Per-query predictions")
    print("-" * 92)
    print(
        f"{'Expected':<9} {'Base':<9} "
        f"{'ClassProto':<17} {'k-NN':<17} {'MultiProto':<17} Text"
    )
    for row in flat_details:
        cp = f"{row['class_prototype_label']}:{row['class_prototype_similarity']:.3f}"
        kn = f"{row['knn_label']}:{row['knn_similarity']:.3f}"
        mp = f"{row['multi_prototype_label']}:{row['multi_prototype_similarity']:.3f}"
        print(
            f"{row['expected_label']:<9} {row['base_label']:<9} "
            f"{cp:<17} {kn:<17} {mp:<17} {row['text']}"
        )

    print()
    print("Method comparison")
    print("-" * 92)
    print(
        f"{'Method':<18} {'Prec':>7} {'Recall':>7} {'Spec':>7} "
        f"{'BalAcc':>7} {'Cover':>7} {'AccptAcc':>8} {'False':>7}"
    )
    for row in summary:
        print(
            f"{row['method']:<18} "
            f"{row['precision']*100:>6.1f}% "
            f"{row['recall']*100:>6.1f}% "
            f"{row['specificity']*100:>6.1f}% "
            f"{row['balanced_accuracy']*100:>6.1f}% "
            f"{row['coverage']*100:>6.1f}% "
            f"{row['accepted_accuracy']*100:>7.1f}% "
            f"{row['false_accept_rate']*100:>6.1f}%"
        )

    print()
    print("Summary CSV:", args.summary_csv)
    print("Detail CSV :", args.detail_csv)


if __name__ == "__main__":
    main()
