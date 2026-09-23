# semantic_projection_eval.py
#
# Compare the original semantic space with the learned projection space.
# Evaluation includes development leave-one-out routing and independent
# holdout open-set validation with class-specific semantic radii.

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F

from model import LanguageModel
from semantic_eval import LabeledSentence, load_benchmark
from semantic_projection import (
    DEFAULT_PROJECTION,
    SemanticProjectionHead,
)
from semantic_radius import (
    DEFAULT_RADIUS_QUANTILE,
    DEFAULT_RADIUS_SCALE,
)
from semantic_router import (
    DEFAULT_ALPHA,
    DEFAULT_BENCHMARK,
    DEFAULT_MODEL,
    DEFAULT_TOKENIZER,
    SemanticRouter,
)
from tokenizer import Tokenizer


DEFAULT_KNOWN_HOLDOUT = "holdout_benchmark.csv"
DEFAULT_UNKNOWN_HOLDOUT = "unknown_holdout.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the learned LLM_SEM semantic projection."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--benchmark", default=DEFAULT_BENCHMARK)
    parser.add_argument("--known-holdout", default=DEFAULT_KNOWN_HOLDOUT)
    parser.add_argument("--unknown-holdout", default=DEFAULT_UNKNOWN_HOLDOUT)
    parser.add_argument("--projection", default=DEFAULT_PROJECTION)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument(
        "--radius-quantile",
        type=float,
        default=DEFAULT_RADIUS_QUANTILE,
    )
    parser.add_argument(
        "--radius-scale",
        type=float,
        default=DEFAULT_RADIUS_SCALE,
    )
    return parser.parse_args()


def cosine_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    return 1.0 - float(F.cosine_similarity(a, b, dim=0).item())


def centroids(
    samples: Sequence[LabeledSentence],
    vectors: Sequence[torch.Tensor],
) -> Dict[str, torch.Tensor]:
    grouped: Dict[str, List[torch.Tensor]] = defaultdict(list)
    for sample, vector in zip(samples, vectors):
        grouped[sample.label].append(vector)
    return {
        label: torch.stack(items).mean(dim=0)
        for label, items in grouped.items()
    }


def nearest(
    vector: torch.Tensor,
    class_centroids: Dict[str, torch.Tensor],
) -> Tuple[str, float]:
    ranked = sorted(
        (
            (label, cosine_distance(vector, center))
            for label, center in class_centroids.items()
        ),
        key=lambda item: item[1],
    )
    return ranked[0]


def loo_accuracy(
    samples: Sequence[LabeledSentence],
    vectors: Sequence[torch.Tensor],
) -> float:
    correct = 0
    for index, sample in enumerate(samples):
        train_samples = [
            other for j, other in enumerate(samples) if j != index
        ]
        train_vectors = [
            vector for j, vector in enumerate(vectors) if j != index
        ]
        label, _ = nearest(
            vectors[index],
            centroids(train_samples, train_vectors),
        )
        correct += int(label == sample.label)
    return correct / len(samples)


def pairwise_margin(
    samples: Sequence[LabeledSentence],
    vectors: Sequence[torch.Tensor],
) -> Tuple[float, float, float]:
    within: List[float] = []
    between: List[float] = []
    for i in range(len(samples)):
        for j in range(i + 1, len(samples)):
            similarity = float(
                F.cosine_similarity(
                    vectors[i],
                    vectors[j],
                    dim=0,
                ).item()
            )
            if samples[i].label == samples[j].label:
                within.append(similarity)
            else:
                between.append(similarity)

    within_mean = sum(within) / len(within)
    between_mean = sum(between) / len(between)
    return within_mean, between_mean, within_mean - between_mean


def class_radii(
    samples: Sequence[LabeledSentence],
    vectors: Sequence[torch.Tensor],
    quantile: float,
    scale: float,
) -> Dict[str, float]:
    grouped: Dict[str, List[float]] = defaultdict(list)

    for index, sample in enumerate(samples):
        train_samples = [
            other for j, other in enumerate(samples) if j != index
        ]
        train_vectors = [
            vector for j, vector in enumerate(vectors) if j != index
        ]
        own_centroid = centroids(
            train_samples,
            train_vectors,
        )[sample.label]
        grouped[sample.label].append(
            cosine_distance(vectors[index], own_centroid)
        )

    result: Dict[str, float] = {}
    for label, distances in grouped.items():
        ordered = sorted(distances)
        q_index = int(round(quantile * (len(ordered) - 1)))
        result[label] = ordered[q_index] * scale
    return result


def mean_nearest_centroid_distance(
    class_centroids: Dict[str, torch.Tensor],
) -> float:
    values: List[float] = []
    labels = sorted(class_centroids)
    for label in labels:
        values.append(
            min(
                cosine_distance(
                    class_centroids[label],
                    class_centroids[other],
                )
                for other in labels
                if other != label
            )
        )
    return sum(values) / len(values)


def holdout_metrics(
    development_samples: Sequence[LabeledSentence],
    development_vectors: Sequence[torch.Tensor],
    known_samples: Sequence[LabeledSentence],
    known_vectors: Sequence[torch.Tensor],
    unknown_vectors: Sequence[torch.Tensor],
    quantile: float,
    scale: float,
) -> Dict[str, float]:
    centers = centroids(development_samples, development_vectors)
    radii = class_radii(
        development_samples,
        development_vectors,
        quantile,
        scale,
    )

    known_base_correct = 0
    known_routed_correct = 0
    known_accepted = 0
    false_unknown = 0

    for sample, vector in zip(known_samples, known_vectors):
        label, distance = nearest(vector, centers)
        base_correct = label == sample.label
        known_base_correct += int(base_correct)

        rejected = distance > radii[label]
        if rejected:
            false_unknown += 1
        else:
            known_accepted += 1
            known_routed_correct += int(base_correct)

    unknown_detected = 0
    for vector in unknown_vectors:
        label, distance = nearest(vector, centers)
        unknown_detected += int(distance > radii[label])

    known_total = len(known_samples)
    unknown_total = len(unknown_vectors)

    known_accuracy = known_base_correct / known_total
    known_recall = known_routed_correct / known_total
    known_accept_rate = known_accepted / known_total
    false_unknown_rate = false_unknown / known_total
    unknown_detection = unknown_detected / unknown_total
    false_known_rate = 1.0 - unknown_detection
    balanced_accuracy = (known_recall + unknown_detection) / 2.0

    return {
        "known_accuracy": known_accuracy,
        "known_recall": known_recall,
        "known_accept_rate": known_accept_rate,
        "false_unknown_rate": false_unknown_rate,
        "unknown_detection": unknown_detection,
        "false_known_rate": false_known_rate,
        "balanced_accuracy": balanced_accuracy,
    }



def grouped_known_metrics(
    development_samples: Sequence[LabeledSentence],
    development_vectors: Sequence[torch.Tensor],
    known_samples: Sequence[LabeledSentence],
    known_vectors: Sequence[torch.Tensor],
    quantile: float,
    scale: float,
    attribute: str,
) -> Dict[str, Dict[str, float]]:
    centers = centroids(development_samples, development_vectors)
    radii = class_radii(
        development_samples,
        development_vectors,
        quantile,
        scale,
    )

    grouped: Dict[str, Dict[str, float]] = defaultdict(
        lambda: {
            "total": 0.0,
            "correct": 0.0,
            "accepted": 0.0,
            "routed_correct": 0.0,
        }
    )

    for sample, vector in zip(known_samples, known_vectors):
        group = str(getattr(sample, attribute, "") or "").strip()
        if not group:
            continue

        label, distance = nearest(vector, centers)
        correct = label == sample.label
        accepted = distance <= radii[label]

        grouped[group]["total"] += 1.0
        grouped[group]["correct"] += float(correct)
        grouped[group]["accepted"] += float(accepted)
        grouped[group]["routed_correct"] += float(correct and accepted)

    result: Dict[str, Dict[str, float]] = {}
    for group, values in grouped.items():
        total = values["total"]
        result[group] = {
            "samples": total,
            "accuracy": values["correct"] / total,
            "accept_rate": values["accepted"] / total,
            "recall": values["routed_correct"] / total,
        }
    return result


def print_grouped_known_report(
    title: str,
    before: Dict[str, Dict[str, float]],
    after: Dict[str, Dict[str, float]],
) -> None:
    groups = sorted(set(before) | set(after))
    if not groups:
        return

    print(title)
    print("=" * len(title))
    print()
    print(
        f"{'Group':<18} {'N':>4} "
        f"{'Acc B':>8} {'Acc A':>8} "
        f"{'Accept B':>9} {'Accept A':>9} "
        f"{'Recall B':>9} {'Recall A':>9}"
    )
    print("-" * 83)

    for group in groups:
        b = before.get(group)
        a = after.get(group)
        n = int((a or b or {}).get("samples", 0.0))
        print(
            f"{group:<18} {n:>4d} "
            f"{(b or {}).get('accuracy', 0.0) * 100:>7.2f}% "
            f"{(a or {}).get('accuracy', 0.0) * 100:>7.2f}% "
            f"{(b or {}).get('accept_rate', 0.0) * 100:>8.2f}% "
            f"{(a or {}).get('accept_rate', 0.0) * 100:>8.2f}% "
            f"{(b or {}).get('recall', 0.0) * 100:>8.2f}% "
            f"{(a or {}).get('recall', 0.0) * 100:>8.2f}%"
        )
    print()


def project_all(
    head: SemanticProjectionHead,
    vectors: Sequence[torch.Tensor],
    device: torch.device,
) -> List[torch.Tensor]:
    batch = torch.stack(vectors).to(device)
    with torch.no_grad():
        projected = head(batch, normalize=True).cpu()
    return [row for row in projected]


def print_space_report(name, samples, vectors) -> None:
    accuracy = loo_accuracy(samples, vectors)
    within, between, margin = pairwise_margin(samples, vectors)
    centers = centroids(samples, vectors)
    nearest_distance = mean_nearest_centroid_distance(centers)

    print(name)
    print("-" * len(name))
    print("LOO routing accuracy        :", f"{accuracy * 100.0:.2f}%")
    print("Within-class similarity     :", f"{within:.6f}")
    print("Between-class similarity    :", f"{between:.6f}")
    print("Semantic margin             :", f"{margin:.6f}")
    print("Mean nearest centroid dist. :", f"{nearest_distance:.6f}")
    print()


def print_holdout_report(name, metrics) -> None:
    print(name)
    print("-" * len(name))
    print(
        "Known routing accuracy :",
        f"{metrics['known_accuracy'] * 100.0:.2f}%",
    )
    print(
        "Known recall           :",
        f"{metrics['known_recall'] * 100.0:.2f}%",
    )
    print(
        "Known accept rate      :",
        f"{metrics['known_accept_rate'] * 100.0:.2f}%",
    )
    print(
        "Unknown detection rate :",
        f"{metrics['unknown_detection'] * 100.0:.2f}%",
    )
    print(
        "False Unknown rate     :",
        f"{metrics['false_unknown_rate'] * 100.0:.2f}%",
    )
    print(
        "False Known rate       :",
        f"{metrics['false_known_rate'] * 100.0:.2f}%",
    )
    print(
        "Balanced accuracy      :",
        f"{metrics['balanced_accuracy'] * 100.0:.2f}%",
    )
    print()


def main() -> None:
    args = parse_args()

    for filename, label in (
        (args.model, "Model checkpoint"),
        (args.tokenizer, "Tokenizer"),
        (args.benchmark, "Development benchmark"),
        (args.known_holdout, "Known holdout"),
        (args.unknown_holdout, "Unknown holdout"),
        (args.projection, "Projection checkpoint"),
    ):
        if not Path(filename).exists():
            raise FileNotFoundError(f"{label} not found: {filename}")

    if not 0.0 <= args.radius_quantile <= 1.0:
        raise ValueError("radius-quantile must be between 0.0 and 1.0.")
    if args.radius_scale <= 0.0:
        raise ValueError("radius-scale must be > 0.")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    tokenizer = Tokenizer.load(args.tokenizer)
    model, checkpoint = LanguageModel.load_checkpoint(
        args.model,
        device=device,
    )
    model.eval()

    head, projection_checkpoint = (
        SemanticProjectionHead.load_checkpoint(
            args.projection,
            device=device,
        )
    )

    development = load_benchmark(args.benchmark)
    known_holdout = load_benchmark(args.known_holdout)
    unknown_holdout = load_benchmark(args.unknown_holdout)

    router = SemanticRouter(
        model=model,
        tokenizer=tokenizer,
        alpha=args.alpha,
    )

    print()
    print("Encoding semantic vectors...")
    development_raw = [
        router._encode_tensor(sample.text)
        for sample in development
    ]
    known_raw = [
        router._encode_tensor(sample.text)
        for sample in known_holdout
    ]
    unknown_raw = [
        router._encode_tensor(sample.text)
        for sample in unknown_holdout
    ]

    development_projected = project_all(
        head, development_raw, device
    )
    known_projected = project_all(
        head, known_raw, device
    )
    unknown_projected = project_all(
        head, unknown_raw, device
    )

    print()
    print("============================================================")
    print(" LLM_SEM Semantic Projection Evaluation")
    print("============================================================")
    print()
    print("Device             :", device)
    if device.type == "cuda":
        print("GPU                :", torch.cuda.get_device_name(0))
    print("Base checkpoint loss:", checkpoint.get("loss"))
    print(
        "Projection best loss:",
        projection_checkpoint.get("best_loss"),
    )
    print("Development samples:", len(development))
    print("Known holdout      :", len(known_holdout))
    print("Unknown holdout    :", len(unknown_holdout))
    print("Radius quantile    :", args.radius_quantile)
    print("Radius scale       :", args.radius_scale)
    print()

    print("Development semantic space")
    print("==========================")
    print()
    print_space_report(
        "Before projection",
        development,
        development_raw,
    )
    print_space_report(
        "After projection",
        development,
        development_projected,
    )

    raw_holdout = holdout_metrics(
        development,
        development_raw,
        known_holdout,
        known_raw,
        unknown_raw,
        args.radius_quantile,
        args.radius_scale,
    )
    projected_holdout = holdout_metrics(
        development,
        development_projected,
        known_holdout,
        known_projected,
        unknown_projected,
        args.radius_quantile,
        args.radius_scale,
    )

    print("Independent holdout open-set validation")
    print("=======================================")
    print()
    print_holdout_report(
        "Before projection",
        raw_holdout,
    )
    print_holdout_report(
        "After projection",
        projected_holdout,
    )

    pattern_before = grouped_known_metrics(
        development,
        development_raw,
        known_holdout,
        known_raw,
        args.radius_quantile,
        args.radius_scale,
        "pattern",
    )
    pattern_after = grouped_known_metrics(
        development,
        development_projected,
        known_holdout,
        known_projected,
        args.radius_quantile,
        args.radius_scale,
        "pattern",
    )
    print_grouped_known_report(
        "Known holdout by utterance pattern",
        pattern_before,
        pattern_after,
    )

    difficulty_before = grouped_known_metrics(
        development,
        development_raw,
        known_holdout,
        known_raw,
        args.radius_quantile,
        args.radius_scale,
        "difficulty",
    )
    difficulty_after = grouped_known_metrics(
        development,
        development_projected,
        known_holdout,
        known_projected,
        args.radius_quantile,
        args.radius_scale,
        "difficulty",
    )
    print_grouped_known_report(
        "Known holdout by difficulty",
        difficulty_before,
        difficulty_after,
    )


if __name__ == "__main__":
    main()
