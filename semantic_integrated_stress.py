# semantic_integrated_stress.py
#
# Integrated Known + Unknown stress test for LLM_SEM v0.2.
#
# Uses:
#   - projected known development centroids/radii from my_benchmark.csv
#   - independent known samples from holdout_benchmark.csv
#   - out-of-domain stress samples from unknown_stress_benchmark.csv
#
# Reports Known Recall, False Unknown Rate, Unknown Detection Rate,
# Balanced Accuracy, and distance/radius separation statistics.

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Dict, List

import torch

from model import LanguageModel
from semantic_eval import load_benchmark
from semantic_projection import DEFAULT_PROJECTION, SemanticProjectionHead
from semantic_projection_eval import centroids, class_radii, cosine_distance
from semantic_radius import DEFAULT_RADIUS_QUANTILE, DEFAULT_RADIUS_SCALE
from semantic_router import (
    DEFAULT_ALPHA,
    DEFAULT_BENCHMARK,
    DEFAULT_MODEL,
    DEFAULT_TOKENIZER,
    SemanticRouter,
)
from tokenizer import Tokenizer


DEFAULT_KNOWN_HOLDOUT = "holdout_benchmark.csv"
DEFAULT_UNKNOWN_STRESS = "unknown_stress_benchmark.csv"
DEFAULT_OUTPUT = "semantic_integrated_stress_results.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Integrated Known + Unknown stress test for LLM_SEM v0.2."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--benchmark", default=DEFAULT_BENCHMARK)
    parser.add_argument("--known-holdout", default=DEFAULT_KNOWN_HOLDOUT)
    parser.add_argument("--unknown-stress", default=DEFAULT_UNKNOWN_STRESS)
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
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    return parser.parse_args()


@torch.no_grad()
def project_batch(
    head: SemanticProjectionHead,
    vectors: List[torch.Tensor],
    device: torch.device,
) -> List[torch.Tensor]:
    batch = torch.stack(vectors).to(device)
    projected = head(batch, normalize=True).cpu()
    return [row for row in projected]


def nearest(
    vector: torch.Tensor,
    centers: Dict[str, torch.Tensor],
):
    ranked = sorted(
        (
            (label, cosine_distance(vector, center))
            for label, center in centers.items()
        ),
        key=lambda item: item[1],
    )
    return ranked[0]


def summary(values: List[float]) -> Dict[str, float]:
    ordered = sorted(values)
    return {
        "mean": mean(ordered),
        "median": median(ordered),
        "min": ordered[0],
        "max": ordered[-1],
    }


def main() -> None:
    args = parse_args()

    for filename, label in (
        (args.model, "Base model checkpoint"),
        (args.tokenizer, "Tokenizer"),
        (args.benchmark, "Development benchmark"),
        (args.known_holdout, "Known holdout"),
        (args.unknown_stress, "Unknown stress benchmark"),
        (args.projection, "Projection checkpoint"),
    ):
        if not Path(filename).exists():
            raise FileNotFoundError(f"{label} not found: {filename}")

    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("alpha must be between 0.0 and 1.0.")
    if not 0.0 <= args.radius_quantile <= 1.0:
        raise ValueError("radius-quantile must be between 0.0 and 1.0.")
    if args.radius_scale <= 0.0:
        raise ValueError("radius-scale must be > 0.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = Tokenizer.load(args.tokenizer)
    model, base_checkpoint = LanguageModel.load_checkpoint(
        args.model,
        device=device,
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    head, projection_checkpoint = SemanticProjectionHead.load_checkpoint(
        args.projection,
        device=device,
    )

    development = load_benchmark(args.benchmark)
    known_holdout = load_benchmark(args.known_holdout)
    unknown_stress = load_benchmark(args.unknown_stress)

    router = SemanticRouter(
        model=model,
        tokenizer=tokenizer,
        alpha=args.alpha,
    )

    print()
    print("Encoding development semantic vectors...")
    dev_raw = [router._encode_tensor(s.text) for s in development]
    dev_projected = project_batch(head, dev_raw, device)

    centers = centroids(development, dev_projected)
    radii = class_radii(
        development,
        dev_projected,
        args.radius_quantile,
        args.radius_scale,
    )

    print("Encoding known holdout semantic vectors...")
    known_raw = [router._encode_tensor(s.text) for s in known_holdout]
    known_projected = project_batch(head, known_raw, device)

    print("Encoding unknown stress semantic vectors...")
    unknown_raw = [router._encode_tensor(s.text) for s in unknown_stress]
    unknown_projected = project_batch(head, unknown_raw, device)

    rows = []
    known_total = len(known_holdout)
    known_base_correct = 0
    known_routed_correct = 0
    false_unknown = 0

    known_ratios: List[float] = []
    unknown_ratios: List[float] = []
    known_category = defaultdict(lambda: {"total": 0, "correct": 0, "rejected": 0})
    unknown_category = defaultdict(lambda: {"total": 0, "detected": 0})

    for sample, vector in zip(known_holdout, known_projected):
        label, distance = nearest(vector, centers)
        radius = radii[label]
        ratio = distance / radius if radius > 0.0 else float("inf")
        rejected = distance > radius
        base_correct = label == sample.label

        known_base_correct += int(base_correct)
        false_unknown += int(rejected)
        known_routed_correct += int((not rejected) and base_correct)
        known_ratios.append(ratio)

        known_category[sample.label]["total"] += 1
        known_category[sample.label]["correct"] += int((not rejected) and base_correct)
        known_category[sample.label]["rejected"] += int(rejected)

        rows.append({
            "source": "known",
            "expected": sample.label,
            "nearest_known_class": label,
            "distance": distance,
            "class_radius": radius,
            "distance_radius_ratio": ratio,
            "unknown": rejected,
            "correct": (not rejected) and base_correct,
            "text": sample.text,
        })

    unknown_detected = 0
    for sample, vector in zip(unknown_stress, unknown_projected):
        label, distance = nearest(vector, centers)
        radius = radii[label]
        ratio = distance / radius if radius > 0.0 else float("inf")
        detected = distance > radius

        unknown_detected += int(detected)
        unknown_ratios.append(ratio)

        unknown_category[sample.label]["total"] += 1
        unknown_category[sample.label]["detected"] += int(detected)

        rows.append({
            "source": "unknown",
            "expected": sample.label,
            "nearest_known_class": label,
            "distance": distance,
            "class_radius": radius,
            "distance_radius_ratio": ratio,
            "unknown": detected,
            "correct": detected,
            "text": sample.text,
        })

    unknown_total = len(unknown_stress)

    base_known_accuracy = known_base_correct / known_total
    known_recall = known_routed_correct / known_total
    false_unknown_rate = false_unknown / known_total
    unknown_detection_rate = unknown_detected / unknown_total
    false_known_rate = 1.0 - unknown_detection_rate
    balanced_accuracy = (known_recall + unknown_detection_rate) / 2.0

    known_stats = summary(known_ratios)
    unknown_stats = summary(unknown_ratios)

    max_known_ratio = max(known_ratios)
    min_unknown_ratio = min(unknown_ratios)
    separation_gap = min_unknown_ratio - max_known_ratio

    with Path(args.output).open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print()
    print("============================================================")
    print(" LLM_SEM v0.2 Integrated Known + Unknown Stress Test")
    print("============================================================")
    print()
    print("Device                 :", device)
    if device.type == "cuda":
        print("GPU                    :", torch.cuda.get_device_name(0))
    print("Base model             :", args.model)
    print("Base checkpoint loss   :", base_checkpoint.get("loss"))
    print("Base LLM frozen        : True")
    print("Projection             :", args.projection)
    print("Projection lambda      :", projection_checkpoint.get("preservation_lambda"))
    print("Projection seed        :", projection_checkpoint.get("seed"))
    print("Development benchmark  :", args.benchmark)
    print("Known holdout          :", args.known_holdout)
    print("Unknown stress         :", args.unknown_stress)
    print()

    print("Classification metrics")
    print("----------------------")
    print("Known samples          :", known_total)
    print("Unknown samples        :", unknown_total)
    print("Known base accuracy    :", f"{base_known_accuracy * 100.0:.2f}%")
    print("Known Recall           :", f"{known_recall * 100.0:.2f}%")
    print("False Unknown Rate     :", f"{false_unknown_rate * 100.0:.2f}%")
    print("Unknown Detection Rate :", f"{unknown_detection_rate * 100.0:.2f}%")
    print("False Known Rate       :", f"{false_known_rate * 100.0:.2f}%")
    print("Balanced Accuracy      :", f"{balanced_accuracy * 100.0:.2f}%")
    print()

    print("Distance / Radius distributions")
    print("-------------------------------")
    print(
        "Known   mean={mean:.3f} median={median:.3f} "
        "min={min:.3f} max={max:.3f}".format(**known_stats)
    )
    print(
        "Unknown mean={mean:.3f} median={median:.3f} "
        "min={min:.3f} max={max:.3f}".format(**unknown_stats)
    )
    print("Max Known ratio        :", f"{max_known_ratio:.3f}")
    print("Min Unknown ratio      :", f"{min_unknown_ratio:.3f}")
    print("Separation gap         :", f"{separation_gap:.3f}")
    print(
        "Distribution overlap  :",
        "No" if separation_gap > 0.0 else "Yes",
    )
    print()

    print("Known category breakdown")
    print("------------------------")
    for label in sorted(known_category):
        stats = known_category[label]
        total = stats["total"]
        print(
            f"{label:<12} "
            f"correct={stats['correct']:>2}/{total:<2} "
            f"false-unknown={stats['rejected']:>2}/{total:<2}"
        )

    print()
    print("Unknown category breakdown")
    print("--------------------------")
    for label in sorted(unknown_category):
        stats = unknown_category[label]
        total = stats["total"]
        print(
            f"{label:<12} "
            f"detected={stats['detected']:>2}/{total:<2} "
            f"{stats['detected'] / total * 100.0:>6.2f}%"
        )

    print()
    print("Results CSV saved:", args.output)


if __name__ == "__main__":
    main()
