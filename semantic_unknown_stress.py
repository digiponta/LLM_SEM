# semantic_unknown_stress.py
#
# Unknown-category stress test for LLM_SEM v0.2.
#
# Evaluates out-of-domain inputs against projected known-class centroids and
# class-specific semantic radii. Reports Unknown Detection Rate, nearest known
# class, and distance/radius ratio.

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
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


DEFAULT_STRESS_BENCHMARK = "unknown_stress_benchmark.csv"
DEFAULT_OUTPUT = "semantic_unknown_stress_results.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an Unknown-category stress test for LLM_SEM v0.2."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--benchmark", default=DEFAULT_BENCHMARK)
    parser.add_argument("--stress-benchmark", default=DEFAULT_STRESS_BENCHMARK)
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


def main() -> None:
    args = parse_args()

    for filename, label in (
        (args.model, "Base model checkpoint"),
        (args.tokenizer, "Tokenizer"),
        (args.benchmark, "Known benchmark"),
        (args.stress_benchmark, "Unknown stress benchmark"),
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

    known_samples = load_benchmark(args.benchmark)
    stress_samples = load_benchmark(args.stress_benchmark)

    known_labels = {sample.label for sample in known_samples}
    overlap = sorted({
        sample.label for sample in stress_samples
        if sample.label in known_labels
    })
    if overlap:
        raise ValueError(
            "Stress benchmark labels must be outside known classes: "
            + ", ".join(overlap)
        )

    router = SemanticRouter(
        model=model,
        tokenizer=tokenizer,
        alpha=args.alpha,
    )

    print()
    print("Encoding known development semantic vectors...")
    known_raw = [
        router._encode_tensor(sample.text)
        for sample in known_samples
    ]
    known_projected = project_batch(head, known_raw, device)

    centers = centroids(known_samples, known_projected)
    radii = class_radii(
        known_samples,
        known_projected,
        args.radius_quantile,
        args.radius_scale,
    )

    print("Encoding unknown stress-test vectors...")
    stress_raw = [
        router._encode_tensor(sample.text)
        for sample in stress_samples
    ]
    stress_projected = project_batch(head, stress_raw, device)

    rows = []
    category_detected: Dict[str, List[bool]] = defaultdict(list)
    nearest_counts: Counter[str] = Counter()

    for sample, vector in zip(stress_samples, stress_projected):
        ranked = sorted(
            (
                (
                    label,
                    cosine_distance(vector, center),
                )
                for label, center in centers.items()
            ),
            key=lambda item: item[1],
        )

        nearest_label, distance = ranked[0]
        radius = radii[nearest_label]
        ratio = distance / radius if radius > 0.0 else float("inf")
        detected = distance > radius
        similarity = 1.0 - distance

        category_detected[sample.label].append(detected)
        nearest_counts[nearest_label] += 1

        rows.append({
            "source_category": sample.label,
            "text": sample.text,
            "nearest_known_class": nearest_label,
            "similarity": similarity,
            "distance": distance,
            "class_radius": radius,
            "distance_radius_ratio": ratio,
            "unknown_detected": detected,
        })

    with Path(args.output).open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    detected_total = sum(int(row["unknown_detected"]) for row in rows)
    detection_rate = detected_total / len(rows)
    ratios = [float(row["distance_radius_ratio"]) for row in rows]

    print()
    print("============================================================")
    print(" LLM_SEM v0.2 Unknown Stress Test")
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
    print("Known benchmark        :", args.benchmark)
    print("Stress benchmark       :", args.stress_benchmark)
    print("Known classes          :", ", ".join(sorted(centers)))
    print("Stress samples         :", len(rows))
    print("Unknown detected       :", f"{detected_total}/{len(rows)}")
    print("Unknown Detection Rate :", f"{detection_rate * 100.0:.2f}%")
    print("Mean distance/radius   :", f"{mean(ratios):.3f}")
    print("Min distance/radius    :", f"{min(ratios):.3f}")
    print()

    print("Category results")
    print("----------------")
    for category in sorted(category_detected):
        values = category_detected[category]
        detected = sum(int(v) for v in values)
        print(
            f"{category:<12} "
            f"{detected:>2}/{len(values):<2} "
            f"{detected / len(values) * 100.0:>6.2f}%"
        )

    print()
    print("Nearest known-class attraction")
    print("------------------------------")
    for label, count in nearest_counts.most_common():
        print(f"{label:<12} {count:>3}")

    print()
    print("Per-sample details")
    print("------------------")
    for row in rows:
        status = "Unknown" if row["unknown_detected"] else "MISSED"
        print(
            f"{row['source_category']:<10} "
            f"-> {row['nearest_known_class']:<10} "
            f"ratio={row['distance_radius_ratio']:.3f} "
            f"distance={row['distance']:.6f} "
            f"radius={row['class_radius']:.6f} "
            f"{status}"
        )

    print()
    print("Results CSV saved:", args.output)


if __name__ == "__main__":
    main()
