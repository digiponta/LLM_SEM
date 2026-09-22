# semantic_radius_sweep.py
#
# Radius-scale sweep for LLM_SEM v0.2.
#
# Uses projected development centroids/radii, independent known holdout,
# and unknown stress samples. Sweeps radius_scale without retraining either
# the base LLM or semantic projection head.

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import torch

from model import LanguageModel
from semantic_eval import load_benchmark
from semantic_projection import DEFAULT_PROJECTION, SemanticProjectionHead
from semantic_projection_eval import centroids, class_radii, cosine_distance
from semantic_radius import DEFAULT_RADIUS_QUANTILE
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
DEFAULT_OUTPUT = "semantic_radius_sweep.csv"
DEFAULT_SCALES = "0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep class-radius scale for LLM_SEM v0.2."
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
    parser.add_argument("--scales", default=DEFAULT_SCALES)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    return parser.parse_args()


def parse_scales(value: str) -> List[float]:
    scales = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        scale = float(item)
        if scale <= 0.0:
            raise ValueError("All radius scales must be > 0.")
        scales.append(scale)
    if not scales:
        raise ValueError("At least one radius scale is required.")
    return scales


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


def evaluate_scale(
    scale: float,
    development,
    development_vectors,
    known_holdout,
    known_vectors,
    unknown_stress,
    unknown_vectors,
    quantile: float,
) -> Dict[str, float]:
    centers = centroids(development, development_vectors)
    radii = class_radii(
        development,
        development_vectors,
        quantile,
        scale,
    )

    known_total = len(known_holdout)
    unknown_total = len(unknown_stress)

    known_base_correct = 0
    known_routed_correct = 0
    known_accepted = 0
    false_unknown = 0

    known_ratios = []
    for sample, vector in zip(known_holdout, known_vectors):
        label, distance = nearest(vector, centers)
        radius = radii[label]
        ratio = distance / radius if radius > 0.0 else float("inf")
        rejected = distance > radius
        base_correct = label == sample.label

        known_base_correct += int(base_correct)
        known_accepted += int(not rejected)
        false_unknown += int(rejected)
        known_routed_correct += int((not rejected) and base_correct)
        known_ratios.append(ratio)

    unknown_detected = 0
    unknown_ratios = []
    for _sample, vector in zip(unknown_stress, unknown_vectors):
        label, distance = nearest(vector, centers)
        radius = radii[label]
        ratio = distance / radius if radius > 0.0 else float("inf")
        detected = distance > radius

        unknown_detected += int(detected)
        unknown_ratios.append(ratio)

    base_known_accuracy = known_base_correct / known_total
    known_recall = known_routed_correct / known_total
    known_accept_rate = known_accepted / known_total
    false_unknown_rate = false_unknown / known_total
    unknown_detection_rate = unknown_detected / unknown_total
    false_known_rate = 1.0 - unknown_detection_rate
    balanced_accuracy = (known_recall + unknown_detection_rate) / 2.0

    max_known_ratio = max(known_ratios)
    min_unknown_ratio = min(unknown_ratios)
    separation_gap = min_unknown_ratio - max_known_ratio

    return {
        "radius_scale": scale,
        "known_base_accuracy": base_known_accuracy,
        "known_recall": known_recall,
        "known_accept_rate": known_accept_rate,
        "false_unknown_rate": false_unknown_rate,
        "unknown_detection_rate": unknown_detection_rate,
        "false_known_rate": false_known_rate,
        "balanced_accuracy": balanced_accuracy,
        "max_known_ratio": max_known_ratio,
        "min_unknown_ratio": min_unknown_ratio,
        "separation_gap": separation_gap,
    }


def main() -> None:
    args = parse_args()
    scales = parse_scales(args.scales)

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
    development_raw = [
        router._encode_tensor(sample.text)
        for sample in development
    ]
    development_projected = project_batch(
        head, development_raw, device
    )

    print("Encoding known holdout semantic vectors...")
    known_raw = [
        router._encode_tensor(sample.text)
        for sample in known_holdout
    ]
    known_projected = project_batch(
        head, known_raw, device
    )

    print("Encoding unknown stress semantic vectors...")
    unknown_raw = [
        router._encode_tensor(sample.text)
        for sample in unknown_stress
    ]
    unknown_projected = project_batch(
        head, unknown_raw, device
    )

    rows = [
        evaluate_scale(
            scale,
            development,
            development_projected,
            known_holdout,
            known_projected,
            unknown_stress,
            unknown_projected,
            args.radius_quantile,
        )
        for scale in scales
    ]

    with Path(args.output).open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    best = max(
        rows,
        key=lambda row: (
            row["balanced_accuracy"],
            row["known_recall"],
            row["unknown_detection_rate"],
            -row["false_unknown_rate"],
        ),
    )

    print()
    print("============================================================")
    print(" LLM_SEM v0.2 Radius Scale Sweep")
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
    print("Radius quantile        :", args.radius_quantile)
    print("Scales                 :", ", ".join(f"{v:.2f}" for v in scales))
    print()

    print(
        f"{'Scale':>6} {'Known':>8} {'F-Unk':>8} "
        f"{'Unknown':>8} {'F-Known':>8} {'BalAcc':>8} {'Gap':>8}"
    )
    print("-" * 62)
    for row in rows:
        print(
            f"{row['radius_scale']:>6.2f} "
            f"{row['known_recall'] * 100.0:>7.2f}% "
            f"{row['false_unknown_rate'] * 100.0:>7.2f}% "
            f"{row['unknown_detection_rate'] * 100.0:>7.2f}% "
            f"{row['false_known_rate'] * 100.0:>7.2f}% "
            f"{row['balanced_accuracy'] * 100.0:>7.2f}% "
            f"{row['separation_gap']:>8.3f}"
        )

    print()
    print("Best candidate")
    print("--------------")
    print("Radius scale           :", f"{best['radius_scale']:.2f}")
    print("Known Recall           :", f"{best['known_recall'] * 100.0:.2f}%")
    print("False Unknown Rate     :", f"{best['false_unknown_rate'] * 100.0:.2f}%")
    print(
        "Unknown Detection Rate :",
        f"{best['unknown_detection_rate'] * 100.0:.2f}%",
    )
    print(
        "Balanced Accuracy      :",
        f"{best['balanced_accuracy'] * 100.0:.2f}%",
    )
    print("Separation gap         :", f"{best['separation_gap']:.3f}")
    print()
    print("Results CSV saved:", args.output)


if __name__ == "__main__":
    main()
