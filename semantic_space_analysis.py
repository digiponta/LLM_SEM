# semantic_space_analysis.py
#
# Semantic-space diagnostics for LLM_SEM.
#
# Measures:
# - class centroid distance matrix
# - within-class radius / dispersion
# - nearest competing centroid
# - centroid separation normalized by class radii
# - leave-one-out misclassification destinations
# - overlap risk between semantic classes

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

import torch
import torch.nn.functional as F

from model import LanguageModel
from semantic_eval import LabeledSentence, load_benchmark
from semantic_radius import (
    DEFAULT_RADIUS_QUANTILE,
    DEFAULT_RADIUS_SCALE,
    fit_class_radii,
)
from semantic_router import (
    DEFAULT_ALPHA,
    DEFAULT_BENCHMARK,
    DEFAULT_MODEL,
    DEFAULT_TOKENIZER,
    SemanticRouter,
    _centroids_from_vectors,
    _rank_against_centroids,
)
from tokenizer import Tokenizer


DEFAULT_SUMMARY_CSV = "semantic_space_summary.csv"
DEFAULT_MATRIX_CSV = "semantic_centroid_distance_matrix.csv"
DEFAULT_ERRORS_CSV = "semantic_space_errors.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze LLM_SEM semantic-space class geometry."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--benchmark", default=DEFAULT_BENCHMARK)
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
    parser.add_argument("--summary-csv", default=DEFAULT_SUMMARY_CSV)
    parser.add_argument("--matrix-csv", default=DEFAULT_MATRIX_CSV)
    parser.add_argument("--errors-csv", default=DEFAULT_ERRORS_CSV)
    return parser.parse_args()


def cosine_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    return 1.0 - float(F.cosine_similarity(a, b, dim=0).item())


def analyze(
    router: SemanticRouter,
    samples: Sequence[LabeledSentence],
    radius_quantile: float,
    radius_scale: float,
):
    vectors = [router._encode_tensor(sample.text) for sample in samples]
    centroids = _centroids_from_vectors(samples, vectors)
    labels = sorted(centroids)

    radii = fit_class_radii(
        router,
        samples,
        quantile=radius_quantile,
        scale=radius_scale,
    )

    matrix: Dict[str, Dict[str, float]] = {}
    for left in labels:
        matrix[left] = {}
        for right in labels:
            matrix[left][right] = cosine_distance(
                centroids[left],
                centroids[right],
            )

    misroutes: Dict[str, Counter] = {
        label: Counter() for label in labels
    }
    correct_count = Counter()
    total_count = Counter()
    error_rows: List[Dict[str, object]] = []

    for index, sample in enumerate(samples):
        train_samples = [
            other for j, other in enumerate(samples) if j != index
        ]
        train_vectors = [
            vector for j, vector in enumerate(vectors) if j != index
        ]
        loo_centroids = _centroids_from_vectors(
            train_samples,
            train_vectors,
        )
        ranked = _rank_against_centroids(
            vectors[index],
            loo_centroids,
        )
        top1 = ranked[0]
        total_count[sample.label] += 1
        if top1.label == sample.label:
            correct_count[sample.label] += 1
        else:
            misroutes[sample.label][top1.label] += 1
            error_rows.append(
                {
                    "expected": sample.label,
                    "predicted": top1.label,
                    "top1_similarity": top1.similarity,
                    "top1_distance": top1.distance,
                    "text": sample.text,
                }
            )

    summary_rows: List[Dict[str, object]] = []

    for label in labels:
        competitors = [
            other for other in labels if other != label
        ]
        nearest = min(
            competitors,
            key=lambda other: matrix[label][other],
        )
        nearest_distance = matrix[label][nearest]
        own_radius = radii[label].radius
        nearest_radius = radii[nearest].radius

        radius_sum = own_radius + nearest_radius
        separation_ratio = (
            nearest_distance / radius_sum
            if radius_sum > 0.0
            else float("inf")
        )
        overlap_gap = nearest_distance - radius_sum
        overlap_risk = separation_ratio < 1.0

        errors = misroutes[label]
        most_common_error = ""
        most_common_error_count = 0
        if errors:
            most_common_error, most_common_error_count = (
                errors.most_common(1)[0]
            )

        total = total_count[label]
        correct = correct_count[label]
        loo_accuracy = correct / total if total else 0.0

        summary_rows.append(
            {
                "label": label,
                "sample_count": total,
                "loo_accuracy": loo_accuracy,
                "radius": own_radius,
                "mean_within_distance": radii[label].mean_distance,
                "max_within_distance": radii[label].max_distance,
                "nearest_class": nearest,
                "nearest_centroid_distance": nearest_distance,
                "nearest_class_radius": nearest_radius,
                "radius_sum": radius_sum,
                "separation_ratio": separation_ratio,
                "overlap_gap": overlap_gap,
                "overlap_risk": overlap_risk,
                "most_common_error": most_common_error,
                "most_common_error_count": most_common_error_count,
            }
        )

    return labels, matrix, summary_rows, error_rows


def write_outputs(
    labels,
    matrix,
    summary_rows,
    error_rows,
    summary_csv,
    matrix_csv,
    errors_csv,
) -> None:
    with Path(summary_csv).open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(summary_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    with Path(matrix_csv).open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        writer = csv.writer(f)
        writer.writerow(["label", *labels])
        for label in labels:
            writer.writerow(
                [label] + [
                    matrix[label][other]
                    for other in labels
                ]
            )

    with Path(errors_csv).open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        fieldnames = [
            "expected",
            "predicted",
            "top1_similarity",
            "top1_distance",
            "text",
        ]
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(error_rows)


def print_report(labels, matrix, summary_rows, error_rows) -> None:
    print()
    print("============================================================")
    print(" LLM_SEM Semantic Space Analysis")
    print("============================================================")
    print()

    print("Class geometry")
    print("--------------")
    print(
        f"{'Class':<12} {'Acc':>8} {'Radius':>10} "
        f"{'Nearest':<12} {'C-Dist':>10} "
        f"{'SepRatio':>10} {'Overlap':>8}"
    )
    for row in summary_rows:
        print(
            f"{row['label']:<12} "
            f"{row['loo_accuracy'] * 100:>7.2f}% "
            f"{row['radius']:>10.6f} "
            f"{row['nearest_class']:<12} "
            f"{row['nearest_centroid_distance']:>10.6f} "
            f"{row['separation_ratio']:>10.3f} "
            f"{str(row['overlap_risk']):>8}"
        )

    print()
    print("Centroid distance matrix")
    print("------------------------")
    header = "class".ljust(12) + "".join(
        f"{label[:9]:>11}" for label in labels
    )
    print(header)
    for label in labels:
        line = label.ljust(12)
        for other in labels:
            line += f"{matrix[label][other]:>11.6f}"
        print(line)

    print()
    print("Primary confusion destinations")
    print("------------------------------")
    for row in summary_rows:
        if row["most_common_error"]:
            print(
                f"{row['label']:<12} -> "
                f"{row['most_common_error']:<12} "
                f"{row['most_common_error_count']} sample(s)"
            )
        else:
            print(f"{row['label']:<12} -> none")

    print()
    print("Interpretation")
    print("--------------")
    print(
        "separation_ratio < 1.0 means the two class-radius regions "
        "overlap geometrically."
    )
    print(
        "Smaller separation ratios and lower leave-one-out accuracy "
        "identify classes that should be improved first."
    )
    print()
    print("Misclassified samples:", len(error_rows))


def main() -> None:
    args = parse_args()

    for filename, label in (
        (args.model, "Model checkpoint"),
        (args.tokenizer, "Tokenizer"),
        (args.benchmark, "Benchmark"),
    ):
        if not Path(filename).exists():
            raise FileNotFoundError(f"{label} not found: {filename}")

    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("alpha must be between 0.0 and 1.0.")
    if not 0.0 <= args.radius_quantile <= 1.0:
        raise ValueError(
            "radius-quantile must be between 0.0 and 1.0."
        )
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
    samples = load_benchmark(args.benchmark)

    print()
    print("LLM_SEM Semantic Space Analyzer")
    print("-------------------------------")
    print("Device          :", device)
    if device.type == "cuda":
        print("GPU             :", torch.cuda.get_device_name(0))
    print("Model           :", args.model)
    print("Benchmark       :", args.benchmark)
    print("Samples         :", len(samples))
    print("Pooling         : raw hybrid")
    print("Alpha           :", args.alpha)
    print("Radius quantile :", args.radius_quantile)
    print("Radius scale    :", args.radius_scale)
    print("Checkpoint loss :", checkpoint.get("loss"))

    router = SemanticRouter(
        model=model,
        tokenizer=tokenizer,
        alpha=args.alpha,
    )

    labels, matrix, summary_rows, error_rows = analyze(
        router=router,
        samples=samples,
        radius_quantile=args.radius_quantile,
        radius_scale=args.radius_scale,
    )

    write_outputs(
        labels=labels,
        matrix=matrix,
        summary_rows=summary_rows,
        error_rows=error_rows,
        summary_csv=args.summary_csv,
        matrix_csv=args.matrix_csv,
        errors_csv=args.errors_csv,
    )

    print_report(
        labels=labels,
        matrix=matrix,
        summary_rows=summary_rows,
        error_rows=error_rows,
    )

    print()
    print("Summary CSV :", args.summary_csv)
    print("Matrix CSV  :", args.matrix_csv)
    print("Errors CSV  :", args.errors_csv)


if __name__ == "__main__":
    main()
