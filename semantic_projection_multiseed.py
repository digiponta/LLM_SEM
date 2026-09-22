# semantic_projection_multiseed.py
#
# Development-only multi-seed sweep for semantic projection preservation_lambda.
# Holdout data are intentionally not loaded.

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List

import torch

from model import LanguageModel
from semantic_eval import load_benchmark
from semantic_projection_sweep import (
    loo_accuracy,
    mean_geometry_drift,
    parse_lambdas,
    semantic_metrics,
    train_one,
)
from semantic_router import (
    DEFAULT_ALPHA,
    DEFAULT_BENCHMARK,
    DEFAULT_MODEL,
    DEFAULT_TOKENIZER,
    SemanticRouter,
)
from tokenizer import Tokenizer


DEFAULT_LAMBDAS = "0.5,1.0,2.0,5.0"
DEFAULT_SEEDS = "1,2,3,4,5"
DEFAULT_DETAIL_CSV = "semantic_projection_multiseed_detail.csv"
DEFAULT_SUMMARY_CSV = "semantic_projection_multiseed_summary.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Development-only multi-seed sweep for semantic projection."
        )
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--benchmark", default=DEFAULT_BENCHMARK)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--lambdas", default=DEFAULT_LAMBDAS)
    parser.add_argument("--seeds", default=DEFAULT_SEEDS)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--temperature", type=float, default=0.10)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--min-delta", type=float, default=1.0e-4)
    parser.add_argument("--detail-csv", default=DEFAULT_DETAIL_CSV)
    parser.add_argument("--summary-csv", default=DEFAULT_SUMMARY_CSV)
    return parser.parse_args()


def parse_seeds(value: str) -> List[int]:
    seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not seeds:
        raise ValueError("At least one seed is required.")
    return seeds


def main() -> None:
    args = parse_args()
    lambdas = parse_lambdas(args.lambdas)
    seeds = parse_seeds(args.seeds)

    for filename, label in (
        (args.model, "Model checkpoint"),
        (args.tokenizer, "Tokenizer"),
        (args.benchmark, "Benchmark"),
    ):
        if not Path(filename).exists():
            raise FileNotFoundError(f"{label} not found: {filename}")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    tokenizer = Tokenizer.load(args.tokenizer)
    model, checkpoint = LanguageModel.load_checkpoint(
        args.model,
        device=device,
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    samples = load_benchmark(args.benchmark)
    labels_sorted = sorted({sample.label for sample in samples})
    label_to_id = {
        label: index for index, label in enumerate(labels_sorted)
    }

    router = SemanticRouter(
        model=model,
        tokenizer=tokenizer,
        alpha=args.alpha,
    )

    print()
    print("Encoding frozen base semantic vectors...")
    base_vectors = torch.stack(
        [router._encode_tensor(sample.text) for sample in samples]
    ).to(device)
    labels = torch.tensor(
        [label_to_id[sample.label] for sample in samples],
        dtype=torch.long,
        device=device,
    )

    raw_vectors = [row.detach().cpu() for row in base_vectors]
    raw_accuracy = loo_accuracy(samples, raw_vectors)
    _, _, raw_margin = semantic_metrics(samples, raw_vectors)

    print()
    print("LLM_SEM Multi-Seed Preservation Sweep")
    print("-------------------------------------")
    print("Device          :", device)
    if device.type == "cuda":
        print("GPU             :", torch.cuda.get_device_name(0))
    print("Benchmark       :", args.benchmark)
    print("Samples         :", len(samples))
    print("Lambdas         :", ", ".join(str(v) for v in lambdas))
    print("Seeds           :", ", ".join(str(v) for v in seeds))
    print("Holdout used    : False")
    print("Raw LOO accuracy:", f"{raw_accuracy * 100.0:.2f}%")
    print("Raw margin      :", f"{raw_margin:.6f}")
    print()

    detail_rows = []

    for preservation_lambda in lambdas:
        for seed in seeds:
            head, training = train_one(
                base_vectors=base_vectors,
                labels=labels,
                preservation_lambda=preservation_lambda,
                hidden_dim=args.hidden_dim,
                epochs=args.epochs,
                lr=args.lr,
                temperature=args.temperature,
                weight_decay=args.weight_decay,
                dropout=args.dropout,
                patience=args.patience,
                min_delta=args.min_delta,
                seed=seed,
            )

            with torch.no_grad():
                projected_batch = head(
                    base_vectors,
                    normalize=True,
                ).cpu()

            projected_vectors = [row for row in projected_batch]
            accuracy = loo_accuracy(samples, projected_vectors)
            within, between, margin = semantic_metrics(
                samples,
                projected_vectors,
            )
            drift = mean_geometry_drift(
                base_vectors,
                projected_batch.to(device),
            )

            detail_rows.append(
                {
                    "preservation_lambda": preservation_lambda,
                    "seed": seed,
                    "best_epoch": int(training["best_epoch"]),
                    "best_total_loss": training["best_loss"],
                    "contrastive_loss": training["contrastive_loss"],
                    "preservation_loss": training["preservation_loss"],
                    "loo_accuracy": accuracy,
                    "within_similarity": within,
                    "between_similarity": between,
                    "semantic_margin": margin,
                    "geometry_drift": drift,
                }
            )

            print(
                f"lambda={preservation_lambda:<4g} "
                f"seed={seed:<3d} "
                f"LOO={accuracy * 100:>6.2f}% "
                f"margin={margin:>8.6f} "
                f"drift={drift:>8.6f}"
            )

    summary_rows = []

    for preservation_lambda in lambdas:
        rows = [
            row for row in detail_rows
            if row["preservation_lambda"] == preservation_lambda
        ]

        accuracies = [row["loo_accuracy"] for row in rows]
        margins = [row["semantic_margin"] for row in rows]
        drifts = [row["geometry_drift"] for row in rows]
        epochs = [row["best_epoch"] for row in rows]

        summary_rows.append(
            {
                "preservation_lambda": preservation_lambda,
                "seed_count": len(rows),
                "mean_loo_accuracy": mean(accuracies),
                "std_loo_accuracy": pstdev(accuracies),
                "mean_semantic_margin": mean(margins),
                "std_semantic_margin": pstdev(margins),
                "mean_geometry_drift": mean(drifts),
                "std_geometry_drift": pstdev(drifts),
                "mean_best_epoch": mean(epochs),
            }
        )

    max_mean_accuracy = max(
        row["mean_loo_accuracy"] for row in summary_rows
    )

    candidates = [
        row for row in summary_rows
        if abs(
            row["mean_loo_accuracy"] - max_mean_accuracy
        ) < 1.0e-12
    ]

    # Stability-aware development-only selection:
    # 1) highest mean LOO accuracy
    # 2) lowest LOO standard deviation
    # 3) require / prefer mean geometry drift <= 0.05
    # 4) highest semantic margin
    stable = [
        row for row in candidates
        if row["mean_geometry_drift"] <= 0.05
    ]
    selection_pool = stable if stable else candidates
    best = sorted(
        selection_pool,
        key=lambda row: (
            row["std_loo_accuracy"],
            -row["mean_semantic_margin"],
            row["mean_geometry_drift"],
        ),
    )[0]

    with Path(args.detail_csv).open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(detail_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(detail_rows)

    with Path(args.summary_csv).open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(summary_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print()
    print("Multi-seed summary")
    print("------------------")
    print(
        f"{'Lambda':>8} {'MeanLOO':>10} {'StdLOO':>10} "
        f"{'MeanMargin':>12} {'MeanDrift':>11}"
    )
    for row in summary_rows:
        print(
            f"{row['preservation_lambda']:>8g} "
            f"{row['mean_loo_accuracy'] * 100:>9.2f}% "
            f"{row['std_loo_accuracy'] * 100:>9.2f}% "
            f"{row['mean_semantic_margin']:>12.6f} "
            f"{row['mean_geometry_drift']:>11.6f}"
        )

    print()
    print("Development-only stable candidate")
    print("---------------------------------")
    print("Lambda            :", best["preservation_lambda"])
    print(
        "Mean LOO accuracy :",
        f"{best['mean_loo_accuracy'] * 100.0:.2f}%",
    )
    print(
        "Std LOO accuracy  :",
        f"{best['std_loo_accuracy'] * 100.0:.2f}%",
    )
    print(
        "Mean margin       :",
        f"{best['mean_semantic_margin']:.6f}",
    )
    print(
        "Mean geometry drift:",
        f"{best['mean_geometry_drift']:.6f}",
    )
    print()
    print("Detail CSV  :", args.detail_csv)
    print("Summary CSV :", args.summary_csv)
    print()
    print(
        "No holdout data were used. "
        "Confirm the selected configuration only on a fresh test set."
    )


if __name__ == "__main__":
    main()
