# semantic_projection_separation_multiseed.py
#
# Multi-seed robustness sweep for LLM_SEM v0.2.
#
# For every separation_lambda x separation_margin combination, train the
# semantic projection with multiple random seeds and aggregate validation
# performance. This reduces the chance of selecting a configuration that
# looks good only for one initialization.
#
# Default seeds: 1,2,3,4,5
# Default grid:
#   lambda = 0.25,0.5,1.0,2.0,5.0
#   margin = 0.02,0.05,0.10,0.20

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List, Sequence

import torch

from model import LanguageModel
from semantic_eval import load_benchmark
from semantic_projection_eval import loo_accuracy, pairwise_margin
from semantic_projection_separation_sweep import (
    DEFAULT_LAMBDAS,
    DEFAULT_MARGINS,
    known_accuracy,
    known_predictions,
    parse_float_list,
    prediction_changes,
    project,
    train_one,
    unknown_detection_rate,
)
from semantic_router import (
    DEFAULT_ALPHA,
    DEFAULT_MODEL,
    DEFAULT_TOKENIZER,
    SemanticRouter,
)
from tokenizer import Tokenizer


DEFAULT_SEEDS = "1,2,3,4,5"
DEFAULT_OUTPUT_CSV = "semantic_projection_separation_multiseed.csv"
DEFAULT_DETAIL_CSV = "semantic_projection_separation_multiseed_detail.csv"


def parse_int_list(value: str) -> List[int]:
    result: List[int] = []
    for item in value.split(","):
        item = item.strip()
        if item:
            result.append(int(item))
    if not result:
        raise argparse.ArgumentTypeError("Expected at least one integer seed.")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a multi-seed semantic projection separation sweep."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--known-holdout", required=True)
    parser.add_argument("--unknown-holdout", default=None)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--temperature", type=float, default=0.10)
    parser.add_argument("--preservation-lambda", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--min-delta", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument(
        "--separation-lambdas",
        type=parse_float_list,
        default=parse_float_list(DEFAULT_LAMBDAS),
    )
    parser.add_argument(
        "--separation-margins",
        type=parse_float_list,
        default=parse_float_list(DEFAULT_MARGINS),
    )
    parser.add_argument(
        "--seeds",
        type=parse_int_list,
        default=parse_int_list(DEFAULT_SEEDS),
    )
    parser.add_argument("--output-csv", default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--detail-csv", default=DEFAULT_DETAIL_CSV)
    return parser.parse_args()


def write_csv(path: str, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()

    for filename, label in (
        (args.model, "Model checkpoint"),
        (args.tokenizer, "Tokenizer"),
        (args.benchmark, "Development benchmark"),
        (args.known_holdout, "Known holdout"),
    ):
        if not Path(filename).exists():
            raise FileNotFoundError(f"{label} not found: {filename}")

    if args.unknown_holdout and not Path(args.unknown_holdout).exists():
        raise FileNotFoundError(
            f"Unknown holdout not found: {args.unknown_holdout}"
        )

    if args.epochs <= 0:
        raise ValueError("epochs must be > 0.")
    if args.lr <= 0.0:
        raise ValueError("lr must be > 0.")
    if args.temperature <= 0.0:
        raise ValueError("temperature must be > 0.")
    if args.preservation_lambda < 0.0:
        raise ValueError("preservation-lambda must be >= 0.")
    if any(value < 0.0 for value in args.separation_lambdas):
        raise ValueError("separation lambdas must be >= 0.")
    if any(value < 0.0 for value in args.separation_margins):
        raise ValueError("separation margins must be >= 0.")

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

    development = load_benchmark(args.benchmark)
    known = load_benchmark(args.known_holdout)
    unknown = (
        load_benchmark(args.unknown_holdout)
        if args.unknown_holdout
        else []
    )

    label_names = sorted({sample.label for sample in development})
    label_to_id = {
        label: index for index, label in enumerate(label_names)
    }

    router = SemanticRouter(
        model=model,
        tokenizer=tokenizer,
        alpha=args.alpha,
    )

    print()
    print("Encoding frozen base semantic vectors...")
    development_raw = [
        router._encode_tensor(sample.text)
        for sample in development
    ]
    known_raw = [
        router._encode_tensor(sample.text)
        for sample in known
    ]
    unknown_raw = [
        router._encode_tensor(sample.text)
        for sample in unknown
    ]

    base_vectors = torch.stack(development_raw).to(device)
    labels = torch.tensor(
        [label_to_id[sample.label] for sample in development],
        dtype=torch.long,
        device=device,
    )

    raw_predictions = known_predictions(
        development,
        development_raw,
        known,
        known_raw,
    )
    raw_known_accuracy = known_accuracy(known, raw_predictions)
    raw_loo = loo_accuracy(development, development_raw)
    _, _, raw_margin = pairwise_margin(
        development,
        development_raw,
    )
    raw_unknown_detection = (
        unknown_detection_rate(
            development,
            development_raw,
            unknown_raw,
        )
        if unknown_raw
        else float("nan")
    )

    combinations = [
        (separation_lambda, separation_margin)
        for separation_lambda in args.separation_lambdas
        for separation_margin in args.separation_margins
    ]
    total_runs = len(combinations) * len(args.seeds)

    print()
    print("============================================================")
    print(" LLM_SEM Separation Multi-Seed Sweep")
    print("============================================================")
    print()
    print("Device             :", device)
    if device.type == "cuda":
        print("GPU                :", torch.cuda.get_device_name(0))
    print("Base checkpoint loss:", checkpoint.get("loss"))
    print("Development samples:", len(development))
    print("Known holdout      :", len(known))
    print("Unknown holdout    :", len(unknown))
    print("Seeds              :", ", ".join(map(str, args.seeds)))
    print("Grid combinations  :", len(combinations))
    print("Total train runs   :", total_runs)
    print("Raw LOO accuracy   :", f"{raw_loo * 100:.2f}%")
    print("Raw known accuracy :", f"{raw_known_accuracy * 100:.2f}%")
    print("Raw semantic margin:", f"{raw_margin:.6f}")
    if unknown_raw:
        print(
            "Raw unknown detect :",
            f"{raw_unknown_detection * 100:.2f}%",
        )
    print()

    detail_rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []

    run_index = 0
    for sep_lambda, sep_margin in combinations:
        known_values: List[float] = []
        loo_values: List[float] = []
        margin_values: List[float] = []
        net_values: List[float] = []
        degraded_values: List[float] = []
        improved_values: List[float] = []
        unknown_values: List[float] = []

        for seed in args.seeds:
            run_index += 1
            result = train_one(
                base_vectors=base_vectors,
                labels=labels,
                input_dim=base_vectors.size(1),
                hidden_dim=args.hidden_dim,
                epochs=args.epochs,
                lr=args.lr,
                temperature=args.temperature,
                preservation_lambda=args.preservation_lambda,
                separation_lambda=sep_lambda,
                separation_margin=sep_margin,
                patience=args.patience,
                min_delta=args.min_delta,
                weight_decay=args.weight_decay,
                dropout=args.dropout,
                seed=seed,
                device=device,
            )

            development_projected = project(
                result.head,
                development_raw,
                device,
            )
            known_projected = project(
                result.head,
                known_raw,
                device,
            )
            unknown_projected = (
                project(result.head, unknown_raw, device)
                if unknown_raw
                else []
            )

            projected_predictions = known_predictions(
                development,
                development_projected,
                known,
                known_projected,
            )
            projected_known_accuracy = known_accuracy(
                known,
                projected_predictions,
            )
            improved, degraded, changed = prediction_changes(
                known,
                raw_predictions,
                projected_predictions,
            )
            projected_loo = loo_accuracy(
                development,
                development_projected,
            )
            within, between, semantic_margin = pairwise_margin(
                development,
                development_projected,
            )
            projected_unknown_detection = (
                unknown_detection_rate(
                    development,
                    development_projected,
                    unknown_projected,
                )
                if unknown_projected
                else float("nan")
            )

            net = improved - degraded

            known_values.append(projected_known_accuracy)
            loo_values.append(projected_loo)
            margin_values.append(semantic_margin)
            net_values.append(float(net))
            degraded_values.append(float(degraded))
            improved_values.append(float(improved))
            if unknown_projected:
                unknown_values.append(projected_unknown_detection)

            detail_rows.append(
                {
                    "separation_lambda": sep_lambda,
                    "separation_margin": sep_margin,
                    "seed": seed,
                    "best_loss": result.best_loss,
                    "best_epoch": result.best_epoch,
                    "known_accuracy": projected_known_accuracy,
                    "known_delta": (
                        projected_known_accuracy - raw_known_accuracy
                    ),
                    "loo_accuracy": projected_loo,
                    "within_similarity": within,
                    "between_similarity": between,
                    "semantic_margin": semantic_margin,
                    "margin_delta": semantic_margin - raw_margin,
                    "improved": improved,
                    "degraded": degraded,
                    "net_improved": net,
                    "prediction_changes": changed,
                    "unknown_detection": projected_unknown_detection,
                }
            )

            unknown_text = (
                f" unknown={projected_unknown_detection * 100:6.2f}%"
                if unknown_projected
                else ""
            )
            print(
                f"[{run_index:>3}/{total_runs}] "
                f"lambda={sep_lambda:<5g} "
                f"margin={sep_margin:<5g} "
                f"seed={seed:<3d} "
                f"known={projected_known_accuracy * 100:6.2f}% "
                f"LOO={projected_loo * 100:6.2f}% "
                f"sem-margin={semantic_margin:.6f} "
                f"net={net:+d}"
                f"{unknown_text}"
            )

        row: Dict[str, object] = {
            "separation_lambda": sep_lambda,
            "separation_margin": sep_margin,
            "seeds": len(args.seeds),
            "mean_known_accuracy": mean(known_values),
            "std_known_accuracy": pstdev(known_values),
            "mean_known_delta": (
                mean(known_values) - raw_known_accuracy
            ),
            "mean_loo_accuracy": mean(loo_values),
            "std_loo_accuracy": pstdev(loo_values),
            "mean_semantic_margin": mean(margin_values),
            "std_semantic_margin": pstdev(margin_values),
            "mean_margin_delta": mean(margin_values) - raw_margin,
            "mean_improved": mean(improved_values),
            "mean_degraded": mean(degraded_values),
            "mean_net_improved": mean(net_values),
            "std_net_improved": pstdev(net_values),
            "mean_unknown_detection": (
                mean(unknown_values)
                if unknown_values
                else float("nan")
            ),
            "std_unknown_detection": (
                pstdev(unknown_values)
                if unknown_values
                else float("nan")
            ),
        }
        summary_rows.append(row)

        print(
            "    aggregate:"
            f" known={row['mean_known_accuracy'] * 100:.2f}%"
            f" +/- {row['std_known_accuracy'] * 100:.2f}"
            f" margin={row['mean_semantic_margin']:.6f}"
            f" net={row['mean_net_improved']:+.2f}"
        )
        print()

    # Robust ranking:
    # 1) highest mean known validation accuracy
    # 2) lowest known-accuracy std
    # 3) highest mean semantic margin
    # 4) highest mean LOO
    # 5) lowest mean degraded samples
    ranked = sorted(
        summary_rows,
        key=lambda row: (
            float(row["mean_known_accuracy"]),
            -float(row["std_known_accuracy"]),
            float(row["mean_semantic_margin"]),
            float(row["mean_loo_accuracy"]),
            -float(row["mean_degraded"]),
        ),
        reverse=True,
    )

    write_csv(args.output_csv, ranked)
    write_csv(args.detail_csv, detail_rows)

    print()
    print("Robust ranking")
    print("==============")
    print()
    print(
        f"{'Rank':>4} {'Lambda':>8} {'Margin':>8} "
        f"{'KnownMean':>10} {'KnownStd':>9} "
        f"{'LOOMean':>9} {'SemMargin':>11} {'Net':>7}"
    )
    print("-" * 82)
    for rank, row in enumerate(ranked[:10], start=1):
        print(
            f"{rank:>4d} "
            f"{float(row['separation_lambda']):>8.3f} "
            f"{float(row['separation_margin']):>8.3f} "
            f"{float(row['mean_known_accuracy']) * 100:>9.2f}% "
            f"{float(row['std_known_accuracy']) * 100:>8.2f}% "
            f"{float(row['mean_loo_accuracy']) * 100:>8.2f}% "
            f"{float(row['mean_semantic_margin']):>11.6f} "
            f"{float(row['mean_net_improved']):>+7.2f}"
        )

    best = ranked[0]
    print()
    print("Robust best configuration")
    print("=========================")
    print()
    print("Separation lambda :", best["separation_lambda"])
    print("Separation margin :", best["separation_margin"])
    print(
        "Mean known accuracy:",
        f"{float(best['mean_known_accuracy']) * 100:.2f}%",
    )
    print(
        "Known accuracy std :",
        f"{float(best['std_known_accuracy']) * 100:.2f}%",
    )
    print(
        "Mean LOO accuracy  :",
        f"{float(best['mean_loo_accuracy']) * 100:.2f}%",
    )
    print(
        "Mean semantic margin:",
        f"{float(best['mean_semantic_margin']):.6f}",
    )
    print(
        "Mean net improved  :",
        f"{float(best['mean_net_improved']):+.2f}",
    )
    if unknown_values:
        print(
            "Mean unknown detect:",
            f"{float(best['mean_unknown_detection']) * 100:.2f}%",
        )

    print()
    print("Summary CSV:", args.output_csv)
    print("Detail CSV :", args.detail_csv)


if __name__ == "__main__":
    main()
