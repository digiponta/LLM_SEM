# semantic_hybrid_eval.py
#
# Sweep alpha for a hybrid semantic vector:
#
#   hybrid = alpha * attention + (1 - alpha) * last
#
# Uses an existing checkpoint only. No retraining is performed.

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean
from typing import Dict, List, Sequence, Tuple

import torch

from model import LanguageModel
from semantic import SemanticData, cosine_similarity, encode_text
from semantic_eval import LabeledSentence, load_benchmark
from tokenizer import Tokenizer


DEFAULT_TOKENIZER = "model/tokenizer.json"
DEFAULT_MODEL = "model/model-gpu-v0.4.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep hybrid semantic pooling alpha without retraining."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--benchmark", default="my_benchmark.csv")
    parser.add_argument("--alpha-start", type=float, default=0.0)
    parser.add_argument("--alpha-stop", type=float, default=1.0)
    parser.add_argument("--alpha-step", type=float, default=0.1)
    parser.add_argument(
        "--mode",
        choices=("raw", "normalized", "both"),
        default="both",
        help="Compare raw hybrid, normalized hybrid, or both.",
    )
    parser.add_argument(
        "--summary-csv",
        default="semantic_hybrid_summary.csv",
    )
    return parser.parse_args()


def alpha_values(start: float, stop: float, step: float) -> List[float]:
    if step <= 0:
        raise ValueError("alpha-step must be > 0.")
    if start < 0.0 or stop > 1.0 or start > stop:
        raise ValueError("alpha range must satisfy 0 <= start <= stop <= 1.")

    values: List[float] = []
    value = start
    while value <= stop + 1.0e-9:
        values.append(round(value, 10))
        value += step
    return values


def encode_benchmark(
    model: LanguageModel,
    tokenizer: Tokenizer,
    benchmark: Sequence[LabeledSentence],
    alpha: float,
    normalized: bool,
) -> List[SemanticData]:
    vectors: List[SemanticData] = []
    for index, sample in enumerate(benchmark, start=1):
        vectors.append(
            encode_text(
                model,
                tokenizer,
                sample.text,
                pooling="hybrid",
                hybrid_alpha=alpha,
                normalize_hybrid=normalized,
            )
        )
        print(
            f"\r[{'norm' if normalized else 'raw ':<4} alpha={alpha:0.2f}] Encoding "
            f"{index}/{len(benchmark)} | {sample.label:<10}",
            end="",
            flush=True,
        )
    print()
    return vectors


def evaluate(
    benchmark: Sequence[LabeledSentence],
    vectors: Sequence[SemanticData],
    alpha: float,
    normalized: bool,
) -> Dict[str, object]:
    within: List[float] = []
    between: List[float] = []

    for i in range(len(benchmark)):
        for j in range(i + 1, len(benchmark)):
            score = cosine_similarity(vectors[i], vectors[j])
            if benchmark[i].label == benchmark[j].label:
                within.append(score)
            else:
                between.append(score)

    correct = 0
    for i in range(len(benchmark)):
        best_j = -1
        best_score = float("-inf")
        for j in range(len(benchmark)):
            if i == j:
                continue
            score = cosine_similarity(vectors[i], vectors[j])
            if score > best_score:
                best_score = score
                best_j = j

        if benchmark[i].label == benchmark[best_j].label:
            correct += 1

    within_mean = mean(within)
    between_mean = mean(between)

    return {
        "mode": "normalized" if normalized else "raw",
        "alpha": alpha,
        "within_similarity": within_mean,
        "between_similarity": between_mean,
        "semantic_margin": within_mean - between_mean,
        "nn_accuracy": correct / len(benchmark),
    }


def write_csv(filename: str, rows: Sequence[Dict[str, object]]) -> None:
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_results(rows: Sequence[Dict[str, object]]) -> None:
    print()
    print("==============================================================")
    print(" Hybrid Semantic Alpha Sweep")
    print(" hybrid = alpha * attention + (1-alpha) * last")
    print("==============================================================")
    print()
    print(
        f"{'Mode':<11} {'Alpha':>7} {'Within':>10} {'Between':>10} "
        f"{'Margin':>10} {'1-NN':>10}"
    )
    print("-" * 66)

    for row in rows:
        print(
            f"{str(row['mode']):<11} "
            f"{float(row['alpha']):>7.2f} "
            f"{float(row['within_similarity']):>10.6f} "
            f"{float(row['between_similarity']):>10.6f} "
            f"{float(row['semantic_margin']):>10.6f} "
            f"{float(row['nn_accuracy']) * 100.0:>9.2f}%"
        )

    best_accuracy = max(
        rows,
        key=lambda row: (
            float(row["nn_accuracy"]),
            float(row["semantic_margin"]),
        ),
    )
    best_margin = max(
        rows,
        key=lambda row: (
            float(row["semantic_margin"]),
            float(row["nn_accuracy"]),
        ),
    )

    print()
    print(
        "Best 1-NN alpha :",
        f"{best_accuracy['mode']} alpha={float(best_accuracy['alpha']):.2f}",
        f"accuracy={float(best_accuracy['nn_accuracy']) * 100.0:.2f}%",
        f"margin={float(best_accuracy['semantic_margin']):.6f}",
    )
    print(
        "Best margin alpha:",
        f"{best_margin['mode']} alpha={float(best_margin['alpha']):.2f}",
        f"margin={float(best_margin['semantic_margin']):.6f}",
        f"accuracy={float(best_margin['nn_accuracy']) * 100.0:.2f}%",
    )


def main() -> None:
    args = parse_args()

    if not Path(args.tokenizer).exists():
        raise FileNotFoundError(f"Tokenizer not found: {args.tokenizer}")
    if not Path(args.model).exists():
        raise FileNotFoundError(f"Model checkpoint not found: {args.model}")

    benchmark = load_benchmark(args.benchmark)
    alphas = alpha_values(
        args.alpha_start,
        args.alpha_stop,
        args.alpha_step,
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print()
    print("LLM_SEM Hybrid Semantic Evaluation")
    print("----------------------------------")
    print("Device    :", device)
    if device.type == "cuda":
        print("GPU       :", torch.cuda.get_device_name(0))
    print("Model     :", args.model)
    print("Tokenizer :", args.tokenizer)
    print("Benchmark :", args.benchmark)
    print("Samples   :", len(benchmark))
    print("Mode      :", args.mode)
    print(
        "Alpha     :",
        f"{args.alpha_start:.2f} .. {args.alpha_stop:.2f}",
        f"step {args.alpha_step:.2f}",
    )
    print()

    tokenizer = Tokenizer.load(args.tokenizer)
    model, checkpoint = LanguageModel.load_checkpoint(
        args.model,
        device=device,
    )

    if tokenizer.vocab_size != model.vocab_size:
        raise ValueError(
            "Tokenizer/model vocabulary mismatch: "
            f"{tokenizer.vocab_size} != {model.vocab_size}"
        )

    print("Checkpoint loss :", checkpoint.get("loss"))
    print("Vector dimension:", model.d_model)
    print()

    rows: List[Dict[str, object]] = []

    modes = (
        (False, True)
        if args.mode == "both"
        else ((args.mode == "normalized"),)
    )

    for normalized in modes:
        for alpha in alphas:
            vectors = encode_benchmark(
                model,
                tokenizer,
                benchmark,
                alpha,
                normalized,
            )
            rows.append(
                evaluate(
                    benchmark,
                    vectors,
                    alpha,
                    normalized,
                )
            )

    print_results(rows)
    write_csv(args.summary_csv, rows)

    print()
    print("Summary CSV saved:", args.summary_csv)


if __name__ == "__main__":
    main()
