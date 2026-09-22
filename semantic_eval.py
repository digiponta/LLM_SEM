# semantic_eval.py
#
# Evaluate whether the existing LLM checkpoint produces useful semantic
# representations without retraining.

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Dict, List, Sequence, Tuple

import torch

from model import LanguageModel
from semantic import SemanticData, cosine_similarity, encode_text
from tokenizer import Tokenizer


DEFAULT_TOKENIZER = "model/tokenizer.json"
DEFAULT_MODEL = "model/model-gpu-v0.4.pt"


@dataclass(frozen=True)
class LabeledSentence:
    label: str
    text: str


DEFAULT_BENCHMARK: Sequence[LabeledSentence] = (
    LabeledSentence("animal", "猫は動物です。"),
    LabeledSentence("animal", "犬は人間と暮らす動物です。"),
    LabeledSentence("animal", "鳥は空を飛ぶ生き物です。"),
    LabeledSentence("animal", "馬は草を食べる動物です。"),
    LabeledSentence("animal", "魚は水の中で暮らします。"),

    LabeledSentence("weather", "東京の天気を教えてください。"),
    LabeledSentence("weather", "今日は雨が降りそうです。"),
    LabeledSentence("weather", "明日の気温を知りたいです。"),
    LabeledSentence("weather", "台風が近づいています。"),
    LabeledSentence("weather", "冬は気温が低くなります。"),

    LabeledSentence("computer", "コンピュータはプログラムを実行します。"),
    LabeledSentence("computer", "GPUは並列計算を高速に実行します。"),
    LabeledSentence("computer", "Pythonでプログラムを書きます。"),
    LabeledSentence("computer", "ニューラルネットワークをGPUで学習します。"),
    LabeledSentence("computer", "CPUは命令を順番に処理します。"),

    LabeledSentence("food", "私は昼食にカレーを食べました。"),
    LabeledSentence("food", "寿司には魚と米を使います。"),
    LabeledSentence("food", "パンを朝食に食べます。"),
    LabeledSentence("food", "りんごは甘い果物です。"),
    LabeledSentence("food", "料理には新鮮な材料を使います。"),

    LabeledSentence("transport", "電車で東京駅へ行きます。"),
    LabeledSentence("transport", "自動車で高速道路を走ります。"),
    LabeledSentence("transport", "飛行機で海外へ移動します。"),
    LabeledSentence("transport", "バスに乗って駅へ向かいます。"),
    LabeledSentence("transport", "船で海を渡ります。"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate semantic vectors exported from an existing LLM_SEM "
            "checkpoint. No retraining is performed."
        )
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument(
        "--benchmark",
        "--benchmark-json",
        dest="benchmark_file",
        default=None,
        help=(
            "Optional benchmark input (.json or .csv). JSON must contain "
            'a list of {"label": "...", "text": "..."} objects. '
            "CSV must contain label,text columns. --benchmark-json is kept "
            "as a backward-compatible alias."
        ),
    )
    parser.add_argument(
        "--csv",
        default="semantic_eval_results.csv",
        help="CSV output for pairwise similarities.",
    )
    return parser.parse_args()


def load_benchmark(filename: str | None) -> List[LabeledSentence]:
    if filename is None:
        return list(DEFAULT_BENCHMARK)

    path = Path(filename)
    suffix = path.suffix.lower()
    result: List[LabeledSentence] = []

    if suffix == ".json":
        # utf-8-sig also accepts ordinary UTF-8 and safely strips a BOM.
        with path.open("r", encoding="utf-8-sig") as f:
            raw = json.load(f)

        for item in raw:
            label = str(item["label"]).strip()
            text = str(item["text"]).strip()
            if not label or not text:
                raise ValueError("Benchmark label/text must not be empty.")
            result.append(LabeledSentence(label=label, text=text))

    elif suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            fields = set(reader.fieldnames or [])

            if not {"label", "text"}.issubset(fields):
                if {"label_a", "label_b", "text_a", "text_b"}.issubset(fields):
                    raise ValueError(
                        f"{filename} is a pairwise evaluation OUTPUT file, "
                        "not a benchmark input. Run 'python semantic_eval.py' "
                        "to use the built-in benchmark, or provide a CSV with "
                        "exactly the benchmark columns 'label,text'."
                    )
                raise ValueError(
                    "Benchmark CSV must contain columns: label,text"
                )

            for row in reader:
                label = str(row["label"]).strip()
                text = str(row["text"]).strip()
                if not label or not text:
                    raise ValueError("Benchmark label/text must not be empty.")
                result.append(LabeledSentence(label=label, text=text))

    else:
        raise ValueError(
            "Benchmark file must be .json or .csv. "
            "Use --benchmark FILE, or omit it for the built-in benchmark."
        )

    if len(result) < 2:
        raise ValueError("Benchmark must contain at least two sentences.")
    return result


def encode_benchmark(
    model: LanguageModel,
    tokenizer: Tokenizer,
    benchmark: Sequence[LabeledSentence],
) -> List[SemanticData]:
    vectors: List[SemanticData] = []
    for index, sample in enumerate(benchmark, start=1):
        semantic = encode_text(model, tokenizer, sample.text)
        vectors.append(semantic)
        print(
            f"\rEncoding {index}/{len(benchmark)} "
            f"| {sample.label:<10} | {sample.text[:30]}",
            end="",
            flush=True,
        )
    print()
    return vectors


def pairwise_scores(
    benchmark: Sequence[LabeledSentence],
    vectors: Sequence[SemanticData],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []

    for i in range(len(benchmark)):
        for j in range(i + 1, len(benchmark)):
            similarity = cosine_similarity(vectors[i], vectors[j])
            same_label = benchmark[i].label == benchmark[j].label
            rows.append(
                {
                    "index_a": i,
                    "index_b": j,
                    "label_a": benchmark[i].label,
                    "label_b": benchmark[j].label,
                    "text_a": benchmark[i].text,
                    "text_b": benchmark[j].text,
                    "same_label": same_label,
                    "similarity": similarity,
                    "distance": 1.0 - similarity,
                }
            )

    return rows


def nearest_neighbor_accuracy(
    benchmark: Sequence[LabeledSentence],
    vectors: Sequence[SemanticData],
) -> Tuple[float, List[Tuple[int, int, float, bool]]]:
    predictions: List[Tuple[int, int, float, bool]] = []
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

        is_correct = benchmark[i].label == benchmark[best_j].label
        correct += int(is_correct)
        predictions.append((i, best_j, best_score, is_correct))

    return correct / len(benchmark), predictions


def write_csv(filename: str, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return

    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_report(
    benchmark: Sequence[LabeledSentence],
    rows: Sequence[Dict[str, object]],
    nn_accuracy: float,
    predictions: Sequence[Tuple[int, int, float, bool]],
) -> None:
    within = [
        float(row["similarity"])
        for row in rows
        if bool(row["same_label"])
    ]
    between = [
        float(row["similarity"])
        for row in rows
        if not bool(row["same_label"])
    ]

    within_mean = mean(within) if within else float("nan")
    between_mean = mean(between) if between else float("nan")
    margin = within_mean - between_mean

    labels = sorted({sample.label for sample in benchmark})

    print()
    print("====================================")
    print(" Semantic Performance Evaluation")
    print("====================================")
    print()
    print("Sentences              :", len(benchmark))
    print("Semantic classes       :", len(labels), ", ".join(labels))
    print("Within-class pairs     :", len(within))
    print("Between-class pairs    :", len(between))
    print()
    print("Mean within similarity :", f"{within_mean:.6f}")
    print("Mean between similarity:", f"{between_mean:.6f}")
    print("Semantic margin        :", f"{margin:.6f}")
    print("1-NN label accuracy    :", f"{nn_accuracy * 100.0:.2f}%")
    print()

    print("Interpretation")
    print("--------------")
    if margin > 0:
        print(
            "PASS signal: semantically grouped sentences are, on average, "
            "closer than sentences from different groups."
        )
    else:
        print(
            "FAIL signal: this benchmark does not show useful semantic "
            "separation with the current representation."
        )

    print()
    print("Nearest-neighbor examples")
    print("-------------------------")
    for source_i, target_i, score, correct in predictions:
        status = "OK" if correct else "MISS"
        print(
            f"[{status}] {benchmark[source_i].label:<10} "
            f"{benchmark[source_i].text}"
        )
        print(
            f"       -> {benchmark[target_i].label:<10} "
            f"{benchmark[target_i].text}"
        )
        print(f"          similarity={score:.6f}")


def main() -> None:
    args = parse_args()

    if not Path(args.tokenizer).exists():
        raise FileNotFoundError(f"Tokenizer not found: {args.tokenizer}")
    if not Path(args.model).exists():
        raise FileNotFoundError(f"Model checkpoint not found: {args.model}")

    benchmark = load_benchmark(args.benchmark_file)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print()
    print("LLM_SEM Semantic Evaluation")
    print("---------------------------")
    print("Device    :", device)
    if device.type == "cuda":
        print("GPU       :", torch.cuda.get_device_name(0))
    print("Model     :", args.model)
    print("Tokenizer :", args.tokenizer)
    print("Samples   :", len(benchmark))
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

    print("Checkpoint loss:", checkpoint.get("loss"))
    print("Vector dimension:", model.d_model)
    print()

    vectors = encode_benchmark(model, tokenizer, benchmark)
    rows = pairwise_scores(benchmark, vectors)
    accuracy, predictions = nearest_neighbor_accuracy(
        benchmark,
        vectors,
    )

    write_csv(args.csv, rows)
    print_report(
        benchmark,
        rows,
        accuracy,
        predictions,
    )

    print()
    print("Pairwise CSV saved:", args.csv)


if __name__ == "__main__":
    main()
