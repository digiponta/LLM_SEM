# semantic_router.py
#
# Semantic routing experiment for LLM_SEM.
#
# Supports:
# - centroid routing
# - leave-one-out known routing evaluation
# - Known/Unknown evaluation without known-sample leakage
# - threshold sweep for similarity/margin trade-offs
#
# Default semantic representation:
#   raw hybrid = 0.35 * attention + 0.65 * last

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F

from model import LanguageModel
from semantic import encode_text
from semantic_eval import LabeledSentence, load_benchmark
from tokenizer import Tokenizer


DEFAULT_MODEL = "model/model-gpu-v0.4.pt"
DEFAULT_TOKENIZER = "model/tokenizer.json"
DEFAULT_BENCHMARK = "my_benchmark.csv"
DEFAULT_UNKNOWN_BENCHMARK = "unknown_benchmark.csv"
DEFAULT_ALPHA = 0.35


@dataclass
class RouteResult:
    label: str
    similarity: float
    distance: float


@dataclass
class KnownScore:
    expected: str
    predicted: str
    top1_similarity: float
    top2_similarity: float
    margin: float
    correct: bool
    text: str


@dataclass
class UnknownScore:
    source_label: str
    top1_label: str
    top1_similarity: float
    top2_similarity: float
    margin: float
    text: str


class SemanticRouter:
    def __init__(
        self,
        model: LanguageModel,
        tokenizer: Tokenizer,
        alpha: float = DEFAULT_ALPHA,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.alpha = alpha
        self.centroids: Dict[str, torch.Tensor] = {}

    def _encode_tensor(self, text: str) -> torch.Tensor:
        semantic = encode_text(
            self.model,
            self.tokenizer,
            text,
            pooling="hybrid",
            hybrid_alpha=self.alpha,
            normalize_hybrid=False,
        )
        return torch.tensor(semantic.vector, dtype=torch.float32)

    def fit(self, samples: List[LabeledSentence]) -> None:
        vectors = [self._encode_tensor(sample.text) for sample in samples]
        self.centroids = _centroids_from_vectors(samples, vectors)

    def route(self, text: str) -> List[RouteResult]:
        if not self.centroids:
            raise RuntimeError("Router has not been fitted.")
        query = self._encode_tensor(text)
        return _rank_against_centroids(query, self.centroids)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Route text to semantic classes using centroid distance."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--benchmark", default=DEFAULT_BENCHMARK)
    parser.add_argument(
        "--unknown-benchmark",
        default=DEFAULT_UNKNOWN_BENCHMARK,
    )
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--text", default=None)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--evaluate-unknown", action="store_true")
    parser.add_argument(
        "--sweep-thresholds",
        action="store_true",
        help="Sweep Known/Unknown similarity and margin thresholds.",
    )
    parser.add_argument(
        "--eval-csv",
        default="semantic_router_eval.csv",
    )
    parser.add_argument(
        "--unknown-eval-csv",
        default="semantic_unknown_eval.csv",
    )
    parser.add_argument(
        "--sweep-csv",
        default="semantic_unknown_threshold_sweep.csv",
    )
    parser.add_argument("--top-k", type=int, default=3)
    return parser.parse_args()


def _centroids_from_vectors(
    samples: Sequence[LabeledSentence],
    vectors: Sequence[torch.Tensor],
) -> Dict[str, torch.Tensor]:
    grouped: Dict[str, List[torch.Tensor]] = defaultdict(list)
    for sample, vector in zip(samples, vectors):
        grouped[sample.label].append(vector)
    return {
        label: torch.stack(items, dim=0).mean(dim=0)
        for label, items in grouped.items()
    }


def _rank_against_centroids(
    query: torch.Tensor,
    centroids: Dict[str, torch.Tensor],
) -> List[RouteResult]:
    results: List[RouteResult] = []
    for label, centroid in centroids.items():
        similarity = float(
            F.cosine_similarity(query, centroid, dim=0).item()
        )
        results.append(
            RouteResult(
                label=label,
                similarity=similarity,
                distance=1.0 - similarity,
            )
        )
    results.sort(key=lambda item: item.similarity, reverse=True)
    return results


def collect_known_loo_scores(
    router: SemanticRouter,
    samples: Sequence[LabeledSentence],
) -> List[KnownScore]:
    vectors = [router._encode_tensor(sample.text) for sample in samples]
    rows: List[KnownScore] = []

    for index, sample in enumerate(samples):
        train_samples = [
            other for j, other in enumerate(samples) if j != index
        ]
        train_vectors = [
            vector for j, vector in enumerate(vectors) if j != index
        ]
        centroids = _centroids_from_vectors(train_samples, train_vectors)
        ranked = _rank_against_centroids(vectors[index], centroids)

        top1 = ranked[0]
        top2 = ranked[1]
        rows.append(
            KnownScore(
                expected=sample.label,
                predicted=top1.label,
                top1_similarity=top1.similarity,
                top2_similarity=top2.similarity,
                margin=top1.similarity - top2.similarity,
                correct=top1.label == sample.label,
                text=sample.text,
            )
        )
    return rows


def collect_unknown_scores(
    router: SemanticRouter,
    known_samples: Sequence[LabeledSentence],
    unknown_samples: Sequence[LabeledSentence],
) -> List[UnknownScore]:
    router.fit(list(known_samples))
    rows: List[UnknownScore] = []
    for sample in unknown_samples:
        ranked = router.route(sample.text)
        top1 = ranked[0]
        top2 = ranked[1]
        rows.append(
            UnknownScore(
                source_label=sample.label,
                top1_label=top1.label,
                top1_similarity=top1.similarity,
                top2_similarity=top2.similarity,
                margin=top1.similarity - top2.similarity,
                text=sample.text,
            )
        )
    return rows


def derive_unknown_thresholds(
    known_scores: Sequence[KnownScore],
) -> Tuple[float, float]:
    correct = [row for row in known_scores if row.correct]
    if not correct:
        raise RuntimeError("No correct known routes are available.")

    similarities = sorted(row.top1_similarity for row in correct)
    margins = sorted(row.margin for row in correct)
    q10_similarity = max(0, int(0.10 * (len(similarities) - 1)))
    q10_margin = max(0, int(0.10 * (len(margins) - 1)))
    return similarities[q10_similarity], margins[q10_margin]


def _unknown_decision(
    top1_similarity: float,
    margin: float,
    similarity_threshold: float,
    margin_threshold: float,
) -> bool:
    return (
        top1_similarity < similarity_threshold
        or margin < margin_threshold
    )


def evaluate_router(
    router: SemanticRouter,
    samples: Sequence[LabeledSentence],
    csv_filename: str,
) -> None:
    scores = collect_known_loo_scores(router, samples)
    labels = sorted({sample.label for sample in samples})

    confusion = {
        expected: {predicted: 0 for predicted in labels}
        for expected in labels
    }
    per_total = {label: 0 for label in labels}
    per_correct = {label: 0 for label in labels}

    for row in scores:
        confusion[row.expected][row.predicted] += 1
        per_total[row.expected] += 1
        per_correct[row.expected] += int(row.correct)

    accuracy = sum(int(row.correct) for row in scores) / len(scores)
    sim_threshold, margin_threshold = derive_unknown_thresholds(scores)

    with Path(csv_filename).open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        writer = csv.writer(f)
        writer.writerow([
            "expected", "predicted", "correct",
            "top1_similarity", "top2_similarity",
            "top1_top2_margin", "text",
        ])
        for row in scores:
            writer.writerow([
                row.expected, row.predicted, row.correct,
                row.top1_similarity, row.top2_similarity,
                row.margin, row.text,
            ])

    print()
    print("============================================================")
    print(" Leave-One-Out Semantic Routing Evaluation")
    print("============================================================")
    print()
    print("Samples          :", len(scores))
    print("Overall accuracy :", f"{accuracy * 100.0:.2f}%")
    print()
    print("Category accuracy")
    print("-----------------")
    for label in labels:
        total = per_total[label]
        correct = per_correct[label]
        value = correct / total if total else 0.0
        print(f"{label:<12} {correct:>2}/{total:<2} {value * 100:>6.2f}%")

    print()
    print("Confusion matrix")
    print("----------------")
    header = "expected\\pred".ljust(14) + "".join(
        f"{label[:10]:>11}" for label in labels
    )
    print(header)
    for expected in labels:
        line = expected.ljust(14)
        for predicted in labels:
            line += f"{confusion[expected][predicted]:>11}"
        print(line)

    print()
    print("Routing confidence")
    print("------------------")
    print(
        "Mean Top-1 similarity :",
        f"{mean(row.top1_similarity for row in scores):.6f}",
    )
    print(
        "Mean Top-1/Top-2 margin:",
        f"{mean(row.margin for row in scores):.6f}",
    )
    print()
    print("Unknown candidate thresholds")
    print("----------------------------")
    print("Top-1 similarity threshold :", f"{sim_threshold:.6f}")
    print("Top-1/Top-2 margin threshold:", f"{margin_threshold:.6f}")
    print()
    print("Evaluation CSV saved:", csv_filename)


def _evaluate_threshold_pair(
    known_scores: Sequence[KnownScore],
    unknown_scores: Sequence[UnknownScore],
    similarity_threshold: float,
    margin_threshold: float,
) -> Dict[str, float]:
    known_total = len(known_scores)
    unknown_total = len(unknown_scores)

    known_accepted = 0
    known_routed_correctly = 0
    false_unknown = 0

    for row in known_scores:
        unknown = _unknown_decision(
            row.top1_similarity,
            row.margin,
            similarity_threshold,
            margin_threshold,
        )
        if unknown:
            false_unknown += 1
        else:
            known_accepted += 1
            if row.correct:
                known_routed_correctly += 1

    unknown_detected = 0
    false_known = 0
    for row in unknown_scores:
        unknown = _unknown_decision(
            row.top1_similarity,
            row.margin,
            similarity_threshold,
            margin_threshold,
        )
        if unknown:
            unknown_detected += 1
        else:
            false_known += 1

    known_recall = (
        known_routed_correctly / known_total if known_total else 0.0
    )
    known_accept_rate = (
        known_accepted / known_total if known_total else 0.0
    )
    unknown_detection = (
        unknown_detected / unknown_total if unknown_total else 0.0
    )
    false_unknown_rate = (
        false_unknown / known_total if known_total else 0.0
    )
    false_known_rate = (
        false_known / unknown_total if unknown_total else 0.0
    )
    balanced_accuracy = (known_recall + unknown_detection) / 2.0

    return {
        "similarity_threshold": similarity_threshold,
        "margin_threshold": margin_threshold,
        "known_recall": known_recall,
        "known_accept_rate": known_accept_rate,
        "unknown_detection_rate": unknown_detection,
        "false_unknown_rate": false_unknown_rate,
        "false_known_rate": false_known_rate,
        "balanced_accuracy": balanced_accuracy,
    }


def evaluate_unknown_router(
    router: SemanticRouter,
    known_samples: Sequence[LabeledSentence],
    unknown_samples: Sequence[LabeledSentence],
    csv_filename: str,
) -> None:
    known_scores = collect_known_loo_scores(router, known_samples)
    unknown_scores = collect_unknown_scores(
        router, known_samples, unknown_samples
    )
    sim_threshold, margin_threshold = derive_unknown_thresholds(known_scores)

    metrics = _evaluate_threshold_pair(
        known_scores,
        unknown_scores,
        sim_threshold,
        margin_threshold,
    )

    rows = []
    for row in known_scores:
        unknown = _unknown_decision(
            row.top1_similarity,
            row.margin,
            sim_threshold,
            margin_threshold,
        )
        rows.append([
            "known", row.expected,
            "unknown" if unknown else row.predicted,
            unknown, row.predicted, row.top1_similarity,
            row.top2_similarity, row.margin, row.text,
        ])

    for row in unknown_scores:
        unknown = _unknown_decision(
            row.top1_similarity,
            row.margin,
            sim_threshold,
            margin_threshold,
        )
        rows.append([
            "unknown", row.source_label,
            "unknown" if unknown else row.top1_label,
            unknown, row.top1_label, row.top1_similarity,
            row.top2_similarity, row.margin, row.text,
        ])

    with Path(csv_filename).open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        writer = csv.writer(f)
        writer.writerow([
            "source", "expected", "predicted", "unknown",
            "top1_label", "top1_similarity", "top2_similarity",
            "top1_top2_margin", "text",
        ])
        writer.writerows(rows)

    print()
    print("============================================================")
    print(" Known / Unknown Semantic Routing Evaluation")
    print("============================================================")
    print()
    print("Known samples          :", len(known_scores))
    print("Unknown samples        :", len(unknown_scores))
    print("Similarity threshold   :", f"{sim_threshold:.6f}")
    print("Margin threshold       :", f"{margin_threshold:.6f}")
    print()
    print("Known recall           :", f"{metrics['known_recall'] * 100:.2f}%")
    print("Known accept rate      :", f"{metrics['known_accept_rate'] * 100:.2f}%")
    print(
        "Unknown detection rate :",
        f"{metrics['unknown_detection_rate'] * 100:.2f}%",
    )
    print(
        "False Unknown rate     :",
        f"{metrics['false_unknown_rate'] * 100:.2f}%",
    )
    print(
        "False Known rate       :",
        f"{metrics['false_known_rate'] * 100:.2f}%",
    )
    print(
        "Balanced accuracy      :",
        f"{metrics['balanced_accuracy'] * 100:.2f}%",
    )

    grouped: Dict[str, List[bool]] = defaultdict(list)
    for row in unknown_scores:
        detected = _unknown_decision(
            row.top1_similarity,
            row.margin,
            sim_threshold,
            margin_threshold,
        )
        grouped[row.source_label].append(detected)

    print()
    print("Unknown category breakdown")
    print("--------------------------")
    for label in sorted(grouped):
        values = grouped[label]
        detected = sum(int(v) for v in values)
        print(
            f"{label:<12} {detected:>2}/{len(values):<2} "
            f"{detected / len(values) * 100:>6.2f}%"
        )

    print()
    print("Unknown evaluation CSV saved:", csv_filename)


def sweep_unknown_thresholds(
    router: SemanticRouter,
    known_samples: Sequence[LabeledSentence],
    unknown_samples: Sequence[LabeledSentence],
    csv_filename: str,
) -> None:
    known_scores = collect_known_loo_scores(router, known_samples)
    unknown_scores = collect_unknown_scores(
        router, known_samples, unknown_samples
    )

    similarities = sorted(
        {round(row.top1_similarity, 6) for row in known_scores + [
            KnownScore(
                expected="unknown",
                predicted=row.top1_label,
                top1_similarity=row.top1_similarity,
                top2_similarity=row.top2_similarity,
                margin=row.margin,
                correct=False,
                text=row.text,
            )
            for row in unknown_scores
        ]}
    )
    margins = sorted(
        {round(row.margin, 6) for row in known_scores + [
            KnownScore(
                expected="unknown",
                predicted=row.top1_label,
                top1_similarity=row.top1_similarity,
                top2_similarity=row.top2_similarity,
                margin=row.margin,
                correct=False,
                text=row.text,
            )
            for row in unknown_scores
        ]}
    )

    # Use compact quantile-like grids instead of every unique value.
    def sample_grid(values: List[float], count: int = 15) -> List[float]:
        if len(values) <= count:
            return values
        return sorted({
            values[round(i * (len(values) - 1) / (count - 1))]
            for i in range(count)
        })

    similarity_grid = sample_grid(similarities)
    margin_grid = sample_grid(margins)

    rows: List[Dict[str, float]] = []
    for sim_threshold in similarity_grid:
        for margin_threshold in margin_grid:
            rows.append(
                _evaluate_threshold_pair(
                    known_scores,
                    unknown_scores,
                    sim_threshold,
                    margin_threshold,
                )
            )

    rows.sort(
        key=lambda row: (
            row["balanced_accuracy"],
            row["unknown_detection_rate"],
            row["known_recall"],
        ),
        reverse=True,
    )

    with Path(csv_filename).open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print()
    print("============================================================")
    print(" Unknown Threshold Sweep")
    print("============================================================")
    print()
    print(
        f"{'SimTh':>8} {'MarginTh':>9} {'Known':>8} "
        f"{'Unknown':>8} {'F-Unk':>8} {'F-Known':>8} {'BalAcc':>8}"
    )
    print("-" * 68)

    for row in rows[:10]:
        print(
            f"{row['similarity_threshold']:>8.4f} "
            f"{row['margin_threshold']:>9.4f} "
            f"{row['known_recall'] * 100:>7.2f}% "
            f"{row['unknown_detection_rate'] * 100:>7.2f}% "
            f"{row['false_unknown_rate'] * 100:>7.2f}% "
            f"{row['false_known_rate'] * 100:>7.2f}% "
            f"{row['balanced_accuracy'] * 100:>7.2f}%"
        )

    best = rows[0]
    print()
    print("Best balanced threshold")
    print("-----------------------")
    print(
        "Similarity threshold:",
        f"{best['similarity_threshold']:.6f}",
    )
    print(
        "Margin threshold    :",
        f"{best['margin_threshold']:.6f}",
    )
    print(
        "Known recall        :",
        f"{best['known_recall'] * 100:.2f}%",
    )
    print(
        "Unknown detection   :",
        f"{best['unknown_detection_rate'] * 100:.2f}%",
    )
    print(
        "Balanced accuracy   :",
        f"{best['balanced_accuracy'] * 100:.2f}%",
    )
    print()
    print("Threshold sweep CSV saved:", csv_filename)


def main() -> None:
    args = parse_args()

    for filename, label in (
        (args.model, "Model checkpoint"),
        (args.tokenizer, "Tokenizer"),
        (args.benchmark, "Benchmark"),
    ):
        if not Path(filename).exists():
            raise FileNotFoundError(f"{label} not found: {filename}")

    if (
        args.evaluate_unknown or args.sweep_thresholds
    ) and not Path(args.unknown_benchmark).exists():
        raise FileNotFoundError(
            f"Unknown benchmark not found: {args.unknown_benchmark}"
        )

    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("alpha must be between 0.0 and 1.0.")

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
    print("LLM_SEM Semantic Router")
    print("-----------------------")
    print("Device     :", device)
    if device.type == "cuda":
        print("GPU        :", torch.cuda.get_device_name(0))
    print("Model      :", args.model)
    print("Benchmark  :", args.benchmark)
    print("Samples    :", len(samples))
    print("Pooling    : raw hybrid")
    print("Alpha      :", args.alpha)
    print("Checkpoint :", checkpoint.get("loss"))
    print()

    router = SemanticRouter(
        model=model,
        tokenizer=tokenizer,
        alpha=args.alpha,
    )

    if args.sweep_thresholds:
        unknown_samples = load_benchmark(args.unknown_benchmark)
        sweep_unknown_thresholds(
            router,
            samples,
            unknown_samples,
            args.sweep_csv,
        )
        return

    if args.evaluate_unknown:
        unknown_samples = load_benchmark(args.unknown_benchmark)
        evaluate_unknown_router(
            router,
            samples,
            unknown_samples,
            args.unknown_eval_csv,
        )
        return

    if args.evaluate:
        evaluate_router(router, samples, args.eval_csv)
        return

    router.fit(samples)
    labels = sorted(router.centroids.keys())
    print("Routes     :", ", ".join(labels))
    print()

    if args.text is not None:
        texts = [args.text]
    else:
        print("Enter text to route. Type 'exit' to quit.")
        print()
        while True:
            text = input("Text> ").strip()
            if text.lower() in {"exit", "quit"}:
                break
            if not text:
                continue
            results = router.route(text)
            print()
            print("Selected route:", results[0].label)
            print("Candidates")
            for result in results[: max(1, args.top_k)]:
                print(
                    f"  {result.label:<12} "
                    f"similarity={result.similarity:.6f} "
                    f"distance={result.distance:.6f}"
                )
            print()
        return

    for text in texts:
        results = router.route(text)
        print("Text          :", text)
        print("Selected route:", results[0].label)
        print("Candidates")
        for result in results[: max(1, args.top_k)]:
            print(
                f"  {result.label:<12} "
                f"similarity={result.similarity:.6f} "
                f"distance={result.distance:.6f}"
            )


if __name__ == "__main__":
    main()
