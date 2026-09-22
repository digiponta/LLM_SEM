# semantic_router.py
#
# Semantic routing experiment for LLM_SEM.
#
# Builds one centroid per semantic class from a labeled benchmark and routes
# new text to the nearest centroid using cosine similarity.
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
from typing import Dict, List, Sequence

import torch
import torch.nn.functional as F

from model import LanguageModel
from semantic import SemanticData, encode_text
from semantic_eval import LabeledSentence, load_benchmark
from tokenizer import Tokenizer


DEFAULT_MODEL = "model/model-gpu-v0.4.pt"
DEFAULT_TOKENIZER = "model/tokenizer.json"
DEFAULT_BENCHMARK = "my_benchmark.csv"
DEFAULT_ALPHA = 0.35


@dataclass
class RouteResult:
    label: str
    similarity: float
    distance: float


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
        return torch.tensor(
            semantic.vector,
            dtype=torch.float32,
        )

    def fit(self, samples: List[LabeledSentence]) -> None:
        grouped: Dict[str, List[torch.Tensor]] = {}

        for sample in samples:
            grouped.setdefault(sample.label, []).append(
                self._encode_tensor(sample.text)
            )

        self.centroids = {
            label: torch.stack(vectors, dim=0).mean(dim=0)
            for label, vectors in grouped.items()
        }

    def route(self, text: str) -> List[RouteResult]:
        if not self.centroids:
            raise RuntimeError("Router has not been fitted.")

        query = self._encode_tensor(text)
        results: List[RouteResult] = []

        for label, centroid in self.centroids.items():
            similarity = float(
                F.cosine_similarity(
                    query,
                    centroid,
                    dim=0,
                ).item()
            )
            results.append(
                RouteResult(
                    label=label,
                    similarity=similarity,
                    distance=1.0 - similarity,
                )
            )

        results.sort(
            key=lambda result: result.similarity,
            reverse=True,
        )
        return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Route text to semantic classes using centroid distance."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--benchmark", default=DEFAULT_BENCHMARK)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--text", default=None)
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run leave-one-out routing evaluation on the benchmark.",
    )
    parser.add_argument(
        "--eval-csv",
        default="semantic_router_eval.csv",
        help="CSV output for leave-one-out routing evaluation.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of candidate routes to display.",
    )
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


def evaluate_router(
    router: SemanticRouter,
    samples: Sequence[LabeledSentence],
    csv_filename: str,
) -> None:
    vectors = [router._encode_tensor(sample.text) for sample in samples]
    labels = sorted({sample.label for sample in samples})
    confusion = {
        expected: {predicted: 0 for predicted in labels}
        for expected in labels
    }
    per_class_total = {label: 0 for label in labels}
    per_class_correct = {label: 0 for label in labels}
    rows = []
    correct_top1 = []
    correct_margin = []

    for index, sample in enumerate(samples):
        train_samples = [
            other for j, other in enumerate(samples) if j != index
        ]
        train_vectors = [
            vector for j, vector in enumerate(vectors) if j != index
        ]
        centroids = _centroids_from_vectors(train_samples, train_vectors)

        query = vectors[index]
        ranked = []
        for label, centroid in centroids.items():
            similarity = float(
                F.cosine_similarity(query, centroid, dim=0).item()
            )
            ranked.append((label, similarity))
        ranked.sort(key=lambda item: item[1], reverse=True)

        predicted = ranked[0][0]
        top1 = ranked[0][1]
        top2 = ranked[1][1] if len(ranked) > 1 else float("nan")
        margin = top1 - top2 if len(ranked) > 1 else float("nan")
        is_correct = predicted == sample.label

        confusion[sample.label][predicted] += 1
        per_class_total[sample.label] += 1
        per_class_correct[sample.label] += int(is_correct)

        if is_correct:
            correct_top1.append(top1)
            correct_margin.append(margin)

        rows.append(
            {
                "index": index,
                "expected": sample.label,
                "predicted": predicted,
                "correct": is_correct,
                "top1_similarity": top1,
                "top2_similarity": top2,
                "top1_top2_margin": margin,
                "text": sample.text,
            }
        )

    accuracy = sum(int(row["correct"]) for row in rows) / len(rows)

    path = Path(csv_filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print()
    print("============================================================")
    print(" Leave-One-Out Semantic Routing Evaluation")
    print("============================================================")
    print()
    print("Samples          :", len(samples))
    print("Overall accuracy :", f"{accuracy * 100.0:.2f}%")
    print()

    print("Category accuracy")
    print("-----------------")
    for label in labels:
        total = per_class_total[label]
        correct = per_class_correct[label]
        value = correct / total if total else 0.0
        print(
            f"{label:<12} {correct:>2}/{total:<2} "
            f"{value * 100.0:>6.2f}%"
        )

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
    all_top1 = [float(row["top1_similarity"]) for row in rows]
    all_margin = [float(row["top1_top2_margin"]) for row in rows]
    print("Mean Top-1 similarity :", f"{mean(all_top1):.6f}")
    print("Mean Top-1/Top-2 margin:", f"{mean(all_margin):.6f}")

    if correct_top1 and correct_margin:
        sorted_top1 = sorted(correct_top1)
        sorted_margin = sorted(correct_margin)
        q10_index_top1 = max(0, int(0.10 * (len(sorted_top1) - 1)))
        q10_index_margin = max(0, int(0.10 * (len(sorted_margin) - 1)))
        similarity_threshold = sorted_top1[q10_index_top1]
        margin_threshold = sorted_margin[q10_index_margin]

        print()
        print("Unknown candidate thresholds")
        print("----------------------------")
        print(
            "Top-1 similarity threshold :",
            f"{similarity_threshold:.6f}",
        )
        print(
            "Top-1/Top-2 margin threshold:",
            f"{margin_threshold:.6f}",
        )
        print(
            "Candidate rule: Unknown if Top-1 similarity is below "
            "the similarity threshold OR the Top-1/Top-2 margin is below "
            "the margin threshold."
        )

    print()
    print("Evaluation CSV saved:", csv_filename)


def main() -> None:
    args = parse_args()

    if not Path(args.model).exists():
        raise FileNotFoundError(f"Model checkpoint not found: {args.model}")
    if not Path(args.tokenizer).exists():
        raise FileNotFoundError(f"Tokenizer not found: {args.tokenizer}")
    if not Path(args.benchmark).exists():
        raise FileNotFoundError(f"Benchmark not found: {args.benchmark}")
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
    router.fit(samples)

    if args.evaluate:
        evaluate_router(
            router=router,
            samples=samples,
            csv_filename=args.eval_csv,
        )
        return

    labels = sorted(router.centroids.keys())
    print("Routes     :", ", ".join(labels))
    print()

    if args.text is not None:
        texts = [args.text]
    else:
        texts = []
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
