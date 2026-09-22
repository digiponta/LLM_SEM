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
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

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
        "--top-k",
        type=int,
        default=3,
        help="Number of candidate routes to display.",
    )
    return parser.parse_args()


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
