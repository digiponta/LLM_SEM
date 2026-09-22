# semantic_infer.py
#
# Operational semantic inference for LLM_SEM v0.2.
#
# Pipeline:
#   text -> frozen LLM_GPU v0.4 -> raw 64-D semantic vector
#        -> learned semantic projection -> projected 64-D vector
#        -> nearest semantic centroid -> class-radius Known/Unknown decision

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run semantic classification with the frozen base LLM and "
            "the trained LLM_SEM v0.2 projection head."
        )
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--benchmark", default=DEFAULT_BENCHMARK)
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
    parser.add_argument("--text", default=None)
    parser.add_argument("--top-k", type=int, default=3)
    return parser.parse_args()


@torch.no_grad()
def project_one(
    head: SemanticProjectionHead,
    vector: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    projected = head(
        vector.to(device).unsqueeze(0),
        normalize=True,
    )
    return projected[0].cpu()


def rank_classes(
    vector: torch.Tensor,
    centers: Dict[str, torch.Tensor],
) -> List[Tuple[str, float, float]]:
    rows: List[Tuple[str, float, float]] = []
    for label, center in centers.items():
        distance = cosine_distance(vector, center)
        rows.append((label, 1.0 - distance, distance))
    rows.sort(key=lambda item: item[1], reverse=True)
    return rows


def main() -> None:
    args = parse_args()

    for filename, label in (
        (args.model, "Base model checkpoint"),
        (args.tokenizer, "Tokenizer"),
        (args.benchmark, "Semantic benchmark"),
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
    if args.top_k <= 0:
        raise ValueError("top-k must be > 0.")

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

    samples = load_benchmark(args.benchmark)
    router = SemanticRouter(
        model=model,
        tokenizer=tokenizer,
        alpha=args.alpha,
    )

    print()
    print("Encoding development semantic vectors...")
    raw_vectors = [
        router._encode_tensor(sample.text)
        for sample in samples
    ]

    batch = torch.stack(raw_vectors).to(device)
    with torch.no_grad():
        projected_batch = head(batch, normalize=True).cpu()
    projected_vectors = [row for row in projected_batch]

    centers = centroids(samples, projected_vectors)
    radii = class_radii(
        samples,
        projected_vectors,
        args.radius_quantile,
        args.radius_scale,
    )

    print()
    print("LLM_SEM v0.2 Semantic Inference")
    print("--------------------------------")
    print("Device              :", device)
    if device.type == "cuda":
        print("GPU                 :", torch.cuda.get_device_name(0))
    print("Base model          :", args.model)
    print("Base checkpoint loss:", base_checkpoint.get("loss"))
    print("Base LLM frozen     : True")
    print("Projection          :", args.projection)
    print("Projection lambda   :", projection_checkpoint.get("preservation_lambda"))
    print("Projection seed     :", projection_checkpoint.get("seed"))
    print("Benchmark           :", args.benchmark)
    print("Classes             :", ", ".join(sorted(centers)))
    print("Radius quantile     :", args.radius_quantile)
    print("Radius scale        :", args.radius_scale)
    print()

    def infer_text(text: str) -> None:
        raw = router._encode_tensor(text)
        projected = project_one(head, raw, device)
        ranked = rank_classes(projected, centers)

        top_label, top_similarity, top_distance = ranked[0]
        radius = radii[top_label]
        is_unknown = top_distance > radius
        selected = "unknown" if is_unknown else top_label

        print("Text           :", text)
        print("Selected route :", selected)
        print("Nearest class  :", top_label)
        print("Similarity     :", f"{top_similarity:.6f}")
        print("Distance       :", f"{top_distance:.6f}")
        print("Class radius   :", f"{radius:.6f}")
        print("Known/Unknown  :", "Unknown" if is_unknown else "Known")
        print("Candidates")
        for label, similarity, distance in ranked[: args.top_k]:
            print(
                f"  {label:<12} "
                f"similarity={similarity:.6f} "
                f"distance={distance:.6f} "
                f"radius={radii[label]:.6f}"
            )
        print()

    if args.text is not None:
        infer_text(args.text)
        return

    print("Enter text to classify. Type 'exit' to quit.")
    print()
    while True:
        text = input("Text> ").strip()
        if text.lower() in {"exit", "quit"}:
            break
        if not text:
            continue
        infer_text(text)


if __name__ == "__main__":
    main()
