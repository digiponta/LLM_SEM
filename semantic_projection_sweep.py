# semantic_projection_sweep.py
#
# Development-only sweep for semantic projection preservation_lambda.
# Independent holdout files are intentionally not loaded here.

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F

from model import LanguageModel
from semantic_eval import LabeledSentence, load_benchmark
from semantic_projection import SemanticProjectionHead
from semantic_projection_train import supervised_contrastive_loss
from semantic_router import (
    DEFAULT_ALPHA,
    DEFAULT_BENCHMARK,
    DEFAULT_MODEL,
    DEFAULT_TOKENIZER,
    SemanticRouter,
)
from tokenizer import Tokenizer


DEFAULT_LAMBDAS = "0.0,0.5,1.0,2.0,5.0"
DEFAULT_OUTPUT = "semantic_projection_sweep.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep projection preservation_lambda using development data only."
        )
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--benchmark", default=DEFAULT_BENCHMARK)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--lambdas", default=DEFAULT_LAMBDAS)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--temperature", type=float, default=0.10)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--min-delta", type=float, default=1.0e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--checkpoint-dir",
        default="model/projection_sweep",
    )
    return parser.parse_args()


def parse_lambdas(value: str) -> List[float]:
    result = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not result:
        raise ValueError("At least one preservation lambda is required.")
    if any(item < 0.0 for item in result):
        raise ValueError("preservation lambdas must be >= 0.")
    return result


def centroids(
    samples: Sequence[LabeledSentence],
    vectors: Sequence[torch.Tensor],
) -> Dict[str, torch.Tensor]:
    grouped: Dict[str, List[torch.Tensor]] = {}
    for sample, vector in zip(samples, vectors):
        grouped.setdefault(sample.label, []).append(vector)
    return {
        label: torch.stack(items).mean(dim=0)
        for label, items in grouped.items()
    }


def loo_accuracy(
    samples: Sequence[LabeledSentence],
    vectors: Sequence[torch.Tensor],
) -> float:
    correct = 0
    for index, sample in enumerate(samples):
        train_samples = [
            other for j, other in enumerate(samples) if j != index
        ]
        train_vectors = [
            vector for j, vector in enumerate(vectors) if j != index
        ]
        centers = centroids(train_samples, train_vectors)
        predicted = min(
            centers,
            key=lambda label: 1.0 - float(
                F.cosine_similarity(
                    vectors[index],
                    centers[label],
                    dim=0,
                ).item()
            ),
        )
        correct += int(predicted == sample.label)
    return correct / len(samples)


def semantic_metrics(
    samples: Sequence[LabeledSentence],
    vectors: Sequence[torch.Tensor],
) -> Tuple[float, float, float]:
    within: List[float] = []
    between: List[float] = []
    for i in range(len(samples)):
        for j in range(i + 1, len(samples)):
            similarity = float(
                F.cosine_similarity(vectors[i], vectors[j], dim=0).item()
            )
            if samples[i].label == samples[j].label:
                within.append(similarity)
            else:
                between.append(similarity)
    within_mean = sum(within) / len(within)
    between_mean = sum(between) / len(between)
    return within_mean, between_mean, within_mean - between_mean


def mean_geometry_drift(
    original: torch.Tensor,
    projected: torch.Tensor,
) -> float:
    original_n = F.normalize(original, p=2, dim=-1)
    projected_n = F.normalize(projected, p=2, dim=-1)
    return float(
        (
            1.0
            - F.cosine_similarity(
                projected_n,
                original_n,
                dim=-1,
            )
        ).mean().item()
    )


def train_one(
    base_vectors: torch.Tensor,
    labels: torch.Tensor,
    preservation_lambda: float,
    hidden_dim: int,
    epochs: int,
    lr: float,
    temperature: float,
    weight_decay: float,
    dropout: float,
    patience: int,
    min_delta: float,
    seed: int,
) -> Tuple[SemanticProjectionHead, Dict[str, float]]:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = base_vectors.device
    head = SemanticProjectionHead(
        input_dim=base_vectors.size(1),
        hidden_dim=hidden_dim,
        output_dim=base_vectors.size(1),
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    normalized_base = F.normalize(base_vectors, p=2, dim=-1)
    best_loss = float("inf")
    best_state = None
    best_epoch = 0
    no_improvement = 0

    for epoch in range(1, epochs + 1):
        head.train()
        optimizer.zero_grad(set_to_none=True)

        projected = head(base_vectors, normalize=True)
        contrastive = supervised_contrastive_loss(
            projected,
            labels,
            temperature=temperature,
        )
        preservation = (
            1.0
            - F.cosine_similarity(
                projected,
                normalized_base,
                dim=-1,
            )
        ).mean()
        total = contrastive + preservation_lambda * preservation

        total.backward()
        optimizer.step()

        value = float(total.item())
        if value < best_loss - min_delta:
            best_loss = value
            best_epoch = epoch
            no_improvement = 0
            best_state = {
                key: tensor.detach().cpu().clone()
                for key, tensor in head.state_dict().items()
            }
        else:
            no_improvement += 1

        if no_improvement >= patience:
            break

    if best_state is not None:
        head.load_state_dict(best_state)

    head.eval()
    with torch.no_grad():
        projected = head(base_vectors, normalize=True)
        final_contrastive = float(
            supervised_contrastive_loss(
                projected,
                labels,
                temperature=temperature,
            ).item()
        )
        final_preservation = mean_geometry_drift(
            base_vectors,
            projected,
        )

    return head, {
        "best_loss": best_loss,
        "best_epoch": float(best_epoch),
        "contrastive_loss": final_contrastive,
        "preservation_loss": final_preservation,
    }


def main() -> None:
    args = parse_args()
    lambdas = parse_lambdas(args.lambdas)

    for filename, label in (
        (args.model, "Model checkpoint"),
        (args.tokenizer, "Tokenizer"),
        (args.benchmark, "Benchmark"),
    ):
        if not Path(filename).exists():
            raise FileNotFoundError(f"{label} not found: {filename}")

    if args.hidden_dim <= 0:
        raise ValueError("hidden-dim must be > 0.")
    if args.epochs <= 0:
        raise ValueError("epochs must be > 0.")
    if args.lr <= 0.0:
        raise ValueError("lr must be > 0.")
    if args.temperature <= 0.0:
        raise ValueError("temperature must be > 0.")
    if args.patience <= 0:
        raise ValueError("patience must be > 0.")

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
    label_names = sorted({sample.label for sample in samples})
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
    raw_within, raw_between, raw_margin = semantic_metrics(
        samples,
        raw_vectors,
    )

    print()
    print("LLM_SEM Preservation Lambda Sweep")
    print("---------------------------------")
    print("Device          :", device)
    if device.type == "cuda":
        print("GPU             :", torch.cuda.get_device_name(0))
    print("Benchmark       :", args.benchmark)
    print("Samples         :", len(samples))
    print("Lambdas         :", ", ".join(str(v) for v in lambdas))
    print("Holdout used    : False")
    print("Raw LOO accuracy:", f"{raw_accuracy * 100.0:.2f}%")
    print("Raw margin      :", f"{raw_margin:.6f}")
    print()

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for preservation_lambda in lambdas:
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
            seed=args.seed,
        )

        with torch.no_grad():
            projected_batch = head(base_vectors, normalize=True).cpu()
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

        lambda_tag = str(preservation_lambda).replace(".", "p")
        checkpoint_path = (
            checkpoint_dir
            / f"semantic-projection-lambda-{lambda_tag}.pt"
        )
        head.save_checkpoint(
            str(checkpoint_path),
            benchmark=args.benchmark,
            base_model=args.model,
            base_checkpoint_loss=checkpoint.get("loss"),
            pooling="hybrid",
            hybrid_alpha=args.alpha,
            preservation_lambda=preservation_lambda,
            best_loss=training["best_loss"],
            best_epoch=int(training["best_epoch"]),
            contrastive_loss=training["contrastive_loss"],
            preservation_loss=training["preservation_loss"],
            development_loo_accuracy=accuracy,
            development_semantic_margin=margin,
            development_geometry_drift=drift,
            seed=args.seed,
        )

        row = {
            "preservation_lambda": preservation_lambda,
            "best_epoch": int(training["best_epoch"]),
            "best_total_loss": training["best_loss"],
            "contrastive_loss": training["contrastive_loss"],
            "preservation_loss": training["preservation_loss"],
            "loo_accuracy": accuracy,
            "within_similarity": within,
            "between_similarity": between,
            "semantic_margin": margin,
            "geometry_drift": drift,
            "checkpoint": str(checkpoint_path),
        }
        rows.append(row)

        print(
            f"lambda={preservation_lambda:<5g} "
            f"LOO={accuracy * 100:>6.2f}% "
            f"margin={margin:>8.6f} "
            f"drift={drift:>8.6f} "
            f"epoch={int(training['best_epoch']):>3}"
        )

    # Development-only ranking:
    # 1) highest LOO accuracy
    # 2) highest semantic margin
    # 3) lowest geometry drift
    ranked = sorted(
        rows,
        key=lambda row: (
            -row["loo_accuracy"],
            -row["semantic_margin"],
            row["geometry_drift"],
        ),
    )
    best = ranked[0]

    with Path(args.output).open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)

    print()
    print("Development-only candidate")
    print("--------------------------")
    print("Lambda          :", best["preservation_lambda"])
    print("LOO accuracy    :", f"{best['loo_accuracy'] * 100.0:.2f}%")
    print("Semantic margin :", f"{best['semantic_margin']:.6f}")
    print("Geometry drift  :", f"{best['geometry_drift']:.6f}")
    print("Checkpoint      :", best["checkpoint"])
    print()
    print("Sweep CSV saved :", args.output)
    print()
    print(
        "No holdout data were used for lambda selection. "
        "Use a fresh independent test set for final confirmation."
    )


if __name__ == "__main__":
    main()
