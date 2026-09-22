# semantic_projection_arch_sweep.py
#
# Projection architecture sweep for LLM_SEM v0.2.
#
# Sweeps hidden dimension and preservation lambda while keeping:
#   - base LLM frozen
#   - semantic pooling fixed
#   - radius_scale fixed at 1.0
#
# Evaluates each candidate on:
#   - development LOO accuracy
#   - geometry drift
#   - known holdout base accuracy
#   - known recall after class-radius rejection
#   - unknown stress detection rate
#   - balanced accuracy

from __future__ import annotations

import argparse
import csv
import copy
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F

from model import LanguageModel
from semantic_eval import load_benchmark
from semantic_projection import SemanticProjectionHead
from semantic_projection_eval import (
    centroids,
    class_radii,
    cosine_distance,
    loo_accuracy,
)
from semantic_projection_train import supervised_contrastive_loss
from semantic_router import (
    DEFAULT_ALPHA,
    DEFAULT_BENCHMARK,
    DEFAULT_MODEL,
    DEFAULT_TOKENIZER,
    SemanticRouter,
)
from tokenizer import Tokenizer


DEFAULT_KNOWN_HOLDOUT = "holdout_benchmark.csv"
DEFAULT_UNKNOWN_STRESS = "unknown_stress_benchmark.csv"
DEFAULT_OUTPUT = "semantic_projection_arch_sweep.csv"
DEFAULT_HIDDEN_DIMS = "32,64,128,256"
DEFAULT_LAMBDAS = "0.5,1.0,2.0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep projection hidden dimension and preservation lambda."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--benchmark", default=DEFAULT_BENCHMARK)
    parser.add_argument("--known-holdout", default=DEFAULT_KNOWN_HOLDOUT)
    parser.add_argument("--unknown-stress", default=DEFAULT_UNKNOWN_STRESS)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--hidden-dims", default=DEFAULT_HIDDEN_DIMS)
    parser.add_argument("--lambdas", default=DEFAULT_LAMBDAS)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--temperature", type=float, default=0.10)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--min-delta", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--radius-quantile", type=float, default=0.90)
    parser.add_argument("--radius-scale", type=float, default=1.0)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    return parser.parse_args()


def parse_int_list(value: str) -> List[int]:
    result = [int(x.strip()) for x in value.split(",") if x.strip()]
    if not result or any(v <= 0 for v in result):
        raise ValueError("hidden dimensions must be positive integers.")
    return result


def parse_float_list(value: str) -> List[float]:
    result = [float(x.strip()) for x in value.split(",") if x.strip()]
    if not result or any(v < 0.0 for v in result):
        raise ValueError("lambdas must be non-negative.")
    return result


@torch.no_grad()
def project_batch(
    head: SemanticProjectionHead,
    vectors: List[torch.Tensor],
    device: torch.device,
) -> List[torch.Tensor]:
    batch = torch.stack(vectors).to(device)
    projected = head(batch, normalize=True).cpu()
    return [row for row in projected]


def nearest(vector: torch.Tensor, centers: Dict[str, torch.Tensor]):
    return min(
        (
            (label, cosine_distance(vector, center))
            for label, center in centers.items()
        ),
        key=lambda item: item[1],
    )


def geometry_drift(
    raw_vectors: List[torch.Tensor],
    projected_vectors: List[torch.Tensor],
) -> float:
    raw = F.normalize(torch.stack(raw_vectors), p=2, dim=-1)
    projected = F.normalize(torch.stack(projected_vectors), p=2, dim=-1)
    return float(
        (1.0 - F.cosine_similarity(raw, projected, dim=-1)).mean().item()
    )


def train_candidate(
    base_vectors: torch.Tensor,
    labels: torch.Tensor,
    hidden_dim: int,
    preservation_lambda: float,
    args: argparse.Namespace,
    device: torch.device,
):
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    head = SemanticProjectionHead(
        input_dim=base_vectors.size(1),
        hidden_dim=hidden_dim,
        output_dim=base_vectors.size(1),
        dropout=0.0,
    ).to(device)

    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    normalized_base = F.normalize(base_vectors, p=2, dim=-1)
    best_loss = float("inf")
    best_epoch = 0
    best_state = None
    no_improvement = 0

    for epoch in range(1, args.epochs + 1):
        head.train()
        optimizer.zero_grad(set_to_none=True)

        projected = head(base_vectors, normalize=True)
        contrastive = supervised_contrastive_loss(
            projected,
            labels,
            temperature=args.temperature,
        )
        preservation = (
            1.0
            - F.cosine_similarity(projected, normalized_base, dim=-1)
        ).mean()
        loss = contrastive + preservation_lambda * preservation
        loss.backward()
        optimizer.step()

        value = float(loss.item())
        if value < best_loss - args.min_delta:
            best_loss = value
            best_epoch = epoch
            no_improvement = 0
            best_state = copy.deepcopy(head.state_dict())
        else:
            no_improvement += 1

        if no_improvement >= args.patience:
            break

    if best_state is not None:
        head.load_state_dict(best_state)
    head.eval()
    return head, best_loss, best_epoch


def evaluate_candidate(
    development,
    development_raw,
    development_projected,
    known_holdout,
    known_projected,
    unknown_stress,
    unknown_projected,
    args,
):
    centers = centroids(development, development_projected)
    radii = class_radii(
        development,
        development_projected,
        args.radius_quantile,
        args.radius_scale,
    )

    known_base_correct = 0
    known_routed_correct = 0
    false_unknown = 0

    for sample, vector in zip(known_holdout, known_projected):
        label, distance = nearest(vector, centers)
        rejected = distance > radii[label]
        base_correct = label == sample.label
        known_base_correct += int(base_correct)
        known_routed_correct += int(base_correct and not rejected)
        false_unknown += int(rejected)

    unknown_detected = 0
    for _sample, vector in zip(unknown_stress, unknown_projected):
        label, distance = nearest(vector, centers)
        unknown_detected += int(distance > radii[label])

    known_total = len(known_holdout)
    unknown_total = len(unknown_stress)

    known_base_accuracy = known_base_correct / known_total
    known_recall = known_routed_correct / known_total
    false_unknown_rate = false_unknown / known_total
    unknown_detection_rate = unknown_detected / unknown_total
    balanced_accuracy = (known_recall + unknown_detection_rate) / 2.0

    return {
        "development_loo_accuracy": loo_accuracy(
            development,
            development_projected,
        ),
        "geometry_drift": geometry_drift(
            development_raw,
            development_projected,
        ),
        "known_base_accuracy": known_base_accuracy,
        "known_recall": known_recall,
        "false_unknown_rate": false_unknown_rate,
        "unknown_detection_rate": unknown_detection_rate,
        "balanced_accuracy": balanced_accuracy,
    }


def main() -> None:
    args = parse_args()
    hidden_dims = parse_int_list(args.hidden_dims)
    lambdas = parse_float_list(args.lambdas)

    for filename, label in (
        (args.model, "Base model checkpoint"),
        (args.tokenizer, "Tokenizer"),
        (args.benchmark, "Development benchmark"),
        (args.known_holdout, "Known holdout"),
        (args.unknown_stress, "Unknown stress benchmark"),
    ):
        if not Path(filename).exists():
            raise FileNotFoundError(f"{label} not found: {filename}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = Tokenizer.load(args.tokenizer)
    model, checkpoint = LanguageModel.load_checkpoint(
        args.model,
        device=device,
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    development = load_benchmark(args.benchmark)
    known_holdout = load_benchmark(args.known_holdout)
    unknown_stress = load_benchmark(args.unknown_stress)

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
        for sample in known_holdout
    ]
    unknown_raw = [
        router._encode_tensor(sample.text)
        for sample in unknown_stress
    ]

    label_names = sorted({sample.label for sample in development})
    label_to_id = {
        label: index for index, label in enumerate(label_names)
    }

    development_batch = torch.stack(development_raw).to(device)
    labels = torch.tensor(
        [label_to_id[sample.label] for sample in development],
        dtype=torch.long,
        device=device,
    )

    rows = []

    print()
    print("============================================================")
    print(" LLM_SEM v0.2 Projection Architecture Sweep")
    print("============================================================")
    print()
    print("Device             :", device)
    if device.type == "cuda":
        print("GPU                :", torch.cuda.get_device_name(0))
    print("Base model         :", args.model)
    print("Checkpoint loss    :", checkpoint.get("loss"))
    print("Base LLM frozen    : True")
    print("Hidden dims        :", ", ".join(map(str, hidden_dims)))
    print("Lambdas            :", ", ".join(map(str, lambdas)))
    print("Seed               :", args.seed)
    print("Radius scale       :", args.radius_scale)
    print()

    for hidden_dim in hidden_dims:
        for preservation_lambda in lambdas:
            head, best_loss, best_epoch = train_candidate(
                development_batch,
                labels,
                hidden_dim,
                preservation_lambda,
                args,
                device,
            )

            development_projected = project_batch(
                head, development_raw, device
            )
            known_projected = project_batch(
                head, known_raw, device
            )
            unknown_projected = project_batch(
                head, unknown_raw, device
            )

            metrics = evaluate_candidate(
                development,
                development_raw,
                development_projected,
                known_holdout,
                known_projected,
                unknown_stress,
                unknown_projected,
                args,
            )

            row = {
                "hidden_dim": hidden_dim,
                "preservation_lambda": preservation_lambda,
                "best_loss": best_loss,
                "best_epoch": best_epoch,
                **metrics,
            }
            rows.append(row)

            print(
                f"hidden={hidden_dim:<3} "
                f"lambda={preservation_lambda:<4.1f} "
                f"LOO={metrics['development_loo_accuracy'] * 100:>6.2f}% "
                f"known={metrics['known_base_accuracy'] * 100:>6.2f}% "
                f"recall={metrics['known_recall'] * 100:>6.2f}% "
                f"unknown={metrics['unknown_detection_rate'] * 100:>6.2f}% "
                f"bal={metrics['balanced_accuracy'] * 100:>6.2f}% "
                f"drift={metrics['geometry_drift']:.4f}"
            )

    with Path(args.output).open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    best = max(
        rows,
        key=lambda row: (
            row["balanced_accuracy"],
            row["known_base_accuracy"],
            row["development_loo_accuracy"],
            row["unknown_detection_rate"],
            -row["geometry_drift"],
        ),
    )

    print()
    print("Best candidate")
    print("--------------")
    print("Hidden dimension       :", best["hidden_dim"])
    print("Preservation lambda    :", best["preservation_lambda"])
    print(
        "Development LOO       :",
        f"{best['development_loo_accuracy'] * 100:.2f}%",
    )
    print(
        "Known base accuracy   :",
        f"{best['known_base_accuracy'] * 100:.2f}%",
    )
    print(
        "Known Recall          :",
        f"{best['known_recall'] * 100:.2f}%",
    )
    print(
        "Unknown Detection     :",
        f"{best['unknown_detection_rate'] * 100:.2f}%",
    )
    print(
        "Balanced Accuracy     :",
        f"{best['balanced_accuracy'] * 100:.2f}%",
    )
    print("Geometry drift        :", f"{best['geometry_drift']:.6f}")
    print("Best epoch            :", best["best_epoch"])
    print()
    print("Results CSV saved:", args.output)


if __name__ == "__main__":
    main()
