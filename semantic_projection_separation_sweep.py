# semantic_projection_separation_sweep.py
#
# Sweep inter-class separation hyperparameters for LLM_SEM v0.2.
#
# The base LLM is frozen and encoded only once. For each combination of
# separation_lambda and separation_margin, a fresh projection head is trained
# with the same seed and evaluated on development + known holdout data.
#
# Reports:
#   - best training loss / epoch
#   - development LOO routing accuracy
#   - within / between similarity and semantic margin
#   - known holdout routing accuracy
#   - improved / degraded known samples vs raw semantic space
#   - optional unknown detection rate
#
# Default grid:
#   lambda = 0.25,0.5,1.0,2.0,5.0
#   margin = 0.02,0.05,0.10,0.20

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F

from model import LanguageModel
from semantic_eval import LabeledSentence, load_benchmark
from semantic_projection import SemanticProjectionHead
from semantic_projection_eval import (
    centroids,
    loo_accuracy,
    pairwise_margin,
)
from semantic_projection_train import (
    inter_class_separation_loss,
    supervised_contrastive_loss,
)
from semantic_router import (
    DEFAULT_ALPHA,
    DEFAULT_MODEL,
    DEFAULT_TOKENIZER,
    SemanticRouter,
)
from tokenizer import Tokenizer


DEFAULT_LAMBDAS = "0.25,0.5,1.0,2.0,5.0"
DEFAULT_MARGINS = "0.02,0.05,0.10,0.20"
DEFAULT_OUTPUT_CSV = "semantic_projection_separation_sweep.csv"
DEFAULT_BEST_OUTPUT = "model/semantic-projection-separation-best.pt"


@dataclass
class TrainResult:
    head: SemanticProjectionHead
    best_loss: float
    best_epoch: int
    final_contrastive: float
    final_preservation: float
    final_separation: float


def parse_float_list(value: str) -> List[float]:
    result = []
    for item in value.split(","):
        item = item.strip()
        if item:
            result.append(float(item))
    if not result:
        raise argparse.ArgumentTypeError("Expected at least one number.")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep semantic projection separation lambda/margin."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--known-holdout", required=True)
    parser.add_argument(
        "--unknown-holdout",
        default=None,
        help="Optional unknown-set CSV for unknown detection comparison.",
    )
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
    parser.add_argument("--seed", type=int, default=42)
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
    parser.add_argument("--output-csv", default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--best-output", default=DEFAULT_BEST_OUTPUT)
    return parser.parse_args()


def cosine_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    return 1.0 - float(F.cosine_similarity(a, b, dim=0).item())


def nearest(
    vector: torch.Tensor,
    centers: Dict[str, torch.Tensor],
) -> str:
    return min(
        centers,
        key=lambda label: cosine_distance(vector, centers[label]),
    )


def known_predictions(
    development_samples: Sequence[LabeledSentence],
    development_vectors: Sequence[torch.Tensor],
    known_samples: Sequence[LabeledSentence],
    known_vectors: Sequence[torch.Tensor],
) -> List[str]:
    centers = centroids(development_samples, development_vectors)
    return [nearest(vector, centers) for vector in known_vectors]


def known_accuracy(
    samples: Sequence[LabeledSentence],
    predictions: Sequence[str],
) -> float:
    return sum(
        int(sample.label == predicted)
        for sample, predicted in zip(samples, predictions)
    ) / len(samples)


def prediction_changes(
    samples: Sequence[LabeledSentence],
    before: Sequence[str],
    after: Sequence[str],
) -> Tuple[int, int, int]:
    improved = 0
    degraded = 0
    changed = 0

    for sample, b, a in zip(samples, before, after):
        if b != a:
            changed += 1
        before_correct = b == sample.label
        after_correct = a == sample.label
        improved += int((not before_correct) and after_correct)
        degraded += int(before_correct and (not after_correct))

    return improved, degraded, changed


def unknown_detection_rate(
    development_samples: Sequence[LabeledSentence],
    development_vectors: Sequence[torch.Tensor],
    unknown_vectors: Sequence[torch.Tensor],
) -> float:
    # For sweep comparison, use the nearest-centroid-distance threshold
    # derived from the maximum in-class development centroid distance.
    centers = centroids(development_samples, development_vectors)

    class_max: Dict[str, float] = {}
    for label, center in centers.items():
        distances = [
            cosine_distance(vector, center)
            for sample, vector in zip(
                development_samples,
                development_vectors,
            )
            if sample.label == label
        ]
        class_max[label] = max(distances)

    detected = 0
    for vector in unknown_vectors:
        ranked = sorted(
            (
                (label, cosine_distance(vector, center))
                for label, center in centers.items()
            ),
            key=lambda item: item[1],
        )
        label, distance = ranked[0]
        detected += int(distance > class_max[label])

    return detected / len(unknown_vectors)


def train_one(
    base_vectors: torch.Tensor,
    labels: torch.Tensor,
    input_dim: int,
    hidden_dim: int,
    epochs: int,
    lr: float,
    temperature: float,
    preservation_lambda: float,
    separation_lambda: float,
    separation_margin: float,
    patience: int,
    min_delta: float,
    weight_decay: float,
    dropout: float,
    seed: int,
    device: torch.device,
) -> TrainResult:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    head = SemanticProjectionHead(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=input_dim,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    normalized_base = F.normalize(base_vectors, p=2, dim=-1)
    best_loss = float("inf")
    best_epoch = 0
    best_state = None
    epochs_without_improvement = 0
    last_values = (float("nan"), float("nan"), float("nan"))

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
        separation = inter_class_separation_loss(
            projected,
            labels,
            margin=separation_margin,
        )
        loss = (
            contrastive
            + preservation_lambda * preservation
            + separation_lambda * separation
        )

        loss.backward()
        optimizer.step()

        value = float(loss.item())
        last_values = (
            float(contrastive.item()),
            float(preservation.item()),
            float(separation.item()),
        )

        if value < best_loss - min_delta:
            best_loss = value
            best_epoch = epoch
            epochs_without_improvement = 0
            best_state = {
                key: tensor.detach().cpu().clone()
                for key, tensor in head.state_dict().items()
            }
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            break

    if best_state is not None:
        head.load_state_dict(best_state)
    head.eval()

    return TrainResult(
        head=head,
        best_loss=best_loss,
        best_epoch=best_epoch,
        final_contrastive=last_values[0],
        final_preservation=last_values[1],
        final_separation=last_values[2],
    )


def project(
    head: SemanticProjectionHead,
    vectors: Sequence[torch.Tensor],
    device: torch.device,
) -> List[torch.Tensor]:
    batch = torch.stack(vectors).to(device)
    with torch.no_grad():
        result = head(batch, normalize=True).cpu()
    return [row for row in result]


def write_csv(path: str, rows: Sequence[Dict[str, object]]) -> None:
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
    raw_within, raw_between, raw_margin = pairwise_margin(
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

    print()
    print("============================================================")
    print(" LLM_SEM Separation Sweep")
    print("============================================================")
    print()
    print("Device             :", device)
    if device.type == "cuda":
        print("GPU                :", torch.cuda.get_device_name(0))
    print("Base checkpoint loss:", checkpoint.get("loss"))
    print("Development samples:", len(development))
    print("Known holdout      :", len(known))
    print("Unknown holdout    :", len(unknown))
    print("Grid combinations  :", len(combinations))
    print("Raw LOO accuracy   :", f"{raw_loo * 100:.2f}%")
    print("Raw known accuracy :", f"{raw_known_accuracy * 100:.2f}%")
    print("Raw semantic margin:", f"{raw_margin:.6f}")
    if unknown_raw:
        print(
            "Raw unknown detect :",
            f"{raw_unknown_detection * 100:.2f}%",
        )
    print()

    rows: List[Dict[str, object]] = []
    best_score = None
    best_head = None
    best_row = None

    for index, (sep_lambda, sep_margin) in enumerate(
        combinations,
        start=1,
    ):
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
            seed=args.seed,
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

        row: Dict[str, object] = {
            "separation_lambda": sep_lambda,
            "separation_margin": sep_margin,
            "best_loss": result.best_loss,
            "best_epoch": result.best_epoch,
            "loo_accuracy": projected_loo,
            "known_accuracy": projected_known_accuracy,
            "known_delta": projected_known_accuracy - raw_known_accuracy,
            "within_similarity": within,
            "between_similarity": between,
            "semantic_margin": semantic_margin,
            "margin_delta": semantic_margin - raw_margin,
            "improved": improved,
            "degraded": degraded,
            "net_improved": improved - degraded,
            "prediction_changes": changed,
            "unknown_detection": projected_unknown_detection,
        }
        rows.append(row)

        # Selection priority:
        # 1) known holdout accuracy
        # 2) semantic margin
        # 3) LOO accuracy
        # 4) fewer degraded samples
        score = (
            projected_known_accuracy,
            semantic_margin,
            projected_loo,
            -degraded,
        )
        if best_score is None or score > best_score:
            best_score = score
            best_head = result.head
            best_row = row

        unknown_text = (
            f" unknown={projected_unknown_detection * 100:6.2f}%"
            if unknown_projected
            else ""
        )
        print(
            f"[{index:>2}/{len(combinations)}] "
            f"lambda={sep_lambda:<5g} margin={sep_margin:<5g} "
            f"known={projected_known_accuracy * 100:6.2f}% "
            f"LOO={projected_loo * 100:6.2f}% "
            f"sem-margin={semantic_margin:.6f} "
            f"net={improved - degraded:+d}"
            f"{unknown_text}"
        )

    write_csv(args.output_csv, rows)

    ranked = sorted(
        rows,
        key=lambda row: (
            float(row["known_accuracy"]),
            float(row["semantic_margin"]),
            float(row["loo_accuracy"]),
            -int(row["degraded"]),
        ),
        reverse=True,
    )

    print()
    print("Top configurations")
    print("==================")
    print()
    print(
        f"{'Rank':>4} {'Lambda':>8} {'Margin':>8} "
        f"{'Known':>9} {'LOO':>9} {'SemMargin':>11} "
        f"{'Net':>5}"
    )
    print("-" * 66)
    for rank, row in enumerate(ranked[:10], start=1):
        print(
            f"{rank:>4d} "
            f"{float(row['separation_lambda']):>8.3f} "
            f"{float(row['separation_margin']):>8.3f} "
            f"{float(row['known_accuracy']) * 100:>8.2f}% "
            f"{float(row['loo_accuracy']) * 100:>8.2f}% "
            f"{float(row['semantic_margin']):>11.6f} "
            f"{int(row['net_improved']):>+5d}"
        )

    if best_head is not None and best_row is not None:
        best_head.save_checkpoint(
            args.best_output,
            benchmark=args.benchmark,
            base_model=args.model,
            base_checkpoint_loss=checkpoint.get("loss"),
            pooling="hybrid",
            hybrid_alpha=args.alpha,
            temperature=args.temperature,
            preservation_lambda=args.preservation_lambda,
            separation_lambda=float(best_row["separation_lambda"]),
            separation_margin=float(best_row["separation_margin"]),
            labels=label_names,
            best_loss=float(best_row["best_loss"]),
            best_epoch=int(best_row["best_epoch"]),
            epochs=args.epochs,
            seed=args.seed,
            sweep_selected=True,
        )

        print()
        print("Best configuration")
        print("==================")
        print()
        print(
            "Separation lambda :",
            best_row["separation_lambda"],
        )
        print(
            "Separation margin :",
            best_row["separation_margin"],
        )
        print(
            "Known accuracy    :",
            f"{float(best_row['known_accuracy']) * 100:.2f}%",
        )
        print(
            "LOO accuracy      :",
            f"{float(best_row['loo_accuracy']) * 100:.2f}%",
        )
        print(
            "Semantic margin   :",
            f"{float(best_row['semantic_margin']):.6f}",
        )
        print(
            "Net improved      :",
            int(best_row["net_improved"]),
        )
        print("Best checkpoint   :", args.best_output)

    print()
    print("Sweep CSV saved   :", args.output_csv)


if __name__ == "__main__":
    main()
