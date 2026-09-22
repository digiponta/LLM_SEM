# semantic_projection_train.py
#
# Train only the semantic projection head using supervised contrastive loss.
# The pretrained LLM remains frozen.

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F

from model import LanguageModel
from semantic_eval import load_benchmark
from semantic_projection import (
    DEFAULT_PROJECTION,
    SemanticProjectionHead,
)
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
        description="Train the LLM_SEM semantic projection head."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--benchmark", default=DEFAULT_BENCHMARK)
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
    parser.add_argument("--output", default=DEFAULT_PROJECTION)
    return parser.parse_args()


def supervised_contrastive_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    if embeddings.dim() != 2:
        raise ValueError("embeddings must have shape [batch, dim].")
    if labels.dim() != 1:
        raise ValueError("labels must have shape [batch].")
    if temperature <= 0.0:
        raise ValueError("temperature must be > 0.")

    z = F.normalize(embeddings, p=2, dim=-1)
    logits = torch.matmul(z, z.T) / temperature

    batch = z.size(0)
    eye = torch.eye(batch, dtype=torch.bool, device=z.device)
    logits = logits.masked_fill(eye, float("-inf"))

    positive_mask = labels.unsqueeze(0) == labels.unsqueeze(1)
    positive_mask = positive_mask & ~eye

    log_denominator = torch.logsumexp(logits, dim=1)
    log_prob = logits - log_denominator.unsqueeze(1)

    positive_count = positive_mask.sum(dim=1)
    if torch.any(positive_count == 0):
        raise ValueError(
            "Each class must contain at least two samples "
            "for supervised contrastive training."
        )

    positive_log_prob = torch.where(
        positive_mask,
        log_prob,
        torch.zeros_like(log_prob),
    ).sum(dim=1) / positive_count

    return -positive_log_prob.mean()


def main() -> None:
    args = parse_args()

    for filename, label in (
        (args.model, "Model checkpoint"),
        (args.tokenizer, "Tokenizer"),
        (args.benchmark, "Benchmark"),
    ):
        if not Path(filename).exists():
            raise FileNotFoundError(f"{label} not found: {filename}")

    if args.epochs <= 0:
        raise ValueError("epochs must be > 0.")
    if args.lr <= 0.0:
        raise ValueError("lr must be > 0.")
    if args.temperature <= 0.0:
        raise ValueError("temperature must be > 0.")
    if args.hidden_dim <= 0:
        raise ValueError("hidden-dim must be > 0.")
    if args.preservation_lambda < 0.0:
        raise ValueError("preservation-lambda must be >= 0.")
    if args.patience <= 0:
        raise ValueError("patience must be > 0.")
    if args.min_delta < 0.0:
        raise ValueError("min-delta must be >= 0.")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

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

    counts = Counter(sample.label for sample in samples)
    for label, count in counts.items():
        if count < 2:
            raise ValueError(
                f"Class {label!r} needs at least two samples."
            )

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

    head = SemanticProjectionHead(
        input_dim=base_vectors.size(1),
        hidden_dim=args.hidden_dim,
        output_dim=base_vectors.size(1),
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    print()
    print("LLM_SEM Semantic Projection Training")
    print("------------------------------------")
    print("Device            :", device)
    if device.type == "cuda":
        print("GPU               :", torch.cuda.get_device_name(0))
    print("Base model        :", args.model)
    print("Checkpoint loss   :", checkpoint.get("loss"))
    print("Benchmark         :", args.benchmark)
    print("Samples           :", len(samples))
    print("Classes           :", len(label_names))
    print("Input dimension   :", base_vectors.size(1))
    print("Projection hidden :", args.hidden_dim)
    print("Output dimension  :", base_vectors.size(1))
    print("Epochs            :", args.epochs)
    print("Learning rate     :", args.lr)
    print("Temperature       :", args.temperature)
    print("Preservation lambda:", args.preservation_lambda)
    print("Early-stop patience:", args.patience)
    print("Early-stop min delta:", args.min_delta)
    print("Base LLM frozen   : True")
    print()

    best_loss = float("inf")
    best_state = None
    best_epoch = 0
    epochs_without_improvement = 0
    normalized_base = F.normalize(base_vectors, p=2, dim=-1)

    for epoch in range(1, args.epochs + 1):
        head.train()
        optimizer.zero_grad(set_to_none=True)

        projected = head(base_vectors, normalize=True)
        contrastive_loss = supervised_contrastive_loss(
            projected,
            labels,
            temperature=args.temperature,
        )
        preservation_loss = (
            1.0
            - F.cosine_similarity(
                projected,
                normalized_base,
                dim=-1,
            )
        ).mean()
        loss = (
            contrastive_loss
            + args.preservation_lambda * preservation_loss
        )
        loss.backward()
        optimizer.step()

        value = float(loss.item())
        contrastive_value = float(contrastive_loss.item())
        preservation_value = float(preservation_loss.item())
        if value < best_loss - args.min_delta:
            best_loss = value
            best_epoch = epoch
            epochs_without_improvement = 0
            best_state = {
                key: tensor.detach().cpu().clone()
                for key, tensor in head.state_dict().items()
            }
        else:
            epochs_without_improvement += 1

        if (
            epoch == 1
            or epoch == args.epochs
            or epoch % max(1, args.epochs // 10) == 0
        ):
            print(
                f"Epoch {epoch:>5}/{args.epochs:<5} "
                f"loss={value:.6f} "
                f"contrastive={contrastive_value:.6f} "
                f"preserve={preservation_value:.6f} "
                f"best={best_loss:.6f}"
            )

        if epochs_without_improvement >= args.patience:
            print(
                f"Early stopping at epoch {epoch}; "
                f"best epoch={best_epoch}"
            )
            break

    if best_state is not None:
        head.load_state_dict(best_state)

    head.save_checkpoint(
        args.output,
        benchmark=args.benchmark,
        base_model=args.model,
        base_checkpoint_loss=checkpoint.get("loss"),
        pooling="hybrid",
        hybrid_alpha=args.alpha,
        temperature=args.temperature,
        preservation_lambda=args.preservation_lambda,
        patience=args.patience,
        min_delta=args.min_delta,
        labels=label_names,
        best_loss=best_loss,
        best_epoch=best_epoch,
        epochs=args.epochs,
        seed=args.seed,
    )

    print()
    print("Projection checkpoint saved:", args.output)
    print("Best training loss          :", f"{best_loss:.6f}")
    print("Best epoch                  :", best_epoch)


if __name__ == "__main__":
    main()
