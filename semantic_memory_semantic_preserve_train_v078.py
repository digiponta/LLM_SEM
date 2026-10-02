# semantic_memory_semantic_preserve_train_v078.py
#
# LLM_SEM v0.8.0
# Semantic-preserving consolidation.
#
# Goal:
#   Move selected Semantic Memory examples toward their taught semantic class
#   while preserving the pre-consolidation semantic geometry for benchmark
#   examples.
#
# Loss:
#   target classification loss
# + benchmark classification loss
# + semantic preservation loss
#
# The LM head is frozen. Training is performed on the final Transformer block
# and final normalization only, so this experiment targets parametric semantic
# memory rather than label-token generation.

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from adaptive_semantic_learning import (
    load_semantic_memory_records,
    update_memory_status,
)
from model import LanguageModel
from semantic_eval import load_benchmark
from tokenizer import Tokenizer


DEFAULT_MEMORY = "data/semantic_memory.jsonl"
DEFAULT_MODEL = "model/model-gpu-v0.4.pt"
DEFAULT_TOKENIZER = "model/tokenizer.json"
DEFAULT_BENCHMARK = "my_benchmark.csv"
DEFAULT_OUTPUT = "model/model-sem-consolidation-v080.pt"


def training_records(path: Path) -> list[dict]:
    return [
        row for row in load_semantic_memory_records(path)
        if row.get("status") == "TRAINING"
    ]


def encode_tensor(
    model: LanguageModel,
    tokenizer: Tokenizer,
    text: str,
    device: torch.device,
    *,
    alpha: float,
) -> torch.Tensor:
    ids = tokenizer.encode(text, add_bos=True, add_eos=True)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    return model.encode_semantic(
        x,
        pooling="hybrid",
        hybrid_alpha=alpha,
        normalize_hybrid=False,
    )[0]


@torch.no_grad()
def build_teacher_vectors(
    teacher: LanguageModel,
    tokenizer: Tokenizer,
    benchmark,
    device: torch.device,
    *,
    alpha: float,
):
    rows: list[tuple[str, str, torch.Tensor]] = []
    grouped: dict[str, list[torch.Tensor]] = {}
    for sample in benchmark:
        vec = encode_tensor(
            teacher, tokenizer, sample.text, device, alpha=alpha
        ).detach()
        rows.append((sample.label, sample.text, vec))
        grouped.setdefault(sample.label, []).append(vec)

    centroids = {
        label: F.normalize(torch.stack(vectors).mean(dim=0), dim=0)
        for label, vectors in grouped.items()
    }
    return rows, centroids


def logits_against_centroids(
    vector: torch.Tensor,
    centroids: dict[str, torch.Tensor],
    labels: list[str],
    temperature: float,
) -> torch.Tensor:
    v = F.normalize(vector, dim=0)
    sims = torch.stack([
        F.cosine_similarity(v, centroids[label], dim=0)
        for label in labels
    ])
    return sims / temperature


def configure_trainable(model: LanguageModel) -> list[torch.nn.Parameter]:
    for parameter in model.parameters():
        parameter.requires_grad = False

    # Constrain updates to the final semantic transformation stage.
    for parameter in model.blocks[-1].parameters():
        parameter.requires_grad = True
    for parameter in model.final_norm.parameters():
        parameter.requires_grad = True

    return [p for p in model.parameters() if p.requires_grad]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LLM_SEM v0.8.0 semantic-preserving consolidation"
    )
    p.add_argument("--memory", default=DEFAULT_MEMORY)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    p.add_argument("--benchmark", default=DEFAULT_BENCHMARK)
    p.add_argument("--output", default=DEFAULT_OUTPUT)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--alpha", type=float, default=0.35)
    p.add_argument("--temperature", type=float, default=0.08)
    p.add_argument("--target-weight", type=float, default=2.0)
    p.add_argument("--benchmark-weight", type=float, default=1.0)
    p.add_argument("--preserve-weight", type=float, default=4.0)
    p.add_argument("--allow-cpu", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    memory_path = Path(args.memory)
    model_path = Path(args.model)
    output_path = Path(args.output)

    records = training_records(memory_path)
    if not records:
        raise RuntimeError(
            "No TRAINING records. Mark the target with "
            "semantic_memory_consolidation.py training first."
        )
    if output_path.resolve() == model_path.resolve():
        raise ValueError("Output checkpoint must differ from source checkpoint.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and not args.allow_cpu:
        raise RuntimeError("CUDA unavailable. Use --allow-cpu only for testing.")

    tokenizer = Tokenizer.load(args.tokenizer)
    benchmark = load_benchmark(args.benchmark)

    teacher, source_checkpoint = LanguageModel.load_checkpoint(
        str(model_path), device=device
    )
    student, _ = LanguageModel.load_checkpoint(str(model_path), device=device)

    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    trainable = configure_trainable(student)
    student.train()

    teacher_rows, centroids = build_teacher_vectors(
        teacher,
        tokenizer,
        benchmark,
        device,
        alpha=args.alpha,
    )

    labels = sorted(centroids)
    label_to_index = {label: i for i, label in enumerate(labels)}

    for row in records:
        if str(row["label"]) not in label_to_index:
            raise ValueError(
                f"Semantic Memory label not in benchmark: {row['label']}"
            )

    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.learning_rate,
        weight_decay=0.01,
    )

    print("=" * 92)
    print(" LLM_SEM v0.8.0 Semantic-Preserving Consolidation")
    print("=" * 92)
    print("Device             :", device)
    if device.type == "cuda":
        print("GPU                :", torch.cuda.get_device_name(0))
    print("Source checkpoint  :", model_path)
    print("Source loss        :", source_checkpoint.get("loss"))
    print("Output checkpoint  :", output_path)
    print("TRAINING records   :", len(records))
    print("Benchmark samples  :", len(benchmark))
    print("Semantic labels    :", ", ".join(labels))
    print("Trainable stage    : final Transformer block + final_norm")
    print("Trainable params   :", sum(p.numel() for p in trainable))
    print("Epochs             :", args.epochs)
    print("Learning rate      :", args.learning_rate)
    print("Target weight      :", args.target_weight)
    print("Benchmark weight   :", args.benchmark_weight)
    print("Preserve weight    :", args.preserve_weight)
    print()

    last_total = None
    for epoch in range(1, max(1, args.epochs) + 1):
        optimizer.zero_grad(set_to_none=True)

        target_losses: list[torch.Tensor] = []
        for row in records:
            text = str(row["text"])
            expected = str(row["label"])
            vec = encode_tensor(
                student, tokenizer, text, device, alpha=args.alpha
            )
            logits = logits_against_centroids(
                vec, centroids, labels, args.temperature
            ).unsqueeze(0)
            target = torch.tensor(
                [label_to_index[expected]], dtype=torch.long, device=device
            )
            target_losses.append(F.cross_entropy(logits, target))

        benchmark_losses: list[torch.Tensor] = []
        preserve_losses: list[torch.Tensor] = []
        for expected, text, teacher_vec in teacher_rows:
            student_vec = encode_tensor(
                student, tokenizer, text, device, alpha=args.alpha
            )
            logits = logits_against_centroids(
                student_vec, centroids, labels, args.temperature
            ).unsqueeze(0)
            target = torch.tensor(
                [label_to_index[expected]], dtype=torch.long, device=device
            )
            benchmark_losses.append(F.cross_entropy(logits, target))

            preserve_losses.append(
                1.0 - F.cosine_similarity(
                    student_vec,
                    teacher_vec,
                    dim=0,
                )
            )

        target_loss = torch.stack(target_losses).mean()
        benchmark_loss = torch.stack(benchmark_losses).mean()
        preserve_loss = torch.stack(preserve_losses).mean()

        total = (
            args.target_weight * target_loss
            + args.benchmark_weight * benchmark_loss
            + args.preserve_weight * preserve_loss
        )

        total.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        last_total = float(total.item())

        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(
                f"Epoch {epoch:>3}/{args.epochs} "
                f"total={float(total.item()):.6f} "
                f"target={float(target_loss.item()):.6f} "
                f"benchmark={float(benchmark_loss.item()):.6f} "
                f"preserve={float(preserve_loss.item()):.6f}"
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    student.save_checkpoint(
        str(output_path),
        optimizer=optimizer,
        epoch=max(1, args.epochs),
        loss=last_total,
    )

    model_version = output_path.name
    for row in records:
        update_memory_status(
            memory_path,
            str(row["text"]),
            "VALIDATING",
            model_version=model_version,
            verified=False,
        )

    print()
    print("Semantic-preserving training completed.")
    print("Saved checkpoint  :", output_path)
    print("Memory transition : TRAINING -> VALIDATING")
    print("Next:")
    print(
        "  python semantic_memory_batch_validate_v080.py "
        "--candidate model/model-sem-consolidation-v080.pt"
    )


if __name__ == "__main__":
    main()
