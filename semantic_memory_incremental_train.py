# semantic_memory_incremental_train.py
#
# LLM_SEM v0.7.3
# Incrementally distill TRAINING Semantic Memory records into a new checkpoint.
#
# Safety properties:
# - never overwrites the source checkpoint
# - only TRAINING records are used as new knowledge
# - mixes replay text to reduce catastrophic forgetting risk
# - moves records to VALIDATING only after checkpoint save succeeds

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from adaptive_semantic_learning import (
    load_semantic_memory_records,
    update_memory_status,
)
from dataset import TokenWindowDataset
from model import LanguageModel
from tokenizer import Tokenizer
from train import Trainer


DEFAULT_MEMORY = "data/semantic_memory.jsonl"
DEFAULT_MODEL = "model/model-gpu-v0.4.pt"
DEFAULT_TOKENIZER = "model/tokenizer.json"
DEFAULT_OUTPUT = "model/model-sem-consolidation-v073.pt"


def find_replay_files() -> list[Path]:
    names = ("general-ja.txt", "data-nagato.txt")
    result: list[Path] = []
    for name in names:
        for candidate in (Path("data") / name, Path("..") / "LLM" / "data" / name):
            if candidate.exists():
                result.append(candidate)
                break
    return result


def training_records(path: Path) -> list[dict]:
    return [
        row for row in load_semantic_memory_records(path)
        if row.get("status") == "TRAINING"
    ]


def build_training_text(
    records: list[dict],
    *,
    replay_chars: int,
    repeat_memory: int,
) -> str:
    if not records:
        raise ValueError("No TRAINING Semantic Memory records found.")

    memory_lines: list[str] = []
    for row in records:
        text = str(row["text"]).strip()
        label = str(row["label"]).strip()
        source = str(row.get("source", "semantic-memory")).strip()
        # Current Semantic Memory stores a taught utterance and semantic label.
        # Preserve both in a natural-text format suitable for next-token training.
        memory_lines.extend([
            f"入力: {text}\n意味分類: {label}\n",
            f"{text}\nこの内容の意味分類は{label}です。\n",
            f"Semantic Memory: {text} -> {label}\n",
        ])

    memory_text = "\n".join(memory_lines)
    memory_text = (memory_text + "\n") * max(1, int(repeat_memory))

    replay_parts: list[str] = []
    remaining = max(0, int(replay_chars))
    for path in find_replay_files():
        if remaining <= 0:
            break
        text = path.read_text(encoding="utf-8")
        take = min(len(text), remaining)
        replay_parts.append(text[:take])
        remaining -= take

    replay_text = "\n\n".join(replay_parts)
    if replay_text:
        return memory_text + "\n\n" + replay_text
    return memory_text


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LLM_SEM v0.7.3 Semantic Memory incremental training"
    )
    p.add_argument("--memory", default=DEFAULT_MEMORY)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    p.add_argument("--output", default=DEFAULT_OUTPUT)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--learning-rate", type=float, default=5e-5)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--replay-chars", type=int, default=20000)
    p.add_argument("--repeat-memory", type=int, default=40)
    p.add_argument("--max-samples", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow CPU training; CUDA is required by default.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    memory_path = Path(args.memory)
    model_path = Path(args.model)
    tokenizer_path = Path(args.tokenizer)
    output_path = Path(args.output)

    if not model_path.exists():
        raise FileNotFoundError(model_path)
    if not tokenizer_path.exists():
        raise FileNotFoundError(tokenizer_path)
    if output_path.resolve() == model_path.resolve():
        raise ValueError("Output checkpoint must differ from source checkpoint.")

    records = training_records(memory_path)
    if not records:
        raise RuntimeError(
            "No TRAINING records. Mark entries with "
            "semantic_memory_consolidation.py training first."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and not args.allow_cpu:
        raise RuntimeError("CUDA unavailable. Use --allow-cpu only for testing.")

    torch.manual_seed(args.seed)

    tokenizer = Tokenizer.load(str(tokenizer_path))
    model, checkpoint = LanguageModel.load_checkpoint(str(model_path), device=device)

    text = build_training_text(
        records,
        replay_chars=args.replay_chars,
        repeat_memory=args.repeat_memory,
    )
    token_ids = tokenizer.encode(text, add_bos=True, add_eos=True)

    unknown_count = sum(1 for token_id in token_ids if token_id == tokenizer.unk_id)
    unknown_ratio = unknown_count / max(1, len(token_ids))

    if len(token_ids) <= model.context_length:
        raise RuntimeError(
            f"Training corpus too short: {len(token_ids)} tokens "
            f"for context length {model.context_length}."
        )

    dataset = TokenWindowDataset(
        token_ids=token_ids,
        context_length=model.context_length,
        max_samples=min(
            max(1, args.max_samples),
            max(1, len(token_ids) - model.context_length),
        ),
        seed=args.seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=max(1, args.batch_size),
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    print("=" * 78)
    print(" LLM_SEM v0.7.3 Semantic Memory Incremental Training")
    print("=" * 78)
    print("Device            :", device)
    if device.type == "cuda":
        print("GPU               :", torch.cuda.get_device_name(0))
    print("Source checkpoint :", model_path)
    print("Source loss       :", checkpoint.get("loss"))
    print("Output checkpoint :", output_path)
    print("TRAINING records  :", len(records))
    print("Training tokens   :", len(token_ids))
    print("Unknown tokens    :", unknown_count)
    print("Unknown ratio     :", f"{unknown_ratio:.4%}")
    print("Replay chars      :", args.replay_chars)
    print("Memory repeat     :", args.repeat_memory)
    print("Samples           :", len(dataset))
    print("Epochs            :", args.epochs)
    print("Learning rate     :", args.learning_rate)
    print()

    trainer = Trainer(
        model=model,
        device=device,
        learning_rate=args.learning_rate,
    )
    history = trainer.train(loader, epochs=max(1, args.epochs))
    final_loss = history[-1] if history else None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_checkpoint(
        str(output_path),
        optimizer=trainer.optimizer,
        epoch=max(1, args.epochs),
        loss=final_loss,
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
    print("Incremental training completed.")
    print("Loss history      :", history)
    print("Saved checkpoint  :", output_path)
    print("Memory transition : TRAINING -> VALIDATING")
    print("Next step         : validate the new checkpoint before PASS.")


if __name__ == "__main__":
    main()
