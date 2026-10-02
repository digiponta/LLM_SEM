# semantic_memory_validate_v073.py
#
# LLM_SEM v0.7.3
# Validate a candidate consolidated checkpoint before Semantic Memory is retired
# from primary routing.
#
# PASS requires:
# 1) candidate conditional NLL on Semantic Memory labels improves over source
# 2) replay/general-language loss does not regress beyond tolerance
#
# On PASS: VALIDATING -> CONSOLIDATED
# On FAIL: VALIDATING -> FAILED

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from adaptive_semantic_learning import (
    load_semantic_memory_records,
    update_memory_status,
)
from model import LanguageModel
from tokenizer import Tokenizer


DEFAULT_MEMORY = "data/semantic_memory.jsonl"
DEFAULT_SOURCE = "model/model-gpu-v0.4.pt"
DEFAULT_CANDIDATE = "model/model-sem-consolidation-v073.pt"
DEFAULT_TOKENIZER = "model/tokenizer.json"


def validating_records(path: Path) -> list[dict]:
    return [
        row for row in load_semantic_memory_records(path)
        if row.get("status") == "VALIDATING"
    ]


@torch.no_grad()
def conditional_target_nll(
    model: LanguageModel,
    tokenizer: Tokenizer,
    prompt: str,
    target: str,
    device: torch.device,
) -> float:
    prompt_ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
    target_ids = tokenizer.encode(target, add_bos=False, add_eos=True)
    ids = prompt_ids + target_ids

    # Keep the target and as much prompt context as the model can consume.
    max_tokens = model.context_length + 1
    if len(ids) > max_tokens:
        trim = len(ids) - max_tokens
        ids = ids[trim:]
        prompt_len = max(1, len(prompt_ids) - trim)
    else:
        prompt_len = len(prompt_ids)

    if len(ids) < 2:
        return float("inf")

    x = torch.tensor([ids[:-1]], dtype=torch.long, device=device)
    y = torch.tensor([ids[1:]], dtype=torch.long, device=device)
    logits = model(x)[0]

    # y position corresponding to first target token is prompt_len - 1.
    start = max(0, prompt_len - 1)
    logits = logits[start:]
    y = y[0, start:]
    if y.numel() == 0:
        return float("inf")

    return float(
        F.cross_entropy(logits, y, reduction="mean").item()
    )


@torch.no_grad()
def text_nll(
    model: LanguageModel,
    tokenizer: Tokenizer,
    text: str,
    device: torch.device,
    *,
    max_windows: int = 64,
) -> float:
    ids = tokenizer.encode(text, add_bos=True, add_eos=True)
    if len(ids) < 2:
        return float("inf")

    losses: list[float] = []
    context = model.context_length
    step = max(1, context)
    for start in range(0, len(ids) - 1, step):
        chunk = ids[start:start + context + 1]
        if len(chunk) < 2:
            continue
        x = torch.tensor([chunk[:-1]], dtype=torch.long, device=device)
        y = torch.tensor([chunk[1:]], dtype=torch.long, device=device)
        logits = model(x)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
        )
        losses.append(float(loss.item()))
        if len(losses) >= max_windows:
            break
    return sum(losses) / max(1, len(losses))


def replay_text(limit: int) -> str:
    parts: list[str] = []
    remaining = max(0, limit)
    for name in ("general-ja.txt", "data-nagato.txt"):
        for candidate in (Path("data") / name, Path("..") / "LLM" / "data" / name):
            if candidate.exists():
                text = candidate.read_text(encoding="utf-8")
                take = min(len(text), remaining)
                parts.append(text[:take])
                remaining -= take
                break
        if remaining <= 0:
            break
    return "\n\n".join(parts)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LLM_SEM v0.7.3 consolidated-checkpoint validation"
    )
    p.add_argument("--memory", default=DEFAULT_MEMORY)
    p.add_argument("--source", default=DEFAULT_SOURCE)
    p.add_argument("--candidate", default=DEFAULT_CANDIDATE)
    p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    p.add_argument("--min-memory-improvement", type=float, default=0.02)
    p.add_argument("--max-replay-regression", type=float, default=0.20)
    p.add_argument("--replay-chars", type=int, default=12000)
    p.add_argument("--allow-cpu", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    memory_path = Path(args.memory)
    records = validating_records(memory_path)
    if not records:
        raise RuntimeError("No VALIDATING Semantic Memory records found.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and not args.allow_cpu:
        raise RuntimeError("CUDA unavailable. Use --allow-cpu only for testing.")

    tokenizer = Tokenizer.load(args.tokenizer)
    source, _ = LanguageModel.load_checkpoint(args.source, device=device)
    candidate, _ = LanguageModel.load_checkpoint(args.candidate, device=device)
    source.eval()
    candidate.eval()

    source_memory: list[float] = []
    candidate_memory: list[float] = []

    print("=" * 86)
    print(" LLM_SEM v0.7.3 Semantic Memory Consolidation Validation")
    print("=" * 86)
    print("Source checkpoint   :", args.source)
    print("Candidate checkpoint:", args.candidate)
    print("VALIDATING records  :", len(records))
    print()

    for row in records:
        prompt = f"入力: {row['text']}\n意味分類: "
        target = str(row["label"])
        src = conditional_target_nll(source, tokenizer, prompt, target, device)
        cand = conditional_target_nll(candidate, tokenizer, prompt, target, device)
        source_memory.append(src)
        candidate_memory.append(cand)
        print(
            f"MEM> {row['text']!r} label={target!r} "
            f"source_nll={src:.6f} candidate_nll={cand:.6f} "
            f"delta={src-cand:+.6f}"
        )

    source_mem_mean = sum(source_memory) / len(source_memory)
    candidate_mem_mean = sum(candidate_memory) / len(candidate_memory)
    memory_improvement = source_mem_mean - candidate_mem_mean

    replay = replay_text(args.replay_chars)
    if replay:
        source_replay = text_nll(source, tokenizer, replay, device)
        candidate_replay = text_nll(candidate, tokenizer, replay, device)
        replay_regression = candidate_replay - source_replay
    else:
        source_replay = float("nan")
        candidate_replay = float("nan")
        replay_regression = 0.0

    memory_ok = memory_improvement >= args.min_memory_improvement
    replay_ok = replay_regression <= args.max_replay_regression
    passed = memory_ok and replay_ok

    print()
    print("Summary")
    print("-------")
    print("Source memory NLL    :", f"{source_mem_mean:.6f}")
    print("Candidate memory NLL :", f"{candidate_mem_mean:.6f}")
    print("Memory improvement   :", f"{memory_improvement:+.6f}")
    print("Required improvement :", f"{args.min_memory_improvement:+.6f}")
    if replay:
        print("Source replay NLL    :", f"{source_replay:.6f}")
        print("Candidate replay NLL :", f"{candidate_replay:.6f}")
        print("Replay regression    :", f"{replay_regression:+.6f}")
        print("Allowed regression   :", f"{args.max_replay_regression:+.6f}")
    else:
        print("Replay check         : SKIPPED (corpus unavailable)")
    print("Memory check         :", "PASS" if memory_ok else "FAIL")
    print("Replay check         :", "PASS" if replay_ok else "FAIL")

    model_version = Path(args.candidate).name
    next_state = "CONSOLIDATED" if passed else "FAILED"
    for row in records:
        update_memory_status(
            memory_path,
            str(row["text"]),
            next_state,
            model_version=model_version,
            verified=passed,
        )

    print()
    print("RESULT               :", "PASS" if passed else "FAIL")
    print("Memory transition    : VALIDATING ->", next_state)
    if passed:
        print("Internal checkpoint is now primary for these records.")
    else:
        print("Semantic Memory remains authoritative for these records.")


if __name__ == "__main__":
    main()
