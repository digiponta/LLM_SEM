# semantic_memory_consolidation.py
#
# LLM_SEM v0.7.2 Semantic Memory Consolidation backend.
#
# Semantic Memory remains authoritative while an entry is ACTIVE, TRAINING,
# VALIDATING, or FAILED. Only a successfully verified entry becomes
# CONSOLIDATED, after which the internal/base model becomes primary and the
# semantic record is retained as a backup/audit record.

from __future__ import annotations

import argparse
from pathlib import Path

from adaptive_semantic_learning import (
    MEMORY_ACTIVE_STATES,
    load_semantic_memory_records,
    memory_status_counts,
    update_memory_status,
)

DEFAULT_MEMORY = "data/semantic_memory.jsonl"


def begin_training(path: Path, text: str, model_version: str = "") -> bool:
    return update_memory_status(
        path,
        text,
        "TRAINING",
        model_version=model_version,
        verified=False,
    )


def begin_validation(path: Path, text: str, model_version: str = "") -> bool:
    return update_memory_status(
        path,
        text,
        "VALIDATING",
        model_version=model_version,
        verified=False,
    )


def complete_validation(
    path: Path,
    text: str,
    *,
    passed: bool,
    model_version: str = "",
) -> bool:
    if passed:
        return update_memory_status(
            path,
            text,
            "CONSOLIDATED",
            model_version=model_version,
            verified=True,
        )
    return update_memory_status(
        path,
        text,
        "FAILED",
        model_version=model_version,
        verified=False,
    )


def print_status(path: Path) -> None:
    counts = memory_status_counts(path)
    print("Semantic Memory Consolidation")
    print("-----------------------------")
    print("Memory:", path)
    for state in ("ACTIVE", "TRAINING", "VALIDATING", "FAILED", "CONSOLIDATED"):
        print(f"{state:<13}: {counts.get(state, 0)}")
    print()
    print("Authoritative memory states:", ", ".join(sorted(MEMORY_ACTIVE_STATES)))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LLM_SEM Semantic Memory Consolidation backend")
    p.add_argument("--memory", default=DEFAULT_MEMORY)
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("status")

    for name in ("training", "validating"):
        q = sub.add_parser(name)
        q.add_argument("text")
        q.add_argument("--model-version", default="")

    q = sub.add_parser("pass")
    q.add_argument("text")
    q.add_argument("--model-version", default="")

    q = sub.add_parser("fail")
    q.add_argument("text")
    q.add_argument("--model-version", default="")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    path = Path(args.memory)

    if args.command == "status":
        print_status(path)
        return

    if args.command == "training":
        changed = begin_training(path, args.text, args.model_version)
    elif args.command == "validating":
        changed = begin_validation(path, args.text, args.model_version)
    elif args.command == "pass":
        changed = complete_validation(
            path,
            args.text,
            passed=True,
            model_version=args.model_version,
        )
    elif args.command == "fail":
        changed = complete_validation(
            path,
            args.text,
            passed=False,
            model_version=args.model_version,
        )
    else:
        raise AssertionError(args.command)

    print("Updated" if changed else "No matching Semantic Memory entry")
    print_status(path)


if __name__ == "__main__":
    main()
