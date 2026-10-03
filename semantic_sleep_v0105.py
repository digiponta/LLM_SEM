# semantic_sleep_v0105.py
#
# LLM_SEM v0.10.5
# /sleep consolidation orchestrator
#
# Moves ACTIVE Semantic Memory into the internal model using the existing
# semantic-preserving training / validation / retention / promotion pipeline.

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from adaptive_semantic_learning import (
    load_semantic_memory_records,
    update_memory_status,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LLM_SEM v0.10.5 semantic sleep")
    p.add_argument("--memory", default="data/semantic_memory.jsonl")
    p.add_argument("--model", required=True)
    p.add_argument("--tokenizer", default="model/tokenizer.json")
    p.add_argument("--benchmark", default="my_benchmark.csv")
    p.add_argument("--candidate", default="model/model-sem-sleep-v0105.pt")
    p.add_argument("--manifest", default="model/active-model.json")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--allow-cpu", action="store_true")
    return p.parse_args()


def run_step(command: list[str], name: str) -> None:
    print()
    print("=" * 96)
    print(" SLEEP STEP:", name)
    print("=" * 96)
    print(">", " ".join(command))
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"{name} failed with exit code {result.returncode}")


def active_records(memory: Path) -> list[dict]:
    return [
        row
        for row in load_semantic_memory_records(memory)
        if row.get("status") == "ACTIVE"
    ]


def records_for_texts(memory: Path, texts: set[str]) -> list[dict]:
    return [
        row
        for row in load_semantic_memory_records(memory)
        if str(row.get("text", "")) in texts
    ]


def main() -> None:
    args = parse_args()

    memory = Path(args.memory)
    source = Path(args.model)
    candidate = Path(args.candidate)

    if not memory.exists():
        raise FileNotFoundError(memory)
    if not source.exists():
        raise FileNotFoundError(source)
    if candidate.resolve() == source.resolve():
        raise ValueError("candidate checkpoint must differ from source model")

    pending = active_records(memory)
    if not pending:
        print("SLEEP> no ACTIVE Semantic Memory records; nothing to consolidate.")
        return

    target_texts = {str(row["text"]) for row in pending}

    print("=" * 96)
    print(" LLM_SEM v0.10.5 Semantic Sleep / Internal Consolidation")
    print("=" * 96)
    print("Source model    :", source)
    print("Candidate model :", candidate)
    print("Memory          :", memory)
    print("ACTIVE records  :", len(pending))
    print()
    for idx, row in enumerate(pending, 1):
        print(
            f"{idx:02d}. label={row.get('label')} "
            f"text={row.get('text')!r} truth={row.get('truth_status', 'UNVERIFIED')}"
        )

    # ACTIVE -> TRAINING before touching model weights.
    for row in pending:
        update_memory_status(
            memory,
            str(row["text"]),
            "TRAINING",
            model_version=source.name,
            verified=False,
        )

    print()
    print("SLEEP> lifecycle ACTIVE -> TRAINING")

    train_cmd = [
        sys.executable,
        "semantic_memory_semantic_preserve_train_v078.py",
        "--memory", str(memory),
        "--model", str(source),
        "--tokenizer", args.tokenizer,
        "--benchmark", args.benchmark,
        "--output", str(candidate),
        "--epochs", str(args.epochs),
    ]
    if args.allow_cpu:
        train_cmd.append("--allow-cpu")

    try:
        run_step(train_cmd, "semantic-preserving internal training")
    except Exception:
        # Restore TRAINING targets to ACTIVE so a transient training failure
        # does not strand the authoritative external memory.
        for row in records_for_texts(memory, target_texts):
            if row.get("status") == "TRAINING":
                update_memory_status(
                    memory,
                    str(row["text"]),
                    "ACTIVE",
                    model_version=source.name,
                    verified=False,
                )
        raise

    validate_cmd = [
        sys.executable,
        "semantic_memory_batch_validate_v080.py",
        "--memory", str(memory),
        "--source", str(source),
        "--candidate", str(candidate),
        "--tokenizer", args.tokenizer,
        "--benchmark", args.benchmark,
    ]
    if args.allow_cpu:
        validate_cmd.append("--allow-cpu")
    run_step(validate_cmd, "batch semantic validation")

    after_validation = records_for_texts(memory, target_texts)
    consolidated = [r for r in after_validation if r.get("status") == "CONSOLIDATED"]
    failed = [r for r in after_validation if r.get("status") != "CONSOLIDATED"]

    print()
    print("SLEEP> validation result")
    print("  consolidated:", len(consolidated))
    print("  not promoted :", len(failed))

    if failed:
        print("SLEEP> promotion blocked because one or more target records failed validation.")
        for row in failed:
            print(
                "  -",
                row.get("status"),
                row.get("label"),
                repr(row.get("text")),
            )
        return

    retention_cmd = [
        sys.executable,
        "consolidated_retention_v094.py",
        "--source", str(source),
        "--candidate", str(candidate),
        "--tokenizer", args.tokenizer,
        "--benchmark", args.benchmark,
        "--memory", str(memory),
    ]
    if args.allow_cpu:
        retention_cmd.append("--allow-cpu")
    run_step(retention_cmd, "consolidated retention validation")

    promote_cmd = [
        sys.executable,
        "promote_active_model_v094.py",
        "--candidate", str(candidate),
        "--manifest", args.manifest,
        "--retention-pass",
        "--note", "v0.10.5 /sleep semantic consolidation",
    ]
    if args.allow_cpu:
        promote_cmd.append("--allow-cpu")
    run_step(promote_cmd, "active model promotion")

    print()
    print("=" * 96)
    print(" SLEEP COMPLETE")
    print("=" * 96)
    print("Promoted model :", candidate)
    print("Manifest       :", args.manifest)
    print("Consolidated   :", len(consolidated))
    print("Restart chat.py to load the promoted internal model.")


if __name__ == "__main__":
    main()
