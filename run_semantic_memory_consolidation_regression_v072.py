# run_semantic_memory_consolidation_regression_v072.py
#
# Regression for Semantic Memory lifecycle priority.

from __future__ import annotations

import tempfile
from pathlib import Path

from adaptive_semantic_learning import (
    append_semantic_memory,
    exact_memory_label,
    load_semantic_memory,
    load_semantic_memory_records,
)
from semantic_memory_consolidation import (
    begin_training,
    begin_validation,
    complete_validation,
)


def check(name: str, condition: bool) -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}")
    if not condition:
        raise AssertionError(name)


def main() -> None:
    print("=" * 78)
    print(" LLM_SEM v0.7.2 Semantic Memory Consolidation Regression")
    print("=" * 78)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "semantic_memory.jsonl"
        text = "量子コンピュータについて教えて"

        check("append-active", append_semantic_memory(path, "computer", text))
        check("active-memory-primary", exact_memory_label(path, text) == "computer")
        check("active-router-visible", len(load_semantic_memory(path)) == 1)

        check("begin-training", begin_training(path, text, "model-v1"))
        check("training-memory-primary", exact_memory_label(path, text) == "computer")
        check("training-router-visible", len(load_semantic_memory(path)) == 1)

        check("begin-validation", begin_validation(path, text, "model-v1"))
        check("validating-memory-primary", exact_memory_label(path, text) == "computer")
        check("validating-router-visible", len(load_semantic_memory(path)) == 1)

        check(
            "validation-fail",
            complete_validation(path, text, passed=False, model_version="model-v1"),
        )
        check("failed-memory-primary", exact_memory_label(path, text) == "computer")
        check("failed-router-visible", len(load_semantic_memory(path)) == 1)

        check("retry-training", begin_training(path, text, "model-v2"))
        check("retry-validating", begin_validation(path, text, "model-v2"))
        check(
            "validation-pass",
            complete_validation(path, text, passed=True, model_version="model-v2"),
        )
        check("consolidated-memory-not-primary", exact_memory_label(path, text) is None)
        check("consolidated-router-hidden", len(load_semantic_memory(path)) == 0)

        records = load_semantic_memory_records(path)
        check("consolidated-record-retained", len(records) == 1)
        check("consolidated-state", records[0]["status"] == "CONSOLIDATED")
        check("consolidated-verified", records[0]["verified"] is True)
        check("consolidated-model-version", records[0]["model_version"] == "model-v2")

    print("-" * 78)
    print("RESULT: PASS")


if __name__ == "__main__":
    main()
