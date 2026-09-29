# run_adaptive_semantic_learning_regression_v03.py
#
# File-level regression tests for the persistent adaptive semantic memory.
# These tests do not require the model checkpoint or GPU.

from __future__ import annotations

import tempfile
from pathlib import Path

from adaptive_semantic_learning import (
    append_semantic_memory,
    load_semantic_memory,
    merge_samples,
)
from semantic_eval import LabeledSentence


def main() -> int:
    passed = 0
    failed = 0

    def check(name: str, condition: bool) -> None:
        nonlocal passed, failed
        if condition:
            passed += 1
            print(f"[PASS] {name}")
        else:
            failed += 1
            print(f"[FAIL] {name}")

    with tempfile.TemporaryDirectory(prefix="llm_sem_v03_") as tmp:
        path = Path(tmp) / "semantic_memory.jsonl"

        check(
            "append first semantic example",
            append_semantic_memory(path, "weather", "明日は晴れますか"),
        )
        check(
            "deduplicate same semantic example",
            not append_semantic_memory(path, "weather", "明日は晴れますか"),
        )
        check(
            "append another label",
            append_semantic_memory(path, "computer", "GPUとは"),
        )

        rows = load_semantic_memory(path)
        check("memory row count", len(rows) == 2)
        check(
            "memory labels",
            {x.label for x in rows} == {"weather", "computer"},
        )

        base = [
            LabeledSentence("weather", "東京の天気"),
            LabeledSentence("computer", "CPUとは"),
        ]
        merged = merge_samples(base, rows)
        check("merge base + adaptive", len(merged) == 4)

    print()
    print("=" * 60)
    print(f"LLM_SEM v0.3 Adaptive Semantic Learning: {passed} passed, {failed} failed")
    print("=" * 60)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
