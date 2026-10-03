# run_multi_relation_composition_regression_v0104.py
from __future__ import annotations

import tempfile
from pathlib import Path

from relation_memory_v0101 import append_relation_fact, compose_subject_facts


def main() -> None:
    failures = 0

    def check(name: str, actual: str, expected: str) -> None:
        nonlocal failures
        ok = actual == expected
        failures += int(not ok)
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
        print("  actual  :", actual)
        print("  expected:", expected)

    print("=" * 96)
    print(" LLM_SEM v0.10.4 Natural Multi-Relation Composition Regression")
    print("=" * 96)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "relations.jsonl"

        append_relation_fact(path, "XはY")
        append_relation_fact(path, "XはZ")
        check(
            "is+is",
            compose_subject_facts(path, "X"),
            "Xは、Yであり、Zである。",
        )

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "relations.jsonl"

        append_relation_fact(path, "文学は言語による芸術表現を研究する分野")
        append_relation_fact(path, "文学は、分類上、数学を含む")
        check(
            "is+includes",
            compose_subject_facts(path, "文学"),
            "文学は、言語による芸術表現を研究する分野であり、分類上、数学を含む。",
        )

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "relations.jsonl"

        append_relation_fact(path, "XはY")
        append_relation_fact(path, "XはAを持つ")
        append_relation_fact(path, "XはBを含む")
        check(
            "is+has+includes",
            compose_subject_facts(path, "X"),
            "Xは、Yであり、Aを持ち、Bを含む。",
        )

    print()
    print("RESULT:", "PASS" if failures == 0 else "FAIL")
    raise SystemExit(0 if failures == 0 else 1)


if __name__ == "__main__":
    main()
