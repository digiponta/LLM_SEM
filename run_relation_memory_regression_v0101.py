# run_relation_memory_regression_v0101.py
from __future__ import annotations

import tempfile
from pathlib import Path

from relation_memory_v0101 import append_relation_fact, load_relation_facts, parse_relation_fact


def main() -> None:
    failures = 0
    def check(name, ok, detail=""):
        nonlocal failures
        failures += int(not ok)
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" : {detail}" if detail else ""))

    print("=" * 88)
    print(" LLM_SEM v0.10.1 Relation Memory Regression")
    print("=" * 88)

    parsed = parse_relation_fact("文学は、分類上、数学を含む")
    check(
        "parse-contextual-relation",
        parsed is not None
        and parsed["subject"] == "文学"
        and parsed["relation"] == "includes"
        and parsed["value"] == "数学"
        and parsed["relation_context"] == "分類上",
        repr(parsed),
    )

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "relation.jsonl"
        check(
            "append-relation",
            append_relation_fact(path, "文学は、分類上、数学を含む"),
        )
        rows = load_relation_facts(path)
        check(
            "reload-relation",
            len(rows) == 1 and rows[0].get("value") == "数学",
            repr(rows),
        )
        check(
            "dedupe-relation",
            not append_relation_fact(path, "文学は、分類上、数学を含む"),
        )

    print()
    print("RESULT:", "PASS" if failures == 0 else "FAIL")
    raise SystemExit(0 if failures == 0 else 1)


if __name__ == "__main__":
    main()
