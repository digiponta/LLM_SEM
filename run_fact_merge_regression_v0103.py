# run_fact_merge_regression_v0103.py
from __future__ import annotations

import tempfile
from pathlib import Path

from relation_memory_v0101 import (
    append_relation_fact,
    compose_subject_facts,
    parse_relation_fact,
)


def main() -> None:
    failures = 0

    def check(name: str, ok: bool, detail: str = "") -> None:
        nonlocal failures
        failures += int(not ok)
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" : {detail}" if detail else ""))

    print("=" * 92)
    print(" LLM_SEM v0.10.3 Fact Merge Learning Regression")
    print("=" * 92)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "relations.jsonl"

        f1 = parse_relation_fact("XはY")
        f2 = parse_relation_fact("XはZ")
        check("parse-short-fact-1", f1 is not None and f1["relation"] == "is", repr(f1))
        check("parse-short-fact-2", f2 is not None and f2["relation"] == "is", repr(f2))

        check("append-X-is-Y", append_relation_fact(path, "XはY"))
        merged1 = compose_subject_facts(path, "X")
        check("single-fact-render", merged1 == "Xは、Yである。", merged1)

        check("append-X-is-Z", append_relation_fact(path, "XはZ"))
        merged2 = compose_subject_facts(path, "X")
        check(
            "two-fact-merge",
            merged2 == "Xは、Yであり、Zである。",
            merged2,
        )

        check("append-X-is-W", append_relation_fact(path, "XはW"))
        merged3 = compose_subject_facts(path, "X")
        check(
            "three-fact-merge",
            merged3 == "Xは、Yであり、Zであり、Wである。",
            merged3,
        )

        check(
            "dedupe-existing-fact",
            not append_relation_fact(path, "XはY"),
        )

    print()
    print("RESULT:", "PASS" if failures == 0 else "FAIL")
    raise SystemExit(0 if failures == 0 else 1)


if __name__ == "__main__":
    main()
