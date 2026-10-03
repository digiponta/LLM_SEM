# run_full_sleep_regression_v0106.py
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from build_sleep_qa_dataset_v0106 import (
    load_base,
    load_jsonl,
    normalize_row,
    relation_rows,
)
from relation_memory_v0101 import append_relation_fact


def main() -> None:
    failures = 0

    def check(name: str, ok: bool, detail: str = "") -> None:
        nonlocal failures
        failures += int(not ok)
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" : {detail}" if detail else ""))

    print("=" * 96)
    print(" LLM_SEM v0.10.6 Full Sleep Dataset Regression")
    print("=" * 96)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        base = root / "base.json"
        learned = root / "learned.jsonl"
        relation = root / "relation.jsonl"

        base.write_text(
            json.dumps(
                {
                    "samples": [
                        {
                            "query": "CPUとは",
                            "answer": "CPUは中央処理装置です。",
                            "label": "computer",
                            "intent": "definition",
                            "concepts": ["CPU"],
                            "truth_status": "TRUE",
                        }
                    ]
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        learned.write_text(
            json.dumps(
                {
                    "query": "文学とは",
                    "answer": "文学は言語による芸術表現を研究する分野です。",
                    "label": "computer",
                    "intent": "definition",
                    "concepts": ["文学"],
                    "truth_status": "UNVERIFIED",
                },
                ensure_ascii=False,
            ) + "\n",
            encoding="utf-8",
        )

        append_relation_fact(relation, "文学は言語による芸術表現を研究する分野")
        append_relation_fact(relation, "文学は、分類上、数学を含む")

        base_rows = [normalize_row(x) for x in load_base(base)]
        learned_rows = [normalize_row(x) for x in load_jsonl(learned)]
        rel_rows = relation_rows(relation)

        check("base-row-loaded", len([x for x in base_rows if x]) == 1)
        check("learned-row-loaded", len([x for x in learned_rows if x]) == 1)

        literature = [
            row for row in rel_rows
            if row["query"] == "文学とは"
        ]
        check("relation-canonical-row-present", len(literature) == 1, repr(literature))
        if literature:
            check(
                "relation-canonical-natural-composition",
                literature[0]["answer"]
                == "文学は、言語による芸術表現を研究する分野であり、分類上、数学を含む。",
                literature[0]["answer"],
            )

    print()
    print("RESULT:", "PASS" if failures == 0 else "FAIL")
    raise SystemExit(0 if failures == 0 else 1)


if __name__ == "__main__":
    main()
