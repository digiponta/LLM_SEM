# run_semantic_sleep_regression_v0105.py
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from semantic_sleep_v0105 import active_records, records_for_texts


def main() -> None:
    failures = 0

    def check(name: str, ok: bool, detail: str = "") -> None:
        nonlocal failures
        failures += int(not ok)
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" : {detail}" if detail else ""))

    print("=" * 92)
    print(" LLM_SEM v0.10.5 Semantic Sleep Regression")
    print("=" * 92)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "semantic_memory.jsonl"
        rows = [
            {
                "label": "computer",
                "text": "量子コンピュータとは何ですか",
                "source": "test",
                "status": "ACTIVE",
                "truth_status": "UNVERIFIED",
            },
            {
                "label": "science",
                "text": "量子力学について教えて",
                "source": "test",
                "status": "CONSOLIDATED",
                "truth_status": "TRUE",
            },
            {
                "label": "computer",
                "text": "古い失敗例",
                "source": "test",
                "status": "FAILED",
                "truth_status": "UNVERIFIED",
            },
        ]
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )

        active = active_records(path)
        check(
            "active-only-selection",
            len(active) == 1 and active[0]["text"] == "量子コンピュータとは何ですか",
            repr(active),
        )

        selected = records_for_texts(
            path,
            {"量子コンピュータとは何ですか"},
        )
        check(
            "target-text-selection",
            len(selected) == 1 and selected[0]["status"] == "ACTIVE",
            repr(selected),
        )

        check(
            "consolidated-not-retrained",
            all(row["status"] != "CONSOLIDATED" for row in active),
        )
        check(
            "failed-not-automatic-retry",
            all(row["status"] != "FAILED" for row in active),
        )

    print()
    print("RESULT:", "PASS" if failures == 0 else "FAIL")
    raise SystemExit(0 if failures == 0 else 1)


if __name__ == "__main__":
    main()
