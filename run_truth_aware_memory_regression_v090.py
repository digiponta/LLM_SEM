# run_truth_aware_memory_regression_v090.py
#
# LLM_SEM v0.9.0
# Regression for truth-aware Semantic Memory.
#
# Verifies:
# - legacy records default to UNVERIFIED
# - FALSE information can be intentionally learned/stored
# - lifecycle and truth state remain independent
# - FALSE produces an explicit warning
# - correction_target is retained
# - truth marking does not silently delete the semantic record

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from adaptive_semantic_learning import (
    append_semantic_memory,
    exact_memory_label,
    load_semantic_memory_records,
    truth_notice,
    update_memory_status,
    update_memory_truth,
)


def check(name: str, condition: bool) -> None:
    print(f"[{'PASS' if condition else 'FAIL'}] {name}")
    if not condition:
        raise AssertionError(name)


def main() -> None:
    print("=" * 86)
    print(" LLM_SEM v0.9.0 Truth-Aware Semantic Memory Regression")
    print("=" * 86)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "semantic_memory.jsonl"

        # Legacy schema compatibility.
        path.write_text(
            json.dumps(
                {
                    "label": "science",
                    "text": "legacy proposition",
                    "status": "ACTIVE",
                },
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )

        legacy = load_semantic_memory_records(path)[0]
        check("legacy-default-unverified", legacy["truth_status"] == "UNVERIFIED")

        false_text = "太陽は地球の周りを公転する"
        correction = "地球は太陽の周りを公転する"

        added = append_semantic_memory(
            path,
            "science",
            false_text,
            source="regression",
            truth_status="FALSE",
            truth_confidence=0.99,
            provenance="controlled-test",
            correction_target=correction,
        )
        check("false-information-storable", added)

        rows = load_semantic_memory_records(path)
        false_row = next(row for row in rows if row["text"] == false_text)

        check("false-state-retained", false_row["truth_status"] == "FALSE")
        check(
            "false-confidence-retained",
            abs(false_row["truth_confidence"] - 0.99) < 1.0e-9,
        )
        check(
            "correction-target-retained",
            false_row["correction_target"] == correction,
        )
        check(
            "false-still-semantic-memory",
            exact_memory_label(path, false_text) == "science",
        )

        notice = truth_notice(false_row) or ""
        check("false-warning-emitted", "FALSE" in notice)
        check("correction-in-warning", correction in notice)

        update_memory_status(
            path,
            false_text,
            "CONSOLIDATED",
            model_version="truth-aware-test.pt",
            verified=True,
        )
        consolidated = next(
            row
            for row in load_semantic_memory_records(path)
            if row["text"] == false_text
        )
        check(
            "lifecycle-independent-of-truth",
            consolidated["status"] == "CONSOLIDATED"
            and consolidated["truth_status"] == "FALSE",
        )

        changed = update_memory_truth(
            path,
            false_text,
            "CONTESTED",
            truth_confidence=0.60,
            provenance="later-review",
        )
        check("truth-state-updatable", changed)

        contested = next(
            row
            for row in load_semantic_memory_records(path)
            if row["text"] == false_text
        )
        check(
            "truth-update-preserves-lifecycle",
            contested["status"] == "CONSOLIDATED"
            and contested["truth_status"] == "CONTESTED",
        )
        check(
            "contested-warning-emitted",
            "CONTESTED" in (truth_notice(contested) or ""),
        )

    print()
    print("RESULT: PASS")
    print(
        "Truth state is independent from lifecycle state; incorrect information "
        "can be retained explicitly as incorrect instead of being rejected."
    )


if __name__ == "__main__":
    main()
