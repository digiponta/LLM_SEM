# run_command_compat_regression_v0102.py
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from adaptive_semantic_learning import (
    append_semantic_memory,
    exact_memory_label,
    forget_semantic_memory,
)
from semantic_answer_memory_v098 import SemanticAnswerMemory


def main() -> None:
    failures = 0

    def check(name: str, ok: bool, detail: str = "") -> None:
        nonlocal failures
        failures += int(not ok)
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" : {detail}" if detail else ""))

    print("=" * 92)
    print(" LLM_SEM v0.10.2 LLM_TRY Command Compatibility Regression")
    print("=" * 92)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        learned = root / "learned.jsonl"
        unified = root / "unified.jsonl"
        base = root / "base.json"
        semantic = root / "semantic.jsonl"

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
        unified.write_text(
            json.dumps(
                {
                    "query": "文学とは",
                    "answer": "分類上、文学は、数学を含む。",
                    "label": "unknown",
                    "intent": "definition",
                    "concepts": ["文学"],
                    "truth_status": "UNVERIFIED",
                },
                ensure_ascii=False,
            ) + "\n",
            encoding="utf-8",
        )
        base.write_text(json.dumps({"samples": []}, ensure_ascii=False), encoding="utf-8")

        memory = SemanticAnswerMemory.load_many([learned, unified, base])
        result = memory.resolve(
            "文学とは",
            label="computer",
            intent="definition",
            concepts=["文学"],
            min_score=7.0,
        )
        check(
            "learned-answer-priority",
            result.matched and result.answer == "文学は言語による芸術表現を研究する分野です。",
            result.answer or "",
        )

        append_semantic_memory(
            semantic,
            label="文学は、言語による芸術を追求する学問",
            text="文学とは",
            source="regression",
        )
        check(
            "bad-label-present",
            exact_memory_label(semantic, "文学とは")
            == "文学は、言語による芸術を追求する学問",
        )
        removed = forget_semantic_memory(semantic, "文学とは")
        check("forget-bad-label", removed)
        check("bad-label-gone", exact_memory_label(semantic, "文学とは") is None)

    print()
    print("RESULT:", "PASS" if failures == 0 else "FAIL")
    raise SystemExit(0 if failures == 0 else 1)


if __name__ == "__main__":
    main()
