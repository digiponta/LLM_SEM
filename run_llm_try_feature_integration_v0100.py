# run_llm_try_feature_integration_v0100.py
#
# LLM_SEM v0.10.0
# Regression for features integrated from LLM_TRY.

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from answer_aware_gate_v099 import apply_answer_aware_gate
from build_unified_semantic_answer_memory_v0100 import (
    render_incoming,
    render_outgoing,
    variants,
)
from semantic_answer_memory_v098 import SemanticAnswerMemory


def check(name: str, ok: bool) -> int:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    return int(not ok)


def main() -> None:
    failures = 0
    print("=" * 92)
    print(" LLM_SEM v0.10.0 LLM_TRY Feature Integration Regression")
    print("=" * 92)

    failures += check(
        "bare-concept-variant",
        "宇宙" in variants("宇宙") and "宇宙って何" in variants("宇宙"),
    )

    fact = {
        "subject": "文学",
        "relation": "includes",
        "value": "数学",
        "condition": "",
        "relation_context": "分類上",
    }
    failures += check(
        "outgoing-relation-render",
        render_outgoing(fact) == "分類上、文学は、数学を含む。",
    )
    failures += check(
        "incoming-relation-render",
        render_incoming(fact) == "数学は、文学に含まれる対象として関係する。",
    )

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        base = root / "base.json"
        learned = root / "learned.jsonl"
        unified = root / "unified.jsonl"

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

        unified.write_text(
            json.dumps(
                {
                    "query": "宇宙",
                    "answer": "宇宙は時空全体です。",
                    "label": "science",
                    "intent": "definition",
                    "concepts": ["宇宙"],
                    "truth_status": "UNVERIFIED",
                },
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )

        memory = SemanticAnswerMemory.load_many([base, unified, learned])
        failures += check("load-many-json-jsonl", len(memory.rows) == 2)

        resolution = memory.resolve(
            "宇宙",
            label="science",
            intent="definition",
            concepts=["宇宙"],
            min_score=7.0,
        )
        failures += check(
            "bare-concept-answer-lookup",
            resolution.matched and resolution.answer == "宇宙は時空全体です。",
        )

        gate = apply_answer_aware_gate(
            "UNKNOWN_KNOWLEDGE",
            resolution=resolution,
            truth_record=None,
            min_promote_score=12.0,
        )
        failures += check(
            "bare-concept-answer-gate-promotion",
            gate.gate == "ACCEPT_ANSWER_MEMORY",
        )

        added = memory.append_persistent(
            learned,
            query="ブラックホールとは",
            answer="ブラックホールは強い重力を持つ天体です。",
            label="science",
            intent="definition",
            concepts=["ブラックホール"],
            truth_status="UNVERIFIED",
        )
        failures += check("persistent-teach-answer-write", added and learned.exists())

        reloaded = SemanticAnswerMemory.load_many([base, unified, learned])
        taught = reloaded.resolve(
            "ブラックホールとは",
            label="science",
            intent="definition",
            concepts=["ブラックホール"],
            min_score=7.0,
        )
        failures += check(
            "persistent-teach-answer-reload",
            taught.matched and "ブラックホール" in (taught.answer or ""),
        )

    print()
    print("RESULT:", "PASS" if failures == 0 else "FAIL")
    raise SystemExit(0 if failures == 0 else 1)


if __name__ == "__main__":
    main()
