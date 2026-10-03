# run_answer_aware_gate_regression_v099.py
#
# LLM_SEM v0.9.9
# Regression for Answer-Aware Gate.

from __future__ import annotations

from answer_aware_gate_v099 import apply_answer_aware_gate
from semantic_answer_memory_v098 import SemanticAnswerMemory


def main() -> None:
    memory = SemanticAnswerMemory.load("data/semantic_guided_qa_v097.json")

    cases = [
        {
            "query": "宇宙とは",
            "label": "science",
            "intent": "definition",
            "concepts": ["宇宙"],
            "gate": "UNKNOWN_KNOWLEDGE",
            "expect": "ACCEPT_ANSWER_MEMORY",
        },
        {
            "query": "暗号とは",
            "label": "computer",
            "intent": "definition",
            "concepts": ["暗号"],
            "gate": "UNKNOWN_KNOWLEDGE",
            "expect": "ACCEPT_ANSWER_MEMORY",
        },
        {
            "query": "CPUとは",
            "label": "computer",
            "intent": "definition",
            "concepts": ["CPU"],
            "gate": "ACCEPT",
            "expect": "ACCEPT",
        },
    ]

    failures = 0
    print("=" * 88)
    print(" LLM_SEM v0.9.9 Answer-Aware Gate Regression")
    print("=" * 88)

    for case in cases:
        resolution = memory.resolve(
            case["query"],
            label=case["label"],
            intent=case["intent"],
            concepts=case["concepts"],
            min_score=7.0,
        )
        decision = apply_answer_aware_gate(
            case["gate"],
            resolution=resolution,
            truth_record=None,
            min_promote_score=12.0,
        )
        ok = decision.gate == case["expect"]
        failures += int(not ok)
        print(
            f"[{'PASS' if ok else 'FAIL'}] {case['query']} "
            f"{case['gate']} -> {decision.gate} "
            f"score={resolution.score:.2f}"
        )

    blocked_resolution = memory.resolve(
        "宇宙とは",
        label="science",
        intent="definition",
        concepts=["宇宙"],
        min_score=7.0,
    )
    blocked = apply_answer_aware_gate(
        "UNKNOWN_KNOWLEDGE",
        resolution=blocked_resolution,
        truth_record={"truth_status": "FALSE"},
        min_promote_score=12.0,
    )
    blocked_ok = blocked.gate == "UNKNOWN_KNOWLEDGE" and not blocked.promoted
    print(
        f"[{'PASS' if blocked_ok else 'FAIL'}] FALSE truth blocks promotion"
    )
    failures += int(not blocked_ok)

    weak = memory.resolve(
        "未登録の概念とは",
        label="science",
        intent="definition",
        concepts=["未登録の概念"],
        min_score=7.0,
    )
    weak_decision = apply_answer_aware_gate(
        "UNKNOWN_KNOWLEDGE",
        resolution=weak,
        truth_record=None,
        min_promote_score=12.0,
    )
    weak_ok = weak_decision.gate == "UNKNOWN_KNOWLEDGE"
    print(
        f"[{'PASS' if weak_ok else 'FAIL'}] weak answer evidence keeps UNKNOWN"
    )
    failures += int(not weak_ok)

    print()
    print("RESULT:", "PASS" if failures == 0 else "FAIL")
    raise SystemExit(0 if failures == 0 else 1)


if __name__ == "__main__":
    main()
