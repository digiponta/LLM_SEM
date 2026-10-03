# run_semantic_answer_memory_regression_v098.py
#
# LLM_SEM v0.9.8
# Regression for Semantic Answer Memory.

from __future__ import annotations

from semantic_answer_memory_v098 import (
    SemanticAnswerMemory,
    truth_allows_answer_memory,
)


def main() -> None:
    memory = SemanticAnswerMemory.load("data/semantic_guided_qa_v097.json")

    cases = [
        ("CPUとは", "computer", "definition", ["CPU"], "CPUは命令を処理する中央処理装置です。"),
        ("宇宙とは", "science", "definition", ["宇宙"], "宇宙は物質とエネルギーを含む時空全体です。"),
        ("暗号とは", "computer", "definition", ["暗号"], "暗号は情報を保護するための変換方式です。"),
        ("GPUって何", "computer", "definition", ["GPU"], "GPUは並列計算を得意とする処理装置です。"),
    ]

    failures = 0
    print("=" * 88)
    print(" LLM_SEM v0.9.8 Semantic Answer Memory Regression")
    print("=" * 88)
    print("Entries:", len(memory.rows))
    print()

    for query, label, intent, concepts, expected in cases:
        r = memory.resolve(
            query,
            label=label,
            intent=intent,
            concepts=concepts,
            min_score=7.0,
        )
        ok = r.matched and r.answer == expected
        failures += int(not ok)
        print(
            f"[{'PASS' if ok else 'FAIL'}] {query} "
            f"score={r.score:.2f} reason={r.reason}"
        )
        print("  AI>", r.answer)

    false_record = {"truth_status": "FALSE"}
    true_record = {"truth_status": "TRUE"}
    sample = memory.resolve(
        "CPUとは",
        label="computer",
        intent="definition",
        concepts=["CPU"],
    )
    false_blocked = not truth_allows_answer_memory(false_record, sample.candidate)
    true_allowed = truth_allows_answer_memory(true_record, sample.candidate)

    print(
        f"[{'PASS' if false_blocked else 'FAIL'}] explicit FALSE blocks stable answer"
    )
    print(
        f"[{'PASS' if true_allowed else 'FAIL'}] TRUE allows stable answer"
    )
    failures += int(not false_blocked)
    failures += int(not true_allowed)

    unknown = memory.resolve(
        "未登録の概念とは",
        label="science",
        intent="definition",
        concepts=["未登録の概念"],
        min_score=7.0,
    )
    fallback_ok = not unknown.matched
    print(
        f"[{'PASS' if fallback_ok else 'FAIL'}] unknown query falls back "
        f"score={unknown.score:.2f}"
    )
    failures += int(not fallback_ok)

    print()
    print("RESULT:", "PASS" if failures == 0 else "FAIL")
    raise SystemExit(0 if failures == 0 else 1)


if __name__ == "__main__":
    main()
