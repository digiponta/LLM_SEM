# run_sleep_memorization_regression_v0109.py
from __future__ import annotations

from semantic_guided_answer_finetune_v097 import split_rows


def main() -> None:
    failures = 0

    def check(name: str, ok: bool, detail: str = "") -> None:
        nonlocal failures
        failures += int(not ok)
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" : {detail}" if detail else ""))

    print("=" * 96)
    print(" LLM_SEM v0.10.9 Sleep Answer Memorization Regression")
    print("=" * 96)

    rows = [
        {
            "query": "CPUとは",
            "answer": "CPUは中央処理装置です。",
            "label": "computer",
            "intent": "definition",
            "concepts": ["CPU"],
            "truth_status": "TRUE",
            "sleep_source": "base",
            "sleep_weight": 1.0,
            "must_train": False,
        },
        {
            "query": "GPUとは",
            "answer": "GPUは並列計算を得意とする処理装置です。",
            "label": "computer",
            "intent": "definition",
            "concepts": ["GPU"],
            "truth_status": "TRUE",
            "sleep_source": "base",
            "sleep_weight": 1.0,
            "must_train": False,
        },
        {
            "query": "文学とは",
            "answer": "文学は、言語による芸術表現を研究する分野であり、分類上、数学を含む。",
            "label": "computer",
            "intent": "definition",
            "concepts": ["文学"],
            "truth_status": "UNVERIFIED",
            "sleep_source": "relation",
            "sleep_weight": 8.0,
            "must_train": True,
        },
    ]

    train_rows, test_rows = split_rows(rows, holdout=0.5, seed=7)
    literature_train = [r for r in train_rows if r["query"] == "文学とは"]
    literature_test = [r for r in test_rows if r["query"] == "文学とは"]

    check(
        "mandatory-row-always-trained",
        len(literature_train) == 1 and not literature_test,
        f"train={len(literature_train)} test={len(literature_test)}",
    )
    check(
        "relation-weight-preserved",
        literature_train[0].get("sleep_weight") == 8.0,
        repr(literature_train[0].get("sleep_weight")),
    )
    check(
        "optional-holdout-still-present",
        len(test_rows) >= 1,
        repr([r["query"] for r in test_rows]),
    )

    print()
    print("RESULT:", "PASS" if failures == 0 else "FAIL")
    raise SystemExit(0 if failures == 0 else 1)


if __name__ == "__main__":
    main()
