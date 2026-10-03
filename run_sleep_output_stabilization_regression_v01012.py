from semantic_guided_answer_finetune_v097 import (
    stabilize_generated_answer,
    generation_quality,
)


def main():
    failed = 0

    def check(name, ok, detail=""):
        nonlocal failed
        failed += int(not ok)
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))

    raw = (
        "文学は、言語による芸術表現を研究する分野であり、数学を含む。"
        "数学を研究する分野であり、数学を含む。"
    )
    stable = stabilize_generated_answer(raw)
    check(
        "first-sentence-stabilization",
        stable == "文学は、言語による芸術表現を研究する分野であり、数学を含む。",
        stable,
    )

    q = generation_quality(stable)
    check("natural-termination", bool(q["terminated"]), repr(q))
    check("low-abnormal-ratio", float(q["abnormal_ratio"]) <= 0.02, repr(q))
    check("low-repetition-ratio", float(q["repetition_ratio"]) <= 0.20, repr(q))

    broken = "文学は、分野であるﾙｾﾞﾛ"
    q2 = generation_quality(broken)
    check(
        "broken-output-not-fully-clean",
        (not bool(q2["terminated"])) or float(q2["abnormal_ratio"]) > 0.02,
        repr(q2),
    )

    print()
    print("RESULT:", "PASS" if failed == 0 else "FAIL")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
