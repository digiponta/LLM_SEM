from semantic_sleep_v0105 import (
    internal_learning_complete,
    next_stall_count,
    round_checkpoint,
)
from pathlib import Path


def main():
    failed = 0

    def check(name, ok, detail=""):
        nonlocal failed
        failed += int(not ok)
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))

    check(
        "complete-at-threshold",
        internal_learning_complete(
            0.80, 0.60, target_mean=0.80, target_min=0.60
        ),
    )
    check(
        "mean-below-target",
        not internal_learning_complete(
            0.79, 0.90, target_mean=0.80, target_min=0.60
        ),
    )
    check(
        "min-below-target",
        not internal_learning_complete(
            0.95, 0.59, target_mean=0.80, target_min=0.60
        ),
    )
    check(
        "stall-increments",
        next_stall_count(0.70, 0.705, 0, min_improvement=0.01) == 1,
    )
    check(
        "stall-resets-on-improvement",
        next_stall_count(0.70, 0.72, 1, min_improvement=0.01) == 0,
    )
    check(
        "round-checkpoint",
        round_checkpoint(Path("model/model-sem-sleep-v0111.pt"), 3).name
        == "model-sem-sleep-v0111.round3.pt",
    )

    print()
    print("RESULT:", "PASS" if failed == 0 else "FAIL")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
