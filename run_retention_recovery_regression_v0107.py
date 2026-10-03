# run_retention_recovery_regression_v0107.py
from __future__ import annotations


def decide(
    *,
    global_pass: bool,
    source_top: str,
    candidate_top: str,
    expected: str,
    candidate_margin: float,
    expected_gain: float,
    min_margin: float = 0.02,
    recovery_min_margin: float = 0.005,
    recovery_min_gain: float = 0.0,
) -> tuple[bool, str]:
    strict_retention = (
        candidate_top == expected
        and candidate_margin >= min_margin
    )
    recovered = (
        source_top != expected
        and candidate_top == expected
        and candidate_margin >= recovery_min_margin
        and expected_gain > recovery_min_gain
    )
    ok = global_pass and (strict_retention or recovered)
    mode = "RETENTION" if strict_retention else ("RECOVERY" if recovered else "FAIL")
    return ok, mode


def main() -> None:
    failures = 0

    def check(name: str, ok: bool, detail: str = "") -> None:
        nonlocal failures
        failures += int(not ok)
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" : {detail}" if detail else ""))

    print("=" * 96)
    print(" LLM_SEM v0.10.7 Retention Recovery Regression")
    print("=" * 96)

    ok, mode = decide(
        global_pass=True,
        source_top="computer",
        candidate_top="science",
        expected="science",
        candidate_margin=0.010451,
        expected_gain=0.007204,
    )
    check("source-wrong-candidate-recovers", ok and mode == "RECOVERY", mode)

    ok, mode = decide(
        global_pass=True,
        source_top="science",
        candidate_top="science",
        expected="science",
        candidate_margin=0.034,
        expected_gain=-0.001,
    )
    check("normal-retention", ok and mode == "RETENTION", mode)

    ok, mode = decide(
        global_pass=True,
        source_top="computer",
        candidate_top="science",
        expected="science",
        candidate_margin=0.001,
        expected_gain=0.01,
    )
    check("recovery-margin-too-low-blocked", (not ok) and mode == "FAIL", mode)

    ok, mode = decide(
        global_pass=True,
        source_top="computer",
        candidate_top="science",
        expected="science",
        candidate_margin=0.01,
        expected_gain=-0.001,
    )
    check("recovery-without-gain-blocked", (not ok) and mode == "FAIL", mode)

    ok, mode = decide(
        global_pass=False,
        source_top="computer",
        candidate_top="science",
        expected="science",
        candidate_margin=0.01,
        expected_gain=0.01,
    )
    check("global-regression-still-blocks", not ok, mode)

    print()
    print("RESULT:", "PASS" if failures == 0 else "FAIL")
    raise SystemExit(0 if failures == 0 else 1)


if __name__ == "__main__":
    main()
