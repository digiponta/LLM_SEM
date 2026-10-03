def runtime_gate(pair_sim, canonical_drop, terminated, abnormal, repetition,
                 min_pair=0.70, max_drop=0.05, max_abnormal=0.02, max_repetition=0.20):
    return (
        pair_sim >= min_pair
        and canonical_drop <= max_drop
        and terminated
        and abnormal <= max_abnormal
        and repetition <= max_repetition
    )


def main():
    cases = [
        ("preserved", runtime_gate(0.88, 0.01, True, 0.00, 0.03), True),
        ("route-induced-answer-drift", runtime_gate(0.55, 0.12, True, 0.00, 0.04), False),
        ("bad-termination", runtime_gate(0.90, 0.00, False, 0.00, 0.02), False),
    ]
    failed = 0
    for name, actual, expected in cases:
        ok = actual == expected
        failed += int(not ok)
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {actual}")
    print()
    print("RESULT:", "PASS" if failed == 0 else "FAIL")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
