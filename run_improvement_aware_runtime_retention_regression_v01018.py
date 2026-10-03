def classify(
    source_canonical,
    candidate_canonical,
    pair_sim,
    *,
    known_threshold=0.70,
    min_gain=0.10,
    min_improved=0.70,
    min_pair=0.70,
    max_drop=0.05,
):
    drop = source_canonical - candidate_canonical
    gain = candidate_canonical - source_canonical
    source_is_known = source_canonical >= known_threshold
    improvement_ok = (
        not source_is_known
        and gain >= min_gain
        and candidate_canonical >= min_improved
    )
    retention_ok = (
        source_is_known
        and pair_sim >= min_pair
        and drop <= max_drop
    )
    if improvement_ok:
        return "IMPROVEMENT", True
    return "RETENTION", retention_ok


def main():
    cases = [
        (
            "literature-retention",
            classify(0.937500, 0.937500, 1.000000),
            ("RETENTION", True),
        ),
        (
            "quantum-crypto-improvement",
            classify(0.164384, 0.891566, 0.205128),
            ("IMPROVEMENT", True),
        ),
        (
            "bad-new-knowledge",
            classify(0.164384, 0.40, 0.20),
            ("RETENTION", False),
        ),
        (
            "known-regression",
            classify(0.90, 0.70, 0.50),
            ("RETENTION", False),
        ),
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
