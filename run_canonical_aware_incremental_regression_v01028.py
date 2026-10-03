def termination_ok(terminated, canonical_similarity):
    return bool(terminated) or canonical_similarity >= 0.999999


def weighted_total(qa, preserve, protect, new_weight, preserve_weight, protect_weight):
    return (
        new_weight * qa
        + preserve_weight * preserve
        + protect_weight * protect
    )


def main():
    checks = [
        ("exact-canonical-without-period-passes", termination_ok(False, 1.0), True),
        ("nonexact-unterminated-fails", termination_ok(False, 0.95), False),
        ("normal-terminated-passes", termination_ok(True, 0.80), True),
        (
            "new-knowledge-weight-amplifies-qa",
            weighted_total(1.0, 0.0, 0.0, 3.0, 5.0, 2.0) == 3.0,
            True,
        ),
    ]

    failed = 0
    for name, actual, expected in checks:
        ok = actual == expected
        failed += int(not ok)
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {actual}")

    print()
    print("RESULT:", "PASS" if failed == 0 else "FAIL")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
