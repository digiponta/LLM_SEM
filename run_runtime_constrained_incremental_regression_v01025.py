def better_safe_candidate(
    known_failures,
    new_failures,
    canonical_mean,
    best_new_failures,
    best_canonical_mean,
):
    safe = known_failures == 0
    return (
        safe
        and (
            new_failures < best_new_failures
            or (
                new_failures == best_new_failures
                and canonical_mean > best_canonical_mean
            )
        )
    )


def main():
    checks = [
        (
            "accept-safe-new-improvement",
            better_safe_candidate(0, 1, 0.82, 2, 0.79),
            True,
        ),
        (
            "reject-known-regression",
            better_safe_candidate(1, 0, 0.90, 2, 0.79),
            False,
        ),
        (
            "prefer-fewer-new-failures",
            better_safe_candidate(0, 0, 0.84, 1, 0.90),
            True,
        ),
        (
            "prefer-higher-mean-on-tie",
            better_safe_candidate(0, 1, 0.86, 1, 0.84),
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
