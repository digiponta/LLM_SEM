def select_runtime_repair(target_before, target_after, other_before, other_after,
                          min_gain=0.01, max_other_drop=0.02):
    gain = target_after - target_before
    drops = [
        b - a
        for b, a in zip(other_before, other_after)
    ]
    return gain >= min_gain and max(drops, default=0.0) <= max_other_drop


def main():
    checks = [
        (
            "accept-runtime-improvement",
            select_runtime_repair(
                0.891566, 0.930000,
                [0.937500], [0.935000],
            ),
            True,
        ),
        (
            "reject-v019-runtime-regression",
            select_runtime_repair(
                0.891566, 0.787234,
                [0.937500], [0.937500],
            ),
            False,
        ),
        (
            "reject-cross-concept-runtime-damage",
            select_runtime_repair(
                0.891566, 0.940000,
                [0.937500], [0.880000],
            ),
            False,
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
