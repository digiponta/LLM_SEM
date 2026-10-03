def answer_retention_pass(
    source_mean,
    candidate_mean,
    item_deltas,
    *,
    max_mean_drop=0.03,
    max_item_drop=0.05,
):
    return (
        source_mean - candidate_mean <= max_mean_drop
        and all(delta >= -max_item_drop for delta in item_deltas)
    )


def main():
    cases = [
        (
            "preserved",
            answer_retention_pass(0.70, 0.69, [-0.01, 0.00, 0.01]),
            True,
        ),
        (
            "mean-drop",
            answer_retention_pass(0.70, 0.64, [-0.02, -0.03]),
            False,
        ),
        (
            "single-item-drop",
            answer_retention_pass(0.70, 0.69, [-0.01, -0.08]),
            False,
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
