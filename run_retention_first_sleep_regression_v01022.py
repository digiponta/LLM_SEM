def should_skip_balanced_sleep(concepts, failures, mean_similarity,
                               min_concepts=2, min_mean=0.75):
    return (
        concepts >= min_concepts
        and failures == 0
        and mean_similarity >= min_mean
    )


def should_noop(source_equals_candidate, pending_semantic):
    return source_equals_candidate and not pending_semantic


def main():
    checks = [
        (
            "v0118-skip-balanced-sleep",
            should_skip_balanced_sleep(2, 0, 0.914533),
            True,
        ),
        (
            "failed-concept-needs-training",
            should_skip_balanced_sleep(2, 1, 0.80),
            False,
        ),
        (
            "insufficient-concepts-needs-training-path",
            should_skip_balanced_sleep(1, 0, 0.95),
            False,
        ),
        (
            "identical-candidate-is-noop",
            should_noop(True, False),
            True,
        ),
        (
            "semantic-pending-prevents-noop",
            should_noop(True, True),
            False,
        ),
        (
            "improved-candidate-not-noop",
            should_noop(False, False),
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
