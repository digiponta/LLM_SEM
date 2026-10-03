def accept_progress(
    before,
    after,
    protected_failures,
    *,
    min_target=0.70,
    min_target_gain=0.10,
    min_progress_gain=0.02,
):
    gain = after - before
    reached = (
        after >= min_target
        and (gain >= min_target_gain or after >= 0.999999)
    )
    progressive = gain >= min_progress_gain and after > before
    return (reached or progressive) and protected_failures == 0


def should_recover(previous_partial, current_sim, threshold=0.70):
    return previous_partial or current_sim >= threshold


def main():
    checks = [
        (
            "subthreshold-progress-accepted",
            accept_progress(0.25, 0.32, 0),
        ),
        (
            "tiny-progress-rejected",
            not accept_progress(0.25, 0.26, 0),
        ),
        (
            "protected-regression-rejected",
            not accept_progress(0.25, 0.50, 1),
        ),
        (
            "target-level-accepted",
            accept_progress(0.25, 0.85, 0),
        ),
        (
            "recover-prior-subthreshold-partial",
            should_recover(True, 0.45),
        ),
        (
            "recover-threshold-row",
            should_recover(False, 0.75),
        ),
    ]

    failed = 0
    for name, ok in checks:
        failed += int(not ok)
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")

    print()
    print("RESULT:", "PASS" if failed == 0 else "FAIL")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
