def latent_accept(
    runtime_gain,
    nll_before,
    nll_after,
    param_delta_rel,
    protected_failures,
    *,
    min_runtime_gain=0.02,
    min_nll_drop=0.05,
    min_nll_rel_drop=0.05,
):
    nll_drop = nll_before - nll_after
    nll_rel = nll_drop / nll_before if nll_before > 0 else 0.0
    runtime_progress = runtime_gain >= min_runtime_gain
    latent_progress = (
        nll_drop >= min_nll_drop
        and nll_rel >= min_nll_rel_drop
        and param_delta_rel > 0.0
    )
    return (runtime_progress or latent_progress) and protected_failures == 0


def main():
    checks = [
        (
            "latent-progress-accepted",
            latent_accept(
                0.0, 2.00, 1.70, 0.001, 0
            ),
        ),
        (
            "no-weight-change-rejected",
            not latent_accept(
                0.0, 2.00, 1.70, 0.0, 0
            ),
        ),
        (
            "tiny-nll-change-rejected",
            not latent_accept(
                0.0, 2.00, 1.98, 0.001, 0
            ),
        ),
        (
            "protected-regression-rejected",
            not latent_accept(
                0.0, 2.00, 1.60, 0.001, 1
            ),
        ),
        (
            "visible-runtime-progress-accepted",
            latent_accept(
                0.10, 2.00, 1.99, 0.001, 0
            ),
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
