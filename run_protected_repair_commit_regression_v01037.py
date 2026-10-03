def should_repair(target_ok, visible_progress, protected_failures):
    return target_ok and visible_progress and protected_failures > 0


def repair_safe(target_before, target_after, protected_failures,
                min_progress_gain=0.02, min_target=0.70):
    gain = target_after - target_before
    target_ok = target_after >= min_target or gain >= min_progress_gain
    return target_ok and protected_failures == 0


def main():
    checks = [
        ("repair-triggered-on-near-safe-candidate", should_repair(True, True, 1)),
        ("latent-only-does-not-force-repair", not should_repair(True, False, 1)),
        ("safe-repair-accepted", repair_safe(0.25, 0.33, 0)),
        ("repair-losing-target-rejected", not repair_safe(0.25, 0.26, 0)),
        ("repair-with-protected-failure-rejected", not repair_safe(0.25, 0.40, 1)),
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
