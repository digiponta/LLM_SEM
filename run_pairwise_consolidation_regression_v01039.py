def pairwise_accept(target_before, target_after, protected_failures,
                    min_gain=0.02, min_target=0.70):
    gain = target_after - target_before
    target_ok = target_after >= min_target or gain >= min_gain
    return target_ok and protected_failures == 0


def main():
    checks = [
        ("pairwise-safe-progress", pairwise_accept(0.25, 0.34, 0)),
        ("pairwise-protected-failure", not pairwise_accept(0.25, 0.40, 1)),
        ("pairwise-target-lost", not pairwise_accept(0.25, 0.26, 0)),
        ("pairwise-target-threshold", pairwise_accept(0.25, 0.75, 0)),
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
