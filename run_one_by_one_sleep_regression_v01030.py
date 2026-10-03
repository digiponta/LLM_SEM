def accept_step(target_before, target_after, protected_failures,
                min_target=0.70, min_gain=0.10):
    gain = target_after - target_before
    target_ok = (
        target_after >= min_target
        and (gain >= min_gain or target_after >= 0.999999)
    )
    return target_ok and protected_failures == 0


def main():
    checks = [
        ("accept-improved-row", accept_step(0.25, 0.90, 0), True),
        ("reject-protected-regression", accept_step(0.25, 0.90, 1), False),
        ("reject-insufficient-learning", accept_step(0.25, 0.50, 0), False),
        ("accept-exact-canonical", accept_step(0.95, 1.0, 0), True),
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
