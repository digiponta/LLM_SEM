def should_repair(sim, min_repair=0.70, target=0.95):
    return min_repair <= sim < target


def main():
    checks = [
        ("literature-surface-target", should_repair(0.937500), True),
        ("quantum-crypto-surface-target", should_repair(0.891566), True),
        ("already-clean", should_repair(0.980000), False),
        ("too-low-semantic-match", should_repair(0.400000), False),
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
