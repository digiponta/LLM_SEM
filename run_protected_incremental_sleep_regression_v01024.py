def protected_incremental_split(precheck):
    failed = {
        row["concept"]
        for row in precheck
        if not row["passed"]
    }
    protected = {
        row["concept"]
        for row in precheck
        if row["passed"]
    }
    return failed, protected


def main():
    precheck = [
        {"concept": "文学", "passed": True},
        {"concept": "量子暗号", "passed": True},
        {"concept": "量子通信", "passed": False},
    ]
    failed, protected = protected_incremental_split(precheck)

    checks = [
        ("new-only-train-target", failed == {"量子通信"}),
        ("literature-protected", "文学" in protected),
        ("quantum-crypto-protected", "量子暗号" in protected),
        ("failed-not-protected", "量子通信" not in protected),
    ]

    failed_count = 0
    for name, ok in checks:
        failed_count += int(not ok)
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")

    print()
    print("Expected incremental policy:")
    print("  train blocks          : 1")
    print("  learning-rate scale   : 0.5")
    print("  protected distill wt  : 8.0")
    print()
    print("RESULT:", "PASS" if failed_count == 0 else "FAIL")
    raise SystemExit(1 if failed_count else 0)


if __name__ == "__main__":
    main()
