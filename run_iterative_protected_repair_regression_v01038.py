def repair_round_state(target_gain, protected_failures, min_gain=0.02):
    target_ok = target_gain >= min_gain
    if target_ok and protected_failures == 0:
        return "SAFE"
    if not target_ok:
        return "TARGET_LOST"
    return "CONTINUE"


def main():
    checks = [
        ("continue-while-target-retained", repair_round_state(0.08, 1) == "CONTINUE"),
        ("safe-when-protection-recovers", repair_round_state(0.05, 0) == "SAFE"),
        ("stop-when-target-lost", repair_round_state(0.01, 1) == "TARGET_LOST"),
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
