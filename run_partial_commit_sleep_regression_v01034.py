def state_for(full_pass, accepted_rows, known_failures):
    if full_pass:
        return "COMPLETE"
    if accepted_rows > 0 and known_failures == 0:
        return "PARTIAL"
    return "UNLEARNED"


def recover_partial(source_canonical, threshold=0.70):
    return source_canonical >= threshold


def main():
    checks = [
        ("complete", state_for(True, 4, 0) == "COMPLETE"),
        ("partial", state_for(False, 2, 0) == "PARTIAL"),
        ("known-regression-not-partial", state_for(False, 2, 1) == "UNLEARNED"),
        ("no-progress-unlearned", state_for(False, 0, 0) == "UNLEARNED"),
        ("recover-prior-partial-row", recover_partial(0.82)),
        ("do-not-recover-unlearned-row", not recover_partial(0.55)),
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
