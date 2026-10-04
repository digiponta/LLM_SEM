BLOCKED = {"FALSE", "CONTESTED", "OUTDATED"}


def allowed(candidate_state, truth_state=None):
    if candidate_state in BLOCKED:
        return False
    if truth_state in BLOCKED:
        return False
    return True


def main():
    checks = [
        ("candidate-false-blocked", not allowed("FALSE", None)),
        ("candidate-contested-blocked", not allowed("CONTESTED", None)),
        ("candidate-outdated-blocked", not allowed("OUTDATED", None)),
        ("semantic-false-blocked", not allowed("TRUE", "FALSE")),
        ("unverified-allowed", allowed("UNVERIFIED", None)),
        ("true-allowed", allowed("TRUE", "TRUE")),
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
