def termination_ok(similarity, terminated):
    canonical_complete = similarity >= 0.999999
    return bool(terminated) or canonical_complete


def main():
    checks = [
        ("exact-canonical-without-period-passes", termination_ok(1.0, False)),
        ("normal-terminated-answer-passes", termination_ok(0.8, True)),
        ("incomplete-unterminated-fails", not termination_ok(0.8, False)),
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
