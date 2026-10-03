def select_incremental_concepts(details):
    return {
        str(row.get("concept", "")).strip()
        for row in details
        if not bool(row.get("passed", False))
        and str(row.get("concept", "")).strip()
    }


def main():
    precheck = [
        {"concept": "文学", "passed": True},
        {"concept": "量子暗号", "passed": True},
        {"concept": "量子通信", "passed": False},
    ]
    failed = select_incremental_concepts(precheck)

    checks = [
        ("only-new-concept-selected", failed == {"量子通信"}),
        ("protected-literature", "文学" not in failed),
        ("protected-quantum-crypto", "量子暗号" not in failed),
        ("new-concept-trainable", "量子通信" in failed),
    ]

    failed_count = 0
    for name, ok in checks:
        failed_count += int(not ok)
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")

    print()
    print("RESULT:", "PASS" if failed_count == 0 else "FAIL")
    raise SystemExit(1 if failed_count else 0)


if __name__ == "__main__":
    main()
