def recover_row(current_sim, min_target, was_partial):
    protect = current_sim >= min_target or was_partial
    retry = current_sim < min_target
    return protect, retry


def anchors_for_target(rows, query):
    return [row for row in rows if row["query"] != query]


def main():
    checks = []

    protect, retry = recover_row(0.45, 0.70, True)
    checks.append(("partial-row-protected-for-others", protect))
    checks.append(("partial-row-retried-for-itself", retry))

    protect, retry = recover_row(0.80, 0.70, True)
    checks.append(("complete-row-protected", protect))
    checks.append(("complete-row-not-retried", not retry))

    rows = [
        {"query": "量子通信とは"},
        {"query": "量子暗号とは"},
    ]
    anchors = anchors_for_target(rows, "量子通信とは")
    checks.append((
        "target-not-duplicated-as-anchor",
        [r["query"] for r in anchors] == ["量子暗号とは"],
    ))

    failed = 0
    for name, ok in checks:
        failed += int(not ok)
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")

    print()
    print("RESULT:", "PASS" if failed == 0 else "FAIL")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
