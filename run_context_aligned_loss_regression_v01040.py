def visible_prefix(full, target_pos, context_length):
    start = max(0, target_pos - context_length)
    return full[start:target_pos]


def main():
    full = list(range(90))
    context = 64

    checks = [
        (
            "early-answer-prefix",
            visible_prefix(full, 59, context) == list(range(59)),
        ),
        (
            "late-answer-window-limited",
            len(visible_prefix(full, 89, context)) == 64,
        ),
        (
            "late-answer-drops-invisible-prefix",
            visible_prefix(full, 89, context)[0] == 25,
        ),
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
