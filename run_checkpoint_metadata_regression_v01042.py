def selected_epoch(prefer_final, best_epoch, epochs, has_best):
    if prefer_final or not has_best:
        return max(1, epochs)
    return best_epoch


def main():
    checks = [
        ("final-state-uses-final-epoch", selected_epoch(True, 12, 80, True) == 80),
        ("best-state-uses-best-epoch", selected_epoch(False, 12, 80, True) == 12),
        ("no-best-uses-final-epoch", selected_epoch(False, 0, 80, False) == 80),
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
