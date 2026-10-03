def classify_row(row):
    if row.get("must_train"):
        return "NEW"
    if row.get("protected"):
        return "PROTECTED"
    return "OPTIONAL"


def main():
    rows = [
        {"query":"量子通信とは","must_train":True,"protected":False},
        {"query":"文学とは","must_train":False,"protected":True},
        {"query":"天気を教えて","must_train":False,"protected":False},
    ]

    checks = [
        ("new-row-visible", classify_row(rows[0]) == "NEW"),
        ("protected-row-visible", classify_row(rows[1]) == "PROTECTED"),
        ("optional-row-visible", classify_row(rows[2]) == "OPTIONAL"),
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
