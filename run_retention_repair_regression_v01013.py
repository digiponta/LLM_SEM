def select_damaged(records):
    return [
        row for row in records
        if row["candidate_top"] != row["expected"]
    ]


def main():
    rows = [
        {
            "text": "量子コンピュータとは何ですか",
            "expected": "computer",
            "candidate_top": "science",
        },
        {
            "text": "量子暗号とは何ですか",
            "expected": "science",
            "candidate_top": "science",
        },
        {
            "text": "暗号",
            "expected": "computer",
            "candidate_top": "computer",
        },
    ]

    damaged = select_damaged(rows)
    checks = [
        (
            "only-misclassified-selected",
            len(damaged) == 1,
        ),
        (
            "observed-quantum-computer-selected",
            damaged[0]["text"] == "量子コンピュータとは何ですか",
        ),
        (
            "correct-records-protected",
            all(
                row["text"] not in {x["text"] for x in damaged}
                for row in rows[1:]
            ),
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
