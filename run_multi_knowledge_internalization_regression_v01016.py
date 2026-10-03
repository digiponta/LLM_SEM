from multi_knowledge_internalization_v01016 import probe_priority, select_probes


def main():
    rows = [
        {
            "query": "文学",
            "answer": "A",
            "concepts": ["文学"],
            "must_train": True,
        },
        {
            "query": "文学とは",
            "answer": "A",
            "concepts": ["文学"],
            "must_train": True,
        },
        {
            "query": "量子暗号とは",
            "answer": "B",
            "concepts": ["量子暗号"],
            "must_train": True,
        },
        {
            "query": "ignored",
            "answer": "C",
            "concepts": ["ignored"],
            "must_train": False,
        },
    ]

    probes = select_probes(rows)
    mapped = {concept: row["query"] for concept, row in probes}

    checks = [
        ("concept-count", len(probes) == 2),
        ("preferred-literature-query", mapped.get("文学") == "文学とは"),
        ("second-concept", mapped.get("量子暗号") == "量子暗号とは"),
        (
            "priority-order",
            probe_priority("文学とは", "文学")
            < probe_priority("文学について教えて", "文学"),
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
