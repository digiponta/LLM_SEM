from semantic_guided_answer_finetune_v097 import row_concept


def balanced_weights(rows):
    groups = {}
    for row in rows:
        if row.get("must_train"):
            groups.setdefault(row_concept(row), []).append(row)
    return {
        concept: 1.0 / len(items)
        for concept, items in groups.items()
    }


def completion(mean_sim, min_sim, concept_mean, concept_min):
    return (
        mean_sim >= 0.80
        and min_sim >= 0.60
        and concept_mean >= 0.75
        and concept_min >= 0.70
    )


def main():
    rows = [
        {"query": "文学", "concepts": ["文学"], "must_train": True},
        {"query": "文学とは", "concepts": ["文学"], "must_train": True},
        {"query": "文学を説明して", "concepts": ["文学"], "must_train": True},
        {"query": "量子暗号とは", "concepts": ["量子暗号"], "must_train": True},
    ]

    weights = balanced_weights(rows)
    checks = [
        ("two-concepts", set(weights) == {"文学", "量子暗号"}),
        ("literature-row-weight", abs(weights["文学"] - 1/3) < 1e-9),
        ("quantum-row-weight", abs(weights["量子暗号"] - 1.0) < 1e-9),
        ("reject-dominant-collapse", completion(0.85, 0.62, 0.76, 0.30) is False),
        ("accept-balanced", completion(0.86, 0.72, 0.82, 0.74) is True),
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
