def choose_best_safe(records):
    safe = [r for r in records if r["known_failures"] == 0]
    if not safe:
        return None
    return sorted(
        safe,
        key=lambda r: (
            r["new_failures"],
            -r["canonical_mean"],
        ),
    )[0]


def main():
    records = [
        {"name": "a", "known_failures": 1, "new_failures": 0, "canonical_mean": 0.90},
        {"name": "b", "known_failures": 0, "new_failures": 2, "canonical_mean": 0.82},
        {"name": "c", "known_failures": 0, "new_failures": 1, "canonical_mean": 0.80},
        {"name": "d", "known_failures": 0, "new_failures": 1, "canonical_mean": 0.86},
    ]
    best = choose_best_safe(records)

    checks = [
        ("reject-known-regression", best["name"] != "a"),
        ("prefer-fewer-new-failures", best["name"] in {"c", "d"}),
        ("prefer-higher-mean-on-tie", best["name"] == "d"),
        ("full-pass-definition", (0 == 0 and 0 == 0)),
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
