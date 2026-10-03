def build_runtime_prompt(query, label):
    return (
        f"質問:{query}\n"
        f"分類:{label}\n"
        "目的:definition\n"
        f"概念:{query.replace('とは','')}\n"
        "真偽:UNKNOWN\n"
        "回答:"
    )


def main():
    row = {
        "query": "量子通信とは",
        "dataset_label": "science",
        "runtime_selected_label": "computer",
    }
    runtime_prompt = build_runtime_prompt(
        row["query"],
        row["runtime_selected_label"],
    )

    checks = [
        ("runtime-label-used", "分類:computer" in runtime_prompt),
        ("dataset-label-not-used", "分類:science" not in runtime_prompt),
        ("internal-probe-shape", "真偽:UNKNOWN" in runtime_prompt),
        ("answer-slot-present", runtime_prompt.endswith("回答:")),
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
