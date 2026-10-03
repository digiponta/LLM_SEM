def replay_target(source_answer, canonical_answer):
    return {
        "source_answer": source_answer,
        "answer": canonical_answer,
        "target_kind": "TRUSTED_CANONICAL",
    }


def main():
    row = replay_target(
        "量子力学の物理法則を利用して、理論上絶対に盗聴されない安全な通信を実現するなも電力です。",
        "量子力学の物理法則を利用して、理論上絶対に盗聴されない安全な通信を実現する技術",
    )

    checks = [
        ("canonical-used-as-target", row["answer"].endswith("技術")),
        ("bad-source-not-used-as-target", "なも電力" not in row["answer"]),
        ("source-kept-for-diagnostics", "なも電力" in row["source_answer"]),
        ("target-kind-explicit", row["target_kind"] == "TRUSTED_CANONICAL"),
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
