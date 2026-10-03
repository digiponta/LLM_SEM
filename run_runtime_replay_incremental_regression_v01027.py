def replay_policy(protected_queries, new_queries):
    return {
        "protected": tuple(protected_queries),
        "new": tuple(new_queries),
    }


def main():
    policy = replay_policy(
        ["文学とは", "量子暗号とは"],
        ["量子通信とは"],
    )

    checks = [
        ("literature-runtime-replay", "文学とは" in policy["protected"]),
        ("quantum-crypto-runtime-replay", "量子暗号とは" in policy["protected"]),
        ("quantum-communication-is-new", "量子通信とは" in policy["new"]),
        ("new-not-protected", "量子通信とは" not in policy["protected"]),
    ]

    failed = 0
    for name, ok in checks:
        failed += int(not ok)
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")

    print()
    print("Expected protection:")
    print("  protected prompt : actual /internal prompt")
    print("  protected target : source-model runtime answer")
    print("  new target       : canonical taught answer")
    print()
    print("RESULT:", "PASS" if failed == 0 else "FAIL")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
