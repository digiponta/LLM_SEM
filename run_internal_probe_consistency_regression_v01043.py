def internal_params():
    return {
        "max_new_tokens": 96,
        "temperature": 0.2,
        "top_k": 1,
        "repetition_penalty": 1.10,
    }


def validator_params():
    return {
        "max_new_tokens": 96,
        "temperature": 0.2,
        "top_k": 1,
        "repetition_penalty": 1.10,
    }


def main():
    checks = [
        ("internal-matches-validator", internal_params() == validator_params()),
        ("internal-is-deterministic-top1", internal_params()["top_k"] == 1),
        ("internal-low-temperature", internal_params()["temperature"] == 0.2),
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
