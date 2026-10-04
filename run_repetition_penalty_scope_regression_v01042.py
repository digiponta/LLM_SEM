def penalty_history(prompt, generated, penalize_prompt=False):
    full = list(prompt) + list(generated)
    prompt_length = len(prompt)
    return full if penalize_prompt else full[prompt_length:]


def main():
    checks = [
        ("prompt-not-penalized", penalty_history([1,2,3], [], False) == []),
        ("generated-penalized", penalty_history([1,2,3], [2,4], False) == [2,4]),
        ("legacy-mode-available", penalty_history([1,2,3], [4], True) == [1,2,3,4]),
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
