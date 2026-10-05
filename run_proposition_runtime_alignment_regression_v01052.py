def prompts_match(training_gate, evaluation_gate):
    return training_gate == evaluation_gate == 'INTERNAL_PROBE'

def main():
    checks = [
        ('runtime-prompt-aligned', prompts_match('INTERNAL_PROBE','INTERNAL_PROBE')),
        ('legacy-accept-prompt-detected', not prompts_match('INTERNAL_PROBE','ACCEPT')),
    ]
    failed = 0
    for name, ok in checks:
        failed += int(not ok)
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    print()
    print('RESULT:', 'PASS' if failed == 0 else 'FAIL')
    raise SystemExit(1 if failed else 0)

if __name__ == '__main__':
    main()
