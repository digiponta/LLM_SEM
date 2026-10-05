from run_proposition_learning_diagnostic_v01054 import classify


def main():
    cases = [
        ('tokenizer', dict(unk_count=1, nll_rel_drop=.5, param_rel=.1, proposition_gain=.5, full_gain=.5, known_failures=0), 'TOKENIZER_BLOCK'),
        ('no-signal', dict(unk_count=0, nll_rel_drop=.001, param_rel=.1, proposition_gain=.5, full_gain=.5, known_failures=0), 'NO_LEARNING_SIGNAL'),
        ('no-param', dict(unk_count=0, nll_rel_drop=.2, param_rel=1e-9, proposition_gain=.5, full_gain=.5, known_failures=0), 'NO_PARAMETER_UPDATE'),
        ('latent-only', dict(unk_count=0, nll_rel_drop=.2, param_rel=.001, proposition_gain=.001, full_gain=.0, known_failures=0), 'LATENT_ONLY'),
        ('retention-conflict', dict(unk_count=0, nll_rel_drop=.2, param_rel=.001, proposition_gain=.2, full_gain=.0, known_failures=1), 'RETENTION_CONFLICT'),
        ('composition-gap', dict(unk_count=0, nll_rel_drop=.2, param_rel=.001, proposition_gain=.2, full_gain=.001, known_failures=0), 'COMPOSITION_GAP'),
        ('end-to-end', dict(unk_count=0, nll_rel_drop=.2, param_rel=.001, proposition_gain=.2, full_gain=.2, known_failures=0), 'END_TO_END_PROGRESS'),
    ]
    failed = 0
    for name, kwargs, expected in cases:
        actual, _ = classify(**kwargs)
        ok = actual == expected
        failed += int(not ok)
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {actual}")
    print()
    print('RESULT:', 'PASS' if failed == 0 else 'FAIL')
    raise SystemExit(1 if failed else 0)


if __name__ == '__main__':
    main()
