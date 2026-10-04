def classify(source_top, cand_top, expected, source_margin, cand_margin, expected_gain):
    strict = cand_top == expected and cand_margin >= 0.02
    degraded = source_top == expected and cand_top == expected and cand_margin >= 0.01
    recovered = source_top != expected and cand_top == expected and cand_margin >= 0.005 and expected_gain > 0.0
    baseline = source_top == expected and source_margin < 0.01 and cand_top == expected and cand_margin >= 0.0 and expected_gain >= -0.01
    if strict: return 'RETENTION'
    if degraded: return 'DEGRADED'
    if recovered: return 'RECOVERY'
    if baseline: return 'BASELINE'
    return 'FAIL'

def main():
    checks = [
        ('current-case', classify('computer','computer','computer',0.005446,0.002411,-0.005551) == 'BASELINE'),
        ('large-drop-fails', classify('computer','computer','computer',0.005,0.002,-0.03) == 'FAIL'),
        ('label-change-fails', classify('computer','science','computer',0.005,0.02,0.01) == 'FAIL'),
        ('stable-retention', classify('science','science','science',0.05,0.04,0.0) == 'RETENTION'),
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
