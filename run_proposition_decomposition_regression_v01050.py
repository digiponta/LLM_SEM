from proposition_bootstrap_v01050 import split_definition

def main():
    concept = '量子センサー'
    text = '量子センサーは、原子や電子などのミクロな「量子」の性質を利用して、磁場、温度、加速度、時間などを極めて高い感度で測定する次世代の計測技術である。'
    parts = split_definition(concept, text)
    expected1 = '量子センサーは、原子や電子などのミクロな「量子」の性質を利用する。'
    expected2 = '量子センサーは、磁場、温度、加速度、時間などを極めて高い感度で測定する次世代の計測技術である。'
    checks = [
        ('two-parts', len(parts) == 2),
        ('first-proposition', len(parts) >= 1 and parts[0] == expected1),
        ('second-proposition', len(parts) >= 2 and parts[1] == expected2),
    ]
    for name, ok in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    print('parts:', parts)
    passed = all(ok for _, ok in checks)
    print()
    print('RESULT:', 'PASS' if passed else 'FAIL')
    raise SystemExit(0 if passed else 1)

if __name__ == '__main__':
    main()
