def eligible_groups(rows, min_group_rows=2):
    groups = {}
    for row in rows:
        if not row.get('must_train', False):
            continue
        concept = (row.get('concepts') or [row.get('query','')])[0]
        groups.setdefault(concept, []).append(row)
    return {k:v for k,v in groups.items() if len(v) >= min_group_rows}

def accept(before_mean, after_mean, known_failures, min_gain=0.05):
    return known_failures == 0 and (after_mean - before_mean) >= min_gain

def main():
    rows = [
        {'query':'量子センサーとは','concepts':['量子センサー'],'must_train':True},
        {'query':'量子センサー','concepts':['量子センサー'],'must_train':True},
        {'query':'量子センサーについて教えて','concepts':['量子センサー'],'must_train':True},
        {'query':'量子センサーを説明して','concepts':['量子センサー'],'must_train':True},
        {'query':'文学とは','concepts':['文学'],'must_train':False},
    ]
    groups = eligible_groups(rows)
    checks = [
        ('multi-row-concept-selected', '量子センサー' in groups and len(groups['量子センサー']) == 4),
        ('protected-not-selected', '文学' not in groups),
        ('safe-progress-accepted', accept(0.30,0.40,0)),
        ('no-progress-rejected', not accept(0.30,0.33,0)),
        ('known-regression-rejected', not accept(0.30,0.50,1)),
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
