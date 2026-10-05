from pathlib import Path
from tempfile import TemporaryDirectory

from proposition_bootstrap_v01050 import remove_stale


def main():
    checks = []
    with TemporaryDirectory() as td:
        p = Path(td) / 'stale.json'
        p.write_text('{"old": true}\n', encoding='utf-8')
        checks.append(('fixture-created', p.exists()))
        remove_stale(p)
        checks.append(('stale-result-removed', not p.exists()))
        remove_stale(p)
        checks.append(('idempotent-remove', not p.exists()))

    semantic_sleep = Path('semantic_sleep_v0105.py').read_text(encoding='utf-8')
    one_by_one = Path('one_by_one_sleep_v01030.py').read_text(encoding='utf-8')
    checks.append((
        'precheck-freshness-guard',
        'precheck_json.unlink()' in semantic_sleep
        and 'did not create fresh result' in semantic_sleep,
    ))
    checks.append((
        'onebyone-runtime-freshness-guard',
        'result_json.unlink()' in one_by_one,
    ))
    checks.append((
        'state-version-current',
        '"version": "v0.10.53"' in one_by_one,
    ))

    failed = 0
    for name, ok in checks:
        failed += int(not ok)
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    print()
    print('RESULT:', 'PASS' if failed == 0 else 'FAIL')
    raise SystemExit(1 if failed else 0)


if __name__ == '__main__':
    main()
