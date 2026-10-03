def safe_candidate(target_before, target_after, protected_failures,
                   min_target=0.70, min_gain=0.10):
    gain = target_after - target_before
    target_ok = (
        target_after >= min_target
        and (gain >= min_gain or target_after >= 0.999999)
    )
    return target_ok and protected_failures == 0


def choose_best(rows):
    safe = [r for r in rows if r["safe"]]
    if not safe:
        return None
    return sorted(
        safe,
        key=lambda r: (
            r["target_after"],
            r["target_gain"],
            -r["max_drop"],
            r["min_pair"],
        ),
        reverse=True,
    )[0]


def main():
    rows = [
        {
            "name":"try1","safe":safe_candidate(0.25,0.80,0),
            "target_after":0.80,"target_gain":0.55,"max_drop":0.01,"min_pair":0.90
        },
        {
            "name":"try2","safe":safe_candidate(0.25,0.92,0),
            "target_after":0.92,"target_gain":0.67,"max_drop":0.03,"min_pair":0.88
        },
        {
            "name":"try3","safe":safe_candidate(0.25,0.95,1),
            "target_after":0.95,"target_gain":0.70,"max_drop":0.08,"min_pair":0.60
        },
    ]
    best = choose_best(rows)
    checks = [
        ("unsafe-best-target-rejected", best["name"] != "try3"),
        ("best-safe-selected", best["name"] == "try2"),
        ("insufficient-learning-rejected", not safe_candidate(0.25,0.50,0)),
        ("protected-regression-rejected", not safe_candidate(0.25,0.90,1)),
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
