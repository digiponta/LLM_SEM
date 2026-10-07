#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LLM_SEM v0.18.1 NDC classification regression."""

from ndc import classify, classify_memory, from_legacy_label


CASES = [
    ("legacy-computer", from_legacy_label("computer"), "007"),
    ("legacy-science", from_legacy_label("science"), "400"),
    ("legacy-animal", from_legacy_label("animal"), "480"),
    ("legacy-weather", from_legacy_label("weather"), "451"),
    ("legacy-food", from_legacy_label("food"), "596"),
    ("legacy-transport", from_legacy_label("transport"), "680"),
    ("information", classify("GPUとCUDAについて説明して"), "007"),
    ("philosophy", classify("哲学とは何か"), "100"),
    ("history", classify("日本の歴史について"), "200"),
    ("social-science", classify("経済と金融市場について"), "330"),
    ("natural-science", classify("量子力学について"), "420"),
    ("engineering", classify("電子工学と回路"), "540"),
    ("industry", classify("鉄道交通について"), "680"),
    ("arts", classify("音楽と楽器"), "760"),
    ("language", classify("英語文法について"), "830"),
    ("literature", classify("日本文学について"), "910"),
    ("memory-combined", classify_memory("GPUとは", "並列計算に適した計算装置"), "007"),
]


def main() -> int:
    print("=" * 88)
    print(" LLM_SEM v0.18.1 NDC Domain Classification Regression")
    print("=" * 88)

    failed = 0
    for name, result, expected in CASES:
        ok = result.code == expected and result.state == "CLASSIFIED"
        print(
            f"[{'PASS' if ok else 'FAIL'}] {name:<20} "
            f"code={result.code!s:<4} main={result.main!s:<2} "
            f"name={result.name} source={result.source}"
        )
        failed += int(not ok)

    unknown = classify("架空概念XYZだけについて")
    unknown_ok = (
        unknown.code is None
        and unknown.main is None
        and unknown.state == "UNKNOWN"
    )
    print(
        f"[{'PASS' if unknown_ok else 'FAIL'}] unknown-not-000       "
        f"code={unknown.code} main={unknown.main} state={unknown.state}"
    )
    failed += int(not unknown_ok)

    print("-" * 88)
    print(f"RESULT: {'PASS' if failed == 0 else 'FAIL'} ({len(CASES) + 1 - failed}/{len(CASES) + 1})")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
