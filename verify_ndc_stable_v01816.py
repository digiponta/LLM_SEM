#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LLM_SEM v0.18.16 Stable NDC Runtime Regression.

Runs both stable NDC layers:
  A) v0.18.11 main-class runtime regression (16/16)
  B) v0.18.15 selected 3-digit regression (33/33)

The release candidate passes only if both layers pass.
"""

from __future__ import annotations

import subprocess
import sys


TESTS = (
    ("main-class", "verify_ndc_runtime_v01811.py"),
    ("selected-3digit", "verify_ndc_hierarchy_stable_v01815.py"),
)


def main() -> int:
    print("=" * 108)
    print(" LLM_SEM v0.18.16 Stable NDC Runtime Regression")
    print("=" * 108)

    failures = []

    for name, script in TESTS:
        print()
        print("=" * 108)
        print(f" {name}: {script}")
        print("=" * 108)

        completed = subprocess.run(
            [sys.executable, script],
            check=False,
        )

        if completed.returncode != 0:
            failures.append((name, script, completed.returncode))

    print()
    print("=" * 108)
    if failures:
        print("RESULT: FAIL")
        for name, script, code in failures:
            print(f"  {name}: {script} exit={code}")
        print("=" * 108)
        return 1

    print("RESULT: PASS")
    print("  main-class regression : PASS (16/16 expected)")
    print("  selected 3-digit      : PASS (33/33 expected)")
    print("  stable runtime        : READY")
    print("=" * 108)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
