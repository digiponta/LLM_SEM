#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LLM_SEM v0.19.8 Stable Runtime Integration Verification.

Runs:
  1) v0.18.16 integrated main-class + selected-3digit regression
  2) v0.19.7 cross-holdout robustness sweep

The purpose is to verify that the promoted v0.19.6 selected-3digit router
coexists with the established stable main-class runtime.
"""

from __future__ import annotations

import subprocess
import sys


CHECKS = (
    ("v0.18.16 stable regression", "verify_ndc_stable_v01816.py"),
    ("v0.19.7 robustness sweep", "verify_ndc_robustness_v0197.py"),
)


def run_check(name: str, script: str) -> bool:
    print()
    print("=" * 112)
    print(name)
    print("=" * 112)
    result = subprocess.run([sys.executable, script])
    ok = result.returncode == 0
    print(f"[{'PASS' if ok else 'FAIL'}] {script} exit={result.returncode}")
    return ok


def main() -> int:
    print("=" * 112)
    print(" LLM_SEM v0.19.8 Stable Runtime Integration Verification")
    print("=" * 112)

    results = [(name, run_check(name, script)) for name, script in CHECKS]
    passed = all(ok for _, ok in results)

    print()
    print("=" * 112)
    print("SUMMARY")
    print("=" * 112)
    for name, ok in results:
        print(f"{name:<36}: {'PASS' if ok else 'FAIL'}")

    print()
    if passed:
        print("stable main-class runtime          : READY")
        print("generalized selected-3digit router: READY")
        print("chat /ndc3 promotion              : READY")
        print("RESULT                             : PASS")
    else:
        print("RESULT                             : FAIL")
    print("=" * 112)

    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
