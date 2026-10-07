#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LLM_SEM v0.18.13 Beam Hierarchical NDC Regression.

Uses the same v0.18.12 cases to measure direct improvement.
"""

from __future__ import annotations

import torch

from model import LanguageModel
from tokenizer import Tokenizer
from ndc_hierarchy_beam_v01813 import BeamHierarchicalNDCRouter
from verify_ndc_hierarchy_v01812 import CASES, UNKNOWN


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer.load("model/tokenizer.json")
    model, checkpoint = LanguageModel.load_checkpoint(
        "model/model-sem-internalized-v01575.pt",
        device=device,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    router = BeamHierarchicalNDCRouter(
        model,
        tokenizer,
        main_bonus=0.025,
        code_similarity_threshold=0.72,
        beam_margin_threshold=0.01,
    )

    print("=" * 112)
    print(" LLM_SEM v0.18.13 Beam Hierarchical Selected 3-digit NDC Regression")
    print("=" * 112)
    print("Device          :", device)
    if device.type == "cuda":
        print("GPU             :", torch.cuda.get_device_name(device))
    print("Checkpoint loss :", checkpoint.get("loss"))
    print()

    passed = 0
    total = 0
    rescued = 0

    for expected, text in CASES:
        total += 1
        r = router.route(text)
        ok = r.state == "ACCEPT" and r.ndc_code == expected
        passed += int(ok)
        rescued += int(ok and r.rescued_main)
        print(
            f"[{'PASS' if ok else 'FAIL'}] expected={expected} "
            f"actual={r.ndc_code or '---'} state={r.state} "
            f"stage1={r.stage1_main or '-'} rescued={r.rescued_main} "
            f"sim={r.code_similarity if r.code_similarity is not None else float('nan'):.3f} "
            f"beam={r.beam_score if r.beam_score is not None else float('nan'):.3f} "
            f"margin={r.beam_margin if r.beam_margin is not None else float('nan'):+.3f} "
            f"text={text}"
        )

    for text in UNKNOWN:
        total += 1
        r = router.route(text)
        ok = r.state == "UNKNOWN" and r.ndc_code is None
        passed += int(ok)
        print(
            f"[{'PASS' if ok else 'FAIL'}] unknown state={r.state} "
            f"stage1={r.stage1_main or '-'} code={r.ndc_code or '---'} "
            f"text={text}"
        )

    print()
    print("=" * 112)
    print(f"Rescued correct cases: {rescued}")
    print(f"RESULT: {'PASS' if passed == total else 'FAIL'} ({passed}/{total})")
    print("=" * 112)

    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
