#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LLM_SEM v0.18.15 Stable Hybrid 3-digit NDC Regression.
"""

from __future__ import annotations

import torch

from model import LanguageModel
from tokenizer import Tokenizer
from ndc_hierarchy_stable_v01815 import StableHybridNDCRouter
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

    router = StableHybridNDCRouter(
        model,
        tokenizer,
        main_bonus=0.025,
        keyword_bonus=0.080,
        code_similarity_threshold=0.72,
        beam_margin_threshold=0.01,
        unknown_rescue_similarity=0.88,
        unknown_rescue_margin=0.04,
    )

    print("=" * 116)
    print(" LLM_SEM v0.18.15 Stable Hybrid Selected 3-digit NDC Regression")
    print("=" * 116)
    print("Device          :", device)
    if device.type == "cuda":
        print("GPU             :", torch.cuda.get_device_name(device))
    print("Checkpoint loss :", checkpoint.get("loss"))
    print()

    passed = 0
    total = 0

    for expected, text in CASES:
        total += 1
        r = router.route(text)
        ok = r.state == "ACCEPT" and r.ndc_code == expected
        passed += int(ok)
        print(
            f"[{'PASS' if ok else 'FAIL'}] expected={expected} "
            f"actual={r.ndc_code or '---'} state={r.state} "
            f"stage1={r.stage1_main or '-'} "
            f"rescued_main={r.rescued_main} "
            f"rescued_unknown={r.rescued_unknown} "
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
    print("=" * 116)
    print(f"RESULT: {'PASS' if passed == total else 'FAIL'} ({passed}/{total})")
    print("=" * 116)

    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
