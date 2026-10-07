#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LLM_SEM v0.18.11 Stable NDC Runtime Regression.
"""

from __future__ import annotations

import torch

from model import LanguageModel
from tokenizer import Tokenizer
from ndc_runtime_v01811 import StableNDCRouter


KNOWN = (
    ("0", "AIモデルをコンピュータ上で動かす"),
    ("1", "心理学で心の働きを研究する"),
    ("2", "近代日本の歴史を学ぶ"),
    ("3", "経済政策と市場の動向"),
    ("4", "化学物質の反応を観察する"),
    ("5", "機械を開発する工学"),
    ("6", "商品の流通と販売"),
    ("7", "音楽を演奏する芸術活動"),
    ("8", "英語の文法と語彙"),
    ("9", "小説という文学作品"),
)

UNKNOWN = (
    "それについてお願いします",
    "何を意味しますか",
    "1414213562",
    "あいまいな何か",
    "対象が不明な説明",
    "無意味列lkjhgf",
)


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

    router = StableNDCRouter(model, tokenizer)

    print("=" * 100)
    print(" LLM_SEM v0.18.11 Stable NDC Runtime Regression")
    print("=" * 100)
    print("Device          :", device)
    if device.type == "cuda":
        print("GPU             :", torch.cuda.get_device_name(device))
    print("Checkpoint loss :", checkpoint.get("loss"))
    print()

    passed = 0
    total = 0

    for expected, text in KNOWN:
        total += 1
        r = router.route(text)
        ok = r.accepted and r.ndc_main == expected
        passed += int(ok)
        print(
            f"[{'PASS' if ok else 'FAIL'}] known expected={expected} "
            f"actual={r.ndc_main} state={r.state} "
            f"known={r.known_similarity:.3f} unknown={r.unknown_similarity:.3f} "
            f"contrast={r.contrast:+.3f} score={r.gate_score:.3f} text={text}"
        )

    for text in UNKNOWN:
        total += 1
        r = router.route(text)
        ok = not r.accepted and r.ndc_main is None
        passed += int(ok)
        print(
            f"[{'PASS' if ok else 'FAIL'}] unknown state={r.state} "
            f"known={r.known_similarity:.3f} unknown={r.unknown_similarity:.3f} "
            f"contrast={r.contrast:+.3f} score={r.gate_score:.3f} text={text}"
        )

    print()
    print("=" * 100)
    print(f"RESULT: {'PASS' if passed == total else 'FAIL'} ({passed}/{total})")
    print("=" * 100)

    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
