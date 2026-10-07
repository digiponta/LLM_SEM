#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LLM_SEM v0.18.12 Hierarchical selected 3-digit NDC routing regression.
"""

from __future__ import annotations

import torch

from model import LanguageModel
from tokenizer import Tokenizer
from ndc_hierarchy_v01812 import HierarchicalNDCRouter


CASES = (
    ("007", "AIとプログラミングについて"),
    ("140", "心理学で認知を研究する"),
    ("150", "倫理的な価値判断について"),
    ("280", "歴史上の人物の生涯を調べる"),
    ("290", "地域の地理を調べる"),
    ("320", "法律と社会制度について"),
    ("330", "経済と市場の動向"),
    ("370", "学校教育制度について"),
    ("410", "数学の微分積分を学ぶ"),
    ("420", "量子力学の基本"),
    ("430", "化学反応を観察する"),
    ("440", "銀河と恒星を研究する"),
    ("451", "台風と気圧の関係"),
    ("460", "遺伝と生物進化を研究する"),
    ("480", "動物の分類と生態"),
    ("490", "病気と治療について"),
    ("530", "機械装置を設計する"),
    ("540", "電気回路を設計する"),
    ("547", "無線通信の技術"),
    ("548", "情報工学と計算機工学"),
    ("596", "料理とレシピについて"),
    ("610", "農作物を育てる"),
    ("670", "商品の販売と流通"),
    ("680", "鉄道と交通の仕組み"),
    ("760", "楽器を演奏する"),
    ("780", "スポーツ競技について"),
    ("810", "日本語文法を学ぶ"),
    ("830", "英語の語彙と文法"),
    ("910", "日本文学と俳句"),
    ("930", "英米文学作品について"),
)

UNKNOWN = (
    "それについてお願いします",
    "何を意味しますか",
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

    router = HierarchicalNDCRouter(
        model,
        tokenizer,
        code_similarity_threshold=0.72,
        code_margin_threshold=0.01,
    )

    print("=" * 108)
    print(" LLM_SEM v0.18.12 Hierarchical Selected 3-digit NDC Regression")
    print("=" * 108)
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
            f"main={r.ndc_main or '-'} "
            f"sim={r.code_similarity if r.code_similarity is not None else float('nan'):.3f} "
            f"margin={r.code_margin if r.code_margin is not None else float('nan'):+.3f} "
            f"text={text}"
        )

    for text in UNKNOWN:
        total += 1
        r = router.route(text)
        ok = r.state == "UNKNOWN" and r.ndc_code is None
        passed += int(ok)
        print(
            f"[{'PASS' if ok else 'FAIL'}] unknown state={r.state} "
            f"main={r.ndc_main or '-'} code={r.ndc_code or '---'} "
            f"text={text}"
        )

    print()
    print("=" * 108)
    print(f"RESULT: {'PASS' if passed == total else 'FAIL'} ({passed}/{total})")
    print("=" * 108)

    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
