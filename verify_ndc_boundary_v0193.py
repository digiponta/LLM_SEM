#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LLM_SEM v0.19.3 Boundary Reinforcement - Independent Holdout V4.

Compares:
  A) v0.19.2 coverage + expanded UNKNOWN gate
  B) v0.19.3 boundary-reinforced router

No calibration on HOLDOUT-V4.
"""

from __future__ import annotations

import torch

from model import LanguageModel
from tokenizer import Tokenizer
from ndc_hierarchy_unknown_v0192 import CoverageUnknownNDCRouter
from ndc_hierarchy_boundary_v0193 import BoundaryReinforcedNDCRouter


HOLDOUT_V4 = (
    ("007", "ソフトウェアで情報処理の手順を実行する"),
    ("140", "人間の心や認知の仕組みを研究する"),
    ("150", "どの行動が道徳的に適切かを考える"),
    ("280", "人物の生涯と功績をまとめる"),
    ("290", "地域の地理的条件や位置関係を調べる"),
    ("320", "権利義務を定める法制度を研究する"),
    ("330", "市場経済の生産と消費の動きを分析する"),
    ("370", "学校教育の制度や方法を研究する"),
    ("410", "数量の関係を数式で解く"),
    ("420", "運動や力の法則を物理学として研究する"),
    ("430", "物質の反応や分子構造を調べる"),
    ("440", "銀河や恒星など宇宙の天体を観測する"),
    ("451", "大気や気圧の変化から天候を分析する"),
    ("460", "遺伝情報と生物進化を研究する"),
    ("480", "動物の種類と生態を分類する"),
    ("490", "症状を診断して治療を検討する"),
    ("530", "機械装置の構造と動作を設計する"),
    ("540", "電子部品で電気回路を設計する"),
    ("547", "電波で信号を送受信する通信技術を扱う"),
    ("548", "計算機システムの構成技術を情報工学として研究する"),
    ("596", "食材を調理して料理を作る"),
    ("610", "農地で作物を育てて収穫する"),
    ("670", "商品を仕入れて販売する商業活動"),
    ("680", "人や貨物を交通手段で輸送する"),
    ("760", "楽器や歌で音楽を表現する"),
    ("780", "ルールに従って身体能力を競う"),
    ("810", "日本語の文法や語彙を研究する"),
    ("830", "英語の文法や語彙を学ぶ"),
    ("910", "日本の小説や詩歌を文学として研究する"),
    ("930", "英国や米国の文学作品を研究する"),
)

UNKNOWN_V4 = (
    "それのことを教えて",
    "この内容を説明して",
    "何を指していますか",
    "対象が不明です",
    "参照先がありません",
    "意味するものがありません",
    "poiuytrewq",
    "mnbvcxz",
    "123123123",
    "未定義対象オメガ",
    "???!!!???",
    "内容なし",
    "対象のない依頼",
    "何についての質問かわかりません",
    "これについて詳しく",
    "説明対象未指定",
)


def evaluate(name, router):
    known_correct = 0
    known_accept = 0
    known_accepted = 0
    unknown_reject = 0

    print()
    print("=" * 118)
    print(name)
    print("=" * 118)

    for expected, text in HOLDOUT_V4:
        r = router.route(text)
        correct = r.ndc_code == expected
        accepted = r.state == "ACCEPT"
        known_correct += int(correct)
        known_accept += int(correct and accepted)
        known_accepted += int(accepted)

        status = "PASS" if correct and accepted else (
            "MISROUTE" if accepted else r.state
        )
        print(
            f"[{status:<9}] expected={expected} actual={r.ndc_code or '---'} "
            f"stage1={r.stage1_main or '-'} "
            f"sim={r.code_similarity if r.code_similarity is not None else float('nan'):.3f} "
            f"margin={r.beam_margin if r.beam_margin is not None else float('nan'):+.3f} "
            f"text={text}"
        )

    for text in UNKNOWN_V4:
        r = router.route(text)
        rejected = r.state == "UNKNOWN"
        unknown_reject += int(rejected)
        print(
            f"[{'UNKNOWN' if rejected else 'FALSE_ACCEPT':<12}] "
            f"actual={r.ndc_code or '---'} state={r.state} text={text}"
        )

    raw = known_correct / len(HOLDOUT_V4)
    accepted_acc = known_accept / len(HOLDOUT_V4)
    coverage = known_accepted / len(HOLDOUT_V4)
    unknown = unknown_reject / len(UNKNOWN_V4)
    balanced = 0.5 * accepted_acc + 0.5 * unknown

    print()
    print(f"Raw known accuracy      : {raw*100.0:.2f}%")
    print(f"Known accepted accuracy : {accepted_acc*100.0:.2f}%")
    print(f"Known coverage          : {coverage*100.0:.2f}%")
    print(f"Unknown reject          : {unknown*100.0:.2f}%")
    print(f"Balanced score          : {balanced*100.0:.2f}%")
    return raw, accepted_acc, coverage, unknown, balanced


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

    a_router = CoverageUnknownNDCRouter(model, tokenizer)
    b_router = BoundaryReinforcedNDCRouter(model, tokenizer)

    print("=" * 118)
    print(" LLM_SEM v0.19.3 Boundary Reinforcement - Independent Holdout V4")
    print("=" * 118)
    print("Device          :", device)
    if device.type == "cuda":
        print("GPU             :", torch.cuda.get_device_name(device))
    print("Checkpoint loss :", checkpoint.get("loss"))
    print("Known holdout   :", len(HOLDOUT_V4))
    print("Unknown holdout :", len(UNKNOWN_V4))
    print("Calibration     : NONE")

    a = evaluate("A) v0.19.2 COVERAGE + UNKNOWN", a_router)
    b = evaluate("B) v0.19.3 BOUNDARY-REINFORCED", b_router)

    print()
    print("=" * 118)
    print("DELTA (v0.19.3 - v0.19.2)")
    print("=" * 118)
    print(f"Raw known accuracy      : {(b[0]-a[0])*100.0:+.2f} pp")
    print(f"Known accepted accuracy : {(b[1]-a[1])*100.0:+.2f} pp")
    print(f"Known coverage          : {(b[2]-a[2])*100.0:+.2f} pp")
    print(f"Unknown reject          : {(b[3]-a[3])*100.0:+.2f} pp")
    print(f"Balanced score          : {(b[4]-a[4])*100.0:+.2f} pp")

    passed = (
        b[0] >= 0.90
        and b[1] >= 0.87
        and b[3] >= 0.85
        and b[4] > a[4]
    )

    print("RESULT:", "PASS" if passed else "EXPERIMENTAL_FAIL")
    print("=" * 118)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
