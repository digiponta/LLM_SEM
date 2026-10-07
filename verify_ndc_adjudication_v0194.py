#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LLM_SEM v0.19.4 Conditional Rescue + Pairwise Adjudication - Holdout V5.

Compares:
  A) v0.19.2 generalized router
  B) v0.19.4 conditional/adjudicated router

No calibration on HOLDOUT-V5.
"""

from __future__ import annotations

import torch

from model import LanguageModel
from tokenizer import Tokenizer
from ndc_hierarchy_unknown_v0192 import CoverageUnknownNDCRouter
from ndc_hierarchy_adjudication_v0194 import ConditionalAdjudicationNDCRouter


HOLDOUT_V5 = (
    ("007", "ソフトウェアで計算処理の手順を実行する"),
    ("140", "人の認知や心の働きを調べる"),
    ("150", "ある行為が道徳的に正しいかを考える"),
    ("280", "一人の人物の生涯と功績を記述する"),
    ("290", "地域の地形や位置関係を地理的に調べる"),
    ("320", "権利と義務を定める法律制度を扱う"),
    ("330", "市場での生産や消費の動きを分析する"),
    ("370", "学校教育の制度と方法を研究する"),
    ("410", "数式で数量関係を解く"),
    ("420", "力と運動に関する物理法則を研究する"),
    ("430", "原子や分子の反応を化学的に調べる"),
    ("440", "星や銀河などの天体を観測研究する"),
    ("451", "気圧や大気の変化から天気を分析する"),
    ("460", "生物の遺伝や進化を研究する"),
    ("480", "動物の種類や生態を分類する"),
    ("490", "患者の症状を診断して治療法を考える"),
    ("530", "機械装置の構造や動作を設計する"),
    ("540", "電子部品を使って電気回路を設計する"),
    ("547", "無線で信号を送受信する通信技術"),
    ("548", "計算機アーキテクチャを情報工学として研究する"),
    ("596", "食材を調理して料理を作る"),
    ("610", "農地で作物を育てて収穫する"),
    ("670", "商品を仕入れて販売する商業活動"),
    ("680", "人や貨物を交通機関で運ぶ"),
    ("760", "楽器や歌によって音楽を表現する"),
    ("780", "身体能力を競うスポーツ競技"),
    ("810", "日本語の文法や語彙を研究する"),
    ("830", "英語の語彙や文法を学ぶ"),
    ("910", "日本で書かれた小説や詩歌を文学として研究する"),
    ("930", "英国や米国の文学作品を研究する"),
)

UNKNOWN_V5 = (
    "それについて説明して",
    "この内容を詳しく",
    "何を指しているのですか",
    "対象がありません",
    "参照先が不明です",
    "意味対象なし",
    "lkjhgfdsa",
    "qazwsxedc",
    "4242424242",
    "未定義対象ベータ",
    "?!?!!?",
    "内容未指定",
    "対象なしの依頼",
    "何の話かわかりません",
    "これについてお願いします",
    "説明する対象がありません",
)


def evaluate(name, router):
    known_correct = 0
    accepted_correct = 0
    accepted = 0
    unknown_reject = 0

    print()
    print("=" * 120)
    print(name)
    print("=" * 120)

    for expected, text in HOLDOUT_V5:
        r = router.route(text)
        correct = r.ndc_code == expected
        is_accepted = r.state == "ACCEPT"
        known_correct += int(correct)
        accepted_correct += int(correct and is_accepted)
        accepted += int(is_accepted)

        status = "PASS" if correct and is_accepted else (
            "MISROUTE" if is_accepted else r.state
        )
        print(
            f"[{status:<9}] expected={expected} actual={r.ndc_code or '---'} "
            f"stage1={r.stage1_main or '-'} "
            f"rescued_unknown={r.rescued_unknown} "
            f"sim={r.code_similarity if r.code_similarity is not None else float('nan'):.3f} "
            f"margin={r.beam_margin if r.beam_margin is not None else float('nan'):+.3f} "
            f"text={text}"
        )

    for text in UNKNOWN_V5:
        r = router.route(text)
        rejected = r.state == "UNKNOWN"
        unknown_reject += int(rejected)
        print(
            f"[{'UNKNOWN' if rejected else 'FALSE_ACCEPT':<12}] "
            f"actual={r.ndc_code or '---'} state={r.state} text={text}"
        )

    raw = known_correct / len(HOLDOUT_V5)
    known_acc = accepted_correct / len(HOLDOUT_V5)
    coverage = accepted / len(HOLDOUT_V5)
    unknown = unknown_reject / len(UNKNOWN_V5)
    balanced = 0.5 * known_acc + 0.5 * unknown

    print()
    print(f"Raw known accuracy      : {raw*100.0:.2f}%")
    print(f"Known accepted accuracy : {known_acc*100.0:.2f}%")
    print(f"Known coverage          : {coverage*100.0:.2f}%")
    print(f"Unknown reject          : {unknown*100.0:.2f}%")
    print(f"Balanced score          : {balanced*100.0:.2f}%")

    return raw, known_acc, coverage, unknown, balanced


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
    b_router = ConditionalAdjudicationNDCRouter(model, tokenizer)

    print("=" * 120)
    print(" LLM_SEM v0.19.4 Conditional Rescue + Pairwise Adjudication - Holdout V5")
    print("=" * 120)
    print("Device          :", device)
    if device.type == "cuda":
        print("GPU             :", torch.cuda.get_device_name(device))
    print("Checkpoint loss :", checkpoint.get("loss"))
    print("Known holdout   :", len(HOLDOUT_V5))
    print("Unknown holdout :", len(UNKNOWN_V5))
    print("Calibration     : NONE")

    a = evaluate("A) v0.19.2 GENERALIZED", a_router)
    b = evaluate("B) v0.19.4 CONDITIONAL / PAIRWISE", b_router)

    print()
    print("=" * 120)
    print("DELTA (v0.19.4 - v0.19.2)")
    print("=" * 120)
    print(f"Raw known accuracy      : {(b[0]-a[0])*100.0:+.2f} pp")
    print(f"Known accepted accuracy : {(b[1]-a[1])*100.0:+.2f} pp")
    print(f"Known coverage          : {(b[2]-a[2])*100.0:+.2f} pp")
    print(f"Unknown reject          : {(b[3]-a[3])*100.0:+.2f} pp")
    print(f"Balanced score          : {(b[4]-a[4])*100.0:+.2f} pp")

    passed = (
        b[0] >= 0.93
        and b[1] >= 0.90
        and b[3] >= 0.90
        and b[4] > a[4]
    )

    print("RESULT:", "PASS" if passed else "EXPERIMENTAL_FAIL")
    print("=" * 120)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
