#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LLM_SEM v0.19.2 Independent Holdout V3.

Compares:
  A) v0.19.1 coverage-expanded router
  B) v0.19.2 coverage + expanded UNKNOWN gate

No calibration is performed on HOLDOUT-V3.
"""

from __future__ import annotations

import torch

from model import LanguageModel
from tokenizer import Tokenizer
from ndc_hierarchy_coverage_v0191 import CoverageExpandedNDCRouter
from ndc_hierarchy_unknown_v0192 import CoverageUnknownNDCRouter


HOLDOUT_V3 = (
    ("007", "計算手続きをソフトウェアとして記述して実行する"),
    ("140", "人の記憶や認知の働きを調査する"),
    ("150", "行為の善悪や望ましさを考察する"),
    ("280", "人物の生涯と業績を時系列でまとめる"),
    ("290", "地域の位置関係や地形的特色を調べる"),
    ("320", "社会で適用される法的ルールを扱う"),
    ("330", "市場における生産や消費の変化を分析する"),
    ("370", "学校教育の制度や教え方を研究する"),
    ("410", "数量関係を数式で表して解く"),
    ("420", "物体に働く力や運動法則を研究する"),
    ("430", "原子や分子の反応による物質変化を調べる"),
    ("440", "恒星や銀河など宇宙の天体を観測する"),
    ("451", "大気や気圧の変化から天候を調べる"),
    ("460", "生物の遺伝と進化の仕組みを研究する"),
    ("480", "動物種の特徴や生態を分類して調べる"),
    ("490", "症状を診断して治療方法を検討する"),
    ("530", "機械装置の動作機構を設計する"),
    ("540", "電気や電子部品を用いた回路を設計する"),
    ("547", "無線信号を使って情報を伝送する"),
    ("548", "計算機システムの情報処理技術を研究する"),
    ("596", "食材を加工して料理を作る方法を扱う"),
    ("610", "農地で作物を栽培し収穫する"),
    ("670", "商品を仕入れて販売する商取引を扱う"),
    ("680", "人や貨物を交通機関で輸送する"),
    ("760", "楽器や歌を使って音楽を表現する"),
    ("780", "身体能力を競う運動競技を行う"),
    ("810", "日本語の文法や語彙の構造を調べる"),
    ("830", "英語の語彙や文法規則を学ぶ"),
    ("910", "日本で書かれた文学作品を研究する"),
    ("930", "英国や米国の文学作品を扱う"),
)

UNKNOWN_V3 = (
    "それについて教えて",
    "詳しく知りたいです",
    "この内容について",
    "何の意味ですか",
    "対象がわかりません",
    "何を説明しているのですか",
    "zxcvbnm",
    "asdfghjkl",
    "2718281828",
    "未定義対象ラムダ",
    "!!!???",
    "説明対象がありません",
    "内容が特定されていません",
    "参照先のない質問",
    "これについてお願いします",
    "意味する対象がない入力",
)


def evaluate(name, router):
    known_correct = 0
    known_accept = 0
    known_accepted = 0
    unknown_reject = 0

    print()
    print("=" * 116)
    print(name)
    print("=" * 116)

    for expected, text in HOLDOUT_V3:
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

    for text in UNKNOWN_V3:
        r = router.route(text)
        rejected = r.state == "UNKNOWN"
        unknown_reject += int(rejected)
        print(
            f"[{'UNKNOWN' if rejected else 'FALSE_ACCEPT':<12}] "
            f"actual={r.ndc_code or '---'} state={r.state} text={text}"
        )

    raw = known_correct / len(HOLDOUT_V3)
    accepted_acc = known_accept / len(HOLDOUT_V3)
    coverage = known_accepted / len(HOLDOUT_V3)
    unknown = unknown_reject / len(UNKNOWN_V3)
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

    a_router = CoverageExpandedNDCRouter(model, tokenizer)
    b_router = CoverageUnknownNDCRouter(model, tokenizer)

    print("=" * 116)
    print(" LLM_SEM v0.19.2 Expanded UNKNOWN Gate - Independent Holdout V3")
    print("=" * 116)
    print("Device          :", device)
    if device.type == "cuda":
        print("GPU             :", torch.cuda.get_device_name(device))
    print("Checkpoint loss :", checkpoint.get("loss"))
    print("Known holdout   :", len(HOLDOUT_V3))
    print("Unknown holdout :", len(UNKNOWN_V3))
    print("Calibration     : NONE")

    a = evaluate("A) v0.19.1 COVERAGE", a_router)
    b = evaluate("B) v0.19.2 COVERAGE + UNKNOWN EXPANSION", b_router)

    print()
    print("=" * 116)
    print("DELTA (v0.19.2 - v0.19.1)")
    print("=" * 116)
    print(f"Raw known accuracy      : {(b[0]-a[0])*100.0:+.2f} pp")
    print(f"Known accepted accuracy : {(b[1]-a[1])*100.0:+.2f} pp")
    print(f"Known coverage          : {(b[2]-a[2])*100.0:+.2f} pp")
    print(f"Unknown reject          : {(b[3]-a[3])*100.0:+.2f} pp")
    print(f"Balanced score          : {(b[4]-a[4])*100.0:+.2f} pp")

    passed = (
        b[0] >= 0.75
        and b[1] >= 0.70
        and b[3] >= 0.80
        and b[4] > a[4]
    )

    print("RESULT:", "PASS" if passed else "EXPERIMENTAL_FAIL")
    print("=" * 116)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
