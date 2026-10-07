#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LLM_SEM v0.19.5 Generalization Stable Candidate - Independent Holdout V6.

Compares:
  A) v0.19.4 conditional/pairwise router
  B) v0.19.5 final local 830 stabilization

No calibration on HOLDOUT-V6.
"""

from __future__ import annotations

import torch

from model import LanguageModel
from tokenizer import Tokenizer
from ndc_hierarchy_adjudication_v0194 import ConditionalAdjudicationNDCRouter
from ndc_hierarchy_final_v0195 import FinalStableNDCRouter


HOLDOUT_V6 = (
    ("007", "計算処理の手続きをソフトウェアで実行する"),
    ("140", "人の認知や記憶の仕組みを心理学的に調べる"),
    ("150", "行為の善悪を倫理の観点から考える"),
    ("280", "ある人物の一生と功績を記録する"),
    ("290", "地域の地理的特徴や位置関係を調査する"),
    ("320", "法律による権利義務の仕組みを研究する"),
    ("330", "市場における生産消費の変化を分析する"),
    ("370", "教育制度や学校での教え方を研究する"),
    ("410", "数学的な式で数量関係を解く"),
    ("420", "力や運動に関する物理法則を調べる"),
    ("430", "物質の反応と分子構造を化学的に研究する"),
    ("440", "恒星や銀河など宇宙の天体を観測する"),
    ("451", "気圧や大気の状態から天気を分析する"),
    ("460", "遺伝や進化など生命現象を研究する"),
    ("480", "動物の分類や生態を調べる"),
    ("490", "患者の症状を診断して治療法を検討する"),
    ("530", "機械装置の構造や機構を設計する"),
    ("540", "電気回路や電子部品の構成を設計する"),
    ("547", "無線通信で信号を送受信する技術を研究する"),
    ("548", "コンピュータアーキテクチャを情報工学として扱う"),
    ("596", "食材を調理して料理を作る方法を扱う"),
    ("610", "農地で作物を栽培し収穫する"),
    ("670", "商品を仕入れ販売する商業活動を扱う"),
    ("680", "人や貨物を交通手段で輸送する"),
    ("760", "楽器演奏や歌唱で音楽を表現する"),
    ("780", "身体能力を競うスポーツ活動を扱う"),
    ("810", "日本語の文法や語彙体系を研究する"),
    ("830", "英語の語彙と文法体系を学ぶ"),
    ("910", "日本で書かれた小説や詩歌を研究する"),
    ("930", "英国や米国で書かれた文学作品を研究する"),
)

UNKNOWN_V6 = (
    "それについて知りたいです",
    "この件を詳しく",
    "何を示していますか",
    "対象がありません",
    "参照対象不明",
    "意味する対象がありません",
    "plmoknijb",
    "wsxedcrfv",
    "9090909090",
    "未定義対象ガンマ2",
    "!!??!!??",
    "内容なしの依頼",
    "対象未指定の質問",
    "何についてなのかわかりません",
    "これを説明して",
    "説明対象が特定されていません",
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

    for expected, text in HOLDOUT_V6:
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

    for text in UNKNOWN_V6:
        r = router.route(text)
        rejected = r.state == "UNKNOWN"
        unknown_reject += int(rejected)
        print(
            f"[{'UNKNOWN' if rejected else 'FALSE_ACCEPT':<12}] "
            f"actual={r.ndc_code or '---'} state={r.state} text={text}"
        )

    raw = known_correct / len(HOLDOUT_V6)
    known_acc = accepted_correct / len(HOLDOUT_V6)
    coverage = accepted / len(HOLDOUT_V6)
    unknown = unknown_reject / len(UNKNOWN_V6)
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

    a_router = ConditionalAdjudicationNDCRouter(model, tokenizer)
    b_router = FinalStableNDCRouter(model, tokenizer)

    print("=" * 120)
    print(" LLM_SEM v0.19.5 Generalization Stable Candidate - Holdout V6")
    print("=" * 120)
    print("Device          :", device)
    if device.type == "cuda":
        print("GPU             :", torch.cuda.get_device_name(device))
    print("Checkpoint loss :", checkpoint.get("loss"))
    print("Known holdout   :", len(HOLDOUT_V6))
    print("Unknown holdout :", len(UNKNOWN_V6))
    print("Calibration     : NONE")

    a = evaluate("A) v0.19.4 CONDITIONAL / PAIRWISE", a_router)
    b = evaluate("B) v0.19.5 FINAL STABLE CANDIDATE", b_router)

    print()
    print("=" * 120)
    print("DELTA (v0.19.5 - v0.19.4)")
    print("=" * 120)
    print(f"Raw known accuracy      : {(b[0]-a[0])*100.0:+.2f} pp")
    print(f"Known accepted accuracy : {(b[1]-a[1])*100.0:+.2f} pp")
    print(f"Known coverage          : {(b[2]-a[2])*100.0:+.2f} pp")
    print(f"Unknown reject          : {(b[3]-a[3])*100.0:+.2f} pp")
    print(f"Balanced score          : {(b[4]-a[4])*100.0:+.2f} pp")

    passed = (
        b[0] >= 0.97
        and b[1] >= 0.97
        and b[2] >= 0.97
        and b[3] >= 0.95
        and b[4] >= 0.96
    )

    print("RESULT:", "PASS" if passed else "EXPERIMENTAL_FAIL")
    print("=" * 120)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
