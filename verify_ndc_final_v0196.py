#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LLM_SEM v0.19.6 Final Generalization Stable Candidate - Holdout V7.

Compares:
  A) v0.19.5
  B) v0.19.6 with local 930 stabilization

No calibration on HOLDOUT-V7.
"""

from __future__ import annotations

import torch

from model import LanguageModel
from tokenizer import Tokenizer
from ndc_hierarchy_final_v0195 import FinalStableNDCRouter
from ndc_hierarchy_final_v0196 import FinalStableNDCRouterV0196


HOLDOUT_V7 = (
    ("007", "ソフトウェアに情報処理手順を実行させる"),
    ("140", "人の認知や記憶の働きを心理学で調べる"),
    ("150", "行為の善悪を倫理的観点から考える"),
    ("280", "人物の一生と業績を記録する"),
    ("290", "地域の地形や位置関係を地理学的に調べる"),
    ("320", "権利義務を定める法律制度を研究する"),
    ("330", "市場の生産や消費の動きを分析する"),
    ("370", "学校教育の仕組みと方法を研究する"),
    ("410", "数学的な式で数量関係を解く"),
    ("420", "力や運動の法則を物理学として調べる"),
    ("430", "物質の反応や分子構造を化学的に研究する"),
    ("440", "恒星や銀河などの天体を観測する"),
    ("451", "気圧や大気の変化から天候を分析する"),
    ("460", "生物の遺伝と進化を研究する"),
    ("480", "動物の種類や生態を分類する"),
    ("490", "患者の症状を診断して治療を考える"),
    ("530", "機械装置の構造や機構を設計する"),
    ("540", "電子部品を用いて電気回路を設計する"),
    ("547", "無線通信で信号を送受信する技術を扱う"),
    ("548", "計算機アーキテクチャを情報工学で研究する"),
    ("596", "食材を調理して料理を作る"),
    ("610", "農地で作物を育てて収穫する"),
    ("670", "商品を仕入れて販売する商業活動"),
    ("680", "人や貨物を交通機関で輸送する"),
    ("760", "楽器や歌で音楽を表現する"),
    ("780", "身体能力を競うスポーツ競技"),
    ("810", "日本語の文法や語彙を研究する"),
    ("830", "英語の文法と語彙体系を学ぶ"),
    ("910", "日本の小説や詩歌を文学として研究する"),
    ("930", "イギリスやアメリカで書かれた文学作品を研究する"),
)

UNKNOWN_V7 = (
    "それについて詳しく",
    "この内容を教えて",
    "何を指していますか",
    "対象は何ですか",
    "参照先がありません",
    "意味する対象なし",
    "qwertasdfg",
    "zmxncbv",
    "5656565656",
    "未定義対象デルタ2",
    "?!??!?!",
    "内容未指定です",
    "対象のない質問",
    "何についてかわかりません",
    "これを詳しく説明して",
    "説明対象なし",
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

    for expected, text in HOLDOUT_V7:
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

    for text in UNKNOWN_V7:
        r = router.route(text)
        rejected = r.state == "UNKNOWN"
        unknown_reject += int(rejected)
        print(
            f"[{'UNKNOWN' if rejected else 'FALSE_ACCEPT':<12}] "
            f"actual={r.ndc_code or '---'} state={r.state} text={text}"
        )

    raw = known_correct / len(HOLDOUT_V7)
    known_acc = accepted_correct / len(HOLDOUT_V7)
    coverage = accepted / len(HOLDOUT_V7)
    unknown = unknown_reject / len(UNKNOWN_V7)
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

    a_router = FinalStableNDCRouter(model, tokenizer)
    b_router = FinalStableNDCRouterV0196(model, tokenizer)

    print("=" * 120)
    print(" LLM_SEM v0.19.6 Final Generalization Stable Candidate - Holdout V7")
    print("=" * 120)
    print("Device          :", device)
    if device.type == "cuda":
        print("GPU             :", torch.cuda.get_device_name(device))
    print("Checkpoint loss :", checkpoint.get("loss"))
    print("Known holdout   :", len(HOLDOUT_V7))
    print("Unknown holdout :", len(UNKNOWN_V7))
    print("Calibration     : NONE")

    a = evaluate("A) v0.19.5", a_router)
    b = evaluate("B) v0.19.6 FINAL", b_router)

    print()
    print("=" * 120)
    print("DELTA (v0.19.6 - v0.19.5)")
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
