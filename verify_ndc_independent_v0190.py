#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LLM_SEM v0.19.0 Independent Holdout Evaluation.

Purpose:
    Evaluate the frozen v0.18.16 stable selected 3-digit NDC runtime on a new,
    independent paraphrase holdout without changing router parameters.

Protocol:
    - router/checkpoint/thresholds: frozen
    - 60 known probes: 2 unseen paraphrases for each of 30 selected codes
    - 12 UNKNOWN probes
    - no calibration on this dataset

This is an evaluation-only branch.
"""

from __future__ import annotations

from collections import defaultdict
import torch

from model import LanguageModel
from tokenizer import Tokenizer
from ndc_hierarchy_stable_v01815 import StableHybridNDCRouter


HOLDOUT = (
    ("007", "計算機上で知的処理を行う仕組み"),
    ("007", "データ処理の手順をソフトウェアとして記述する"),
    ("140", "人が物事をどう認識するかを調べる"),
    ("140", "心の動きや認知過程を研究対象にする"),
    ("150", "何が正しい行いかを考察する"),
    ("150", "善悪や価値の基準について論じる"),
    ("280", "ある人物が歩んだ人生を年代順に追う"),
    ("280", "著名人の生涯と業績をまとめる"),
    ("290", "土地ごとの特徴や位置関係を調べる"),
    ("290", "地域ごとの地形や特色を記述する"),
    ("320", "社会で守るべき規則の体系を扱う"),
    ("320", "権利や義務を定める制度について学ぶ"),
    ("330", "物価や雇用の変化を分析する"),
    ("330", "生産と消費の動きを社会全体で捉える"),
    ("370", "子どもへの学びの仕組みを考える"),
    ("370", "学校で知識を教える制度を研究する"),
    ("410", "数の関係を式で表して解く"),
    ("410", "関数の変化率や面積を数式で求める"),
    ("420", "物体の運動を支配する法則を調べる"),
    ("420", "光や電磁場の性質を自然法則として扱う"),
    ("430", "物質が別の物質へ変わる過程を調べる"),
    ("430", "原子や分子の結びつき方を研究する"),
    ("440", "遠方の天体や星の集まりを観測する"),
    ("440", "宇宙空間にある天体の性質を調べる"),
    ("451", "大気の状態から雨や風の変化を調べる"),
    ("451", "気圧配置から天候の変化を考える"),
    ("460", "生き物の遺伝情報が世代へ伝わる仕組み"),
    ("460", "生命が長い時間をかけて変化する過程"),
    ("480", "生き物のうち動物の種類や生態を調べる"),
    ("480", "鳥や哺乳類などの特徴を分類する"),
    ("490", "患者の症状から原因を調べて治療する"),
    ("490", "健康を損なう状態を診断し回復を目指す"),
    ("530", "歯車や軸を使う装置の構造を考える"),
    ("530", "動く機構を持つ装置を設計する"),
    ("540", "電流や電圧を利用する装置を設計する"),
    ("540", "電子部品を組み合わせて回路を構成する"),
    ("547", "離れた場所へ信号を送る技術を扱う"),
    ("547", "電波を使って情報を伝送する仕組み"),
    ("548", "計算機を利用した情報処理技術を研究する"),
    ("548", "コンピュータシステムの構成技術を扱う"),
    ("596", "材料を加熱したり混ぜたりして食事を作る"),
    ("596", "食材から料理を作る手順を考える"),
    ("610", "畑で作物を育て収穫する"),
    ("610", "土地を使って食料となる植物を生産する"),
    ("670", "商品を仕入れて客へ販売する"),
    ("670", "売買や店舗運営などの商いを扱う"),
    ("680", "人や荷物を場所から場所へ運ぶ"),
    ("680", "鉄道や道路を使った移動の仕組み"),
    ("760", "音を組み合わせて作品として表現する"),
    ("760", "楽器や歌で旋律を表現する"),
    ("780", "ルールに従って身体能力を競う"),
    ("780", "競技として身体を動かす活動"),
    ("810", "日本で使われる言葉の構造を調べる"),
    ("810", "助詞や活用など日本語の仕組みを学ぶ"),
    ("830", "英語で文を組み立てる規則を学ぶ"),
    ("830", "英単語の意味や英文の構造を調べる"),
    ("910", "日本で書かれた小説や詩歌を研究する"),
    ("910", "和歌や俳句を含む国内の文学作品"),
    ("930", "英語圏で書かれた文学作品を研究する"),
    ("930", "英国や米国の小説・詩を扱う"),
)

UNKNOWN = (
    "それはどうなっていますか",
    "もう少し詳しく",
    "この件について",
    "何のことですか",
    "abcdefghij",
    "9876543210",
    "未定義対象デルタ",
    "説明してほしいもの",
    "内容未指定の質問",
    "???!!!",
    "対象が書かれていない依頼",
    "意味を特定できない入力",
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

    router = StableHybridNDCRouter(model, tokenizer)

    print("=" * 120)
    print(" LLM_SEM v0.19.0 Independent Selected 3-digit NDC Holdout")
    print("=" * 120)
    print("Device             :", device)
    if device.type == "cuda":
        print("GPU                :", torch.cuda.get_device_name(device))
    print("Checkpoint loss    :", checkpoint.get("loss"))
    print("Router             : v0.18.16 stable, frozen")
    print("Known holdout      :", len(HOLDOUT))
    print("Unknown holdout    :", len(UNKNOWN))
    print("Calibration        : NONE")
    print()

    per_code = defaultdict(lambda: [0, 0])
    known_correct = 0
    accepted_correct = 0
    known_accepted = 0

    for expected, text in HOLDOUT:
        r = router.route(text)
        correct = r.ndc_code == expected
        accepted = r.state == "ACCEPT"

        known_correct += int(correct)
        accepted_correct += int(correct and accepted)
        known_accepted += int(accepted)
        per_code[expected][1] += 1
        per_code[expected][0] += int(correct and accepted)

        status = "PASS" if correct and accepted else (
            "MISROUTE" if accepted else r.state
        )
        print(
            f"[{status:<9}] expected={expected} actual={r.ndc_code or '---'} "
            f"stage1={r.stage1_main or '-'} "
            f"rescued_main={r.rescued_main} "
            f"rescued_unknown={r.rescued_unknown} "
            f"sim={r.code_similarity if r.code_similarity is not None else float('nan'):.3f} "
            f"margin={r.beam_margin if r.beam_margin is not None else float('nan'):+.3f} "
            f"text={text}"
        )

    unknown_reject = 0
    for text in UNKNOWN:
        r = router.route(text)
        rejected = r.state == "UNKNOWN"
        unknown_reject += int(rejected)
        print(
            f"[{'UNKNOWN' if rejected else 'FALSE_ACCEPT':<12}] "
            f"actual={r.ndc_code or '---'} state={r.state} text={text}"
        )

    raw_accuracy = known_correct / len(HOLDOUT)
    known_accept_accuracy = accepted_correct / len(HOLDOUT)
    known_coverage = known_accepted / len(HOLDOUT)
    unknown_reject_rate = unknown_reject / len(UNKNOWN)
    balanced = 0.5 * known_accept_accuracy + 0.5 * unknown_reject_rate

    print()
    print("Per-code accepted accuracy")
    print("-" * 120)
    for code in sorted(per_code):
        hit, total = per_code[code]
        print(f"NDC {code}: {hit}/{total} = {hit/total*100.0:6.2f}%")

    print()
    print("=" * 120)
    print("SUMMARY")
    print("=" * 120)
    print(f"Raw known accuracy       : {raw_accuracy*100.0:.2f}%")
    print(f"Known accepted accuracy  : {known_accept_accuracy*100.0:.2f}%")
    print(f"Known coverage           : {known_coverage*100.0:.2f}%")
    print(f"Unknown reject           : {unknown_reject_rate*100.0:.2f}%")
    print(f"Balanced score           : {balanced*100.0:.2f}%")

    passed = (
        raw_accuracy >= 0.80
        and known_accept_accuracy >= 0.75
        and unknown_reject_rate >= 0.80
    )

    print(
        "RESULT:",
        "PASS" if passed else "EXPERIMENTAL_FAIL",
    )
    print("=" * 120)

    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
