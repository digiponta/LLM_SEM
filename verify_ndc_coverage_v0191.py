#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LLM_SEM v0.19.1 Independent Holdout V2.

Compares:
  A) frozen v0.18.16 stable router
  B) v0.19.1 coverage-expanded router

Evaluation uses a new holdout V2 not used in v0.19.0 and not copied into the
coverage prototype set.
"""

from __future__ import annotations

from collections import defaultdict
import torch

from model import LanguageModel
from tokenizer import Tokenizer
from ndc_hierarchy_stable_v01815 import StableHybridNDCRouter
from ndc_hierarchy_coverage_v0191 import CoverageExpandedNDCRouter


HOLDOUT_V2 = (
    ("007", "情報を処理する手続きを計算機に実行させる"),
    ("007", "ソフトウェアで知的な計算処理を実現する"),
    ("140", "知覚や記憶など人の心の機能を調べる"),
    ("140", "人間の認知と行動の関係を研究する"),
    ("150", "行動が道徳的に正しいかを考える"),
    ("150", "価値判断の基準と善悪を論じる"),
    ("280", "ある人の人生の歩みと功績を記録する"),
    ("280", "人物の一生を経歴としてまとめる"),
    ("290", "地域ごとの位置や自然条件を調べる"),
    ("290", "土地の特徴を地理的に記述する"),
    ("320", "法律で定められた権利と責任を扱う"),
    ("320", "社会秩序を支える法的な制度を研究する"),
    ("330", "賃金や物価など経済指標の変化をみる"),
    ("330", "市場での生産販売消費の動きを分析する"),
    ("370", "学校で教える仕組みや教育方法を考える"),
    ("370", "学習を支える教育制度を研究する"),
    ("410", "数式を用いて変化量や面積を求める"),
    ("410", "数量の関係を数学的に解明する"),
    ("420", "力と運動の関係を自然法則として扱う"),
    ("420", "電気や磁気の現象を物理学的に研究する"),
    ("430", "分子構造と物質の変化を調べる"),
    ("430", "物質同士が反応する仕組みを研究する"),
    ("440", "星や銀河など遠方天体を研究する"),
    ("440", "宇宙にある天体の形成や性質を調べる"),
    ("451", "大気の変化から雨や風を予測する"),
    ("451", "気圧や気温の推移を気象として調べる"),
    ("460", "遺伝情報が世代間で伝わる仕組みを研究する"),
    ("460", "生命の進化や細胞の働きを調べる"),
    ("480", "哺乳類や鳥類の生態を研究する"),
    ("480", "動物種の特徴を比較して分類する"),
    ("490", "病気を診断して適切な治療を行う"),
    ("490", "患者の症状を医学的に評価する"),
    ("530", "機械の動きを生む機構を設計する"),
    ("530", "回転部品を組み合わせた装置を考える"),
    ("540", "電気信号を扱う回路を設計する"),
    ("540", "電子部品で電気回路を構成する"),
    ("547", "電波を利用して情報を遠くへ送る"),
    ("547", "通信信号の送受信方式を研究する"),
    ("548", "計算機システムの情報処理構成を研究する"),
    ("548", "コンピュータ工学の技術を扱う"),
    ("596", "食材を加工して料理として仕上げる"),
    ("596", "調理方法や料理の作り方を扱う"),
    ("610", "農地で穀物や野菜を生産する"),
    ("610", "作物を栽培して収穫する"),
    ("670", "商品を仕入れ客に販売する活動"),
    ("670", "店舗での売買や小売を扱う"),
    ("680", "人員や貨物を交通機関で運ぶ"),
    ("680", "鉄道道路を使った輸送を扱う"),
    ("760", "旋律や和音を使って音楽を表現する"),
    ("760", "楽器演奏や歌唱による芸術活動"),
    ("780", "ルールの下で身体能力を競う"),
    ("780", "競技として行う運動活動を扱う"),
    ("810", "日本語の語順や助詞の使い方を調べる"),
    ("810", "日本語の文法体系を研究する"),
    ("830", "英文の語順や時制の規則を学ぶ"),
    ("830", "英語の単語と文構造を分析する"),
    ("910", "日本の小説や詩歌を文学として研究する"),
    ("910", "俳句や和歌など日本固有の文学を扱う"),
    ("930", "英国や米国で書かれた文学を研究する"),
    ("930", "英語圏の小説や詩を文学作品として扱う"),
)

UNKNOWN_V2 = (
    "それについて知りたい",
    "詳しく説明してください",
    "この話について",
    "どんな意味ですか",
    "qwertyuiop",
    "3141592653",
    "未定義項目シータ",
    "対象未指定",
    "何かを説明する依頼",
    "!!!!!",
    "内容が示されていない質問",
    "意味対象なし",
)


def evaluate(name, router, holdout, unknown):
    correct = 0
    accepted_correct = 0
    accepted = 0
    unknown_reject = 0
    per_code = defaultdict(lambda: [0, 0])

    print()
    print("=" * 120)
    print(name)
    print("=" * 120)

    for expected, text in holdout:
        r = router.route(text)
        is_correct = r.ndc_code == expected
        is_accepted = r.state == "ACCEPT"
        correct += int(is_correct)
        accepted_correct += int(is_correct and is_accepted)
        accepted += int(is_accepted)
        per_code[expected][1] += 1
        per_code[expected][0] += int(is_correct and is_accepted)

        status = "PASS" if is_correct and is_accepted else (
            "MISROUTE" if is_accepted else r.state
        )
        print(
            f"[{status:<9}] expected={expected} actual={r.ndc_code or '---'} "
            f"stage1={r.stage1_main or '-'} "
            f"sim={r.code_similarity if r.code_similarity is not None else float('nan'):.3f} "
            f"margin={r.beam_margin if r.beam_margin is not None else float('nan'):+.3f} "
            f"text={text}"
        )

    for text in unknown:
        r = router.route(text)
        rejected = r.state == "UNKNOWN"
        unknown_reject += int(rejected)
        print(
            f"[{'UNKNOWN' if rejected else 'FALSE_ACCEPT':<12}] "
            f"actual={r.ndc_code or '---'} state={r.state} text={text}"
        )

    raw = correct / len(holdout)
    known_acc = accepted_correct / len(holdout)
    coverage = accepted / len(holdout)
    unknown_rate = unknown_reject / len(unknown)
    balanced = 0.5 * known_acc + 0.5 * unknown_rate

    print()
    print("Per-code accepted accuracy")
    print("-" * 120)
    for code in sorted(per_code):
        hit, total = per_code[code]
        print(f"NDC {code}: {hit}/{total} = {hit/total*100.0:6.2f}%")

    print()
    print(f"Raw known accuracy      : {raw*100.0:.2f}%")
    print(f"Known accepted accuracy : {known_acc*100.0:.2f}%")
    print(f"Known coverage          : {coverage*100.0:.2f}%")
    print(f"Unknown reject          : {unknown_rate*100.0:.2f}%")
    print(f"Balanced score          : {balanced*100.0:.2f}%")

    return raw, known_acc, coverage, unknown_rate, balanced


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

    stable = StableHybridNDCRouter(model, tokenizer)
    expanded = CoverageExpandedNDCRouter(model, tokenizer)

    print("=" * 120)
    print(" LLM_SEM v0.19.1 Coverage Expansion - Independent Holdout V2")
    print("=" * 120)
    print("Device          :", device)
    if device.type == "cuda":
        print("GPU             :", torch.cuda.get_device_name(device))
    print("Checkpoint loss :", checkpoint.get("loss"))
    print("Known holdout   :", len(HOLDOUT_V2))
    print("Unknown holdout :", len(UNKNOWN_V2))
    print("Calibration     : NONE")

    a = evaluate("A) v0.18.16 STABLE", stable, HOLDOUT_V2, UNKNOWN_V2)
    b = evaluate("B) v0.19.1 COVERAGE-EXPANDED", expanded, HOLDOUT_V2, UNKNOWN_V2)

    print()
    print("=" * 120)
    print("DELTA (EXPANDED - STABLE)")
    print("=" * 120)
    print(f"Raw known accuracy      : {(b[0]-a[0])*100.0:+.2f} pp")
    print(f"Known accepted accuracy : {(b[1]-a[1])*100.0:+.2f} pp")
    print(f"Known coverage          : {(b[2]-a[2])*100.0:+.2f} pp")
    print(f"Unknown reject          : {(b[3]-a[3])*100.0:+.2f} pp")
    print(f"Balanced score          : {(b[4]-a[4])*100.0:+.2f} pp")

    passed = (
        b[0] >= 0.65
        and b[1] >= 0.60
        and b[3] >= 0.75
        and b[4] > a[4]
    )

    print("RESULT:", "PASS" if passed else "EXPERIMENTAL_FAIL")
    print("=" * 120)

    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
