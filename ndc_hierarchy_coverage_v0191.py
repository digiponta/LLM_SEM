#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LLM_SEM v0.19.1 Coverage-Expanded Stable NDC Router.

Goal:
    Improve generalization without changing the frozen semantic encoder.

Method:
    Expand selected 3-digit prototype coverage using additional paraphrase
    anchors that are distinct from the v0.19.0 independent holdout wording.

The stable v0.18.16 beam/UNKNOWN logic is preserved.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch.nn.functional as F

from ndc_semantic_router_v0182 import encode_text
from ndc_hierarchy_stable_v01815 import StableHybridNDCRouter


COVERAGE_EXPANSION: Dict[str, Tuple[str, ...]] = {
    "007": (
        "計算処理をソフトウェアで自動化する",
        "情報処理アルゴリズムを計算機で実行する",
    ),
    "140": (
        "人の知覚や認知の仕組みを研究する",
        "心の状態や行動を心理学的に調べる",
    ),
    "150": (
        "行為の善悪や価値基準を検討する",
        "道徳的に望ましい行動を考える",
    ),
    "280": (
        "一人の人物の人生と業績を記録する",
        "人物の経歴を生涯に沿ってまとめる",
    ),
    "290": (
        "土地ごとの地形や位置関係を調査する",
        "地域の地理的特徴を記述する",
    ),
    "320": (
        "権利義務を定める法制度を扱う",
        "社会のルールを法律として研究する",
    ),
    "330": (
        "雇用や物価など経済活動の変化を分析する",
        "生産消費と市場の動きを調べる",
    ),
    "370": (
        "学校教育の制度や方法を研究する",
        "学習を支える教育の仕組みを考える",
    ),
    "410": (
        "数式を使って数量関係を解く",
        "関数や微積分を用いて数理問題を扱う",
    ),
    "420": (
        "運動や力に関する自然法則を研究する",
        "光や電磁気の性質を物理学で扱う",
    ),
    "430": (
        "原子分子と物質変化を研究する",
        "物質同士の反応と組成を調べる",
    ),
    "440": (
        "恒星や銀河など天体を観測研究する",
        "宇宙にある天体の構造を調べる",
    ),
    "451": (
        "大気の変化から天候を調べる",
        "気圧や風から天気の推移を分析する",
    ),
    "460": (
        "遺伝情報と生物進化の仕組みを研究する",
        "生命の変化や細胞の働きを調べる",
    ),
    "480": (
        "動物の種類と生態を分類研究する",
        "鳥類や哺乳類の特徴を調べる",
    ),
    "490": (
        "病気の診断と治療を医学的に行う",
        "患者の健康状態を診て治療法を考える",
    ),
    "530": (
        "機械の構造や動作機構を設計する",
        "歯車や軸を使う装置を工学的に設計する",
    ),
    "540": (
        "電流電圧を利用する電気回路を設計する",
        "電子部品を使って回路を構成する",
    ),
    "547": (
        "無線や電波で情報を遠距離伝送する",
        "通信信号を送受信する技術を扱う",
    ),
    "548": (
        "コンピュータシステムの情報処理技術を扱う",
        "計算機構成と情報工学を研究する",
    ),
    "596": (
        "食材を調理して料理を作る",
        "食品の調理手順や献立を扱う",
    ),
    "610": (
        "畑で作物を育てて収穫する",
        "農地を使い植物性食料を生産する",
    ),
    "670": (
        "商品を仕入れて販売する商業活動",
        "小売や売買の仕組みを扱う",
    ),
    "680": (
        "人や荷物を交通手段で輸送する",
        "道路鉄道などの移動輸送を扱う",
    ),
    "760": (
        "旋律や音を使って音楽作品を表現する",
        "歌や楽器による音楽活動を扱う",
    ),
    "780": (
        "身体能力を競うスポーツ競技を行う",
        "ルールに従って運動競技を行う",
    ),
    "810": (
        "日本語の助詞や活用など文法を研究する",
        "日本語の語彙や文構造を調べる",
    ),
    "830": (
        "英語の語順や文法規則を学ぶ",
        "英単語と英文構造を研究する",
    ),
    "910": (
        "日本で生まれた文学作品を研究する",
        "和歌俳句を含む日本文学を扱う",
    ),
    "930": (
        "英語圏の文学作品を研究する",
        "英国米国の小説や詩を扱う",
    ),
}


class CoverageExpandedNDCRouter(StableHybridNDCRouter):
    def __init__(self, model, tokenizer, **kwargs) -> None:
        super().__init__(model, tokenizer, **kwargs)

        for code, texts in COVERAGE_EXPANSION.items():
            extra = tuple(
                F.normalize(
                    encode_text(
                        model,
                        tokenizer,
                        text,
                        pooling=self.pooling,
                    ),
                    p=2,
                    dim=-1,
                )
                for text in texts
            )
            self.code_prototypes[code] = tuple(self.code_prototypes[code]) + extra
