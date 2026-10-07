#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LLM_SEM v0.18.7 Confusion-Aware NDC Prototype Set.

Purpose
-------
v0.18.3 reached 70% raw routing without projection.
v0.18.6 preserved strong unknown rejection but did not improve raw routing.

Observed confusions are concentrated around:
  - NDC 2 / 3 / 4
  - NDC 8 / 9
  - selected NDC 7 -> 9 cases

v0.18.7 therefore augments semantic prototypes with additional representative
phrases for those confused regions while keeping the base LLM frozen and
avoiding use of final-test strings as prototypes.
"""

from __future__ import annotations

from typing import Dict, Tuple

from ndc_semantic_router_v0182 import NDC_MAIN_SEEDS


TRAIN_ONLY_NDC_SEEDS: Dict[str, Tuple[str, ...]] = {
    main: tuple(texts[:4])
    for main, texts in NDC_MAIN_SEEDS.items()
}

AUGMENTED_NDC_SEEDS: Dict[str, Tuple[str, ...]] = {
    **TRAIN_ONLY_NDC_SEEDS,

    "2": TRAIN_ONLY_NDC_SEEDS["2"] + (
        "江戸時代と日本史",
        "歴史上の人物の生涯",
        "近代史と社会の変化",
        "古代文明の歴史",
        "地域の歴史と地理",
        "歴史資料を読む",
    ),

    "3": TRAIN_ONLY_NDC_SEEDS["3"] + (
        "景気と経済市場",
        "教育制度と学校政策",
        "法律と社会制度の関係",
        "政治制度と行政",
        "社会保障と福祉政策",
        "経済学と社会科学",
    ),

    "4": TRAIN_ONLY_NDC_SEEDS["4"] + (
        "ブラックホールと宇宙物理",
        "化学反応と物質",
        "生物進化と生命科学",
        "物理学の自然現象",
        "地球科学と自然環境",
        "科学的観測と実験",
    ),

    "7": TRAIN_ONLY_NDC_SEEDS["7"] + (
        "絵画作品を鑑賞する",
        "美術館と芸術作品",
        "音楽作品と演奏",
        "スポーツ競技と体育",
    ),

    "8": TRAIN_ONLY_NDC_SEEDS["8"] + (
        "英単語と語彙",
        "日本語文法と文章表現",
        "翻訳と言語変換",
        "言葉の意味と用法",
        "外国語を学ぶ",
        "言語表現を比較する",
    ),

    "9": TRAIN_ONLY_NDC_SEEDS["9"] + (
        "詩の表現と文学",
        "作家と文学作品",
        "小説と物語文学",
        "文学作品を鑑賞する",
        "詩歌と文学表現",
        "文学史と作家",
    ),
}
