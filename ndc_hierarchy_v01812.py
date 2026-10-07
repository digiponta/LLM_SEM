#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LLM_SEM v0.18.12 Hierarchical 3-digit NDC Router.

Stage 1:
    StableNDCRouter -> NDC main class 0-9 or UNKNOWN

Stage 2:
    Restrict candidate 3-digit NDC codes to the accepted main class,
    then choose the nearest selected-code semantic prototype.

Important:
- This router only covers the selected 3-digit codes already defined in ndc.py.
- It does NOT attempt to reproduce the full NDC schedule.
- UNKNOWN remains separate from NDC 000.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import torch
import torch.nn.functional as F

from ndc import NDC_CODES
from ndc_semantic_router_v0182 import encode_text
from ndc_runtime_v01811 import StableNDCRouter


NDC_CODE_SEEDS: Dict[str, Tuple[str, ...]] = {
    "002": (
        "知識と学問について",
        "学術研究と知識体系",
        "学問一般を扱う",
    ),
    "007": (
        "情報科学とコンピュータ",
        "人工知能とプログラミング",
        "データベースとソフトウェア",
        "計算機科学とアルゴリズム",
    ),
    "100": (
        "哲学と思想について",
        "哲学的な考察",
        "思想と論理について",
    ),
    "140": (
        "心理学と認知",
        "心の働きと心理",
        "認知機能を研究する",
    ),
    "150": (
        "倫理学と道徳",
        "善悪と価値判断",
        "倫理的な判断について",
    ),
    "200": (
        "歴史と史学",
        "歴史的な出来事",
        "時代の変化を学ぶ",
    ),
    "280": (
        "人物の伝記",
        "歴史人物の生涯",
        "人物の経歴を調べる",
    ),
    "290": (
        "地理と地域",
        "地誌と地域研究",
        "地域の地理を調べる",
    ),
    "300": (
        "社会科学一般",
        "社会制度を研究する",
        "社会科学の分野",
    ),
    "320": (
        "法律と法学",
        "社会の法制度",
        "法律の仕組み",
    ),
    "330": (
        "経済と市場",
        "景気と経済政策",
        "金融と経済活動",
    ),
    "360": (
        "社会問題と福祉",
        "社会保障と福祉",
        "社会生活の問題",
    ),
    "370": (
        "教育と学校制度",
        "教育政策と学校",
        "学習と教育制度",
    ),
    "400": (
        "自然科学一般",
        "自然現象を科学的に研究する",
        "自然科学の分野",
    ),
    "410": (
        "数学と数理",
        "代数や幾何を扱う",
        "微分積分と数学",
    ),
    "420": (
        "物理学と自然法則",
        "力学と電磁気",
        "量子力学と物理",
    ),
    "430": (
        "化学と物質",
        "化学反応を研究する",
        "分子や化合物を扱う",
    ),
    "440": (
        "天文学と宇宙科学",
        "星や銀河を研究する",
        "宇宙と天体について",
    ),
    "450": (
        "地球科学と地学",
        "地球の構造を研究する",
        "地質と地球科学",
    ),
    "451": (
        "気象学と天気",
        "台風や気圧を研究する",
        "天候と気象について",
    ),
    "460": (
        "生物科学と生命",
        "遺伝と進化を研究する",
        "細胞と生命科学",
    ),
    "480": (
        "動物学",
        "動物の分類と生態",
        "哺乳類や鳥類を研究する",
    ),
    "490": (
        "医学と医療",
        "病気と治療",
        "医療と疾患について",
    ),
    "500": (
        "技術と工学一般",
        "工学技術について",
        "工業技術の分野",
    ),
    "501": (
        "工業の基礎理論",
        "工業基礎学",
        "技術基礎と工学基礎",
    ),
    "530": (
        "機械工学",
        "機械装置を設計する",
        "機械の設計と製造",
    ),
    "540": (
        "電気工学と電子工学",
        "電気回路を設計する",
        "電子回路と電気技術",
    ),
    "547": (
        "通信工学",
        "無線通信とネットワーク",
        "通信技術を研究する",
    ),
    "548": (
        "情報工学",
        "コンピュータ工学",
        "計算機工学と情報技術",
    ),
    "590": (
        "家政学と生活科学",
        "家庭生活と生活科学",
        "家政と生活技術",
    ),
    "596": (
        "食品と料理",
        "調理とレシピ",
        "食べ物と料理について",
    ),
    "600": (
        "産業一般",
        "産業活動について",
        "産業の仕組み",
    ),
    "610": (
        "農業と作物生産",
        "農作物を育てる",
        "農業生産について",
    ),
    "670": (
        "商業と流通",
        "商品の販売と小売",
        "マーケティングと商業",
    ),
    "680": (
        "運輸と交通",
        "鉄道や航空による輸送",
        "物流と交通の仕組み",
    ),
    "690": (
        "通信事業",
        "通信サービス産業",
        "通信事業について",
    ),
    "700": (
        "芸術と美術一般",
        "絵画や彫刻を鑑賞する",
        "美術作品について",
    ),
    "760": (
        "音楽",
        "楽器を演奏する",
        "作曲と音楽活動",
    ),
    "780": (
        "スポーツと体育",
        "競技スポーツ",
        "体育と運動競技",
    ),
    "800": (
        "言語一般と言語学",
        "言語の仕組み",
        "文法と言語学",
    ),
    "810": (
        "日本語",
        "日本語文法",
        "日本語の語彙と表現",
    ),
    "830": (
        "英語",
        "英語文法と語彙",
        "英語表現を学ぶ",
    ),
    "900": (
        "文学一般",
        "文学作品について",
        "小説や詩を文学として読む",
    ),
    "910": (
        "日本文学",
        "日本の文学作品",
        "俳句や和歌と日本文学",
    ),
    "930": (
        "英米文学",
        "英語圏の文学作品",
        "英文学と英米文学",
    ),
}


@dataclass(frozen=True)
class HierarchicalNDCDecision:
    text: str
    state: str
    ndc_main: str | None
    ndc_main_name: str | None
    ndc_code: str | None
    ndc_code_name: str | None
    main_gate_score: float
    code_similarity: float | None
    code_margin: float | None


def _normalize(vector: torch.Tensor) -> torch.Tensor:
    return F.normalize(vector, p=2, dim=-1)


class HierarchicalNDCRouter:
    def __init__(
        self,
        model,
        tokenizer,
        *,
        pooling: str = "mean",
        code_similarity_threshold: float = 0.72,
        code_margin_threshold: float = 0.01,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.pooling = pooling
        self.code_similarity_threshold = float(code_similarity_threshold)
        self.code_margin_threshold = float(code_margin_threshold)

        self.main_router = StableNDCRouter(
            model,
            tokenizer,
            pooling=pooling,
        )

        self.code_prototypes: Dict[str, Tuple[torch.Tensor, ...]] = {}
        for code, texts in NDC_CODE_SEEDS.items():
            if code not in NDC_CODES:
                raise ValueError(f"unsupported selected NDC code: {code}")
            self.code_prototypes[code] = tuple(
                _normalize(
                    encode_text(
                        model,
                        tokenizer,
                        text,
                        pooling=pooling,
                    )
                )
                for text in texts
            )

    @torch.no_grad()
    def route(self, text: str) -> HierarchicalNDCDecision:
        main = self.main_router.route(text)

        if not main.accepted or main.ndc_main is None:
            return HierarchicalNDCDecision(
                text=text,
                state="UNKNOWN",
                ndc_main=None,
                ndc_main_name=None,
                ndc_code=None,
                ndc_code_name=None,
                main_gate_score=main.gate_score,
                code_similarity=None,
                code_margin=None,
            )

        vector = _normalize(
            encode_text(
                self.model,
                self.tokenizer,
                text,
                pooling=self.pooling,
            )
        )

        candidates = {
            code: prototypes
            for code, prototypes in self.code_prototypes.items()
            if code.startswith(main.ndc_main)
        }

        if not candidates:
            return HierarchicalNDCDecision(
                text=text,
                state="MAIN_ONLY",
                ndc_main=main.ndc_main,
                ndc_main_name=main.ndc_name,
                ndc_code=None,
                ndc_code_name=None,
                main_gate_score=main.gate_score,
                code_similarity=None,
                code_margin=None,
            )

        scored = []
        for code, prototypes in candidates.items():
            nearest = max(
                float(torch.dot(vector, proto).item())
                for proto in prototypes
            )
            scored.append((nearest, code))

        scored.sort(reverse=True)
        top1_sim, top1_code = scored[0]
        top2_sim = scored[1][0] if len(scored) >= 2 else -1.0
        margin = top1_sim - top2_sim

        if (
            top1_sim < self.code_similarity_threshold
            or margin < self.code_margin_threshold
        ):
            return HierarchicalNDCDecision(
                text=text,
                state="MAIN_ONLY",
                ndc_main=main.ndc_main,
                ndc_main_name=main.ndc_name,
                ndc_code=None,
                ndc_code_name=None,
                main_gate_score=main.gate_score,
                code_similarity=top1_sim,
                code_margin=margin,
            )

        return HierarchicalNDCDecision(
            text=text,
            state="ACCEPT",
            ndc_main=main.ndc_main,
            ndc_main_name=main.ndc_name,
            ndc_code=top1_code,
            ndc_code_name=NDC_CODES[top1_code],
            main_gate_score=main.gate_score,
            code_similarity=top1_sim,
            code_margin=margin,
        )
