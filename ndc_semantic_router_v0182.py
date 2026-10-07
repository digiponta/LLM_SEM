#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LLM_SEM v0.18.2 Semantic Vector + NDC centroid router.

Stage 1 routes semantic vectors to the ten NDC main classes (0-9).
UNKNOWN is a confidence state outside NDC; it is never mapped to NDC 000.

The router is intentionally training-free:
  text -> model.encode_semantic() -> L2 normalize -> class centroids
       -> cosine top-1 + margin -> ACCEPT / UNKNOWN

This module is an experiment layer and does not replace ndc.py keyword routing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn.functional as F

from ndc import NDC_MAIN
from tokenizer import Tokenizer
from model import LanguageModel


NDC_MAIN_SEEDS: Dict[str, Tuple[str, ...]] = {
    "0": (
        "人工知能と大規模言語モデル",
        "コンピュータと情報科学",
        "Pythonプログラミング",
        "CPUとGPUによる計算",
        "データベースとソフトウェア",
        "情報検索と知識体系",
    ),
    "1": (
        "哲学とは何か",
        "論理学と推論",
        "心理学と認知",
        "倫理と道徳",
        "思想と価値観",
        "人間の心について考える",
    ),
    "2": (
        "日本の歴史",
        "世界史の出来事",
        "人物の伝記",
        "地理と地域",
        "古代から現代までの歴史",
        "国や都市の地誌",
    ),
    "3": (
        "経済と金融市場",
        "法律と社会制度",
        "教育と学校",
        "社会福祉",
        "政治と行政",
        "社会科学の研究",
    ),
    "4": (
        "数学と自然科学",
        "量子力学と物理学",
        "化学反応と分子",
        "宇宙と天文学",
        "気象と地球科学",
        "生物と動物",
    ),
    "5": (
        "機械工学と設計",
        "電気電子工学",
        "情報工学と通信工学",
        "建築と土木技術",
        "製造技術と工業",
        "食品と料理の技術",
    ),
    "6": (
        "農業と作物",
        "商業と流通",
        "鉄道と運輸",
        "航空と交通",
        "通信事業",
        "産業と企業活動",
    ),
    "7": (
        "絵画と美術",
        "音楽と演奏",
        "映画と写真",
        "スポーツと体育",
        "芸術作品",
        "デザインと造形",
    ),
    "8": (
        "日本語の文法",
        "英語の文法",
        "言語学",
        "語彙と意味",
        "翻訳と言語表現",
        "発音と会話",
    ),
    "9": (
        "日本文学と小説",
        "英米文学",
        "詩と物語",
        "文学作品を読む",
        "作家と作品",
        "俳句と和歌",
    ),
}


@dataclass(frozen=True)
class RouteResult:
    predicted_main: str
    predicted_name: str
    top1_similarity: float
    top2_similarity: float
    margin: float
    state: str

    @property
    def accepted(self) -> bool:
        return self.state == "ACCEPT"


@torch.no_grad()
def encode_text(
    model: LanguageModel,
    tokenizer: Tokenizer,
    text: str,
    *,
    pooling: str = "mean",
) -> torch.Tensor:
    ids = tokenizer.encode(text, add_bos=True, add_eos=False)
    ids = ids[-model.context_length:]
    x = torch.tensor(
        [ids],
        dtype=torch.long,
        device=next(model.parameters()).device,
    )
    vector = model.encode_semantic(x, pooling=pooling)[0]
    return F.normalize(vector, p=2, dim=-1)


@torch.no_grad()
def build_centroids(
    model: LanguageModel,
    tokenizer: Tokenizer,
    *,
    seeds: Dict[str, Sequence[str]] | None = None,
    pooling: str = "mean",
) -> Dict[str, torch.Tensor]:
    source = seeds or NDC_MAIN_SEEDS
    centroids: Dict[str, torch.Tensor] = {}

    for main, texts in source.items():
        vectors = torch.stack([
            encode_text(model, tokenizer, text, pooling=pooling)
            for text in texts
        ])
        centroid = vectors.mean(dim=0)
        centroids[main] = F.normalize(centroid, p=2, dim=-1)

    return centroids


@torch.no_grad()
def route_vector(
    vector: torch.Tensor,
    centroids: Dict[str, torch.Tensor],
    *,
    similarity_threshold: float,
    margin_threshold: float = 0.0,
) -> RouteResult:
    scores = sorted(
        (
            (
                main,
                float(
                    F.cosine_similarity(
                        vector.unsqueeze(0),
                        centroid.unsqueeze(0),
                    ).item()
                ),
            )
            for main, centroid in centroids.items()
        ),
        key=lambda item: item[1],
        reverse=True,
    )
    if len(scores) < 2:
        raise RuntimeError("At least two NDC centroids are required.")

    top1_main, top1 = scores[0]
    _, top2 = scores[1]
    margin = top1 - top2
    accepted = (
        top1 >= similarity_threshold
        and margin >= margin_threshold
    )

    return RouteResult(
        predicted_main=top1_main,
        predicted_name=NDC_MAIN[top1_main],
        top1_similarity=top1,
        top2_similarity=top2,
        margin=margin,
        state="ACCEPT" if accepted else "UNKNOWN",
    )


@torch.no_grad()
def route_text(
    model: LanguageModel,
    tokenizer: Tokenizer,
    text: str,
    centroids: Dict[str, torch.Tensor],
    *,
    similarity_threshold: float,
    margin_threshold: float = 0.0,
    pooling: str = "mean",
) -> RouteResult:
    vector = encode_text(model, tokenizer, text, pooling=pooling)
    return route_vector(
        vector,
        centroids,
        similarity_threshold=similarity_threshold,
        margin_threshold=margin_threshold,
    )
