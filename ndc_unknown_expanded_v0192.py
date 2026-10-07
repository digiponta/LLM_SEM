#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LLM_SEM v0.19.2 Expanded Unknown Runtime Gate.

This module strengthens UNKNOWN detection without changing NDC code
classification prototypes or the frozen semantic encoder.

It extends the v0.18.11 contrastive UNKNOWN prototype set with additional
underspecified, referent-free, and gibberish-like examples.
"""

from __future__ import annotations

from typing import Tuple

from ndc_runtime_v01811 import StableNDCRouter
from ndc_semantic_router_v0182 import encode_text


EXTRA_UNKNOWN_TEXTS: Tuple[str, ...] = (
    "どんな意味ですか",
    "意味を教えてください",
    "これは何ですか",
    "この件について",
    "この話について",
    "それについて知りたい",
    "詳しく説明してください",
    "もう少し詳しく",
    "対象未指定",
    "対象がありません",
    "何を指しているかわからない",
    "意味対象なし",
    "内容が示されていない質問",
    "説明してほしいもの",
    "何かを説明する依頼",
    "abcdefghij",
    "qwertyuiop",
    "!!!!!",
    "???!!!",
    "未定義項目シータ",
)


class ExpandedUnknownNDCRouter(StableNDCRouter):
    def __init__(self, model, tokenizer, **kwargs) -> None:
        super().__init__(model, tokenizer, **kwargs)

        extras = tuple(
            encode_text(
                model,
                tokenizer,
                text,
                pooling=self.pooling,
            )
            for text in EXTRA_UNKNOWN_TEXTS
        )
        self.unknown_prototypes = tuple(self.unknown_prototypes) + extras
