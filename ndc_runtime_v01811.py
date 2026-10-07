#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LLM_SEM v0.18.11 Stable NDC Runtime Router.

This module packages the v0.18.10 experiment into a reusable runtime component.

Architecture
------------
Frozen semantic encoder
    -> confusion-aware augmented NDC classifier
    -> contrastive known-vs-unknown gate

The classifier decides the NDC main class.
The contrastive gate decides ACCEPT vs UNKNOWN.

Default parameters are taken from the v0.18.10 DEV-selected operating point:
    contrast_weight = 2.00
    margin_weight = 1.50
    known_similarity_weight = 0.25
    gate_threshold = 0.240370
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import torch

from ndc import NDC_MAIN
from ndc_semantic_router_v0182 import encode_text
from ndc_semantic_router_v0183 import build_prototypes
from ndc_prototypes_v0187 import AUGMENTED_NDC_SEEDS
from ndc_unknown_gate_v01810 import route_contrastive


DEFAULT_UNKNOWN_TEXTS: Tuple[str, ...] = (
    "これはどういうこと",
    "それの説明",
    "XYZABC",
    "???",
    "未定義対象ガンマ",
    "意味不明な文字列qqq",
)


@dataclass(frozen=True)
class NDCRuntimeDecision:
    text: str
    ndc_main: str | None
    ndc_name: str | None
    accepted: bool
    state: str
    known_similarity: float
    unknown_similarity: float
    contrast: float
    margin: float
    gate_score: float


class StableNDCRouter:
    def __init__(
        self,
        model,
        tokenizer,
        *,
        pooling: str = "mean",
        contrast_weight: float = 2.00,
        margin_weight: float = 1.50,
        known_similarity_weight: float = 0.25,
        gate_threshold: float = 0.240370,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.pooling = pooling
        self.contrast_weight = float(contrast_weight)
        self.margin_weight = float(margin_weight)
        self.known_similarity_weight = float(known_similarity_weight)
        self.gate_threshold = float(gate_threshold)

        self.augmented_prototypes = build_prototypes(
            model,
            tokenizer,
            seeds=AUGMENTED_NDC_SEEDS,
            pooling=pooling,
        )
        self.unknown_prototypes = tuple(
            encode_text(
                model,
                tokenizer,
                text,
                pooling=pooling,
            )
            for text in DEFAULT_UNKNOWN_TEXTS
        )

    @torch.no_grad()
    def route(self, text: str) -> NDCRuntimeDecision:
        vector = encode_text(
            self.model,
            self.tokenizer,
            text,
            pooling=self.pooling,
        )

        result = route_contrastive(
            vector,
            self.augmented_prototypes,
            self.unknown_prototypes,
            top_k=3,
            max_weight=1.0,
            contrast_weight=self.contrast_weight,
            margin_weight=self.margin_weight,
            known_similarity_weight=self.known_similarity_weight,
            gate_threshold=self.gate_threshold,
        )

        if result.accepted:
            ndc_main = result.predicted_main
            ndc_name = NDC_MAIN.get(ndc_main)
            state = "ACCEPT"
        else:
            ndc_main = None
            ndc_name = None
            state = "UNKNOWN"

        return NDCRuntimeDecision(
            text=text,
            ndc_main=ndc_main,
            ndc_name=ndc_name,
            accepted=result.accepted,
            state=state,
            known_similarity=result.known_similarity,
            unknown_similarity=result.unknown_similarity,
            contrast=result.contrast,
            margin=result.class_margin,
            gate_score=result.gate_score,
        )
