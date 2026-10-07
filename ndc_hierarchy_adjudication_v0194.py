#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LLM_SEM v0.19.4 Conditional Rescue + Pairwise Adjudication.

Builds on v0.19.2 stable-generalization candidate.

Local fixes only:
1) lexical-supported UNKNOWN rescue:
   If the main UNKNOWN gate rejects a query but deterministic NDC evidence
   supports the same top semantic code, allow a lower rescue threshold.

2) pairwise adjudication:
   - 007 vs 548: prefer 548 when engineering/system-architecture evidence exists
   - 900 vs 910: prefer 910 when Japanese-literature evidence exists

Global thresholds, encoder, and expanded UNKNOWN prototypes are unchanged.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Iterable

import torch
import torch.nn.functional as F

from ndc import NDC_CODES, classify, normalize_text
from ndc_semantic_router_v0182 import encode_text
from ndc_hierarchy_unknown_v0192 import CoverageUnknownNDCRouter
from ndc_hierarchy_rescue_v01814 import RescueNDCDecision


PAIRWISE_548_MARKERS = (
    "情報工学",
    "コンピュータ工学",
    "計算機工学",
    "計算機アーキテクチャ",
    "コンピュータアーキテクチャ",
    "システム構成",
    "計算機システム",
)

PAIRWISE_910_MARKERS = (
    "日本文学",
    "日本の文学",
    "日本で書かれ",
    "国内の文学",
    "和歌",
    "俳句",
)

LEXICAL_RESCUE_CODES = {"150", "490"}


def contains_any(text: str, markers: Iterable[str]) -> bool:
    normalized = normalize_text(text)
    return any(normalize_text(marker) in normalized for marker in markers)


class ConditionalAdjudicationNDCRouter(CoverageUnknownNDCRouter):
    def __init__(
        self,
        model,
        tokenizer,
        *,
        lexical_rescue_similarity: float = 0.84,
        lexical_rescue_margin: float = 0.04,
        pair_bonus_548: float = 0.08,
        pair_bonus_910: float = 0.08,
        **kwargs,
    ) -> None:
        super().__init__(model, tokenizer, **kwargs)
        self.lexical_rescue_similarity = float(lexical_rescue_similarity)
        self.lexical_rescue_margin = float(lexical_rescue_margin)
        self.pair_bonus_548 = float(pair_bonus_548)
        self.pair_bonus_910 = float(pair_bonus_910)
        self._adjudication_text = ""

    def _rank_codes(self, vector, stage1_main):
        scored = super()._rank_codes(vector, stage1_main)
        text = self._adjudication_text

        adjusted = []
        for score, similarity, code in scored:
            bonus = 0.0
            if code == "548" and contains_any(text, PAIRWISE_548_MARKERS):
                bonus += self.pair_bonus_548
            if code == "910" and contains_any(text, PAIRWISE_910_MARKERS):
                bonus += self.pair_bonus_910
            adjusted.append((score + bonus, similarity, code))

        adjusted.sort(reverse=True)
        return adjusted

    @torch.no_grad()
    def route(self, text: str) -> RescueNDCDecision:
        self._adjudication_text = text
        try:
            stage1 = self.main_router.route(text)

            vector = F.normalize(
                encode_text(
                    self.model,
                    self.tokenizer,
                    text,
                    pooling=self.pooling,
                ),
                p=2,
                dim=-1,
            )

            scored = self._rank_codes(
                vector,
                stage1.ndc_main if stage1.accepted else None,
            )

            top_score, top_similarity, top_code = scored[0]
            second_score = scored[1][0] if len(scored) > 1 else -1.0
            margin = top_score - second_score

            if not stage1.accepted:
                lexical = classify(text)
                lexical_supported = (
                    lexical.state == "CLASSIFIED"
                    and lexical.code == top_code
                    and top_code in LEXICAL_RESCUE_CODES
                )

                strict_rescue = (
                    top_similarity >= self.unknown_rescue_similarity
                    and margin >= self.unknown_rescue_margin
                )
                lexical_rescue = (
                    lexical_supported
                    and top_similarity >= self.lexical_rescue_similarity
                    and margin >= self.lexical_rescue_margin
                )

                if strict_rescue or lexical_rescue:
                    return RescueNDCDecision(
                        text=text,
                        state="ACCEPT",
                        stage1_main=None,
                        stage1_name=None,
                        ndc_code=top_code,
                        ndc_code_name=NDC_CODES[top_code],
                        code_similarity=top_similarity,
                        beam_score=top_score,
                        beam_margin=margin,
                        rescued_main=False,
                        rescued_unknown=True,
                        main_gate_score=stage1.gate_score,
                    )

                return RescueNDCDecision(
                    text=text,
                    state="UNKNOWN",
                    stage1_main=None,
                    stage1_name=None,
                    ndc_code=None,
                    ndc_code_name=None,
                    code_similarity=top_similarity,
                    beam_score=top_score,
                    beam_margin=margin,
                    rescued_main=False,
                    rescued_unknown=False,
                    main_gate_score=stage1.gate_score,
                )

            if (
                top_similarity < self.code_similarity_threshold
                or margin < self.beam_margin_threshold
            ):
                return RescueNDCDecision(
                    text=text,
                    state="MAIN_ONLY",
                    stage1_main=stage1.ndc_main,
                    stage1_name=stage1.ndc_name,
                    ndc_code=None,
                    ndc_code_name=None,
                    code_similarity=top_similarity,
                    beam_score=top_score,
                    beam_margin=margin,
                    rescued_main=False,
                    rescued_unknown=False,
                    main_gate_score=stage1.gate_score,
                )

            rescued_main = not top_code.startswith(stage1.ndc_main or "")
            return RescueNDCDecision(
                text=text,
                state="ACCEPT",
                stage1_main=stage1.ndc_main,
                stage1_name=stage1.ndc_name,
                ndc_code=top_code,
                ndc_code_name=NDC_CODES[top_code],
                code_similarity=top_similarity,
                beam_score=top_score,
                beam_margin=margin,
                rescued_main=rescued_main,
                rescued_unknown=False,
                main_gate_score=stage1.gate_score,
            )
        finally:
            self._adjudication_text = ""
