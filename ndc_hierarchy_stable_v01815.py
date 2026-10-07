#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LLM_SEM v0.18.15 Stable Hybrid 3-digit NDC Router.

Based on v0.18.14 rescue beam routing.

Final refinement:
  - semantic beam score remains primary
  - deterministic ndc.py keyword classification contributes only a small
    exact-code prior for selected 3-digit codes

This is intended as a tie-breaker for semantic boundary cases such as
490 (medicine) vs 596 (food/cooking), not as a replacement for semantic routing.
"""

from __future__ import annotations

from dataclasses import dataclass

from ndc import classify
from ndc_hierarchy_rescue_v01814 import RescueBeamNDCRouter, RescueNDCDecision


class StableHybridNDCRouter(RescueBeamNDCRouter):
    def __init__(
        self,
        model,
        tokenizer,
        *,
        pooling: str = "mean",
        main_bonus: float = 0.025,
        keyword_bonus: float = 0.080,
        code_similarity_threshold: float = 0.72,
        beam_margin_threshold: float = 0.01,
        unknown_rescue_similarity: float = 0.88,
        unknown_rescue_margin: float = 0.04,
    ) -> None:
        super().__init__(
            model,
            tokenizer,
            pooling=pooling,
            main_bonus=main_bonus,
            code_similarity_threshold=code_similarity_threshold,
            beam_margin_threshold=beam_margin_threshold,
            unknown_rescue_similarity=unknown_rescue_similarity,
            unknown_rescue_margin=unknown_rescue_margin,
        )
        self.keyword_bonus = float(keyword_bonus)
        self._active_keyword_code = None

    def _rank_codes(self, vector, stage1_main):
        lexical = self._active_keyword_code
        scored = []

        for code, prototypes in self.code_prototypes.items():
            similarity = max(
                float((vector * prototype).sum().item())
                for prototype in prototypes
            )

            score = similarity
            if stage1_main is not None and code.startswith(stage1_main):
                score += self.main_bonus
            if lexical == code:
                score += self.keyword_bonus

            scored.append((score, similarity, code))

        scored.sort(reverse=True)
        return scored

    def route(self, text: str) -> RescueNDCDecision:
        lexical = classify(text)
        self._active_keyword_code = (
            lexical.code
            if lexical.state == "CLASSIFIED"
            and lexical.code in self.code_prototypes
            else None
        )
        try:
            return super().route(text)
        finally:
            self._active_keyword_code = None
