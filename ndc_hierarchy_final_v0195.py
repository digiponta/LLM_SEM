#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LLM_SEM v0.19.5 Final 830 Boundary Stabilization.

Builds on v0.19.4.

Adds only a small local bonus for NDC 830 when the query contains explicit
English-language evidence. All global thresholds, UNKNOWN logic, and other
pairwise adjudication rules remain unchanged.
"""

from __future__ import annotations

from ndc import normalize_text
from ndc_hierarchy_adjudication_v0194 import ConditionalAdjudicationNDCRouter


ENGLISH_830_MARKERS = (
    "英語",
    "英文",
    "英単語",
    "英文法",
    "english",
)


class FinalStableNDCRouter(ConditionalAdjudicationNDCRouter):
    def __init__(
        self,
        model,
        tokenizer,
        *,
        bonus_830: float = 0.025,
        **kwargs,
    ) -> None:
        super().__init__(model, tokenizer, **kwargs)
        self.bonus_830 = float(bonus_830)

    def _rank_codes(self, vector, stage1_main):
        scored = super()._rank_codes(vector, stage1_main)
        text = normalize_text(self._adjudication_text)
        has_english = any(
            normalize_text(marker) in text
            for marker in ENGLISH_830_MARKERS
        )

        if not has_english:
            return scored

        adjusted = []
        for score, similarity, code in scored:
            if code == "830":
                score += self.bonus_830
            adjusted.append((score, similarity, code))

        adjusted.sort(reverse=True)
        return adjusted
