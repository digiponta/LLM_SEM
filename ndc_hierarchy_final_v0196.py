#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LLM_SEM v0.19.6 Final 930 Boundary Stabilization.

Builds on v0.19.5.

Adds only a small local bonus for NDC 930 when explicit English/American
literature evidence is present. All global thresholds, UNKNOWN logic,
and previous local adjudication rules remain unchanged.
"""

from __future__ import annotations

from ndc import normalize_text
from ndc_hierarchy_final_v0195 import FinalStableNDCRouter


LITERATURE_930_MARKERS = (
    "英米文学",
    "英国文学",
    "米国文学",
    "イギリス文学",
    "アメリカ文学",
    "英国",
    "米国",
    "英語圏",
)


class FinalStableNDCRouterV0196(FinalStableNDCRouter):
    def __init__(
        self,
        model,
        tokenizer,
        *,
        bonus_930: float = 0.025,
        **kwargs,
    ) -> None:
        super().__init__(model, tokenizer, **kwargs)
        self.bonus_930 = float(bonus_930)

    def _rank_codes(self, vector, stage1_main):
        scored = super()._rank_codes(vector, stage1_main)
        text = normalize_text(self._adjudication_text)
        has_930_evidence = any(
            normalize_text(marker) in text
            for marker in LITERATURE_930_MARKERS
        )

        if not has_930_evidence:
            return scored

        adjusted = []
        for score, similarity, code in scored:
            if code == "930":
                score += self.bonus_930
            adjusted.append((score, similarity, code))

        adjusted.sort(reverse=True)
        return adjusted
