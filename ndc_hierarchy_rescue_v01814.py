#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LLM_SEM v0.18.14 Hierarchical Beam Router with Fine-Code Rescue.

Targets the three remaining v0.18.13 failures:
  - 490 (medicine) confused with 596 (food/cooking)
  - 596 rejected by main unknown gate before fine routing
  - 930 (English/American literature) confused with generic 900 literature

Changes:
  1) add targeted fine-code prototypes / hard negatives for 490, 596, 900, 930
  2) if Stage-1 returns UNKNOWN, allow a restricted fine-code rescue only when
     a selected 3-digit code has very high similarity and sufficient margin
  3) keep UNKNOWN protection by requiring a stricter rescue threshold

This still uses only selected codes already defined in ndc.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from ndc import NDC_CODES
from ndc_semantic_router_v0182 import encode_text
from ndc_runtime_v01811 import StableNDCRouter
from ndc_hierarchy_v01812 import NDC_CODE_SEEDS


EXTRA_CODE_SEEDS: Dict[str, Tuple[str, ...]] = {
    "490": (
        "病気と治療を行う医学",
        "医療と診断について",
        "疾患の治療と医学",
        "患者の病気を治療する",
    ),
    "596": (
        "料理とレシピを扱う",
        "食品を調理する",
        "食事と調理法について",
        "料理の作り方と食品",
    ),
    "900": (
        "文学一般について",
        "文学作品全般を論じる",
        "文学という分野",
    ),
    "930": (
        "英米文学作品を読む",
        "英語圏の小説と文学",
        "イギリス文学とアメリカ文学",
        "英文学作品について",
    ),
}


@dataclass(frozen=True)
class RescueNDCDecision:
    text: str
    state: str
    stage1_main: str | None
    stage1_name: str | None
    ndc_code: str | None
    ndc_code_name: str | None
    code_similarity: float | None
    beam_score: float | None
    beam_margin: float | None
    rescued_main: bool
    rescued_unknown: bool
    main_gate_score: float


class RescueBeamNDCRouter:
    def __init__(
        self,
        model,
        tokenizer,
        *,
        pooling: str = "mean",
        main_bonus: float = 0.025,
        code_similarity_threshold: float = 0.72,
        beam_margin_threshold: float = 0.01,
        unknown_rescue_similarity: float = 0.88,
        unknown_rescue_margin: float = 0.04,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.pooling = pooling
        self.main_bonus = float(main_bonus)
        self.code_similarity_threshold = float(code_similarity_threshold)
        self.beam_margin_threshold = float(beam_margin_threshold)
        self.unknown_rescue_similarity = float(unknown_rescue_similarity)
        self.unknown_rescue_margin = float(unknown_rescue_margin)

        self.main_router = StableNDCRouter(model, tokenizer, pooling=pooling)

        seeds = {
            code: tuple(texts) + EXTRA_CODE_SEEDS.get(code, ())
            for code, texts in NDC_CODE_SEEDS.items()
        }

        self.code_prototypes: Dict[str, Tuple[torch.Tensor, ...]] = {}
        for code, texts in seeds.items():
            self.code_prototypes[code] = tuple(
                F.normalize(
                    encode_text(model, tokenizer, text, pooling=pooling),
                    p=2,
                    dim=-1,
                )
                for text in texts
            )

    @torch.no_grad()
    def _rank_codes(
        self,
        vector: torch.Tensor,
        stage1_main: str | None,
    ):
        scored = []
        for code, prototypes in self.code_prototypes.items():
            similarity = max(
                float(torch.dot(vector, prototype).item())
                for prototype in prototypes
            )
            prior = (
                self.main_bonus
                if stage1_main is not None and code.startswith(stage1_main)
                else 0.0
            )
            scored.append((similarity + prior, similarity, code))

        scored.sort(reverse=True)
        return scored

    @torch.no_grad()
    def route(self, text: str) -> RescueNDCDecision:
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
            if (
                top_similarity >= self.unknown_rescue_similarity
                and margin >= self.unknown_rescue_margin
            ):
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
