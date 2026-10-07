#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LLM_SEM v0.18.13 Hierarchical Beam NDC Router.

v0.18.12 used a hard Stage-1 main-class gate. If Stage 1 chose the wrong main
class, the correct 3-digit code was never considered.

v0.18.13 replaces that hard restriction with a soft main-class prior:
  score(code) = code_similarity + main_bonus(if code matches Stage-1 main)

All selected 3-digit codes remain eligible. The Stage-1 main decision becomes
a prior rather than an irreversible filter.

UNKNOWN remains controlled by the stable contrastive main router.
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


@dataclass(frozen=True)
class BeamNDCDecision:
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
    main_gate_score: float


class BeamHierarchicalNDCRouter:
    def __init__(
        self,
        model,
        tokenizer,
        *,
        pooling: str = "mean",
        main_bonus: float = 0.025,
        code_similarity_threshold: float = 0.72,
        beam_margin_threshold: float = 0.01,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.pooling = pooling
        self.main_bonus = float(main_bonus)
        self.code_similarity_threshold = float(code_similarity_threshold)
        self.beam_margin_threshold = float(beam_margin_threshold)

        self.main_router = StableNDCRouter(model, tokenizer, pooling=pooling)

        self.code_prototypes: Dict[str, Tuple[torch.Tensor, ...]] = {}
        for code, texts in NDC_CODE_SEEDS.items():
            self.code_prototypes[code] = tuple(
                F.normalize(
                    encode_text(model, tokenizer, text, pooling=pooling),
                    p=2,
                    dim=-1,
                )
                for text in texts
            )

    @torch.no_grad()
    def route(self, text: str) -> BeamNDCDecision:
        stage1 = self.main_router.route(text)

        if not stage1.accepted:
            return BeamNDCDecision(
                text=text,
                state="UNKNOWN",
                stage1_main=None,
                stage1_name=None,
                ndc_code=None,
                ndc_code_name=None,
                code_similarity=None,
                beam_score=None,
                beam_margin=None,
                rescued_main=False,
                main_gate_score=stage1.gate_score,
            )

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

        scored = []
        for code, prototypes in self.code_prototypes.items():
            similarity = max(
                float(torch.dot(vector, prototype).item())
                for prototype in prototypes
            )
            prior = self.main_bonus if code.startswith(stage1.ndc_main or "") else 0.0
            beam_score = similarity + prior
            scored.append((beam_score, similarity, code))

        scored.sort(reverse=True)
        top_score, top_similarity, top_code = scored[0]
        second_score = scored[1][0] if len(scored) > 1 else -1.0
        margin = top_score - second_score

        if (
            top_similarity < self.code_similarity_threshold
            or margin < self.beam_margin_threshold
        ):
            return BeamNDCDecision(
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
                main_gate_score=stage1.gate_score,
            )

        rescued = not top_code.startswith(stage1.ndc_main or "")

        return BeamNDCDecision(
            text=text,
            state="ACCEPT",
            stage1_main=stage1.ndc_main,
            stage1_name=stage1.ndc_name,
            ndc_code=top_code,
            ndc_code_name=NDC_CODES[top_code],
            code_similarity=top_similarity,
            beam_score=top_score,
            beam_margin=margin,
            rescued_main=rescued,
            main_gate_score=stage1.gate_score,
        )
