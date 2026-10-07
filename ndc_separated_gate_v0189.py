#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LLM_SEM v0.18.9 Separated NDC Classification and Unknown Gate.

Classification:
    Always use the confusion-aware augmented prototype router.

Unknown detection:
    Evaluate independent evidence from
      - baseline/augmented router agreement
      - augmented nearest similarity
      - baseline nearest similarity
      - augmented class margin
      - baseline class margin

The classifier and gate are intentionally decoupled. The NDC label comes from
the augmented router; the gate only decides ACCEPT versus UNKNOWN.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

import torch

from ndc_semantic_router_v0183 import route_vector_multi


@dataclass(frozen=True)
class SeparatedGateResult:
    predicted_main: str
    baseline_main: str
    augmented_main: str
    agreed: bool
    baseline_similarity: float
    augmented_similarity: float
    baseline_margin: float
    augmented_margin: float
    evidence_score: float
    accepted: bool
    state: str


@torch.no_grad()
def route_separated(
    vector: torch.Tensor,
    baseline_prototypes: Dict[str, Sequence[torch.Tensor]],
    augmented_prototypes: Dict[str, Sequence[torch.Tensor]],
    *,
    top_k: int,
    max_weight: float,
    agreement_bonus: float,
    augmented_similarity_weight: float,
    baseline_similarity_weight: float,
    augmented_margin_weight: float,
    baseline_margin_weight: float,
    evidence_threshold: float,
) -> SeparatedGateResult:
    baseline = route_vector_multi(
        vector,
        baseline_prototypes,
        similarity_threshold=-1.0,
        margin_threshold=-1.0,
        top_k=top_k,
        max_weight=max_weight,
    )
    augmented = route_vector_multi(
        vector,
        augmented_prototypes,
        similarity_threshold=-1.0,
        margin_threshold=-1.0,
        top_k=top_k,
        max_weight=max_weight,
    )

    agreed = baseline.predicted_main == augmented.predicted_main

    # Classification is ALWAYS the augmented router.
    predicted_main = augmented.predicted_main

    evidence = 0.0
    if agreed:
        evidence += agreement_bonus

    evidence += augmented_similarity_weight * augmented.nearest_similarity
    evidence += baseline_similarity_weight * baseline.nearest_similarity
    evidence += augmented_margin_weight * augmented.class_margin
    evidence += baseline_margin_weight * baseline.class_margin

    accepted = evidence >= evidence_threshold

    return SeparatedGateResult(
        predicted_main=predicted_main,
        baseline_main=baseline.predicted_main,
        augmented_main=augmented.predicted_main,
        agreed=agreed,
        baseline_similarity=baseline.nearest_similarity,
        augmented_similarity=augmented.nearest_similarity,
        baseline_margin=baseline.class_margin,
        augmented_margin=augmented.class_margin,
        evidence_score=evidence,
        accepted=accepted,
        state="ACCEPT" if accepted else "UNKNOWN",
    )
