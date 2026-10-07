#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LLM_SEM v0.18.10 Contrastive Unknown Gate.

The NDC classifier is fixed to the confusion-aware augmented prototype router.

Unknown detection is separated into a contrastive decision:
    known evidence  = nearest NDC prototype similarity
    unknown evidence = nearest UNKNOWN prototype similarity
    contrast = known evidence - unknown evidence

The gate also uses the augmented class margin as secondary evidence.

This directly models "known-domain vs underspecified/OOD" instead of trying to
infer unknownness from NDC confidence alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import torch
import torch.nn.functional as F

from ndc_semantic_router_v0183 import route_vector_multi


@dataclass(frozen=True)
class ContrastiveGateResult:
    predicted_main: str
    known_similarity: float
    unknown_similarity: float
    contrast: float
    class_margin: float
    gate_score: float
    accepted: bool
    state: str


@torch.no_grad()
def nearest_similarity(
    vector: torch.Tensor,
    prototypes: Sequence[torch.Tensor],
) -> float:
    if not prototypes:
        raise ValueError("prototypes must not be empty")
    return max(
        float(
            F.cosine_similarity(
                vector.unsqueeze(0),
                proto.unsqueeze(0),
            ).item()
        )
        for proto in prototypes
    )


@torch.no_grad()
def route_contrastive(
    vector: torch.Tensor,
    augmented_prototypes: Dict[str, Sequence[torch.Tensor]],
    unknown_prototypes: Sequence[torch.Tensor],
    *,
    top_k: int = 3,
    max_weight: float = 1.0,
    contrast_weight: float = 1.0,
    margin_weight: float = 0.5,
    known_similarity_weight: float = 0.25,
    gate_threshold: float = 0.0,
) -> ContrastiveGateResult:
    ndc = route_vector_multi(
        vector,
        augmented_prototypes,
        similarity_threshold=-1.0,
        margin_threshold=-1.0,
        top_k=top_k,
        max_weight=max_weight,
    )

    known_similarity = ndc.nearest_similarity
    unknown_similarity = nearest_similarity(vector, unknown_prototypes)
    contrast = known_similarity - unknown_similarity

    gate_score = (
        contrast_weight * contrast
        + margin_weight * ndc.class_margin
        + known_similarity_weight * known_similarity
    )

    accepted = gate_score >= gate_threshold

    return ContrastiveGateResult(
        predicted_main=ndc.predicted_main,
        known_similarity=known_similarity,
        unknown_similarity=unknown_similarity,
        contrast=contrast,
        class_margin=ndc.class_margin,
        gate_score=gate_score,
        accepted=accepted,
        state="ACCEPT" if accepted else "UNKNOWN",
    )
