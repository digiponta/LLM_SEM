#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LLM_SEM v0.18.8 Dual-Router Consensus Gate.

The v0.18.7 augmented prototype router improved raw routing but required a
harsh global threshold that severely reduced known acceptance.

v0.18.8 separates two concerns:
  1) domain routing
  2) unknown rejection

Two independent semantic routers are evaluated:
  - TRAIN-only baseline prototypes
  - confusion-aware augmented prototypes

Consensus rule:
  - if both routers predict the same NDC main class, evaluate a relaxed
    agreement threshold;
  - if they disagree, reject as UNKNOWN.

This uses agreement as structural confidence instead of forcing one global
similarity threshold to do both classification and OOD rejection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

import torch

from ndc_semantic_router_v0183 import route_vector_multi


@dataclass(frozen=True)
class ConsensusResult:
    predicted_main: str | None
    baseline_main: str
    augmented_main: str
    baseline_similarity: float
    augmented_similarity: float
    baseline_margin: float
    augmented_margin: float
    agreed: bool
    accepted: bool
    state: str


@torch.no_grad()
def route_consensus(
    vector: torch.Tensor,
    baseline_prototypes: Dict[str, Sequence[torch.Tensor]],
    augmented_prototypes: Dict[str, Sequence[torch.Tensor]],
    *,
    top_k: int = 3,
    max_weight: float = 1.0,
    similarity_threshold: float = 0.80,
    margin_threshold: float = 0.0,
) -> ConsensusResult:
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

    similarity_ok = (
        min(
            baseline.nearest_similarity,
            augmented.nearest_similarity,
        )
        >= similarity_threshold
    )
    margin_ok = (
        min(
            baseline.class_margin,
            augmented.class_margin,
        )
        >= margin_threshold
    )

    accepted = agreed and similarity_ok and margin_ok
    predicted_main = baseline.predicted_main if agreed else None

    return ConsensusResult(
        predicted_main=predicted_main,
        baseline_main=baseline.predicted_main,
        augmented_main=augmented.predicted_main,
        baseline_similarity=baseline.nearest_similarity,
        augmented_similarity=augmented.nearest_similarity,
        baseline_margin=baseline.class_margin,
        augmented_margin=augmented.class_margin,
        agreed=agreed,
        accepted=accepted,
        state="ACCEPT" if accepted else "UNKNOWN",
    )
