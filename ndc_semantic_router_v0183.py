#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LLM_SEM v0.18.3 Multi-Prototype NDC Semantic Router.

Motivation
----------
v0.18.2 showed that one centroid per NDC main class over-compresses the current
semantic space.  v0.18.3 keeps multiple semantic prototypes per class and scores
a query against the strongest prototypes instead of only a single class mean.

Pipeline:
  text
    -> semantic vector (mean pooling)
    -> cosine similarity to every NDC prototype
    -> per-class top-k prototype aggregation
    -> top-1 class / top-2 class margin
    -> calibrated ACCEPT / UNKNOWN

UNKNOWN remains outside NDC and is never mapped to 000.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F

from model import LanguageModel
from tokenizer import Tokenizer
from ndc import NDC_MAIN
from ndc_semantic_router_v0182 import NDC_MAIN_SEEDS, encode_text


@dataclass(frozen=True)
class PrototypeRouteResult:
    predicted_main: str
    predicted_name: str
    top1_score: float
    top2_score: float
    class_margin: float
    nearest_similarity: float
    state: str

    @property
    def accepted(self) -> bool:
        return self.state == "ACCEPT"


@torch.no_grad()
def build_prototypes(
    model: LanguageModel,
    tokenizer: Tokenizer,
    *,
    seeds: Dict[str, Sequence[str]] | None = None,
    pooling: str = "mean",
) -> Dict[str, Tuple[torch.Tensor, ...]]:
    source = seeds or NDC_MAIN_SEEDS
    result: Dict[str, Tuple[torch.Tensor, ...]] = {}
    for main, texts in source.items():
        result[main] = tuple(
            encode_text(model, tokenizer, text, pooling=pooling)
            for text in texts
        )
    return result


def _class_score(
    vector: torch.Tensor,
    prototypes: Sequence[torch.Tensor],
    *,
    top_k: int,
    max_weight: float,
) -> Tuple[float, float]:
    if not prototypes:
        raise ValueError("prototype list must not be empty")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if not 0.0 <= max_weight <= 1.0:
        raise ValueError("max_weight must be between 0 and 1")

    sims = sorted(
        (
            float(
                F.cosine_similarity(
                    vector.unsqueeze(0),
                    proto.unsqueeze(0),
                ).item()
            )
            for proto in prototypes
        ),
        reverse=True,
    )
    selected = sims[: min(top_k, len(sims))]
    mean_top = sum(selected) / len(selected)
    nearest = sims[0]

    # Blend nearest-prototype evidence with local prototype consistency.
    score = max_weight * nearest + (1.0 - max_weight) * mean_top
    return score, nearest


@torch.no_grad()
def route_vector_multi(
    vector: torch.Tensor,
    prototypes: Dict[str, Sequence[torch.Tensor]],
    *,
    similarity_threshold: float,
    margin_threshold: float,
    top_k: int = 2,
    max_weight: float = 0.60,
) -> PrototypeRouteResult:
    scores: List[Tuple[str, float, float]] = []
    for main, class_prototypes in prototypes.items():
        score, nearest = _class_score(
            vector,
            class_prototypes,
            top_k=top_k,
            max_weight=max_weight,
        )
        scores.append((main, score, nearest))

    scores.sort(key=lambda row: row[1], reverse=True)
    if len(scores) < 2:
        raise RuntimeError("At least two NDC classes are required.")

    top1_main, top1_score, top1_nearest = scores[0]
    _, top2_score, _ = scores[1]
    margin = top1_score - top2_score

    accepted = (
        top1_nearest >= similarity_threshold
        and margin >= margin_threshold
    )

    return PrototypeRouteResult(
        predicted_main=top1_main,
        predicted_name=NDC_MAIN[top1_main],
        top1_score=top1_score,
        top2_score=top2_score,
        class_margin=margin,
        nearest_similarity=top1_nearest,
        state="ACCEPT" if accepted else "UNKNOWN",
    )
