# semantic_radius.py
#
# Class-specific semantic radius detector for LLM_SEM.
#
# Radii are learned only from known development samples using leave-one-out
# class centroids. Unknown development samples are not required.

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import torch
import torch.nn.functional as F

from semantic_eval import LabeledSentence
from semantic_router import SemanticRouter, _centroids_from_vectors


DEFAULT_RADIUS_QUANTILE = 0.90
DEFAULT_RADIUS_SCALE = 1.00


@dataclass
class RadiusStats:
    label: str
    radius: float
    sample_count: int
    mean_distance: float
    max_distance: float


def _nearest_rank_quantile(values: Sequence[float], q: float) -> float:
    if not values:
        raise ValueError("values must not be empty.")
    if not 0.0 <= q <= 1.0:
        raise ValueError("q must be between 0.0 and 1.0.")
    ordered = sorted(float(value) for value in values)
    index = int(round(q * (len(ordered) - 1)))
    return ordered[index]


def fit_class_radii(
    router: SemanticRouter,
    samples: Sequence[LabeledSentence],
    quantile: float = DEFAULT_RADIUS_QUANTILE,
    scale: float = DEFAULT_RADIUS_SCALE,
) -> Dict[str, RadiusStats]:
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be between 0.0 and 1.0.")
    if scale <= 0.0:
        raise ValueError("scale must be > 0.")

    vectors = [router._encode_tensor(sample.text) for sample in samples]
    distances: Dict[str, List[float]] = {}

    for index, sample in enumerate(samples):
        train_samples = [
            other for j, other in enumerate(samples) if j != index
        ]
        train_vectors = [
            vector for j, vector in enumerate(vectors) if j != index
        ]
        centroids = _centroids_from_vectors(train_samples, train_vectors)

        if sample.label not in centroids:
            continue

        similarity = float(
            F.cosine_similarity(
                vectors[index],
                centroids[sample.label],
                dim=0,
            ).item()
        )
        distance = 1.0 - similarity
        distances.setdefault(sample.label, []).append(distance)

    stats: Dict[str, RadiusStats] = {}
    for label, values in distances.items():
        base_radius = _nearest_rank_quantile(values, quantile)
        stats[label] = RadiusStats(
            label=label,
            radius=base_radius * scale,
            sample_count=len(values),
            mean_distance=sum(values) / len(values),
            max_distance=max(values),
        )

    return stats


def classify_with_class_radius(
    ranked,
    radii: Dict[str, RadiusStats],
):
    if not ranked:
        raise ValueError("ranked routes must not be empty.")

    top1 = ranked[0]
    stats = radii.get(top1.label)
    if stats is None:
        raise KeyError(f"No radius available for class: {top1.label}")

    is_unknown = top1.distance > stats.radius
    return is_unknown, stats.radius
