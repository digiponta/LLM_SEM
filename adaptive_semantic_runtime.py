# adaptive_semantic_runtime.py
#
# LLM_SEM v0.4.6 Local Evidence Runtime
#
# Runtime policy distilled from v0.4.1-v0.4.5 experiments:
#   - Base Router is fixed on the base benchmark.
#   - Adaptive Memory is independent and can change online.
#   - Multi-Prototype memory: 2 prototypes per label by default.
#   - Normal acceptance:
#         memory_sim >= 0.80 AND memory_label == base_label
#   - Conditional adaptive override:
#         memory_label != base_label
#         AND memory_sim >= 0.92
#         AND top-k local majority label == memory_label
#         AND local purity >= 0.60
#   - Default local k = 3.
#   - Prototype margin remains diagnostic only.

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Sequence

import torch
import torch.nn.functional as F

from semantic_eval import LabeledSentence


@dataclass(frozen=True)
class RuntimeMemoryVector:
    label: str
    text: str
    vector: torch.Tensor


@dataclass(frozen=True)
class RuntimeDecision:
    action: str
    label: str
    memory_similarity: float
    memory_margin: float | None
    local_majority_label: str
    local_purity: float
    local_k: int
    base_label: str
    base_similarity: float
    agreement: bool


def unit(v: torch.Tensor) -> torch.Tensor:
    return F.normalize(v, p=2, dim=0)


def encode_memory(router, rows: Sequence[LabeledSentence]) -> List[RuntimeMemoryVector]:
    return [
        RuntimeMemoryVector(
            label=row.label,
            text=row.text,
            vector=unit(router._encode_tensor(row.text)),
        )
        for row in rows
    ]


def spherical_kmeans(
    vectors: Sequence[torch.Tensor],
    k: int,
    iterations: int = 20,
) -> List[torch.Tensor]:
    if not vectors:
        return []

    k = max(1, min(k, len(vectors)))
    if k == 1:
        return [unit(torch.stack(list(vectors), dim=0).mean(dim=0))]
    if len(vectors) <= k:
        return [unit(v.clone()) for v in vectors]

    centers = [vectors[0].clone()]
    while len(centers) < k:
        best_i = 0
        best_d = -1.0
        for i, vector in enumerate(vectors):
            nearest = max(float(torch.dot(vector, c).item()) for c in centers)
            distance = 1.0 - nearest
            if distance > best_d:
                best_d = distance
                best_i = i
        centers.append(vectors[best_i].clone())

    centers = [unit(c) for c in centers]

    for _ in range(iterations):
        groups = [[] for _ in range(k)]
        for vector in vectors:
            sims = [float(torch.dot(vector, c).item()) for c in centers]
            groups[max(range(k), key=lambda i: sims[i])].append(vector)

        new_centers = []
        for i, group in enumerate(groups):
            if group:
                new_centers.append(unit(torch.stack(group, dim=0).mean(dim=0)))
            else:
                new_centers.append(centers[i])

        if all(
            float(torch.dot(a, b).item()) > 0.999999
            for a, b in zip(centers, new_centers)
        ):
            centers = new_centers
            break

        centers = new_centers

    return centers


def build_multi_prototypes(
    memory: Sequence[RuntimeMemoryVector],
    per_label: int = 2,
) -> Dict[str, List[torch.Tensor]]:
    grouped: Dict[str, List[torch.Tensor]] = defaultdict(list)
    for row in memory:
        grouped[row.label].append(row.vector)

    return {
        label: spherical_kmeans(vectors, per_label)
        for label, vectors in grouped.items()
    }


def rank_multi_prototypes(
    query: torch.Tensor,
    prototypes: Dict[str, List[torch.Tensor]],
):
    scored = []
    for label, centers in prototypes.items():
        similarity = max(float(torch.dot(query, c).item()) for c in centers)
        scored.append((similarity, label))
    scored.sort(reverse=True)
    return scored


def local_evidence(
    query: torch.Tensor,
    memory: Sequence[RuntimeMemoryVector],
    k: int = 3,
):
    if not memory:
        return "", 0.0, []

    neighbors = sorted(
        (
            (
                float(torch.dot(query, row.vector).item()),
                row.label,
                row.text,
            )
            for row in memory
        ),
        reverse=True,
    )[:max(1, min(k, len(memory)))]

    counts = Counter(label for _, label, _ in neighbors)
    majority_label, majority_count = counts.most_common(1)[0]
    purity = majority_count / len(neighbors)
    return majority_label, purity, neighbors


def decide_adaptive_route(
    router,
    text: str,
    memory: Sequence[RuntimeMemoryVector],
    prototypes: Dict[str, List[torch.Tensor]],
    *,
    base_similarity_threshold: float = 0.80,
    override_similarity_threshold: float = 0.92,
    local_k: int = 3,
    local_purity_threshold: float = 0.60,
) -> RuntimeDecision | None:
    if not prototypes:
        return None

    query = unit(router._encode_tensor(text))
    ranked_memory = rank_multi_prototypes(query, prototypes)
    mem_sim, mem_label = ranked_memory[0]
    mem_margin = (
        mem_sim - ranked_memory[1][0]
        if len(ranked_memory) > 1
        else None
    )

    base_top1 = router.route(text)[0]
    agreement = mem_label == base_top1.label

    majority_label, purity, _ = local_evidence(
        query,
        memory,
        k=local_k,
    )

    if mem_sim >= base_similarity_threshold and agreement:
        action = "ACCEPT"
    elif (
        not agreement
        and mem_sim >= override_similarity_threshold
        and majority_label == mem_label
        and purity >= local_purity_threshold
    ):
        action = "ADAPTIVE_OVERRIDE"
    elif mem_sim >= base_similarity_threshold:
        action = "GATE_REVIEW"
    else:
        action = "BASE_FALLBACK"

    return RuntimeDecision(
        action=action,
        label=mem_label,
        memory_similarity=mem_sim,
        memory_margin=mem_margin,
        local_majority_label=majority_label,
        local_purity=purity,
        local_k=min(local_k, len(memory)),
        base_label=base_top1.label,
        base_similarity=base_top1.similarity,
        agreement=agreement,
    )
