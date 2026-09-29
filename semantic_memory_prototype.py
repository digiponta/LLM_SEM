# semantic_memory_prototype.py
#
# LLM_SEM v0.3.4 Semantic Memory Prototype utilities.
#
# Multiple adaptive examples are grouped by label.  Each label prototype is
# the mean semantic vector of its taught examples.  Queries are ranked against
# prototypes and the top1-top2 cosine gap becomes the memory margin.

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import torch
import torch.nn.functional as F

from semantic_eval import LabeledSentence


@dataclass(frozen=True)
class MemoryPrototype:
    label: str
    vector: torch.Tensor
    count: int


@dataclass(frozen=True)
class PrototypeScore:
    label: str
    similarity: float
    count: int


def build_prototypes(
    router,
    rows: Sequence[LabeledSentence],
) -> Dict[str, MemoryPrototype]:
    grouped: dict[str, list[torch.Tensor]] = defaultdict(list)

    for row in rows:
        grouped[row.label].append(router._encode_tensor(row.text))

    prototypes: Dict[str, MemoryPrototype] = {}
    for label, vectors in grouped.items():
        mean_vector = torch.stack(vectors, dim=0).mean(dim=0)
        prototypes[label] = MemoryPrototype(
            label=label,
            vector=mean_vector,
            count=len(vectors),
        )
    return prototypes


def rank_prototypes(
    router,
    text: str,
    prototypes: Dict[str, MemoryPrototype],
) -> List[PrototypeScore]:
    if not prototypes:
        return []

    query = router._encode_tensor(text)
    scores: List[PrototypeScore] = []

    for label, prototype in prototypes.items():
        similarity = float(
            F.cosine_similarity(query, prototype.vector, dim=0).item()
        )
        scores.append(
            PrototypeScore(
                label=label,
                similarity=similarity,
                count=prototype.count,
            )
        )

    scores.sort(key=lambda x: x.similarity, reverse=True)
    return scores


def prototype_margin(scores: Sequence[PrototypeScore]) -> float | None:
    if len(scores) < 2:
        return None
    return scores[0].similarity - scores[1].similarity
