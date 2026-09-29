# semantic_contrastive_enrichment_policy.py
#
# LLM_SEM v0.4.10 Contrastive Enrichment Policy
#
# Goal:
#   Rank teaching examples by expected semantic-boundary utility rather than
#   raw query similarity alone.
#
# Candidate score:
#   boundary_score =
#       0.70 * query_similarity
#     + 0.20 * novelty_vs_same_label_memory
#     + 0.10 * label_deficit
#
# where:
#   novelty = 1 - max cosine similarity to existing same-label memory
#   label_deficit = relative under-representation of the candidate label
#
# Safety / data-quality rules:
#   - Never auto-write to Adaptive Memory.
#   - Exclude exact query text.
#   - Exclude already registered memory text.
#   - Exclude near-duplicates of existing same-label memory (>= 0.995 cosine).
#   - Preserve competing labels so the user can inspect the boundary.

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import torch

from adaptive_semantic_learning import normalize_text
from adaptive_semantic_runtime import RuntimeDecision, RuntimeMemoryVector, unit
from semantic_eval import LabeledSentence


@dataclass(frozen=True)
class ContrastiveSuggestion:
    label: str
    text: str
    query_similarity: float
    novelty: float
    label_deficit: float
    boundary_score: float
    reason: str


def load_enrichment_pool(path: Path) -> List[LabeledSentence]:
    if not path.exists():
        return []

    rows: List[LabeledSentence] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = normalize_text(str(row.get("label", "")))
            text = normalize_text(str(row.get("text", "")))
            if label and text:
                rows.append(LabeledSentence(label=label, text=text))
    return rows


def competing_labels(decision: RuntimeDecision | None):
    if decision is None:
        return []

    ordered = []

    def add(label: str, reason: str):
        if label and all(existing != label for existing, _ in ordered):
            ordered.append((label, reason))

    add(decision.label, "multi-prototype side")
    add(decision.local_majority_label, "local-neighborhood side")
    add(decision.base_label, "base-router side")
    return ordered


def _label_deficits(memory: Sequence[RuntimeMemoryVector], labels):
    counts = Counter(row.label for row in memory)
    if not labels:
        return {}

    max_count = max([counts.get(label, 0) for label, _ in labels] + [1])
    return {
        label: (max_count - counts.get(label, 0)) / max_count
        for label, _ in labels
    }


def _same_label_max_similarity(
    candidate_vector: torch.Tensor,
    memory: Sequence[RuntimeMemoryVector],
    label: str,
) -> float:
    sims = [
        float(torch.dot(candidate_vector, row.vector).item())
        for row in memory
        if row.label == label
    ]
    return max(sims) if sims else 0.0


def suggest_contrastive_enrichment(
    router,
    query_text: str,
    decision: RuntimeDecision | None,
    memory: Sequence[RuntimeMemoryVector],
    pool: Sequence[LabeledSentence],
    *,
    max_per_label: int = 2,
    max_total: int = 4,
    near_duplicate_threshold: float = 0.995,
) -> List[ContrastiveSuggestion]:
    labels = competing_labels(decision)
    if not labels or not pool:
        return []

    query_norm = normalize_text(query_text)
    query_vector = unit(router._encode_tensor(query_text))

    existing_texts = {
        (row.label, normalize_text(row.text))
        for row in memory
    }

    deficits = _label_deficits(memory, labels)
    suggestions = []

    for label, reason in labels:
        scored = []

        for row in pool:
            if row.label != label:
                continue

            candidate_text = normalize_text(row.text)

            # v0.4.9 issue: do not suggest the query itself.
            if candidate_text == query_norm:
                continue

            if (row.label, candidate_text) in existing_texts:
                continue

            candidate_vector = unit(router._encode_tensor(row.text))
            query_similarity = float(
                torch.dot(query_vector, candidate_vector).item()
            )

            same_label_similarity = _same_label_max_similarity(
                candidate_vector,
                memory,
                label,
            )

            # Avoid adding an almost identical example to an existing memory item.
            if same_label_similarity >= near_duplicate_threshold:
                continue

            novelty = max(0.0, 1.0 - same_label_similarity)
            deficit = deficits.get(label, 0.0)

            boundary_score = (
                0.70 * query_similarity
                + 0.20 * novelty
                + 0.10 * deficit
            )

            scored.append(
                ContrastiveSuggestion(
                    label=label,
                    text=row.text,
                    query_similarity=query_similarity,
                    novelty=novelty,
                    label_deficit=deficit,
                    boundary_score=boundary_score,
                    reason=reason,
                )
            )

        scored.sort(
            key=lambda item: (
                item.boundary_score,
                item.query_similarity,
                item.novelty,
            ),
            reverse=True,
        )
        suggestions.extend(scored[:max_per_label])

    # Keep high-utility proposals while retaining both sides where possible.
    suggestions.sort(
        key=lambda item: (
            item.boundary_score,
            item.query_similarity,
        ),
        reverse=True,
    )
    return suggestions[:max_total]
