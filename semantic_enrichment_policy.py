# semantic_enrichment_policy.py
#
# LLM_SEM v0.4.9 Enrichment Policy
#
# Suggests nearby teaching examples for uncertain/reviewed inputs.
# It NEVER writes to Adaptive Memory automatically.
#
# Candidate-label policy:
#   1. Prefer labels supported by both Multi-Prototype and Local Evidence.
#   2. Include Base label when it disagrees, so the user can inspect both sides.
#   3. Rank unused enrichment-pool examples by cosine similarity to the query.
#
# This is a proposal mechanism, not an automatic relabeling mechanism.

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import torch

from adaptive_semantic_learning import normalize_text
from adaptive_semantic_runtime import RuntimeDecision, RuntimeMemoryVector, unit
from semantic_eval import LabeledSentence


@dataclass(frozen=True)
class EnrichmentSuggestion:
    label: str
    text: str
    similarity: float
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


def candidate_labels(decision: RuntimeDecision | None) -> List[tuple[str, str]]:
    if decision is None:
        return []

    ordered: List[tuple[str, str]] = []

    def add(label: str, reason: str):
        if label and all(existing != label for existing, _ in ordered):
            ordered.append((label, reason))

    if decision.local_majority_label == decision.label:
        add(
            decision.label,
            "prototype and local-neighbor evidence agree",
        )
    elif decision.local_majority_label:
        add(
            decision.local_majority_label,
            "local-neighbor majority",
        )

    add(decision.label, "multi-prototype candidate")
    add(decision.base_label, "base-router candidate")
    return ordered


def suggest_enrichment(
    router,
    query_text: str,
    decision: RuntimeDecision | None,
    memory: Sequence[RuntimeMemoryVector],
    pool: Sequence[LabeledSentence],
    *,
    max_per_label: int = 2,
    max_total: int = 4,
) -> List[EnrichmentSuggestion]:
    labels = candidate_labels(decision)
    if not labels or not pool:
        return []

    existing = {
        (row.label, normalize_text(row.text))
        for row in memory
    }

    query = unit(router._encode_tensor(query_text))
    suggestions: List[EnrichmentSuggestion] = []

    for label, reason in labels:
        scored = []
        for row in pool:
            if row.label != label:
                continue
            if (row.label, normalize_text(row.text)) in existing:
                continue

            vector = unit(router._encode_tensor(row.text))
            similarity = float(torch.dot(query, vector).item())
            scored.append((similarity, row.text))

        scored.sort(reverse=True)

        for similarity, text in scored[:max_per_label]:
            suggestions.append(
                EnrichmentSuggestion(
                    label=label,
                    text=text,
                    similarity=similarity,
                    reason=reason,
                )
            )

    # Keep strongest proposals first while preserving cross-label inspection.
    suggestions.sort(key=lambda x: x.similarity, reverse=True)
    return suggestions[:max_total]
