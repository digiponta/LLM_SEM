# semantic_structural_counterfactual.py
#
# LLM_SEM v0.4.12 Structural Counterfactual Score
#
# Distinguishes:
#   - DECISION_ONLY_ACCEPT:
#       the runtime action becomes ACCEPT/ADAPTIVE_OVERRIDE, but the local
#       semantic structure remains inconsistent.
#   - STRUCTURAL_IMPROVEMENT:
#       prototype label, local-majority label, and Base label become aligned,
#       with local purity maintained or improved.
#
# Also reports PARTIAL_STRUCTURE_IMPROVEMENT and NO_IMPROVEMENT.
# Persistent Adaptive Memory is never modified.

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from adaptive_semantic_runtime import (
    RuntimeDecision,
    RuntimeMemoryVector,
    build_multi_prototypes,
    decide_adaptive_route,
    unit,
)
from semantic_eval import LabeledSentence


@dataclass(frozen=True)
class StructuralCounterfactualResult:
    label: str
    text: str
    category: str
    before_action: str
    after_action: str
    base_label: str
    before_memory_label: str
    after_memory_label: str
    before_local_label: str
    after_local_label: str
    before_purity: float
    after_purity: float
    before_similarity: float
    after_similarity: float
    decision_gain: float
    prototype_base_gain: float
    local_base_gain: float
    prototype_local_gain: float
    purity_gain: float
    similarity_gain: float
    structural_score: float


def _resolved(action: str) -> float:
    return 1.0 if action in ("ACCEPT", "ADAPTIVE_OVERRIDE") else 0.0


def _eq(a: str, b: str) -> float:
    return 1.0 if a == b else 0.0


def simulate_structural_candidate(
    router,
    query_text: str,
    before: RuntimeDecision,
    memory: Sequence[RuntimeMemoryVector],
    candidate: LabeledSentence,
    *,
    prototypes_per_label: int = 2,
    base_similarity_threshold: float = 0.80,
    override_similarity_threshold: float = 0.92,
    local_k: int = 3,
    local_purity_threshold: float = 1.00,
) -> StructuralCounterfactualResult:
    candidate_vector = RuntimeMemoryVector(
        label=candidate.label,
        text=candidate.text,
        vector=unit(router._encode_tensor(candidate.text)),
    )

    simulated_memory = list(memory) + [candidate_vector]
    simulated_prototypes = build_multi_prototypes(
        simulated_memory,
        per_label=prototypes_per_label,
    )

    after = decide_adaptive_route(
        router,
        query_text,
        simulated_memory,
        simulated_prototypes,
        base_similarity_threshold=base_similarity_threshold,
        override_similarity_threshold=override_similarity_threshold,
        local_k=local_k,
        local_purity_threshold=local_purity_threshold,
    )
    if after is None:
        raise RuntimeError("Counterfactual simulation produced no decision.")

    decision_gain = _resolved(after.action) - _resolved(before.action)

    prototype_base_gain = (
        _eq(after.label, after.base_label)
        - _eq(before.label, before.base_label)
    )
    local_base_gain = (
        _eq(after.local_majority_label, after.base_label)
        - _eq(before.local_majority_label, before.base_label)
    )
    prototype_local_gain = (
        _eq(after.label, after.local_majority_label)
        - _eq(before.label, before.local_majority_label)
    )

    purity_gain = after.local_purity - before.local_purity
    similarity_gain = after.memory_similarity - before.memory_similarity

    after_full_alignment = (
        after.label == after.local_majority_label == after.base_label
    )
    before_full_alignment = (
        before.label == before.local_majority_label == before.base_label
    )

    purity_not_worse = after.local_purity >= before.local_purity - 1e-12
    purity_improved = after.local_purity > before.local_purity + 1e-12

    structural_changed = (
        prototype_base_gain > 0
        or local_base_gain > 0
        or prototype_local_gain > 0
        or purity_improved
    )

    if after_full_alignment and not before_full_alignment and purity_not_worse:
        category = "STRUCTURAL_IMPROVEMENT"
    elif decision_gain > 0 and not structural_changed:
        category = "DECISION_ONLY_ACCEPT"
    elif decision_gain > 0 and not after_full_alignment:
        category = "DECISION_ONLY_ACCEPT"
    elif structural_changed:
        category = "PARTIAL_STRUCTURE_IMPROVEMENT"
    else:
        category = "NO_IMPROVEMENT"

    # Structural consistency dominates the score.
    structural_score = (
        0.10 * decision_gain
        + 0.25 * prototype_base_gain
        + 0.25 * local_base_gain
        + 0.20 * prototype_local_gain
        + 0.15 * purity_gain
        + 0.05 * similarity_gain
    )

    # Make full structural repair clearly outrank decision-only transitions.
    if category == "STRUCTURAL_IMPROVEMENT":
        structural_score += 1.0
    elif category == "DECISION_ONLY_ACCEPT":
        structural_score -= 0.25

    return StructuralCounterfactualResult(
        label=candidate.label,
        text=candidate.text,
        category=category,
        before_action=before.action,
        after_action=after.action,
        base_label=after.base_label,
        before_memory_label=before.label,
        after_memory_label=after.label,
        before_local_label=before.local_majority_label,
        after_local_label=after.local_majority_label,
        before_purity=before.local_purity,
        after_purity=after.local_purity,
        before_similarity=before.memory_similarity,
        after_similarity=after.memory_similarity,
        decision_gain=decision_gain,
        prototype_base_gain=prototype_base_gain,
        local_base_gain=local_base_gain,
        prototype_local_gain=prototype_local_gain,
        purity_gain=purity_gain,
        similarity_gain=similarity_gain,
        structural_score=structural_score,
    )
