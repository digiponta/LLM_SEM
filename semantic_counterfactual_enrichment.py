# semantic_counterfactual_enrichment.py
#
# LLM_SEM v0.4.11 Counterfactual Enrichment Evaluation
#
# Simulate adding one candidate teaching example WITHOUT modifying persistent
# Adaptive Memory, rebuild Multi-Prototype / Local Evidence, and measure how
# the current query's semantic boundary changes.
#
# Counterfactual utility rewards:
#   - GATE_REVIEW/BASE_FALLBACK -> ACCEPT/ADAPTIVE_OVERRIDE
#   - memory label becoming aligned with the candidate teaching label
#   - local majority alignment
#   - local purity increase
#   - memory similarity increase
#
# No persistent writes are performed.

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
class CounterfactualResult:
    label: str
    text: str
    before_action: str
    after_action: str
    before_memory_label: str
    after_memory_label: str
    before_similarity: float
    after_similarity: float
    before_local_label: str
    after_local_label: str
    before_purity: float
    after_purity: float
    action_gain: float
    label_gain: float
    purity_gain: float
    similarity_gain: float
    counterfactual_score: float


def _action_value(action: str) -> float:
    return {
        "BASE_FALLBACK": 0.0,
        "GATE_REVIEW": 0.0,
        "ACCEPT": 1.0,
        "ADAPTIVE_OVERRIDE": 1.0,
    }.get(action, 0.0)


def simulate_candidate(
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
) -> CounterfactualResult:
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

    action_gain = _action_value(after.action) - _action_value(before.action)

    before_label_ok = 1.0 if before.label == candidate.label else 0.0
    after_label_ok = 1.0 if after.label == candidate.label else 0.0
    label_gain = after_label_ok - before_label_ok

    purity_gain = after.local_purity - before.local_purity
    similarity_gain = after.memory_similarity - before.memory_similarity

    # Primary objective: turn an unresolved boundary into a resolvable one.
    # Secondary objectives: align the memory/local region with the teaching
    # label and strengthen local purity. Similarity gain is a light tie-breaker.
    counterfactual_score = (
        0.50 * action_gain
        + 0.25 * label_gain
        + 0.20 * purity_gain
        + 0.05 * similarity_gain
    )

    return CounterfactualResult(
        label=candidate.label,
        text=candidate.text,
        before_action=before.action,
        after_action=after.action,
        before_memory_label=before.label,
        after_memory_label=after.label,
        before_similarity=before.memory_similarity,
        after_similarity=after.memory_similarity,
        before_local_label=before.local_majority_label,
        after_local_label=after.local_majority_label,
        before_purity=before.local_purity,
        after_purity=after.local_purity,
        action_gain=action_gain,
        label_gain=label_gain,
        purity_gain=purity_gain,
        similarity_gain=similarity_gain,
        counterfactual_score=counterfactual_score,
    )
