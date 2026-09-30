# semantic_runtime_v2.py
#
# Runtime adapter for LLM_SEM Semantic Data v2.0.
#
# This module bridges the current adaptive runtime (memory/base/prototype/
# local-majority/gate signals) to the structured SemanticDataV2 contract.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from model import LanguageModel
from tokenizer import Tokenizer
from semantic import (
    SemanticConcept,
    SemanticContext,
    SemanticDataV2,
    SemanticPurpose,
    SemanticRelation,
    SemanticVector,
    encode_semantic_v2,
)


@dataclass
class RuntimeSemanticSignals:
    """Signals exported by the v0.4.x adaptive semantic runtime."""

    memory_label: Optional[str] = None
    memory_similarity: Optional[float] = None
    memory_margin: Optional[float] = None

    local_majority: Optional[str] = None
    local_purity: Optional[float] = None
    local_k: Optional[int] = None

    base_label: Optional[str] = None
    base_similarity: Optional[float] = None

    prototype_labels: List[str] = field(default_factory=list)
    prototype_scores: Dict[str, float] = field(default_factory=dict)

    gate_state: Optional[str] = None
    selected_label: Optional[str] = None

    confidence: Optional[float] = None
    uncertainty: Optional[float] = None

    adaptive_enabled: Optional[bool] = None
    adaptive_samples: Optional[int] = None
    memory_labels: Optional[int] = None

    metadata: Dict[str, str] = field(default_factory=dict)


def _clamp01(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return max(0.0, min(1.0, float(value)))


def estimate_runtime_confidence(signals: RuntimeSemanticSignals) -> float:
    """Derive a conservative confidence score from available runtime signals.

    Priority:
      explicit confidence
      -> memory similarity weighted by local purity
      -> base similarity
      -> neutral fallback
    """
    if signals.confidence is not None:
        return _clamp01(signals.confidence) or 0.0

    if signals.memory_similarity is not None:
        purity = (
            _clamp01(signals.local_purity)
            if signals.local_purity is not None
            else 1.0
        )
        return _clamp01(float(signals.memory_similarity) * float(purity)) or 0.0

    if signals.base_similarity is not None:
        return _clamp01(signals.base_similarity) or 0.0

    return 0.5


def estimate_runtime_uncertainty(
    signals: RuntimeSemanticSignals,
    confidence: Optional[float] = None,
) -> float:
    """Estimate uncertainty while respecting review/gate ambiguity."""
    if signals.uncertainty is not None:
        return _clamp01(signals.uncertainty) or 0.0

    conf = estimate_runtime_confidence(signals) if confidence is None else confidence
    uncertainty = 1.0 - conf

    if signals.memory_margin is not None:
        # Small margin means ambiguity. Use a simple bounded penalty without
        # assuming any particular future calibration method.
        margin = max(0.0, float(signals.memory_margin))
        ambiguity = max(0.0, 1.0 - min(1.0, margin / 0.05))
        uncertainty = max(uncertainty, ambiguity)

    if signals.gate_state and "REVIEW" in signals.gate_state.upper():
        uncertainty = max(uncertainty, 0.75)

    return _clamp01(uncertainty) or 0.0


def runtime_relations(signals: RuntimeSemanticSignals) -> List[SemanticRelation]:
    """Represent runtime decisions as explicit semantic relations."""
    relations: List[SemanticRelation] = []

    if signals.memory_label:
        relations.append(
            SemanticRelation(
                subject="query",
                predicate="memory_candidate",
                object=signals.memory_label,
                confidence=_clamp01(signals.memory_similarity),
            )
        )

    if signals.local_majority:
        relations.append(
            SemanticRelation(
                subject="query",
                predicate="local_majority",
                object=signals.local_majority,
                confidence=_clamp01(signals.local_purity),
                attributes={
                    "k": str(signals.local_k) if signals.local_k is not None else "",
                },
            )
        )

    if signals.base_label:
        relations.append(
            SemanticRelation(
                subject="query",
                predicate="base_candidate",
                object=signals.base_label,
                confidence=_clamp01(signals.base_similarity),
            )
        )

    if signals.selected_label:
        relations.append(
            SemanticRelation(
                subject="query",
                predicate="selected_label",
                object=signals.selected_label,
                confidence=estimate_runtime_confidence(signals),
            )
        )

    for label in signals.prototype_labels:
        relations.append(
            SemanticRelation(
                subject=label,
                predicate="represented_by",
                object="semantic_prototype",
                confidence=_clamp01(signals.prototype_scores.get(label)),
            )
        )

    return relations


def runtime_context(signals: RuntimeSemanticSignals) -> SemanticContext:
    attrs: Dict[str, str] = {}

    if signals.gate_state is not None:
        attrs["gate_state"] = signals.gate_state
    if signals.local_k is not None:
        attrs["local_k"] = str(signals.local_k)
    if signals.adaptive_enabled is not None:
        attrs["adaptive_enabled"] = str(signals.adaptive_enabled)
    if signals.adaptive_samples is not None:
        attrs["adaptive_samples"] = str(signals.adaptive_samples)
    if signals.memory_labels is not None:
        attrs["memory_labels"] = str(signals.memory_labels)

    attrs.update(signals.metadata)

    return SemanticContext(
        text="LLM_SEM adaptive semantic runtime",
        vector=None,
        attributes=attrs,
    )


def build_runtime_semantic_v2(
    model: LanguageModel,
    tokenizer: Tokenizer,
    text: str,
    signals: RuntimeSemanticSignals,
    *,
    concept_texts: Optional[Sequence[str]] = None,
    purpose_text: Optional[str] = None,
    intent: Optional[str] = None,
) -> SemanticDataV2:
    """Build Semantic Data v2.0 from one runtime decision.

    Existing runtime values become structured fields:
      memory/base/local/prototype -> relations + context
      selected label              -> concept and relation
      gate confidence             -> confidence / uncertainty
      query meaning               -> global vector
      task intent                 -> purpose vector
    """

    confidence = estimate_runtime_confidence(signals)
    uncertainty = estimate_runtime_uncertainty(signals, confidence)

    concepts = list(concept_texts or [])
    if signals.selected_label and signals.selected_label not in concepts:
        concepts.append(signals.selected_label)
    if signals.memory_label and signals.memory_label not in concepts:
        concepts.append(signals.memory_label)
    if signals.base_label and signals.base_label not in concepts:
        concepts.append(signals.base_label)

    semantic = encode_semantic_v2(
        model,
        tokenizer,
        text,
        concepts=concepts or None,
        purpose_text=purpose_text,
        intent=intent,
        relations=runtime_relations(signals),
        context_text=None,
        context_attributes=runtime_context(signals).attributes,
        confidence=confidence,
        uncertainty=uncertainty,
        metadata={
            "runtime_adapter": "v0.5.1",
            "source_runtime": "LLM_SEM v0.4.x adaptive",
            **signals.metadata,
        },
    )

    return semantic


def from_runtime_dict(
    model: LanguageModel,
    tokenizer: Tokenizer,
    text: str,
    runtime: Dict[str, object],
    *,
    concept_texts: Optional[Sequence[str]] = None,
    purpose_text: Optional[str] = None,
    intent: Optional[str] = None,
) -> SemanticDataV2:
    """Convenience adapter for chat.py-style dictionaries.

    Unknown keys are ignored so the adapter can tolerate incremental runtime
    evolution without forcing immediate changes to the Semantic Data schema.
    """

    allowed = RuntimeSemanticSignals.__dataclass_fields__.keys()
    clean = {k: v for k, v in runtime.items() if k in allowed}
    signals = RuntimeSemanticSignals(**clean)

    return build_runtime_semantic_v2(
        model,
        tokenizer,
        text,
        signals,
        concept_texts=concept_texts,
        purpose_text=purpose_text,
        intent=intent,
    )


def runtime_summary(semantic: SemanticDataV2) -> Dict[str, object]:
    """Compact summary useful for chat/debug output."""
    return {
        "schema_version": semantic.schema_version,
        "confidence": semantic.confidence,
        "uncertainty": semantic.uncertainty,
        "concepts": [c.name for c in semantic.concept_vectors],
        "purpose": semantic.purpose.text,
        "intent": semantic.purpose.intent,
        "relations": [
            {
                "subject": r.subject,
                "predicate": r.predicate,
                "object": r.object,
                "confidence": r.confidence,
            }
            for r in semantic.relations
        ],
        "context": (
            semantic.context.attributes if semantic.context is not None else {}
        ),
    }
