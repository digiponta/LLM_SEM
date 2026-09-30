# semantic_relations_v035.py
#
# Rule-based semantic relation generation for LLM_SEM v0.3.5.
#
# Relations are derived from the already extracted Concept / Intent / Purpose
# structure. Runtime routing evidence is generated separately by
# semantic_runtime_v2.py and merged into SemanticDataV2.relations[].

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from semantic import SemanticRelation
from semantic_intent_v034 import PurposeIntentResult


def generate_semantic_relations(
    extracted: PurposeIntentResult,
) -> List[SemanticRelation]:
    """Generate explicit semantic relations from Concept / Intent / Purpose."""
    relations: List[SemanticRelation] = []

    predicate_map = {
        "definition": "requests_definition_of",
        "explain": "requests_explanation_of",
        "how_to": "requests_how_to_for",
        "why": "requests_reason_for",
        "query": "requests_information_about",
        "unspecified": "mentions",
    }
    predicate = predicate_map.get(extracted.intent, "mentions")

    for concept in extracted.concept_texts:
        relations.append(
            SemanticRelation(
                subject="query",
                predicate=predicate,
                object=concept,
                confidence=extracted.confidence,
                attributes={
                    "intent": extracted.intent,
                    "rule": extracted.rule,
                },
            )
        )

        relations.append(
            SemanticRelation(
                subject=extracted.purpose_text,
                predicate="targets_concept",
                object=concept,
                confidence=extracted.confidence,
                attributes={
                    "intent": extracted.intent,
                },
            )
        )

    relations.append(
        SemanticRelation(
            subject="query",
            predicate="has_purpose",
            object=extracted.purpose_text,
            confidence=extracted.confidence,
            attributes={
                "intent": extracted.intent,
                "rule": extracted.rule,
            },
        )
    )

    return relations
