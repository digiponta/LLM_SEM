# semantic_relations_v035.py
#
# Rule-based semantic relation generation for LLM_SEM v0.3.5.
#
# Relations are derived from the already extracted Concept / Intent / Purpose
# structure. Runtime routing evidence is generated separately by
# semantic_runtime_v2.py and merged into SemanticDataV2.relations[].

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from semantic import SemanticRelation
from semantic_intent_v034 import PurposeIntentResult
from semantic_proposition_v036 import Proposition


def generate_semantic_relations(
    extracted: PurposeIntentResult,
    propositions: Optional[List[Proposition]] = None,
    *,
    purpose_text: Optional[str] = None,
    concept_texts: Optional[List[str]] = None,
) -> List[SemanticRelation]:
    """Generate semantic relations from Concept / Intent / Purpose / Proposition."""
    relations: List[SemanticRelation] = []
    effective_purpose = purpose_text or extracted.purpose_text
    effective_concepts = concept_texts or extracted.concept_texts

    predicate_map = {
        "definition": "requests_definition_of",
        "explain": "requests_explanation_of",
        "how_to": "requests_how_to_for",
        "why": "requests_reason_for",
        "query": "requests_information_about",
        "unspecified": "mentions",
    }
    predicate = predicate_map.get(extracted.intent, "mentions")

    proposition_list = propositions or []

    if proposition_list:
        # Bind query intent to proposition nodes rather than independently to
        # every concept. This preserves the fact that, e.g., a "why" query is
        # about the whole statement "GPU is fast", not GPU and "fast" as two
        # unrelated request targets.
        for index, proposition in enumerate(proposition_list, 1):
            proposition_id = f"proposition_{index}"

            query_predicate = (
                "asserts"
                if extracted.intent == "unspecified"
                else predicate
            )
            relations.append(
                SemanticRelation(
                    subject="query",
                    predicate=query_predicate,
                    object=proposition_id,
                    confidence=extracted.confidence,
                    attributes={
                        "intent": extracted.intent,
                        "rule": extracted.rule,
                        "relation_family": "query_proposition_binding",
                    },
                )
            )

            relations.append(
                SemanticRelation(
                    subject=effective_purpose,
                    predicate="targets_proposition",
                    object=proposition_id,
                    confidence=extracted.confidence,
                    attributes={
                        "intent": extracted.intent,
                    },
                )
            )

            relations.append(
                SemanticRelation(
                    subject=proposition_id,
                    predicate="subject",
                    object=proposition.subject,
                    confidence=proposition.confidence,
                    attributes={"relation_family": "proposition_binding"},
                )
            )
            relations.append(
                SemanticRelation(
                    subject=proposition_id,
                    predicate="predicate",
                    object=proposition.predicate,
                    confidence=proposition.confidence,
                    attributes={"relation_family": "proposition_binding"},
                )
            )
            relations.append(
                SemanticRelation(
                    subject=proposition_id,
                    predicate="object",
                    object=proposition.object,
                    confidence=proposition.confidence,
                    attributes={"relation_family": "proposition_binding"},
                )
            )

            relations.append(
                SemanticRelation(
                    subject=proposition.subject,
                    predicate=proposition.predicate,
                    object=proposition.object,
                    confidence=proposition.confidence,
                    attributes={
                        "particle": proposition.particle,
                        "rule": proposition.rule,
                        "relation_family": "proposition",
                        "proposition_id": proposition_id,
                    },
                )
            )
    else:
        for concept in effective_concepts:
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
                    subject=effective_purpose,
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
            object=effective_purpose,
            confidence=extracted.confidence,
            attributes={
                "intent": extracted.intent,
                "rule": extracted.rule,
            },
        )
    )

    return relations
