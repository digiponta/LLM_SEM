# semantic_proposition_v036.py
#
# Lightweight Japanese proposition extraction for LLM_SEM v0.3.6.
#
# This module intentionally uses transparent particle-based rules. It extracts
# a small proposition structure when confidence is sufficient and otherwise
# returns no proposition rather than inventing one.

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import List, Optional


@dataclass
class Proposition:
    subject: str
    predicate: str
    object: str
    particle: str
    confidence: float
    rule: str


_PROPERTY_WORDS = {
    "高速",
    "低速",
    "速い",
    "遅い",
    "高い",
    "低い",
    "大きい",
    "小さい",
    "安全",
    "危険",
    "可能",
    "必要",
    "重要",
    "有効",
    "無効",
}


def _clean(text: str) -> str:
    return " ".join(text.strip().split())


def _predicate_for_topic(rest: str) -> str:
    if rest in _PROPERTY_WORDS:
        return "has_property"
    if any(rest.endswith(word) for word in _PROPERTY_WORDS):
        return "has_property"
    return "has_predicate"


def extract_propositions(text: str) -> List[Proposition]:
    """Extract conservative Japanese proposition structures.

    Supported baseline particles:
      は / が -> subject + predicate phrase
      を      -> omitted_subject --acts_on--> object
      に      -> omitted_subject --targets--> object
      で      -> omitted_subject --context_of_action--> object
      と      -> omitted_subject --related_with--> object

    The は/が form is the primary v0.3.6 target.
    """
    value = _clean(text)
    if not value:
        return []

    # Subject/topic + predicate.
    m = re.match(r"^(?P<subject>.+?)(?P<particle>は|が)(?P<rest>.+)$", value)
    if m:
        subject = _clean(m.group("subject"))
        rest = _clean(m.group("rest"))
        if subject and rest:
            return [
                Proposition(
                    subject=subject,
                    predicate=_predicate_for_topic(rest),
                    object=rest,
                    particle=m.group("particle"),
                    confidence=0.92,
                    rule="subject-topic-predicate",
                )
            ]

    # Object/target/context relations when the grammatical subject is omitted.
    particle_rules = [
        ("を", "acts_on", 0.78),
        ("に", "targets", 0.72),
        ("で", "context_of_action", 0.70),
        ("と", "related_with", 0.70),
    ]
    for particle, predicate, confidence in particle_rules:
        if particle in value:
            left, right = value.split(particle, 1)
            left = _clean(left)
            right = _clean(right)
            if left and right:
                return [
                    Proposition(
                        subject="implicit_subject",
                        predicate=predicate,
                        object=left,
                        particle=particle,
                        confidence=confidence,
                        rule=f"particle:{particle}",
                    )
                ]

    return []


def refine_purpose(
    intent: str,
    purpose_text: str,
    propositions: List[Proposition],
) -> str:
    """Refine purpose text when a proposition gives a clearer structure."""
    if intent == "why" and propositions:
        p = propositions[0]
        if p.subject != "implicit_subject":
            return f"explain_reason({p.subject}, {p.object})"
    return purpose_text
