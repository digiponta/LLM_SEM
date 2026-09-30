# semantic_intent_v034.py
#
# Lightweight rule-based Concept / Intent / Purpose extraction for LLM_SEM v0.3.4.
# The goal is to separate semantic topic from user intent without introducing a
# new learned model. Ambiguous inputs fall back to the original text.

from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass
class PurposeIntentResult:
    concept_texts: list[str]
    intent: str
    purpose_text: str
    rule: str
    confidence: float


def _clean(text: str) -> str:
    return " ".join(text.strip().split())


def _clean_concept(text: str) -> str:
    concept = _clean(text)
    if concept.endswith("の") and len(concept) > 1:
        concept = concept[:-1]
    return concept


def extract_purpose_intent(text: str) -> PurposeIntentResult:
    original = _clean(text)
    if not original:
        raise ValueError("text must not be empty")

    rules = [
        (
            "definition",
            re.compile(r"^(?P<concept>.+?)(?:とは|って何|とは何|とは何ですか)[？?。]*$"),
            lambda c: f"explain_definition({c})",
            "definition-pattern",
            0.95,
        ),
        (
            "explain",
            re.compile(r"^(?P<concept>.+?)(?:について)?(?:教えて|説明して|説明してください)[。]*$"),
            lambda c: f"explain({c})",
            "explain-pattern",
            0.90,
        ),
        (
            "how_to",
            re.compile(r"^(?P<concept>.+?)(?:方法|やり方|使い方|仕方)(?:を)?(?:教えて|説明して|知りたい)[。]*$"),
            lambda c: f"how_to({c})",
            "how-to-pattern",
            0.90,
        ),
        (
            "why",
            re.compile(r"^(?P<concept>.+?)(?:なぜ|どうして)(?:ですか)?[？?。]*$"),
            lambda c: f"explain_reason({c})",
            "why-pattern",
            0.85,
        ),
    ]

    for intent, pattern, purpose_fn, rule, confidence in rules:
        m = pattern.match(original)
        if m:
            concept = _clean_concept(m.group("concept"))
            if concept:
                return PurposeIntentResult(
                    concept_texts=[concept],
                    intent=intent,
                    purpose_text=purpose_fn(concept),
                    rule=rule,
                    confidence=confidence,
                )

    # Common Japanese interrogative suffixes.
    suffix_rules = [
        ("definition", "とは", "explain_definition", 0.92),
        ("explain", "について教えて", "explain", 0.90),
        ("explain", "を教えて", "explain", 0.88),
        ("query", "を知りたい", "obtain_information", 0.85),
    ]
    for intent, suffix, purpose_name, confidence in suffix_rules:
        if original.endswith(suffix):
            concept = _clean_concept(original[: -len(suffix)])
            if concept:
                return PurposeIntentResult(
                    concept_texts=[concept],
                    intent=intent,
                    purpose_text=f"{purpose_name}({concept})",
                    rule=f"suffix:{suffix}",
                    confidence=confidence,
                )

    # Conservative fallback: keep the original text as the concept and purpose.
    return PurposeIntentResult(
        concept_texts=[original],
        intent="unspecified",
        purpose_text=original,
        rule="fallback",
        confidence=0.50,
    )
