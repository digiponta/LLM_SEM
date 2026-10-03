# semantic_answer_memory_v098.py
#
# LLM_SEM v0.9.8
# Semantic Answer Memory / Answer Resolver
#
# Resolve stable answers from semantic state:
#   query + selected label + intent + concepts
#
# This is deliberately separate from Semantic Memory lifecycle. Semantic Memory
# stores learned semantic knowledge/truth metadata; Answer Memory stores
# user-visible response candidates.

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable


DEFAULT_ANSWER_MEMORY = "data/semantic_guided_qa_v097.json"


def normalize_text(text: str) -> str:
    return "".join(str(text).strip().split()).lower()


@dataclass
class AnswerCandidate:
    query: str
    label: str
    intent: str
    concepts: list[str]
    truth_status: str
    answer: str


@dataclass
class AnswerResolution:
    matched: bool
    answer: str | None
    score: float
    reason: str
    candidate: AnswerCandidate | None


class SemanticAnswerMemory:
    def __init__(self, rows: Iterable[AnswerCandidate]):
        self.rows = list(rows)

    @classmethod
    def load(cls, path: str | Path = DEFAULT_ANSWER_MEMORY):
        p = Path(path)
        obj = json.loads(p.read_text(encoding="utf-8"))
        rows: list[AnswerCandidate] = []
        for item in obj.get("samples", []):
            query = str(item.get("query", "")).strip()
            label = str(item.get("label", "")).strip()
            answer = str(item.get("answer", "")).strip()
            if not query or not label or not answer:
                continue
            rows.append(
                AnswerCandidate(
                    query=query,
                    label=label,
                    intent=str(item.get("intent", "")).strip() or "general",
                    concepts=[
                        str(x).strip()
                        for x in item.get("concepts", [])
                        if str(x).strip()
                    ],
                    truth_status=str(
                        item.get("truth_status", "UNVERIFIED")
                    ).strip().upper(),
                    answer=answer,
                )
            )
        return cls(rows)

    def resolve(
        self,
        query: str,
        *,
        label: str,
        intent: str | None,
        concepts: list[str] | None,
        min_score: float = 7.0,
    ) -> AnswerResolution:
        nq = normalize_text(query)
        nlabel = normalize_text(label)
        nintent = normalize_text(intent or "general")
        nconcepts = {normalize_text(x) for x in (concepts or []) if str(x).strip()}

        best: tuple[float, AnswerCandidate, list[str]] | None = None

        for row in self.rows:
            score = 0.0
            reasons: list[str] = []

            if normalize_text(row.query) == nq:
                score += 10.0
                reasons.append("query-exact")

            if normalize_text(row.label) == nlabel:
                score += 4.0
                reasons.append("label")

            row_intent = normalize_text(row.intent)
            if row_intent == nintent:
                score += 2.0
                reasons.append("intent")

            row_concepts = {
                normalize_text(x) for x in row.concepts if str(x).strip()
            }
            overlap = nconcepts & row_concepts
            if overlap:
                score += 5.0 + max(0, len(overlap) - 1)
                reasons.append("concept")

            # Definition paraphrase support: if the concept itself appears in
            # the query, allow semantic state to resolve the canonical answer.
            if (
                row_concepts
                and normalize_text(row.label) == nlabel
                and row_intent == nintent
                and any(c and c in nq for c in row_concepts)
            ):
                score += 2.0
                reasons.append("concept-in-query")

            if best is None or score > best[0]:
                best = (score, row, reasons)

        if best is None:
            return AnswerResolution(False, None, 0.0, "no-candidates", None)

        score, row, reasons = best
        if score < float(min_score):
            return AnswerResolution(
                False,
                None,
                score,
                "below-threshold:" + ",".join(reasons),
                row,
            )

        return AnswerResolution(
            True,
            row.answer,
            score,
            ",".join(reasons) or "semantic-match",
            row,
        )


def truth_allows_answer_memory(
    truth_record: dict | None,
    candidate: AnswerCandidate | None,
) -> bool:
    """Do not silently turn an explicit FALSE/CONTESTED record into a TRUE answer."""
    if truth_record is None:
        return True

    state = str(truth_record.get("truth_status", "UNVERIFIED")).upper()
    if state in {"FALSE", "CONTESTED", "OUTDATED"}:
        return False

    # UNVERIFIED is still allowed as a response candidate, but the runtime
    # retains and displays its explicit warning.
    return True
