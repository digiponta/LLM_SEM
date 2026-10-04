# semantic_answer_memory_v098.py
#
# LLM_SEM v0.10.0
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
DEFAULT_LEARNED_ANSWER_MEMORY = "data/semantic_answer_memory_learned.jsonl"
DEFAULT_UNIFIED_ANSWER_MEMORY = "data/unified_semantic_answer_memory_v0100.jsonl"


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
    def _rows_from_path(cls, path: str | Path) -> list[AnswerCandidate]:
        p = Path(path)
        if not p.exists():
            return []

        items: list[dict] = []
        if p.suffix.lower() == ".jsonl":
            for raw in p.read_text(encoding="utf-8").splitlines():
                if not raw.strip():
                    continue
                try:
                    row = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict):
                    items.append(row)
        else:
            obj = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(obj, dict):
                items = [x for x in obj.get("samples", []) if isinstance(x, dict)]

        rows: list[AnswerCandidate] = []
        for item in items:
            query = str(item.get("query", item.get("user", ""))).strip()
            label = str(item.get("label", item.get("semantic_label", ""))).strip()
            answer = str(item.get("answer", item.get("assistant", ""))).strip()
            concept = str(item.get("concept", "")).strip()
            concepts = [
                str(x).strip()
                for x in item.get("concepts", [])
                if str(x).strip()
            ]
            if concept and concept not in concepts:
                concepts.insert(0, concept)

            if not query or not answer:
                continue
            rows.append(
                AnswerCandidate(
                    query=query,
                    label=label or "unknown",
                    intent=str(item.get("intent", "definition")).strip() or "general",
                    concepts=concepts,
                    truth_status=str(
                        item.get("truth_status", "UNVERIFIED")
                    ).strip().upper(),
                    answer=answer,
                )
            )
        return rows

    @classmethod
    def load(cls, path: str | Path = DEFAULT_ANSWER_MEMORY):
        return cls(cls._rows_from_path(path))

    @classmethod
    def load_many(cls, paths: Iterable[str | Path]):
        rows: list[AnswerCandidate] = []
        seen: set[tuple[str, str, str]] = set()
        for path in paths:
            for row in cls._rows_from_path(path):
                key = (
                    normalize_text(row.query),
                    normalize_text(row.answer),
                    normalize_text(row.label),
                )
                if key in seen:
                    continue
                seen.add(key)
                rows.append(row)
        return cls(rows)

    def append_persistent(
        self,
        path: str | Path,
        *,
        query: str,
        answer: str,
        label: str,
        intent: str | None,
        concepts: list[str] | None,
        truth_status: str = "UNVERIFIED",
        source: str = "chat-teach-answer",
    ) -> bool:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        candidate = AnswerCandidate(
            query=query.strip(),
            label=(label or "unknown").strip(),
            intent=(intent or "general").strip(),
            concepts=[str(x).strip() for x in (concepts or []) if str(x).strip()],
            truth_status=(truth_status or "UNVERIFIED").strip().upper(),
            answer=answer.strip(),
        )
        if not candidate.query or not candidate.answer:
            return False

        fingerprint = (
            normalize_text(candidate.query),
            normalize_text(candidate.answer),
            normalize_text(candidate.label),
        )
        for row in self.rows:
            old = (
                normalize_text(row.query),
                normalize_text(row.answer),
                normalize_text(row.label),
            )
            if old == fingerprint:
                return False

        payload = {
            "query": candidate.query,
            "answer": candidate.answer,
            "label": candidate.label,
            "intent": candidate.intent,
            "concepts": candidate.concepts,
            "truth_status": candidate.truth_status,
            "source": source,
        }
        with p.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self.rows.append(candidate)
        return True

    def upsert_persistent(
        self,
        path: str | Path,
        *,
        query: str,
        answer: str,
        label: str,
        intent: str | None,
        concepts: list[str] | None,
        truth_status: str = "UNVERIFIED",
        source: str = "chat-fact-merge",
    ) -> bool:
        """Replace prior learned answer for the same query/concept and append the new canonical answer."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        qn = normalize_text(query)
        concept_set = {
            normalize_text(x)
            for x in (concepts or [])
            if str(x).strip()
        }

        kept: list[dict] = []
        changed = False
        if p.exists():
            for raw in p.read_text(encoding="utf-8").splitlines():
                if not raw.strip():
                    continue
                try:
                    row = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                row_query = normalize_text(row.get("query", row.get("user", "")))
                row_concepts = {
                    normalize_text(x)
                    for x in row.get("concepts", [])
                    if str(x).strip()
                }
                same_query = row_query == qn
                same_concept = bool(concept_set and concept_set & row_concepts)
                if same_query or same_concept:
                    changed = True
                    continue
                kept.append(row)

        payload = {
            "query": query.strip(),
            "answer": answer.strip(),
            "label": (label or "unknown").strip(),
            "intent": (intent or "general").strip(),
            "concepts": [str(x).strip() for x in (concepts or []) if str(x).strip()],
            "truth_status": (truth_status or "UNVERIFIED").strip().upper(),
            "source": source,
        }
        kept.append(payload)
        p.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in kept),
            encoding="utf-8",
        )

        self.rows = [
            row
            for row in self.rows
            if normalize_text(row.query) != qn
            and not (
                concept_set
                and concept_set
                & {normalize_text(x) for x in row.concepts if str(x).strip()}
            )
        ]
        self.rows.insert(
            0,
            AnswerCandidate(
                query=payload["query"],
                label=payload["label"],
                intent=payload["intent"],
                concepts=payload["concepts"],
                truth_status=payload["truth_status"],
                answer=payload["answer"],
            ),
        )
        return True

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
    """Reject answer-memory candidates marked unsafe by either truth source."""
    blocked = {"FALSE", "CONTESTED", "OUTDATED"}

    if candidate is not None:
        candidate_state = str(
            candidate.truth_status or "UNVERIFIED"
        ).upper()
        if candidate_state in blocked:
            return False

    if truth_record is not None:
        state = str(
            truth_record.get("truth_status", "UNVERIFIED")
        ).upper()
        if state in blocked:
            return False

    # TRUE/UNVERIFIED candidates remain usable. UNVERIFIED still carries
    # the runtime warning from the semantic truth record when available.
    return True
