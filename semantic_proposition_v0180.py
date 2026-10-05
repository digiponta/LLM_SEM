#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LLM_SEM v0.18.0 Semantic Proposition Compose / Decompose

Canonical internal form:
    Proposition(subject="X", value="Y")

Bidirectional normalization:
    XはYである。
    XはZである。
        -> compose -> Xは、Yであり、Zである。

    Xは、Yであり、Zである。
        -> decompose -> (X,Y), (X,Z)

The proposition store is deliberately symbolic and auditable.  It is separate
from Semantic Memory and from model weights so that later versions can choose
whether propositions remain external, are consolidated, or are internalized.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import json
import re
import unicodedata
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


TRAILING = "。．.!！?？"
LEADING_SEPARATORS = "、，, "
VALUE_SEPARATORS = re.compile(r"\s*(?:、|，|,)\s*")


@dataclass(frozen=True)
class Proposition:
    subject: str
    value: str

    def normalized(self) -> "Proposition":
        return Proposition(
            subject=normalize_fragment(self.subject),
            value=normalize_fragment(self.value),
        )


def normalize_fragment(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text)).strip()
    text = text.strip(TRAILING).strip()
    return text


def _strip_subject_marker(text: str):
    # Prefer the first Japanese topic marker.  v0.18.0 intentionally supports
    # the canonical declarative form used by the LLM_SEM experiments.
    if "は" not in text:
        return None
    subject, body = text.split("は", 1)
    subject = normalize_fragment(subject)
    body = body.lstrip(LEADING_SEPARATORS).strip()
    if not subject or not body:
        return None
    return subject, body


def decompose_statement(text: str) -> List[Proposition]:
    """Decompose one canonical Japanese copular statement.

    Supported examples:
      XはYである
      Xは、Yである。
      Xは、Yであり、Zである。
      XはYでありZである

    Returns [] when the input is outside the supported proposition grammar.
    """
    normalized = unicodedata.normalize("NFKC", str(text)).strip()
    normalized = normalized.rstrip(TRAILING).strip()

    split = _strip_subject_marker(normalized)
    if split is None:
        return []

    subject, body = split

    # Canonical compound form uses "であり" for every non-final item and
    # "である" for the final item.
    if not body.endswith("である"):
        return []

    body = body[: -len("である")].strip()
    if not body:
        return []

    # "Yであり、Z" -> ["Y", "Z"]
    raw_parts = re.split(r"\s*であり\s*(?:、|，|,)?\s*", body)
    values: List[str] = []

    for raw in raw_parts:
        value = normalize_fragment(raw.lstrip(LEADING_SEPARATORS))
        if value and value not in values:
            values.append(value)

    return [
        Proposition(subject=subject, value=value).normalized()
        for value in values
    ]


def compose_propositions(
    propositions: Sequence[Proposition],
    subject: str | None = None,
) -> str:
    """Compose atomic propositions for one subject into canonical Japanese."""
    normalized = [p.normalized() for p in propositions]

    if subject is not None:
        target = normalize_fragment(subject)
        normalized = [p for p in normalized if p.subject == target]

    if not normalized:
        return ""

    subjects = {p.subject for p in normalized}
    if len(subjects) != 1:
        raise ValueError(
            "compose_propositions requires propositions for exactly one subject"
        )

    subject_value = normalized[0].subject
    values: List[str] = []
    for p in normalized:
        if p.value not in values:
            values.append(p.value)

    if len(values) == 1:
        return f"{subject_value}は、{values[0]}である。"

    head = "であり、".join(values[:-1])
    return f"{subject_value}は、{head}であり、{values[-1]}である。"


def merge_propositions(
    existing: Iterable[Proposition],
    incoming: Iterable[Proposition],
) -> List[Proposition]:
    """Stable de-duplicating union of propositions."""
    result: List[Proposition] = []
    seen = set()

    for proposition in list(existing) + list(incoming):
        p = proposition.normalized()
        key = (p.subject, p.value)
        if not p.subject or not p.value or key in seen:
            continue
        seen.add(key)
        result.append(p)

    return result


def group_by_subject(
    propositions: Iterable[Proposition],
) -> Dict[str, List[Proposition]]:
    grouped: Dict[str, List[Proposition]] = {}
    for p in propositions:
        q = p.normalized()
        grouped.setdefault(q.subject, []).append(q)
    return grouped


def load_propositions(path: Path) -> List[Proposition]:
    if not path.exists():
        return []

    rows: List[Proposition] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                item = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid proposition JSON at line {line_no}: {exc}"
                ) from exc

            subject = normalize_fragment(item.get("subject", ""))
            value = normalize_fragment(item.get("value", ""))
            if not subject or not value:
                raise ValueError(
                    f"invalid proposition at line {line_no}: {item!r}"
                )
            rows.append(Proposition(subject, value))

    return merge_propositions([], rows)


def save_propositions(path: Path, propositions: Iterable[Proposition]) -> None:
    rows = merge_propositions([], propositions)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for p in rows:
            f.write(
                json.dumps(asdict(p), ensure_ascii=False) + "\n"
            )


def add_statement(path: Path, statement: str) -> List[Proposition]:
    incoming = decompose_statement(statement)
    if not incoming:
        return []

    current = load_propositions(path)
    merged = merge_propositions(current, incoming)
    save_propositions(path, merged)
    return incoming


def compose_subject(path: Path, subject: str) -> str:
    rows = load_propositions(path)
    return compose_propositions(rows, subject=subject)
