# relation_memory_v0101.py
#
# LLM_SEM v0.10.1
# Persistent relation-fact memory adapted from LLM_TRY v10.7.x.

from __future__ import annotations

import json
import re
from pathlib import Path


DEFAULT_RELATION_MEMORY = "data/relation_memory_v0101.jsonl"


def _normalize_condition(condition: str) -> tuple[str, bool]:
    text = condition.strip()
    if not text:
        return "", True
    polarity = not any(marker in text for marker in ("でない", "ではない", "ない"))
    predicate = re.sub(r"(?:でない|ではない|ない)$", "", text).strip()
    return predicate, polarity


def parse_relation_fact(text: str) -> dict | None:
    text = re.sub(r"[。．]+$", "", text.strip())
    if not text:
        return None

    condition = ""
    m_cond = re.fullmatch(r"^(?:(.+?)(?:のとき|とき)[、,]?\s*)?(.+)$", text)
    if not m_cond:
        return None
    condition = (m_cond.group(1) or "").strip()
    body = m_cond.group(2).strip()

    m_def = re.fullmatch(r"^([^\s。、！？?]{1,32})とは、?(.+)$", body)
    if m_def:
        subject = m_def.group(1).strip()
        value = m_def.group(2).strip(" 、,")
        pred, pol = _normalize_condition(condition)
        return {
            "subject": subject,
            "relation": "definition",
            "value": value,
            "condition": condition,
            "condition_predicate": pred,
            "condition_polarity": pol,
            "relation_context": "",
        }

    verb_patterns = (
        ("includes", r"^([^\s。、！？?]{1,32})は、?(?:(.+?上)[、,]?)?(.+?)を含む$"),
        ("belongs_to", r"^([^\s。、！？?]{1,32})は、?(?:(.+?上)[、,]?)?(.+?)に属する$"),
        ("has", r"^([^\s。、！？?]{1,32})は、?(?:(.+?上)[、,]?)?(.+?)を持つ$"),
        ("used_for", r"^([^\s。、！？?]{1,32})は、?(?:(.+?上)[、,]?)?(.+?)に(?:使われる|利用される)$"),
    )
    for relation, pattern in verb_patterns:
        m = re.fullmatch(pattern, body)
        if m:
            subject = m.group(1).strip()
            context = (m.group(2) or "").strip(" 、,")
            value = m.group(3).strip(" 、,")
            pred, pol = _normalize_condition(condition)
            return {
                "subject": subject,
                "relation": relation,
                "value": value,
                "condition": condition,
                "condition_predicate": pred,
                "condition_polarity": pol,
                "relation_context": context,
            }

    m_attr = re.fullmatch(
        r"^([^\s。、！？?]{1,32})の([^\s。、！？?]{1,24})は、?(.+?)(?:である|です)$",
        body,
    )
    if m_attr:
        pred, pol = _normalize_condition(condition)
        return {
            "subject": m_attr.group(1).strip(),
            "relation": m_attr.group(2).strip(),
            "value": m_attr.group(3).strip(" 、,"),
            "condition": condition,
            "condition_predicate": pred,
            "condition_polarity": pol,
            "relation_context": "",
        }

    m_is = re.fullmatch(
        r"^([^\s。、！？?]{1,32})は、?(.+?)(?:である|です)$",
        body,
    )
    if m_is:
        pred, pol = _normalize_condition(condition)
        return {
            "subject": m_is.group(1).strip(),
            "relation": "is",
            "value": m_is.group(2).strip(" 、,"),
            "condition": condition,
            "condition_predicate": pred,
            "condition_polarity": pol,
            "relation_context": "",
        }

    # LLM_TRY-compatible short fact: XはY
    m_short_is = re.fullmatch(
        r"^([^\s。、！？?]{1,32})は、?([^。、！？?]{1,80})$",
        body,
    )
    if m_short_is:
        pred, pol = _normalize_condition(condition)
        return {
            "subject": m_short_is.group(1).strip(),
            "relation": "is",
            "value": m_short_is.group(2).strip(" 、,"),
            "condition": condition,
            "condition_predicate": pred,
            "condition_polarity": pol,
            "relation_context": "",
        }
    return None


def append_relation_fact(path: str | Path, text: str, source: str = "chat-teach-answer") -> bool:
    fact = parse_relation_fact(text)
    if fact is None:
        return False

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    key = (
        fact["subject"].lower(),
        fact["relation"].lower(),
        fact["value"].lower(),
        fact["condition"].lower(),
        fact["relation_context"].lower(),
    )
    if p.exists():
        for raw in p.read_text(encoding="utf-8").splitlines():
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError:
                continue
            old = (
                str(row.get("subject", "")).strip().lower(),
                str(row.get("relation", "")).strip().lower(),
                str(row.get("value", "")).strip().lower(),
                str(row.get("condition", "")).strip().lower(),
                str(row.get("relation_context", "")).strip().lower(),
            )
            if old == key:
                return False

    payload = dict(fact)
    payload["source"] = source
    with p.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return True


def load_relation_facts(path: str | Path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    out = []
    for raw in p.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict) and row.get("subject") and row.get("relation") and row.get("value"):
            out.append(row)
    return out


def facts_for_subject(path: str | Path, subject: str) -> list[dict]:
    target = subject.strip().lower()
    rows = []
    seen = set()
    for row in load_relation_facts(path):
        if str(row.get("subject", "")).strip().lower() != target:
            continue
        key = (
            str(row.get("relation", "")).strip().lower(),
            str(row.get("value", "")).strip().lower(),
            str(row.get("condition", "")).strip().lower(),
            str(row.get("relation_context", "")).strip().lower(),
        )
        if key in seen:
            continue
        seen.add(key)
        rows.append(row)
    return rows


def compose_fact_answer(subject: str, values: list[str]) -> str:
    """Compose multiple unconditional 'is' facts for one subject."""
    clean = []
    seen = set()
    for value in values:
        v = str(value).strip().rstrip("。")
        if not v:
            continue
        key = v.lower()
        if key in seen:
            continue
        seen.add(key)
        clean.append(v)

    if not clean:
        return ""
    if len(clean) == 1:
        return f"{subject}は、{clean[0]}である。"
    if len(clean) == 2:
        return f"{subject}は、{clean[0]}であり、{clean[1]}である。"
    head = "、".join(f"{v}であり" for v in clean[:-1])
    return f"{subject}は、{head}、{clean[-1]}である。"


def _continuation_clause(subject: str, fact: dict) -> str:
    """Render one unconditional fact as a continuation after 'Xは、...'."""
    relation = str(fact.get("relation", "")).strip()
    value = str(fact.get("value", "")).strip()
    context = str(fact.get("relation_context", "")).strip()

    if relation == "definition":
        return f"{value}であり"
    if relation == "includes":
        prefix = f"{context}、" if context else ""
        return f"{prefix}{value}を含み"
    if relation == "belongs_to":
        prefix = f"{context}、" if context else ""
        return f"{prefix}{value}に属し"
    if relation == "has":
        prefix = f"{context}、" if context else ""
        return f"{prefix}{value}を持ち"
    if relation == "used_for":
        prefix = f"{context}、" if context else ""
        return f"{prefix}{value}に利用され"
    if relation == "is":
        return f"{value}であり"
    return f"{relation}が{value}であり"


def _final_clause(subject: str, fact: dict) -> str:
    """Render one unconditional fact as the sentence-final predicate."""
    relation = str(fact.get("relation", "")).strip()
    value = str(fact.get("value", "")).strip()
    context = str(fact.get("relation_context", "")).strip()

    if relation == "definition":
        return f"{value}である"
    if relation == "includes":
        prefix = f"{context}、" if context else ""
        return f"{prefix}{value}を含む"
    if relation == "belongs_to":
        prefix = f"{context}、" if context else ""
        return f"{prefix}{value}に属する"
    if relation == "has":
        prefix = f"{context}、" if context else ""
        return f"{prefix}{value}を持つ"
    if relation == "used_for":
        prefix = f"{context}、" if context else ""
        return f"{prefix}{value}に利用される"
    if relation == "is":
        return f"{value}である"
    return f"{relation}が{value}である"


def compose_subject_facts(path: str | Path, subject: str) -> str:
    """Naturally compose compatible facts for one subject.

    Unconditional facts are compressed into one Japanese sentence when
    possible. Conditional facts remain separate so their scope is preserved.
    """
    facts = facts_for_subject(path, subject)
    if not facts:
        return ""

    unconditional = [
        f for f in facts
        if not str(f.get("condition", "")).strip()
    ]
    conditional = [
        f for f in facts
        if str(f.get("condition", "")).strip()
    ]

    sentences: list[str] = []

    if unconditional:
        # Prefer generic identity/definition information first, then relations.
        priority = {
            "definition": 0,
            "is": 1,
            "belongs_to": 2,
            "has": 3,
            "includes": 4,
            "used_for": 5,
        }
        ordered = sorted(
            unconditional,
            key=lambda f: priority.get(str(f.get("relation", "")).strip(), 9),
        )

        if len(ordered) == 1:
            sentences.append(
                f"{subject}は、{_final_clause(subject, ordered[0])}。"
            )
        else:
            heads = [
                _continuation_clause(subject, fact)
                for fact in ordered[:-1]
            ]
            tail = _final_clause(subject, ordered[-1])
            sentences.append(
                f"{subject}は、" + "、".join(heads + [tail]) + "。"
            )

    for fact in conditional:
        condition = str(fact.get("condition", "")).strip()
        body = _final_clause(subject, fact)
        sentences.append(f"{condition}のとき、{subject}は、{body}。")

    return "".join(sentences)
