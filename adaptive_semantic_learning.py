# adaptive_semantic_learning.py
#
# LLM_SEM v0.3 Adaptive Semantic Learning.
#
# Adds a small persistent semantic memory on top of the frozen base encoder.
# Teaching a previously unknown utterance adds a labeled example, after which
# the router can be refit immediately without retraining the Transformer.

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

from semantic_eval import LabeledSentence


@dataclass(frozen=True)
class SemanticMemoryEntry:
    label: str
    text: str
    source: str = "manual"
    timestamp: str = ""


def normalize_text(text: str) -> str:
    return " ".join(text.strip().split())


def load_semantic_memory(path: Path) -> List[LabeledSentence]:
    if not path.exists():
        return []

    rows: List[LabeledSentence] = []
    seen: set[tuple[str, str]] = set()

    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            item = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc

        label = normalize_text(str(item.get("label", "")))
        text = normalize_text(str(item.get("text", "")))
        if not label or not text:
            continue

        key = (label, text)
        if key not in seen:
            rows.append(LabeledSentence(label=label, text=text))
            seen.add(key)

    return rows


def append_semantic_memory(
    path: Path,
    label: str,
    text: str,
    source: str = "manual",
) -> bool:
    label = normalize_text(label)
    text = normalize_text(text)

    if not label:
        raise ValueError("label must not be empty")
    if not text:
        raise ValueError("text must not be empty")

    existing = {
        (row.label, row.text)
        for row in load_semantic_memory(path)
    }
    if (label, text) in existing:
        return False

    path.parent.mkdir(parents=True, exist_ok=True)
    entry = SemanticMemoryEntry(
        label=label,
        text=text,
        source=source,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    )
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(entry), ensure_ascii=False) + "\n")
    return True


def merge_samples(
    base_samples: Sequence[LabeledSentence],
    adaptive_samples: Iterable[LabeledSentence],
) -> List[LabeledSentence]:
    merged: List[LabeledSentence] = []
    seen: set[tuple[str, str]] = set()

    for row in list(base_samples) + list(adaptive_samples):
        key = (row.label, normalize_text(row.text))
        if key in seen:
            continue
        merged.append(LabeledSentence(label=row.label, text=normalize_text(row.text)))
        seen.add(key)

    return merged


def pending_count(path: Path) -> int:
    return len(load_semantic_memory(path))


def exact_memory_label(
    path: Path,
    text: str,
) -> str | None:
    """Return the explicitly taught label for an exact normalized utterance."""
    target = normalize_text(text)
    for row in load_semantic_memory(path):
        if normalize_text(row.text) == target:
            return row.label
    return None


def forget_semantic_memory(
    path: Path,
    text: str,
) -> bool:
    """Remove all adaptive-memory entries matching the normalized text."""
    target = normalize_text(text)
    if not path.exists():
        return False

    kept = []
    removed = False
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        item = json.loads(raw)
        item_text = normalize_text(str(item.get("text", "")))
        if item_text == target:
            removed = True
            continue
        kept.append(item)

    if removed:
        path.write_text(
            "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in kept),
            encoding="utf-8",
        )
    return removed


def relabel_semantic_memory(
    path: Path,
    text: str,
    new_label: str,
) -> bool:
    """Replace the label for all adaptive-memory entries matching text."""
    target = normalize_text(text)
    new_label = normalize_text(new_label)
    if not new_label:
        raise ValueError("new_label must not be empty")
    if not path.exists():
        return False

    rows = []
    changed = False
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        item = json.loads(raw)
        item_text = normalize_text(str(item.get("text", "")))
        if item_text == target:
            item["label"] = new_label
            item["source"] = "chat-relabel"
            item["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            changed = True
        rows.append(item)

    if changed:
        path.write_text(
            "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in rows),
            encoding="utf-8",
        )
    return changed
