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


MEMORY_ACTIVE_STATES = {"ACTIVE", "TRAINING", "VALIDATING", "FAILED"}
MEMORY_ALL_STATES = MEMORY_ACTIVE_STATES | {"CONSOLIDATED"}
TRUTH_STATES = {"TRUE", "FALSE", "UNVERIFIED", "CONTESTED", "OUTDATED"}


@dataclass(frozen=True)
class SemanticMemoryEntry:
    label: str
    text: str
    source: str = "manual"
    timestamp: str = ""
    status: str = "ACTIVE"
    model_version: str = ""
    verified: bool = False
    truth_status: str = "UNVERIFIED"
    truth_confidence: float = 0.0
    provenance: str = ""
    correction_target: str = ""


def normalize_text(text: str) -> str:
    return " ".join(text.strip().split())


def load_semantic_memory_records(path: Path) -> list[dict]:
    """Load raw semantic-memory records with lifecycle defaults."""
    if not path.exists():
        return []

    records: list[dict] = []
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
        status = str(item.get("status", "ACTIVE")).upper().strip() or "ACTIVE"
        if status not in MEMORY_ALL_STATES:
            status = "ACTIVE"
        item["label"] = label
        item["text"] = text
        item["status"] = status
        item["model_version"] = str(item.get("model_version", ""))
        item["verified"] = bool(item.get("verified", False))
        truth_status = str(item.get("truth_status", "UNVERIFIED")).upper().strip() or "UNVERIFIED"
        if truth_status not in TRUTH_STATES:
            truth_status = "UNVERIFIED"
        item["truth_status"] = truth_status
        try:
            item["truth_confidence"] = float(item.get("truth_confidence", 0.0))
        except (TypeError, ValueError):
            item["truth_confidence"] = 0.0
        item["truth_confidence"] = max(0.0, min(1.0, item["truth_confidence"]))
        item["provenance"] = str(item.get("provenance", item.get("source", "")))
        item["correction_target"] = normalize_text(str(item.get("correction_target", "")))
        records.append(item)
    return records


def load_semantic_memory(path: Path) -> List[LabeledSentence]:
    if not path.exists():
        return []

    rows: List[LabeledSentence] = []
    seen: set[tuple[str, str]] = set()

    for item in load_semantic_memory_records(path):
        if item["status"] not in MEMORY_ACTIVE_STATES:
            continue

        label = item["label"]
        text = item["text"]
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
    *,
    truth_status: str = "UNVERIFIED",
    truth_confidence: float = 0.0,
    provenance: str = "",
    correction_target: str = "",
) -> bool:
    label = normalize_text(label)
    text = normalize_text(text)

    if not label:
        raise ValueError("label must not be empty")
    if not text:
        raise ValueError("text must not be empty")
    truth_status = truth_status.upper().strip()
    if truth_status not in TRUTH_STATES:
        raise ValueError(f"invalid truth_status: {truth_status}")
    truth_confidence = max(0.0, min(1.0, float(truth_confidence)))
    provenance = provenance.strip() or source
    correction_target = normalize_text(correction_target)

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
        status="ACTIVE",
        model_version="",
        verified=False,
        truth_status=truth_status,
        truth_confidence=truth_confidence,
        provenance=provenance,
        correction_target=correction_target,
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


def memory_status_counts(path: Path) -> dict[str, int]:
    counts = {state: 0 for state in sorted(MEMORY_ALL_STATES)}
    for item in load_semantic_memory_records(path):
        counts[item["status"]] = counts.get(item["status"], 0) + 1
    return counts


def update_memory_status(
    path: Path,
    text: str,
    status: str,
    *,
    model_version: str = "",
    verified: bool | None = None,
) -> bool:
    """Update lifecycle state for matching records.

    Semantic Memory remains authoritative for ACTIVE/TRAINING/VALIDATING/FAILED.
    Only CONSOLIDATED records are removed from normal adaptive routing.
    """
    target = normalize_text(text)
    status = status.upper().strip()
    if status not in MEMORY_ALL_STATES:
        raise ValueError(f"invalid semantic-memory status: {status}")
    if not path.exists():
        return False

    rows = []
    changed = False
    now = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    for item in load_semantic_memory_records(path):
        if normalize_text(item["text"]) == target:
            item["status"] = status
            item["timestamp"] = now
            if model_version:
                item["model_version"] = model_version
            if verified is not None:
                item["verified"] = bool(verified)
            elif status != "CONSOLIDATED":
                item["verified"] = False
            changed = True
        rows.append(item)

    if changed:
        path.write_text(
            "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in rows),
            encoding="utf-8",
        )
    return changed


def exact_memory_record(path: Path, text: str) -> dict | None:
    target = normalize_text(text)
    for item in load_semantic_memory_records(path):
        if item["status"] in MEMORY_ACTIVE_STATES and normalize_text(item["text"]) == target:
            return item
    return None


def exact_memory_label(
    path: Path,
    text: str,
) -> str | None:
    """Return the explicitly taught label for an exact normalized utterance."""
    item = exact_memory_record(path, text)
    return str(item["label"]) if item is not None else None


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


def truth_status_counts(path: Path) -> dict[str, int]:
    counts = {state: 0 for state in sorted(TRUTH_STATES)}
    for item in load_semantic_memory_records(path):
        state = str(item.get("truth_status", "UNVERIFIED"))
        counts[state] = counts.get(state, 0) + 1
    return counts


def update_memory_truth(
    path: Path,
    text: str,
    truth_status: str,
    *,
    truth_confidence: float | None = None,
    provenance: str | None = None,
    correction_target: str | None = None,
) -> bool:
    """Update truth metadata without changing lifecycle state."""
    target = normalize_text(text)
    truth_status = truth_status.upper().strip()
    if truth_status not in TRUTH_STATES:
        raise ValueError(f"invalid truth_status: {truth_status}")
    if not path.exists():
        return False

    rows = []
    changed = False
    now = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    for item in load_semantic_memory_records(path):
        if normalize_text(item["text"]) == target:
            item["truth_status"] = truth_status
            if truth_confidence is not None:
                item["truth_confidence"] = max(
                    0.0, min(1.0, float(truth_confidence))
                )
            if provenance is not None:
                item["provenance"] = str(provenance).strip()
            if correction_target is not None:
                item["correction_target"] = normalize_text(correction_target)
            item["timestamp"] = now
            changed = True
        rows.append(item)

    if changed:
        path.write_text(
            "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in rows),
            encoding="utf-8",
        )
    return changed


def truth_notice(item: dict | None) -> str | None:
    """Return a user-facing notice for non-TRUE truth states."""
    if item is None:
        return None
    state = str(item.get("truth_status", "UNVERIFIED")).upper()
    if state == "TRUE":
        return None
    if state == "FALSE":
        correction = normalize_text(str(item.get("correction_target", "")))
        suffix = f" Corrected information: {correction}" if correction else ""
        return "This information is stored as FALSE / incorrect." + suffix
    if state == "CONTESTED":
        return "This information is stored as CONTESTED; multiple interpretations or claims may exist."
    if state == "OUTDATED":
        return "This information is stored as OUTDATED and may no longer be current."
    return "This information is stored as UNVERIFIED and has not been confirmed."

def exact_truth_record(path: Path, text: str) -> dict | None:
    """Return truth metadata for an exact record regardless of lifecycle state."""
    target = normalize_text(text)
    matches = [
        item for item in load_semantic_memory_records(path)
        if normalize_text(item["text"]) == target
    ]
    if not matches:
        return None

    # Prefer active/authoritative memory while migration is incomplete.
    for item in matches:
        if item["status"] in MEMORY_ACTIVE_STATES:
            return item
    return matches[0]
