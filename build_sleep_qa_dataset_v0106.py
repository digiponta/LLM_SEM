# build_sleep_qa_dataset_v0106.py
#
# LLM_SEM v0.10.6
# Build one QA dataset for /sleep from base QA, learned Answer Memory,
# and canonicalized Relation Memory.

from __future__ import annotations

import argparse
import json
from pathlib import Path

from relation_memory_v0101 import compose_subject_facts, load_relation_facts


def load_base(path: Path) -> list[dict]:
    if not path.exists():
        return []
    obj = json.loads(path.read_text(encoding="utf-8"))
    return [x for x in obj.get("samples", []) if isinstance(x, dict)]


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def normalize_row(row: dict) -> dict | None:
    query = str(row.get("query", row.get("user", ""))).strip()
    answer = str(row.get("answer", row.get("assistant", ""))).strip()
    label = str(row.get("label", row.get("semantic_label", "unknown"))).strip() or "unknown"
    concepts = [
        str(x).strip()
        for x in row.get("concepts", [])
        if str(x).strip()
    ]
    concept = str(row.get("concept", "")).strip()
    if concept and concept not in concepts:
        concepts.insert(0, concept)
    if not query or not answer:
        return None
    return {
        "query": query,
        "answer": answer,
        "label": label,
        "intent": str(row.get("intent", "definition")).strip() or "definition",
        "concepts": concepts,
        "truth_status": str(row.get("truth_status", "UNVERIFIED")).strip().upper(),
    }


def relation_rows(path: Path, default_label: str = "unknown") -> list[dict]:
    facts = load_relation_facts(path)
    subjects = []
    seen = set()
    for fact in facts:
        subject = str(fact.get("subject", "")).strip()
        if subject and subject not in seen:
            seen.add(subject)
            subjects.append(subject)

    rows = []
    for subject in subjects:
        answer = compose_subject_facts(path, subject)
        if not answer:
            continue
        for query in (
            subject,
            f"{subject}とは",
            f"{subject}について教えて",
            f"{subject}を説明して",
        ):
            rows.append({
                "query": query,
                "answer": answer,
                "label": default_label,
                "intent": "definition",
                "concepts": [subject],
                "truth_status": "UNVERIFIED",
            })
    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LLM_SEM v0.10.6 sleep QA dataset builder")
    p.add_argument("--base", default="data/semantic_guided_qa_v097.json")
    p.add_argument("--learned", default="data/semantic_answer_memory_learned.jsonl")
    p.add_argument("--relation", default="data/relation_memory_v0101.jsonl")
    p.add_argument("--output", default="data/semantic_sleep_qa_v0106.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rows = []

    for raw in load_base(Path(args.base)):
        row = normalize_row(raw)
        if row:
            rows.append(row)

    # Learned human answers come after base rows and replace exact-query duplicates.
    learned = []
    for raw in load_jsonl(Path(args.learned)):
        row = normalize_row(raw)
        if row:
            learned.append(row)

    relations = relation_rows(Path(args.relation))

    by_query: dict[str, dict] = {}
    order: list[str] = []

    def put(row: dict) -> None:
        key = "".join(row["query"].split()).lower()
        if key not in by_query:
            order.append(key)
        by_query[key] = row

    for row in rows:
        put(row)
    for row in relations:
        put(row)
    for row in learned:
        put(row)

    merged = [by_query[key] for key in order]

    # Fill unknown relation labels from exact/overlapping learned/base concept knowledge.
    concept_labels: dict[str, str] = {}
    for row in merged:
        if row["label"] == "unknown":
            continue
        for concept in row["concepts"]:
            concept_labels.setdefault(concept, row["label"])

    for row in merged:
        if row["label"] == "unknown":
            for concept in row["concepts"]:
                if concept in concept_labels:
                    row["label"] = concept_labels[concept]
                    break

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "version": "0.10.6",
                "purpose": "full-sleep-consolidation",
                "samples": merged,
            },
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    print("=" * 92)
    print(" LLM_SEM v0.10.6 Sleep QA Dataset Builder")
    print("=" * 92)
    print("Base rows    :", len(rows))
    print("Relation rows:", len(relations))
    print("Learned rows :", len(learned))
    print("Merged rows  :", len(merged))
    print("Saved        :", out)


if __name__ == "__main__":
    main()
