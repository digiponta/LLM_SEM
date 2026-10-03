# build_incremental_sleep_dataset_v01023.py
#
# LLM_SEM v0.10.23
# Build a QA sleep dataset containing only failed/new mandatory concepts,
# while retaining ordinary optional/base rows for stabilization.

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(
        description="LLM_SEM v0.10.23 Incremental Sleep Dataset Builder"
    )
    p.add_argument("--dataset", required=True)
    p.add_argument("--precheck", required=True)
    p.add_argument("--output", required=True)
    return p.parse_args()


def concept_of(row: dict) -> str:
    concepts = [
        str(x).strip()
        for x in row.get("concepts", [])
        if str(x).strip()
    ]
    if concepts:
        return concepts[0]
    return str(row.get("query", "")).strip()


def main():
    args = parse_args()
    dataset_path = Path(args.dataset)
    precheck_path = Path(args.precheck)
    output_path = Path(args.output)

    obj = json.loads(dataset_path.read_text(encoding="utf-8"))
    precheck = json.loads(precheck_path.read_text(encoding="utf-8"))

    failed_concepts = {
        str(item.get("concept", "")).strip()
        for item in precheck.get("details", [])
        if not bool(item.get("passed", False))
        and str(item.get("concept", "")).strip()
    }

    samples = list(obj.get("samples", []))
    selected = []
    protected = 0
    mandatory_selected = 0

    for row in samples:
        must_train = bool(row.get("must_train", False))
        if not must_train:
            selected.append(row)
            continue

        concept = concept_of(row)
        if concept in failed_concepts:
            selected.append(row)
            mandatory_selected += 1
        else:
            protected += 1

    out = dict(obj)
    out["samples"] = selected
    out["incremental_sleep"] = {
        "version": "v0.10.23",
        "failed_concepts": sorted(failed_concepts),
        "protected_mandatory_rows": protected,
        "selected_mandatory_rows": mandatory_selected,
        "source_dataset": str(dataset_path),
        "precheck": str(precheck_path),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(out, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("=" * 96)
    print(" LLM_SEM v0.10.23 Incremental Sleep Dataset Builder")
    print("=" * 96)
    print("Failed/new concepts       :", ", ".join(sorted(failed_concepts)) or "(none)")
    print("Protected mandatory rows  :", protected)
    print("Selected mandatory rows   :", mandatory_selected)
    print("Optional/base rows retained:", len(selected) - mandatory_selected)
    print("Output                    :", output_path)

    if not failed_concepts or mandatory_selected == 0:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
