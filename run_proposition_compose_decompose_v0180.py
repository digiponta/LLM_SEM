#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LLM_SEM v0.18.0 Semantic Proposition Compose / Decompose Regression
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from semantic_proposition_v0180 import (
    Proposition,
    add_statement,
    compose_propositions,
    compose_subject,
    decompose_statement,
    load_propositions,
    merge_propositions,
)


def check(name: str, condition: bool, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" : {detail}" if detail else ""))
    if not condition:
        raise AssertionError(name)


def main():
    print("=" * 88)
    print(" LLM_SEM v0.18.0 Semantic Proposition Compose / Decompose Regression")
    print("=" * 88)

    # A) Atomic decomposition
    p = decompose_statement("XはYである。")
    check(
        "decompose-atomic",
        p == [Proposition("X", "Y")],
        str(p),
    )

    # B) Compound decomposition
    p = decompose_statement("Xは、Yであり、Zである。")
    check(
        "decompose-compound",
        p == [Proposition("X", "Y"), Proposition("X", "Z")],
        str(p),
    )

    # C) Composition
    text = compose_propositions([
        Proposition("X", "Y"),
        Proposition("X", "Z"),
    ])
    check(
        "compose-two",
        text == "Xは、Yであり、Zである。",
        text,
    )

    # D) Reverse round trip
    original = "量子センサーは、高感度であり、量子的性質を利用する技術である。"
    atoms = decompose_statement(original)
    rebuilt = compose_propositions(atoms)
    check(
        "roundtrip-compound",
        rebuilt == original,
        rebuilt,
    )

    # E) Merge from separate teachings
    merged = merge_propositions(
        [Proposition("X", "Y")],
        [Proposition("X", "Z")],
    )
    check(
        "merge-separate",
        compose_propositions(merged) == "Xは、Yであり、Zである。",
        compose_propositions(merged),
    )

    # F) Deduplication
    merged = merge_propositions(
        [Proposition("X", "Y")],
        [Proposition("X", "Y"), Proposition("X", "Z")],
    )
    check(
        "deduplicate",
        merged == [Proposition("X", "Y"), Proposition("X", "Z")],
        str(merged),
    )

    # G) Persistent store: separate -> composed
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "props.jsonl"
        a = add_statement(path, "架空装置は高速である。")
        b = add_statement(path, "架空装置は低消費電力である。")
        saved = load_propositions(path)
        rendered = compose_subject(path, "架空装置")

        check("store-first-teach", len(a) == 1, str(a))
        check("store-second-teach", len(b) == 1, str(b))
        check("store-count", len(saved) == 2, str(saved))
        check(
            "store-compose",
            rendered == "架空装置は、高速であり、低消費電力である。",
            rendered,
        )

    # H) Persistent store: compound -> atoms
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "props.jsonl"
        added = add_statement(
            path,
            "架空装置は、高速であり、低消費電力である。",
        )
        saved = load_propositions(path)
        check("compound-store-split-count", len(added) == 2, str(added))
        check("compound-store-atomic-count", len(saved) == 2, str(saved))

    # I) Unsupported grammar must fail closed
    check(
        "unsupported-query-fails-closed",
        decompose_statement("Xとは何ですか") == [],
    )

    print()
    print("=" * 88)
    print(" RESULT")
    print("=" * 88)
    print("Compose                  : PASS")
    print("Decompose                : PASS")
    print("Bidirectional round-trip : PASS")
    print("Persistent merge         : PASS")
    print("STATUS                   : PROPOSITION_COMPOSE_DECOMPOSE_VALIDATED")


if __name__ == "__main__":
    main()
