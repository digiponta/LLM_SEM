# run_v065_regression.py
#
# LLM_SEM v0.6.5 Adaptive Composition Runtime Regression
#
# Fast checks for the promoted composition runtime. No base model checkpoint is
# loaded. The test verifies source integration and the frozen gate policy.
#
# Run:
#   python run_v065_regression.py

from __future__ import annotations

import ast
from pathlib import Path

from adaptive_composition_runtime_v065 import (
    AdaptiveCompositionRuntime,
    BALANCED,
    RELATION_AWARE,
)


ROOT = Path(__file__).resolve().parent


def check(name, condition, detail=""):
    if not condition:
        suffix = f": {detail}" if detail else ""
        raise AssertionError(f"{name}{suffix}")
    print(f"[PASS] {name}")


def parse_file(name):
    source = (ROOT / name).read_text(encoding="utf-8")
    ast.parse(source)
    check(f"syntax:{name}", True)
    return source


def source_checks():
    chat = parse_file("chat.py")
    semantic = parse_file("semantic.py")
    runtime = parse_file("semantic_runtime_v2.py")
    comp = parse_file("adaptive_composition_runtime_v065.py")

    check(
        "chat:v065-banner",
        "LLM_SEM v0.6.5 Unified Semantic Runtime" in chat,
    )
    check(
        "chat:composition-import",
        "AdaptiveCompositionRuntime" in chat
        and "enrich_proposition_specs" in chat,
    )
    check("chat:composition-flag", "--composition" in chat)
    check(
        "chat:composition-enrichment",
        chat.count("enrich_proposition_specs(") >= 2,
    )
    check(
        "semantic:precomputed-vector",
        'precomputed_vector = spec.get("vector")' in semantic
        and 'model_type="adaptive-composition"' in semantic,
    )
    check(
        "summary:provenance",
        '"vector_role": p.vector.role' in runtime
        and '"attributes": dict(p.attributes)' in runtime,
    )
    check(
        "composition:confirmed-policy",
        'reason = "unseen_relation_seen_concepts"' in comp
        and 'reason = "otherwise_balanced"' in comp,
    )


def gate_checks():
    # decision() does not depend on checkpoint-loaded instance attributes.
    obj = AdaptiveCompositionRuntime.__new__(AdaptiveCompositionRuntime)

    known = obj.decision("new-subject", "has_property", "new-object")
    check("gate:known-relation-mode", known.mode == "balanced", known.mode)
    check("gate:known-relation-weights", known.weights == BALANCED)

    unseen_seen = obj.decision("GPU", "used_for", "data")
    check(
        "gate:unseen-relation-seen-concepts-mode",
        unseen_seen.mode == "relation_aware",
        unseen_seen.mode,
    )
    check(
        "gate:unseen-relation-seen-concepts-weights",
        unseen_seen.weights == RELATION_AWARE,
    )

    partial = obj.decision("GPU", "causes", "発熱")
    check("gate:partial-novelty-mode", partial.mode == "balanced", partial.mode)
    check("gate:partial-novelty-weights", partial.weights == BALANCED)

    novel = obj.decision("SSD", "is_a", "記憶装置")
    check("gate:novel-mode", novel.mode == "balanced", novel.mode)
    check("gate:novel-weights", novel.weights == BALANCED)


def required_files():
    names = [
        "adaptive_composition_runtime_v065.py",
        "run_adaptive_semantic_composition_v063.py",
        "run_adaptive_composition_confirmatory_v064.py",
        "semantic.py",
        "semantic_runtime_v2.py",
        "chat.py",
    ]
    for name in names:
        check(f"file:{name}", (ROOT / name).is_file())


def main():
    print("=" * 78)
    print(" LLM_SEM v0.6.5 Adaptive Composition Runtime Regression")
    print("=" * 78)

    required_files()
    source_checks()
    gate_checks()

    print("-" * 78)
    print("RESULT: PASS")
    print(
        "v0.6.5 runtime promotion is consistent: confirmed adaptive "
        "composition -> proposition vector + provenance -> SemanticDataV2."
    )


if __name__ == "__main__":
    main()
