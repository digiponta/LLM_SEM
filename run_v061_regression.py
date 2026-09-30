# run_v061_regression.py
#
# LLM_SEM v0.6.1 Unified Semantic Runtime Regression
#
# Fast regression for the v0.6.1 integration contract. It does not require
# loading the language-model checkpoint. It verifies:
#   1) chat.py syntax and v0.6.1 integration markers
#   2) Semantic Data v2 / relation / proposition integration
#   3) adaptive multi-prototype + local-evidence runtime wiring
#   4) local-evidence decision behavior with deterministic synthetic vectors
#
# Run:
#   python run_v061_regression.py

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from adaptive_semantic_runtime import (
    build_multi_prototypes,
    decide_adaptive_route,
    encode_memory,
)
from semantic_eval import LabeledSentence


ROOT = Path(__file__).resolve().parent
CHAT = ROOT / "chat.py"


@dataclass
class RouteRow:
    label: str
    similarity: float


class FakeRouter:
    """Minimal router contract used by adaptive_semantic_runtime."""

    def __init__(self, vectors, base_routes):
        self.vectors = {
            text: F.normalize(torch.tensor(vector, dtype=torch.float32), dim=0)
            for text, vector in vectors.items()
        }
        self.base_routes = base_routes

    def _encode_tensor(self, text):
        return self.vectors[text]

    def route(self, text):
        label, similarity = self.base_routes[text]
        return [RouteRow(label, float(similarity))]


def check(name, condition, detail=""):
    if not condition:
        suffix = f": {detail}" if detail else ""
        raise AssertionError(f"{name}{suffix}")
    print(f"[PASS] {name}")


def source_checks():
    source = CHAT.read_text(encoding="utf-8")
    ast.parse(source)

    check("chat:syntax", True)
    check("chat:version", "LLM_SEM v0.6.1 Unified Semantic Runtime" in source)
    check(
        "chat:adaptive-runtime-import",
        "from adaptive_semantic_runtime import" in source
        and "build_multi_prototypes" in source
        and "decide_adaptive_route" in source
        and "encode_memory" in source,
    )
    check(
        "chat:semantic-v2",
        "from semantic_runtime_v2 import from_runtime_dict, runtime_summary"
        in source,
    )
    check(
        "chat:relations",
        "from semantic_relations_v035 import generate_semantic_relations"
        in source,
    )
    check(
        "chat:propositions",
        "from semantic_proposition_v036 import" in source
        and "extract_propositions" in source
        and "proposition_specs" in source,
    )
    check("chat:local-runtime-helper", "def evaluate_local_runtime(" in source)
    check("chat:adaptive-override", 'gate = "ACCEPT_ADAPTIVE"' in source)
    check("chat:runtime-command", 'if text == "/runtime":' in source)
    check("chat:runtime-metadata", '"local_purity": local_decision.local_purity' in source)


def adaptive_runtime_checks():
    # Two compact semantic regions. q_agree belongs to computer and the fixed
    # base also says computer -> normal ACCEPT.
    vectors = {
        "computer-1": [1.0, 0.0],
        "computer-2": [0.98, 0.10],
        "science-1": [0.0, 1.0],
        "science-2": [0.10, 0.98],
        "q_agree": [0.99, 0.04],
        # Strong local computer evidence, but fixed base says science:
        # should become ADAPTIVE_OVERRIDE under the v0.4.6/v0.6.1 policy.
        "q_override": [0.995, 0.02],
        # Middle region with insufficient memory similarity:
        "q_fallback": [0.71, 0.70],
    }
    base_routes = {
        "q_agree": ("computer", 0.90),
        "q_override": ("science", 0.86),
        "q_fallback": ("science", 0.80),
    }

    router = FakeRouter(vectors, base_routes)
    rows = [
        LabeledSentence("computer", "computer-1"),
        LabeledSentence("computer", "computer-2"),
        LabeledSentence("science", "science-1"),
        LabeledSentence("science", "science-2"),
    ]

    memory = encode_memory(router, rows)
    prototypes = build_multi_prototypes(memory, per_label=2)

    check("runtime:prototype-labels", set(prototypes) == {"computer", "science"})
    check(
        "runtime:prototype-count",
        all(len(value) == 2 for value in prototypes.values()),
    )

    agree = decide_adaptive_route(
        router,
        "q_agree",
        memory,
        prototypes,
        base_similarity_threshold=0.80,
        override_similarity_threshold=0.92,
        local_k=2,
        local_purity_threshold=1.0,
    )
    check("runtime:agree-not-none", agree is not None)
    check("runtime:agree-action", agree.action == "ACCEPT", agree.action)
    check("runtime:agree-label", agree.label == "computer", agree.label)

    override = decide_adaptive_route(
        router,
        "q_override",
        memory,
        prototypes,
        base_similarity_threshold=0.80,
        override_similarity_threshold=0.92,
        local_k=2,
        local_purity_threshold=1.0,
    )
    check("runtime:override-not-none", override is not None)
    check(
        "runtime:override-action",
        override.action == "ADAPTIVE_OVERRIDE",
        override.action,
    )
    check("runtime:override-label", override.label == "computer", override.label)
    check(
        "runtime:override-purity",
        abs(override.local_purity - 1.0) < 1e-9,
        str(override.local_purity),
    )

    fallback = decide_adaptive_route(
        router,
        "q_fallback",
        memory,
        prototypes,
        base_similarity_threshold=0.90,
        override_similarity_threshold=0.98,
        local_k=2,
        local_purity_threshold=1.0,
    )
    check("runtime:fallback-not-none", fallback is not None)
    check(
        "runtime:fallback-safe",
        fallback.action in {"BASE_FALLBACK", "GATE_REVIEW"},
        fallback.action,
    )


def required_files():
    names = [
        "semantic.py",
        "semantic_runtime_v2.py",
        "semantic_intent_v034.py",
        "semantic_relations_v035.py",
        "semantic_proposition_v036.py",
        "adaptive_semantic_learning.py",
        "adaptive_semantic_runtime.py",
        "run_role_synergy_v045.py",
    ]
    for name in names:
        check(f"file:{name}", (ROOT / name).is_file())


def main():
    print("=" * 78)
    print(" LLM_SEM v0.6.1 Unified Semantic Runtime Regression")
    print("=" * 78)

    required_files()
    source_checks()
    adaptive_runtime_checks()

    print("-" * 78)
    print("RESULT: PASS")
    print(
        "v0.6.1 integration contract is consistent: Semantic Data v2 + "
        "relations/propositions + local-evidence adaptive runtime."
    )


if __name__ == "__main__":
    main()
