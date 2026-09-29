# run_semantic_memory_prototype_regression_v034.py
#
# Lightweight regression for prototype ranking/margin behavior.
# No LLM checkpoint is required.

from __future__ import annotations

from types import SimpleNamespace

import torch

from semantic_eval import LabeledSentence
from semantic_memory_prototype import (
    build_prototypes,
    prototype_margin,
    rank_prototypes,
)


class FakeRouter:
    def __init__(self):
        self.vectors = {
            "c1": torch.tensor([1.0, 0.0]),
            "c2": torch.tensor([0.8, 0.2]),
            "s1": torch.tensor([0.0, 1.0]),
            "s2": torch.tensor([0.2, 0.8]),
            "cq": torch.tensor([0.95, 0.05]),
            "sq": torch.tensor([0.05, 0.95]),
        }

    def _encode_tensor(self, text):
        return self.vectors[text]


def main() -> int:
    router = FakeRouter()
    rows = [
        LabeledSentence("computer", "c1"),
        LabeledSentence("computer", "c2"),
        LabeledSentence("science", "s1"),
        LabeledSentence("science", "s2"),
    ]

    prototypes = build_prototypes(router, rows)
    checks = []

    checks.append(("two prototypes", set(prototypes) == {"computer", "science"}))
    checks.append(("computer count", prototypes["computer"].count == 2))
    checks.append(("science count", prototypes["science"].count == 2))

    c_scores = rank_prototypes(router, "cq", prototypes)
    checks.append(("computer query top1", c_scores[0].label == "computer"))
    checks.append(("computer margin positive", prototype_margin(c_scores) > 0.0))

    s_scores = rank_prototypes(router, "sq", prototypes)
    checks.append(("science query top1", s_scores[0].label == "science"))
    checks.append(("science margin positive", prototype_margin(s_scores) > 0.0))

    failed = 0
    for name, ok in checks:
        if ok:
            print(f"[PASS] {name}")
        else:
            failed += 1
            print(f"[FAIL] {name}")

    print()
    print("=" * 60)
    print(
        f"LLM_SEM v0.3.4 Semantic Memory Prototype: "
        f"{len(checks) - failed} passed, {failed} failed"
    )
    print("=" * 60)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
