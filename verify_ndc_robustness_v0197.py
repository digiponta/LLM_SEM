#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LLM_SEM v0.19.7 Cross-Holdout Robustness Sweep.

Purpose:
    Measure robustness across multiple previously-created independent holdouts
    instead of adding more local rules to chase a single evaluation set.

Routers:
    A) v0.19.2 Coverage + Expanded UNKNOWN
    B) v0.19.4 Conditional / Pairwise
    C) v0.19.5 + local 830 stabilization
    D) v0.19.6 + local 930 stabilization

Holdouts:
    V3, V4, V5, V6, V7

Important:
    This is a retrospective robustness evaluation. Because later router
    variants were designed after observing earlier holdout failures, this
    script must NOT be interpreted as a fully independent final benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple

import torch

from model import LanguageModel
from tokenizer import Tokenizer
from ndc_hierarchy_unknown_v0192 import CoverageUnknownNDCRouter
from ndc_hierarchy_adjudication_v0194 import ConditionalAdjudicationNDCRouter
from ndc_hierarchy_final_v0195 import FinalStableNDCRouter
from ndc_hierarchy_final_v0196 import FinalStableNDCRouterV0196

from verify_ndc_unknown_v0192 import HOLDOUT_V3, UNKNOWN_V3
from verify_ndc_boundary_v0193 import HOLDOUT_V4, UNKNOWN_V4
from verify_ndc_adjudication_v0194 import HOLDOUT_V5, UNKNOWN_V5
from verify_ndc_final_v0195 import HOLDOUT_V6, UNKNOWN_V6
from verify_ndc_final_v0196 import HOLDOUT_V7, UNKNOWN_V7


@dataclass(frozen=True)
class Metrics:
    raw: float
    accepted_acc: float
    coverage: float
    unknown_reject: float
    balanced: float


SETS = (
    ("V3", HOLDOUT_V3, UNKNOWN_V3),
    ("V4", HOLDOUT_V4, UNKNOWN_V4),
    ("V5", HOLDOUT_V5, UNKNOWN_V5),
    ("V6", HOLDOUT_V6, UNKNOWN_V6),
    ("V7", HOLDOUT_V7, UNKNOWN_V7),
)


def evaluate(router, known, unknown) -> Metrics:
    correct = 0
    accepted_correct = 0
    accepted = 0
    rejected_unknown = 0

    for expected, text in known:
        r = router.route(text)
        is_correct = r.ndc_code == expected
        is_accepted = r.state == "ACCEPT"
        correct += int(is_correct)
        accepted_correct += int(is_correct and is_accepted)
        accepted += int(is_accepted)

    for text in unknown:
        r = router.route(text)
        rejected_unknown += int(r.state == "UNKNOWN")

    raw = correct / len(known)
    accepted_acc = accepted_correct / len(known)
    coverage = accepted / len(known)
    unknown_reject = rejected_unknown / len(unknown)
    balanced = 0.5 * accepted_acc + 0.5 * unknown_reject

    return Metrics(
        raw=raw,
        accepted_acc=accepted_acc,
        coverage=coverage,
        unknown_reject=unknown_reject,
        balanced=balanced,
    )


def mean(values):
    return sum(values) / len(values)


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer.load("model/tokenizer.json")
    model, checkpoint = LanguageModel.load_checkpoint(
        "model/model-sem-internalized-v01575.pt",
        device=device,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    routers = (
        ("v0.19.2", CoverageUnknownNDCRouter(model, tokenizer)),
        ("v0.19.4", ConditionalAdjudicationNDCRouter(model, tokenizer)),
        ("v0.19.5", FinalStableNDCRouter(model, tokenizer)),
        ("v0.19.6", FinalStableNDCRouterV0196(model, tokenizer)),
    )

    print("=" * 132)
    print(" LLM_SEM v0.19.7 Cross-Holdout Robustness Sweep")
    print("=" * 132)
    print("Device          :", device)
    if device.type == "cuda":
        print("GPU             :", torch.cuda.get_device_name(device))
    print("Checkpoint loss :", checkpoint.get("loss"))
    print("Holdouts        : V3, V4, V5, V6, V7")
    print("Interpretation  : retrospective robustness, not fully independent final benchmark")
    print()

    overall = {}

    for name, router in routers:
        rows = []

        print("=" * 132)
        print(name)
        print("=" * 132)
        print(
            f"{'Set':<6} {'Raw':>8} {'Accepted':>10} {'Coverage':>10} "
            f"{'Unknown':>10} {'Balanced':>10}"
        )
        print("-" * 132)

        for set_name, known, unknown in SETS:
            m = evaluate(router, known, unknown)
            rows.append(m)
            print(
                f"{set_name:<6} "
                f"{m.raw*100:7.2f}% "
                f"{m.accepted_acc*100:9.2f}% "
                f"{m.coverage*100:9.2f}% "
                f"{m.unknown_reject*100:9.2f}% "
                f"{m.balanced*100:9.2f}%"
            )

        aggregate = Metrics(
            raw=mean([m.raw for m in rows]),
            accepted_acc=mean([m.accepted_acc for m in rows]),
            coverage=mean([m.coverage for m in rows]),
            unknown_reject=mean([m.unknown_reject for m in rows]),
            balanced=mean([m.balanced for m in rows]),
        )
        min_known = min(m.accepted_acc for m in rows)
        min_unknown = min(m.unknown_reject for m in rows)
        min_balanced = min(m.balanced for m in rows)

        overall[name] = (aggregate, min_known, min_unknown, min_balanced)

        print("-" * 132)
        print(
            f"{'MEAN':<6} "
            f"{aggregate.raw*100:7.2f}% "
            f"{aggregate.accepted_acc*100:9.2f}% "
            f"{aggregate.coverage*100:9.2f}% "
            f"{aggregate.unknown_reject*100:9.2f}% "
            f"{aggregate.balanced*100:9.2f}%"
        )
        print(
            f"MIN accepted={min_known*100:.2f}% "
            f"MIN unknown={min_unknown*100:.2f}% "
            f"MIN balanced={min_balanced*100:.2f}%"
        )
        print()

    ranked = sorted(
        overall.items(),
        key=lambda kv: (
            kv[1][3],             # min balanced first
            kv[1][0].balanced,    # then mean balanced
            kv[1][1],             # then min known
        ),
        reverse=True,
    )

    print("=" * 132)
    print("ROBUSTNESS RANKING")
    print("=" * 132)
    for index, (name, (m, min_known, min_unknown, min_balanced)) in enumerate(ranked, 1):
        print(
            f"{index}. {name:<8} "
            f"mean_balanced={m.balanced*100:6.2f}% "
            f"min_balanced={min_balanced*100:6.2f}% "
            f"mean_known={m.accepted_acc*100:6.2f}% "
            f"mean_unknown={m.unknown_reject*100:6.2f}%"
        )

    best_name, (best, min_known, min_unknown, min_balanced) = ranked[0]

    robust_candidate = (
        best.accepted_acc >= 0.90
        and best.unknown_reject >= 0.90
        and min_known >= 0.85
        and min_unknown >= 0.85
        and min_balanced >= 0.87
    )

    print()
    print("=" * 132)
    print("SUMMARY")
    print("=" * 132)
    print("Best router       :", best_name)
    print(f"Mean known        : {best.accepted_acc*100:.2f}%")
    print(f"Mean unknown      : {best.unknown_reject*100:.2f}%")
    print(f"Mean balanced     : {best.balanced*100:.2f}%")
    print(f"Minimum known     : {min_known*100:.2f}%")
    print(f"Minimum unknown   : {min_unknown*100:.2f}%")
    print(f"Minimum balanced  : {min_balanced*100:.2f}%")
    print(
        "RESULT            :",
        "ROBUST_CANDIDATE" if robust_candidate else "NOT_YET_ROBUST",
    )
    print("=" * 132)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
