# run_adaptive_semantic_composition_v063.py
#
# LLM_SEM v0.6.3 Adaptive Semantic Composition Gate
#
# Exploratory follow-up to v0.6.2.
#
# v0.6.2 showed:
#   - balanced role-aware composition was strongest overall,
#   - predicate/relation emphasis helped the unseen-relation / seen-concept split,
#   - learned composition improved TRAIN but did not beat frozen role-aware
#     composition on holdout generalization.
#
# v0.6.3 therefore tests a constrained, interpretable composition gate.
# The gate uses ONLY novelty relative to TRAIN_CASES:
#
#   known relation
#       -> balanced
#
#   unseen relation + both concepts seen
#       -> relation_aware
#
#   unseen relation + at least one unseen concept
#       -> balanced
#
# This rule is deliberately small and non-neural. It does not inspect holdout
# labels, margins, positive vectors, or counterfactual scores at decision time.
#
# IMPORTANT:
#   This is an exploratory/post-hoc experiment motivated by v0.6.2 results.
#   It should not be treated as an independent confirmatory benchmark.
#
# Run:
#   python run_adaptive_semantic_composition_v063.py

from __future__ import annotations

from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F

from model import LanguageModel
from tokenizer import Tokenizer
from run_relation_unseen_generalization_v043 import (
    MODEL,
    TOKENIZER,
    TRAIN_CASES,
    HOLDOUT_SEEN_REL,
    HOLDOUT_UNSEEN_REL_SEEN_CONCEPTS,
    HOLDOUT_UNSEEN_REL_UNSEEN_CONCEPTS,
    StructuralRoleProjection,
    prepare,
    summarize,
    validate_cases,
)


ROLE_CHECKPOINT = "model/structural-role-projection-v043.pt"
OUT = "results/adaptive_semantic_composition_v063.pt"

BALANCED = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
RELATION_AWARE = (0.25, 0.50, 0.25)
CONCEPT_AWARE = (0.40, 0.20, 0.40)

METHODS = (
    "simple_mean",
    "balanced",
    "relation_aware",
    "concept_aware",
    "adaptive_gate",
)


def training_vocabulary():
    relations = {case.predicate for case in TRAIN_CASES}
    concepts = set()
    for case in TRAIN_CASES:
        concepts.add(case.subject)
        concepts.add(case.object)
    return relations, concepts


TRAIN_RELATIONS, TRAIN_CONCEPTS = training_vocabulary()


def projected_roles(role_model, item):
    return (
        role_model.subject(item["s"]),
        role_model.predicate(item["p"]),
        role_model.object(item["o"]),
    )


def weighted_compose(role_model, item, weights):
    ws, wp, wo = weights
    rs, rp, ro = projected_roles(role_model, item)
    return F.normalize(ws * rs + wp * rp + wo * ro, dim=0)


def simple_mean(item):
    return F.normalize(
        torch.stack(
            [
                F.normalize(item["s"], dim=0),
                F.normalize(item["p"], dim=0),
                F.normalize(item["o"], dim=0),
            ]
        ).mean(dim=0),
        dim=0,
    )


def novelty_state(case):
    relation_seen = case.predicate in TRAIN_RELATIONS
    subject_seen = case.subject in TRAIN_CONCEPTS
    object_seen = case.object in TRAIN_CONCEPTS

    if relation_seen:
        mode = "balanced"
        reason = "known_relation"
        weights = BALANCED
    elif subject_seen and object_seen:
        mode = "relation_aware"
        reason = "unseen_relation_seen_concepts"
        weights = RELATION_AWARE
    else:
        mode = "balanced"
        reason = "unseen_relation_with_novel_concept"
        weights = BALANCED

    return {
        "relation_seen": relation_seen,
        "subject_seen": subject_seen,
        "object_seen": object_seen,
        "mode": mode,
        "reason": reason,
        "weights": weights,
    }


def evaluate_vector(composed, item):
    pos = float(F.cosine_similarity(composed, item["positive"], dim=0).item())
    cfs = [
        float(F.cosine_similarity(composed, cf, dim=0).item())
        for cf in item["counterfactuals"]
    ]
    best_cf = max(cfs)
    margin = pos - best_cf
    preservation = float(
        F.cosine_similarity(composed, item["raw"], dim=0).item()
    )
    return (
        item["case"].name,
        pos,
        cfs[0],
        cfs[1],
        cfs[2],
        best_cf,
        margin,
        preservation,
    )


def evaluate_method(method, role_model, prepared):
    rows = []
    decisions = []

    for item in prepared:
        if method == "simple_mean":
            composed = simple_mean(item)
            decision = None
        elif method == "balanced":
            composed = weighted_compose(role_model, item, BALANCED)
            decision = None
        elif method == "relation_aware":
            composed = weighted_compose(role_model, item, RELATION_AWARE)
            decision = None
        elif method == "concept_aware":
            composed = weighted_compose(role_model, item, CONCEPT_AWARE)
            decision = None
        elif method == "adaptive_gate":
            decision = novelty_state(item["case"])
            composed = weighted_compose(
                role_model,
                item,
                decision["weights"],
            )
        else:
            raise ValueError(method)

        rows.append(evaluate_vector(composed, item))
        decisions.append(decision)

    return rows, decisions


def print_method_table(title, results, count):
    print(title)
    print("-" * 116)
    print(
        f"{'Method':<20} {'Positive':>10} {'MeanMargin':>12} "
        f"{'MinMargin':>11} {'Preserve':>10}"
    )
    print("-" * 116)

    for method in METHODS:
        s = summarize(results[method]["rows"])
        print(
            f"{method:<20} "
            f"{s['positive_cases']:>3}/{count:<6} "
            f"{s['mean_margin']:+12.6f} "
            f"{s['min_margin']:+11.6f} "
            f"{s['mean_preservation']:10.6f}"
        )
    print()


def print_gate_decisions(label, prepared, decisions):
    print(label)
    print("-" * 116)
    print(
        f"{'Case':<24} {'Relation':<18} {'S-seen':>7} {'O-seen':>7} "
        f"{'Mode':<16} {'Reason'}"
    )
    print("-" * 116)

    for item, d in zip(prepared, decisions):
        if d is None:
            continue
        case = item["case"]
        print(
            f"{case.name:<24} "
            f"{case.predicate:<18} "
            f"{str(d['subject_seen']):>7} "
            f"{str(d['object_seen']):>7} "
            f"{d['mode']:<16} "
            f"{d['reason']}"
        )
    print()


def aggregate_holdout(results):
    total = (
        len(HOLDOUT_SEEN_REL)
        + len(HOLDOUT_UNSEEN_REL_SEEN_CONCEPTS)
        + len(HOLDOUT_UNSEEN_REL_UNSEEN_CONCEPTS)
    )

    aggregate = {}
    for method in METHODS:
        summaries = [
            summarize(results[split][method]["rows"])
            for split in ("A", "B", "C")
        ]
        aggregate[method] = {
            "positive_cases": sum(x["positive_cases"] for x in summaries),
            "total_cases": total,
            "mean_margin": sum(x["mean_margin"] for x in summaries) / 3.0,
            "mean_preservation": (
                sum(x["mean_preservation"] for x in summaries) / 3.0
            ),
        }
    return aggregate


def main():
    for path in (MODEL, TOKENIZER, ROLE_CHECKPOINT):
        if not Path(path).exists():
            raise FileNotFoundError(path)

    validate_cases("TRAIN", TRAIN_CASES)
    validate_cases("A", HOLDOUT_SEEN_REL)
    validate_cases("B", HOLDOUT_UNSEEN_REL_SEEN_CONCEPTS)
    validate_cases("C", HOLDOUT_UNSEEN_REL_UNSEEN_CONCEPTS)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer.load(TOKENIZER)
    base_model, base_checkpoint = LanguageModel.load_checkpoint(
        MODEL,
        device=device,
    )
    base_model.eval()

    checkpoint = torch.load(
        ROLE_CHECKPOINT,
        map_location=device,
        weights_only=False,
    )
    dim = int(checkpoint["dimension"])
    role_seed = int(checkpoint["best_seed"])

    role_model = StructuralRoleProjection(dim, seed=role_seed).to(device)
    role_model.load_state_dict(checkpoint["state_dict"])
    role_model.eval()

    train = prepare(base_model, tokenizer, device, TRAIN_CASES)
    split_a = prepare(
        base_model,
        tokenizer,
        device,
        HOLDOUT_SEEN_REL,
    )
    split_b = prepare(
        base_model,
        tokenizer,
        device,
        HOLDOUT_UNSEEN_REL_SEEN_CONCEPTS,
    )
    split_c = prepare(
        base_model,
        tokenizer,
        device,
        HOLDOUT_UNSEEN_REL_UNSEEN_CONCEPTS,
    )

    splits = {
        "TRAIN": train,
        "A": split_a,
        "B": split_b,
        "C": split_c,
    }

    print("=" * 116)
    print(" LLM_SEM v0.6.3 Adaptive Semantic Composition Gate")
    print("=" * 116)
    print("Device               :", device)
    if device.type == "cuda":
        print("GPU                  :", torch.cuda.get_device_name(0))
    print("Base checkpoint loss :", base_checkpoint.get("loss"))
    print("Role checkpoint      :", ROLE_CHECKPOINT)
    print("Role projection seed :", role_seed)
    print("Vector dimension     :", dim)
    print("Train relations      :", sorted(TRAIN_RELATIONS))
    print("Train concepts       :", len(TRAIN_CONCEPTS))
    print("Gate training        : False")
    print("Holdout optimization : False")
    print("Experiment type      : exploratory / post-hoc")
    print()

    print("Composition modes")
    print("-" * 116)
    print(f"balanced       : S={BALANCED[0]:.3f} P={BALANCED[1]:.3f} O={BALANCED[2]:.3f}")
    print(
        f"relation_aware : S={RELATION_AWARE[0]:.3f} "
        f"P={RELATION_AWARE[1]:.3f} O={RELATION_AWARE[2]:.3f}"
    )
    print(
        f"concept_aware  : S={CONCEPT_AWARE[0]:.3f} "
        f"P={CONCEPT_AWARE[1]:.3f} O={CONCEPT_AWARE[2]:.3f}"
    )
    print()

    results = {}
    for split_name, prepared in splits.items():
        results[split_name] = {}
        for method in METHODS:
            rows, decisions = evaluate_method(
                method,
                role_model,
                prepared,
            )
            results[split_name][method] = {
                "rows": rows,
                "decisions": decisions,
            }

    print_method_table(
        "TRAIN: seen relations / training propositions",
        results["TRAIN"],
        len(train),
    )
    print_method_table(
        "A) Seen relation / unseen proposition",
        results["A"],
        len(split_a),
    )
    print_method_table(
        "B) Unseen relation / seen concepts",
        results["B"],
        len(split_b),
    )
    print_method_table(
        "C) Unseen relation / unseen concepts",
        results["C"],
        len(split_c),
    )

    print_gate_decisions(
        "Adaptive gate decisions: A",
        split_a,
        results["A"]["adaptive_gate"]["decisions"],
    )
    print_gate_decisions(
        "Adaptive gate decisions: B",
        split_b,
        results["B"]["adaptive_gate"]["decisions"],
    )
    print_gate_decisions(
        "Adaptive gate decisions: C",
        split_c,
        results["C"]["adaptive_gate"]["decisions"],
    )

    aggregate = aggregate_holdout(results)

    print("Cross-split generalization")
    print("-" * 116)
    print(
        f"{'Method':<20} {'Positive':>10} {'MeanMargin':>12} "
        f"{'Preserve':>10}"
    )
    print("-" * 116)
    for method in METHODS:
        a = aggregate[method]
        print(
            f"{method:<20} "
            f"{a['positive_cases']:>3}/{a['total_cases']:<6} "
            f"{a['mean_margin']:+12.6f} "
            f"{a['mean_preservation']:10.6f}"
        )
    print()

    mode_counts = Counter()
    for split in ("A", "B", "C"):
        for d in results[split]["adaptive_gate"]["decisions"]:
            mode_counts[d["mode"]] += 1

    print("Adaptive mode usage")
    print("-" * 116)
    for mode in ("balanced", "relation_aware", "concept_aware"):
        print(f"{mode:<20}: {mode_counts.get(mode, 0)}")
    print()

    print("Interpretation")
    print("-" * 116)
    print(
        "The adaptive gate is constrained by semantic novelty relative to TRAIN "
        "and cannot inspect evaluation margins or counterfactual outcomes."
    )
    print(
        "Improvement over balanced composition would support the hypothesis "
        "that composition policy should depend on semantic novelty rather than "
        "use one fixed role weighting for every proposition."
    )
    print(
        "Because this rule was motivated by v0.6.2, any gain is exploratory. "
        "A new independently designed holdout set is required for confirmation."
    )

    output = {
        "version": "v0.6.3",
        "experiment_type": "exploratory_posthoc",
        "base_checkpoint_loss": base_checkpoint.get("loss"),
        "role_checkpoint": ROLE_CHECKPOINT,
        "role_seed": role_seed,
        "train_relations": sorted(TRAIN_RELATIONS),
        "train_concepts": sorted(TRAIN_CONCEPTS),
        "weights": {
            "balanced": BALANCED,
            "relation_aware": RELATION_AWARE,
            "concept_aware": CONCEPT_AWARE,
        },
        "gate_rule": {
            "known_relation": "balanced",
            "unseen_relation_seen_concepts": "relation_aware",
            "unseen_relation_with_novel_concept": "balanced",
        },
        "train": {
            method: summarize(results["TRAIN"][method]["rows"])
            for method in METHODS
        },
        "split_A": {
            method: summarize(results["A"][method]["rows"])
            for method in METHODS
        },
        "split_B": {
            method: summarize(results["B"][method]["rows"])
            for method in METHODS
        },
        "split_C": {
            method: summarize(results["C"][method]["rows"])
            for method in METHODS
        },
        "aggregate_holdout": aggregate,
        "gate_decisions": {
            split: results[split]["adaptive_gate"]["decisions"]
            for split in ("A", "B", "C")
        },
        "mode_counts": dict(mode_counts),
    }

    out_path = Path(OUT)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, out_path)
    print()
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
