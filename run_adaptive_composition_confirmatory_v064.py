# run_adaptive_composition_confirmatory_v064.py
#
# LLM_SEM v0.6.4 Confirmatory Adaptive Semantic Composition
#
# Purpose:
#   Confirm the v0.6.3 adaptive composition policy on NEW holdout cases that
#   were not used to define the rule.
#
# Frozen v0.6.3 policy:
#
#   known relation
#       -> balanced
#
#   unseen relation + subject/object both seen in TRAIN
#       -> relation_aware
#
#   otherwise
#       -> balanced
#
# No gate training, no weight tuning, no threshold tuning, no seed selection,
# and no use of confirmatory margins to alter the policy.
#
# Confirmatory splits:
#   D) known relation / new concepts
#   E) unseen relation / seen concepts
#   F) unseen relation / partially seen concepts
#   G) unseen relation / new concepts
#
# Run:
#   python run_adaptive_composition_confirmatory_v064.py

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from model import LanguageModel
from tokenizer import Tokenizer
from run_relation_unseen_generalization_v043 import (
    MODEL,
    TOKENIZER,
    TRAIN_CASES,
    StructuralRoleProjection,
    Case,
    prepare,
    summarize,
    validate_cases,
)


ROLE_CHECKPOINT = "model/structural-role-projection-v043.pt"
OUT = "results/adaptive_composition_confirmatory_v064.pt"

BALANCED = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
RELATION_AWARE = (0.25, 0.50, 0.25)

METHODS = (
    "balanced",
    "relation_aware",
    "adaptive_gate",
)


# D: relation seen in TRAIN, but concepts are new.
CONFIRM_D = [
    Case("disk-fast", "ディスク", "has_property", "高速転送", "メモリ", "targets", "低速転送"),
    Case("parser-code", "パーサ", "targets", "コード", "GPU", "related_with", "画像"),
    Case("switch-network", "スイッチ", "related_with", "ネットワーク機器", "CPU", "has_property", "低速"),
]

# E: unseen relation, both subject/object concepts were seen in TRAIN.
CONFIRM_E = [
    Case("cpu-is-computer", "CPU", "is_a", "computer", "GPU", "targets", "weather"),
    Case("gpu-used-data", "GPU", "used_for", "data", "Python", "related_with", "image"),
    Case("python-part-computer", "Python", "part_of", "computer", "CUDA", "targets", "weather"),
]

# F: unseen relation, one side seen and one side novel.
CONFIRM_F = [
    Case("gpu-causes-heat", "GPU", "causes", "発熱", "CPU", "has_property", "低温"),
    Case("python-used-analysis", "Python", "used_for", "解析", "CUDA", "targets", "画像"),
    Case("cuda-part-system", "CUDA", "part_of", "計算基盤", "GPU", "related_with", "weather"),
]

# G: unseen relation, both subject/object concepts novel.
CONFIRM_G = [
    Case("ssd-is-storage", "SSD", "is_a", "記憶装置", "HDD", "has_property", "演算装置"),
    Case("compiler-causes-binary", "コンパイラ", "causes", "バイナリ生成", "リンカ", "targets", "画像"),
    Case("sensor-used-measurement", "温度センサー", "used_for", "温度測定", "カメラ", "related_with", "画像"),
    Case("cache-hierarchy", "L2キャッシュ", "part_of", "メモリ階層", "SSD", "targets", "ストレージ"),
]


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


def gate_decision(case):
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
        reason = "otherwise_balanced"
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


def evaluate(method, role_model, prepared):
    rows = []
    decisions = []

    for item in prepared:
        if method == "balanced":
            weights = BALANCED
            decision = None
        elif method == "relation_aware":
            weights = RELATION_AWARE
            decision = None
        elif method == "adaptive_gate":
            decision = gate_decision(item["case"])
            weights = decision["weights"]
        else:
            raise ValueError(method)

        composed = weighted_compose(role_model, item, weights)
        rows.append(evaluate_vector(composed, item))
        decisions.append(decision)

    return rows, decisions


def print_table(title, results, count):
    print(title)
    print("-" * 118)
    print(
        f"{'Method':<20} {'Positive':>10} {'MeanMargin':>12} "
        f"{'MinMargin':>11} {'Preserve':>10}"
    )
    print("-" * 118)

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


def print_gate(title, prepared, decisions):
    print(title)
    print("-" * 118)
    print(
        f"{'Case':<26} {'Relation':<18} {'S-seen':>7} {'O-seen':>7} "
        f"{'Mode':<16} {'Reason'}"
    )
    print("-" * 118)
    for item, decision in zip(prepared, decisions):
        case = item["case"]
        print(
            f"{case.name:<26} "
            f"{case.predicate:<18} "
            f"{str(decision['subject_seen']):>7} "
            f"{str(decision['object_seen']):>7} "
            f"{decision['mode']:<16} "
            f"{decision['reason']}"
        )
    print()


def main():
    for path in (MODEL, TOKENIZER, ROLE_CHECKPOINT):
        if not Path(path).exists():
            raise FileNotFoundError(path)

    for label, cases in (
        ("D", CONFIRM_D),
        ("E", CONFIRM_E),
        ("F", CONFIRM_F),
        ("G", CONFIRM_G),
    ):
        validate_cases(label, cases)

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

    prepared = {
        "D": prepare(base_model, tokenizer, device, CONFIRM_D),
        "E": prepare(base_model, tokenizer, device, CONFIRM_E),
        "F": prepare(base_model, tokenizer, device, CONFIRM_F),
        "G": prepare(base_model, tokenizer, device, CONFIRM_G),
    }

    print("=" * 118)
    print(" LLM_SEM v0.6.4 Confirmatory Adaptive Semantic Composition")
    print("=" * 118)
    print("Device               :", device)
    if device.type == "cuda":
        print("GPU                  :", torch.cuda.get_device_name(0))
    print("Base checkpoint loss :", base_checkpoint.get("loss"))
    print("Role checkpoint      :", ROLE_CHECKPOINT)
    print("Role projection seed :", role_seed)
    print("Vector dimension     :", dim)
    print("Gate policy          : FROZEN from v0.6.3")
    print("Gate training        : False")
    print("Weight tuning        : False")
    print("Threshold tuning     : False")
    print("Seed selection       : False")
    print("Experiment type      : confirmatory")
    print()

    results = {}

    for split, rows in prepared.items():
        results[split] = {}
        for method in METHODS:
            out_rows, decisions = evaluate(method, role_model, rows)
            results[split][method] = {
                "rows": out_rows,
                "decisions": decisions,
            }

    titles = {
        "D": "D) Known relation / new concepts",
        "E": "E) Unseen relation / seen concepts",
        "F": "F) Unseen relation / partially seen concepts",
        "G": "G) Unseen relation / new concepts",
    }

    for split in ("D", "E", "F", "G"):
        print_table(
            titles[split],
            results[split],
            len(prepared[split]),
        )
        print_gate(
            f"Adaptive gate decisions: {split}",
            prepared[split],
            results[split]["adaptive_gate"]["decisions"],
        )

    total = sum(len(prepared[x]) for x in ("D", "E", "F", "G"))

    print("Confirmatory aggregate")
    print("-" * 118)
    print(
        f"{'Method':<20} {'Positive':>10} {'MeanMargin':>12} "
        f"{'Preserve':>10}"
    )
    print("-" * 118)

    aggregate = {}
    for method in METHODS:
        summaries = [
            summarize(results[split][method]["rows"])
            for split in ("D", "E", "F", "G")
        ]
        aggregate[method] = {
            "positive_cases": sum(x["positive_cases"] for x in summaries),
            "total_cases": total,
            "mean_margin": sum(x["mean_margin"] for x in summaries) / 4.0,
            "mean_preservation": (
                sum(x["mean_preservation"] for x in summaries) / 4.0
            ),
        }
        a = aggregate[method]
        print(
            f"{method:<20} "
            f"{a['positive_cases']:>3}/{a['total_cases']:<6} "
            f"{a['mean_margin']:+12.6f} "
            f"{a['mean_preservation']:10.6f}"
        )
    print()

    balanced = aggregate["balanced"]
    adaptive = aggregate["adaptive_gate"]

    delta_margin = adaptive["mean_margin"] - balanced["mean_margin"]
    delta_positive = (
        adaptive["positive_cases"] - balanced["positive_cases"]
    )

    print("Confirmatory comparison")
    print("-" * 118)
    print(f"Positive-case delta : {delta_positive:+d}")
    print(f"Mean-margin delta   : {delta_margin:+.6f}")
    print(
        "Primary criterion    : adaptive_gate >= balanced on positive cases "
        "and mean margin"
    )
    confirmed = (
        adaptive["positive_cases"] >= balanced["positive_cases"]
        and adaptive["mean_margin"] >= balanced["mean_margin"]
    )
    print("Confirmed            :", confirmed)
    print()

    print("Interpretation")
    print("-" * 118)
    print(
        "This script does not alter the v0.6.3 gate from confirmatory results."
    )
    print(
        "If Confirmed=True, the existing adaptive composition policy has "
        "replicated on an independently introduced holdout set."
    )
    print(
        "If Confirmed=False, the v0.6.3 gain should remain classified as "
        "exploratory and the gate should not be promoted to the runtime."
    )

    output = {
        "version": "v0.6.4",
        "experiment_type": "confirmatory",
        "base_checkpoint_loss": base_checkpoint.get("loss"),
        "role_checkpoint": ROLE_CHECKPOINT,
        "role_seed": role_seed,
        "gate_policy": {
            "known_relation": "balanced",
            "unseen_relation_seen_concepts": "relation_aware",
            "otherwise": "balanced",
        },
        "weights": {
            "balanced": BALANCED,
            "relation_aware": RELATION_AWARE,
        },
        "split_D": {
            method: summarize(results["D"][method]["rows"])
            for method in METHODS
        },
        "split_E": {
            method: summarize(results["E"][method]["rows"])
            for method in METHODS
        },
        "split_F": {
            method: summarize(results["F"][method]["rows"])
            for method in METHODS
        },
        "split_G": {
            method: summarize(results["G"][method]["rows"])
            for method in METHODS
        },
        "aggregate": aggregate,
        "delta_positive": delta_positive,
        "delta_margin": delta_margin,
        "confirmed": confirmed,
        "gate_decisions": {
            split: results[split]["adaptive_gate"]["decisions"]
            for split in ("D", "E", "F", "G")
        },
    }

    out_path = Path(OUT)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, out_path)
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
