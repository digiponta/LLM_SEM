# run_role_synergy_v045.py
#
# LLM_SEM v0.4.5 Role Interaction / Synergy Analysis
#
# Compare all non-empty Subject / Predicate / Object role combinations using
# the fixed v0.4.3 trained projection. No retraining.
#
# Variants:
#   subject_only
#   predicate_only
#   object_only
#   subject_predicate
#   subject_object
#   predicate_object
#   full
#
# Synergy is measured as:
#   pair_synergy(A,B) = margin(A+B) - max(margin(A), margin(B))
#
# Triple synergy is measured as:
#   triple_synergy = margin(full) - max(pair margins)
#
# Positive synergy means combining roles yields more structural margin than the
# best constituent configuration.

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from model import LanguageModel
from tokenizer import Tokenizer

from run_relation_unseen_generalization_v043 import (
    MODEL,
    TOKENIZER,
    HOLDOUT_SEEN_REL,
    HOLDOUT_UNSEEN_REL_SEEN_CONCEPTS,
    HOLDOUT_UNSEEN_REL_UNSEEN_CONCEPTS,
    StructuralRoleProjection,
    prepare,
    summarize,
    validate_cases,
)


CHECKPOINT = "model/structural-role-projection-v043.pt"

VARIANTS = {
    "subject_only": (True, False, False),
    "predicate_only": (False, True, False),
    "object_only": (False, False, True),
    "subject_predicate": (True, True, False),
    "subject_object": (True, False, True),
    "predicate_object": (False, True, True),
    "full": (True, True, True),
}


def compose_variant(
    role_model,
    subject_v,
    predicate_v,
    object_v,
    mask,
):
    use_subject, use_predicate, use_object = mask
    parts = []

    if use_subject:
        parts.append(role_model.subject(subject_v))
    if use_predicate:
        parts.append(role_model.predicate(predicate_v))
    if use_object:
        parts.append(role_model.object(object_v))

    if not parts:
        raise ValueError("empty role composition is not allowed")

    return F.normalize(torch.stack(parts).mean(dim=0), dim=0)


def evaluate_variant(role_model, prepared, mask):
    rows = []

    for item in prepared:
        composed = compose_variant(
            role_model,
            item["s"],
            item["p"],
            item["o"],
            mask,
        )

        positive = float(
            F.cosine_similarity(composed, item["positive"], dim=0).item()
        )
        cfs = [
            float(F.cosine_similarity(composed, cf, dim=0).item())
            for cf in item["counterfactuals"]
        ]
        best_cf = max(cfs)
        margin = positive - best_cf
        preservation = float(
            F.cosine_similarity(composed, item["raw"], dim=0).item()
        )

        rows.append(
            (
                item["case"].name,
                positive,
                cfs[0],
                cfs[1],
                cfs[2],
                best_cf,
                margin,
                preservation,
            )
        )

    return rows


def evaluate_all(role_model, prepared):
    return {
        name: {
            "rows": evaluate_variant(role_model, prepared, mask),
        }
        for name, mask in VARIANTS.items()
    }


def attach_summaries(results):
    for value in results.values():
        value["summary"] = summarize(value["rows"])
    return results


def compute_synergy(results):
    m = {
        name: value["summary"]["mean_margin"]
        for name, value in results.items()
    }

    pair_synergy = {
        "subject_predicate": (
            m["subject_predicate"]
            - max(m["subject_only"], m["predicate_only"])
        ),
        "subject_object": (
            m["subject_object"]
            - max(m["subject_only"], m["object_only"])
        ),
        "predicate_object": (
            m["predicate_object"]
            - max(m["predicate_only"], m["object_only"])
        ),
    }

    triple_synergy = (
        m["full"]
        - max(
            m["subject_predicate"],
            m["subject_object"],
            m["predicate_object"],
        )
    )

    return pair_synergy, triple_synergy


def print_variant_table(label, results, count):
    print(label)
    print("-" * 122)
    print(
        f"{'Variant':<20} {'Positive':>10} {'MeanMargin':>12} "
        f"{'MinMargin':>11} {'Preserve':>10}"
    )
    print("-" * 122)

    for name in VARIANTS:
        summary = results[name]["summary"]
        print(
            f"{name:<20} "
            f"{summary['positive_cases']:>3}/{count:<6} "
            f"{summary['mean_margin']:+12.6f} "
            f"{summary['min_margin']:+11.6f} "
            f"{summary['mean_preservation']:10.6f}"
        )
    print()


def print_synergy(label, pair_synergy, triple_synergy):
    print(label)
    print("-" * 122)
    print(
        f"Subject + Predicate synergy : "
        f"{pair_synergy['subject_predicate']:+.6f}"
    )
    print(
        f"Subject + Object synergy    : "
        f"{pair_synergy['subject_object']:+.6f}"
    )
    print(
        f"Predicate + Object synergy  : "
        f"{pair_synergy['predicate_object']:+.6f}"
    )
    print(f"Triple synergy              : {triple_synergy:+.6f}")
    print()


def main():
    if not Path(MODEL).exists():
        raise FileNotFoundError(MODEL)
    if not Path(TOKENIZER).exists():
        raise FileNotFoundError(TOKENIZER)
    if not Path(CHECKPOINT).exists():
        raise FileNotFoundError(
            f"{CHECKPOINT} not found. Run "
            "run_relation_unseen_generalization_v043.py first."
        )

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
        CHECKPOINT,
        map_location=device,
        weights_only=False,
    )

    dim = int(checkpoint["dimension"])
    seed = int(checkpoint["best_seed"])

    role_model = StructuralRoleProjection(dim, seed=seed).to(device)
    role_model.load_state_dict(checkpoint["state_dict"])
    role_model.eval()

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

    print("=" * 122)
    print(" LLM_SEM v0.4.5 Role Interaction / Synergy Analysis")
    print("=" * 122)
    print("Device               :", device)
    if device.type == "cuda":
        print("GPU                  :", torch.cuda.get_device_name(0))
    print("Base checkpoint loss :", base_checkpoint.get("loss"))
    print("Projection checkpoint:", CHECKPOINT)
    print("Projection seed      :", seed)
    print("Vector dimension     :", dim)
    print("Retraining           : False")
    print()

    results = {}
    synergies = {}

    for split_name, prepared in (
        ("A", split_a),
        ("B", split_b),
        ("C", split_c),
    ):
        split_results = attach_summaries(
            evaluate_all(role_model, prepared)
        )
        pair_synergy, triple_synergy = compute_synergy(split_results)

        results[split_name] = split_results
        synergies[split_name] = {
            "pair": pair_synergy,
            "triple": triple_synergy,
        }

        title = {
            "A": "A) Seen relation / unseen proposition",
            "B": "B) Unseen relation / seen concepts",
            "C": "C) Unseen relation / unseen concepts",
        }[split_name]

        print_variant_table(
            title,
            split_results,
            len(prepared),
        )
        print_synergy(
            f"{title} synergy",
            pair_synergy,
            triple_synergy,
        )

    print("Cross-split mean synergy")
    print("-" * 122)

    pair_names = (
        "subject_predicate",
        "subject_object",
        "predicate_object",
    )

    aggregate_pair = {}
    for pair in pair_names:
        value = sum(
            synergies[split]["pair"][pair]
            for split in ("A", "B", "C")
        ) / 3.0
        aggregate_pair[pair] = value
        print(f"{pair:<24}: {value:+.6f}")

    aggregate_triple = sum(
        synergies[split]["triple"]
        for split in ("A", "B", "C")
    ) / 3.0
    print(f"{'triple':<24}: {aggregate_triple:+.6f}")
    print()

    print("Interpretation")
    print("-" * 122)
    print(
        "Positive pair synergy means a role pair produces more mean structural "
        "margin than either role alone."
    )
    print(
        "Positive triple synergy means Full exceeds the strongest two-role "
        "configuration."
    )
    print(
        "Negative synergy means the additional role does not add useful "
        "structural information under that split and may introduce interference."
    )

    output = {
        "version": "v0.4.5",
        "source_checkpoint": CHECKPOINT,
        "best_seed": seed,
        "split_A": {
            name: value["summary"]
            for name, value in results["A"].items()
        },
        "split_B": {
            name: value["summary"]
            for name, value in results["B"].items()
        },
        "split_C": {
            name: value["summary"]
            for name, value in results["C"].items()
        },
        "synergy_A": synergies["A"],
        "synergy_B": synergies["B"],
        "synergy_C": synergies["C"],
        "cross_split_pair_synergy": aggregate_pair,
        "cross_split_triple_synergy": aggregate_triple,
    }

    out_path = Path("results/role_synergy_v045.pt")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, out_path)
    print()
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
