# run_role_ablation_v044.py
#
# LLM_SEM v0.4.4 Role Ablation / Causal Contribution Analysis
#
# Evaluate the trained v0.4.3 Structural Role Projection while removing one
# structural role at a time:
#
#   Full         = Ws(S) + Wp(P) + Wo(O)
#   No Subject   =         Wp(P) + Wo(O)
#   No Predicate = Ws(S)         + Wo(O)
#   No Object    = Ws(S) + Wp(P)
#
# No parameters are retrained during ablation. Therefore the performance drop
# measures the functional contribution of each learned role pathway.

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

ABLATIONS = {
    "full": (True, True, True),
    "no_subject": (False, True, True),
    "no_predicate": (True, False, True),
    "no_object": (True, True, False),
}


def compose_ablated(
    role_model,
    subject_v,
    predicate_v,
    object_v,
    use_subject: bool,
    use_predicate: bool,
    use_object: bool,
):
    parts = []

    if use_subject:
        parts.append(role_model.subject(subject_v))
    if use_predicate:
        parts.append(role_model.predicate(predicate_v))
    if use_object:
        parts.append(role_model.object(object_v))

    if not parts:
        raise ValueError("at least one role must remain active")

    return F.normalize(torch.stack(parts).mean(dim=0), dim=0)


def evaluate_ablation(role_model, prepared, mask):
    use_subject, use_predicate, use_object = mask
    rows = []

    for item in prepared:
        composed = compose_ablated(
            role_model,
            item["s"],
            item["p"],
            item["o"],
            use_subject,
            use_predicate,
            use_object,
        )

        positive = float(
            F.cosine_similarity(composed, item["positive"], dim=0).item()
        )
        counterfactuals = [
            float(F.cosine_similarity(composed, cf, dim=0).item())
            for cf in item["counterfactuals"]
        ]

        best_cf = max(counterfactuals)
        margin = positive - best_cf
        preservation = float(
            F.cosine_similarity(composed, item["raw"], dim=0).item()
        )

        rows.append(
            (
                item["case"].name,
                positive,
                counterfactuals[0],
                counterfactuals[1],
                counterfactuals[2],
                best_cf,
                margin,
                preservation,
            )
        )

    return rows


def report_split(name, role_model, prepared):
    results = {}
    for ablation_name, mask in ABLATIONS.items():
        rows = evaluate_ablation(role_model, prepared, mask)
        results[ablation_name] = {
            "rows": rows,
            "summary": summarize(rows),
        }

    full = results["full"]["summary"]

    print(name)
    print("-" * 118)
    print(
        f"{'Variant':<16} {'Positive':>10} {'MeanMargin':>12} "
        f"{'MinMargin':>11} {'Preserve':>10} {'DeltaMargin':>12} "
        f"{'LostCases':>10}"
    )
    print("-" * 118)

    for variant in ABLATIONS:
        summary = results[variant]["summary"]
        delta_margin = summary["mean_margin"] - full["mean_margin"]
        lost_cases = full["positive_cases"] - summary["positive_cases"]

        print(
            f"{variant:<16} "
            f"{summary['positive_cases']:>3}/{len(prepared):<6} "
            f"{summary['mean_margin']:+12.6f} "
            f"{summary['min_margin']:+11.6f} "
            f"{summary['mean_preservation']:10.6f} "
            f"{delta_margin:+12.6f} "
            f"{lost_cases:+10d}"
        )

    print()

    role_drops = {
        "Subject": full["mean_margin"]
        - results["no_subject"]["summary"]["mean_margin"],
        "Predicate": full["mean_margin"]
        - results["no_predicate"]["summary"]["mean_margin"],
        "Object": full["mean_margin"]
        - results["no_object"]["summary"]["mean_margin"],
    }

    print("Role contribution by mean-margin drop")
    print("-" * 118)
    for role, drop in role_drops.items():
        print(f"{role:<12}: {drop:+.6f}")
    print()

    return results, role_drops


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

    print("=" * 118)
    print(" LLM_SEM v0.4.4 Role Ablation / Causal Contribution Analysis")
    print("=" * 118)
    print("Device              :", device)
    if device.type == "cuda":
        print("GPU                 :", torch.cuda.get_device_name(0))
    print("Base checkpoint loss:", base_checkpoint.get("loss"))
    print("Projection checkpoint:", CHECKPOINT)
    print("Projection seed     :", seed)
    print("Vector dimension    :", dim)
    print("Retraining          : False")
    print()

    all_drops = {}

    results_a, drops_a = report_split(
        "A) Seen relation / unseen proposition",
        role_model,
        split_a,
    )
    results_b, drops_b = report_split(
        "B) Unseen relation / seen concepts",
        role_model,
        split_b,
    )
    results_c, drops_c = report_split(
        "C) Unseen relation / unseen concepts",
        role_model,
        split_c,
    )

    all_drops["A"] = drops_a
    all_drops["B"] = drops_b
    all_drops["C"] = drops_c

    print("Cross-split causal contribution summary")
    print("-" * 118)
    print(
        f"{'Role removed':<18} {'A drop':>12} {'B drop':>12} "
        f"{'C drop':>12} {'Mean drop':>12}"
    )
    print("-" * 118)

    aggregate = {}
    for role in ("Subject", "Predicate", "Object"):
        a = all_drops["A"][role]
        b = all_drops["B"][role]
        c = all_drops["C"][role]
        mean_drop = (a + b + c) / 3.0
        aggregate[role] = mean_drop

        print(
            f"{role:<18} {a:+12.6f} {b:+12.6f} "
            f"{c:+12.6f} {mean_drop:+12.6f}"
        )

    most_important = max(aggregate, key=aggregate.get)

    print()
    print("Interpretation")
    print("-" * 118)
    print(
        "Positive drop means removing that role reduces structural margin, "
        "so the role contributes causally to the trained composition."
    )
    print(
        "Near-zero drop means that role is largely redundant under this "
        "evaluation. A negative drop means removal unexpectedly improves "
        "margin and indicates interference or over-reliance elsewhere."
    )
    print(f"Largest mean contribution: {most_important}")

    output = {
        "version": "v0.4.4",
        "source_checkpoint": CHECKPOINT,
        "best_seed": seed,
        "role_mean_margin_drop": aggregate,
        "split_A": {
            k: v["summary"] for k, v in results_a.items()
        },
        "split_B": {
            k: v["summary"] for k, v in results_b.items()
        },
        "split_C": {
            k: v["summary"] for k, v in results_c.items()
        },
    }

    out_path = Path("results/role_ablation_v044.pt")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, out_path)
    print()
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
