# run_semantic_composition_v062.py
#
# LLM_SEM v0.6.2 Semantic Composition Experiment
#
# Goal:
#   Test whether Subject + Predicate + Object can be composed into a
#   proposition-level semantic representation that discriminates the correct
#   proposition from structural counterfactuals and generalizes to unseen
#   propositions, relations, and concepts.
#
# Compared methods:
#   1) simple_mean       : mean(normalized S, P, O)
#   2) role_aware        : frozen v0.4.3 role projection
#   3) relation_weighted : role projection with stronger predicate contribution
#   4) learned_comp      : train-only composition head over projected roles
#
# Leakage control:
#   - v0.4.3 checkpoint is loaded but never retrained here.
#   - learned_comp is optimized only on TRAIN_CASES.
#   - A/B/C holdouts never enter optimization or seed selection.
#
# Run:
#   python run_semantic_composition_v062.py

from __future__ import annotations

from pathlib import Path
import random

import torch
import torch.nn as nn
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
OUT = "results/semantic_composition_v062.pt"

SEEDS = [1, 2, 3, 4, 5]
EPOCHS = 300
LR = 5e-4
TARGET_MARGIN = 0.08
PRESERVATION_LAMBDA = 0.25
HIDDEN_SCALE = 2

METHODS = (
    "simple_mean",
    "role_aware",
    "relation_weighted",
    "learned_comp",
)


class LearnedComposer(nn.Module):
    """Train-only proposition composer over frozen role-projected vectors."""

    def __init__(self, dim: int, seed: int):
        super().__init__()
        hidden = dim * HIDDEN_SCALE

        random.seed(seed)
        torch.manual_seed(seed)

        self.net = nn.Sequential(
            nn.Linear(dim * 3, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )
        self.skip = nn.Linear(dim * 3, dim, bias=False)

        with torch.no_grad():
            self.skip.weight.zero_()
            eye = torch.eye(dim)
            self.skip.weight[:, 0:dim].copy_(eye / 3.0)
            self.skip.weight[:, dim:2 * dim].copy_(eye / 3.0)
            self.skip.weight[:, 2 * dim:3 * dim].copy_(eye / 3.0)

    def forward(self, rs, rp, ro):
        x = torch.cat([rs, rp, ro], dim=-1)
        return F.normalize(self.skip(x) + self.net(x), dim=-1)


def projected_roles(role_model, item):
    return (
        role_model.subject(item["s"]),
        role_model.predicate(item["p"]),
        role_model.object(item["o"]),
    )


def compose_simple(item):
    return F.normalize(
        torch.stack([
            F.normalize(item["s"], dim=0),
            F.normalize(item["p"], dim=0),
            F.normalize(item["o"], dim=0),
        ]).mean(dim=0),
        dim=0,
    )


def compose_role_aware(role_model, item):
    rs, rp, ro = projected_roles(role_model, item)
    return F.normalize((rs + rp + ro) / 3.0, dim=0)


def compose_relation_weighted(role_model, item):
    # Predicate encodes the explicit relation label in the v0.4.x benchmark.
    # This variant tests whether emphasizing relation identity improves
    # proposition discrimination without learning new parameters.
    rs, rp, ro = projected_roles(role_model, item)
    return F.normalize(0.25 * rs + 0.50 * rp + 0.25 * ro, dim=0)


def compose_learned(role_model, composer, item):
    rs, rp, ro = projected_roles(role_model, item)
    return composer(rs, rp, ro)


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


def evaluate_method(method, role_model, composer, prepared):
    rows = []
    for item in prepared:
        if method == "simple_mean":
            composed = compose_simple(item)
        elif method == "role_aware":
            composed = compose_role_aware(role_model, item)
        elif method == "relation_weighted":
            composed = compose_relation_weighted(role_model, item)
        elif method == "learned_comp":
            if composer is None:
                raise ValueError("learned_comp requires a trained composer")
            composed = compose_learned(role_model, composer, item)
        else:
            raise ValueError(method)
        rows.append(evaluate_vector(composed, item))
    return rows


def train_composer(role_model, dim, train, device, seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    composer = LearnedComposer(dim, seed).to(device)
    optimizer = torch.optim.Adam(composer.parameters(), lr=LR)

    role_model.eval()
    for parameter in role_model.parameters():
        parameter.requires_grad_(False)

    for _ in range(EPOCHS):
        optimizer.zero_grad()
        ranking_losses = []
        preservation_losses = []

        for item in train:
            composed = compose_learned(role_model, composer, item)
            pos = F.cosine_similarity(composed, item["positive"], dim=0)

            for cf in item["counterfactuals"]:
                neg = F.cosine_similarity(composed, cf, dim=0)
                ranking_losses.append(
                    F.relu(TARGET_MARGIN - pos + neg)
                )

            # Keep the learned proposition near the base proposition embedding.
            preservation_losses.append(1.0 - pos)

        loss = (
            torch.stack(ranking_losses).mean()
            + PRESERVATION_LAMBDA * torch.stack(preservation_losses).mean()
        )
        loss.backward()
        optimizer.step()

    return composer


def print_method_table(title, results, count):
    print(title)
    print("-" * 112)
    print(
        f"{'Method':<20} {'Positive':>10} {'MeanMargin':>12} "
        f"{'MinMargin':>11} {'Preserve':>10}"
    )
    print("-" * 112)
    for method in METHODS:
        s = summarize(results[method])
        print(
            f"{method:<20} "
            f"{s['positive_cases']:>3}/{count:<6} "
            f"{s['mean_margin']:+12.6f} "
            f"{s['min_margin']:+11.6f} "
            f"{s['mean_preservation']:10.6f}"
        )
    print()


def generalization_score(results):
    # Seed selection uses TRAIN only. This function is for reporting holdout
    # aggregate after the seed has already been fixed.
    summaries = [summarize(results[x]["learned_comp"]) for x in ("A", "B", "C")]
    return {
        "positive_cases": sum(s["positive_cases"] for s in summaries),
        "mean_margin": sum(s["mean_margin"] for s in summaries) / len(summaries),
        "mean_preservation": sum(s["mean_preservation"] for s in summaries) / len(summaries),
    }


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
        base_model, tokenizer, device, HOLDOUT_SEEN_REL
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

    print("=" * 112)
    print(" LLM_SEM v0.6.2 Semantic Composition Experiment")
    print("=" * 112)
    print("Device               :", device)
    if device.type == "cuda":
        print("GPU                  :", torch.cuda.get_device_name(0))
    print("Base checkpoint loss :", base_checkpoint.get("loss"))
    print("Role checkpoint      :", ROLE_CHECKPOINT)
    print("Role projection seed :", role_seed)
    print("Vector dimension     :", dim)
    print("Composer seeds       :", SEEDS)
    print("Composer epochs      :", EPOCHS)
    print("Holdout in training  : False")
    print("Holdout seed select  : False")
    print()

    # Train-only seed selection.
    candidates = []
    for seed in SEEDS:
        composer = train_composer(
            role_model,
            dim,
            train,
            device,
            seed,
        )
        rows = evaluate_method(
            "learned_comp",
            role_model,
            composer,
            train,
        )
        s = summarize(rows)
        score = (
            s["positive_cases"],
            s["mean_margin"],
            s["min_margin"],
            s["mean_preservation"],
        )
        candidates.append((score, seed, composer, rows))
        print(
            f"seed={seed:<2} train={s['positive_cases']}/{len(train)} "
            f"margin={s['mean_margin']:+.6f} "
            f"min={s['min_margin']:+.6f} "
            f"preserve={s['mean_preservation']:.6f}"
        )

    _, best_seed, composer, train_learned = max(
        candidates,
        key=lambda x: x[0],
    )
    composer.eval()
    print()
    print("Best composer seed selected by TRAIN only:", best_seed)
    print()

    splits = {
        "TRAIN": train,
        "A": split_a,
        "B": split_b,
        "C": split_c,
    }
    results = {}

    for split_name, prepared in splits.items():
        results[split_name] = {}
        for method in METHODS:
            if split_name == "TRAIN" and method == "learned_comp":
                rows = train_learned
            else:
                rows = evaluate_method(
                    method,
                    role_model,
                    composer,
                    prepared,
                )
            results[split_name][method] = rows

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

    print("Cross-split generalization")
    print("-" * 112)
    print(
        f"{'Method':<20} {'Positive':>10} {'MeanMargin':>12} "
        f"{'Preserve':>10}"
    )
    print("-" * 112)

    aggregate = {}
    total_holdout = len(split_a) + len(split_b) + len(split_c)
    for method in METHODS:
        ss = [summarize(results[x][method]) for x in ("A", "B", "C")]
        agg = {
            "positive_cases": sum(x["positive_cases"] for x in ss),
            "mean_margin": sum(x["mean_margin"] for x in ss) / 3.0,
            "mean_preservation": sum(x["mean_preservation"] for x in ss) / 3.0,
        }
        aggregate[method] = agg
        print(
            f"{method:<20} "
            f"{agg['positive_cases']:>3}/{total_holdout:<6} "
            f"{agg['mean_margin']:+12.6f} "
            f"{agg['mean_preservation']:10.6f}"
        )
    print()

    print("Interpretation")
    print("-" * 112)
    print(
        "A positive counterfactual margin means the composed semantic vector "
        "is closer to the correct proposition than to all one-role replacements."
    )
    print(
        "A/B/C separate proposition generalization, unseen-relation transfer, "
        "and the strict unseen-relation + unseen-concept condition."
    )
    print(
        "If learned_comp improves TRAIN but not B/C, the composer is likely "
        "overfitting proposition patterns rather than learning reusable "
        "semantic composition."
    )
    print(
        "If role_aware or relation_weighted remain competitive on B/C, the "
        "frozen structural roles are providing transferable composition bias."
    )

    output = {
        "version": "v0.6.2",
        "base_checkpoint_loss": base_checkpoint.get("loss"),
        "role_checkpoint": ROLE_CHECKPOINT,
        "role_seed": role_seed,
        "best_composer_seed": best_seed,
        "epochs": EPOCHS,
        "lr": LR,
        "target_margin": TARGET_MARGIN,
        "preservation_lambda": PRESERVATION_LAMBDA,
        "methods": list(METHODS),
        "train": {
            m: summarize(results["TRAIN"][m]) for m in METHODS
        },
        "split_A": {
            m: summarize(results["A"][m]) for m in METHODS
        },
        "split_B": {
            m: summarize(results["B"][m]) for m in METHODS
        },
        "split_C": {
            m: summarize(results["C"][m]) for m in METHODS
        },
        "aggregate_holdout": aggregate,
        "composer_state_dict": composer.state_dict(),
    }

    out_path = Path(OUT)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, out_path)
    print()
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
