# run_structural_generalization_holdout_v042.py
#
# LLM_SEM v0.4.2 Structural Generalization / Holdout Evaluation
#
# Purpose:
#   Verify that Structural Role Projection learns role binding rather than
#   merely memorizing the propositions used during optimization.
#
# Protocol:
#   - Base LLM_SEM encoder is frozen.
#   - Train and holdout propositions are disjoint.
#   - Only Subject / Predicate / Object projection matrices are trainable.
#   - Training uses margin ranking + semantic preservation.
#   - Seed selection uses TRAIN metrics only.
#   - HOLDOUT is reported after model selection and is never used for fitting.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import LanguageModel
from tokenizer import Tokenizer
from semantic import encode_text


MODEL = "model/model-gpu-v0.4.pt"
TOKENIZER = "model/tokenizer.json"
OUT = "model/structural-role-projection-v042.pt"

SEEDS = [1, 2, 3, 4, 5]
EPOCHS = 300
LR = 5e-4
TARGET_MARGIN = 0.08
PRESERVATION_LAMBDA = 0.35
INIT_NOISE_STD = 0.01


@dataclass(frozen=True)
class Case:
    name: str
    subject: str
    predicate: str
    object: str
    subject_cf: str
    predicate_cf: str
    object_cf: str


# Training propositions. These are the only examples used for optimization.
TRAIN_CASES = [
    Case("gpu-fast", "GPU", "has_property", "高速", "CPU", "has_predicate", "低速"),
    Case("cpu-executes", "CPU", "has_predicate", "命令を実行する", "GPU", "has_property", "画像を表示する"),
    Case("cuda-gpu", "CUDA", "targets", "GPU", "Python", "has_property", "weather"),
    Case("python-computer", "Python", "related_with", "computer", "weather", "has_property", "food"),
    Case("gpu-parallel", "GPU", "has_property", "並列処理に強い", "CPU", "related_with", "低速"),
    Case("cuda-parallel", "CUDA", "targets", "並列計算", "Python", "has_predicate", "逐次処理"),
    Case("python-program", "Python", "is_a", "プログラミング言語", "CUDA", "targets", "GPU"),
    Case("cpu-processor", "CPU", "is_a", "プロセッサ", "GPU", "has_property", "高速"),
]


# Holdout propositions are never included in the optimizer loss.
# Subjects/objects are chosen to differ from the training examples where
# practical, while reusing relation roles so the test measures role binding.
HOLDOUT_CASES = [
    Case("memory-storage", "メモリ", "is_a", "記憶装置", "CPU", "has_property", "演算装置"),
    Case("network-connects", "ネットワーク", "related_with", "通信", "GPU", "targets", "画像"),
    Case("compiler-code", "コンパイラ", "targets", "プログラム", "CUDA", "is_a", "ハードウェア"),
    Case("database-data", "データベース", "has_property", "データを保持する", "CPU", "related_with", "命令"),
    Case("sensor-measures", "センサー", "has_predicate", "値を測定する", "GPU", "has_property", "高速"),
    Case("router-packet", "ルータ", "targets", "パケット", "Python", "related_with", "computer"),
]


class StructuralRoleProjection(nn.Module):
    def __init__(self, dim: int, seed: int):
        super().__init__()
        self.subject = nn.Linear(dim, dim, bias=False)
        self.predicate = nn.Linear(dim, dim, bias=False)
        self.object = nn.Linear(dim, dim, bias=False)

        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        eye = torch.eye(dim)

        with torch.no_grad():
            for layer in (self.subject, self.predicate, self.object):
                noise = torch.randn(
                    dim,
                    dim,
                    generator=generator,
                    dtype=eye.dtype,
                ) * INIT_NOISE_STD
                layer.weight.copy_(eye + noise)

    def forward(
        self,
        subject_v: torch.Tensor,
        predicate_v: torch.Tensor,
        object_v: torch.Tensor,
    ) -> torch.Tensor:
        out = (
            self.subject(subject_v)
            + self.predicate(predicate_v)
            + self.object(object_v)
        ) / 3.0
        return F.normalize(out, dim=-1)


def encode(model, tokenizer, text: str, device: torch.device) -> torch.Tensor:
    data = encode_text(model, tokenizer, text)
    return torch.tensor(data.vector, dtype=torch.float32, device=device)


def prop_text(subject: str, predicate: str, object_: str) -> str:
    return f"{subject} {predicate} {object_}"


def raw_compose(
    subject_v: torch.Tensor,
    predicate_v: torch.Tensor,
    object_v: torch.Tensor,
) -> torch.Tensor:
    stacked = torch.stack(
        [
            F.normalize(subject_v, dim=0),
            F.normalize(predicate_v, dim=0),
            F.normalize(object_v, dim=0),
        ]
    )
    return F.normalize(stacked.mean(dim=0), dim=0)


def prepare(model, tokenizer, device, cases):
    prepared = []

    for case in cases:
        subject_v = encode(model, tokenizer, case.subject, device)
        predicate_v = encode(model, tokenizer, case.predicate, device)
        object_v = encode(model, tokenizer, case.object, device)

        positive_v = F.normalize(
            encode(
                model,
                tokenizer,
                prop_text(case.subject, case.predicate, case.object),
                device,
            ),
            dim=0,
        )

        counterfactuals = [
            F.normalize(
                encode(
                    model,
                    tokenizer,
                    prop_text(case.subject_cf, case.predicate, case.object),
                    device,
                ),
                dim=0,
            ),
            F.normalize(
                encode(
                    model,
                    tokenizer,
                    prop_text(case.subject, case.predicate_cf, case.object),
                    device,
                ),
                dim=0,
            ),
            F.normalize(
                encode(
                    model,
                    tokenizer,
                    prop_text(case.subject, case.predicate, case.object_cf),
                    device,
                ),
                dim=0,
            ),
        ]

        prepared.append(
            {
                "case": case,
                "subject": subject_v,
                "predicate": predicate_v,
                "object": object_v,
                "positive": positive_v,
                "counterfactuals": counterfactuals,
                "raw_composed": raw_compose(subject_v, predicate_v, object_v),
            }
        )

    return prepared


def evaluate(role_model, prepared):
    rows = []

    for item in prepared:
        composed = role_model(
            item["subject"],
            item["predicate"],
            item["object"],
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

        raw_preservation = float(
            F.cosine_similarity(composed, item["raw_composed"], dim=0).item()
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
                raw_preservation,
            )
        )

    return rows


def summarize(rows):
    margins = [row[6] for row in rows]
    preservation = [row[7] for row in rows]
    return {
        "positive_cases": sum(1 for margin in margins if margin > 0.0),
        "mean_margin": sum(margins) / len(margins),
        "min_margin": min(margins),
        "mean_preservation": sum(preservation) / len(preservation),
    }


def train_one_seed(dim, train_prepared, device, seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    role_model = StructuralRoleProjection(dim, seed=seed).to(device)
    optimizer = torch.optim.Adam(role_model.parameters(), lr=LR)

    before = evaluate(role_model, train_prepared)

    for _ in range(EPOCHS):
        optimizer.zero_grad()
        ranking_losses = []
        preservation_losses = []

        for item in train_prepared:
            composed = role_model(
                item["subject"],
                item["predicate"],
                item["object"],
            )
            positive = F.cosine_similarity(
                composed,
                item["positive"],
                dim=0,
            )

            for counterfactual in item["counterfactuals"]:
                negative = F.cosine_similarity(
                    composed,
                    counterfactual,
                    dim=0,
                )
                ranking_losses.append(
                    F.relu(TARGET_MARGIN - positive + negative)
                )

            preservation_losses.append(1.0 - positive)

        ranking_loss = torch.stack(ranking_losses).mean()
        preservation_loss = torch.stack(preservation_losses).mean()
        loss = ranking_loss + PRESERVATION_LAMBDA * preservation_loss

        loss.backward()
        optimizer.step()

    after = evaluate(role_model, train_prepared)
    return role_model, before, after


def print_rows(title, rows):
    print(title)
    print("-" * 114)
    print(
        f"{'Case':<20} {'Positive':>9} {'CF-subj':>9} {'CF-pred':>9} "
        f"{'CF-obj':>9} {'CF-best':>9} {'Margin':>9} {'Preserve':>9}"
    )
    print("-" * 114)

    for name, pos, cfs, cfp, cfo, best_cf, margin, preservation in rows:
        print(
            f"{name:<20} {pos:9.6f} {cfs:9.6f} {cfp:9.6f} "
            f"{cfo:9.6f} {best_cf:9.6f} {margin:+9.6f} "
            f"{preservation:9.6f}"
        )
    print()


def format_summary(label, before, after, n_cases):
    print(label)
    print("-" * 114)
    print(
        f"Positive-margin cases : "
        f"{before['positive_cases']}/{n_cases} -> "
        f"{after['positive_cases']}/{n_cases}"
    )
    print(
        f"Mean structural margin: "
        f"{before['mean_margin']:+.6f} -> {after['mean_margin']:+.6f}"
    )
    print(
        f"Minimum margin        : "
        f"{before['min_margin']:+.6f} -> {after['min_margin']:+.6f}"
    )
    print(
        f"Mean preservation     : "
        f"{before['mean_preservation']:.6f} -> "
        f"{after['mean_preservation']:.6f}"
    )
    print()


def main():
    if not Path(MODEL).exists():
        raise FileNotFoundError(MODEL)
    if not Path(TOKENIZER).exists():
        raise FileNotFoundError(TOKENIZER)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer.load(TOKENIZER)
    model, checkpoint = LanguageModel.load_checkpoint(MODEL, device=device)
    model.eval()

    train_prepared = prepare(model, tokenizer, device, TRAIN_CASES)
    holdout_prepared = prepare(model, tokenizer, device, HOLDOUT_CASES)
    dim = len(train_prepared[0]["subject"])

    print("=" * 114)
    print(" LLM_SEM v0.4.2 Structural Generalization / Holdout Evaluation")
    print("=" * 114)
    print("Device              :", device)
    if device.type == "cuda":
        print("GPU                 :", torch.cuda.get_device_name(0))
    print("Checkpoint loss     :", checkpoint.get("loss"))
    print("Vector dimension    :", dim)
    print("Train cases         :", len(TRAIN_CASES))
    print("Holdout cases       :", len(HOLDOUT_CASES))
    print("Holdout in training : False")
    print("Seeds               :", SEEDS)
    print("Epochs              :", EPOCHS)
    print("Learning rate       :", LR)
    print("Target margin       :", TARGET_MARGIN)
    print("Preservation lambda :", PRESERVATION_LAMBDA)
    print("Init noise std      :", INIT_NOISE_STD)
    print()

    candidates = []

    for seed in SEEDS:
        role_model, train_before, train_after = train_one_seed(
            dim,
            train_prepared,
            device,
            seed,
        )

        train_before_summary = summarize(train_before)
        train_after_summary = summarize(train_after)
        holdout_before = evaluate(
            StructuralRoleProjection(dim, seed=seed).to(device),
            holdout_prepared,
        )
        holdout_after = evaluate(role_model, holdout_prepared)
        holdout_before_summary = summarize(holdout_before)
        holdout_after_summary = summarize(holdout_after)

        # Model selection is TRAIN ONLY. Holdout metrics are printed for
        # transparency but never enter the score.
        train_score = (
            train_after_summary["positive_cases"],
            train_after_summary["mean_margin"],
            train_after_summary["min_margin"],
            train_after_summary["mean_preservation"],
        )

        candidates.append(
            {
                "score": train_score,
                "seed": seed,
                "model": role_model,
                "train_before": train_before,
                "train_after": train_after,
                "holdout_before": holdout_before,
                "holdout_after": holdout_after,
            }
        )

        print(
            f"seed={seed:<2} "
            f"train "
            f"{train_before_summary['positive_cases']}/{len(TRAIN_CASES)} "
            f"{train_before_summary['mean_margin']:+.5f} -> "
            f"{train_after_summary['positive_cases']}/{len(TRAIN_CASES)} "
            f"{train_after_summary['mean_margin']:+.5f}  "
            f"holdout "
            f"{holdout_before_summary['positive_cases']}/{len(HOLDOUT_CASES)} "
            f"{holdout_before_summary['mean_margin']:+.5f} -> "
            f"{holdout_after_summary['positive_cases']}/{len(HOLDOUT_CASES)} "
            f"{holdout_after_summary['mean_margin']:+.5f}"
        )

    best = max(candidates, key=lambda item: item["score"])
    best_seed = best["seed"]

    train_before_summary = summarize(best["train_before"])
    train_after_summary = summarize(best["train_after"])
    holdout_before_summary = summarize(best["holdout_before"])
    holdout_after_summary = summarize(best["holdout_after"])

    print()
    print("Best seed selected by TRAIN metrics only:", best_seed)
    print()

    print_rows("TRAIN before projection", best["train_before"])
    print_rows("TRAIN after projection", best["train_after"])
    format_summary(
        "TRAIN summary",
        train_before_summary,
        train_after_summary,
        len(TRAIN_CASES),
    )

    print_rows("HOLDOUT before projection", best["holdout_before"])
    print_rows("HOLDOUT after projection", best["holdout_after"])
    format_summary(
        "HOLDOUT summary",
        holdout_before_summary,
        holdout_after_summary,
        len(HOLDOUT_CASES),
    )

    generalization_gain = (
        holdout_after_summary["mean_margin"]
        - holdout_before_summary["mean_margin"]
    )

    print("Generalization verdict")
    print("-" * 114)
    print(
        "Holdout mean-margin gain : "
        f"{generalization_gain:+.6f}"
    )
    print(
        "Holdout positive cases   : "
        f"{holdout_before_summary['positive_cases']}/{len(HOLDOUT_CASES)} -> "
        f"{holdout_after_summary['positive_cases']}/{len(HOLDOUT_CASES)}"
    )
    print(
        "Holdout preservation     : "
        f"{holdout_after_summary['mean_preservation']:.6f}"
    )
    print()
    print(
        "Evidence for structural generalization requires improvement on HOLDOUT, "
        "not only on TRAIN. A train-only improvement is compatible with memorization."
    )

    checkpoint_out = {
        "version": "v0.4.2",
        "dimension": dim,
        "best_seed": best_seed,
        "epochs": EPOCHS,
        "lr": LR,
        "target_margin": TARGET_MARGIN,
        "preservation_lambda": PRESERVATION_LAMBDA,
        "init_noise_std": INIT_NOISE_STD,
        "state_dict": best["model"].state_dict(),
        "train_before": train_before_summary,
        "train_after": train_after_summary,
        "holdout_before": holdout_before_summary,
        "holdout_after": holdout_after_summary,
        "train_cases": [case.__dict__ for case in TRAIN_CASES],
        "holdout_cases": [case.__dict__ for case in HOLDOUT_CASES],
    }

    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint_out, OUT)
    print("Saved:", OUT)


if __name__ == "__main__":
    main()
