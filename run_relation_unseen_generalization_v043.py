# run_relation_unseen_generalization_v043.py
#
# LLM_SEM v0.4.3 Relation-Unseen Structural Generalization
#
# Evaluate whether role projection generalizes beyond relation labels observed
# during training.
#
# Splits:
#   A) seen-relation / unseen-proposition
#   B) unseen-relation / seen-concepts
#   C) unseen-relation / unseen-concepts
#
# Holdout splits never enter optimization or model selection.

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
OUT = "model/structural-role-projection-v043.pt"

SEEDS = [1, 2, 3, 4, 5]
EPOCHS = 350
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


TRAIN_CASES = [
    Case("gpu-fast", "GPU", "has_property", "高速", "CPU", "targets", "低速"),
    Case("gpu-parallel", "GPU", "has_property", "並列処理に強い", "CPU", "related_with", "低速"),
    Case("cuda-gpu", "CUDA", "targets", "GPU", "Python", "has_property", "weather"),
    Case("cuda-parallel", "CUDA", "targets", "並列計算", "Python", "related_with", "逐次処理"),
    Case("python-computer", "Python", "related_with", "computer", "weather", "targets", "food"),
    Case("network-data", "network", "related_with", "data", "GPU", "has_property", "image"),
]

SEEN_RELATIONS = {"has_property", "targets", "related_with"}
UNSEEN_RELATIONS = {"is_a", "has_predicate", "used_for", "part_of", "causes"}

# A: relation was seen in TRAIN, but proposition is new.
HOLDOUT_SEEN_REL = [
    Case("memory-fast", "メモリ", "has_property", "高速アクセス", "CPU", "targets", "低速"),
    Case("compiler-program", "コンパイラ", "targets", "プログラム", "CUDA", "related_with", "GPU"),
    Case("router-network", "ルータ", "related_with", "ネットワーク", "GPU", "has_property", "画像"),
]

# B: relation is unseen, while subject/object reuse concepts represented in TRAIN.
HOLDOUT_UNSEEN_REL_SEEN_CONCEPTS = [
    Case("gpu-processor", "GPU", "is_a", "computer", "CPU", "has_property", "weather"),
    Case("cuda-executes", "CUDA", "has_predicate", "並列計算", "Python", "targets", "逐次処理"),
    Case("python-used-for", "Python", "used_for", "data", "GPU", "related_with", "image"),
]

# C: both relation and core concepts are unseen.
HOLDOUT_UNSEEN_REL_UNSEEN_CONCEPTS = [
    Case("sensor-device", "センサー", "is_a", "測定装置", "カメラ", "has_property", "入力装置"),
    Case("database-stores", "データベース", "has_predicate", "データを保持する", "ファイル", "targets", "データ"),
    Case("cache-part", "キャッシュ", "part_of", "メモリ階層", "CPU", "related_with", "ストレージ階層"),
    Case("heat-causes", "発熱", "causes", "温度上昇", "電圧", "has_property", "温度低下"),
    Case("compiler-used-for", "コンパイラ", "used_for", "コード変換", "CPU", "targets", "画像生成"),
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
                noise = torch.randn(dim, dim, generator=generator) * INIT_NOISE_STD
                layer.weight.copy_(eye + noise)

    def forward(self, subject_v, predicate_v, object_v):
        out = (
            self.subject(subject_v)
            + self.predicate(predicate_v)
            + self.object(object_v)
        ) / 3.0
        return F.normalize(out, dim=-1)


def encode(model, tokenizer, text, device):
    item = encode_text(model, tokenizer, text)
    return torch.tensor(item.vector, dtype=torch.float32, device=device)


def prop_text(s, p, o):
    return f"{s} {p} {o}"


def raw_compose(s, p, o):
    stacked = torch.stack([
        F.normalize(s, dim=0),
        F.normalize(p, dim=0),
        F.normalize(o, dim=0),
    ])
    return F.normalize(stacked.mean(dim=0), dim=0)


def prepare(model, tokenizer, device, cases):
    prepared = []
    for case in cases:
        s = encode(model, tokenizer, case.subject, device)
        p = encode(model, tokenizer, case.predicate, device)
        o = encode(model, tokenizer, case.object, device)

        positive = F.normalize(
            encode(model, tokenizer, prop_text(case.subject, case.predicate, case.object), device),
            dim=0,
        )
        cfs = [
            F.normalize(encode(model, tokenizer, prop_text(case.subject_cf, case.predicate, case.object), device), dim=0),
            F.normalize(encode(model, tokenizer, prop_text(case.subject, case.predicate_cf, case.object), device), dim=0),
            F.normalize(encode(model, tokenizer, prop_text(case.subject, case.predicate, case.object_cf), device), dim=0),
        ]

        prepared.append({
            "case": case,
            "s": s,
            "p": p,
            "o": o,
            "positive": positive,
            "counterfactuals": cfs,
            "raw": raw_compose(s, p, o),
        })
    return prepared


def evaluate(role_model, prepared):
    rows = []
    for item in prepared:
        composed = role_model(item["s"], item["p"], item["o"])
        pos = float(F.cosine_similarity(composed, item["positive"], dim=0).item())
        cfs = [
            float(F.cosine_similarity(composed, cf, dim=0).item())
            for cf in item["counterfactuals"]
        ]
        best_cf = max(cfs)
        margin = pos - best_cf
        preservation = float(F.cosine_similarity(composed, item["raw"], dim=0).item())
        rows.append((item["case"].name, pos, *cfs, best_cf, margin, preservation))
    return rows


def summarize(rows):
    margins = [r[6] for r in rows]
    preservation = [r[7] for r in rows]
    return {
        "positive_cases": sum(1 for m in margins if m > 0),
        "mean_margin": sum(margins) / len(margins),
        "min_margin": min(margins),
        "mean_preservation": sum(preservation) / len(preservation),
    }


def train_one_seed(dim, train_prepared, device, seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    role_model = StructuralRoleProjection(dim, seed).to(device)
    optimizer = torch.optim.Adam(role_model.parameters(), lr=LR)

    before = evaluate(role_model, train_prepared)

    for _ in range(EPOCHS):
        optimizer.zero_grad()
        ranking_losses = []
        preservation_losses = []

        for item in train_prepared:
            composed = role_model(item["s"], item["p"], item["o"])
            pos = F.cosine_similarity(composed, item["positive"], dim=0)

            for cf in item["counterfactuals"]:
                neg = F.cosine_similarity(composed, cf, dim=0)
                ranking_losses.append(F.relu(TARGET_MARGIN - pos + neg))

            preservation_losses.append(1.0 - pos)

        loss = (
            torch.stack(ranking_losses).mean()
            + PRESERVATION_LAMBDA * torch.stack(preservation_losses).mean()
        )
        loss.backward()
        optimizer.step()

    return role_model, before, evaluate(role_model, train_prepared)


def print_rows(title, rows):
    print(title)
    print("-" * 114)
    print(
        f"{'Case':<22} {'Positive':>9} {'CF-subj':>9} {'CF-pred':>9} "
        f"{'CF-obj':>9} {'CF-best':>9} {'Margin':>9} {'Preserve':>9}"
    )
    print("-" * 114)
    for name, pos, cfs, cfp, cfo, best_cf, margin, preserve in rows:
        print(
            f"{name:<22} {pos:9.6f} {cfs:9.6f} {cfp:9.6f} "
            f"{cfo:9.6f} {best_cf:9.6f} {margin:+9.6f} {preserve:9.6f}"
        )
    print()


def print_summary(label, before, after, count):
    b = summarize(before)
    a = summarize(after)
    print(label)
    print("-" * 114)
    print(f"Positive-margin cases : {b['positive_cases']}/{count} -> {a['positive_cases']}/{count}")
    print(f"Mean structural margin: {b['mean_margin']:+.6f} -> {a['mean_margin']:+.6f}")
    print(f"Minimum margin        : {b['min_margin']:+.6f} -> {a['min_margin']:+.6f}")
    print(f"Mean preservation     : {b['mean_preservation']:.6f} -> {a['mean_preservation']:.6f}")
    print()


def validate_cases(label, cases):
    """Reject degenerate counterfactuals identical to the positive proposition."""
    for case in cases:
        positive = (case.subject, case.predicate, case.object)
        variants = {
            "subject": (case.subject_cf, case.predicate, case.object),
            "predicate": (case.subject, case.predicate_cf, case.object),
            "object": (case.subject, case.predicate, case.object_cf),
        }
        for kind, candidate in variants.items():
            if candidate == positive:
                raise ValueError(
                    f"{label}:{case.name} has degenerate {kind} counterfactual "
                    f"identical to positive: {positive}"
                )


def main():
    if not Path(MODEL).exists():
        raise FileNotFoundError(MODEL)
    if not Path(TOKENIZER).exists():
        raise FileNotFoundError(TOKENIZER)

    validate_cases("TRAIN", TRAIN_CASES)
    validate_cases("A", HOLDOUT_SEEN_REL)
    validate_cases("B", HOLDOUT_UNSEEN_REL_SEEN_CONCEPTS)
    validate_cases("C", HOLDOUT_UNSEEN_REL_UNSEEN_CONCEPTS)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer.load(TOKENIZER)
    model, checkpoint = LanguageModel.load_checkpoint(MODEL, device=device)
    model.eval()

    train = prepare(model, tokenizer, device, TRAIN_CASES)
    split_a = prepare(model, tokenizer, device, HOLDOUT_SEEN_REL)
    split_b = prepare(model, tokenizer, device, HOLDOUT_UNSEEN_REL_SEEN_CONCEPTS)
    split_c = prepare(model, tokenizer, device, HOLDOUT_UNSEEN_REL_UNSEEN_CONCEPTS)

    dim = len(train[0]["s"])

    print("=" * 114)
    print(" LLM_SEM v0.4.3.1 Relation-Unseen Structural Generalization")
    print("=" * 114)
    print("Device                    :", device)
    if device.type == "cuda":
        print("GPU                       :", torch.cuda.get_device_name(0))
    print("Checkpoint loss           :", checkpoint.get("loss"))
    print("Vector dimension          :", dim)
    print("Train cases               :", len(TRAIN_CASES))
    print("Seen-rel holdout          :", len(HOLDOUT_SEEN_REL))
    print("Unseen-rel/seen-concepts  :", len(HOLDOUT_UNSEEN_REL_SEEN_CONCEPTS))
    print("Unseen-rel/unseen-concepts:", len(HOLDOUT_UNSEEN_REL_UNSEEN_CONCEPTS))
    print("Seen relations            :", sorted(SEEN_RELATIONS))
    print("Unseen relations          :", sorted(UNSEEN_RELATIONS))
    print("Holdout in training       : False")
    print("Holdout in seed selection : False")
    print()

    candidates = []

    for seed in SEEDS:
        role_model, train_before, train_after = train_one_seed(dim, train, device, seed)
        train_summary = summarize(train_after)

        a_before = evaluate(StructuralRoleProjection(dim, seed).to(device), split_a)
        b_before = evaluate(StructuralRoleProjection(dim, seed).to(device), split_b)
        c_before = evaluate(StructuralRoleProjection(dim, seed).to(device), split_c)

        a_after = evaluate(role_model, split_a)
        b_after = evaluate(role_model, split_b)
        c_after = evaluate(role_model, split_c)

        score = (
            train_summary["positive_cases"],
            train_summary["mean_margin"],
            train_summary["min_margin"],
            train_summary["mean_preservation"],
        )

        candidates.append({
            "score": score,
            "seed": seed,
            "model": role_model,
            "train_before": train_before,
            "train_after": train_after,
            "a_before": a_before,
            "a_after": a_after,
            "b_before": b_before,
            "b_after": b_after,
            "c_before": c_before,
            "c_after": c_after,
        })

        sa = summarize(a_after)
        sb = summarize(b_after)
        sc = summarize(c_after)

        print(
            f"seed={seed:<2} "
            f"train={train_summary['positive_cases']}/{len(TRAIN_CASES)} "
            f"{train_summary['mean_margin']:+.5f} | "
            f"A={sa['positive_cases']}/{len(HOLDOUT_SEEN_REL)} {sa['mean_margin']:+.5f} | "
            f"B={sb['positive_cases']}/{len(HOLDOUT_UNSEEN_REL_SEEN_CONCEPTS)} {sb['mean_margin']:+.5f} | "
            f"C={sc['positive_cases']}/{len(HOLDOUT_UNSEEN_REL_UNSEEN_CONCEPTS)} {sc['mean_margin']:+.5f}"
        )

    best = max(candidates, key=lambda x: x["score"])
    print()
    print("Best seed selected by TRAIN metrics only:", best["seed"])
    print()

    print_rows("TRAIN after projection", best["train_after"])
    print_rows("A) Seen relation / unseen proposition", best["a_after"])
    print_rows("B) Unseen relation / seen concepts", best["b_after"])
    print_rows("C) Unseen relation / unseen concepts", best["c_after"])

    print_summary(
        "A summary: Seen relation / unseen proposition",
        best["a_before"], best["a_after"], len(HOLDOUT_SEEN_REL)
    )
    print_summary(
        "B summary: Unseen relation / seen concepts",
        best["b_before"], best["b_after"], len(HOLDOUT_UNSEEN_REL_SEEN_CONCEPTS)
    )
    print_summary(
        "C summary: Unseen relation / unseen concepts",
        best["c_before"], best["c_after"], len(HOLDOUT_UNSEEN_REL_UNSEEN_CONCEPTS)
    )

    b_after = summarize(best["b_after"])
    c_after = summarize(best["c_after"])

    print("Generalization interpretation")
    print("-" * 114)
    print(
        "A tests proposition generalization with familiar relation labels. "
        "B tests unseen relation labels on familiar concepts. "
        "C is the strictest split: unseen relations with unseen concepts."
    )
    print(
        "Strong evidence for structural role learning requires positive gains "
        "in B and preferably C, not only in A."
    )
    print(
        f"Unseen relation result B: {b_after['positive_cases']}/{len(HOLDOUT_UNSEEN_REL_SEEN_CONCEPTS)} "
        f"mean={b_after['mean_margin']:+.6f}"
    )
    print(
        f"Strict result C         : {c_after['positive_cases']}/{len(HOLDOUT_UNSEEN_REL_UNSEEN_CONCEPTS)} "
        f"mean={c_after['mean_margin']:+.6f}"
    )

    checkpoint_out = {
        "version": "v0.4.3",
        "dimension": dim,
        "best_seed": best["seed"],
        "epochs": EPOCHS,
        "lr": LR,
        "target_margin": TARGET_MARGIN,
        "preservation_lambda": PRESERVATION_LAMBDA,
        "init_noise_std": INIT_NOISE_STD,
        "seen_relations": sorted(SEEN_RELATIONS),
        "unseen_relations": sorted(UNSEEN_RELATIONS),
        "state_dict": best["model"].state_dict(),
        "train_after": summarize(best["train_after"]),
        "holdout_seen_relation": summarize(best["a_after"]),
        "holdout_unseen_relation_seen_concepts": summarize(best["b_after"]),
        "holdout_unseen_relation_unseen_concepts": summarize(best["c_after"]),
    }

    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint_out, OUT)
    print()
    print("Saved:", OUT)


if __name__ == "__main__":
    main()
