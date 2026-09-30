# run_structural_role_projection_v041.py
#
# LLM_SEM v0.4.1 Structural Role Encoding / Projection
#
# Learn separate Subject / Predicate / Object projections while freezing the
# base semantic encoder. Compare structural margins before and after training.

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
OUT = "model/structural-role-projection-v041.pt"

SEEDS = [1, 2, 3, 4, 5]
EPOCHS = 250
LR = 1e-3
TARGET_MARGIN = 0.10


@dataclass
class Case:
    name: str
    subject: str
    predicate: str
    object: str
    subject_cf: str
    predicate_cf: str
    object_cf: str


CASES = [
    Case("gpu-fast", "GPU", "has_property", "高速", "CPU", "has_predicate", "低速"),
    Case(
        "cpu-executes",
        "CPU",
        "has_predicate",
        "命令を実行する",
        "GPU",
        "has_property",
        "画像を表示する",
    ),
    Case("cuda-gpu", "CUDA", "targets", "GPU", "Python", "has_property", "weather"),
    Case(
        "python-computer",
        "Python",
        "related_with",
        "computer",
        "weather",
        "has_property",
        "food",
    ),
]


class StructuralRoleProjection(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.subject = nn.Linear(dim, dim, bias=False)
        self.predicate = nn.Linear(dim, dim, bias=False)
        self.object = nn.Linear(dim, dim, bias=False)

        eye = torch.eye(dim)
        with torch.no_grad():
            self.subject.weight.copy_(eye)
            self.predicate.weight.copy_(eye)
            self.object.weight.copy_(eye)

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


def prop_text(s: str, p: str, o: str) -> str:
    return f"{s} {p} {o}"


def prepare(model, tokenizer, device):
    prepared = []
    for case in CASES:
        s = encode(model, tokenizer, case.subject, device)
        p = encode(model, tokenizer, case.predicate, device)
        o = encode(model, tokenizer, case.object, device)

        positive = F.normalize(
            encode(model, tokenizer, prop_text(case.subject, case.predicate, case.object), device),
            dim=0,
        )
        cf_subject = F.normalize(
            encode(model, tokenizer, prop_text(case.subject_cf, case.predicate, case.object), device),
            dim=0,
        )
        cf_predicate = F.normalize(
            encode(model, tokenizer, prop_text(case.subject, case.predicate_cf, case.object), device),
            dim=0,
        )
        cf_object = F.normalize(
            encode(model, tokenizer, prop_text(case.subject, case.predicate, case.object_cf), device),
            dim=0,
        )

        prepared.append(
            {
                "case": case,
                "s": s,
                "p": p,
                "o": o,
                "positive": positive,
                "counterfactuals": [cf_subject, cf_predicate, cf_object],
            }
        )
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
        rows.append((item["case"].name, pos, *cfs, best_cf, margin))
    return rows


def summarize(rows):
    margins = [r[-1] for r in rows]
    return {
        "mean_margin": sum(margins) / len(margins),
        "positive_cases": sum(1 for m in margins if m > 0.0),
        "min_margin": min(margins),
    }


def train_one_seed(dim, prepared, device, seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    role_model = StructuralRoleProjection(dim).to(device)
    optimizer = torch.optim.Adam(role_model.parameters(), lr=LR)

    before = evaluate(role_model, prepared)

    for _ in range(EPOCHS):
        optimizer.zero_grad()
        losses = []

        for item in prepared:
            composed = role_model(item["s"], item["p"], item["o"])
            pos = F.cosine_similarity(composed, item["positive"], dim=0)

            for cf in item["counterfactuals"]:
                neg = F.cosine_similarity(composed, cf, dim=0)
                losses.append(F.relu(TARGET_MARGIN - pos + neg))

        loss = torch.stack(losses).mean()
        loss.backward()
        optimizer.step()

    after = evaluate(role_model, prepared)
    return role_model, before, after


def print_rows(title, rows):
    print(title)
    print("-" * 100)
    print(
        f"{'Case':<18} {'Positive':>9} {'CF-subj':>9} {'CF-pred':>9} "
        f"{'CF-obj':>9} {'CF-best':>9} {'Margin':>9}"
    )
    print("-" * 100)
    for name, pos, cfs, cfp, cfo, best_cf, margin in rows:
        print(
            f"{name:<18} {pos:9.6f} {cfs:9.6f} {cfp:9.6f} "
            f"{cfo:9.6f} {best_cf:9.6f} {margin:+9.6f}"
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

    prepared = prepare(model, tokenizer, device)
    dim = len(prepared[0]["s"])

    print("=" * 100)
    print(" LLM_SEM v0.4.1 Structural Role Encoding / Projection")
    print("=" * 100)
    print("Device          :", device)
    if device.type == "cuda":
        print("GPU             :", torch.cuda.get_device_name(0))
    print("Checkpoint loss :", checkpoint.get("loss"))
    print("Vector dimension:", dim)
    print("Cases           :", len(CASES))
    print("Seeds           :", SEEDS)
    print("Epochs          :", EPOCHS)
    print("Target margin   :", TARGET_MARGIN)
    print()

    results = []
    best = None

    for seed in SEEDS:
        role_model, before, after = train_one_seed(dim, prepared, device, seed)
        sb = summarize(before)
        sa = summarize(after)

        results.append((seed, sb, sa))

        score = (sa["positive_cases"], sa["mean_margin"], sa["min_margin"])
        if best is None or score > best[0]:
            best = (score, seed, role_model, before, after)

        print(
            f"seed={seed:<2} "
            f"before={sb['positive_cases']}/4 {sb['mean_margin']:+.6f} "
            f"after={sa['positive_cases']}/4 {sa['mean_margin']:+.6f} "
            f"min={sa['min_margin']:+.6f}"
        )

    assert best is not None
    _, best_seed, best_model, before_rows, after_rows = best

    print()
    print(f"Best seed: {best_seed}")
    print()
    print_rows("Before projection", before_rows)
    print_rows("After projection", after_rows)

    before_summary = summarize(before_rows)
    after_summary = summarize(after_rows)

    print("Summary")
    print("-" * 100)
    print(
        f"Positive-margin cases : "
        f"{before_summary['positive_cases']}/4 -> {after_summary['positive_cases']}/4"
    )
    print(
        f"Mean structural margin: "
        f"{before_summary['mean_margin']:+.6f} -> {after_summary['mean_margin']:+.6f}"
    )
    print(
        f"Minimum margin        : "
        f"{before_summary['min_margin']:+.6f} -> {after_summary['min_margin']:+.6f}"
    )

    checkpoint_out = {
        "version": "v0.4.1",
        "dimension": dim,
        "best_seed": best_seed,
        "epochs": EPOCHS,
        "lr": LR,
        "target_margin": TARGET_MARGIN,
        "state_dict": best_model.state_dict(),
        "before": before_summary,
        "after": after_summary,
        "cases": [c.__dict__ for c in CASES],
    }
    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint_out, OUT)

    print()
    print("Saved:", OUT)


if __name__ == "__main__":
    main()
