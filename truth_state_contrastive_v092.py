# truth_state_contrastive_v092.py
#
# LLM_SEM v0.9.2
# Contrastive Truth-State Representation
#
# Frozen semantic encoder.
# Trainable truth projector creates a dedicated truth-state space:
#   semantic 64 -> truth 16
# A classifier operates on the projected space.
#
# Loss:
#   cross entropy
# + supervised contrastive center loss
# + inter-class separation loss
#
# This is supervised truth-state learning, not autonomous factual verification.

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from adaptive_semantic_learning import TRUTH_STATES
from model import LanguageModel
from tokenizer import Tokenizer


DEFAULT_MODEL = "model/model-sem-consolidation-v081.pt"
DEFAULT_TOKENIZER = "model/tokenizer.json"
DEFAULT_DATASET = "data/truth_state_v091.json"
DEFAULT_OUTPUT = "model/truth-state-contrastive-v092.pt"


class TruthProjector(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        projection_dim: int,
        num_classes: int,
    ):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, projection_dim),
        )
        self.classifier = nn.Linear(projection_dim, num_classes)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.projector(x), dim=-1)

    def forward(self, x: torch.Tensor):
        z = self.encode(x)
        return z, self.classifier(z)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LLM_SEM v0.9.2 Contrastive Truth-State Representation"
    )
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    p.add_argument("--dataset", default=DEFAULT_DATASET)
    p.add_argument("--output", default=DEFAULT_OUTPUT)
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--learning-rate", type=float, default=7e-4)
    p.add_argument("--hidden-dim", type=int, default=32)
    p.add_argument("--projection-dim", type=int, default=16)
    p.add_argument("--alpha", type=float, default=0.35)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--holdout-per-class", type=int, default=1)
    p.add_argument("--center-weight", type=float, default=1.0)
    p.add_argument("--separation-weight", type=float, default=0.5)
    p.add_argument("--separation-margin", type=float, default=0.20)
    p.add_argument("--multi-seed", type=int, default=10)
    p.add_argument("--allow-cpu", action="store_true")
    return p.parse_args()


def load_dataset(path: Path) -> list[dict]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for row in obj.get("samples", []):
        text = str(row.get("text", "")).strip()
        truth = str(row.get("truth_status", "")).upper().strip()
        if text and truth in TRUTH_STATES:
            rows.append({"text": text, "truth_status": truth})
    if not rows:
        raise RuntimeError("No valid truth-state samples found.")
    return rows


@torch.no_grad()
def encode_text(
    model: LanguageModel,
    tokenizer: Tokenizer,
    text: str,
    device: torch.device,
    alpha: float,
) -> torch.Tensor:
    ids = tokenizer.encode(text, add_bos=True, add_eos=True)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    vec = model.encode_semantic(
        x,
        pooling="hybrid",
        hybrid_alpha=alpha,
        normalize_hybrid=False,
    )[0]
    return F.normalize(vec, dim=0)


def stratified_split(rows, labels, holdout_per_class, seed):
    rng = random.Random(seed)
    by_label = defaultdict(list)
    for row in rows:
        by_label[row["truth_status"]].append(row)

    train, test = [], []
    for label in labels:
        items = list(by_label[label])
        if len(items) <= holdout_per_class:
            raise RuntimeError(
                f"Not enough samples for {label}: {len(items)}"
            )
        rng.shuffle(items)
        test.extend(items[:holdout_per_class])
        train.extend(items[holdout_per_class:])
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def build_matrix(
    rows,
    model,
    tokenizer,
    device,
    alpha,
    label_to_id,
):
    x = torch.stack([
        encode_text(model, tokenizer, row["text"], device, alpha)
        for row in rows
    ])
    y = torch.tensor(
        [label_to_id[row["truth_status"]] for row in rows],
        dtype=torch.long,
        device=device,
    )
    return x, y


def class_centers(z: torch.Tensor, y: torch.Tensor, num_classes: int):
    centers = []
    present = []
    for class_id in range(num_classes):
        mask = y == class_id
        if bool(mask.any()):
            center = F.normalize(z[mask].mean(dim=0), dim=0)
            centers.append(center)
            present.append(class_id)
    return torch.stack(centers), present


def center_loss(
    z: torch.Tensor,
    y: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    losses = []
    for class_id in range(num_classes):
        mask = y == class_id
        if not bool(mask.any()):
            continue
        center = F.normalize(z[mask].mean(dim=0), dim=0)
        sims = F.cosine_similarity(z[mask], center.unsqueeze(0), dim=-1)
        losses.append((1.0 - sims).mean())
    if not losses:
        return torch.tensor(0.0, device=z.device)
    return torch.stack(losses).mean()


def separation_loss(
    z: torch.Tensor,
    y: torch.Tensor,
    num_classes: int,
    margin: float,
) -> torch.Tensor:
    centers, _ = class_centers(z, y, num_classes)
    if centers.size(0) < 2:
        return torch.tensor(0.0, device=z.device)

    penalties = []
    for i in range(centers.size(0)):
        for j in range(i + 1, centers.size(0)):
            sim = F.cosine_similarity(centers[i], centers[j], dim=0)
            penalties.append(F.relu(sim - margin))
    return torch.stack(penalties).mean()


@torch.no_grad()
def evaluate(model, x, y, id_to_label):
    model.eval()
    z, logits = model(x)
    probs = F.softmax(logits, dim=-1)
    pred = probs.argmax(dim=-1)
    accuracy = float((pred == y).float().mean().item())

    centers, present = class_centers(z, y, len(id_to_label))
    within = []
    for center, class_id in zip(centers, present):
        mask = y == class_id
        sims = F.cosine_similarity(
            z[mask],
            center.unsqueeze(0),
            dim=-1,
        )
        within.extend(sims.tolist())

    between = []
    for i in range(centers.size(0)):
        for j in range(i + 1, centers.size(0)):
            between.append(
                float(
                    F.cosine_similarity(
                        centers[i],
                        centers[j],
                        dim=0,
                    ).item()
                )
            )

    details = []
    for i in range(x.size(0)):
        top_prob, top_id = probs[i].max(dim=0)
        details.append({
            "expected": id_to_label[int(y[i].item())],
            "predicted": id_to_label[int(top_id.item())],
            "confidence": float(top_prob.item()),
        })

    return {
        "accuracy": accuracy,
        "details": details,
        "within": sum(within) / len(within) if within else 0.0,
        "between": sum(between) / len(between) if between else 0.0,
    }


def train_one(
    *,
    seed,
    rows,
    labels,
    model,
    tokenizer,
    device,
    args,
):
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    label_to_id = {label: i for i, label in enumerate(labels)}
    id_to_label = {i: label for label, i in label_to_id.items()}

    train_rows, test_rows = stratified_split(
        rows,
        labels,
        max(1, args.holdout_per_class),
        seed,
    )
    x_train, y_train = build_matrix(
        train_rows,
        model,
        tokenizer,
        device,
        args.alpha,
        label_to_id,
    )
    x_test, y_test = build_matrix(
        test_rows,
        model,
        tokenizer,
        device,
        args.alpha,
        label_to_id,
    )

    head = TruthProjector(
        input_dim=model.d_model,
        hidden_dim=args.hidden_dim,
        projection_dim=args.projection_dim,
        num_classes=len(labels),
    ).to(device)

    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=args.learning_rate,
        weight_decay=0.01,
    )

    last = None
    for epoch in range(1, args.epochs + 1):
        head.train()
        optimizer.zero_grad(set_to_none=True)

        z, logits = head(x_train)
        ce = F.cross_entropy(logits, y_train)
        ctr = center_loss(z, y_train, len(labels))
        sep = separation_loss(
            z,
            y_train,
            len(labels),
            args.separation_margin,
        )
        loss = (
            ce
            + args.center_weight * ctr
            + args.separation_weight * sep
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        optimizer.step()

        last = (float(loss.item()), float(ce.item()), float(ctr.item()), float(sep.item()))

    train_eval = evaluate(head, x_train, y_train, id_to_label)
    test_eval = evaluate(head, x_test, y_test, id_to_label)

    return {
        "seed": seed,
        "head": head,
        "train_rows": train_rows,
        "test_rows": test_rows,
        "train_eval": train_eval,
        "test_eval": test_eval,
        "last_loss": last,
        "labels": labels,
    }


def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and not args.allow_cpu:
        raise RuntimeError("CUDA unavailable. Use --allow-cpu only for testing.")

    tokenizer = Tokenizer.load(args.tokenizer)
    model, checkpoint = LanguageModel.load_checkpoint(
        args.model,
        device=device,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    rows = load_dataset(Path(args.dataset))
    labels = sorted(TRUTH_STATES)

    seeds = list(range(args.seed, args.seed + max(1, args.multi_seed)))
    results = []

    print("=" * 100)
    print(" LLM_SEM v0.9.2 Contrastive Truth-State Representation")
    print("=" * 100)
    print("Device              :", device)
    if device.type == "cuda":
        print("GPU                 :", torch.cuda.get_device_name(0))
    print("Frozen checkpoint   :", args.model)
    print("Checkpoint loss     :", checkpoint.get("loss"))
    print("Semantic dim        :", model.d_model)
    print("Truth projection    :", f"{model.d_model}->{args.hidden_dim}->{args.projection_dim}")
    print("Truth classes       :", ", ".join(labels))
    print("Dataset samples     :", len(rows))
    print("Holdout/class       :", args.holdout_per_class)
    print("Seeds               :", len(seeds))
    print("Center weight       :", args.center_weight)
    print("Separation weight   :", args.separation_weight)
    print("Separation margin   :", args.separation_margin)
    print("Encoder trainable   : False")
    print()

    best = None
    for seed in seeds:
        result = train_one(
            seed=seed,
            rows=rows,
            labels=labels,
            model=model,
            tokenizer=tokenizer,
            device=device,
            args=args,
        )
        results.append(result)
        tr = result["train_eval"]
        te = result["test_eval"]
        loss, ce, ctr, sep = result["last_loss"]
        print(
            f"seed={seed:<3} "
            f"train={tr['accuracy']*100:6.2f}% "
            f"holdout={te['accuracy']*100:6.2f}% "
            f"within={te['within']:+.4f} "
            f"between={te['between']:+.4f} "
            f"loss={loss:.4f}"
        )
        if best is None or te["accuracy"] > best["test_eval"]["accuracy"]:
            best = result

    assert best is not None

    holdouts = [r["test_eval"]["accuracy"] for r in results]
    mean = sum(holdouts) / len(holdouts)
    variance = sum((x - mean) ** 2 for x in holdouts) / len(holdouts)
    std = variance ** 0.5

    print()
    print("Multi-seed summary")
    print("------------------")
    print("Holdout mean      :", f"{mean*100:.2f}%")
    print("Holdout std       :", f"{std*100:.2f} pp")
    print("Holdout min       :", f"{min(holdouts)*100:.2f}%")
    print("Holdout max       :", f"{max(holdouts)*100:.2f}%")
    print("Best seed         :", best["seed"])

    print()
    print("Best-seed holdout detail")
    print("------------------------")
    by_expected = Counter()
    by_correct = Counter()
    for row, source in zip(
        best["test_eval"]["details"],
        best["test_rows"],
    ):
        mark = "PASS" if row["expected"] == row["predicted"] else "FAIL"
        print(
            f"[{mark}] expected={row['expected']:<11} "
            f"predicted={row['predicted']:<11} "
            f"conf={row['confidence']:.3f} "
            f"text={source['text']!r}"
        )
        by_expected[row["expected"]] += 1
        by_correct[row["expected"]] += int(
            row["expected"] == row["predicted"]
        )

    print()
    print("Per-class best-seed holdout")
    print("---------------------------")
    for label in labels:
        print(
            f"{label:<11}: "
            f"{by_correct[label]}/{by_expected[label]}"
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "llm-sem-truth-state-contrastive-v092",
            "base_model": args.model,
            "base_checkpoint_loss": checkpoint.get("loss"),
            "alpha": args.alpha,
            "input_dim": model.d_model,
            "hidden_dim": args.hidden_dim,
            "projection_dim": args.projection_dim,
            "labels": labels,
            "state_dict": best["head"].state_dict(),
            "best_seed": best["seed"],
            "holdout_mean": mean,
            "holdout_std": std,
            "best_holdout_accuracy": best["test_eval"]["accuracy"],
            "center_weight": args.center_weight,
            "separation_weight": args.separation_weight,
            "separation_margin": args.separation_margin,
        },
        output,
    )

    print()
    print("Saved projection  :", output)
    print()
    print(
        "Interpretation: v0.9.2 learns a dedicated truth-state space on top "
        "of the frozen semantic representation. Improvement over v0.9.1 "
        "would indicate that truth-state supervision benefits from an "
        "explicit geometry rather than a plain classifier alone."
    )


if __name__ == "__main__":
    main()
