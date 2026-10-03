# truth_state_projection_v091.py
#
# LLM_SEM v0.9.1
# Truth-State Projection
#
# Frozen semantic encoder + trainable projection head.
# The head predicts:
#   TRUE / FALSE / UNVERIFIED / CONTESTED / OUTDATED
#
# Important:
# This is supervised truth-state classification. It does not independently
# discover objective truth from text alone.

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
DEFAULT_OUTPUT = "model/truth-state-projection-v091.pt"


class TruthStateHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LLM_SEM v0.9.1 Truth-State Projection"
    )
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    p.add_argument("--dataset", default=DEFAULT_DATASET)
    p.add_argument("--output", default=DEFAULT_OUTPUT)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--hidden-dim", type=int, default=32)
    p.add_argument("--alpha", type=float, default=0.35)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--holdout-per-class", type=int, default=1)
    p.add_argument("--allow-cpu", action="store_true")
    return p.parse_args()


def load_dataset(path: Path) -> list[dict]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    rows = list(obj.get("samples", []))
    out = []
    for row in rows:
        text = str(row.get("text", "")).strip()
        truth = str(row.get("truth_status", "")).upper().strip()
        if text and truth in TRUTH_STATES:
            out.append({"text": text, "truth_status": truth})
    if not out:
        raise RuntimeError("No valid truth-state samples found.")
    return out


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


def stratified_split(
    rows: list[dict],
    labels: list[str],
    holdout_per_class: int,
    seed: int,
):
    rng = random.Random(seed)
    by_label = defaultdict(list)
    for row in rows:
        by_label[row["truth_status"]].append(row)

    train = []
    test = []
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
    vectors = []
    targets = []
    for row in rows:
        vectors.append(
            encode_text(
                model,
                tokenizer,
                row["text"],
                device,
                alpha,
            )
        )
        targets.append(label_to_id[row["truth_status"]])
    x = torch.stack(vectors)
    y = torch.tensor(targets, dtype=torch.long, device=device)
    return x, y


@torch.no_grad()
def evaluate(head, x, y, id_to_label):
    head.eval()
    logits = head(x)
    probs = F.softmax(logits, dim=-1)
    pred = probs.argmax(dim=-1)
    accuracy = float((pred == y).float().mean().item())
    rows = []
    for i in range(x.size(0)):
        top_prob, top_id = probs[i].max(dim=0)
        rows.append({
            "expected": id_to_label[int(y[i].item())],
            "predicted": id_to_label[int(top_id.item())],
            "confidence": float(top_prob.item()),
        })
    return accuracy, rows


def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and not args.allow_cpu:
        raise RuntimeError("CUDA unavailable. Use --allow-cpu only for testing.")

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

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
    label_to_id = {label: i for i, label in enumerate(labels)}
    id_to_label = {i: label for label, i in label_to_id.items()}

    train_rows, test_rows = stratified_split(
        rows,
        labels,
        max(1, args.holdout_per_class),
        args.seed,
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

    head = TruthStateHead(
        input_dim=model.d_model,
        hidden_dim=args.hidden_dim,
        num_classes=len(labels),
    ).to(device)

    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=args.learning_rate,
        weight_decay=0.01,
    )

    print("=" * 92)
    print(" LLM_SEM v0.9.1 Truth-State Projection")
    print("=" * 92)
    print("Device            :", device)
    if device.type == "cuda":
        print("GPU               :", torch.cuda.get_device_name(0))
    print("Frozen checkpoint :", args.model)
    print("Checkpoint loss   :", checkpoint.get("loss"))
    print("Semantic dim      :", model.d_model)
    print("Truth classes     :", ", ".join(labels))
    print("Dataset samples   :", len(rows))
    print("Train samples     :", len(train_rows))
    print("Holdout samples   :", len(test_rows))
    print("Head              :", f"{model.d_model}->{args.hidden_dim}->{len(labels)}")
    print("Encoder trainable : False")
    print("Head trainable    : True")
    print()

    for epoch in range(1, args.epochs + 1):
        head.train()
        optimizer.zero_grad(set_to_none=True)
        logits = head(x_train)
        loss = F.cross_entropy(logits, y_train)
        loss.backward()
        optimizer.step()

        if epoch == 1 or epoch % 50 == 0 or epoch == args.epochs:
            train_acc, _ = evaluate(head, x_train, y_train, id_to_label)
            test_acc, _ = evaluate(head, x_test, y_test, id_to_label)
            print(
                f"Epoch {epoch:>3}/{args.epochs} "
                f"loss={float(loss.item()):.6f} "
                f"train={train_acc*100:6.2f}% "
                f"holdout={test_acc*100:6.2f}%"
            )

    train_acc, _ = evaluate(head, x_train, y_train, id_to_label)
    test_acc, test_detail = evaluate(head, x_test, y_test, id_to_label)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "llm-sem-truth-state-projection-v091",
            "base_model": args.model,
            "base_checkpoint_loss": checkpoint.get("loss"),
            "alpha": args.alpha,
            "input_dim": model.d_model,
            "hidden_dim": args.hidden_dim,
            "labels": labels,
            "state_dict": head.state_dict(),
            "train_accuracy": train_acc,
            "holdout_accuracy": test_acc,
            "seed": args.seed,
        },
        output,
    )

    print()
    print("Holdout detail")
    print("--------------")
    for row, source in zip(test_detail, test_rows):
        mark = "PASS" if row["expected"] == row["predicted"] else "FAIL"
        print(
            f"[{mark}] expected={row['expected']:<11} "
            f"predicted={row['predicted']:<11} "
            f"conf={row['confidence']:.3f} "
            f"text={source['text']!r}"
        )

    by_expected = Counter()
    by_correct = Counter()
    for row in test_detail:
        by_expected[row["expected"]] += 1
        by_correct[row["expected"]] += int(
            row["expected"] == row["predicted"]
        )

    print()
    print("Per-class holdout")
    print("-----------------")
    for label in labels:
        print(
            f"{label:<11}: "
            f"{by_correct[label]}/{by_expected[label]}"
        )

    print()
    print("Train accuracy   :", f"{train_acc*100:.2f}%")
    print("Holdout accuracy :", f"{test_acc*100:.2f}%")
    print("Saved projection :", output)
    print()
    print(
        "Interpretation: the frozen Semantic Vector can now feed a separate "
        "truth-state classifier. This does not prove factual truth discovery; "
        "it measures supervised truth-state separability/generalization."
    )


if __name__ == "__main__":
    main()
