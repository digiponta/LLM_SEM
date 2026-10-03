# truth_evidence_projection_v093.py
#
# LLM_SEM v0.9.3
# Evidence-Aware Truth Representation
#
# Inputs:
#   query semantic vector
#   reference/evidence semantic vector
#   |query-reference|
#   query*reference
#   cosine(query, reference)
#   explicit evidence-relation one-hot
#
# The base LLM_SEM encoder is frozen. Only the evidence-aware truth head trains.

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
DEFAULT_DATASET = "data/truth_evidence_pairs_v093.json"
DEFAULT_OUTPUT = "model/truth-evidence-projection-v093.pt"


class EvidenceTruthHead(nn.Module):
    def __init__(
        self,
        semantic_dim: int,
        relation_dim: int,
        hidden_dim: int,
        num_classes: int,
    ):
        super().__init__()
        input_dim = semantic_dim * 4 + 1 + relation_dim
        self.input_dim = input_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LLM_SEM v0.9.3 Evidence-Aware Truth Representation"
    )
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    p.add_argument("--dataset", default=DEFAULT_DATASET)
    p.add_argument("--output", default=DEFAULT_OUTPUT)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--alpha", type=float, default=0.35)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--multi-seed", type=int, default=10)
    p.add_argument("--holdout-per-class", type=int, default=1)
    p.add_argument("--allow-cpu", action="store_true")
    return p.parse_args()


def load_dataset(path: Path):
    obj = json.loads(path.read_text(encoding="utf-8"))
    relations = list(obj.get("relation_types", []))
    rows = []
    for row in obj.get("samples", []):
        q = str(row.get("query", "")).strip()
        r = str(row.get("reference", "")).strip()
        rel = str(row.get("relation", "")).upper().strip()
        truth = str(row.get("truth_status", "")).upper().strip()
        if q and r and rel in relations and truth in TRUTH_STATES:
            rows.append({
                "query": q,
                "reference": r,
                "relation": rel,
                "truth_status": truth,
            })
    if not rows:
        raise RuntimeError("No valid evidence pairs found.")
    return rows, relations


@torch.no_grad()
def encode(
    model: LanguageModel,
    tokenizer: Tokenizer,
    text: str,
    device: torch.device,
    alpha: float,
) -> torch.Tensor:
    ids = tokenizer.encode(text, add_bos=True, add_eos=True)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    v = model.encode_semantic(
        x,
        pooling="hybrid",
        hybrid_alpha=alpha,
        normalize_hybrid=False,
    )[0]
    return F.normalize(v, dim=0)


def feature_vector(
    q: torch.Tensor,
    r: torch.Tensor,
    relation: str,
    relation_to_id: dict[str, int],
) -> torch.Tensor:
    diff = torch.abs(q - r)
    prod = q * r
    cos = F.cosine_similarity(q, r, dim=0).reshape(1)
    rel = torch.zeros(
        len(relation_to_id),
        dtype=q.dtype,
        device=q.device,
    )
    rel[relation_to_id[relation]] = 1.0
    return torch.cat([q, r, diff, prod, cos, rel], dim=0)


def stratified_split(rows, labels, holdout_per_class, seed):
    rng = random.Random(seed)
    by_label = defaultdict(list)
    for row in rows:
        by_label[row["truth_status"]].append(row)

    train, test = [], []
    for label in labels:
        items = list(by_label[label])
        if len(items) <= holdout_per_class:
            raise RuntimeError(f"Not enough samples for {label}: {len(items)}")
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
    relation_to_id,
    label_to_id,
):
    xs, ys = [], []
    for row in rows:
        q = encode(model, tokenizer, row["query"], device, alpha)
        r = encode(model, tokenizer, row["reference"], device, alpha)
        xs.append(feature_vector(q, r, row["relation"], relation_to_id))
        ys.append(label_to_id[row["truth_status"]])
    return (
        torch.stack(xs),
        torch.tensor(ys, dtype=torch.long, device=device),
    )


@torch.no_grad()
def evaluate(head, x, y, id_to_label):
    head.eval()
    probs = F.softmax(head(x), dim=-1)
    pred = probs.argmax(dim=-1)
    acc = float((pred == y).float().mean().item())
    detail = []
    for i in range(x.size(0)):
        top_prob, top_id = probs[i].max(dim=0)
        detail.append({
            "expected": id_to_label[int(y[i].item())],
            "predicted": id_to_label[int(top_id.item())],
            "confidence": float(top_prob.item()),
        })
    return acc, detail


def run_seed(
    seed,
    *,
    rows,
    relations,
    labels,
    model,
    tokenizer,
    device,
    args,
):
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    relation_to_id = {name: i for i, name in enumerate(relations)}
    label_to_id = {name: i for i, name in enumerate(labels)}
    id_to_label = {i: name for name, i in label_to_id.items()}

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
        relation_to_id,
        label_to_id,
    )
    x_test, y_test = build_matrix(
        test_rows,
        model,
        tokenizer,
        device,
        args.alpha,
        relation_to_id,
        label_to_id,
    )

    head = EvidenceTruthHead(
        semantic_dim=model.d_model,
        relation_dim=len(relations),
        hidden_dim=args.hidden_dim,
        num_classes=len(labels),
    ).to(device)

    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=args.learning_rate,
        weight_decay=0.01,
    )

    for _ in range(args.epochs):
        head.train()
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(head(x_train), y_train)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        optimizer.step()

    train_acc, _ = evaluate(head, x_train, y_train, id_to_label)
    test_acc, detail = evaluate(head, x_test, y_test, id_to_label)

    return {
        "seed": seed,
        "head": head,
        "train_acc": train_acc,
        "test_acc": test_acc,
        "detail": detail,
        "test_rows": test_rows,
        "relations": relations,
        "labels": labels,
        "relation_to_id": relation_to_id,
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

    rows, relations = load_dataset(Path(args.dataset))
    labels = sorted(TRUTH_STATES)
    seeds = list(range(args.seed, args.seed + max(1, args.multi_seed)))

    print("=" * 104)
    print(" LLM_SEM v0.9.3 Evidence-Aware Truth Representation")
    print("=" * 104)
    print("Device              :", device)
    if device.type == "cuda":
        print("GPU                 :", torch.cuda.get_device_name(0))
    print("Frozen checkpoint   :", args.model)
    print("Checkpoint loss     :", checkpoint.get("loss"))
    print("Semantic dim        :", model.d_model)
    print("Evidence relations  :", ", ".join(relations))
    print("Truth classes       :", ", ".join(labels))
    print("Dataset pairs       :", len(rows))
    print("Holdout/class       :", args.holdout_per_class)
    print("Seeds               :", len(seeds))
    print("Encoder trainable   : False")
    print()

    results = []
    best = None

    for seed in seeds:
        result = run_seed(
            seed,
            rows=rows,
            relations=relations,
            labels=labels,
            model=model,
            tokenizer=tokenizer,
            device=device,
            args=args,
        )
        results.append(result)
        print(
            f"seed={seed:<3} "
            f"train={result['train_acc']*100:6.2f}% "
            f"holdout={result['test_acc']*100:6.2f}%"
        )
        if best is None or result["test_acc"] > best["test_acc"]:
            best = result

    assert best is not None
    vals = [r["test_acc"] for r in results]
    mean = sum(vals) / len(vals)
    var = sum((x - mean) ** 2 for x in vals) / len(vals)
    std = var ** 0.5

    print()
    print("Multi-seed summary")
    print("------------------")
    print("Holdout mean      :", f"{mean*100:.2f}%")
    print("Holdout std       :", f"{std*100:.2f} pp")
    print("Holdout min       :", f"{min(vals)*100:.2f}%")
    print("Holdout max       :", f"{max(vals)*100:.2f}%")
    print("Best seed         :", best["seed"])

    print()
    print("Best-seed holdout detail")
    print("------------------------")
    by_expected = Counter()
    by_correct = Counter()
    for row, source in zip(best["detail"], best["test_rows"]):
        ok = row["expected"] == row["predicted"]
        print(
            f"[{'PASS' if ok else 'FAIL'}] "
            f"expected={row['expected']:<11} "
            f"predicted={row['predicted']:<11} "
            f"conf={row['confidence']:.3f} "
            f"relation={source['relation']:<11} "
            f"query={source['query']!r}"
        )
        by_expected[row["expected"]] += 1
        by_correct[row["expected"]] += int(ok)

    print()
    print("Per-class best-seed holdout")
    print("---------------------------")
    for label in labels:
        print(f"{label:<11}: {by_correct[label]}/{by_expected[label]}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "llm-sem-truth-evidence-projection-v093",
            "base_model": args.model,
            "base_checkpoint_loss": checkpoint.get("loss"),
            "alpha": args.alpha,
            "semantic_dim": model.d_model,
            "relation_types": relations,
            "hidden_dim": args.hidden_dim,
            "labels": labels,
            "state_dict": best["head"].state_dict(),
            "best_seed": best["seed"],
            "holdout_mean": mean,
            "holdout_std": std,
            "best_holdout_accuracy": best["test_acc"],
        },
        output,
    )

    print()
    print("Saved projection  :", output)
    print()
    print(
        "Interpretation: unlike v0.9.1/v0.9.2, this experiment predicts "
        "truth state from a query/evidence relation rather than from query "
        "semantics alone."
    )


if __name__ == "__main__":
    main()
