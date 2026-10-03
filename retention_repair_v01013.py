# retention_repair_v01013.py
#
# LLM_SEM v0.10.13
# Targeted repair for consolidated semantic memories damaged by sleep QA learning.

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from adaptive_semantic_learning import load_semantic_memory_records
from model import LanguageModel
from semantic_eval import load_benchmark
from semantic_router import SemanticRouter
from tokenizer import Tokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LLM_SEM v0.10.13 retention repair")
    p.add_argument("--model", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--memory", default="data/semantic_memory.jsonl")
    p.add_argument("--tokenizer", default="model/tokenizer.json")
    p.add_argument("--benchmark", default="my_benchmark.csv")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--learning-rate", type=float, default=3e-6)
    p.add_argument("--target-weight", type=float, default=6.0)
    p.add_argument("--benchmark-weight", type=float, default=1.0)
    p.add_argument("--preserve-weight", type=float, default=6.0)
    p.add_argument("--consolidated-weight", type=float, default=3.0)
    p.add_argument("--alpha", type=float, default=0.35)
    p.add_argument("--temperature", type=float, default=0.06)
    p.add_argument("--allow-cpu", action="store_true")
    return p.parse_args()


def encode(model, tokenizer, text, device, alpha):
    ids = tokenizer.encode(text, add_bos=True, add_eos=True)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    return model.encode_semantic(
        x,
        pooling="hybrid",
        hybrid_alpha=alpha,
        normalize_hybrid=False,
    )[0]


def configure_trainable(model):
    for p in model.parameters():
        p.requires_grad = False
    for p in model.blocks[-1].parameters():
        p.requires_grad = True
    for p in model.final_norm.parameters():
        p.requires_grad = True
    return [p for p in model.parameters() if p.requires_grad]


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and not args.allow_cpu:
        raise RuntimeError("CUDA unavailable. Use --allow-cpu only for testing.")

    tokenizer = Tokenizer.load(args.tokenizer)
    benchmark = load_benchmark(args.benchmark)
    consolidated = [
        row for row in load_semantic_memory_records(Path(args.memory))
        if row.get("status") == "CONSOLIDATED"
    ]

    teacher, ckpt = LanguageModel.load_checkpoint(args.model, device=device)
    student, _ = LanguageModel.load_checkpoint(args.model, device=device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    router = SemanticRouter(teacher, tokenizer, alpha=args.alpha)
    router.fit(benchmark)

    damaged = []
    for row in consolidated:
        ranked = router.route(str(row["text"]))
        top = ranked[0].label if ranked else "(none)"
        if top != str(row["label"]):
            damaged.append(row)

    print("=" * 96)
    print(" LLM_SEM v0.10.13 Targeted Retention Repair")
    print("=" * 96)
    print("Device              :", device)
    print("Source checkpoint   :", args.model)
    print("Source loss         :", ckpt.get("loss"))
    print("Consolidated records:", len(consolidated))
    print("Damaged records     :", len(damaged))

    if not damaged:
        Path(args.output).write_bytes(Path(args.model).read_bytes())
        print("RESULT: NO_REPAIR_NEEDED")
        return

    benchmark_refs = []
    grouped = {}
    with torch.no_grad():
        for sample in benchmark:
            vec = encode(
                teacher, tokenizer, sample.text, device, args.alpha
            ).detach()
            benchmark_refs.append((sample.label, sample.text, vec))
            grouped.setdefault(sample.label, []).append(vec)

        consolidated_refs = [
            (
                str(row["label"]),
                str(row["text"]),
                encode(
                    teacher, tokenizer, str(row["text"]), device, args.alpha
                ).detach(),
            )
            for row in consolidated
        ]

    centroids = {
        label: F.normalize(torch.stack(vectors).mean(dim=0), dim=0)
        for label, vectors in grouped.items()
    }
    labels = sorted(centroids)
    label_to_index = {label: i for i, label in enumerate(labels)}

    def logits_for(vec):
        v = F.normalize(vec, dim=0)
        return torch.stack([
            F.cosine_similarity(v, centroids[label], dim=0)
            for label in labels
        ]) / args.temperature

    trainable = configure_trainable(student)
    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.learning_rate,
        weight_decay=0.01,
    )

    for idx, row in enumerate(damaged, 1):
        print(
            f"{idx:02d}. expected={row['label']} text={row['text']!r}"
        )

    last_total = None
    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad(set_to_none=True)

        target_losses = []
        for row in damaged:
            vec = encode(
                student, tokenizer, str(row["text"]), device, args.alpha
            )
            target = torch.tensor(
                [label_to_index[str(row["label"])]],
                dtype=torch.long,
                device=device,
            )
            target_losses.append(
                F.cross_entropy(logits_for(vec).unsqueeze(0), target)
            )

        bench_losses = []
        preserve_losses = []
        for expected, text, teacher_vec in benchmark_refs:
            vec = encode(student, tokenizer, text, device, args.alpha)
            target = torch.tensor(
                [label_to_index[expected]],
                dtype=torch.long,
                device=device,
            )
            bench_losses.append(
                F.cross_entropy(logits_for(vec).unsqueeze(0), target)
            )
            preserve_losses.append(
                1.0 - F.cosine_similarity(vec, teacher_vec, dim=0)
            )

        consolidated_losses = []
        for expected, text, teacher_vec in consolidated_refs:
            vec = encode(student, tokenizer, text, device, args.alpha)
            target = torch.tensor(
                [label_to_index[expected]],
                dtype=torch.long,
                device=device,
            )
            cls = F.cross_entropy(logits_for(vec).unsqueeze(0), target)
            preserve = 1.0 - F.cosine_similarity(vec, teacher_vec, dim=0)
            consolidated_losses.append(cls + preserve)

        target_loss = torch.stack(target_losses).mean()
        bench_loss = torch.stack(bench_losses).mean()
        preserve_loss = torch.stack(preserve_losses).mean()
        consolidated_loss = torch.stack(consolidated_losses).mean()

        total = (
            args.target_weight * target_loss
            + args.benchmark_weight * bench_loss
            + args.preserve_weight * preserve_loss
            + args.consolidated_weight * consolidated_loss
        )
        total.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        last_total = float(total.item())

        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(
                f"Epoch {epoch:>3}/{args.epochs} "
                f"total={float(total.item()):.6f} "
                f"target={float(target_loss.item()):.6f} "
                f"benchmark={float(bench_loss.item()):.6f} "
                f"preserve={float(preserve_loss.item()):.6f}"
            )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    student.save_checkpoint(
        str(out),
        optimizer=optimizer,
        epoch=args.epochs,
        loss=last_total,
    )
    print("Saved checkpoint:", out)
    print("RESULT: REPAIRED")


if __name__ == "__main__":
    main()
