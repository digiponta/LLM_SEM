# surface_generation_repair_v01019.py
#
# LLM_SEM v0.10.19
# Repair surface realization for already-internalized knowledge while
# preserving semantic representations and concept balance.

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from model import LanguageModel
from semantic_eval import load_benchmark
from semantic_guided_answer_finetune_v097 import (
    answer_lm_loss,
    generation_similarity,
    load_dataset,
    row_concept,
    semantic_vector,
    semantic_vector_grad,
)
from tokenizer import Tokenizer


def parse_args():
    p = argparse.ArgumentParser(
        description="LLM_SEM v0.10.19 Surface Generation Repair"
    )
    p.add_argument("--model", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--dataset", default="data/semantic_sleep_qa_v0106.json")
    p.add_argument("--benchmark", default="my_benchmark.csv")
    p.add_argument("--tokenizer", default="model/tokenizer.json")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--learning-rate", type=float, default=2e-6)
    p.add_argument("--min-repair-sim", type=float, default=0.70)
    p.add_argument("--target-sim", type=float, default=0.95)
    p.add_argument("--preserve-weight", type=float, default=5.0)
    p.add_argument("--alpha", type=float, default=0.35)
    p.add_argument("--allow-cpu", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and not args.allow_cpu:
        raise RuntimeError("CUDA unavailable. Use --allow-cpu only for testing.")

    tokenizer = Tokenizer.load(args.tokenizer)
    rows = [
        r for r in load_dataset(Path(args.dataset))
        if bool(r.get("must_train", False))
    ]
    benchmark = load_benchmark(args.benchmark)

    model, ckpt = LanguageModel.load_checkpoint(args.model, device=device)
    teacher, _ = LanguageModel.load_checkpoint(args.model, device=device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    repair_rows = []
    print("=" * 96)
    print(" LLM_SEM v0.10.19 Surface Generation Repair")
    print("=" * 96)
    print("Source:", args.model)

    for row in rows:
        sim, generated = generation_similarity(model, tokenizer, row, device)
        if args.min_repair_sim <= sim < args.target_sim:
            repair_rows.append(row)
            print(
                f"TARGET concept={row_concept(row)!r} "
                f"query={row['query']!r} sim={sim:.6f}"
            )
            print("  generated:", generated)

    if not repair_rows:
        print("RESULT: SKIP (no surface-repair targets)")
        Path(args.output).write_bytes(Path(args.model).read_bytes())
        return

    for p in model.parameters():
        p.requires_grad = False
    for p in model.final_norm.parameters():
        p.requires_grad = True
    for p in model.lm_head.parameters():
        p.requires_grad = True

    params = [
        p for p in model.parameters() if p.requires_grad
    ]
    optimizer = torch.optim.AdamW(params, lr=args.learning_rate, weight_decay=0.01)

    teacher_vectors = {
        row.text: semantic_vector(
            teacher, tokenizer, row.text, device, args.alpha
        ).detach()
        for row in benchmark
    }

    # Equal contribution per concept.
    grouped = {}
    for row in repair_rows:
        grouped.setdefault(row_concept(row), []).append(row)

    last_loss = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        concept_losses = []
        for concept_rows in grouped.values():
            losses = [
                answer_lm_loss(model, tokenizer, row, device)
                for row in concept_rows
            ]
            concept_losses.append(torch.stack(losses).mean())
        qa_loss = torch.stack(concept_losses).mean()

        preserve = []
        for row in benchmark:
            sv = semantic_vector_grad(
                model, tokenizer, row.text, device, args.alpha
            )
            preserve.append(
                1.0 - F.cosine_similarity(
                    sv, teacher_vectors[row.text], dim=0
                )
            )
        preserve_loss = torch.stack(preserve).mean()
        total = qa_loss + args.preserve_weight * preserve_loss
        total.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        last_loss = float(total.item())

        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(
                f"Epoch {epoch:>3}/{args.epochs} "
                f"total={last_loss:.6f} "
                f"qa={float(qa_loss.item()):.6f} "
                f"preserve={float(preserve_loss.item()):.6f}"
            )

    model.save_checkpoint(
        args.output,
        optimizer=optimizer,
        epoch=args.epochs,
        loss=last_loss,
    )

    print()
    failures = 0
    for row in repair_rows:
        sim, generated = generation_similarity(model, tokenizer, row, device)
        ok = sim >= args.target_sim
        failures += int(not ok)
        print(
            f"[{'PASS' if ok else 'REVIEW'}] "
            f"{row['query']} sim={sim:.6f}"
        )
        print("  generated:", generated)

    print("Saved:", args.output)
    print("RESULT:", "PASS" if failures == 0 else "REVIEW")


if __name__ == "__main__":
    main()
