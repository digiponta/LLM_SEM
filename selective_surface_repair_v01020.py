# selective_surface_repair_v01020.py
#
# LLM_SEM v0.10.20
# Concept-by-concept optional surface repair with automatic rollback.

from __future__ import annotations

import argparse
import shutil
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
        description="LLM_SEM v0.10.20 Selective Surface Repair"
    )
    p.add_argument("--model", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--dataset", default="data/semantic_sleep_qa_v0106.json")
    p.add_argument("--benchmark", default="my_benchmark.csv")
    p.add_argument("--tokenizer", default="model/tokenizer.json")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--learning-rate", type=float, default=1e-6)
    p.add_argument("--min-repair-sim", type=float, default=0.70)
    p.add_argument("--target-sim", type=float, default=0.95)
    p.add_argument("--min-target-gain", type=float, default=0.01)
    p.add_argument("--max-other-drop", type=float, default=0.02)
    p.add_argument("--preserve-weight", type=float, default=8.0)
    p.add_argument("--alpha", type=float, default=0.35)
    p.add_argument("--allow-cpu", action="store_true")
    return p.parse_args()


@torch.no_grad()
def concept_scores(model, tokenizer, rows, device):
    grouped = {}
    for row in rows:
        grouped.setdefault(row_concept(row), []).append(row)

    out = {}
    for concept, concept_rows in grouped.items():
        vals = []
        for row in concept_rows:
            sim, _ = generation_similarity(model, tokenizer, row, device)
            vals.append(sim)
        out[concept] = sum(vals) / len(vals) if vals else 0.0
    return out


def train_one_concept(
    source_path: str,
    output_path: str,
    *,
    target_concept: str,
    target_rows: list[dict],
    tokenizer,
    benchmark,
    device,
    args,
):
    model, _ = LanguageModel.load_checkpoint(source_path, device=device)
    teacher, _ = LanguageModel.load_checkpoint(source_path, device=device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    for p in model.parameters():
        p.requires_grad = False
    for p in model.final_norm.parameters():
        p.requires_grad = True
    for p in model.lm_head.parameters():
        p.requires_grad = True

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        params,
        lr=args.learning_rate,
        weight_decay=0.01,
    )

    teacher_vectors = {
        row.text: semantic_vector(
            teacher, tokenizer, row.text, device, args.alpha
        ).detach()
        for row in benchmark
    }

    last_loss = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        qa_losses = [
            answer_lm_loss(model, tokenizer, row, device)
            for row in target_rows
        ]
        qa_loss = torch.stack(qa_losses).mean()

        preserve_losses = []
        for row in benchmark:
            vec = semantic_vector_grad(
                model, tokenizer, row.text, device, args.alpha
            )
            preserve_losses.append(
                1.0 - F.cosine_similarity(
                    vec,
                    teacher_vectors[row.text],
                    dim=0,
                )
            )
        preserve_loss = torch.stack(preserve_losses).mean()

        total = qa_loss + args.preserve_weight * preserve_loss
        total.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        last_loss = float(total.item())

        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(
                f"  [{target_concept}] epoch {epoch:>3}/{args.epochs} "
                f"total={last_loss:.6f} "
                f"qa={float(qa_loss.item()):.6f} "
                f"preserve={float(preserve_loss.item()):.6f}"
            )

    model.save_checkpoint(
        output_path,
        optimizer=optimizer,
        epoch=args.epochs,
        loss=last_loss,
    )


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and not args.allow_cpu:
        raise RuntimeError("CUDA unavailable. Use --allow-cpu only for testing.")

    tokenizer = Tokenizer.load(args.tokenizer)
    benchmark = load_benchmark(args.benchmark)
    rows = [
        row for row in load_dataset(Path(args.dataset))
        if bool(row.get("must_train", False))
    ]

    grouped = {}
    for row in rows:
        grouped.setdefault(row_concept(row), []).append(row)

    current = Path(args.model)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 104)
    print(" LLM_SEM v0.10.20 Selective Surface Repair")
    print("=" * 104)
    print("Source model       :", current)
    print("Concepts           :", len(grouped))
    print("Min repair sim     :", args.min_repair_sim)
    print("Target sim         :", args.target_sim)
    print("Min target gain    :", args.min_target_gain)
    print("Max other drop     :", args.max_other_drop)

    accepted = 0
    rejected = 0

    for concept, concept_rows in grouped.items():
        base_model, _ = LanguageModel.load_checkpoint(current, device=device)
        before = concept_scores(
            base_model,
            tokenizer,
            rows,
            device,
        )
        target_before = before.get(concept, 0.0)

        if not (args.min_repair_sim <= target_before < args.target_sim):
            print()
            print(
                f"SKIP concept={concept!r} "
                f"similarity={target_before:.6f}"
            )
            continue

        trial = output.with_name(
            f"{output.stem}.{concept}.trial{output.suffix}"
        )

        print()
        print(
            f"REPAIR concept={concept!r} "
            f"before={target_before:.6f}"
        )
        train_one_concept(
            str(current),
            str(trial),
            target_concept=concept,
            target_rows=concept_rows,
            tokenizer=tokenizer,
            benchmark=benchmark,
            device=device,
            args=args,
        )

        trial_model, _ = LanguageModel.load_checkpoint(
            str(trial),
            device=device,
        )
        after = concept_scores(
            trial_model,
            tokenizer,
            rows,
            device,
        )

        target_after = after.get(concept, 0.0)
        target_gain = target_after - target_before

        other_drops = []
        for other, before_score in before.items():
            if other == concept:
                continue
            other_drops.append(
                before_score - after.get(other, 0.0)
            )
        max_other_drop = max(other_drops, default=0.0)

        accept = (
            target_gain >= args.min_target_gain
            and max_other_drop <= args.max_other_drop
        )

        print(
            f"SELECT concept={concept!r} "
            f"target={target_before:.6f}->{target_after:.6f} "
            f"gain={target_gain:+.6f} "
            f"max_other_drop={max_other_drop:+.6f} "
            f"decision={'ACCEPT' if accept else 'REJECT'}"
        )

        if accept:
            shutil.copy2(trial, output)
            current = output
            accepted += 1
        else:
            rejected += 1

    if accepted == 0:
        shutil.copy2(args.model, output)
        print()
        print("RESULT: KEEP_SOURCE (no repair candidate improved safely)")
    else:
        if current.resolve() != output.resolve():
            shutil.copy2(current, output)
        print()
        print(
            f"RESULT: SELECTED accepted={accepted} rejected={rejected}"
        )

    print("Selected output:", output)


if __name__ == "__main__":
    main()
