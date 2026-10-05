#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LLM_SEM v0.15.7 Contrastive Decoder Crossing

Purpose
-------
Move newly internalized semantic knowledge across the decoder decision boundary
without broadly perturbing the Transformer representation.

Trainable:
    final_norm
    lm_head

Frozen:
    embedding
    Transformer blocks

Loss:
    positive canonical NLL
    + lambda_margin * pairwise contrastive margin loss
    + lambda_replay * replay NLL

This script NEVER promotes a checkpoint automatically. It only writes a
candidate checkpoint for subsequent v0.15.6.1 prompt-aware validation and
runtime retention testing.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from model import LanguageModel
from tokenizer import Tokenizer


TRAIN_CASES = [
    {
        "prompt": "量子センサー",
        "positive": "は、原子や電子などのミクロな「量子」の性質を利用する。",
        "negative": "は、量子状態を利用して情報を伝送する通信技術である。",
    },
    {
        "prompt": "量子センサー",
        "positive": "は、磁場、温度、加速度、時間などを極めて高い感度で測定する次世代の計測技術である。",
        "negative": "は、量子状態を利用して、理論上絶対に盗聴されない安全な通信を実現する技術である。",
    },
    {
        "prompt": "量子センサーとは",
        "positive": "、原子や電子などのミクロな「量子」の性質を利用する。",
        "negative": "、量子状態を利用して情報を伝送する通信技術である。",
    },
    {
        "prompt": "量子センサーとは",
        "positive": "、磁場、温度、加速度、時間などを極めて高い感度で測定する次世代の計測技術である。",
        "negative": "、量子状態を利用して、理論上絶対に盗聴されない安全な通信を実現する技術である。",
    },
    {
        "prompt": "量子センサーについて教えて",
        "positive": "。量子センサーは、原子や電子などのミクロな「量子」の性質を利用する。",
        "negative": "。量子通信は、量子状態を利用して情報を伝送する通信技術である。",
    },
    {
        "prompt": "量子センサーを説明して",
        "positive": "。量子センサーは、磁場、温度、加速度、時間などを極めて高い感度で測定する次世代の計測技術である。",
        "negative": "。量子通信は、量子状態を利用して、理論上絶対に盗聴されない安全な通信を実現する技術である。",
    },
]


REPLAY_CASES = [
    ("CPUとは", "、命令を実行する中央処理装置である。"),
    ("GPUとは", "、多数の演算を並列に処理するプロセッサである。"),
    ("AIとは", "、人工知能である。"),
    ("数学とは", "、数や構造、空間、変化などを扱う学問である。"),
    ("文学とは", "、言語による芸術表現である。"),
    ("量子通信とは", "、量子状態を利用して情報を伝送する通信技術である。"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LLM_SEM v0.15.7 Contrastive Decoder Crossing"
    )
    p.add_argument("--model", required=True)
    p.add_argument("--tokenizer", default="model/tokenizer.json")
    p.add_argument("--output", default="model/model-sem-decoder-v0157.candidate.pt")
    p.add_argument("--json-out", default="results/decoder_crossing_v0157.json")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--lr-final-norm", type=float, default=2.0e-6)
    p.add_argument("--lr-lm-head", type=float, default=7.5e-7)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--margin", type=float, default=0.15)
    p.add_argument("--lambda-margin", type=float, default=1.0)
    p.add_argument("--lambda-replay", type=float, default=0.35)
    p.add_argument("--clip-grad", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--report-every", type=int, default=5)
    return p.parse_args()


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(name)


def encode_prompt(tokenizer: Tokenizer, text: str) -> List[int]:
    return tokenizer.encode(text, add_bos=True, add_eos=False)


def encode_continuation(tokenizer: Tokenizer, text: str) -> List[int]:
    return tokenizer.encode(text, add_bos=False, add_eos=True)


def sequence_mean_logp(
    model: LanguageModel,
    tokenizer: Tokenizer,
    prompt: str,
    continuation: str,
) -> torch.Tensor:
    prefix = encode_prompt(tokenizer, prompt)
    targets = encode_continuation(tokenizer, continuation)

    token_logps: List[torch.Tensor] = []
    device = next(model.parameters()).device

    for target in targets:
        context = prefix[-model.context_length:]
        x = torch.tensor([context], dtype=torch.long, device=device)
        logits = model(x)[0, -1, :]
        log_probs = F.log_softmax(logits, dim=-1)
        token_logps.append(log_probs[int(target)])
        prefix.append(int(target))

    return torch.stack(token_logps).mean()


@torch.no_grad()
def evaluate(
    model: LanguageModel,
    tokenizer: Tokenizer,
) -> Dict[str, object]:
    model.eval()

    rows = []
    for case in TRAIN_CASES:
        pos = float(
            sequence_mean_logp(
                model, tokenizer, case["prompt"], case["positive"]
            ).item()
        )
        neg = float(
            sequence_mean_logp(
                model, tokenizer, case["prompt"], case["negative"]
            ).item()
        )
        rows.append({
            "prompt": case["prompt"],
            "positive_mean_logp": pos,
            "negative_mean_logp": neg,
            "margin": pos - neg,
        })

    margins = [x["margin"] for x in rows]
    return {
        "mean_margin": sum(margins) / len(margins),
        "min_margin": min(margins),
        "positive_count": sum(x > 0.0 for x in margins),
        "case_count": len(rows),
        "rows": rows,
    }


def freeze_backbone(model: LanguageModel) -> None:
    for param in model.parameters():
        param.requires_grad = False

    for param in model.final_norm.parameters():
        param.requires_grad = True

    for param in model.lm_head.parameters():
        param.requires_grad = True


def trainable_counts(model: LanguageModel) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    return total, trainable


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = choose_device(args.device)

    if not Path(args.model).exists():
        raise FileNotFoundError(args.model)
    if not Path(args.tokenizer).exists():
        raise FileNotFoundError(args.tokenizer)

    tokenizer = Tokenizer.load(args.tokenizer)
    model, checkpoint = LanguageModel.load_checkpoint(
        args.model, device=device
    )

    if model.vocab_size != tokenizer.vocab_size:
        raise ValueError("model/tokenizer vocabulary mismatch")

    freeze_backbone(model)
    total_params, trainable_params = trainable_counts(model)

    optimizer = torch.optim.AdamW(
        [
            {
                "params": list(model.final_norm.parameters()),
                "lr": args.lr_final_norm,
            },
            {
                "params": list(model.lm_head.parameters()),
                "lr": args.lr_lm_head,
            },
        ],
        weight_decay=args.weight_decay,
    )

    baseline = evaluate(model, tokenizer)

    print("=" * 108)
    print(" LLM_SEM v0.15.7 Contrastive Decoder Crossing")
    print("=" * 108)
    print("Device                :", device)
    if device.type == "cuda":
        print("GPU                   :", torch.cuda.get_device_name(0))
    print("Source checkpoint     :", args.model)
    print("Source loss           :", checkpoint.get("loss"))
    print("Tokenizer             :", args.tokenizer)
    print("Trainable modules     : final_norm + lm_head")
    print("Frozen modules        : embedding + Transformer blocks")
    print("Total parameters      :", total_params)
    print("Trainable parameters  :", trainable_params)
    print("LR final_norm         :", args.lr_final_norm)
    print("LR lm_head            :", args.lr_lm_head)
    print("Target margin         :", args.margin)
    print("Lambda margin         :", args.lambda_margin)
    print("Lambda replay         :", args.lambda_replay)
    print("Epochs                :", args.epochs)
    print()
    print("Baseline mean margin  : %+0.6f" % baseline["mean_margin"])
    print("Baseline min margin   : %+0.6f" % baseline["min_margin"])
    print(
        "Baseline positive     : %d/%d"
        % (baseline["positive_count"], baseline["case_count"])
    )
    print()

    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        positive_losses = []
        margin_losses = []

        order = list(range(len(TRAIN_CASES)))
        random.shuffle(order)

        for idx in order:
            case = TRAIN_CASES[idx]
            pos_score = sequence_mean_logp(
                model,
                tokenizer,
                case["prompt"],
                case["positive"],
            )
            neg_score = sequence_mean_logp(
                model,
                tokenizer,
                case["prompt"],
                case["negative"],
            )

            positive_losses.append(-pos_score)
            margin_losses.append(
                F.relu(args.margin - pos_score + neg_score)
            )

        replay_losses = []
        for prompt, continuation in REPLAY_CASES:
            replay_score = sequence_mean_logp(
                model,
                tokenizer,
                prompt,
                continuation,
            )
            replay_losses.append(-replay_score)

        positive_loss = torch.stack(positive_losses).mean()
        margin_loss = torch.stack(margin_losses).mean()
        replay_loss = torch.stack(replay_losses).mean()

        total_loss = (
            positive_loss
            + args.lambda_margin * margin_loss
            + args.lambda_replay * replay_loss
        )

        total_loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            max_norm=args.clip_grad,
        )
        optimizer.step()

        if (
            epoch == 1
            or epoch == args.epochs
            or epoch % args.report_every == 0
        ):
            metrics = evaluate(model, tokenizer)
            row = {
                "epoch": epoch,
                "total_loss": float(total_loss.detach().item()),
                "positive_loss": float(positive_loss.detach().item()),
                "margin_loss": float(margin_loss.detach().item()),
                "replay_loss": float(replay_loss.detach().item()),
                "grad_norm": float(grad_norm),
                "mean_margin": float(metrics["mean_margin"]),
                "min_margin": float(metrics["min_margin"]),
                "positive_count": int(metrics["positive_count"]),
            }
            history.append(row)

            print(
                "epoch=%3d total=%.6f pos=%.6f margin=%.6f "
                "replay=%.6f grad=%.6f mean_margin=%+.6f "
                "min_margin=%+.6f positive=%d/%d"
                % (
                    epoch,
                    row["total_loss"],
                    row["positive_loss"],
                    row["margin_loss"],
                    row["replay_loss"],
                    row["grad_norm"],
                    row["mean_margin"],
                    row["min_margin"],
                    row["positive_count"],
                    len(TRAIN_CASES),
                )
            )

    final_metrics = evaluate(model, tokenizer)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model.save_checkpoint(
        str(output_path),
        optimizer=None,
        epoch=args.epochs,
        loss=float(history[-1]["total_loss"]) if history else None,
    )

    result = (
        "CANDIDATE_BOUNDARY_CROSSED"
        if final_metrics["positive_count"] == len(TRAIN_CASES)
        and final_metrics["min_margin"] > 0.0
        else "CANDIDATE_NOT_CROSSED"
    )

    print()
    print("=" * 108)
    print(" FINAL CANDIDATE SUMMARY")
    print("=" * 108)
    print("Baseline mean margin :", "%+.6f" % baseline["mean_margin"])
    print("Final mean margin    :", "%+.6f" % final_metrics["mean_margin"])
    print("Margin gain          :", "%+.6f" % (
        final_metrics["mean_margin"] - baseline["mean_margin"]
    ))
    print("Final min margin     :", "%+.6f" % final_metrics["min_margin"])
    print(
        "Positive margins     : %d/%d"
        % (final_metrics["positive_count"], final_metrics["case_count"])
    )
    print("RESULT               :", result)
    print("Candidate checkpoint :", output_path)
    print("NOTE                 : candidate only; not promoted")
    print()

    report = {
        "version": "v0.15.7",
        "experiment": "contrastive_decoder_crossing",
        "source_checkpoint": args.model,
        "candidate_checkpoint": str(output_path),
        "tokenizer": args.tokenizer,
        "config": {
            "epochs": args.epochs,
            "lr_final_norm": args.lr_final_norm,
            "lr_lm_head": args.lr_lm_head,
            "weight_decay": args.weight_decay,
            "margin": args.margin,
            "lambda_margin": args.lambda_margin,
            "lambda_replay": args.lambda_replay,
            "clip_grad": args.clip_grad,
            "seed": args.seed,
        },
        "baseline": baseline,
        "final": final_metrics,
        "history": history,
        "result": result,
        "promotion_policy": {
            "automatic_promotion": False,
            "next_validation": [
                "run_decoder_boundary_diagnostic_v01561.py",
                "runtime answer retention / protected-known validation",
            ],
        },
    }

    json_path = Path(args.json_out)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("Summary JSON         :", json_path)

    return 0 if result == "CANDIDATE_BOUNDARY_CROSSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
