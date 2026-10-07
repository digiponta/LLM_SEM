#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LLM_SEM v0.15.7.2 Aggressive Decoder Crossing Sweep

Focuses on the best v0.15.7.1 source (v0154) and sweeps higher decoder LRs.
Stops no training internally; every run remains a candidate-only experiment.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


LR_PAIRS = [
    (1.0e-4, 5.0e-5),
    (2.0e-4, 1.0e-4),
    (5.0e-4, 2.0e-4),
    (1.0e-3, 5.0e-4),
]


def parse_args():
    p = argparse.ArgumentParser(
        description="LLM_SEM v0.15.7.2 Aggressive Decoder Crossing Sweep"
    )
    p.add_argument(
        "--source",
        default="model/model-sem-diagnostic-v0154.pt",
    )
    p.add_argument("--tokenizer", default="model/tokenizer.json")
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--margin", type=float, default=0.15)
    p.add_argument("--lambda-margin", type=float, default=1.0)
    p.add_argument("--lambda-replay", type=float, default=0.35)
    p.add_argument("--clip-grad", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument(
        "--output-dir",
        default="model/v01572_sweep",
    )
    p.add_argument(
        "--results-dir",
        default="results/v01572_sweep",
    )
    p.add_argument(
        "--summary",
        default="results/decoder_crossing_aggressive_sweep_v01572.json",
    )
    return p.parse_args()


def tag_lr(x: float) -> str:
    return f"{x:.0e}".replace("+", "").replace("-", "m")


def main():
    args = parse_args()

    source = Path(args.source)
    tokenizer = Path(args.tokenizer)
    trainer = Path("run_contrastive_decoder_crossing_v0157.py")

    for path in (source, tokenizer, trainer):
        if not path.exists():
            raise FileNotFoundError(path)

    out_dir = Path(args.output_dir)
    res_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 120)
    print(" LLM_SEM v0.15.7.2 Aggressive Decoder Crossing Sweep")
    print("=" * 120)
    print("Source            :", source)
    print("Epochs / run      :", args.epochs)
    print("Target margin     :", args.margin)
    print("Lambda margin     :", args.lambda_margin)
    print("Lambda replay     :", args.lambda_replay)
    print("Gradient clip     :", args.clip_grad)
    print("LR pairs          :")
    for fn_lr, head_lr in LR_PAIRS:
        print(f"  final_norm={fn_lr:g}  lm_head={head_lr:g}")
    print()

    rows = []

    for i, (fn_lr, head_lr) in enumerate(LR_PAIRS, 1):
        label = f"fn{tag_lr(fn_lr)}_head{tag_lr(head_lr)}"
        candidate = out_dir / (
            f"model-sem-decoder-v01572-{label}.candidate.pt"
        )
        result_json = res_dir / f"decoder-v01572-{label}.json"

        cmd = [
            sys.executable,
            str(trainer),
            "--model", str(source),
            "--tokenizer", str(tokenizer),
            "--output", str(candidate),
            "--json-out", str(result_json),
            "--epochs", str(args.epochs),
            "--lr-final-norm", str(fn_lr),
            "--lr-lm-head", str(head_lr),
            "--margin", str(args.margin),
            "--lambda-margin", str(args.lambda_margin),
            "--lambda-replay", str(args.lambda_replay),
            "--clip-grad", str(args.clip_grad),
            "--seed", str(args.seed),
            "--device", str(args.device),
            "--report-every", "10",
        ]

        print("=" * 120)
        print(
            f"[RUN {i}/{len(LR_PAIRS)}] "
            f"final_norm_lr={fn_lr:g} lm_head_lr={head_lr:g}"
        )
        print("=" * 120)

        proc = subprocess.run(cmd)

        if not result_json.exists():
            rows.append({
                "lr_final_norm": fn_lr,
                "lr_lm_head": head_lr,
                "return_code": proc.returncode,
                "status": "NO_RESULT_JSON",
            })
            continue

        payload = json.loads(result_json.read_text(encoding="utf-8"))
        baseline = payload["baseline"]
        final = payload["final"]
        history = payload.get("history", [])

        replay_start = history[0]["replay_loss"] if history else None
        replay_end = history[-1]["replay_loss"] if history else None
        replay_delta = (
            replay_end - replay_start
            if replay_start is not None and replay_end is not None
            else None
        )

        row = {
            "lr_final_norm": fn_lr,
            "lr_lm_head": head_lr,
            "candidate": str(candidate),
            "result_json": str(result_json),
            "return_code": proc.returncode,
            "result": payload["result"],
            "baseline_mean_margin": baseline["mean_margin"],
            "final_mean_margin": final["mean_margin"],
            "margin_gain": final["mean_margin"] - baseline["mean_margin"],
            "final_min_margin": final["min_margin"],
            "positive_count": final["positive_count"],
            "case_count": final["case_count"],
            "replay_loss_start": replay_start,
            "replay_loss_end": replay_end,
            "replay_loss_delta": replay_delta,
        }
        rows.append(row)

        print(
            "[SUMMARY] positive=%d/%d mean=%+.6f min=%+.6f "
            "gain=%+.6f replay_delta=%s result=%s"
            % (
                row["positive_count"],
                row["case_count"],
                row["final_mean_margin"],
                row["final_min_margin"],
                row["margin_gain"],
                (
                    f"{row['replay_loss_delta']:+.6f}"
                    if row["replay_loss_delta"] is not None
                    else "n/a"
                ),
                row["result"],
            )
        )
        print()

    valid = [r for r in rows if "positive_count" in r]
    valid.sort(
        key=lambda r: (
            r["positive_count"],
            r["final_min_margin"],
            r["final_mean_margin"],
        ),
        reverse=True,
    )

    print("=" * 120)
    print(" GLOBAL AGGRESSIVE SWEEP RANKING")
    print("=" * 120)
    print(
        "Rank final_norm    lm_head      Positive   "
        "MinMargin    MeanMargin   Gain        ReplayDelta"
    )
    print("-" * 120)

    for rank, row in enumerate(valid, 1):
        rd = (
            f"{row['replay_loss_delta']:+.6f}"
            if row["replay_loss_delta"] is not None
            else "n/a"
        )
        print(
            f"{rank:4d} "
            f"{row['lr_final_norm']:11.2e} "
            f"{row['lr_lm_head']:11.2e} "
            f"{row['positive_count']:2d}/{row['case_count']:<2d}      "
            f"{row['final_min_margin']:+.6f} "
            f"{row['final_mean_margin']:+.6f} "
            f"{row['margin_gain']:+.6f} "
            f"{rd}"
        )

    best = valid[0] if valid else None

    if best is None:
        overall = "NO_VALID_RUN"
    elif (
        best["positive_count"] == best["case_count"]
        and best["final_min_margin"] > 0.0
    ):
        overall = "AGGRESSIVE_SWEEP_FOUND_CROSSED_CANDIDATE"
    else:
        overall = "AGGRESSIVE_SWEEP_NO_CROSSING"

    print()
    print("RESULT             :", overall)
    if best:
        print("Best final_norm LR :", best["lr_final_norm"])
        print("Best lm_head LR    :", best["lr_lm_head"])
        print("Best positive      :", f"{best['positive_count']}/{best['case_count']}")
        print("Best min margin    :", f"{best['final_min_margin']:+.6f}")
        print("Best mean margin   :", f"{best['final_mean_margin']:+.6f}")
        print("Best margin gain   :", f"{best['margin_gain']:+.6f}")
        print("Best candidate     :", best["candidate"])
        if best["replay_loss_delta"] is not None:
            print("Replay loss delta  :", f"{best['replay_loss_delta']:+.6f}")
        print("NOTE               : candidate only; not promoted")

    summary = {
        "version": "v0.15.7.2",
        "experiment": "aggressive_decoder_crossing_sweep",
        "source": str(source),
        "config": {
            "epochs": args.epochs,
            "margin": args.margin,
            "lambda_margin": args.lambda_margin,
            "lambda_replay": args.lambda_replay,
            "clip_grad": args.clip_grad,
            "seed": args.seed,
        },
        "lr_pairs": [
            {"lr_final_norm": a, "lr_lm_head": b}
            for a, b in LR_PAIRS
        ],
        "runs": rows,
        "ranking": valid,
        "best": best,
        "result": overall,
        "promotion_policy": {
            "automatic_promotion": False,
            "next_validation": [
                "run_decoder_boundary_diagnostic_v01561.py",
                "protected-known/runtime retention validation",
            ],
        },
    }

    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("Summary JSON       :", summary_path)

    return 0 if overall == "AGGRESSIVE_SWEEP_FOUND_CROSSED_CANDIDATE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
