#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LLM_SEM v0.15.7.1 Decoder Crossing LR Sweep

Runs v0.15.7 contrastive decoder crossing over multiple source checkpoints
and learning-rate pairs, then ranks candidates by:
  1) positive margin count
  2) minimum margin
  3) mean margin

No checkpoint is promoted automatically.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


DEFAULT_SOURCES = [
    "model/model-sem-diagnostic-v0154.pt",
    "model/model-sem-diagnostic-v0155.pt",
]

DEFAULT_LR_PAIRS = [
    (5.0e-6, 2.0e-6),
    (1.0e-5, 5.0e-6),
    (2.0e-5, 1.0e-5),
    (5.0e-5, 2.0e-5),
]


def parse_args():
    p = argparse.ArgumentParser(
        description="LLM_SEM v0.15.7.1 Decoder Crossing LR Sweep"
    )
    p.add_argument(
        "--source",
        action="append",
        dest="sources",
        help="Source checkpoint. Repeat for multiple sources.",
    )
    p.add_argument("--tokenizer", default="model/tokenizer.json")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--margin", type=float, default=0.15)
    p.add_argument("--lambda-margin", type=float, default=1.0)
    p.add_argument("--lambda-replay", type=float, default=0.35)
    p.add_argument("--clip-grad", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument(
        "--output-dir",
        default="model/v01571_sweep",
    )
    p.add_argument(
        "--results-dir",
        default="results/v01571_sweep",
    )
    p.add_argument(
        "--summary",
        default="results/decoder_crossing_lr_sweep_v01571.json",
    )
    return p.parse_args()


def source_tag(path: str) -> str:
    stem = Path(path).stem
    if stem.startswith("model-sem-diagnostic-"):
        return stem.replace("model-sem-diagnostic-", "")
    return stem


def main() -> int:
    args = parse_args()
    sources = args.sources or DEFAULT_SOURCES

    tokenizer = Path(args.tokenizer)
    if not tokenizer.exists():
        raise FileNotFoundError(tokenizer)

    script = Path("run_contrastive_decoder_crossing_v0157.py")
    if not script.exists():
        raise FileNotFoundError(script)

    output_dir = Path(args.output_dir)
    results_dir = Path(args.results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    print("=" * 120)
    print(" LLM_SEM v0.15.7.1 Decoder Crossing LR Sweep")
    print("=" * 120)
    print("Sources          :", ", ".join(sources))
    print("Epochs / run     :", args.epochs)
    print("Target margin    :", args.margin)
    print("Lambda margin    :", args.lambda_margin)
    print("Lambda replay    :", args.lambda_replay)
    print("LR pairs         :")
    for fn_lr, head_lr in DEFAULT_LR_PAIRS:
        print(
            "  final_norm=%g  lm_head=%g"
            % (fn_lr, head_lr)
        )
    print()

    run_index = 0

    for source in sources:
        source_path = Path(source)
        if not source_path.exists():
            print("[SKIP] missing source:", source)
            continue

        tag = source_tag(source)

        for fn_lr, head_lr in DEFAULT_LR_PAIRS:
            run_index += 1
            lr_tag = (
                f"fn{fn_lr:.0e}_head{head_lr:.0e}"
                .replace("+", "")
                .replace("-", "m")
            )

            candidate = output_dir / (
                f"model-sem-decoder-v01571-{tag}-{lr_tag}.candidate.pt"
            )
            result_json = results_dir / (
                f"decoder-v01571-{tag}-{lr_tag}.json"
            )

            cmd = [
                sys.executable,
                str(script),
                "--model", str(source_path),
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
            ]

            print("=" * 120)
            print(
                f"[RUN {run_index}] source={tag} "
                f"final_norm_lr={fn_lr:g} lm_head_lr={head_lr:g}"
            )
            print("=" * 120)

            proc = subprocess.run(cmd)

            if not result_json.exists():
                rows.append({
                    "source": str(source_path),
                    "source_tag": tag,
                    "lr_final_norm": fn_lr,
                    "lr_lm_head": head_lr,
                    "candidate": str(candidate),
                    "result_json": str(result_json),
                    "return_code": proc.returncode,
                    "status": "NO_RESULT_JSON",
                })
                continue

            payload = json.loads(
                result_json.read_text(encoding="utf-8")
            )
            baseline = payload["baseline"]
            final = payload["final"]

            row = {
                "source": str(source_path),
                "source_tag": tag,
                "lr_final_norm": fn_lr,
                "lr_lm_head": head_lr,
                "candidate": str(candidate),
                "result_json": str(result_json),
                "return_code": proc.returncode,
                "result": payload["result"],
                "baseline_mean_margin": baseline["mean_margin"],
                "final_mean_margin": final["mean_margin"],
                "margin_gain": (
                    final["mean_margin"] - baseline["mean_margin"]
                ),
                "final_min_margin": final["min_margin"],
                "positive_count": final["positive_count"],
                "case_count": final["case_count"],
            }
            rows.append(row)

            print(
                "[SUMMARY] positive=%d/%d mean=%+.6f "
                "min=%+.6f gain=%+.6f result=%s"
                % (
                    row["positive_count"],
                    row["case_count"],
                    row["final_mean_margin"],
                    row["final_min_margin"],
                    row["margin_gain"],
                    row["result"],
                )
            )
            print()

    valid = [
        r for r in rows
        if "positive_count" in r
    ]

    valid.sort(
        key=lambda r: (
            r["positive_count"],
            r["final_min_margin"],
            r["final_mean_margin"],
        ),
        reverse=True,
    )

    print()
    print("=" * 120)
    print(" GLOBAL LR SWEEP RANKING")
    print("=" * 120)
    print(
        "Rank Source   final_norm    lm_head      "
        "Positive   MinMargin    MeanMargin   Gain"
    )
    print("-" * 120)

    for rank, row in enumerate(valid, 1):
        print(
            f"{rank:4d} "
            f"{row['source_tag']:8s} "
            f"{row['lr_final_norm']:11.2e} "
            f"{row['lr_lm_head']:11.2e} "
            f"{row['positive_count']:2d}/{row['case_count']:<2d}      "
            f"{row['final_min_margin']:+.6f} "
            f"{row['final_mean_margin']:+.6f} "
            f"{row['margin_gain']:+.6f}"
        )

    best = valid[0] if valid else None

    print()
    if best is None:
        overall = "NO_VALID_RUN"
        print("RESULT            :", overall)
    else:
        all_positive = (
            best["positive_count"] == best["case_count"]
            and best["final_min_margin"] > 0.0
        )
        overall = (
            "SWEEP_FOUND_CROSSED_CANDIDATE"
            if all_positive
            else "SWEEP_NO_CROSSING"
        )
        print("RESULT            :", overall)
        print("Best source       :", best["source"])
        print("Best final_norm LR:", best["lr_final_norm"])
        print("Best lm_head LR   :", best["lr_lm_head"])
        print("Best positive     :", f"{best['positive_count']}/{best['case_count']}")
        print("Best min margin   :", f"{best['final_min_margin']:+.6f}")
        print("Best mean margin  :", f"{best['final_mean_margin']:+.6f}")
        print("Best candidate    :", best["candidate"])
        print()
        print("NOTE              : candidate only; not promoted")

    summary = {
        "version": "v0.15.7.1",
        "experiment": "decoder_crossing_lr_sweep",
        "sources": sources,
        "lr_pairs": [
            {
                "lr_final_norm": fn_lr,
                "lr_lm_head": head_lr,
            }
            for fn_lr, head_lr in DEFAULT_LR_PAIRS
        ],
        "config": {
            "epochs": args.epochs,
            "margin": args.margin,
            "lambda_margin": args.lambda_margin,
            "lambda_replay": args.lambda_replay,
            "clip_grad": args.clip_grad,
            "seed": args.seed,
        },
        "runs": rows,
        "ranking": valid,
        "best": best,
        "result": overall,
        "promotion_policy": {
            "automatic_promotion": False,
            "next_validation": (
                "Run run_decoder_boundary_diagnostic_v01561.py "
                "with the best candidate, then protected-known/runtime retention."
            ),
        },
    }

    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("Summary JSON      :", summary_path)

    return 0 if overall == "SWEEP_FOUND_CROSSED_CANDIDATE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
