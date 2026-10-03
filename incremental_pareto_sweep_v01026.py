# incremental_pareto_sweep_v01026.py
#
# LLM_SEM v0.10.28
# Independent protected-incremental candidates using actual /internal replay anchors.
# Select only a candidate that passes the full actual /internal runtime gate.

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(
        description="LLM_SEM v0.10.26 Incremental Pareto Sweep"
    )
    p.add_argument("--source", required=True)
    p.add_argument("--incremental-dataset", required=True)
    p.add_argument("--full-dataset", required=True)
    p.add_argument("--benchmark", default="my_benchmark.csv")
    p.add_argument("--tokenizer", default="model/tokenizer.json")
    p.add_argument("--output", required=True)
    p.add_argument("--base-learning-rate", type=float, default=1e-5)
    p.add_argument("--base-lm-head-lr", type=float, default=5e-5)
    p.add_argument("--preserve-weight", type=float, default=5.0)
    p.add_argument("--train-blocks", type=int, default=1)
    p.add_argument("--allow-cpu", action="store_true")
    return p.parse_args()


def run(cmd):
    return subprocess.run(cmd, check=False).returncode


def main():
    args = parse_args()
    source = Path(args.source)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    # Search around the v0.10.24 near-pass region.
    # (epochs, lr_scale, distillation_weight, new_knowledge_weight)
    configs = [
        (240, 0.50, 1.0, 2.0),
        (240, 0.50, 2.0, 2.0),
        (320, 0.50, 1.0, 3.0),
        (320, 0.50, 2.0, 3.0),
        (480, 0.40, 1.0, 4.0),
        (480, 0.50, 2.0, 4.0),
    ]

    print("=" * 104)
    print(" LLM_SEM v0.10.28 Canonical-Aware / New-Knowledge-Boost Sweep")
    print("=" * 104)
    print("Source             :", source)
    print("Incremental dataset:", args.incremental_dataset)
    print("Full dataset       :", args.full_dataset)
    print("Candidates         :", len(configs))

    safe = []
    best = None

    for idx, (epochs, lr_scale, distill, new_weight) in enumerate(configs, 1):
        candidate = output.with_name(
            f"{output.stem}.sweep{idx}{output.suffix}"
        )
        train_json = output.with_name(
            f"{output.stem}.sweep{idx}.train.json"
        )
        runtime_json = output.with_name(
            f"{output.stem}.sweep{idx}.runtime.json"
        )

        lr = args.base_learning_rate * lr_scale
        lm_lr = args.base_lm_head_lr * lr_scale

        print()
        print("-" * 104)
        print(
            f"SWEEP {idx}/{len(configs)} "
            f"epochs={epochs} lr_scale={lr_scale:.2f} "
            f"distill={distill:.1f} new_weight={new_weight:.1f}"
        )

        train_cmd = [
            sys.executable,
            "semantic_guided_answer_finetune_v097.py",
            "--model", str(source),
            "--tokenizer", args.tokenizer,
            "--dataset", args.incremental_dataset,
            "--benchmark", args.benchmark,
            "--output", str(candidate),
            "--epochs", str(epochs),
            "--learning-rate", str(lr),
            "--lm-head-lr", str(lm_lr),
            "--preserve-weight", str(args.preserve_weight),
            "--train-blocks", str(args.train_blocks),
            "--protected-distill-weight", str(distill),
            "--new-knowledge-weight", str(new_weight),
            "--min-generation-sim", "0.0",
            "--result-json", str(train_json),
            "--prefer-final-state",
            "--concept-balanced",
        ]
        if args.allow_cpu:
            train_cmd.append("--allow-cpu")

        if run(train_cmd) != 0:
            print("SWEEP> training failed; skip candidate")
            continue

        runtime_cmd = [
            sys.executable,
            "runtime_answer_retention_v01015.py",
            "--source", str(source),
            "--candidate", str(candidate),
            "--dataset", args.full_dataset,
            "--benchmark", args.benchmark,
            "--tokenizer", args.tokenizer,
            "--result-json", str(runtime_json),
        ]
        if args.allow_cpu:
            runtime_cmd.append("--allow-cpu")

        runtime_code = run(runtime_cmd)
        if not runtime_json.exists():
            print("SWEEP> runtime result missing; skip candidate")
            continue

        m = json.loads(runtime_json.read_text(encoding="utf-8"))
        known = int(m.get("known_failures", 999))
        new = int(m.get("new_failures", 999))
        mean = float(m.get("candidate_canonical_mean", 0.0))
        failures = int(m.get("failures", 999))

        print(
            "SWEEP> result "
            f"known_failures={known} "
            f"new_failures={new} "
            f"failures={failures} "
            f"canonical_mean={mean:.6f}"
        )

        record = {
            "index": idx,
            "candidate": str(candidate),
            "epochs": epochs,
            "lr_scale": lr_scale,
            "distill": distill,
            "new_knowledge_weight": new_weight,
            "known_failures": known,
            "new_failures": new,
            "failures": failures,
            "canonical_mean": mean,
        }

        if known == 0:
            safe.append(record)
            if (
                best is None
                or new < best["new_failures"]
                or (
                    new == best["new_failures"]
                    and mean > best["canonical_mean"]
                )
            ):
                best = record

        if runtime_code == 0 and failures == 0:
            shutil.copy2(candidate, output)
            summary = output.with_suffix(".sweep-summary.json")
            summary.write_text(
                json.dumps(
                    {
                        "result": "PASS",
                        "selected": record,
                        "safe_candidates": safe,
                    },
                    ensure_ascii=False,
                    indent=2,
                ) + "\n",
                encoding="utf-8",
            )
            print()
            print("SWEEP> FULL RUNTIME PASS")
            print("SWEEP> selected:", candidate)
            print("SWEEP> output  :", output)
            raise SystemExit(0)

    summary = output.with_suffix(".sweep-summary.json")
    summary.write_text(
        json.dumps(
            {
                "result": "FAIL",
                "best_safe": best,
                "safe_candidates": safe,
            },
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 104)
    print(" SWEEP COMPLETE WITHOUT FULL RUNTIME PASS")
    print("=" * 104)
    if best is not None:
        print(
            "Best safe diagnostic candidate:",
            best["candidate"],
            f"new_failures={best['new_failures']}",
            f"canonical_mean={best['canonical_mean']:.6f}",
        )
    else:
        print("No candidate preserved all known/protected probes.")
    print("Source model remains active.")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
