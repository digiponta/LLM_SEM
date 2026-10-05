#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LLM_SEM v0.15.7.4 Trajectory Preservation Validation

Rationale:
Greedy string similarity is too brittle for a small autoregressive model:
one early token change can cause large downstream text divergence even when
the local decoder distribution is still largely preserved.

This validator therefore uses:
  A) prompt-aware boundary validation (v0.15.6.1)
  B) prompt-local JS divergence + top1 retention
  C) source-trajectory JS divergence under teacher-forced source prefixes
  D) source-reference NLL delta

No checkpoint is promoted automatically.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from model import LanguageModel
from tokenizer import Tokenizer


DEFAULT_CANDIDATES = [
    "model/v01572_sweep/model-sem-decoder-v01572-fn5em04_head2em04.candidate.pt",
    "model/v01572_sweep/model-sem-decoder-v01572-fn1em03_head5em04.candidate.pt",
]

PROTECTED_PROMPTS = [
    "コンピュータとは",
    "Pythonとは",
    "科学とは",
    "宇宙とは",
    "時間とは",
    "動物とは",
    "天気とは",
    "食べ物とは",
    "交通とは",
    "なぜGPUは高速",
    "CPUの役割",
    "GPUの役割",
]


def parse_args():
    p = argparse.ArgumentParser(
        description="LLM_SEM v0.15.7.4 Trajectory Preservation Validation"
    )
    p.add_argument(
        "--source",
        default="model/model-sem-diagnostic-v0154.pt",
    )
    p.add_argument(
        "--candidate",
        action="append",
        dest="candidates",
    )
    p.add_argument("--tokenizer", default="model/tokenizer.json")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--trajectory-tokens", type=int, default=32)
    p.add_argument("--max-prompt-js", type=float, default=0.08)
    p.add_argument("--max-trajectory-js", type=float, default=0.10)
    p.add_argument("--min-trajectory-top1", type=float, default=0.75)
    p.add_argument("--max-source-nll-delta", type=float, default=0.50)
    p.add_argument(
        "--summary",
        default="results/trajectory_preservation_v01574.json",
    )
    return p.parse_args()


def choose_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(name)


def encode_prompt(tokenizer, text):
    return tokenizer.encode(text, add_bos=True, add_eos=False)


@torch.no_grad()
def logits_for_prefix(model, prefix):
    device = next(model.parameters()).device
    context = prefix[-model.context_length:]
    x = torch.tensor([context], dtype=torch.long, device=device)
    return model(x)[0, -1, :]


def js_divergence_from_logits(a, b):
    pa = F.softmax(a, dim=-1).clamp_min(1.0e-12)
    pb = F.softmax(b, dim=-1).clamp_min(1.0e-12)
    m = 0.5 * (pa + pb)
    kl_a = torch.sum(pa * (torch.log(pa) - torch.log(m)))
    kl_b = torch.sum(pb * (torch.log(pb) - torch.log(m)))
    return float((0.5 * (kl_a + kl_b)).item())


@torch.no_grad()
def source_greedy_ids(model, tokenizer, prompt, max_tokens):
    prefix = encode_prompt(tokenizer, prompt)
    generated = []
    for _ in range(max_tokens):
        logits = logits_for_prefix(model, prefix)
        tid = int(torch.argmax(logits).item())
        generated.append(tid)
        prefix.append(tid)
        if tokenizer.eos_id is not None and tid == tokenizer.eos_id:
            break
    return generated


@torch.no_grad()
def prompt_local_metrics(source, candidate, tokenizer, prompt):
    prefix = encode_prompt(tokenizer, prompt)
    src_logits = logits_for_prefix(source, prefix)
    cand_logits = logits_for_prefix(candidate, prefix)

    src_top1 = int(torch.argmax(src_logits).item())
    cand_top1 = int(torch.argmax(cand_logits).item())

    return {
        "prompt_js": js_divergence_from_logits(src_logits, cand_logits),
        "prompt_top1_same": src_top1 == cand_top1,
        "source_top1": src_top1,
        "candidate_top1": cand_top1,
    }


@torch.no_grad()
def trajectory_metrics(
    source,
    candidate,
    tokenizer,
    prompt,
    max_tokens,
):
    source_prefix = encode_prompt(tokenizer, prompt)
    source_ids = source_greedy_ids(
        source, tokenizer, prompt, max_tokens
    )

    js_values = []
    top1_same = []
    src_nll = []
    cand_nll = []

    prefix = list(source_prefix)

    for target in source_ids:
        src_logits = logits_for_prefix(source, prefix)
        cand_logits = logits_for_prefix(candidate, prefix)

        js_values.append(
            js_divergence_from_logits(src_logits, cand_logits)
        )

        src_top1 = int(torch.argmax(src_logits).item())
        cand_top1 = int(torch.argmax(cand_logits).item())
        top1_same.append(src_top1 == cand_top1)

        src_lp = F.log_softmax(src_logits, dim=-1)[target]
        cand_lp = F.log_softmax(cand_logits, dim=-1)[target]
        src_nll.append(float(-src_lp.item()))
        cand_nll.append(float(-cand_lp.item()))

        prefix.append(int(target))

    mean_js = sum(js_values) / max(1, len(js_values))
    top1_retention = (
        sum(top1_same) / max(1, len(top1_same))
    )
    mean_src_nll = sum(src_nll) / max(1, len(src_nll))
    mean_cand_nll = sum(cand_nll) / max(1, len(cand_nll))

    return {
        "trajectory_tokens": len(source_ids),
        "trajectory_js": mean_js,
        "trajectory_top1_retention": top1_retention,
        "source_reference_nll": mean_src_nll,
        "candidate_reference_nll": mean_cand_nll,
        "source_nll_delta": mean_cand_nll - mean_src_nll,
        "source_text": tokenizer.decode(
            source_ids, skip_special_tokens=True
        ),
    }


def run_prompt_aware(source, candidate, tokenizer, tag):
    script = Path("run_decoder_boundary_diagnostic_v01561.py")
    if not script.exists():
        raise FileNotFoundError(script)

    out = Path("results") / f"prompt_aware_v01574_{tag}.json"

    cmd = [
        sys.executable,
        str(script),
        "--before", str(source),
        "--after", str(candidate),
        "--tokenizer", str(tokenizer),
        "--json-out", str(out),
    ]

    proc = subprocess.run(cmd)

    if not out.exists():
        return {
            "return_code": proc.returncode,
            "result": "NO_RESULT_JSON",
        }

    payload = json.loads(out.read_text(encoding="utf-8"))
    return {
        "return_code": proc.returncode,
        "result": payload["summary"]["result"],
        "summary": payload["summary"],
        "json": str(out),
    }


def candidate_tag(path):
    return Path(path).stem.replace(".candidate", "")


def main():
    args = parse_args()
    device = choose_device(args.device)

    source_path = Path(args.source)
    tokenizer_path = Path(args.tokenizer)
    candidate_paths = [
        Path(x) for x in (args.candidates or DEFAULT_CANDIDATES)
    ]

    for path in [source_path, tokenizer_path, *candidate_paths]:
        if not path.exists():
            raise FileNotFoundError(path)

    tokenizer = Tokenizer.load(str(tokenizer_path))
    source, _ = LanguageModel.load_checkpoint(
        str(source_path), device=device
    )
    source.eval()

    print("=" * 120)
    print(" LLM_SEM v0.15.7.4 Trajectory Preservation Validation")
    print("=" * 120)
    print("Source checkpoint       :", source_path)
    print("Candidates              :", len(candidate_paths))
    print("Protected prompts       :", len(PROTECTED_PROMPTS))
    print("Trajectory tokens       :", args.trajectory_tokens)
    print("Max prompt JS           :", args.max_prompt_js)
    print("Max trajectory JS       :", args.max_trajectory_js)
    print("Min trajectory top1     :", args.min_trajectory_top1)
    print("Max source NLL delta    :", args.max_source_nll_delta)
    print()

    results = []

    for candidate_path in candidate_paths:
        candidate, _ = LanguageModel.load_checkpoint(
            str(candidate_path), device=device
        )
        candidate.eval()

        tag = candidate_tag(candidate_path)

        print("=" * 120)
        print("CANDIDATE:", candidate_path)
        print("=" * 120)

        boundary = run_prompt_aware(
            source_path,
            candidate_path,
            tokenizer_path,
            tag,
        )

        rows = []

        for prompt in PROTECTED_PROMPTS:
            local = prompt_local_metrics(
                source, candidate, tokenizer, prompt
            )
            traj = trajectory_metrics(
                source,
                candidate,
                tokenizer,
                prompt,
                args.trajectory_tokens,
            )

            row = {
                "prompt": prompt,
                **local,
                **traj,
            }
            rows.append(row)

            print(
                f"{prompt!r:18s} "
                f"prompt_js={row['prompt_js']:.4f} "
                f"traj_js={row['trajectory_js']:.4f} "
                f"traj_top1={row['trajectory_top1_retention']:.1%} "
                f"nll_delta={row['source_nll_delta']:+.4f}"
            )

        mean_prompt_js = sum(
            x["prompt_js"] for x in rows
        ) / len(rows)
        prompt_top1_retention = sum(
            x["prompt_top1_same"] for x in rows
        ) / len(rows)
        mean_trajectory_js = sum(
            x["trajectory_js"] for x in rows
        ) / len(rows)
        mean_trajectory_top1 = sum(
            x["trajectory_top1_retention"] for x in rows
        ) / len(rows)
        mean_source_nll_delta = sum(
            x["source_nll_delta"] for x in rows
        ) / len(rows)

        boundary_ok = (
            boundary.get("result") == "PROMPT_AWARE_PASS"
        )
        preservation_ok = (
            mean_prompt_js <= args.max_prompt_js
            and mean_trajectory_js <= args.max_trajectory_js
            and mean_trajectory_top1 >= args.min_trajectory_top1
            and mean_source_nll_delta <= args.max_source_nll_delta
        )
        qualified = boundary_ok and preservation_ok

        print()
        print("TRAJECTORY PRESERVATION SUMMARY")
        print("-" * 120)
        print("Prompt-aware result      :", boundary.get("result"))
        print(f"Mean prompt JS           : {mean_prompt_js:.6f}")
        print(f"Prompt top1 retention    : {prompt_top1_retention:.1%}")
        print(f"Mean trajectory JS       : {mean_trajectory_js:.6f}")
        print(f"Mean trajectory top1     : {mean_trajectory_top1:.1%}")
        print(f"Mean source NLL delta    : {mean_source_nll_delta:+.6f}")
        print("Boundary check           :", "PASS" if boundary_ok else "FAIL")
        print(
            "Trajectory preservation :", "PASS" if preservation_ok else "FAIL"
        )
        print("QUALIFIED                :", "YES" if qualified else "NO")
        print()

        results.append({
            "candidate": str(candidate_path),
            "boundary": boundary,
            "mean_prompt_js": mean_prompt_js,
            "prompt_top1_retention": prompt_top1_retention,
            "mean_trajectory_js": mean_trajectory_js,
            "mean_trajectory_top1": mean_trajectory_top1,
            "mean_source_nll_delta": mean_source_nll_delta,
            "boundary_ok": boundary_ok,
            "preservation_ok": preservation_ok,
            "qualified": qualified,
            "rows": rows,
        })

    qualified = [r for r in results if r["qualified"]]
    qualified.sort(
        key=lambda r: (
            -r["mean_trajectory_js"],
            r["mean_trajectory_top1"],
            -r["mean_source_nll_delta"],
        ),
        reverse=True,
    )

    print("=" * 120)
    print(" GLOBAL TRAJECTORY VALIDATION")
    print("=" * 120)

    if qualified:
        overall = "QUALIFIED_CANDIDATE_FOUND"
        best = qualified[0]
        print("RESULT                 :", overall)
        print("Best candidate         :", best["candidate"])
        print(
            "Mean trajectory JS     :",
            f"{best['mean_trajectory_js']:.6f}",
        )
        print(
            "Trajectory top1        :",
            f"{best['mean_trajectory_top1']:.1%}",
        )
        print(
            "Mean source NLL delta  :",
            f"{best['mean_source_nll_delta']:+.6f}",
        )
        print("NOTE                   : still not auto-promoted")
    else:
        overall = "NO_QUALIFIED_CANDIDATE"
        best = None
        print("RESULT                 :", overall)
        print("NOTE                   : do not promote")

    payload = {
        "version": "v0.15.7.4",
        "experiment": "trajectory_preservation_validation",
        "source": str(source_path),
        "tokenizer": str(tokenizer_path),
        "thresholds": {
            "max_prompt_js": args.max_prompt_js,
            "max_trajectory_js": args.max_trajectory_js,
            "min_trajectory_top1": args.min_trajectory_top1,
            "max_source_nll_delta": args.max_source_nll_delta,
        },
        "results": results,
        "best": best,
        "result": overall,
        "promotion_policy": {
            "automatic_promotion": False,
            "required_next": (
                "Manual runtime review before promotion."
            ),
        },
    }

    out = Path(args.summary)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("Summary JSON           :", out)

    return 0 if overall == "QUALIFIED_CANDIDATE_FOUND" else 1


if __name__ == "__main__":
    raise SystemExit(main())
