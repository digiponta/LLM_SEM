#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LLM_SEM v0.15.7.3 Candidate Preservation Validation

Compares crossed decoder candidates against the protected source checkpoint.

Validation has two independent parts:
  A) target boundary: re-run v0.15.6.1 prompt-aware decoder diagnostic
  B) behavior preservation: compare source/candidate on holdout prompts that
     were NOT used by the v0.15.7 contrastive target set.

This script does not promote checkpoints.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from difflib import SequenceMatcher
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
        description="LLM_SEM v0.15.7.3 Candidate Preservation Validation"
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
    p.add_argument("--max-new-tokens", type=int, default=48)
    p.add_argument("--min-mean-similarity", type=float, default=0.70)
    p.add_argument("--min-top1-retention", type=float, default=0.75)
    p.add_argument("--max-mean-js", type=float, default=0.15)
    p.add_argument(
        "--summary",
        default="results/candidate_preservation_v01573.json",
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
def greedy(model, tokenizer, prompt, max_new_tokens):
    ids = encode_prompt(tokenizer, prompt)
    out = model.generate(
        ids,
        max_new_tokens=max_new_tokens,
        eos_id=tokenizer.eos_id,
        temperature=0.0,
        top_k=None,
        repetition_penalty=1.0,
    )
    return tokenizer.decode(out[len(ids):], skip_special_tokens=True)


@torch.no_grad()
def next_distribution(model, tokenizer, prompt):
    ids = encode_prompt(tokenizer, prompt)
    context = ids[-model.context_length:]
    device = next(model.parameters()).device
    x = torch.tensor([context], dtype=torch.long, device=device)
    logits = model(x)[0, -1, :]
    return F.softmax(logits, dim=-1)


def js_divergence(p, q):
    eps = 1.0e-12
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    m = 0.5 * (p + q)
    kl_pm = torch.sum(p * (torch.log(p) - torch.log(m)))
    kl_qm = torch.sum(q * (torch.log(q) - torch.log(m)))
    return float((0.5 * (kl_pm + kl_qm)).item())


def text_similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()


def run_prompt_aware(source, candidate, tokenizer, tag):
    script = Path("run_decoder_boundary_diagnostic_v01561.py")
    if not script.exists():
        raise FileNotFoundError(script)

    out = Path("results") / f"prompt_aware_v01573_{tag}.json"
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
    candidates = [Path(x) for x in (args.candidates or DEFAULT_CANDIDATES)]

    for path in [source_path, tokenizer_path, *candidates]:
        if not path.exists():
            raise FileNotFoundError(path)

    tokenizer = Tokenizer.load(str(tokenizer_path))
    source, _ = LanguageModel.load_checkpoint(str(source_path), device=device)
    source.eval()

    print("=" * 120)
    print(" LLM_SEM v0.15.7.3 Candidate Preservation Validation")
    print("=" * 120)
    print("Source checkpoint     :", source_path)
    print("Tokenizer             :", tokenizer_path)
    print("Candidates            :", len(candidates))
    print("Protected prompts     :", len(PROTECTED_PROMPTS))
    print("Min mean similarity   :", args.min_mean_similarity)
    print("Min top1 retention    :", args.min_top1_retention)
    print("Max mean JS           :", args.max_mean_js)
    print()

    source_cache = {}
    for prompt in PROTECTED_PROMPTS:
        text = greedy(source, tokenizer, prompt, args.max_new_tokens)
        dist = next_distribution(source, tokenizer, prompt)
        source_cache[prompt] = {
            "text": text,
            "dist": dist,
            "top1": int(torch.argmax(dist).item()),
        }

    results = []

    for candidate_path in candidates:
        tag = candidate_tag(candidate_path)
        print("=" * 120)
        print("CANDIDATE:", candidate_path)
        print("=" * 120)

        prompt_aware = run_prompt_aware(
            source_path, candidate_path, tokenizer_path, tag
        )

        candidate, _ = LanguageModel.load_checkpoint(
            str(candidate_path), device=device
        )
        candidate.eval()

        rows = []
        for prompt in PROTECTED_PROMPTS:
            src = source_cache[prompt]
            cand_text = greedy(
                candidate, tokenizer, prompt, args.max_new_tokens
            )
            cand_dist = next_distribution(candidate, tokenizer, prompt)
            cand_top1 = int(torch.argmax(cand_dist).item())

            sim = text_similarity(src["text"], cand_text)
            js = js_divergence(src["dist"], cand_dist)
            same_top1 = cand_top1 == src["top1"]

            rows.append({
                "prompt": prompt,
                "source_text": src["text"],
                "candidate_text": cand_text,
                "text_similarity": sim,
                "js_divergence": js,
                "source_top1": src["top1"],
                "candidate_top1": cand_top1,
                "same_top1": same_top1,
            })

            print(
                f"{prompt!r:18s} sim={sim:.4f} js={js:.4f} "
                f"top1={'KEEP' if same_top1 else 'CHANGED'}"
            )

        mean_sim = sum(x["text_similarity"] for x in rows) / len(rows)
        min_sim = min(x["text_similarity"] for x in rows)
        mean_js = sum(x["js_divergence"] for x in rows) / len(rows)
        top1_kept = sum(x["same_top1"] for x in rows)
        top1_retention = top1_kept / len(rows)

        boundary_ok = (
            prompt_aware.get("result") == "PROMPT_AWARE_PASS"
        )
        preservation_ok = (
            mean_sim >= args.min_mean_similarity
            and top1_retention >= args.min_top1_retention
            and mean_js <= args.max_mean_js
        )
        qualified = boundary_ok and preservation_ok

        print()
        print("VALIDATION SUMMARY")
        print("-" * 120)
        print("Prompt-aware result :", prompt_aware.get("result"))
        print(f"Mean text similarity: {mean_sim:.6f}")
        print(f"Min text similarity : {min_sim:.6f}")
        print(f"Mean JS divergence  : {mean_js:.6f}")
        print(
            f"Top1 retention      : {top1_kept}/{len(rows)} "
            f"({top1_retention:.1%})"
        )
        print("Boundary check      :", "PASS" if boundary_ok else "FAIL")
        print("Preservation check  :", "PASS" if preservation_ok else "FAIL")
        print("QUALIFIED           :", "YES" if qualified else "NO")
        print()

        results.append({
            "candidate": str(candidate_path),
            "prompt_aware": prompt_aware,
            "mean_text_similarity": mean_sim,
            "min_text_similarity": min_sim,
            "mean_js_divergence": mean_js,
            "top1_retention": top1_retention,
            "top1_kept": top1_kept,
            "protected_count": len(rows),
            "boundary_ok": boundary_ok,
            "preservation_ok": preservation_ok,
            "qualified": qualified,
            "rows": rows,
        })

    qualified = [x for x in results if x["qualified"]]
    qualified.sort(
        key=lambda x: (
            x["mean_text_similarity"],
            x["top1_retention"],
            -x["mean_js_divergence"],
        ),
        reverse=True,
    )

    print("=" * 120)
    print(" GLOBAL CANDIDATE VALIDATION")
    print("=" * 120)

    if qualified:
        overall = "QUALIFIED_CANDIDATE_FOUND"
        best = qualified[0]
        print("RESULT             :", overall)
        print("Best candidate     :", best["candidate"])
        print(
            "Mean similarity    :",
            f"{best['mean_text_similarity']:.6f}",
        )
        print(
            "Top1 retention     :",
            f"{best['top1_retention']:.1%}",
        )
        print(
            "Mean JS divergence :",
            f"{best['mean_js_divergence']:.6f}",
        )
        print("NOTE               : still not auto-promoted")
    else:
        overall = "NO_QUALIFIED_CANDIDATE"
        best = None
        print("RESULT             :", overall)
        print("NOTE               : do not promote")

    payload = {
        "version": "v0.15.7.3",
        "experiment": "candidate_preservation_validation",
        "source": str(source_path),
        "tokenizer": str(tokenizer_path),
        "thresholds": {
            "min_mean_similarity": args.min_mean_similarity,
            "min_top1_retention": args.min_top1_retention,
            "max_mean_js": args.max_mean_js,
        },
        "results": results,
        "best": best,
        "result": overall,
        "promotion_policy": {
            "automatic_promotion": False,
            "required_next": (
                "Manual runtime/protected-known review before promotion."
            ),
        },
    }

    out = Path(args.summary)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("Summary JSON       :", out)

    return 0 if overall == "QUALIFIED_CANDIDATE_FOUND" else 1


if __name__ == "__main__":
    raise SystemExit(main())
