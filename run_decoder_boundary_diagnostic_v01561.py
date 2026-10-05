#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LLM_SEM v0.15.6.1 Prompt-Aware Decoder Boundary Diagnostic

This diagnostic fixes the v0.15.6 scoring artifact where the prompt subject
could be duplicated in the teacher-forced target.

Instead of scoring:
    prompt = "量子センサー"
    target = "量子センサーは、..."

it scores prompt-aware continuations such as:
    prompt = "量子センサー"
    canonical continuation = "は、原子や電子..."
    confuser continuation   = "は、量子状態を利用して情報を伝送する..."

The experiment is diagnostic only. It never promotes or overwrites checkpoints.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F

from model import LanguageModel
from tokenizer import Tokenizer


DEFAULT_CASES = [
    {
        "probe": "量子センサーとは",
        "canonical": [
            "、原子や電子などのミクロな「量子」の性質を利用する。",
            "、磁場、温度、加速度、時間などを極めて高い感度で測定する次世代の計測技術である。",
        ],
        "confuser": [
            "、量子状態を利用して情報を伝送する通信技術である。",
            "、量子状態を利用して、理論上絶対に盗聴されない安全な通信を実現する技術である。",
        ],
    },
    {
        "probe": "量子センサー",
        "canonical": [
            "は、原子や電子などのミクロな「量子」の性質を利用する。",
            "は、磁場、温度、加速度、時間などを極めて高い感度で測定する次世代の計測技術である。",
        ],
        "confuser": [
            "は、量子状態を利用して情報を伝送する通信技術である。",
            "は、量子状態を利用して、理論上絶対に盗聴されない安全な通信を実現する技術である。",
        ],
    },
    {
        "probe": "量子センサーについて教えて",
        "canonical": [
            "。量子センサーは、原子や電子などのミクロな「量子」の性質を利用する。",
            "。量子センサーは、磁場、温度、加速度、時間などを極めて高い感度で測定する次世代の計測技術である。",
        ],
        "confuser": [
            "。量子通信は、量子状態を利用して情報を伝送する通信技術である。",
            "。量子通信は、量子状態を利用して、理論上絶対に盗聴されない安全な通信を実現する技術である。",
        ],
    },
    {
        "probe": "量子センサーを説明して",
        "canonical": [
            "。量子センサーは、原子や電子などのミクロな「量子」の性質を利用する。",
            "。量子センサーは、磁場、温度、加速度、時間などを極めて高い感度で測定する次世代の計測技術である。",
        ],
        "confuser": [
            "。量子通信は、量子状態を利用して情報を伝送する通信技術である。",
            "。量子通信は、量子状態を利用して、理論上絶対に盗聴されない安全な通信を実現する技術である。",
        ],
    },
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LLM_SEM v0.15.6.1 Prompt-Aware Decoder Boundary Diagnostic"
    )
    p.add_argument("--before", required=True)
    p.add_argument("--after", required=True)
    p.add_argument("--tokenizer", default="model/tokenizer.json")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--top-k", type=int, default=12)
    p.add_argument("--early-tokens", type=int, default=16)
    p.add_argument("--max-new-tokens", type=int, default=80)
    p.add_argument(
        "--json-out",
        default="results/decoder_boundary_v01561.json",
    )
    return p.parse_args()


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(name)


def prompt_ids(tokenizer: Tokenizer, text: str) -> List[int]:
    return tokenizer.encode(text, add_bos=True, add_eos=False)


def continuation_ids(tokenizer: Tokenizer, text: str) -> List[int]:
    return tokenizer.encode(text, add_bos=False, add_eos=True)


def token_text(tokenizer: Tokenizer, token_id: int) -> str:
    t = tokenizer.id_to_token.get(int(token_id), "<UNK>")
    return (
        t.replace("\n", "\\n")
         .replace("\t", "\\t")
         .replace(" ", "␠")
    )


@torch.no_grad()
def score_continuation(
    model: LanguageModel,
    tokenizer: Tokenizer,
    prompt: str,
    continuation: str,
) -> Dict[str, object]:
    prefix = prompt_ids(tokenizer, prompt)
    targets = continuation_ids(tokenizer, continuation)
    device = next(model.parameters()).device

    logps = []
    logits_at_target = []
    ranks = []

    for target in targets:
        context = prefix[-model.context_length:]
        x = torch.tensor([context], dtype=torch.long, device=device)
        logits = model(x)[0, -1, :]
        log_probs = F.log_softmax(logits, dim=-1)

        target_lp = float(log_probs[target].item())
        target_logit = float(logits[target].item())
        rank = int((logits > logits[target]).sum().item()) + 1

        logps.append(target_lp)
        logits_at_target.append(target_logit)
        ranks.append(rank)
        prefix.append(int(target))

    mean_lp = sum(logps) / max(1, len(logps))
    nll = -mean_lp

    return {
        "continuation": continuation,
        "ids": targets,
        "token_log_probs": logps,
        "token_logits": logits_at_target,
        "token_ranks": ranks,
        "mean_log_prob": mean_lp,
        "nll": nll,
        "ppl": math.exp(min(50.0, nll)),
    }


@torch.no_grad()
def next_top(model, tokenizer, prompt, k):
    ids = prompt_ids(tokenizer, prompt)
    context = ids[-model.context_length:]
    device = next(model.parameters()).device
    x = torch.tensor([context], dtype=torch.long, device=device)
    logits = model(x)[0, -1, :]
    log_probs = F.log_softmax(logits, dim=-1)

    values, token_ids = torch.topk(log_probs, min(k, log_probs.numel()))
    rows = []

    for rank, (tid, value) in enumerate(
        zip(token_ids.tolist(), values.tolist()),
        start=1,
    ):
        rows.append({
            "rank": rank,
            "id": int(tid),
            "token": token_text(tokenizer, tid),
            "log_prob": float(value),
            "prob": math.exp(float(value)),
        })

    return rows


@torch.no_grad()
def greedy(model, tokenizer, prompt, max_new_tokens):
    ids = prompt_ids(tokenizer, prompt)
    out = model.generate(
        ids,
        max_new_tokens=max_new_tokens,
        eos_id=tokenizer.eos_id,
        temperature=0.0,
        top_k=None,
        repetition_penalty=1.0,
    )
    return tokenizer.decode(out[len(ids):], skip_special_tokens=True)


def best(rows):
    return max(rows, key=lambda row: row["mean_log_prob"])


def classify(before_margin, after_margin, eps=1e-6):
    if before_margin <= eps and after_margin > eps:
        return "BOUNDARY_CROSSED"
    if before_margin > eps and after_margin > eps:
        if after_margin + eps >= before_margin:
            return "STABLE_CANONICAL"
        return "CANONICAL_WEAKENED"
    if before_margin > eps and after_margin <= eps:
        return "REGRESSED"
    if after_margin > before_margin + eps:
        return "LATENT_ONLY_IMPROVING"
    if after_margin < before_margin - eps:
        return "LATENT_ONLY_REGRESSED"
    return "LATENT_ONLY"


def first_token_delta(before_row, after_row):
    if not before_row["token_log_probs"] or not after_row["token_log_probs"]:
        return 0.0
    return (
        after_row["token_log_probs"][0]
        - before_row["token_log_probs"][0]
    )


def print_top(title, rows):
    print(title)
    print("-" * 104)
    for row in rows:
        print(
            f"{row['rank']:2d}. id={row['id']:5d} "
            f"token={row['token']!r:12s} "
            f"logp={row['log_prob']:+.6f} p={row['prob']:.6f}"
        )


def print_early(title, row, tokenizer, limit):
    print(title)
    print("-" * 104)
    for i, (tid, lp, rank) in enumerate(
        zip(row["ids"], row["token_log_probs"], row["token_ranks"])
    ):
        if i >= limit:
            break
        print(
            f"{i:2d} token={token_text(tokenizer, tid)!r:12s} "
            f"id={tid:5d} logp={lp:+.6f} rank={rank:4d}"
        )


def main() -> int:
    args = parse_args()
    device = choose_device(args.device)

    for path in (args.before, args.after, args.tokenizer):
        if not Path(path).exists():
            raise FileNotFoundError(path)

    tokenizer = Tokenizer.load(args.tokenizer)
    before, before_ckpt = LanguageModel.load_checkpoint(
        args.before, device=device
    )
    after, after_ckpt = LanguageModel.load_checkpoint(
        args.after, device=device
    )
    before.eval()
    after.eval()

    if before.vocab_size != tokenizer.vocab_size:
        raise ValueError("before/tokenizer vocabulary mismatch")
    if after.vocab_size != tokenizer.vocab_size:
        raise ValueError("after/tokenizer vocabulary mismatch")

    print("=" * 120)
    print(" LLM_SEM v0.15.6.1 Prompt-Aware Decoder Boundary Diagnostic")
    print("=" * 120)
    print("Device            :", device)
    if device.type == "cuda":
        print("GPU               :", torch.cuda.get_device_name(0))
    print("Tokenizer         :", args.tokenizer)
    print("Before checkpoint :", args.before)
    print("After checkpoint  :", args.after)
    print("Before loss       :", before_ckpt.get("loss"))
    print("After loss        :", after_ckpt.get("loss"))
    print("Scoring           : prompt-aware continuation")
    print()

    results = []

    for case in DEFAULT_CASES:
        probe = case["probe"]

        print("=" * 120)
        print("PROBE:", probe)
        print("=" * 120)

        before_greedy = greedy(
            before, tokenizer, probe, args.max_new_tokens
        )
        after_greedy = greedy(
            after, tokenizer, probe, args.max_new_tokens
        )

        print("BEFORE greedy:", before_greedy)
        print("AFTER  greedy:", after_greedy)
        print()

        before_top = next_top(before, tokenizer, probe, args.top_k)
        after_top = next_top(after, tokenizer, probe, args.top_k)

        print_top("BEFORE next-token top-k", before_top)
        print()
        print_top("AFTER next-token top-k", after_top)
        print()

        before_canon = best([
            score_continuation(before, tokenizer, probe, x)
            for x in case["canonical"]
        ])
        after_canon = best([
            score_continuation(after, tokenizer, probe, x)
            for x in case["canonical"]
        ])
        before_conf = best([
            score_continuation(before, tokenizer, probe, x)
            for x in case["confuser"]
        ])
        after_conf = best([
            score_continuation(after, tokenizer, probe, x)
            for x in case["confuser"]
        ])

        before_margin = (
            before_canon["mean_log_prob"]
            - before_conf["mean_log_prob"]
        )
        after_margin = (
            after_canon["mean_log_prob"]
            - after_conf["mean_log_prob"]
        )
        margin_gain = after_margin - before_margin
        boundary_state = classify(before_margin, after_margin)

        canonical_first_gain = first_token_delta(
            before_canon, after_canon
        )
        confuser_first_gain = first_token_delta(
            before_conf, after_conf
        )

        print("PROMPT-AWARE SCORE SUMMARY")
        print("-" * 120)
        print("Best canonical continuation BEFORE :", before_canon["continuation"])
        print("Best canonical continuation AFTER  :", after_canon["continuation"])
        print("Best confuser continuation BEFORE  :", before_conf["continuation"])
        print("Best confuser continuation AFTER   :", after_conf["continuation"])
        print()
        print(
            f"BEFORE canonical mean logp : "
            f"{before_canon['mean_log_prob']:+.6f} "
            f"NLL={before_canon['nll']:.6f}"
        )
        print(
            f"BEFORE confuser  mean logp : "
            f"{before_conf['mean_log_prob']:+.6f} "
            f"NLL={before_conf['nll']:.6f}"
        )
        print(f"BEFORE margin              : {before_margin:+.6f}")
        print(
            f"AFTER  canonical mean logp : "
            f"{after_canon['mean_log_prob']:+.6f} "
            f"NLL={after_canon['nll']:.6f}"
        )
        print(
            f"AFTER  confuser  mean logp : "
            f"{after_conf['mean_log_prob']:+.6f} "
            f"NLL={after_conf['nll']:.6f}"
        )
        print(f"AFTER  margin              : {after_margin:+.6f}")
        print(f"Margin gain                : {margin_gain:+.6f}")
        print(f"Canonical first-token gain : {canonical_first_gain:+.6f}")
        print(f"Confuser first-token gain  : {confuser_first_gain:+.6f}")
        print(f"Boundary state             : {boundary_state}")
        print()

        print_early(
            "BEFORE best canonical continuation",
            before_canon,
            tokenizer,
            args.early_tokens,
        )
        print()
        print_early(
            "AFTER best canonical continuation",
            after_canon,
            tokenizer,
            args.early_tokens,
        )
        print()
        print_early(
            "AFTER best confuser continuation",
            after_conf,
            tokenizer,
            args.early_tokens,
        )
        print()

        results.append({
            "probe": probe,
            "before_greedy": before_greedy,
            "after_greedy": after_greedy,
            "before_margin": before_margin,
            "after_margin": after_margin,
            "margin_gain": margin_gain,
            "canonical_first_token_gain": canonical_first_gain,
            "confuser_first_token_gain": confuser_first_gain,
            "boundary_state": boundary_state,
            "before_best_canonical": before_canon["continuation"],
            "after_best_canonical": after_canon["continuation"],
            "before_best_confuser": before_conf["continuation"],
            "after_best_confuser": after_conf["continuation"],
            "before_next_top": before_top,
            "after_next_top": after_top,
        })

    mean_before_margin = sum(
        x["before_margin"] for x in results
    ) / len(results)
    mean_after_margin = sum(
        x["after_margin"] for x in results
    ) / len(results)
    mean_margin_gain = mean_after_margin - mean_before_margin

    crossed = sum(
        x["boundary_state"] == "BOUNDARY_CROSSED"
        for x in results
    )
    stable = sum(
        x["boundary_state"] == "STABLE_CANONICAL"
        for x in results
    )
    regressed = sum(
        x["boundary_state"] == "REGRESSED"
        for x in results
    )
    latent_improving = sum(
        x["boundary_state"] == "LATENT_ONLY_IMPROVING"
        for x in results
    )

    if crossed + stable == len(results):
        overall = "PROMPT_AWARE_PASS"
    elif regressed > 0:
        overall = "FAIL_REGRESSION"
    elif mean_margin_gain > 0.0:
        overall = "LATENT_ONLY_IMPROVING"
    else:
        overall = "LATENT_ONLY"

    print("=" * 120)
    print(" GLOBAL PROMPT-AWARE DECODER SUMMARY")
    print("=" * 120)
    print(f"Mean margin BEFORE   : {mean_before_margin:+.6f}")
    print(f"Mean margin AFTER    : {mean_after_margin:+.6f}")
    print(f"Mean margin gain     : {mean_margin_gain:+.6f}")
    print(f"Boundary crossed     : {crossed}/{len(results)}")
    print(f"Stable canonical     : {stable}/{len(results)}")
    print(f"Latent improving     : {latent_improving}/{len(results)}")
    print(f"Regressed            : {regressed}/{len(results)}")
    print(f"RESULT               : {overall}")
    print()

    if overall == "PROMPT_AWARE_PASS":
        print("Interpretation: canonical continuation wins across all probes.")
        print("Next step: runtime-retention validation before promotion.")
    elif overall == "LATENT_ONLY_IMPROVING":
        print("Interpretation: prompt-aware decoder preference improved,")
        print("but canonical continuation has not fully crossed the boundary.")
        print("Next step: v0.15.7 contrastive decoder crossing.")
    elif overall == "FAIL_REGRESSION":
        print("Interpretation: at least one probe lost canonical preference.")
        print("Do not promote.")
    else:
        print("Interpretation: canonical continuation still does not improve")
        print("enough relative to the confuser.")
        print("Next step: v0.15.7 contrastive decoder crossing.")
    print()

    output = {
        "version": "v0.15.6.1",
        "experiment": "prompt_aware_decoder_boundary_diagnostic",
        "before_checkpoint": args.before,
        "after_checkpoint": args.after,
        "tokenizer": args.tokenizer,
        "summary": {
            "mean_margin_before": mean_before_margin,
            "mean_margin_after": mean_after_margin,
            "mean_margin_gain": mean_margin_gain,
            "boundary_crossed": crossed,
            "stable_canonical": stable,
            "latent_improving": latent_improving,
            "regressed": regressed,
            "probe_count": len(results),
            "result": overall,
        },
        "results": results,
    }

    out = Path(args.json_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("Summary JSON         :", out)

    return 0 if overall == "PROMPT_AWARE_PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
