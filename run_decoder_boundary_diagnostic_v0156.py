#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LLM_SEM v0.15.6 Decoder Boundary Diagnostic

Compares BEFORE and AFTER checkpoints at the decoder level.
The diagnostic never promotes or overwrites a checkpoint.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from model import LanguageModel
from tokenizer import Tokenizer


DEFAULT_PROBES = [
    "量子センサーとは",
    "量子センサー",
    "量子センサーについて教えて",
    "量子センサーを説明して",
]

DEFAULT_CANONICAL = [
    "量子センサーは、原子や電子などのミクロな「量子」の性質を利用する。",
    "量子センサーは、磁場、温度、加速度、時間などを極めて高い感度で測定する次世代の計測技術である。",
]

DEFAULT_CONFUSERS = [
    "量子通信は、量子状態を利用して情報を伝送する通信技術である。",
    "量子通信は、量子状態を利用して、理論上絶対に盗聴されない安全な通信を実現する技術である。",
]


def parse_args():
    p = argparse.ArgumentParser(
        description="LLM_SEM v0.15.6 Decoder Boundary Diagnostic"
    )
    p.add_argument("--before", required=True)
    p.add_argument("--after", required=True)
    p.add_argument("--tokenizer", default="model/tokenizer.json")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--top-k", type=int, default=12)
    p.add_argument("--early-tokens", type=int, default=16)
    p.add_argument("--max-new-tokens", type=int, default=80)
    p.add_argument("--probe", action="append", dest="probes")
    p.add_argument("--canonical", action="append", dest="canonicals")
    p.add_argument("--confuser", action="append", dest="confusers")
    p.add_argument(
        "--json-out",
        default="results/decoder_boundary_v0156.json",
    )
    return p.parse_args()


def choose_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(name)


def token_text(tokenizer, token_id):
    t = tokenizer.id_to_token.get(int(token_id), "<UNK>")
    return t.replace("\n", "\\n").replace("\t", "\\t").replace(" ", "␠")


def prompt_ids(tokenizer, text):
    return tokenizer.encode(text, add_bos=True, add_eos=False)


def answer_ids(tokenizer, text):
    return tokenizer.encode(text, add_bos=False, add_eos=True)


@torch.no_grad()
def score_answer(model, tokenizer, prompt, answer):
    prefix = prompt_ids(tokenizer, prompt)
    targets = answer_ids(tokenizer, answer)
    device = next(model.parameters()).device

    logps = []
    ranks = []
    logits_at_target = []

    for target in targets:
        ctx = prefix[-model.context_length:]
        x = torch.tensor([ctx], dtype=torch.long, device=device)
        logits = model(x)[0, -1, :]
        lp = F.log_softmax(logits, dim=-1)

        logps.append(float(lp[target].item()))
        logits_at_target.append(float(logits[target].item()))
        ranks.append(int((logits > logits[target]).sum().item()) + 1)
        prefix.append(int(target))

    mean_lp = sum(logps) / max(1, len(logps))
    nll = -mean_lp
    return {
        "answer": answer,
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
    ctx = ids[-model.context_length:]
    device = next(model.parameters()).device
    x = torch.tensor([ctx], dtype=torch.long, device=device)
    logits = model(x)[0, -1, :]
    lp = F.log_softmax(logits, dim=-1)
    vals, tids = torch.topk(lp, min(k, lp.numel()))

    out = []
    for rank, (tid, val) in enumerate(zip(tids.tolist(), vals.tolist()), 1):
        out.append({
            "rank": rank,
            "id": int(tid),
            "token": token_text(tokenizer, tid),
            "log_prob": float(val),
            "prob": math.exp(float(val)),
        })
    return out


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


def state(before_margin, after_margin, eps=1e-6):
    if before_margin <= eps and after_margin > eps:
        return "BOUNDARY_CROSSED"
    if before_margin > eps and after_margin > eps:
        return "STABLE_CANONICAL" if after_margin + eps >= before_margin else "CANONICAL_WEAKENED"
    if before_margin > eps and after_margin <= eps:
        return "REGRESSED"
    if after_margin > before_margin + eps:
        return "LATENT_ONLY_IMPROVING"
    if after_margin < before_margin - eps:
        return "LATENT_ONLY_REGRESSED"
    return "LATENT_ONLY"


def print_top(title, rows):
    print(title)
    print("-" * 96)
    for row in rows:
        print(
            f"{row['rank']:2d}. id={row['id']:5d} "
            f"token={row['token']!r:12s} "
            f"logp={row['log_prob']:+.6f} p={row['prob']:.6f}"
        )


def print_early(title, row, tokenizer, limit):
    print(title)
    print("-" * 96)
    for i, (tid, lp, rank) in enumerate(
        zip(row["ids"], row["token_log_probs"], row["token_ranks"])
    ):
        if i >= limit:
            break
        print(
            f"{i:2d} token={token_text(tokenizer, tid)!r:12s} "
            f"id={tid:5d} logp={lp:+.6f} rank={rank:4d}"
        )


def main():
    args = parse_args()
    device = choose_device(args.device)

    tokenizer = Tokenizer.load(args.tokenizer)
    before, before_ckpt = LanguageModel.load_checkpoint(args.before, device=device)
    after, after_ckpt = LanguageModel.load_checkpoint(args.after, device=device)
    before.eval()
    after.eval()

    if before.vocab_size != tokenizer.vocab_size:
        raise ValueError("before/tokenizer vocabulary mismatch")
    if after.vocab_size != tokenizer.vocab_size:
        raise ValueError("after/tokenizer vocabulary mismatch")

    probes = args.probes or DEFAULT_PROBES
    canonicals = args.canonicals or DEFAULT_CANONICAL
    confusers = args.confusers or DEFAULT_CONFUSERS

    print("=" * 112)
    print(" LLM_SEM v0.15.6 Decoder Boundary Diagnostic")
    print("=" * 112)
    print("Device            :", device)
    if device.type == "cuda":
        print("GPU               :", torch.cuda.get_device_name(0))
    print("Tokenizer         :", args.tokenizer)
    print("Before checkpoint :", args.before)
    print("After checkpoint  :", args.after)
    print("Before loss       :", before_ckpt.get("loss"))
    print("After loss        :", after_ckpt.get("loss"))
    print()

    results = []

    for probe in probes:
        print("=" * 112)
        print("PROBE:", probe)
        print("=" * 112)

        bg = greedy(before, tokenizer, probe, args.max_new_tokens)
        ag = greedy(after, tokenizer, probe, args.max_new_tokens)
        print("BEFORE greedy:", bg)
        print("AFTER  greedy:", ag)
        print()

        bt = next_top(before, tokenizer, probe, args.top_k)
        at = next_top(after, tokenizer, probe, args.top_k)
        print_top("BEFORE next-token top-k", bt)
        print()
        print_top("AFTER next-token top-k", at)
        print()

        bc = best([score_answer(before, tokenizer, probe, x) for x in canonicals])
        ac = best([score_answer(after, tokenizer, probe, x) for x in canonicals])
        bf = best([score_answer(before, tokenizer, probe, x) for x in confusers])
        af = best([score_answer(after, tokenizer, probe, x) for x in confusers])

        bm = bc["mean_log_prob"] - bf["mean_log_prob"]
        am = ac["mean_log_prob"] - af["mean_log_prob"]
        gain = am - bm
        st = state(bm, am)

        print("DECODER SCORE SUMMARY")
        print("-" * 112)
        print(f"BEFORE canonical mean logp : {bc['mean_log_prob']:+.6f} NLL={bc['nll']:.6f}")
        print(f"BEFORE confuser  mean logp : {bf['mean_log_prob']:+.6f} NLL={bf['nll']:.6f}")
        print(f"BEFORE margin              : {bm:+.6f}")
        print(f"AFTER  canonical mean logp : {ac['mean_log_prob']:+.6f} NLL={ac['nll']:.6f}")
        print(f"AFTER  confuser  mean logp : {af['mean_log_prob']:+.6f} NLL={af['nll']:.6f}")
        print(f"AFTER  margin              : {am:+.6f}")
        print(f"Margin gain                : {gain:+.6f}")
        print(f"Boundary state             : {st}")
        print()

        print_early("BEFORE best canonical early tokens", bc, tokenizer, args.early_tokens)
        print()
        print_early("AFTER best canonical early tokens", ac, tokenizer, args.early_tokens)
        print()
        print_early("AFTER best confuser early tokens", af, tokenizer, args.early_tokens)
        print()

        results.append({
            "probe": probe,
            "before_greedy": bg,
            "after_greedy": ag,
            "before_margin": bm,
            "after_margin": am,
            "margin_gain": gain,
            "boundary_state": st,
            "before_best_canonical": bc["answer"],
            "after_best_canonical": ac["answer"],
            "before_best_confuser": bf["answer"],
            "after_best_confuser": af["answer"],
            "before_next_top": bt,
            "after_next_top": at,
        })

    mb = sum(x["before_margin"] for x in results) / len(results)
    ma = sum(x["after_margin"] for x in results) / len(results)
    mg = ma - mb
    crossed = sum(x["boundary_state"] == "BOUNDARY_CROSSED" for x in results)
    regressed = sum(x["boundary_state"] == "REGRESSED" for x in results)
    latent = sum(x["boundary_state"].startswith("LATENT_ONLY") for x in results)

    if crossed == len(results):
        overall = "BOUNDARY_CROSSED"
    elif regressed:
        overall = "FAIL_REGRESSION"
    elif ma > 0 and crossed:
        overall = "PARTIAL_BOUNDARY_CROSSING"
    elif mg > 0:
        overall = "LATENT_ONLY_IMPROVING"
    else:
        overall = "LATENT_ONLY"

    print("=" * 112)
    print(" GLOBAL DECODER BOUNDARY SUMMARY")
    print("=" * 112)
    print(f"Mean margin BEFORE : {mb:+.6f}")
    print(f"Mean margin AFTER  : {ma:+.6f}")
    print(f"Mean margin gain   : {mg:+.6f}")
    print(f"Boundary crossed   : {crossed}/{len(results)}")
    print(f"Latent-only probes : {latent}/{len(results)}")
    print(f"Regressed probes   : {regressed}/{len(results)}")
    print("RESULT             :", overall)

    output = {
        "version": "v0.15.6",
        "experiment": "decoder_boundary_diagnostic",
        "before_checkpoint": args.before,
        "after_checkpoint": args.after,
        "tokenizer": args.tokenizer,
        "summary": {
            "mean_margin_before": mb,
            "mean_margin_after": ma,
            "mean_margin_gain": mg,
            "boundary_crossed": crossed,
            "latent_only": latent,
            "regressed": regressed,
            "probe_count": len(results),
            "result": overall,
        },
        "results": results,
    }

    out = Path(args.json_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Summary JSON       :", out)

    return 0 if overall == "BOUNDARY_CROSSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
