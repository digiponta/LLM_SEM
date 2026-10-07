#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LLM_SEM v0.17.5 False-Activation Repair

Goal:
  Narrow the attractor basin of the newly internalized quantum-sensor fact
  without losing the validated protected knowledge.

Method:
  - source checkpoint: model-sem-sleep-v0172.pt
  - positives: quantum-sensor fact + spelling variant
  - protected: 14 canonical facts
  - negatives: near and unrelated probes that must NOT emit the quantum-sensor answer
  - loss: positive/protected NLL + negative token unlikelihood
  - trainable: final_norm + lm_head
  - runtime checks use real generate() settings
"""

from __future__ import annotations

import argparse
import copy
import difflib
import json
import unicodedata
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from model import LanguageModel
from tokenizer import Tokenizer


DEFAULT_MODEL = "model/model-sem-sleep-v0172.pt"
DEFAULT_TOKENIZER = "model/tokenizer.json"
DEFAULT_PROTECTED = "data/protected_knowledge_v0167.jsonl"
DEFAULT_OUTPUT = "model/model-sem-stage5-v0175.pt"
DEFAULT_RESULTS = "results/false_activation_repair_v0175.json"

TRAILING_PUNCTUATION = "、，,。．.!！?？:：;；"

NEW_PROMPTS = [
    "量子センサーとは",
    "量子センサとは",
]
NEW_ANSWER = "原子や電子などの量子的性質を利用する高感度な計測技術である。"

NEGATIVE_PROMPTS = [
    "量子通信とは",
    "量子コンピュータとは",
    "量子暗号とは",
    "暗号",
    "文学とは",
    "ブラックホールとは",
]


def parse_args():
    p = argparse.ArgumentParser(
        description="LLM_SEM v0.17.5 False-Activation Repair"
    )
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    p.add_argument("--protected", default=DEFAULT_PROTECTED)
    p.add_argument("--output", default=DEFAULT_OUTPUT)
    p.add_argument("--results", default=DEFAULT_RESULTS)
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--epochs", type=int, default=800)
    p.add_argument("--lr-final-norm", type=float, default=2.0e-4)
    p.add_argument("--lr-lm-head", type=float, default=1.0e-4)
    p.add_argument("--positive-weight", type=float, default=2.0)
    p.add_argument("--protected-weight", type=float, default=1.0)
    p.add_argument("--negative-weight", type=float, default=3.0)
    p.add_argument("--negative-prefix-tokens", type=int, default=12)
    p.add_argument("--clip-grad", type=float, default=1.0)
    p.add_argument("--check-every", type=int, default=20)
    p.add_argument("--max-new-tokens", type=int, default=96)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--repetition-penalty", type=float, default=1.10)
    p.add_argument("--false-activation-similarity", type=float, default=0.70)
    return p.parse_args()


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(name)


def normalize_prompt(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).strip()
    return text.rstrip(TRAILING_PUNCTUATION).strip()


def load_protected(path: Path) -> List[Dict[str, str]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, 1):
            raw = raw.strip()
            if not raw:
                continue
            item = json.loads(raw)
            prompt = str(item.get("prompt", "")).strip()
            answer = str(item.get("answer", "")).strip()
            if not prompt or not answer:
                raise ValueError(f"invalid protected row at line {line_no}")
            rows.append({"prompt": prompt, "answer": answer})
    return rows


def encode_prompt(tokenizer: Tokenizer, text: str) -> List[int]:
    return tokenizer.encode(
        normalize_prompt(text),
        add_bos=True,
        add_eos=False,
    )


def continuation_nll(
    model: LanguageModel,
    tokenizer: Tokenizer,
    prompt: str,
    answer: str,
) -> torch.Tensor:
    device = next(model.parameters()).device
    prompt_ids = encode_prompt(tokenizer, prompt)
    answer_ids = tokenizer.encode(answer, add_bos=False, add_eos=True)
    if not answer_ids:
        raise ValueError("empty answer")

    full = prompt_ids + answer_ids
    if len(full) > model.context_length:
        keep = model.context_length
        full = full[-keep:]
        prompt_count = max(1, keep - len(answer_ids))
    else:
        prompt_count = len(prompt_ids)

    x = torch.tensor([full[:-1]], dtype=torch.long, device=device)
    y = torch.tensor([full[1:]], dtype=torch.long, device=device)
    logits = model(x)

    start = max(0, prompt_count - 1)
    logits = logits[:, start:, :]
    y = y[:, start:]

    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        y.reshape(-1),
    )


def negative_unlikelihood(
    model: LanguageModel,
    tokenizer: Tokenizer,
    prompt: str,
    forbidden_answer: str,
    prefix_tokens: int,
) -> torch.Tensor:
    """Penalize the forbidden answer path under a negative prompt.

    Teacher-force the forbidden continuation, then apply token-level
    unlikelihood loss to the first N continuation tokens.  Early tokens matter
    most for preventing the runtime decoder from entering the wrong attractor.
    """
    device = next(model.parameters()).device
    prompt_ids = encode_prompt(tokenizer, prompt)
    answer_ids = tokenizer.encode(
        forbidden_answer,
        add_bos=False,
        add_eos=False,
    )
    if not answer_ids:
        raise ValueError("empty forbidden answer")

    answer_ids = answer_ids[: max(1, prefix_tokens)]
    losses = []
    generated = list(prompt_ids)

    for target_id in answer_ids:
        context = generated[-model.context_length:]
        x = torch.tensor([context], dtype=torch.long, device=device)
        logits = model(x)[0, -1, :]
        log_probs = F.log_softmax(logits, dim=-1)
        p_target = log_probs[int(target_id)].exp().clamp(max=1.0 - 1.0e-6)
        losses.append(-torch.log1p(-p_target))
        generated.append(int(target_id))

    return torch.stack(losses).mean()


@torch.no_grad()
def generate_answer(
    model: LanguageModel,
    tokenizer: Tokenizer,
    prompt: str,
    args,
) -> str:
    prefix = encode_prompt(tokenizer, prompt)
    generated = model.generate(
        prefix,
        max_new_tokens=args.max_new_tokens,
        eos_id=tokenizer.eos_id,
        temperature=args.temperature,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
    )
    return tokenizer.decode(
        generated[len(prefix):],
        skip_special_tokens=True,
    ).strip()


def similarity_to_new_answer(text: str) -> float:
    a = unicodedata.normalize("NFKC", text).strip()
    b = unicodedata.normalize("NFKC", NEW_ANSWER).strip()
    if not a:
        return 0.0
    if a == b or a in b or b in a:
        return 1.0
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()


@torch.no_grad()
def runtime_eval(
    model: LanguageModel,
    tokenizer: Tokenizer,
    protected_rows: List[Dict[str, str]],
    args,
):
    positive = {}
    for prompt in NEW_PROMPTS:
        actual = generate_answer(model, tokenizer, prompt, args)
        positive[prompt] = {
            "actual": actual,
            "pass": actual == NEW_ANSWER,
        }

    protected = {}
    for item in protected_rows:
        actual = generate_answer(model, tokenizer, item["prompt"], args)
        protected[item["prompt"]] = {
            "expected": item["answer"],
            "actual": actual,
            "pass": actual == item["answer"],
        }

    negatives = {}
    for prompt in NEGATIVE_PROMPTS:
        actual = generate_answer(model, tokenizer, prompt, args)
        sim = similarity_to_new_answer(actual)
        negatives[prompt] = {
            "actual": actual,
            "similarity": sim,
            "pass": sim < args.false_activation_similarity,
        }

    positive_pass = sum(int(v["pass"]) for v in positive.values())
    protected_pass = sum(int(v["pass"]) for v in protected.values())
    negative_pass = sum(int(v["pass"]) for v in negatives.values())

    return {
        "positive": positive,
        "protected": protected,
        "negatives": negatives,
        "positive_pass": positive_pass,
        "positive_total": len(positive),
        "protected_pass": protected_pass,
        "protected_total": len(protected),
        "negative_pass": negative_pass,
        "negative_total": len(negatives),
        "all_pass": (
            positive_pass == len(positive)
            and protected_pass == len(protected)
            and negative_pass == len(negatives)
        ),
    }


def freeze_for_repair(model: LanguageModel) -> None:
    for p in model.parameters():
        p.requires_grad = False
    for p in model.final_norm.parameters():
        p.requires_grad = True
    for p in model.lm_head.parameters():
        p.requires_grad = True


def score_eval(result) -> float:
    # Lower is better.  Runtime correctness dominates NLL.
    return (
        10.0 * (result["positive_total"] - result["positive_pass"])
        + 5.0 * (result["protected_total"] - result["protected_pass"])
        + 10.0 * (result["negative_total"] - result["negative_pass"])
    )


def main():
    args = parse_args()
    device = choose_device(args.device)

    model_path = Path(args.model)
    tokenizer_path = Path(args.tokenizer)
    protected_path = Path(args.protected)
    output_path = Path(args.output)
    results_path = Path(args.results)

    tokenizer = Tokenizer.load(str(tokenizer_path))
    protected_rows = load_protected(protected_path)
    model, checkpoint = LanguageModel.load_checkpoint(
        str(model_path),
        device=device,
    )
    model.eval()

    baseline = runtime_eval(model, tokenizer, protected_rows, args)

    print("=" * 88)
    print(" LLM_SEM v0.17.5 False-Activation Repair")
    print("=" * 88)
    print("Model                  :", model_path)
    print("Protected entries      :", len(protected_rows))
    print("Negative probes        :", len(NEGATIVE_PROMPTS))
    print("Epochs                 :", args.epochs)
    print("LR final_norm          :", args.lr_final_norm)
    print("LR lm_head             :", args.lr_lm_head)
    print("Positive weight        :", args.positive_weight)
    print("Protected weight       :", args.protected_weight)
    print("Negative weight        :", args.negative_weight)
    print("Negative prefix tokens :", args.negative_prefix_tokens)
    print("False-activation th    :", args.false_activation_similarity)
    print("Device                 :", device)
    print()
    print(
        "Baseline runtime      : "
        f"positive={baseline['positive_pass']}/{baseline['positive_total']} "
        f"protected={baseline['protected_pass']}/{baseline['protected_total']} "
        f"negative={baseline['negative_pass']}/{baseline['negative_total']}"
    )
    print()

    freeze_for_repair(model)
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
        weight_decay=0.0,
    )

    best_state = copy.deepcopy(model.state_dict())
    best_eval = baseline
    best_score = score_eval(baseline)
    best_epoch = 0
    stop_reason = "MAX_EPOCHS"

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        positive_loss = torch.stack([
            continuation_nll(model, tokenizer, prompt, NEW_ANSWER)
            for prompt in NEW_PROMPTS
        ]).mean()

        protected_loss = torch.stack([
            continuation_nll(
                model,
                tokenizer,
                item["prompt"],
                item["answer"],
            )
            for item in protected_rows
        ]).mean()

        negative_loss = torch.stack([
            negative_unlikelihood(
                model,
                tokenizer,
                prompt,
                NEW_ANSWER,
                args.negative_prefix_tokens,
            )
            for prompt in NEGATIVE_PROMPTS
        ]).mean()

        loss = (
            args.positive_weight * positive_loss
            + args.protected_weight * protected_loss
            + args.negative_weight * negative_loss
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            args.clip_grad,
        )
        optimizer.step()

        should_check = (
            epoch == 1
            or epoch == args.epochs
            or epoch % max(1, args.check_every) == 0
        )
        if not should_check:
            continue

        model.eval()
        current = runtime_eval(model, tokenizer, protected_rows, args)
        current_score = score_eval(current)

        print(
            f"epoch={epoch:3d}/{args.epochs} "
            f"loss={float(loss.item()):.6f} "
            f"pos_nll={float(positive_loss.item()):.6f} "
            f"prot_nll={float(protected_loss.item()):.6f} "
            f"neg_ul={float(negative_loss.item()):.6f} "
            f"positive={current['positive_pass']}/{current['positive_total']} "
            f"protected={current['protected_pass']}/{current['protected_total']} "
            f"negative={current['negative_pass']}/{current['negative_total']}"
        )

        if current_score < best_score:
            best_score = current_score
            best_state = copy.deepcopy(model.state_dict())
            best_eval = current
            best_epoch = epoch

        if current["all_pass"]:
            best_state = copy.deepcopy(model.state_dict())
            best_eval = current
            best_epoch = epoch
            stop_reason = "FALSE_ACTIVATION_REPAIRED"
            print("REPAIR> all positive/protected/negative runtime checks passed")
            break

    model.load_state_dict(best_state)
    model.eval()

    final_eval = runtime_eval(model, tokenizer, protected_rows, args)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    new_checkpoint = dict(checkpoint)
    new_checkpoint["model_state_dict"] = model.state_dict()
    new_checkpoint["stage5_repair"] = {
        "version": "v0.17.5",
        "source_checkpoint": str(model_path),
        "best_epoch": best_epoch,
        "stop_reason": stop_reason,
        "negative_prompts": NEGATIVE_PROMPTS,
        "negative_prefix_tokens": args.negative_prefix_tokens,
        "false_activation_similarity": args.false_activation_similarity,
        "runtime": final_eval,
        "status": (
            "FALSE_ACTIVATION_REPAIRED"
            if final_eval["all_pass"]
            else "FALSE_ACTIVATION_REPAIR_CANDIDATE"
        ),
    }
    torch.save(new_checkpoint, output_path)

    results = {
        "version": "v0.17.5",
        "source_model": str(model_path),
        "output_model": str(output_path),
        "baseline": baseline,
        "final": final_eval,
        "best_epoch": best_epoch,
        "stop_reason": stop_reason,
        "status": (
            "PASS" if final_eval["all_pass"] else "FAIL"
        ),
    }
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print()
    print("=" * 88)
    print(" FALSE-ACTIVATION REPAIR RESULT")
    print("=" * 88)
    print(
        "Positive runtime      : "
        f"{final_eval['positive_pass']}/{final_eval['positive_total']}"
    )
    print(
        "Protected runtime     : "
        f"{final_eval['protected_pass']}/{final_eval['protected_total']}"
    )
    print(
        "Negative rejection    : "
        f"{final_eval['negative_pass']}/{final_eval['negative_total']}"
    )
    print("Selected epoch        :", best_epoch)
    print("Stop reason           :", stop_reason)
    print("Output model          :", output_path)
    print("Results               :", results_path)
    print(
        "STATUS                :",
        "PASS" if final_eval["all_pass"] else "FAIL",
    )

    for prompt, item in final_eval["negatives"].items():
        status = "PASS" if item["pass"] else "FAIL"
        print(
            f"[{status}] {prompt!r} "
            f"sim={item['similarity']:.3f} -> {item['actual']}"
        )

    raise SystemExit(0 if final_eval["all_pass"] else 1)


if __name__ == "__main__":
    main()
