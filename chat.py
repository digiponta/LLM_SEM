#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LLM_SEM v0.16.7 interactive chat with Semantic Memory /sleep.

Commands
--------
/help
/teach <prompt> => <answer>
/memory
/sleep [epochs]
/model
/reload
/quit

/sleep performs monitored decoder-only internalization:
  - trainable: final_norm + lm_head
  - target: taught prompt/answer NLL
  - crossing: canonical-vs-pre-sleep-confuser sequence margin
  - greedy alignment: canonical token vs strongest local competitor margin
  - preservation: validated canonical protected knowledge (NLL + token margin)
  - output: model/model-sem-sleep-v0167.pt
  - memory remains on disk after sleep for auditability

This is an experimental online internalization path.  It does not run the full
v0.15.7.x promotion suite automatically.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from model import LanguageModel
from tokenizer import Tokenizer


DEFAULT_MODEL = "model/model-sem-internalized-v01575.pt"
DEFAULT_TOKENIZER = "model/tokenizer.json"
DEFAULT_MEMORY = "data/semantic_memory_v0160.jsonl"
DEFAULT_PROTECTED = "data/protected_knowledge_v0167.jsonl"
DEFAULT_SLEEP_MODEL = "model/model-sem-sleep-v0167.pt"

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
    "CPUとは",
    "GPUとは",
    "CPUの役割",
    "GPUの役割",
]


def parse_args():
    p = argparse.ArgumentParser(
        description="LLM_SEM v0.16.7 Chat + /sleep internalization"
    )
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    p.add_argument("--memory", default=DEFAULT_MEMORY)
    p.add_argument("--protected", default=DEFAULT_PROTECTED)
    p.add_argument("--sleep-output", default=DEFAULT_SLEEP_MODEL)
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--max-new-tokens", type=int, default=96)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--repetition-penalty", type=float, default=1.10)

    # v0.16.7 uses the decoder LR scale that actually crossed the boundary
    # in v0.15.7.2, while monitoring preservation and stopping early.
    p.add_argument("--sleep-epochs", type=int, default=240)
    p.add_argument("--sleep-lr-final-norm", type=float, default=5.0e-4)
    p.add_argument("--sleep-lr-lm-head", type=float, default=2.0e-4)
    p.add_argument("--sleep-kl", type=float, default=0.50)
    p.add_argument("--sleep-protected-nll-weight", type=float, default=1.00)
    p.add_argument("--sleep-protected-token-weight", type=float, default=1.50)
    p.add_argument("--sleep-protected-hard-weight", type=float, default=1.00)
    p.add_argument("--sleep-protected-target-margin", type=float, default=0.05)
    p.add_argument("--sleep-min-protected-top1", type=float, default=1.00)
    p.add_argument("--sleep-max-protected-nll-delta", type=float, default=0.25)
    p.add_argument("--sleep-margin-weight", type=float, default=1.00)
    p.add_argument("--sleep-target-margin", type=float, default=0.15)
    p.add_argument("--sleep-token-margin-weight", type=float, default=2.00)
    p.add_argument("--sleep-hard-token-weight", type=float, default=2.00)
    p.add_argument("--sleep-target-token-margin", type=float, default=0.25)
    p.add_argument("--sleep-min-token-top1", type=float, default=1.00)
    p.add_argument("--sleep-clip-grad", type=float, default=1.0)
    p.add_argument("--sleep-min-nll-gain", type=float, default=0.50)
    p.add_argument("--sleep-max-prompt-js", type=float, default=0.08)
    p.add_argument("--sleep-check-every", type=int, default=10)
    return p.parse_args()


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(name)


def load_knowledge(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []

    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                item = json.loads(raw)
            except json.JSONDecodeError as exc:
                print(f"[memory] skip invalid JSON line {line_no}: {exc}")
                continue
            prompt = str(item.get("prompt", "")).strip()
            answer = str(item.get("answer", "")).strip()
            if prompt and answer:
                rows.append({"prompt": prompt, "answer": answer})
    return rows


def validate_protected_knowledge(
    rows: List[Dict[str, str]],
) -> None:
    required = {"CPUとは", "GPUとは"}
    prompts = {row["prompt"] for row in rows}
    missing = sorted(required - prompts)
    if missing:
        raise RuntimeError(
            "Protected knowledge is missing required anchors: "
            + ", ".join(missing)
        )


def append_memory(path: Path, prompt: str, answer: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    item = {"prompt": prompt, "answer": answer}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


def encode_prompt(tokenizer: Tokenizer, text: str) -> List[int]:
    return tokenizer.encode(text, add_bos=True, add_eos=False)


@torch.no_grad()
def generate_answer(
    model: LanguageModel,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    repetition_penalty: float,
) -> str:
    prefix = encode_prompt(tokenizer, prompt)
    generated = model.generate(
        prefix,
        max_new_tokens=max_new_tokens,
        eos_id=tokenizer.eos_id,
        temperature=temperature,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
    )
    return tokenizer.decode(
        generated[len(prefix):],
        skip_special_tokens=True,
    ).strip()


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
        raise ValueError("answer token sequence is empty")

    full = prompt_ids + answer_ids
    if len(full) > model.context_length:
        # Preserve the complete answer when possible and trim old prompt tokens.
        keep = model.context_length
        full = full[-keep:]
        prompt_count = max(1, keep - len(answer_ids))
    else:
        prompt_count = len(prompt_ids)

    x = torch.tensor(
        [full[:-1]],
        dtype=torch.long,
        device=device,
    )
    targets = torch.tensor(
        [full[1:]],
        dtype=torch.long,
        device=device,
    )

    logits = model(x)

    # Target positions corresponding to the continuation.  Because targets are
    # shifted by one, the first answer token is predicted from the last prompt
    # token at index prompt_count - 1.
    start = max(0, prompt_count - 1)
    logits = logits[:, start:, :]
    targets = targets[:, start:]

    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
    )


def continuation_margin(
    model: LanguageModel,
    tokenizer: Tokenizer,
    prompt: str,
    canonical: str,
    confuser: str,
) -> torch.Tensor:
    """Return canonical score minus confuser score.

    continuation_nll() is mean token NLL, so:
      score = -NLL
      margin = score(canonical) - score(confuser)
             = NLL(confuser) - NLL(canonical)
    Positive margin means the canonical continuation is preferred.
    """
    canonical_nll = continuation_nll(
        model, tokenizer, prompt, canonical
    )
    confuser_nll = continuation_nll(
        model, tokenizer, prompt, confuser
    )
    return confuser_nll - canonical_nll


def canonical_token_margin_stats(
    model: LanguageModel,
    tokenizer: Tokenizer,
    prompt: str,
    answer: str,
    target_margin: float,
    hard_token_weight: float = 0.0,
):
    """Teacher-forced token-level argmax margin for the canonical answer.

    For every canonical continuation token:
        token_margin = target_logit - max(non_target_logits)

    The differentiable loss pushes every canonical token above its strongest
    local competitor.  top1_ratio == 1.0 means the complete teacher-forced
    canonical path is locally greedy-compatible.
    """
    device = next(model.parameters()).device

    prompt_ids = encode_prompt(tokenizer, prompt)
    answer_ids = tokenizer.encode(answer, add_bos=False, add_eos=True)
    if not answer_ids:
        raise ValueError("answer token sequence is empty")

    full = prompt_ids + answer_ids
    if len(full) > model.context_length:
        keep = model.context_length
        full = full[-keep:]
        prompt_count = max(1, keep - len(answer_ids))
    else:
        prompt_count = len(prompt_ids)

    x = torch.tensor([full[:-1]], dtype=torch.long, device=device)
    targets = torch.tensor([full[1:]], dtype=torch.long, device=device)

    logits = model(x)
    start = max(0, prompt_count - 1)
    logits = logits[:, start:, :]
    targets = targets[:, start:]

    target_logits = logits.gather(
        dim=-1,
        index=targets.unsqueeze(-1),
    ).squeeze(-1)

    competitor_logits = logits.clone()
    competitor_logits.scatter_(
        dim=-1,
        index=targets.unsqueeze(-1),
        value=float("-inf"),
    )
    strongest_competitor = competitor_logits.max(dim=-1).values

    margins = target_logits - strongest_competitor
    margin_target = torch.as_tensor(
        target_margin,
        dtype=margins.dtype,
        device=margins.device,
    )
    hinge = F.relu(margin_target - margins)
    mean_hinge = hinge.mean()
    worst_hinge = hinge.max()
    loss = mean_hinge + hard_token_weight * worst_hinge

    top1_ratio = (margins >= 0.0).float().mean()
    min_margin = margins.min()
    mean_margin = margins.mean()

    return loss, top1_ratio, min_margin, mean_margin


@torch.no_grad()
def mean_token_margin_metrics(
    model: LanguageModel,
    tokenizer: Tokenizer,
    memory: List[Dict[str, str]],
    target_margin: float,
):
    ratios = []
    min_margins = []
    mean_margins = []

    for item in memory:
        _, ratio, min_margin, mean_margin = canonical_token_margin_stats(
            model,
            tokenizer,
            item["prompt"],
            item["answer"],
            target_margin,
            0.0,
        )
        ratios.append(float(ratio.item()))
        min_margins.append(float(min_margin.item()))
        mean_margins.append(float(mean_margin.item()))

    return {
        "top1_ratio": sum(ratios) / max(1, len(ratios)),
        "min_margin": min(min_margins) if min_margins else float("-inf"),
        "mean_margin": (
            sum(mean_margins) / max(1, len(mean_margins))
        ),
    }


@torch.no_grad()
def print_hard_token_diagnostics(
    model: LanguageModel,
    tokenizer: Tokenizer,
    memory: List[Dict[str, str]],
    limit: int = 8,
) -> None:
    """Print the worst canonical token decisions under teacher forcing."""
    for item in memory:
        device = next(model.parameters()).device
        prompt_ids = encode_prompt(tokenizer, item["prompt"])
        answer_ids = tokenizer.encode(
            item["answer"],
            add_bos=False,
            add_eos=True,
        )
        full = prompt_ids + answer_ids
        if len(full) > model.context_length:
            keep = model.context_length
            full = full[-keep:]
            prompt_count = max(1, keep - len(answer_ids))
        else:
            prompt_count = len(prompt_ids)

        x = torch.tensor([full[:-1]], dtype=torch.long, device=device)
        targets = torch.tensor([full[1:]], dtype=torch.long, device=device)
        logits = model(x)
        start = max(0, prompt_count - 1)
        logits = logits[:, start:, :]
        targets = targets[:, start:]

        target_logits = logits.gather(
            -1, targets.unsqueeze(-1)
        ).squeeze(-1)
        competitor = logits.clone()
        competitor.scatter_(
            -1, targets.unsqueeze(-1), float("-inf")
        )
        comp_values, comp_ids = competitor.max(dim=-1)
        margins = target_logits - comp_values

        rows = []
        for i in range(margins.size(1)):
            target_id = int(targets[0, i].item())
            comp_id = int(comp_ids[0, i].item())
            rows.append(
                (
                    float(margins[0, i].item()),
                    i,
                    tokenizer.decode([target_id]),
                    tokenizer.decode([comp_id]),
                )
            )
        rows.sort(key=lambda row: row[0])

        print(f"HARD> prompt={item['prompt']!r}")
        for margin, pos, target_tok, comp_tok in rows[:limit]:
            print(
                f"  pos={pos:02d} target={target_tok!r} "
                f"competitor={comp_tok!r} margin={margin:+.4f}"
            )


@torch.no_grad()
def capture_confusers(
    reference: LanguageModel,
    tokenizer: Tokenizer,
    memory: List[Dict[str, str]],
) -> List[str]:
    confusers = []
    for item in memory:
        text = generate_answer(
            model=reference,
            tokenizer=tokenizer,
            prompt=item["prompt"],
            max_new_tokens=48,
            temperature=0.0,
            top_k=40,
            repetition_penalty=1.10,
        )
        if not text:
            text = "。"
        confusers.append(text)
    return confusers


@torch.no_grad()
def mean_memory_margin(
    model: LanguageModel,
    tokenizer: Tokenizer,
    memory: List[Dict[str, str]],
    confusers: List[str],
) -> float:
    values = []
    for item, confuser in zip(memory, confusers):
        margin = continuation_margin(
            model,
            tokenizer,
            item["prompt"],
            item["answer"],
            confuser,
        )
        values.append(float(margin.item()))
    return sum(values) / max(1, len(values))


def next_logits(
    model: LanguageModel,
    tokenizer: Tokenizer,
    prompt: str,
) -> torch.Tensor:
    device = next(model.parameters()).device
    ids = encode_prompt(tokenizer, prompt)
    ids = ids[-model.context_length:]
    x = torch.tensor([ids], dtype=torch.long, device=device)
    return model(x)[0, -1, :]


def kl_to_reference(
    model: LanguageModel,
    reference: LanguageModel,
    tokenizer: Tokenizer,
    prompts: List[str],
) -> torch.Tensor:
    device = next(model.parameters()).device
    losses = []

    for prompt in prompts:
        current_logits = next_logits(model, tokenizer, prompt)
        with torch.no_grad():
            reference_logits = next_logits(reference, tokenizer, prompt)

        ref_prob = F.softmax(reference_logits, dim=-1)
        current_log = F.log_softmax(current_logits, dim=-1)
        ref_log = F.log_softmax(reference_logits, dim=-1)

        losses.append(
            torch.sum(ref_prob * (ref_log - current_log))
        )

    if not losses:
        return torch.zeros((), device=device)

    return torch.stack(losses).mean()


def mean_memory_nll(
    model: LanguageModel,
    tokenizer: Tokenizer,
    memory: List[Dict[str, str]],
) -> float:
    if not memory:
        return float("nan")

    model.eval()
    values = []
    with torch.no_grad():
        for item in memory:
            value = continuation_nll(
                model,
                tokenizer,
                item["prompt"],
                item["answer"],
            )
            values.append(float(value.item()))
    return sum(values) / len(values)


def mean_prompt_js(
    before: LanguageModel,
    after: LanguageModel,
    tokenizer: Tokenizer,
    prompts: List[str],
) -> float:
    values = []
    with torch.no_grad():
        for prompt in prompts:
            a = next_logits(before, tokenizer, prompt)
            b = next_logits(after, tokenizer, prompt)

            pa = F.softmax(a, dim=-1).clamp_min(1.0e-12)
            pb = F.softmax(b, dim=-1).clamp_min(1.0e-12)
            m = 0.5 * (pa + pb)

            js = 0.5 * torch.sum(pa * (torch.log(pa) - torch.log(m)))
            js += 0.5 * torch.sum(pb * (torch.log(pb) - torch.log(m)))
            values.append(float(js.item()))

    return sum(values) / max(1, len(values))


@torch.no_grad()
def capture_replay_targets(
    reference: LanguageModel,
    tokenizer: Tokenizer,
    prompts: List[str],
) -> List[Dict[str, str]]:
    rows = []
    for prompt in prompts:
        answer = generate_answer(
            model=reference,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=48,
            temperature=0.0,
            top_k=40,
            repetition_penalty=1.10,
        )
        if answer:
            rows.append({"prompt": prompt, "answer": answer})
    return rows


def replay_loss(
    model: LanguageModel,
    tokenizer: Tokenizer,
    replay_rows: List[Dict[str, str]],
) -> torch.Tensor:
    device = next(model.parameters()).device
    if not replay_rows:
        return torch.zeros((), device=device)

    losses = [
        continuation_nll(
            model,
            tokenizer,
            item["prompt"],
            item["answer"],
        )
        for item in replay_rows
    ]
    return torch.stack(losses).mean()


@torch.no_grad()
def mean_replay_nll(
    model: LanguageModel,
    tokenizer: Tokenizer,
    replay_rows: List[Dict[str, str]],
) -> float:
    if not replay_rows:
        return 0.0
    values = [
        float(
            continuation_nll(
                model,
                tokenizer,
                item["prompt"],
                item["answer"],
            ).item()
        )
        for item in replay_rows
    ]
    return sum(values) / len(values)


def apply_repetition_penalty_to_logits(
    logits: torch.Tensor,
    seen_ids: List[int],
    penalty: float,
) -> torch.Tensor:
    """Differentiably apply the same repetition penalty used by generate()."""
    adjusted = logits.clone()
    if penalty == 1.0:
        return adjusted

    unique_ids = sorted({
        int(token_id)
        for token_id in seen_ids
        if 0 <= int(token_id) < adjusted.numel()
    })
    if not unique_ids:
        return adjusted

    index = torch.tensor(
        unique_ids,
        dtype=torch.long,
        device=adjusted.device,
    )
    values = adjusted.index_select(0, index)
    penalized = torch.where(
        values >= 0,
        values / penalty,
        values * penalty,
    )
    adjusted = adjusted.index_copy(0, index, penalized)
    return adjusted


def runtime_replay_margin_stats(
    model: LanguageModel,
    tokenizer: Tokenizer,
    replay_rows: List[Dict[str, str]],
    repetition_penalty: float,
    target_margin: float,
):
    """Preserve source greedy decisions under the real generation rule.

    Replay targets were captured from the source model with temperature=0 and
    the same repetition penalty.  For every answer position, the source next
    token is forced to stay above the strongest competing token after applying
    that penalty to the current model logits.
    """
    device = next(model.parameters()).device
    losses = []
    all_margins = []

    for item in replay_rows:
        prompt_ids = encode_prompt(tokenizer, item["prompt"])
        answer_ids = tokenizer.encode(
            item["answer"],
            add_bos=False,
            add_eos=False,
        )
        if not answer_ids:
            continue

        full = prompt_ids + answer_ids
        x = torch.tensor(
            [full[:-1]],
            dtype=torch.long,
            device=device,
        )
        logits_all = model(x)[0]

        prompt_count = len(prompt_ids)
        for answer_pos, target_id in enumerate(answer_ids):
            full_pos = prompt_count - 1 + answer_pos
            if full_pos >= logits_all.size(0):
                break

            prefix_end = prompt_count + answer_pos
            prefix = full[:prefix_end]
            if len(prefix) > model.context_length:
                # The vectorized pass above no longer matches generate() once
                # the rolling context window truncates.  Skip those late
                # positions rather than train on a mismatched trajectory.
                break

            logits = apply_repetition_penalty_to_logits(
                logits_all[full_pos],
                prefix,
                repetition_penalty,
            )
            target_logit = logits[int(target_id)]
            competitor = logits.clone()
            competitor[int(target_id)] = float("-inf")
            strongest = competitor.max()
            margin = target_logit - strongest
            all_margins.append(margin)

            margin_target = torch.as_tensor(
                target_margin,
                dtype=margin.dtype,
                device=margin.device,
            )
            losses.append(F.relu(margin_target - margin))

    if not losses:
        zero = torch.zeros((), device=device)
        return zero, zero, zero, zero

    loss = torch.stack(losses).mean()
    margins = torch.stack(all_margins)
    top1 = (margins >= 0.0).float().mean()
    return loss, top1, margins.min(), margins.mean()


@torch.no_grad()
def mean_runtime_replay_metrics(
    model: LanguageModel,
    tokenizer: Tokenizer,
    replay_rows: List[Dict[str, str]],
    repetition_penalty: float,
    target_margin: float,
):
    _, top1, min_margin, mean_margin = runtime_replay_margin_stats(
        model,
        tokenizer,
        replay_rows,
        repetition_penalty,
        target_margin,
    )
    return {
        "top1_ratio": float(top1.item()),
        "min_margin": float(min_margin.item()),
        "mean_margin": float(mean_margin.item()),
    }


def freeze_for_sleep(model: LanguageModel) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False

    for parameter in model.final_norm.parameters():
        parameter.requires_grad = True

    for parameter in model.lm_head.parameters():
        parameter.requires_grad = True


def save_sleep_checkpoint(
    output: Path,
    model: LanguageModel,
    source_checkpoint: Dict[str, object],
    source_path: Path,
    memory_path: Path,
    memory_count: int,
    epochs: int,
    before_nll: float,
    after_nll: float,
    prompt_js: float,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = dict(source_checkpoint)
    checkpoint["model_state_dict"] = model.state_dict()
    checkpoint["loss"] = after_nll
    checkpoint["sleep"] = {
        "version": "v0.16.7",
        "source_checkpoint": str(source_path),
        "memory_file": str(memory_path),
        "protected_file": str(protected_path),
        "memory_count": memory_count,
        "epochs": epochs,
        "before_memory_nll": before_nll,
        "after_memory_nll": after_nll,
        "protected_prompt_js": prompt_js,
        "trainable": ["final_norm", "lm_head"],
        "status": "SLEEP_CANDIDATE",
    }
    torch.save(checkpoint, output)


def run_sleep(
    model: LanguageModel,
    checkpoint: Dict[str, object],
    tokenizer: Tokenizer,
    memory: List[Dict[str, str]],
    memory_path: Path,
    protected_path: Path,
    source_path: Path,
    output_path: Path,
    epochs: int,
    lr_final_norm: float,
    lr_lm_head: float,
    lambda_kl: float,
    protected_rows: List[Dict[str, str]],
    protected_nll_weight: float,
    protected_token_weight: float,
    protected_hard_weight: float,
    protected_target_margin: float,
    min_protected_top1: float,
    max_protected_nll_delta: float,
    margin_weight: float,
    target_margin: float,
    token_margin_weight: float,
    hard_token_weight: float,
    target_token_margin: float,
    min_token_top1: float,
    clip_grad: float,
    min_nll_gain: float,
    max_prompt_js: float,
    check_every: int,
) -> Tuple[LanguageModel, Dict[str, object], Path]:
    if not memory:
        print("SLEEP> no Semantic Memory entries; nothing to internalize")
        return model, checkpoint, source_path

    device = next(model.parameters()).device

    # Keep an immutable pre-sleep reference for preservation.
    reference = copy.deepcopy(model).to(device)
    reference.eval()
    for parameter in reference.parameters():
        parameter.requires_grad = False

    before_nll = mean_memory_nll(
        model,
        tokenizer,
        memory,
    )
    validate_protected_knowledge(protected_rows)
    before_protected_nll = mean_memory_nll(
        model,
        tokenizer,
        protected_rows,
    )
    before_protected_token = mean_token_margin_metrics(
        model,
        tokenizer,
        protected_rows,
        protected_target_margin,
    )
    confusers = capture_confusers(
        reference,
        tokenizer,
        memory,
    )
    before_margin = mean_memory_margin(
        model,
        tokenizer,
        memory,
        confusers,
    )

    print("SLEEP> captured pre-sleep confusers")
    for index, (item, confuser) in enumerate(
        zip(memory, confusers), 1
    ):
        print(
            f"  {index:02d}. prompt={item['prompt']!r} "
            f"confuser={confuser!r}"
        )

    freeze_for_sleep(model)

    optimizer = torch.optim.AdamW(
        [
            {
                "params": list(model.final_norm.parameters()),
                "lr": lr_final_norm,
            },
            {
                "params": list(model.lm_head.parameters()),
                "lr": lr_lm_head,
            },
        ],
        weight_decay=0.0,
    )

    print()
    print("=" * 72)
    print(" LLM_SEM v0.16.7 /sleep")
    print("=" * 72)
    print("Memory entries       :", len(memory))
    print("Epochs               :", epochs)
    print("LR final_norm        :", lr_final_norm)
    print("LR lm_head           :", lr_lm_head)
    print("KL preservation      :", lambda_kl)
    print("Protected entries    :", len(protected_rows))
    print("Protected NLL wt     :", protected_nll_weight)
    print("Protected token wt   :", protected_token_weight)
    print("Protected hard wt    :", protected_hard_weight)
    print("Protected margin     :", protected_target_margin)
    print("Min protected top1   :", min_protected_top1)
    print("Max protected dNLL   :", max_protected_nll_delta)
    print("Sequence margin wt   :", margin_weight)
    print("Sequence target      :", target_margin)
    print("Token margin wt      :", token_margin_weight)
    print("Hard-token weight    :", hard_token_weight)
    print("Token target margin  :", target_token_margin)
    print("Min token top1       :", min_token_top1)
    print("Min NLL gain target  :", min_nll_gain)
    print("Max protected JS     :", max_prompt_js)
    print("Check every          :", check_every)
    print("Trainable            : final_norm + lm_head")
    print(f"Memory NLL before    : {before_nll:.6f}")
    print(f"Protected NLL before : {before_protected_nll:.6f}")
    print(
        "Protected top1 before: "
        f"{before_protected_token['top1_ratio']:.1%}"
    )
    print(
        "Protected min margin : "
        f"{before_protected_token['min_margin']:+.6f}"
    )
    before_token = mean_token_margin_metrics(
        model,
        tokenizer,
        memory,
        target_token_margin,
    )
    print(f"Memory margin before : {before_margin:+.6f}")
    print(
        "Token top1 before    : "
        f"{before_token['top1_ratio']:.1%}"
    )
    print(
        "Min token margin     : "
        f"{before_token['min_margin']:+.6f}"
    )
    print()

    best_state = copy.deepcopy(model.state_dict())
    best_nll = before_nll
    best_js = 0.0
    best_epoch = 0
    stop_reason = "MAX_EPOCHS"

    model.train()

    for epoch in range(1, epochs + 1):
        total_target = 0.0
        total_margin = 0.0
        total_token_margin = 0.0
        total_protected_nll = 0.0
        total_protected_token = 0.0
        total_kl = 0.0

        for item, confuser in zip(memory, confusers):
            optimizer.zero_grad(set_to_none=True)

            target_loss = continuation_nll(
                model,
                tokenizer,
                item["prompt"],
                item["answer"],
            )
            margin = continuation_margin(
                model,
                tokenizer,
                item["prompt"],
                item["answer"],
                confuser,
            )
            margin_loss = F.relu(
                torch.as_tensor(
                    target_margin,
                    dtype=margin.dtype,
                    device=margin.device,
                ) - margin
            )
            (
                token_margin_loss,
                _token_top1,
                _token_min_margin,
                _token_mean_margin,
            ) = canonical_token_margin_stats(
                model,
                tokenizer,
                item["prompt"],
                item["answer"],
                target_token_margin,
                hard_token_weight,
            )
            preserve_loss = kl_to_reference(
                model,
                reference,
                tokenizer,
                PROTECTED_PROMPTS,
            )
            protected_nll_loss = torch.stack([
                continuation_nll(
                    model,
                    tokenizer,
                    protected["prompt"],
                    protected["answer"],
                )
                for protected in protected_rows
            ]).mean()
            protected_token_losses = [
                canonical_token_margin_stats(
                    model,
                    tokenizer,
                    protected["prompt"],
                    protected["answer"],
                    protected_target_margin,
                    protected_hard_weight,
                )[0]
                for protected in protected_rows
            ]
            protected_token_loss = torch.stack(
                protected_token_losses
            ).mean()

            loss = (
                target_loss
                + margin_weight * margin_loss
                + token_margin_weight * token_margin_loss
                + protected_nll_weight * protected_nll_loss
                + protected_token_weight * protected_token_loss
                + lambda_kl * preserve_loss
            )
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                [
                    p
                    for p in model.parameters()
                    if p.requires_grad
                ],
                clip_grad,
            )
            optimizer.step()

            total_target += float(target_loss.item())
            total_margin += float(margin_loss.item())
            total_token_margin += float(token_margin_loss.item())
            total_protected_nll += float(protected_nll_loss.item())
            total_protected_token += float(protected_token_loss.item())
            total_kl += float(preserve_loss.item())

        should_check = (
            epoch == 1
            or epoch == epochs
            or epoch % max(1, check_every) == 0
        )

        if should_check:
            model.eval()
            current_nll = mean_memory_nll(
                model,
                tokenizer,
                memory,
            )
            current_js = mean_prompt_js(
                reference,
                model,
                tokenizer,
                PROTECTED_PROMPTS,
            )
            current_margin = mean_memory_margin(
                model,
                tokenizer,
                memory,
                confusers,
            )
            current_protected_nll = mean_memory_nll(
                model,
                tokenizer,
                protected_rows,
            )
            protected_delta = (
                current_protected_nll - before_protected_nll
            )
            protected_metrics = mean_token_margin_metrics(
                model,
                tokenizer,
                protected_rows,
                protected_target_margin,
            )
            token_metrics = mean_token_margin_metrics(
                model,
                tokenizer,
                memory,
                target_token_margin,
            )
            gain = before_nll - current_nll
            count = len(memory)

            print(
                f"epoch={epoch:3d}/{epochs} "
                f"target_nll={total_target / count:.6f} "
                f"eval_nll={current_nll:.6f} "
                f"gain={gain:+.6f} "
                f"margin={current_margin:+.6f} "
                f"margin_loss={total_margin / count:.6f} "
                f"token_top1={token_metrics['top1_ratio']:.1%} "
                f"token_min={token_metrics['min_margin']:+.4f} "
                f"token_loss={total_token_margin / count:.6f} "
                f"protected_nll={current_protected_nll:.6f} "
                f"protected_delta={protected_delta:+.6f} "
                f"protected_top1={protected_metrics['top1_ratio']:.1%} "
                f"protected_min={protected_metrics['min_margin']:+.4f} "
                f"protected_nll_loss={total_protected_nll / count:.6f} "
                f"protected_token_loss={total_protected_token / count:.6f} "
                f"preserve_kl={total_kl / count:.6f} "
                f"prompt_js={current_js:.6f}"
            )

            if (
                current_js <= max_prompt_js
                and protected_delta <= max_protected_nll_delta
                and protected_metrics["top1_ratio"] >= min_protected_top1
                and protected_metrics["min_margin"] >= 0.0
                and current_nll < best_nll
            ):
                best_nll = current_nll
                best_js = current_js
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())

            if (
                current_js > max_prompt_js
                or protected_delta > max_protected_nll_delta
            ):
                stop_reason = "PRESERVATION_LIMIT"
                print(
                    "SLEEP> preservation/protected limit reached; "
                    "restoring best safe checkpoint"
                )
                break

            if (
                gain >= min_nll_gain
                and current_margin >= target_margin
                and token_metrics["top1_ratio"] >= min_token_top1
                and token_metrics["min_margin"] >= target_token_margin
                and protected_delta <= max_protected_nll_delta
                and protected_metrics["top1_ratio"] >= min_protected_top1
                and protected_metrics["min_margin"] >= protected_target_margin
            ):
                stop_reason = "GREEDY_CANONICAL_TARGET_REACHED"
                print(
                    "SLEEP> NLL, sequence margin, and token-level "
                    "greedy targets reached; stopping"
                )
                break

            model.train()

    model.load_state_dict(best_state)
    model.eval()

    after_nll = mean_memory_nll(
        model,
        tokenizer,
        memory,
    )
    after_margin = mean_memory_margin(
        model,
        tokenizer,
        memory,
        confusers,
    )
    prompt_js = mean_prompt_js(
        reference,
        model,
        tokenizer,
        PROTECTED_PROMPTS,
    )
    after_protected_nll = mean_memory_nll(
        model,
        tokenizer,
        protected_rows,
    )
    protected_delta = (
        after_protected_nll - before_protected_nll
    )
    after_protected_token = mean_token_margin_metrics(
        model,
        tokenizer,
        protected_rows,
        protected_target_margin,
    )
    after_token = mean_token_margin_metrics(
        model,
        tokenizer,
        memory,
        target_token_margin,
    )

    save_sleep_checkpoint(
        output=output_path,
        model=model,
        source_checkpoint=checkpoint,
        source_path=source_path,
        memory_path=memory_path,
        protected_path=protected_path,
        memory_count=len(memory),
        epochs=best_epoch,
        before_nll=before_nll,
        after_nll=after_nll,
        prompt_js=prompt_js,
    )

    # Reload exactly what was serialized and use it for subsequent chat.
    reloaded, new_checkpoint = LanguageModel.load_checkpoint(
        str(output_path),
        device=device,
    )
    reloaded.eval()

    print()
    print("SLEEP RESULT")
    print("-" * 72)
    print(f"Memory NLL           : {before_nll:.6f} -> {after_nll:.6f}")
    print(f"NLL gain             : {before_nll - after_nll:+.6f}")
    print(
        f"Decoder margin       : "
        f"{before_margin:+.6f} -> {after_margin:+.6f}"
    )
    print(
        "Canonical token top1 : "
        f"{after_token['top1_ratio']:.1%}"
    )
    print(
        "Min token margin     : "
        f"{after_token['min_margin']:+.6f}"
    )
    print(
        "Mean token margin    : "
        f"{after_token['mean_margin']:+.6f}"
    )
    print(
        f"Protected NLL        : "
        f"{before_protected_nll:.6f} -> {after_protected_nll:.6f}"
    )
    print(f"Protected NLL delta  : {protected_delta:+.6f}")
    print(
        "Protected token top1 : "
        f"{after_protected_token['top1_ratio']:.1%}"
    )
    print(
        "Protected min margin : "
        f"{after_protected_token['min_margin']:+.6f}"
    )
    print(f"Protected prompt JS  : {prompt_js:.6f}")
    print_hard_token_diagnostics(
        model,
        tokenizer,
        memory,
    )
    print("Selected epoch       :", best_epoch)
    print("Stop reason          :", stop_reason)
    print("Candidate checkpoint :", output_path)
    print("Status               : SLEEP_CANDIDATE")
    print(
        "Note                 : run the v0.15.7.x validation suite "
        "before formal promotion"
    )
    print()

    return reloaded, new_checkpoint, output_path


def print_help() -> None:
    print(
        """
Commands:
  /teach <prompt> => <answer>
      Add one persistent Semantic Memory item.

  /memory
      Show current Semantic Memory.

  /sleep [epochs]
      Internalize current Semantic Memory into final_norm + lm_head.
      The result is saved as a candidate checkpoint and becomes the live model.

  /model
      Show the currently loaded checkpoint.

  /reload
      Reload the current checkpoint from disk.

  /help
      Show this help.

  /quit
      Exit.
""".strip()
    )


def main():
    args = parse_args()
    device = choose_device(args.device)

    model_path = Path(args.model)
    tokenizer_path = Path(args.tokenizer)
    memory_path = Path(args.memory)
    protected_path = Path(args.protected)
    sleep_output = Path(args.sleep_output)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            "Run the v0.15.7.5 promotion gate first or pass --model."
        )
    if not tokenizer_path.exists():
        raise FileNotFoundError(tokenizer_path)

    tokenizer = Tokenizer.load(str(tokenizer_path))
    model, checkpoint = LanguageModel.load_checkpoint(
        str(model_path),
        device=device,
    )
    model.eval()

    current_model_path = model_path
    memory = load_knowledge(memory_path)
    protected_rows = load_knowledge(protected_path)
    validate_protected_knowledge(protected_rows)

    print("=" * 72)
    print(" LLM_SEM Chat - v0.16.7 Semantic Memory /sleep")
    print("=" * 72)
    print("Device          :", device)
    if device.type == "cuda":
        print("GPU             :", torch.cuda.get_device_name(device))
    print("Model           :", current_model_path)
    print("Tokenizer       :", tokenizer_path)
    print("Parameters      :", f"{model.parameter_count:,}")
    print("Context length  :", model.context_length)
    print("Semantic memory :", memory_path)
    print("Memory entries  :", len(memory))
    print("Protected file  :", protected_path)
    print("Protected count :", len(protected_rows))
    print("Sleep output    :", sleep_output)
    print()
    print("Commands: /teach, /memory, /sleep, /model, /reload, /help, /quit")
    print()

    while True:
        try:
            raw = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not raw:
            continue

        if raw in {"/quit", "/exit", "quit", "exit"}:
            break

        if raw == "/help":
            print_help()
            continue

        if raw == "/memory":
            memory = load_knowledge(memory_path)
            if not memory:
                print("MEM> empty")
            else:
                print(f"MEM> {len(memory)} entries")
                for index, item in enumerate(memory, 1):
                    print(
                        f"  {index:02d}. {item['prompt']} => {item['answer']}"
                    )
            continue

        if raw.startswith("/teach"):
            body = raw[len("/teach"):].strip()
            if "=>" not in body:
                print("MEM> usage: /teach <prompt> => <answer>")
                continue

            prompt, answer = body.split("=>", 1)
            prompt = prompt.strip()
            answer = answer.strip()

            if not prompt or not answer:
                print("MEM> prompt and answer must both be non-empty")
                continue

            append_memory(memory_path, prompt, answer)
            memory = load_knowledge(memory_path)
            print(
                f"MEM> stored #{len(memory)}: "
                f"{prompt} => {answer}"
            )
            continue

        if raw.startswith("/sleep"):
            tail = raw[len("/sleep"):].strip()
            epochs = args.sleep_epochs
            if tail:
                try:
                    epochs = int(tail)
                    if epochs <= 0:
                        raise ValueError
                except ValueError:
                    print("SLEEP> usage: /sleep [positive_epoch_count]")
                    continue

            memory = load_knowledge(memory_path)
            model, checkpoint, current_model_path = run_sleep(
                model=model,
                checkpoint=checkpoint,
                tokenizer=tokenizer,
                memory=memory,
                memory_path=memory_path,
                protected_path=protected_path,
                source_path=current_model_path,
                output_path=sleep_output,
                epochs=epochs,
                lr_final_norm=args.sleep_lr_final_norm,
                lr_lm_head=args.sleep_lr_lm_head,
                lambda_kl=args.sleep_kl,
                protected_rows=protected_rows,
                protected_nll_weight=args.sleep_protected_nll_weight,
                protected_token_weight=args.sleep_protected_token_weight,
                protected_hard_weight=args.sleep_protected_hard_weight,
                protected_target_margin=args.sleep_protected_target_margin,
                min_protected_top1=args.sleep_min_protected_top1,
                max_protected_nll_delta=args.sleep_max_protected_nll_delta,
                margin_weight=args.sleep_margin_weight,
                target_margin=args.sleep_target_margin,
                token_margin_weight=args.sleep_token_margin_weight,
                hard_token_weight=args.sleep_hard_token_weight,
                target_token_margin=args.sleep_target_token_margin,
                min_token_top1=args.sleep_min_token_top1,
                clip_grad=args.sleep_clip_grad,
                min_nll_gain=args.sleep_min_nll_gain,
                max_prompt_js=args.sleep_max_prompt_js,
                check_every=args.sleep_check_every,
            )
            continue

        if raw == "/model":
            print("MODEL>", current_model_path)
            sleep_meta = checkpoint.get("sleep")
            promotion_meta = checkpoint.get("promotion")
            if promotion_meta:
                print(
                    "MODEL> promotion=",
                    promotion_meta.get("status"),
                    promotion_meta.get("version"),
                )
            if sleep_meta:
                print(
                    "MODEL> sleep=",
                    sleep_meta.get("status"),
                    "entries=",
                    sleep_meta.get("memory_count"),
                    "epochs=",
                    sleep_meta.get("epochs"),
                )
            continue

        if raw == "/reload":
            model, checkpoint = LanguageModel.load_checkpoint(
                str(current_model_path),
                device=device,
            )
            model.eval()
            print("MODEL> reloaded", current_model_path)
            continue

        answer = generate_answer(
            model=model,
            tokenizer=tokenizer,
            prompt=raw,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
        )
        print("LLM>", answer)


if __name__ == "__main__":
    main()
