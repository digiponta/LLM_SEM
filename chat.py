#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LLM_SEM v0.18.11 interactive chat with Stable NDC Runtime + Semantic Memory /sleep.

Commands
--------
/help
/ndc <text>
/teach <prompt> => <answer>
/memory
/protected
/propteach <statement>
/prop <subject>
/props
/repair [epochs]
/sleep [epochs]
/model
/reload
/quit

/sleep performs monitored decoder-only internalization:
  - trainable: final_norm + lm_head
  - target: taught prompt/answer NLL
  - crossing: canonical-vs-pre-sleep-confuser sequence margin
  - greedy alignment: canonical token vs strongest local competitor margin
  - retention/correction: validated protected knowledge
    (NLL + teacher-forced margin + runtime-aligned greedy margin)
  - output: model/model-sem-sleep-v0174.pt
  - memory remains on disk after sleep for auditability

This is an experimental online internalization path.  It does not run the full
v0.15.7.x promotion suite automatically.
"""

from __future__ import annotations

import argparse
import copy
import json
import unicodedata
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from model import LanguageModel
from tokenizer import Tokenizer
from semantic_proposition_v0180 import (
    add_statement,
    compose_subject,
    load_propositions,
)
from ndc import classify_memory, enrich_memory_item
from ndc_runtime_v01811 import StableNDCRouter
from ndc_hierarchy_v01812 import HierarchicalNDCRouter
from ndc_hierarchy_beam_v01813 import BeamHierarchicalNDCRouter
from ndc_hierarchy_rescue_v01814 import RescueBeamNDCRouter
from ndc_hierarchy_stable_v01815 import StableHybridNDCRouter


DEFAULT_BASE_MODEL = "model/model-sem-internalized-v01575.pt"
DEFAULT_NDC_MODEL = DEFAULT_BASE_MODEL
DEFAULT_TOKENIZER = "model/tokenizer.json"
DEFAULT_MEMORY = "data/semantic_memory_v0160.jsonl"
DEFAULT_PROTECTED = "data/protected_knowledge_v0167.jsonl"
DEFAULT_SLEEP_MODEL = "model/model-sem-sleep-v0174.pt"
DEFAULT_REPAIR_MODEL = "model/model-sem-canonical-base-v0174.pt"
DEFAULT_MODEL_STATE = "data/runtime_model_state_v0174.json"
DEFAULT_GATE_BASE_MODEL = "model/model-sem-canonical-base-v0172.pt"
DEFAULT_GATE_INTERNALIZED_MODEL = "model/model-sem-sleep-v0172.pt"
DEFAULT_GATE_THRESHOLD = 0.069273
DEFAULT_UNKNOWN_THRESHOLD = 0.943319
DEFAULT_UNKNOWN_RESPONSE = "その質問については、現在の知識では確実に答えられません。"
DEFAULT_PROPOSITION_STORE = "data/semantic_propositions_v0180.jsonl"

GATE_POSITIVE_SEEDS = [
    "量子センサーとは",
    "量子センサとは",
    "量子センサーとは。",
    "量子センサとは？",
]

GATE_NEGATIVE_SEEDS = [
    "量子通信とは",
    "量子コンピュータとは",
    "量子暗号とは",
    "暗号",
    "文学とは",
    "ブラックホールとは",
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
    "CPUとは",
    "GPUとは",
    "CPUの役割",
    "GPUの役割",
]


def parse_args():
    p = argparse.ArgumentParser(
        description="LLM_SEM v0.18.1 NDC Chat + /sleep internalization"
    )
    p.add_argument(
        "--model",
        default=None,
        help=(
            "Explicit startup checkpoint. If omitted, v0.18.0 restores the "
            "last persisted live model."
        ),
    )
    p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    p.add_argument(
        "--ndc-model",
        default=DEFAULT_NDC_MODEL,
        help="Frozen checkpoint used by the stable v0.18.11 NDC runtime router.",
    )
    p.add_argument("--memory", default=DEFAULT_MEMORY)
    p.add_argument("--protected", default=DEFAULT_PROTECTED)
    p.add_argument("--propositions", default=DEFAULT_PROPOSITION_STORE)
    p.add_argument("--sleep-output", default=DEFAULT_SLEEP_MODEL)
    p.add_argument("--repair-output", default=DEFAULT_REPAIR_MODEL)
    p.add_argument("--model-state", default=DEFAULT_MODEL_STATE)
    p.add_argument("--gate-base-model", default=DEFAULT_GATE_BASE_MODEL)
    p.add_argument(
        "--gate-internalized-model",
        default=DEFAULT_GATE_INTERNALIZED_MODEL,
    )
    p.add_argument(
        "--gate-threshold",
        type=float,
        default=DEFAULT_GATE_THRESHOLD,
    )
    p.add_argument(
        "--unknown-threshold",
        type=float,
        default=DEFAULT_UNKNOWN_THRESHOLD,
    )
    p.add_argument(
        "--unknown-response",
        default=DEFAULT_UNKNOWN_RESPONSE,
    )
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--max-new-tokens", type=int, default=96)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--repetition-penalty", type=float, default=1.10)

    # v0.18.0 treats validated protected knowledge as authoritative
    # multi-task supervision, not as a source-model preservation constraint.
    p.add_argument("--sleep-epochs", type=int, default=600)
    p.add_argument("--sleep-lr-final-norm", type=float, default=5.0e-4)
    p.add_argument("--sleep-lr-lm-head", type=float, default=2.0e-4)
    # The source model is known to be wrong on several protected prompts.
    # Do not pull the corrected model back toward those wrong distributions.
    p.add_argument("--sleep-kl", type=float, default=0.0)
    p.add_argument("--sleep-protected-nll-weight", type=float, default=1.00)
    p.add_argument("--sleep-protected-token-weight", type=float, default=3.00)
    p.add_argument("--sleep-protected-hard-weight", type=float, default=2.00)
    p.add_argument("--sleep-protected-target-margin", type=float, default=0.05)
    p.add_argument("--sleep-protected-runtime-weight", type=float, default=4.00)
    p.add_argument("--sleep-protected-runtime-margin", type=float, default=0.10)
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

    # v0.18.0 Phase A: repair the validated canonical base first.
    p.add_argument("--repair-epochs", type=int, default=1600)
    p.add_argument("--repair-lr-final-norm", type=float, default=5.0e-4)
    p.add_argument("--repair-lr-lm-head", type=float, default=2.0e-4)
    p.add_argument("--repair-lr-last-block", type=float, default=5.0e-5)
    p.add_argument("--repair-unfreeze-top1", type=float, default=0.95)
    p.add_argument("--repair-nll-weight", type=float, default=1.0)
    p.add_argument("--repair-runtime-weight", type=float, default=4.0)
    p.add_argument("--repair-runtime-hard-weight", type=float, default=0.50)
    p.add_argument("--repair-hard-start-top1", type=float, default=0.98)
    p.add_argument("--repair-runtime-margin", type=float, default=0.05)
    p.add_argument("--repair-check-every", type=int, default=10)
    return p.parse_args()


def save_runtime_model_state(
    state_path: Path,
    model_path: Path,
    source: str,
) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": "v0.18.0",
        "model": str(model_path),
        "source": source,
    }
    with state_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def resolve_startup_model(
    explicit_model: str | None,
    state_path: Path,
) -> Tuple[Path, str]:
    if explicit_model:
        return Path(explicit_model), "explicit"

    if state_path.exists():
        try:
            with state_path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            saved = Path(str(payload.get("model", "")))
            if saved.exists():
                return saved, "persisted"
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass

    # Migration convenience for users coming from v0.17.2.
    legacy_candidates = [
        Path("model/model-sem-sleep-v0172.pt"),
        Path("model/model-sem-canonical-base-v0172.pt"),
    ]
    for candidate in legacy_candidates:
        if candidate.exists():
            return candidate, "legacy-auto"

    return Path(DEFAULT_BASE_MODEL), "base-fallback"



def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(name)


def load_knowledge(path: Path) -> List[Dict[str, object]]:
    """Load knowledge and attach NDC metadata without rewriting old files.

    Historical Semantic Memory rows containing only prompt/answer remain valid.
    Missing NDC fields are classified on read; newly taught rows persist them.
    UNKNOWN is represented separately from NDC 000.
    """
    if not path.exists():
        return []

    rows: List[Dict[str, object]] = []
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
                item["prompt"] = prompt
                item["answer"] = answer
                rows.append(enrich_memory_item(item))
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


def append_memory(path: Path, prompt: str, answer: str) -> Dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    ndc = classify_memory(prompt, answer)
    item: Dict[str, object] = {
        "prompt": prompt,
        "answer": answer,
        "ndc_code": ndc.code,
        "ndc_main": ndc.main,
        "ndc_name": ndc.name,
        "classification_state": ndc.state,
        "ndc_source": ndc.source,
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
    return item


RUNTIME_TRAILING_PUNCTUATION = "、，,。．.!！?？:：;；"


def normalize_runtime_prompt(text: str) -> str:
    """Normalize harmless surface variants before runtime generation.

    Training/protected prompts are stored without sentence-final punctuation.
    Keep the semantic content intact while removing only trailing punctuation
    and normalizing Unicode width/compatibility forms.
    """
    normalized = unicodedata.normalize("NFKC", text).strip()
    normalized = normalized.rstrip(RUNTIME_TRAILING_PUNCTUATION).strip()
    return normalized


def encode_prompt(tokenizer: Tokenizer, text: str) -> List[int]:
    return tokenizer.encode(text, add_bos=True, add_eos=False)


@torch.no_grad()
def gate_semantic_vector(
    model: LanguageModel,
    tokenizer: Tokenizer,
    text: str,
) -> torch.Tensor:
    ids = tokenizer.encode(text, add_bos=True, add_eos=False)
    ids = ids[-model.context_length:]
    x = torch.tensor(
        [ids],
        dtype=torch.long,
        device=next(model.parameters()).device,
    )
    vector = model.encode_semantic(
        x,
        pooling="mean",
    )[0]
    return F.normalize(vector, p=2, dim=-1)


@torch.no_grad()
def build_internalized_gate(
    model: LanguageModel,
    tokenizer: Tokenizer,
) -> Tuple[torch.Tensor, torch.Tensor]:
    positive = torch.stack([
        gate_semantic_vector(model, tokenizer, text)
        for text in GATE_POSITIVE_SEEDS
    ]).mean(dim=0)
    negative = torch.stack([
        gate_semantic_vector(model, tokenizer, text)
        for text in GATE_NEGATIVE_SEEDS
    ]).mean(dim=0)
    return (
        F.normalize(positive, p=2, dim=-1),
        F.normalize(negative, p=2, dim=-1),
    )


@torch.no_grad()
def internalized_gate_score(
    model: LanguageModel,
    tokenizer: Tokenizer,
    prompt: str,
    positive_centroid: torch.Tensor,
    negative_centroid: torch.Tensor,
) -> Dict[str, float]:
    vector = gate_semantic_vector(model, tokenizer, prompt)
    pos_sim = float(
        F.cosine_similarity(
            vector.unsqueeze(0),
            positive_centroid.unsqueeze(0),
        ).item()
    )
    neg_sim = float(
        F.cosine_similarity(
            vector.unsqueeze(0),
            negative_centroid.unsqueeze(0),
        ).item()
    )
    return {
        "positive_similarity": pos_sim,
        "negative_similarity": neg_sim,
        "margin": pos_sim - neg_sim,
    }


@torch.no_grad()
def build_canonical_known_anchors(
    model: LanguageModel,
    tokenizer: Tokenizer,
    protected_rows: List[Dict[str, str]],
) -> Tuple[List[str], List[torch.Tensor]]:
    prompts = [item["prompt"] for item in protected_rows]
    vectors = [
        gate_semantic_vector(model, tokenizer, prompt)
        for prompt in prompts
    ]
    return prompts, vectors


@torch.no_grad()
def canonical_known_score(
    model: LanguageModel,
    tokenizer: Tokenizer,
    prompt: str,
    anchor_prompts: List[str],
    anchor_vectors: List[torch.Tensor],
) -> Dict[str, object]:
    vector = gate_semantic_vector(model, tokenizer, prompt)
    scores = [
        float(
            F.cosine_similarity(
                vector.unsqueeze(0),
                anchor.unsqueeze(0),
            ).item()
        )
        for anchor in anchor_vectors
    ]
    best_index = max(range(len(scores)), key=scores.__getitem__)
    return {
        "score": scores[best_index],
        "nearest": anchor_prompts[best_index],
    }


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
    hard_weight: float = 0.0,
):
    """Runtime-aligned canonical token margin.

    Uses the same rolling context and repetition penalty as LanguageModel.generate().
    Short trajectories are evaluated in one vectorized forward pass.  Longer
    trajectories fall back to per-token rolling-context evaluation.
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

        if len(full) <= model.context_length:
            x = torch.tensor(
                [full[:-1]],
                dtype=torch.long,
                device=device,
            )
            logits_all = model(x)[0]
            prompt_count = len(prompt_ids)

            for answer_pos, target_id in enumerate(answer_ids):
                full_pos = prompt_count - 1 + answer_pos
                prefix = full[:prompt_count + answer_pos]
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
        else:
            # Exact generate()-aligned rolling context for long trajectories.
            generated = list(prompt_ids)
            for target_id in answer_ids:
                context = generated[-model.context_length:]
                x = torch.tensor(
                    [context],
                    dtype=torch.long,
                    device=device,
                )
                logits = model(x)[0, -1, :]
                logits = apply_repetition_penalty_to_logits(
                    logits,
                    generated,
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
                generated.append(int(target_id))

    if not losses:
        zero = torch.zeros((), device=device)
        return zero, zero, zero, zero

    hinge_losses = torch.stack(losses)
    loss = hinge_losses.mean() + hard_weight * hinge_losses.max()
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


def save_repair_checkpoint(
    output: Path,
    model: LanguageModel,
    source_checkpoint: Dict[str, object],
    source_path: Path,
    protected_path: Path,
    protected_count: int,
    epochs: int,
    before_nll: float,
    after_nll: float,
    runtime_top1: float,
    runtime_min_margin: float,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = dict(source_checkpoint)
    checkpoint["model_state_dict"] = model.state_dict()
    checkpoint["loss"] = after_nll
    checkpoint["repair"] = {
        "version": "v0.18.0",
        "source_checkpoint": str(source_path),
        "protected_file": str(protected_path),
        "protected_count": protected_count,
        "epochs": epochs,
        "before_protected_nll": before_nll,
        "after_protected_nll": after_nll,
        "runtime_top1": runtime_top1,
        "runtime_min_margin": runtime_min_margin,
        "trainable": ["final_norm", "lm_head", "last_block_when_needed"],
        "status": "CANONICAL_BASE_CANDIDATE",
    }
    torch.save(checkpoint, output)


def run_repair(
    model: LanguageModel,
    checkpoint: Dict[str, object],
    tokenizer: Tokenizer,
    protected_rows: List[Dict[str, str]],
    protected_path: Path,
    source_path: Path,
    output_path: Path,
    state_path: Path,
    epochs: int,
    lr_final_norm: float,
    lr_lm_head: float,
    lr_last_block: float,
    unfreeze_top1: float,
    nll_weight: float,
    runtime_weight: float,
    runtime_hard_weight: float,
    hard_start_top1: float,
    runtime_margin: float,
    repetition_penalty: float,
    clip_grad: float,
    check_every: int,
) -> Tuple[LanguageModel, Dict[str, object], Path]:
    validate_protected_knowledge(protected_rows)
    device = next(model.parameters()).device

    before_nll = mean_memory_nll(
        model,
        tokenizer,
        protected_rows,
    )
    before_runtime = mean_runtime_replay_metrics(
        model,
        tokenizer,
        protected_rows,
        repetition_penalty,
        runtime_margin,
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
    print(" LLM_SEM v0.18.0 /repair - Canonical Base Repair")
    print("=" * 72)
    print("Protected entries    :", len(protected_rows))
    print("Epochs               :", epochs)
    print("LR final_norm        :", lr_final_norm)
    print("LR lm_head           :", lr_lm_head)
    print("LR last block        :", lr_last_block)
    print("Unfreeze top1        :", unfreeze_top1)
    print("NLL weight           :", nll_weight)
    print("Runtime margin wt    :", runtime_weight)
    print("Runtime hard wt      :", runtime_hard_weight)
    print("Hard start top1      :", hard_start_top1)
    print("Runtime target margin:", runtime_margin)
    print("Repetition penalty   :", repetition_penalty)
    print("Trainable            : final_norm + lm_head")
    print(f"Protected NLL before : {before_nll:.6f}")
    print(
        "Runtime top1 before  : "
        f"{before_runtime['top1_ratio']:.1%}"
    )
    print(
        "Runtime min before   : "
        f"{before_runtime['min_margin']:+.6f}"
    )
    print()

    best_state = copy.deepcopy(model.state_dict())
    best_score = (
        before_nll
        + 10.0 * (1.0 - before_runtime["top1_ratio"])
        + max(0.0, -before_runtime["min_margin"])
    )
    best_epoch = 0
    stop_reason = "MAX_EPOCHS"
    hard_focus_active = False
    last_block_active = False

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        nll_loss = torch.stack([
            continuation_nll(
                model,
                tokenizer,
                item["prompt"],
                item["answer"],
            )
            for item in protected_rows
        ]).mean()

        (
            runtime_loss,
            _runtime_top1,
            _runtime_min,
            _runtime_mean,
        ) = runtime_replay_margin_stats(
            model,
            tokenizer,
            protected_rows,
            repetition_penalty,
            runtime_margin,
            runtime_hard_weight if hard_focus_active else 0.0,
        )

        loss = (
            nll_weight * nll_loss
            + runtime_weight * runtime_loss
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

        should_check = (
            epoch == 1
            or epoch == epochs
            or epoch % max(1, check_every) == 0
        )
        if not should_check:
            continue

        model.eval()
        current_nll = mean_memory_nll(
            model,
            tokenizer,
            protected_rows,
        )
        runtime_metrics = mean_runtime_replay_metrics(
            model,
            tokenizer,
            protected_rows,
            repetition_penalty,
            runtime_margin,
        )

        print(
            f"epoch={epoch:3d}/{epochs} "
            f"protected_nll={current_nll:.6f} "
            f"runtime_top1={runtime_metrics['top1_ratio']:.1%} "
            f"runtime_min={runtime_metrics['min_margin']:+.4f} "
            f"runtime_mean={runtime_metrics['mean_margin']:+.4f} "
            f"runtime_loss={float(runtime_loss.item()):.6f} "
            f"last_block={'ON' if last_block_active else 'OFF'} "
            f"hard_focus={'ON' if hard_focus_active else 'OFF'}"
        )

        if (
            not last_block_active
            and runtime_metrics["top1_ratio"] >= unfreeze_top1
        ):
            last_block_active = True
            for parameter in model.blocks[-1].parameters():
                parameter.requires_grad = True
            optimizer.add_param_group({
                "params": list(model.blocks[-1].parameters()),
                "lr": lr_last_block,
                "weight_decay": 0.0,
            })
            print(
                "REPAIR> representation stage enabled: "
                "last Transformer block unfrozen "
                f"at top1={runtime_metrics['top1_ratio']:.1%}"
            )

        if (
            not hard_focus_active
            and runtime_metrics["top1_ratio"] >= hard_start_top1
        ):
            hard_focus_active = True
            print(
                "REPAIR> curriculum switch: hard-token focus enabled "
                f"at top1={runtime_metrics['top1_ratio']:.1%}"
            )

        score = (
            current_nll
            + 10.0 * (1.0 - runtime_metrics["top1_ratio"])
            + max(0.0, -runtime_metrics["min_margin"])
        )
        if score < best_score:
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

        if (
            runtime_metrics["top1_ratio"] >= 1.0
            and runtime_metrics["min_margin"] >= runtime_margin
        ):
            stop_reason = "CANONICAL_BASE_REPAIRED"
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            print(
                "REPAIR> all protected canonical tokens are greedy-safe; "
                "stopping"
            )
            break

    model.load_state_dict(best_state)
    model.eval()

    after_nll = mean_memory_nll(
        model,
        tokenizer,
        protected_rows,
    )
    after_runtime = mean_runtime_replay_metrics(
        model,
        tokenizer,
        protected_rows,
        repetition_penalty,
        runtime_margin,
    )

    save_repair_checkpoint(
        output=output_path,
        model=model,
        source_checkpoint=checkpoint,
        source_path=source_path,
        protected_path=protected_path,
        protected_count=len(protected_rows),
        epochs=best_epoch,
        before_nll=before_nll,
        after_nll=after_nll,
        runtime_top1=after_runtime["top1_ratio"],
        runtime_min_margin=after_runtime["min_margin"],
    )

    reloaded, new_checkpoint = LanguageModel.load_checkpoint(
        str(output_path),
        device=device,
    )
    reloaded.eval()
    save_runtime_model_state(
        state_path,
        output_path,
        "repair",
    )

    print()
    print("REPAIR RESULT")
    print("-" * 72)
    print(
        f"Protected NLL        : "
        f"{before_nll:.6f} -> {after_nll:.6f}"
    )
    print(
        "Runtime greedy top1  : "
        f"{after_runtime['top1_ratio']:.1%}"
    )
    print(
        "Runtime min margin   : "
        f"{after_runtime['min_margin']:+.6f}"
    )
    print(
        "Runtime mean margin  : "
        f"{after_runtime['mean_margin']:+.6f}"
    )
    print("Selected epoch       :", best_epoch)
    print("Stop reason          :", stop_reason)
    print("Canonical base       :", output_path)
    print("Status               : CANONICAL_BASE_CANDIDATE")
    print()

    return reloaded, new_checkpoint, output_path



def save_sleep_checkpoint(
    output: Path,
    model: LanguageModel,
    source_checkpoint: Dict[str, object],
    source_path: Path,
    memory_path: Path,
    protected_path: Path,
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
        "version": "v0.18.0",
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
    state_path: Path,
    epochs: int,
    lr_final_norm: float,
    lr_lm_head: float,
    lambda_kl: float,
    protected_rows: List[Dict[str, str]],
    protected_nll_weight: float,
    protected_token_weight: float,
    protected_hard_weight: float,
    protected_target_margin: float,
    protected_runtime_weight: float,
    protected_runtime_margin: float,
    repetition_penalty: float,
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
    before_protected_runtime = mean_runtime_replay_metrics(
        model,
        tokenizer,
        protected_rows,
        repetition_penalty,
        protected_runtime_margin,
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
    print(" LLM_SEM v0.18.0 /sleep")
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
    print("Protected runtime wt :", protected_runtime_weight)
    print("Protected runtime mg :", protected_runtime_margin)
    print("Repetition penalty   :", repetition_penalty)
    print("Min protected top1   :", min_protected_top1)
    print("Max protected dNLL   :", max_protected_nll_delta, "(diagnostic only)")
    print("Sequence margin wt   :", margin_weight)
    print("Sequence target      :", target_margin)
    print("Token margin wt      :", token_margin_weight)
    print("Hard-token weight    :", hard_token_weight)
    print("Token target margin  :", target_token_margin)
    print("Min token top1       :", min_token_top1)
    print("Min NLL gain target  :", min_nll_gain)
    print("Source prompt JS     :", max_prompt_js, "(diagnostic only)")
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
    print(
        "Protected runtime t1 : "
        f"{before_protected_runtime['top1_ratio']:.1%}"
    )
    print(
        "Protected runtime min: "
        f"{before_protected_runtime['min_margin']:+.6f}"
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
    best_score = float("inf")
    best_epoch = 0
    stop_reason = "MAX_EPOCHS"

    model.train()

    for epoch in range(1, epochs + 1):
        total_target = 0.0
        total_margin = 0.0
        total_token_margin = 0.0
        total_protected_nll = 0.0
        total_protected_token = 0.0
        total_protected_runtime = 0.0
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
            (
                protected_runtime_loss,
                _protected_runtime_top1,
                _protected_runtime_min,
                _protected_runtime_mean,
            ) = runtime_replay_margin_stats(
                model,
                tokenizer,
                protected_rows,
                repetition_penalty,
                protected_runtime_margin,
            )

            loss = (
                target_loss
                + margin_weight * margin_loss
                + token_margin_weight * token_margin_loss
                + protected_nll_weight * protected_nll_loss
                + protected_token_weight * protected_token_loss
                + protected_runtime_weight * protected_runtime_loss
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
            total_protected_runtime += float(protected_runtime_loss.item())
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
            protected_runtime_metrics = mean_runtime_replay_metrics(
                model,
                tokenizer,
                protected_rows,
                repetition_penalty,
                protected_runtime_margin,
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
                f"protected_runtime_top1={protected_runtime_metrics['top1_ratio']:.1%} "
                f"protected_runtime_min={protected_runtime_metrics['min_margin']:+.4f} "
                f"protected_runtime_loss={total_protected_runtime / count:.6f} "
                f"preserve_kl={total_kl / count:.6f} "
                f"prompt_js={current_js:.6f}"
            )

            # v0.18.0: choose the best multi-task checkpoint by canonical
            # progress.  Source-model JS is diagnostic only because the source
            # answers are known to be wrong for some protected prompts.
            progress_score = (
                current_nll
                + current_protected_nll
                + 5.0 * (1.0 - token_metrics["top1_ratio"])
                + 5.0 * (1.0 - protected_metrics["top1_ratio"])
                + 10.0 * (1.0 - protected_runtime_metrics["top1_ratio"])
                + max(0.0, -token_metrics["min_margin"])
                + max(0.0, -protected_metrics["min_margin"])
                + 2.0 * max(0.0, -protected_runtime_metrics["min_margin"])
            )
            if progress_score < best_score:
                best_score = progress_score
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())

            if not torch.isfinite(
                torch.as_tensor(
                    current_nll + current_protected_nll,
                    device=device,
                )
            ):
                stop_reason = "NUMERICAL_FAILURE"
                print("SLEEP> numerical failure; restoring best checkpoint")
                break

            if (
                gain >= min_nll_gain
                and current_margin >= target_margin
                and token_metrics["top1_ratio"] >= min_token_top1
                and token_metrics["min_margin"] >= target_token_margin
                and protected_metrics["top1_ratio"] >= min_protected_top1
                and protected_metrics["min_margin"] >= protected_target_margin
                and protected_runtime_metrics["top1_ratio"] >= min_protected_top1
                and protected_runtime_metrics["min_margin"] >= protected_runtime_margin
            ):
                stop_reason = "NEW_AND_PROTECTED_CANONICAL_REACHED"
                print(
                    "SLEEP> new memory and validated protected knowledge "
                    "both reached canonical greedy targets; stopping"
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
    after_protected_runtime = mean_runtime_replay_metrics(
        model,
        tokenizer,
        protected_rows,
        repetition_penalty,
        protected_runtime_margin,
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
    save_runtime_model_state(
        state_path,
        output_path,
        "sleep",
    )

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
    print(
        "Protected runtime t1 : "
        f"{after_protected_runtime['top1_ratio']:.1%}"
    )
    print(
        "Protected runtime min: "
        f"{after_protected_runtime['min_margin']:+.6f}"
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
  /ndc <text>
      Route text with the stable v0.18.11 NDC main-class classifier +
      contrastive unknown gate.

  /ndc3 <text>
      Run the stable v0.18.16 selected 3-digit NDC router.
      This is the default production path.

  /ndc3beam <text>
      Run v0.18.13 beam hierarchical routing.

  /ndc3rescue <text>
      Run v0.18.14 rescue beam routing.

  /ndc3stable <text>
      Run v0.18.15 stable hybrid routing: semantic beam routing plus a small
      deterministic selected-code keyword prior for fine-code tie breaking.

  /teach <prompt> => <answer>
      Add one persistent Semantic Memory item.

  /memory
      Show current Semantic Memory with NDC domain metadata.

  /protected
      Show validated canonical knowledge.

  /propteach <statement>
      Decompose a canonical proposition statement into atomic propositions
      and persist them. Examples:
        /propteach XはYである。
        /propteach Xは、Yであり、Zである。

  /prop <subject>
      Compose all stored atomic propositions for one subject.

  /props
      Show all stored atomic propositions.

  /repair [epochs]
      Phase A: repair the model using protected canonical knowledge only.
      Saves model-sem-canonical-base-v0169.pt and makes it the live model.

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

    tokenizer_path = Path(args.tokenizer)
    ndc_model_path = Path(args.ndc_model)
    memory_path = Path(args.memory)
    protected_path = Path(args.protected)
    proposition_path = Path(args.propositions)
    sleep_output = Path(args.sleep_output)
    repair_output = Path(args.repair_output)
    model_state_path = Path(args.model_state)
    gate_base_path = Path(args.gate_base_model)
    gate_internalized_path = Path(args.gate_internalized_model)
    model_path, startup_source = resolve_startup_model(
        args.model,
        model_state_path,
    )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            "Run the v0.15.7.5 promotion gate first or pass --model."
        )
    if not tokenizer_path.exists():
        raise FileNotFoundError(tokenizer_path)
    if not ndc_model_path.exists():
        raise FileNotFoundError(
            f"NDC runtime model not found: {ndc_model_path}"
        )
    if not gate_base_path.exists():
        raise FileNotFoundError(
            f"Gate base model not found: {gate_base_path}"
        )
    if not gate_internalized_path.exists():
        raise FileNotFoundError(
            f"Gate internalized model not found: {gate_internalized_path}"
        )

    tokenizer = Tokenizer.load(str(tokenizer_path))
    model, checkpoint = LanguageModel.load_checkpoint(
        str(model_path),
        device=device,
    )
    model.eval()

    ndc_model, ndc_checkpoint = LanguageModel.load_checkpoint(
        str(ndc_model_path),
        device=device,
    )
    ndc_model.eval()
    for parameter in ndc_model.parameters():
        parameter.requires_grad_(False)
    ndc_router = StableNDCRouter(
        ndc_model,
        tokenizer,
    )
    ndc3_router = HierarchicalNDCRouter(
        ndc_model,
        tokenizer,
    )
    ndc3_beam_router = BeamHierarchicalNDCRouter(
        ndc_model,
        tokenizer,
    )
    ndc3_rescue_router = RescueBeamNDCRouter(
        ndc_model,
        tokenizer,
    )
    ndc3_stable_router = StableHybridNDCRouter(
        ndc_model,
        tokenizer,
    )

    gate_base_model, _gate_base_checkpoint = LanguageModel.load_checkpoint(
        str(gate_base_path),
        device=device,
    )
    gate_base_model.eval()

    gate_internalized_model, _gate_internalized_checkpoint = (
        LanguageModel.load_checkpoint(
            str(gate_internalized_path),
            device=device,
        )
    )
    gate_internalized_model.eval()

    gate_positive_centroid, gate_negative_centroid = build_internalized_gate(
        gate_base_model,
        tokenizer,
    )

    current_model_path = model_path
    memory = load_knowledge(memory_path)
    protected_rows = load_knowledge(protected_path)
    validate_protected_knowledge(protected_rows)
    canonical_anchor_prompts, canonical_anchor_vectors = (
        build_canonical_known_anchors(
            gate_base_model,
            tokenizer,
            protected_rows,
        )
    )

    print("=" * 72)
    print(" LLM_SEM Chat - v0.18.16 Stable Hierarchical NDC Runtime + Semantic Memory /sleep")
    print("=" * 72)
    print("Device          :", device)
    if device.type == "cuda":
        print("GPU             :", torch.cuda.get_device_name(device))
    print("Model           :", current_model_path)
    print("Startup source  :", startup_source)
    print("Model state     :", model_state_path)
    print("Tokenizer       :", tokenizer_path)
    print("NDC model       :", ndc_model_path)
    print("NDC ckpt loss   :", ndc_checkpoint.get("loss"))
    print("NDC runtime     : v0.18.11 contrastive stable")
    print("Parameters      :", f"{model.parameter_count:,}")
    print("Context length  :", model.context_length)
    print("Semantic memory :", memory_path)
    print("Memory entries  :", len(memory))
    print("Protected file  :", protected_path)
    print("Protected count :", len(protected_rows))
    print("Proposition db  :", proposition_path)
    print("Propositions    :", len(load_propositions(proposition_path)))
    print("Sleep output    :", sleep_output)
    print("Repair output   :", repair_output)
    print("Gate base       :", gate_base_path)
    print("Gate internal   :", gate_internalized_path)
    print("Gate pooling    : mean")
    print("Gate threshold  :", args.gate_threshold)
    print("Unknown pooling : mean")
    print("Unknown th      :", args.unknown_threshold)
    print()
    print(
        "Commands: /ndc, /ndc3, /ndc3beam, /ndc3rescue, /ndc3stable, /teach, /memory, /protected, /propteach, /prop, /props, "
        "/repair, /sleep, /model, /reload, /help, /quit"
    )
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

        if raw.startswith("/ndc3stable"):
            text = raw[len("/ndc3stable"):].strip()
            if not text:
                print("NDC3S> usage: /ndc3stable <text>")
                continue

            decision = ndc3_stable_router.route(text)
            if decision.state == "ACCEPT":
                print(
                    f"NDC3S> ACCEPT stage1={decision.stage1_main} "
                    f"code={decision.ndc_code} "
                    f"name={decision.ndc_code_name} "
                    f"rescued_main={decision.rescued_main} "
                    f"rescued_unknown={decision.rescued_unknown} "
                    f"sim={decision.code_similarity:.6f} "
                    f"beam={decision.beam_score:.6f} "
                    f"margin={decision.beam_margin:+.6f}"
                )
            elif decision.state == "MAIN_ONLY":
                print(
                    f"NDC3S> MAIN_ONLY stage1={decision.stage1_main} "
                    f"sim={decision.code_similarity:.6f} "
                    f"beam={decision.beam_score:.6f} "
                    f"margin={decision.beam_margin:+.6f}"
                )
            else:
                print(
                    f"NDC3S> UNKNOWN "
                    f"sim={decision.code_similarity:.6f} "
                    f"beam={decision.beam_score:.6f} "
                    f"margin={decision.beam_margin:+.6f}"
                )
            continue

        if raw.startswith("/ndc3rescue"):
            text = raw[len("/ndc3rescue"):].strip()
            if not text:
                print("NDC3R> usage: /ndc3rescue <text>")
                continue

            decision = ndc3_rescue_router.route(text)
            if decision.state == "ACCEPT":
                print(
                    f"NDC3R> ACCEPT stage1={decision.stage1_main} "
                    f"code={decision.ndc_code} "
                    f"name={decision.ndc_code_name} "
                    f"rescued_main={decision.rescued_main} "
                    f"rescued_unknown={decision.rescued_unknown} "
                    f"sim={decision.code_similarity:.6f} "
                    f"beam={decision.beam_score:.6f} "
                    f"margin={decision.beam_margin:+.6f}"
                )
            elif decision.state == "MAIN_ONLY":
                print(
                    f"NDC3R> MAIN_ONLY stage1={decision.stage1_main} "
                    f"sim={decision.code_similarity:.6f} "
                    f"beam={decision.beam_score:.6f} "
                    f"margin={decision.beam_margin:+.6f}"
                )
            else:
                print(
                    f"NDC3R> UNKNOWN "
                    f"sim={decision.code_similarity:.6f} "
                    f"beam={decision.beam_score:.6f} "
                    f"margin={decision.beam_margin:+.6f}"
                )
            continue

        if raw.startswith("/ndc3beam"):
            text = raw[len("/ndc3beam"):].strip()
            if not text:
                print("NDC3B> usage: /ndc3beam <text>")
                continue

            decision = ndc3_beam_router.route(text)
            if decision.state == "ACCEPT":
                print(
                    f"NDC3B> ACCEPT stage1={decision.stage1_main} "
                    f"code={decision.ndc_code} "
                    f"name={decision.ndc_code_name} "
                    f"rescued={decision.rescued_main} "
                    f"sim={decision.code_similarity:.6f} "
                    f"beam={decision.beam_score:.6f} "
                    f"margin={decision.beam_margin:+.6f}"
                )
            elif decision.state == "MAIN_ONLY":
                print(
                    f"NDC3B> MAIN_ONLY stage1={decision.stage1_main} "
                    f"sim={decision.code_similarity:.6f} "
                    f"beam={decision.beam_score:.6f} "
                    f"margin={decision.beam_margin:+.6f}"
                )
            else:
                print("NDC3B> UNKNOWN")
            continue

        if raw.startswith("/ndc3"):
            text = raw[len("/ndc3"):].strip()
            if not text:
                print("NDC3> usage: /ndc3 <text>")
                continue

            decision = ndc3_stable_router.route(text)
            if decision.state == "ACCEPT":
                print(
                    f"NDC3> ACCEPT stage1={decision.stage1_main} "
                    f"code={decision.ndc_code} "
                    f"name={decision.ndc_code_name} "
                    f"rescued_main={decision.rescued_main} "
                    f"rescued_unknown={decision.rescued_unknown} "
                    f"sim={decision.code_similarity:.6f} "
                    f"beam={decision.beam_score:.6f} "
                    f"margin={decision.beam_margin:+.6f}"
                )
            elif decision.state == "MAIN_ONLY":
                print(
                    f"NDC3> MAIN_ONLY stage1={decision.stage1_main} "
                    f"sim={decision.code_similarity:.6f} "
                    f"beam={decision.beam_score:.6f} "
                    f"margin={decision.beam_margin:+.6f}"
                )
            else:
                print(
                    f"NDC3> UNKNOWN "
                    f"sim={decision.code_similarity:.6f} "
                    f"beam={decision.beam_score:.6f} "
                    f"margin={decision.beam_margin:+.6f}"
                )
            continue

        if raw.startswith("/ndc"):
            text = raw[len("/ndc"):].strip()
            if not text:
                print("NDC> usage: /ndc <text>")
                continue

            decision = ndc_router.route(text)
            if decision.accepted:
                print(
                    f"NDC> ACCEPT main={decision.ndc_main} "
                    f"name={decision.ndc_name} "
                    f"known={decision.known_similarity:.6f} "
                    f"unknown={decision.unknown_similarity:.6f} "
                    f"contrast={decision.contrast:+.6f} "
                    f"margin={decision.margin:+.6f} "
                    f"score={decision.gate_score:.6f}"
                )
            else:
                print(
                    "NDC> UNKNOWN "
                    f"known={decision.known_similarity:.6f} "
                    f"unknown={decision.unknown_similarity:.6f} "
                    f"contrast={decision.contrast:+.6f} "
                    f"margin={decision.margin:+.6f} "
                    f"score={decision.gate_score:.6f}"
                )
            continue

        if raw == "/memory":
            memory = load_knowledge(memory_path)
            if not memory:
                print("MEM> empty")
            else:
                print(f"MEM> {len(memory)} entries")
                for index, item in enumerate(memory, 1):
                    code = item.get("ndc_code") or "---"
                    name = item.get("ndc_name") or "未分類"
                    state = item.get("classification_state") or "UNKNOWN"
                    print(
                        f"  {index:02d}. [NDC {code} {name} / {state}] "
                        f"{item['prompt']} => {item['answer']}"
                    )
            continue

        if raw == "/protected":
            print(f"PROTECTED> {len(protected_rows)} entries")
            for index, item in enumerate(protected_rows, 1):
                print(
                    f"  {index:02d}. {item['prompt']} => {item['answer']}"
                )
            continue

        if raw == "/props":
            propositions = load_propositions(proposition_path)
            if not propositions:
                print("PROP> empty")
            else:
                print(f"PROP> {len(propositions)} atomic propositions")
                for index, item in enumerate(propositions, 1):
                    print(
                        f"  {index:02d}. "
                        f"subject={item.subject!r} value={item.value!r}"
                    )
            continue

        if raw.startswith("/propteach"):
            statement = raw[len("/propteach"):].strip()
            if not statement:
                print("PROP> usage: /propteach <statement>")
                continue

            added = add_statement(proposition_path, statement)
            if not added:
                print(
                    "PROP> unsupported proposition grammar; "
                    "expected XはYである or Xは、Yであり、Zである"
                )
                continue

            print(f"PROP> decomposed {len(added)} proposition(s)")
            for item in added:
                print(
                    f"  + subject={item.subject!r} value={item.value!r}"
                )
            composed = compose_subject(
                proposition_path,
                added[0].subject,
            )
            print("PROP> composed:", composed)
            continue

        if raw.startswith("/prop "):
            subject = raw[len("/prop "):].strip()
            if not subject:
                print("PROP> usage: /prop <subject>")
                continue

            composed = compose_subject(proposition_path, subject)
            if composed:
                print("PROP>", composed)
            else:
                print(f"PROP> no propositions for {subject!r}")
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

            stored = append_memory(memory_path, prompt, answer)
            memory = load_knowledge(memory_path)
            code = stored.get("ndc_code") or "---"
            name = stored.get("ndc_name") or "未分類"
            state = stored.get("classification_state") or "UNKNOWN"
            print(
                f"MEM> stored #{len(memory)} [NDC {code} {name} / {state}]: "
                f"{prompt} => {answer}"
            )
            continue

        if raw.startswith("/repair"):
            tail = raw[len("/repair"):].strip()
            epochs = args.repair_epochs
            if tail:
                try:
                    epochs = int(tail)
                    if epochs <= 0:
                        raise ValueError
                except ValueError:
                    print("REPAIR> usage: /repair [positive_epoch_count]")
                    continue

            model, checkpoint, current_model_path = run_repair(
                model=model,
                checkpoint=checkpoint,
                tokenizer=tokenizer,
                protected_rows=protected_rows,
                protected_path=protected_path,
                source_path=current_model_path,
                output_path=repair_output,
                state_path=model_state_path,
                epochs=epochs,
                lr_final_norm=args.repair_lr_final_norm,
                lr_lm_head=args.repair_lr_lm_head,
                lr_last_block=args.repair_lr_last_block,
                unfreeze_top1=args.repair_unfreeze_top1,
                nll_weight=args.repair_nll_weight,
                runtime_weight=args.repair_runtime_weight,
                runtime_hard_weight=args.repair_runtime_hard_weight,
                hard_start_top1=args.repair_hard_start_top1,
                runtime_margin=args.repair_runtime_margin,
                repetition_penalty=args.repetition_penalty,
                clip_grad=args.sleep_clip_grad,
                check_every=args.repair_check_every,
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
                state_path=model_state_path,
                epochs=epochs,
                lr_final_norm=args.sleep_lr_final_norm,
                lr_lm_head=args.sleep_lr_lm_head,
                lambda_kl=args.sleep_kl,
                protected_rows=protected_rows,
                protected_nll_weight=args.sleep_protected_nll_weight,
                protected_token_weight=args.sleep_protected_token_weight,
                protected_hard_weight=args.sleep_protected_hard_weight,
                protected_target_margin=args.sleep_protected_target_margin,
                protected_runtime_weight=args.sleep_protected_runtime_weight,
                protected_runtime_margin=args.sleep_protected_runtime_margin,
                repetition_penalty=args.repetition_penalty,
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
            print("MODEL> state=", model_state_path)
            sleep_meta = checkpoint.get("sleep")
            repair_meta = checkpoint.get("repair")
            promotion_meta = checkpoint.get("promotion")
            if promotion_meta:
                print(
                    "MODEL> promotion=",
                    promotion_meta.get("status"),
                    promotion_meta.get("version"),
                )
            if repair_meta:
                print(
                    "MODEL> repair=",
                    repair_meta.get("status"),
                    "protected=",
                    repair_meta.get("protected_count"),
                    "epochs=",
                    repair_meta.get("epochs"),
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

        runtime_prompt = normalize_runtime_prompt(raw)
        if not runtime_prompt:
            print("LLM> empty prompt after normalization")
            continue
        if runtime_prompt != raw:
            print(f"NORM> {raw!r} -> {runtime_prompt!r}")

        gate = internalized_gate_score(
            gate_base_model,
            tokenizer,
            runtime_prompt,
            gate_positive_centroid,
            gate_negative_centroid,
        )
        use_internalized = gate["margin"] >= args.gate_threshold
        if use_internalized:
            print(
                "GATE> "
                "route=INTERNALIZED "
                f"pos={gate['positive_similarity']:.6f} "
                f"neg={gate['negative_similarity']:.6f} "
                f"margin={gate['margin']:+.6f} "
                f"threshold={args.gate_threshold:+.6f}"
            )
            answer = generate_answer(
                model=gate_internalized_model,
                tokenizer=tokenizer,
                prompt=runtime_prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                repetition_penalty=args.repetition_penalty,
            )
            print("LLM>", answer)
            continue

        known = canonical_known_score(
            gate_base_model,
            tokenizer,
            runtime_prompt,
            canonical_anchor_prompts,
            canonical_anchor_vectors,
        )
        is_known = float(known["score"]) >= args.unknown_threshold
        route_name = "CANONICAL_KNOWN" if is_known else "UNKNOWN"
        print(
            "GATE> "
            f"route={route_name} "
            f"pos={gate['positive_similarity']:.6f} "
            f"neg={gate['negative_similarity']:.6f} "
            f"margin={gate['margin']:+.6f} "
            f"threshold={args.gate_threshold:+.6f}"
        )
        print(
            "UNKNOWN> "
            f"score={float(known['score']):.6f} "
            f"threshold={args.unknown_threshold:.6f} "
            f"nearest={known['nearest']!r}"
        )

        if not is_known:
            print("LLM>", args.unknown_response)
            continue

        answer = generate_answer(
            model=gate_base_model,
            tokenizer=tokenizer,
            prompt=runtime_prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
        )
        print("LLM>", answer)


if __name__ == "__main__":
    main()
