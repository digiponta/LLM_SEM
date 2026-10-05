#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LLM_SEM v0.16.0 interactive chat with Semantic Memory /sleep.

Commands
--------
/help
/teach <prompt> => <answer>
/memory
/sleep [epochs]
/model
/reload
/quit

/sleep performs conservative decoder-only internalization:
  - trainable: final_norm + lm_head
  - target: taught prompt/answer NLL
  - preservation: KL divergence against the pre-sleep model on protected prompts
  - output: model/model-sem-sleep-v0160.pt
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
DEFAULT_SLEEP_MODEL = "model/model-sem-sleep-v0160.pt"

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
        description="LLM_SEM v0.16.0 Chat + /sleep internalization"
    )
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    p.add_argument("--memory", default=DEFAULT_MEMORY)
    p.add_argument("--sleep-output", default=DEFAULT_SLEEP_MODEL)
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--max-new-tokens", type=int, default=96)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--repetition-penalty", type=float, default=1.10)

    # Conservative online sleep defaults.  Explicit v0.15.7.x promotion
    # experiments may use more aggressive values.
    p.add_argument("--sleep-epochs", type=int, default=30)
    p.add_argument("--sleep-lr-final-norm", type=float, default=5.0e-5)
    p.add_argument("--sleep-lr-lm-head", type=float, default=2.0e-5)
    p.add_argument("--sleep-kl", type=float, default=0.50)
    p.add_argument("--sleep-clip-grad", type=float, default=1.0)
    return p.parse_args()


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(name)


def load_memory(path: Path) -> List[Dict[str, str]]:
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
        "version": "v0.16.0",
        "source_checkpoint": str(source_path),
        "memory_file": str(memory_path),
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
    source_path: Path,
    output_path: Path,
    epochs: int,
    lr_final_norm: float,
    lr_lm_head: float,
    lambda_kl: float,
    clip_grad: float,
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
    print(" LLM_SEM v0.16.0 /sleep")
    print("=" * 72)
    print("Memory entries       :", len(memory))
    print("Epochs               :", epochs)
    print("LR final_norm        :", lr_final_norm)
    print("LR lm_head           :", lr_lm_head)
    print("KL preservation      :", lambda_kl)
    print("Trainable            : final_norm + lm_head")
    print(f"Memory NLL before    : {before_nll:.6f}")
    print()

    model.train()

    for epoch in range(1, epochs + 1):
        total_target = 0.0
        total_kl = 0.0

        for item in memory:
            optimizer.zero_grad(set_to_none=True)

            target_loss = continuation_nll(
                model,
                tokenizer,
                item["prompt"],
                item["answer"],
            )
            preserve_loss = kl_to_reference(
                model,
                reference,
                tokenizer,
                PROTECTED_PROMPTS,
            )

            loss = target_loss + lambda_kl * preserve_loss
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
            total_kl += float(preserve_loss.item())

        if (
            epoch == 1
            or epoch == epochs
            or epoch % max(1, epochs // 6) == 0
        ):
            count = len(memory)
            print(
                f"epoch={epoch:3d}/{epochs} "
                f"target_nll={total_target / count:.6f} "
                f"preserve_kl={total_kl / count:.6f}"
            )

    model.eval()

    after_nll = mean_memory_nll(
        model,
        tokenizer,
        memory,
    )
    prompt_js = mean_prompt_js(
        reference,
        model,
        tokenizer,
        PROTECTED_PROMPTS,
    )

    save_sleep_checkpoint(
        output=output_path,
        model=model,
        source_checkpoint=checkpoint,
        source_path=source_path,
        memory_path=memory_path,
        memory_count=len(memory),
        epochs=epochs,
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
    print(f"Protected prompt JS  : {prompt_js:.6f}")
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
    memory = load_memory(memory_path)

    print("=" * 72)
    print(" LLM_SEM Chat - v0.16.0 Semantic Memory /sleep")
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

        if raw in {"/quit", "/exit"}:
            break

        if raw == "/help":
            print_help()
            continue

        if raw == "/memory":
            memory = load_memory(memory_path)
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
            memory = load_memory(memory_path)
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

            memory = load_memory(memory_path)
            model, checkpoint, current_model_path = run_sleep(
                model=model,
                checkpoint=checkpoint,
                tokenizer=tokenizer,
                memory=memory,
                memory_path=memory_path,
                source_path=current_model_path,
                output_path=sleep_output,
                epochs=epochs,
                lr_final_norm=args.sleep_lr_final_norm,
                lr_lm_head=args.sleep_lr_lm_head,
                lambda_kl=args.sleep_kl,
                clip_grad=args.sleep_clip_grad,
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
