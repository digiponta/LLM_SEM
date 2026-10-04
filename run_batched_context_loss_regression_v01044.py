from __future__ import annotations

import copy

import torch
import torch.nn.functional as F

from model import LanguageModel
from tokenizer import Tokenizer
from semantic_guided_answer_finetune_v097 import context_window_answer_loss


def reference_loss(model, tokenizer, prompt, answer, device):
    prompt_ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
    answer_ids = tokenizer.encode(answer, add_bos=False, add_eos=True)
    full = prompt_ids + answer_ids

    losses = []
    first_target = len(prompt_ids)
    context_length = max(1, int(model.context_length))

    for target_pos in range(first_target, len(full)):
        start = max(0, target_pos - context_length)
        prefix = full[start:target_pos]
        if not prefix:
            continue
        x = torch.tensor([prefix], dtype=torch.long, device=device)
        target = torch.tensor([full[target_pos]], dtype=torch.long, device=device)
        logits = model(x)[0, -1, :].unsqueeze(0)
        losses.append(F.cross_entropy(logits, target))

    if not losses:
        return torch.tensor(0.0, device=device, requires_grad=True)
    return torch.stack(losses).mean()


def grad_vector(model):
    rows = []
    for p in model.parameters():
        if p.grad is not None:
            rows.append(p.grad.detach().reshape(-1).cpu())
    return torch.cat(rows) if rows else torch.zeros(1)


def main():
    torch.manual_seed(1234)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = Tokenizer()
    tokenizer.fit("質問:量子通信とは分類:computer回答:量子通信は量子状態を利用する技術である。")

    base = LanguageModel(
        vocab_size=tokenizer.vocab_size,
        d_model=16,
        num_layers=1,
        hidden_dim=32,
        context_length=8,
    ).to(device)

    ref_model = copy.deepcopy(base)
    batched_model = copy.deepcopy(base)

    prompt = "質問:量子通信とは\n分類:computer\n回答:"
    answer = "量子通信は量子状態を利用する技術である。"

    ref = reference_loss(ref_model, tokenizer, prompt, answer, device)
    ref.backward()
    ref_grad = grad_vector(ref_model)

    batched = context_window_answer_loss(
        batched_model,
        tokenizer,
        prompt,
        answer,
        device,
    )
    batched.backward()
    batched_grad = grad_vector(batched_model)

    loss_diff = abs(float(ref.item()) - float(batched.item()))
    grad_diff = float((ref_grad - batched_grad).abs().max().item())

    checks = [
        ("loss-equivalent", loss_diff < 1.0e-6),
        ("gradient-equivalent", grad_diff < 1.0e-5),
        ("context-shorter-than-sequence", base.context_length < len(tokenizer.encode(prompt + answer))),
    ]

    for name, ok in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")

    print(f"reference loss : {float(ref.item()):.9f}")
    print(f"batched loss   : {float(batched.item()):.9f}")
    print(f"loss diff      : {loss_diff:.12f}")
    print(f"max grad diff  : {grad_diff:.12f}")
    print()
    passed = all(ok for _, ok in checks)
    print("RESULT:", "PASS" if passed else "FAIL")
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
