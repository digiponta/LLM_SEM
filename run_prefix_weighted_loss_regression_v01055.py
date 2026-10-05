from __future__ import annotations

import torch
import torch.nn.functional as F

from model import LanguageModel
from proposition_bootstrap_v01050 import proposition_rows
from semantic_guided_answer_finetune_v097 import context_window_answer_loss
from tokenizer import Tokenizer


def old_reference_loss(model, tokenizer, prompt, answer, device):
    prompt_ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
    answer_ids = tokenizer.encode(answer, add_bos=False, add_eos=True)
    full = prompt_ids + answer_ids
    first_target = len(prompt_ids)
    context_length = max(1, int(model.context_length))

    losses = []
    for target_pos in range(first_target, len(full)):
        start = max(0, target_pos - context_length)
        prefix = full[start:target_pos]
        if not prefix:
            continue
        x = torch.tensor([prefix], dtype=torch.long, device=device)
        target = torch.tensor(
            [int(full[target_pos])],
            dtype=torch.long,
            device=device,
        )
        logits = model(x)[:, -1, :]
        losses.append(F.cross_entropy(logits, target))
    return torch.stack(losses).mean()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer.load("model/tokenizer.json")
    model, _ = LanguageModel.load_checkpoint(
        "model/model-sem-sleep-v0144.pt",
        device=device,
    )
    model.eval()

    prompt = "量子センサーとは"
    answer = "量子センサーは、量子の性質を利用する。"

    with torch.no_grad():
        old = old_reference_loss(model, tokenizer, prompt, answer, device)
        new_default = context_window_answer_loss(
            model, tokenizer, prompt, answer, device
        )
        weighted = context_window_answer_loss(
            model,
            tokenizer,
            prompt,
            answer,
            device,
            prefix_tokens=8,
            prefix_weight=4.0,
        )

    targets = [
        {
            "query": "量子センサーとは",
            "answer": answer,
            "concepts": ["量子センサー"],
            "must_train": True,
        },
        {
            "query": "量子センサー",
            "answer": answer,
            "concepts": ["量子センサー"],
            "must_train": True,
        },
    ]
    rows = proposition_rows(
        "量子センサー",
        targets,
        [
            "量子センサーは、量子の性質を利用する。",
            "量子センサーは、高感度で測定する技術である。",
        ],
    )

    checks = [
        (
            "default-loss-equivalent",
            abs(float(old.item()) - float(new_default.item())) < 1.0e-6,
        ),
        (
            "weighted-loss-different",
            abs(float(weighted.item()) - float(new_default.item())) > 1.0e-8,
        ),
        (
            "proposition-prefix-tokens",
            all(int(r.get("loss_prefix_tokens", 0)) == 8 for r in rows),
        ),
        (
            "proposition-prefix-weight",
            all(abs(float(r.get("loss_prefix_weight", 0.0)) - 4.0) < 1.0e-9 for r in rows),
        ),
    ]

    print("old reference :", f"{float(old.item()):.9f}")
    print("new default   :", f"{float(new_default.item()):.9f}")
    print("prefix weighted:", f"{float(weighted.item()):.9f}")
    print()

    failed = 0
    for name, ok in checks:
        failed += int(not ok)
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")

    print()
    print("RESULT:", "PASS" if failed == 0 else "FAIL")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
