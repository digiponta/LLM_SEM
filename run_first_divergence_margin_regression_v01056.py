from __future__ import annotations

import torch

from model import LanguageModel
from semantic_guided_answer_finetune_v097 import (
    build_first_divergence_specs,
    first_divergence_margin_loss,
    first_divergence_margin_report,
)
from semantic_guided_answer_finetune_v097 import load_dataset
from tokenizer import Tokenizer


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer.load("model/tokenizer.json")
    model, _ = LanguageModel.load_checkpoint(
        "model/model-sem-sleep-v0144.pt",
        device=device,
    )
    rows = [
        row for row in load_dataset(
            __import__("pathlib").Path("data/semantic_sleep_qa_v0106.json")
        )
        if bool(row.get("must_train", False))
        and str((row.get("concepts") or [""])[0]) == "量子センサー"
    ]
    # Use a direct runtime-like prompt if available; otherwise the helper
    # will build the semantic prompt in its normal path.
    specs = build_first_divergence_specs(
        model,
        tokenizer,
        rows,
    )

    checks = [
        ("specs-present", len(specs) > 0),
        (
            "different-target-competitor",
            all(int(x["target_id"]) != int(x["competitor_id"]) for x in specs),
        ),
    ]

    if specs:
        report = first_divergence_margin_report(
            model, tokenizer, specs, device
        )
        loss = first_divergence_margin_loss(
            model,
            tokenizer,
            specs,
            device,
            margin=1.0,
        )
        checks.extend([
            ("report-count", len(report) == len(specs)),
            ("finite-loss", torch.isfinite(loss).item()),
            ("nonnegative-loss", float(loss.item()) >= 0.0),
        ])
        for item in report:
            print(
                f"{item['query']!r}: idx={item['divergence_index']} "
                f"target={item['target_token']!r} "
                f"competitor={item['competitor_token']!r} "
                f"margin={item['margin']:+.6f}"
            )
        print("margin loss:", f"{float(loss.item()):.6f}")

    failed = 0
    print()
    for name, ok in checks:
        failed += int(not ok)
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")

    print()
    print("RESULT:", "PASS" if failed == 0 else "FAIL")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
