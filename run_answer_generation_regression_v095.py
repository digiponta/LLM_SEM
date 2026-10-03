# run_answer_generation_regression_v095.py
#
# LLM_SEM v0.9.5
# Smoke regression for restored answer generation.

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from chat import generate_answer
from model import LanguageModel
from tokenizer import Tokenizer


def parse_args():
    p = argparse.ArgumentParser(
        description="LLM_SEM v0.9.5 answer-generation regression"
    )
    p.add_argument("--model", default="model/model-sem-consolidation-v081.pt")
    p.add_argument("--tokenizer", default="model/tokenizer.json")
    p.add_argument("--max-new-tokens", type=int, default=40)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--allow-cpu", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    for path in (args.model, args.tokenizer):
        if not Path(path).exists():
            raise FileNotFoundError(path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and not args.allow_cpu:
        raise RuntimeError("CUDA unavailable. Use --allow-cpu only for testing.")

    tokenizer = Tokenizer.load(args.tokenizer)
    model, checkpoint = LanguageModel.load_checkpoint(args.model, device=device)
    model.eval()

    # generate_answer expects the runtime flag to exist.
    args.answer = True

    tests = [
        "CPUとは",
        "宇宙とは",
        "暗号",
    ]

    print("=" * 84)
    print(" LLM_SEM v0.9.5 Answer Generation Regression")
    print("=" * 84)
    print("Device :", device)
    if device.type == "cuda":
        print("GPU    :", torch.cuda.get_device_name(0))
    print("Model  :", args.model)
    print("Loss   :", checkpoint.get("loss"))
    print()

    failures = 0
    for text in tests:
        answer, mode = generate_answer(model, tokenizer, text, args)
        ok = bool(answer.strip()) and answer != "(generation produced no visible tokens)"
        print(f"[{'PASS' if ok else 'FAIL'}] {text}")
        print("  AI>", answer)
        failures += int(not ok)

    print()
    print("RESULT:", "PASS" if failures == 0 else "FAIL")
    raise SystemExit(0 if failures == 0 else 1)


if __name__ == "__main__":
    main()
