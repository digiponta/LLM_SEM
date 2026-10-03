# run_semantic_guided_answer_regression_v096.py
#
# LLM_SEM v0.9.6
# Regression for semantic-guided answer generation.

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from chat import build_semantic_generation_prompt, generate_answer
from model import LanguageModel
from tokenizer import Tokenizer


def parse_args():
    p = argparse.ArgumentParser(
        description="LLM_SEM v0.9.6 semantic-guided answer regression"
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
    args.answer = True

    tests = [
        {
            "text": "CPUとは",
            "label": "computer",
            "gate": "ACCEPT",
            "intent": "definition",
            "concepts": ["CPU"],
            "truth": None,
        },
        {
            "text": "宇宙とは",
            "label": "science",
            "gate": "GATE_REVIEW",
            "intent": "definition",
            "concepts": ["宇宙"],
            "truth": None,
        },
        {
            "text": "暗号",
            "label": "computer",
            "gate": "ACCEPT",
            "intent": "unspecified",
            "concepts": ["暗号"],
            "truth": {"truth_status": "UNVERIFIED"},
        },
    ]

    print("=" * 92)
    print(" LLM_SEM v0.9.6 Semantic-Guided Answer Regression")
    print("=" * 92)
    print("Device :", device)
    if device.type == "cuda":
        print("GPU    :", torch.cuda.get_device_name(0))
    print("Model  :", args.model)
    print("Loss   :", checkpoint.get("loss"))
    print()

    failures = 0
    for case in tests:
        prompt = build_semantic_generation_prompt(
            case["text"],
            selected_label=case["label"],
            gate=case["gate"],
            intent=case["intent"],
            concepts=case["concepts"],
            truth_record=case["truth"],
        )
        prompt_ok = (
            f"質問:{case['text']}" in prompt
            and f"分類:{case['label']}" in prompt
            and "回答:" in prompt
        )

        answer, mode = generate_answer(
            model,
            tokenizer,
            case["text"],
            args,
            selected_label=case["label"],
            gate=case["gate"],
            intent=case["intent"],
            concepts=case["concepts"],
            truth_record=case["truth"],
        )
        answer_ok = (
            mode == "semantic-guided"
            and bool(answer.strip())
            and answer != "(generation produced no visible tokens)"
        )
        ok = prompt_ok and answer_ok
        failures += int(not ok)

        print(f"[{'PASS' if ok else 'FAIL'}] {case['text']}")
        print("  mode  :", mode)
        print("  prompt:")
        for line in prompt.splitlines():
            print("   ", line)
        print("  AI>   ", answer)

    print()
    print("RESULT:", "PASS" if failures == 0 else "FAIL")
    raise SystemExit(0 if failures == 0 else 1)


if __name__ == "__main__":
    main()
