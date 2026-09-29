from __future__ import annotations

import argparse
from pathlib import Path
import torch

from adaptive_semantic_learning import (
    append_semantic_memory,
    exact_memory_label,
    load_semantic_memory,
    merge_samples,
)
from model import LanguageModel
from semantic_eval import load_benchmark
from semantic_router import SemanticRouter, get_policy_thresholds
from tokenizer import Tokenizer

DEFAULT_MODEL = "model/model-gpu-v0.4.pt"
DEFAULT_TOKENIZER = "model/tokenizer.json"
DEFAULT_BENCHMARK = "my_benchmark.csv"
DEFAULT_UNKNOWN_BENCHMARK = "unknown_benchmark.csv"
DEFAULT_MEMORY = "data/semantic_memory.jsonl"


def classify_gate(ranked, thresholds):
    top1 = ranked[0]
    top2 = ranked[1] if len(ranked) > 1 else None
    margin = top1.similarity - (top2.similarity if top2 else -1.0)
    if top1.similarity < float(thresholds["similarity_threshold"]):
        return "UNKNOWN_KNOWLEDGE", top1, margin
    if margin < float(thresholds["margin_threshold"]):
        return "GATE_REVIEW", top1, margin
    return "ACCEPT", top1, margin


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    p.add_argument("--benchmark", default=DEFAULT_BENCHMARK)
    p.add_argument("--unknown-benchmark", default=DEFAULT_UNKNOWN_BENCHMARK)
    p.add_argument("--memory", default=DEFAULT_MEMORY)
    p.add_argument("--policy", default="balanced",
                   choices=["known-first", "balanced", "discovery-first"])
    p.add_argument("--alpha", type=float, default=0.35)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer.load(args.tokenizer)
    model, checkpoint = LanguageModel.load_checkpoint(args.model, device=device)

    base = load_benchmark(args.benchmark)
    unknown = load_benchmark(args.unknown_benchmark)
    memory_path = Path(args.memory)
    router = SemanticRouter(model, tokenizer, alpha=args.alpha)
    last_text = None

    def rebuild():
        adaptive = load_semantic_memory(memory_path)
        samples = merge_samples(base, adaptive)
        router.fit(samples)
        thresholds = get_policy_thresholds(router, samples, unknown, args.policy)
        return adaptive, thresholds

    adaptive, thresholds = rebuild()

    print()
    print("============================================================")
    print(" LLM_SEM v0.3.1 Adaptive Semantic Learning")
    print("============================================================")
    print("Device          :", device)
    if device.type == "cuda":
        print("GPU             :", torch.cuda.get_device_name(0))
    print("Checkpoint loss :", checkpoint.get("loss"))
    print("Adaptive samples:", len(adaptive))
    print()
    print("Commands: /teach <label>, /memory, /quit")
    print()

    while True:
        text = input("You> ").strip()
        if not text:
            continue
        if text in ("/quit", "/exit", "quit", "exit"):
            break
        if text == "/memory":
            adaptive = load_semantic_memory(memory_path)
            print("Adaptive samples:", len(adaptive))
            continue
        if text.startswith("/teach"):
            parts = text.split(maxsplit=1)
            if len(parts) != 2 or last_text is None:
                print("Usage: /teach <label>")
                continue
            added = append_semantic_memory(
                memory_path, parts[1].strip(), last_text, source="chat-manual"
            )
            adaptive, thresholds = rebuild()
            print("Learned." if added else "Already learned.")
            continue

        last_text = text

        taught = exact_memory_label(memory_path, text)
        if taught is not None:
            print(
                f"SEM> ACCEPT  label={taught} "
                "sim=1.000000 margin=1.000000"
            )
            print("SEM> Routed by explicit adaptive memory.")
            continue

        ranked = router.route(text)
        gate, top1, margin = classify_gate(ranked, thresholds)
        print(
            f"SEM> {gate}  label={top1.label} "
            f"sim={top1.similarity:.6f} margin={margin:.6f}"
        )
        if gate == "ACCEPT":
            print(f"SEM> Routed to semantic class: {top1.label}")
        elif gate == "UNKNOWN_KNOWLEDGE":
            print("SEM> Unknown semantic region. Teach with: /teach <label>")
        else:
            print("SEM> Ambiguous semantic region.")


if __name__ == "__main__":
    main()
