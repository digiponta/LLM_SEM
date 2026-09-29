# chat.py
#
# LLM_SEM v0.3 Adaptive Semantic Learning interactive shell.
#
# The base Transformer remains frozen. New semantic examples are stored in a
# persistent JSONL memory and centroids are immediately refit.

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from adaptive_semantic_learning import (
    append_semantic_memory,
    load_semantic_memory,
    merge_samples,
)
from model import LanguageModel
from semantic_eval import load_benchmark
from semantic_router import (
    SemanticRouter,
    get_policy_thresholds,
)
from tokenizer import Tokenizer


DEFAULT_MODEL = "model/model-gpu-v0.4.pt"
DEFAULT_TOKENIZER = "model/tokenizer.json"
DEFAULT_BENCHMARK = "my_benchmark.csv"
DEFAULT_UNKNOWN_BENCHMARK = "unknown_benchmark.csv"
DEFAULT_MEMORY = "data/semantic_memory.jsonl"
DEFAULT_POLICY = "balanced"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LLM_SEM v0.3 Adaptive Semantic Learning shell."
    )
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    p.add_argument("--benchmark", default=DEFAULT_BENCHMARK)
    p.add_argument("--unknown-benchmark", default=DEFAULT_UNKNOWN_BENCHMARK)
    p.add_argument("--memory", default=DEFAULT_MEMORY)
    p.add_argument("--policy", default=DEFAULT_POLICY,
                   choices=["known-first", "balanced", "discovery-first"])
    p.add_argument("--alpha", type=float, default=0.35)
    p.add_argument("--learn", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def classify_gate(ranked, thresholds):
    top1 = ranked[0]
    top2 = ranked[1] if len(ranked) > 1 else None
    margin = top1.similarity - (top2.similarity if top2 else -1.0)

    sim_th = float(thresholds["similarity_threshold"])
    margin_th = float(thresholds["margin_threshold"])

    if top1.similarity < sim_th:
        return "UNKNOWN_KNOWLEDGE", top1, margin
    if margin < margin_th:
        return "GATE_REVIEW", top1, margin
    return "ACCEPT", top1, margin


def main() -> None:
    args = parse_args()

    for filename, label in (
        (args.model, "Model checkpoint"),
        (args.tokenizer, "Tokenizer"),
        (args.benchmark, "Benchmark"),
        (args.unknown_benchmark, "Unknown benchmark"),
    ):
        if not Path(filename).exists():
            raise FileNotFoundError(f"{label} not found: {filename}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer.load(args.tokenizer)
    model, checkpoint = LanguageModel.load_checkpoint(args.model, device=device)

    base_samples = load_benchmark(args.benchmark)
    unknown_samples = load_benchmark(args.unknown_benchmark)
    memory_path = Path(args.memory)
    learning_enabled = bool(args.learn)
    last_text: str | None = None

    router = SemanticRouter(model, tokenizer, alpha=args.alpha)

    def rebuild():
        adaptive = load_semantic_memory(memory_path)
        samples = merge_samples(base_samples, adaptive)
        router.fit(samples)
        thresholds = get_policy_thresholds(
            router,
            samples,
            unknown_samples,
            args.policy,
        )
        return samples, adaptive, thresholds

    samples, adaptive, thresholds = rebuild()

    print()
    print("============================================================")
    print(" LLM_SEM v0.3 Adaptive Semantic Learning")
    print("============================================================")
    print("Device          :", device)
    if device.type == "cuda":
        print("GPU             :", torch.cuda.get_device_name(0))
    print("Checkpoint loss :", checkpoint.get("loss"))
    print("Policy          :", args.policy)
    print("Base samples    :", len(base_samples))
    print("Adaptive samples:", len(adaptive))
    print("Learning        :", "ON" if learning_enabled else "OFF")
    print()
    print("Commands:")
    print("  /learn on|off|status")
    print("  /teach <label>       teach the previous user utterance")
    print("  /memory               show adaptive sample count")
    print("  /quit")
    print()

    while True:
        text = input("You> ").strip()
        if not text:
            continue

        if text in ("/quit", "/exit", "quit", "exit"):
            break

        if text.startswith("/learn"):
            parts = text.split()
            if len(parts) == 1 or parts[1] == "status":
                print("Learning:", "ON" if learning_enabled else "OFF")
            elif parts[1] == "on":
                learning_enabled = True
                print("Learning: ON")
            elif parts[1] == "off":
                learning_enabled = False
                print("Learning: OFF")
            else:
                print("Usage: /learn on|off|status")
            continue

        if text == "/memory":
            adaptive = load_semantic_memory(memory_path)
            print(f"Adaptive samples: {len(adaptive)} -> {memory_path}")
            continue

        if text.startswith("/teach"):
            parts = text.split(maxsplit=1)
            if len(parts) != 2 or not parts[1].strip():
                print("Usage: /teach <label>")
                continue
            if not learning_enabled:
                print("Learning is OFF. Use /learn on first.")
                continue
            if last_text is None:
                print("No previous utterance is available to teach.")
                continue

            label = parts[1].strip()
            added = append_semantic_memory(
                memory_path,
                label=label,
                text=last_text,
                source="chat-manual",
            )
            samples, adaptive, thresholds = rebuild()
            if added:
                print(
                    f"Learned: label={label!r}, text={last_text!r} "
                    f"(adaptive samples={len(adaptive)})"
                )
            else:
                print("Already learned.")
            continue

        last_text = text
        ranked = router.route(text)
        gate, top1, margin = classify_gate(ranked, thresholds)

        print(
            f"SEM> {gate}  label={top1.label} "
            f"sim={top1.similarity:.6f} margin={margin:.6f}"
        )

        if gate == "UNKNOWN_KNOWLEDGE":
            print("SEM> Unknown semantic region. Teach with: /teach <label>")
        elif gate == "GATE_REVIEW":
            print("SEM> Ambiguous semantic region. Review or teach a better label.")
        else:
            print(f"SEM> Routed to semantic class: {top1.label}")


if __name__ == "__main__":
    main()
