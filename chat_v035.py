from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import torch

from adaptive_semantic_learning import (
    append_semantic_memory,
    exact_memory_label,
    forget_semantic_memory,
    load_semantic_memory,
    merge_samples,
    relabel_semantic_memory,
)
from model import LanguageModel
from semantic_eval import load_benchmark
from semantic_memory_prototype import (
    build_prototypes,
    prototype_margin,
    rank_prototypes,
)
from semantic_router import SemanticRouter, get_policy_thresholds
from tokenizer import Tokenizer

DEFAULT_MODEL = "model/model-gpu-v0.4.pt"
DEFAULT_TOKENIZER = "model/tokenizer.json"
DEFAULT_BENCHMARK = "my_benchmark.csv"
DEFAULT_UNKNOWN_BENCHMARK = "unknown_benchmark.csv"
DEFAULT_MEMORY = "data/semantic_memory.jsonl"
DEFAULT_MEMORY_SIM_THRESHOLD = 0.90
DEFAULT_MEMORY_MARGIN_THRESHOLD = 0.02


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
    p = argparse.ArgumentParser(
        description="LLM_SEM v0.3.5 Semantic Memory Correction shell."
    )
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    p.add_argument("--benchmark", default=DEFAULT_BENCHMARK)
    p.add_argument("--unknown-benchmark", default=DEFAULT_UNKNOWN_BENCHMARK)
    p.add_argument("--memory", default=DEFAULT_MEMORY)
    p.add_argument("--policy", default="balanced",
                   choices=["known-first", "balanced", "discovery-first"])
    p.add_argument("--alpha", type=float, default=0.35)
    p.add_argument(
        "--memory-sim-th",
        type=float,
        default=DEFAULT_MEMORY_SIM_THRESHOLD,
    )
    p.add_argument(
        "--memory-margin-th",
        type=float,
        default=DEFAULT_MEMORY_MARGIN_THRESHOLD,
    )
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
        prototypes = build_prototypes(router, adaptive)
        return adaptive, thresholds, prototypes

    adaptive, thresholds, prototypes = rebuild()

    print()
    print("============================================================")
    print(" LLM_SEM v0.3.4 Semantic Memory Prototype")
    print("============================================================")
    print("Device             :", device)
    if device.type == "cuda":
        print("GPU                :", torch.cuda.get_device_name(0))
    print("Checkpoint loss    :", checkpoint.get("loss"))
    print("Adaptive samples   :", len(adaptive))
    print("Memory labels      :", len(prototypes))
    print("Memory similarity  :", f"{args.memory_sim_th:.3f}")
    print("Memory margin      :", f"{args.memory_margin_th:.3f}")
    print()
    print("Commands: /teach <label>, /relabel <label>, /forget, /memory, /quit")
    print()

    while True:
        text = input("You> ").strip()
        if not text:
            continue
        if text in ("/quit", "/exit", "quit", "exit"):
            break

        if text == "/memory":
            adaptive = load_semantic_memory(memory_path)
            counts = Counter(row.label for row in adaptive)
            print("Adaptive samples:", len(adaptive))
            for label in sorted(counts):
                print(f"  {label}: {counts[label]} sample(s)")
            continue

        if text.startswith("/teach"):
            parts = text.split(maxsplit=1)
            if len(parts) != 2 or last_text is None:
                print("Usage: /teach <label>")
                continue

            added = append_semantic_memory(
                memory_path,
                parts[1].strip(),
                last_text,
                source="chat-manual",
            )
            adaptive, thresholds, prototypes = rebuild()
            print("Learned." if added else "Already learned.")
            continue

        if text.startswith("/relabel"):
            parts = text.split(maxsplit=1)
            if len(parts) != 2 or last_text is None:
                print("Usage: /relabel <label>")
                continue
            changed = relabel_semantic_memory(
                memory_path,
                last_text,
                parts[1].strip(),
            )
            adaptive, thresholds, prototypes = rebuild()
            print("Relabeled." if changed else "No exact memory entry to relabel.")
            continue

        if text == "/forget":
            if last_text is None:
                print("No previous utterance is available to forget.")
                continue
            removed = forget_semantic_memory(memory_path, last_text)
            adaptive, thresholds, prototypes = rebuild()
            print("Forgotten." if removed else "No exact memory entry to forget.")
            continue

        last_text = text

        taught = exact_memory_label(memory_path, text)
        if taught is not None:
            print(
                f"SEM> ACCEPT  label={taught} "
                "mem_sim=1.000000 mem_margin=EXACT"
            )
            print("SEM> Routed by exact adaptive memory.")
            continue

        ranked = router.route(text)
        base_gate, base_top1, base_margin = classify_gate(ranked, thresholds)

        proto_scores = rank_prototypes(router, text, prototypes)
        if proto_scores:
            mem_top1 = proto_scores[0]
            mem_margin = prototype_margin(proto_scores)
            margin_ok = (
                mem_margin is not None
                and mem_margin >= args.memory_margin_th
            )
            sim_ok = mem_top1.similarity >= args.memory_sim_th

            if len(proto_scores) == 1:
                margin_text = "N/A"
            else:
                margin_text = f"{mem_margin:.6f}"

            print(
                f"MEM> top1={mem_top1.label} "
                f"sim={mem_top1.similarity:.6f} "
                f"margin={margin_text} "
                f"samples={mem_top1.count}"
            )

            if sim_ok and margin_ok:
                if mem_top1.label == base_top1.label:
                    print(
                        f"SEM> ACCEPT  label={mem_top1.label} "
                        f"mem_sim={mem_top1.similarity:.6f} "
                        f"mem_margin={margin_text} "
                        f"base_sim={base_top1.similarity:.6f}"
                    )
                    print(
                        "SEM> Prototype/Base agreement: "
                        f"{mem_top1.label} == {base_top1.label}"
                    )
                    continue

                print(
                    f"SEM> GATE_REVIEW  memory={mem_top1.label} "
                    f"mem_sim={mem_top1.similarity:.6f} "
                    f"mem_margin={margin_text} "
                    f"base={base_top1.label} "
                    f"base_sim={base_top1.similarity:.6f}"
                )
                print(
                    "SEM> Prototype confident but Base disagrees; "
                    "manual review or additional teaching recommended."
                )
                continue

            if sim_ok and mem_margin is None:
                if mem_top1.label == base_top1.label:
                    print(
                        f"SEM> ACCEPT  label={mem_top1.label} "
                        f"mem_sim={mem_top1.similarity:.6f} "
                        "mem_margin=N/A "
                        f"base_sim={base_top1.similarity:.6f}"
                    )
                    print(
                        "SEM> Single-label memory; accepted only by Base agreement."
                    )
                    continue

                print(
                    f"SEM> GATE_REVIEW  memory={mem_top1.label} "
                    f"mem_sim={mem_top1.similarity:.6f} "
                    "mem_margin=N/A "
                    f"base={base_top1.label} "
                    f"base_sim={base_top1.similarity:.6f}"
                )
                print(
                    "SEM> Single-label memory and Base disagreement."
                )
                continue

            if sim_ok and not margin_ok:
                print(
                    "SEM> GATE_REVIEW  reason=low-memory-margin "
                    f"memory={mem_top1.label} "
                    f"mem_margin={margin_text}"
                )
                continue

        print(
            f"SEM> {base_gate}  label={base_top1.label} "
            f"sim={base_top1.similarity:.6f} margin={base_margin:.6f}"
        )

        if base_gate == "ACCEPT":
            print(f"SEM> Routed to semantic class: {base_top1.label}")
        elif base_gate == "UNKNOWN_KNOWLEDGE":
            print("SEM> Unknown semantic region. Teach with: /teach <label>")
        else:
            print("SEM> Ambiguous semantic region.")


if __name__ == "__main__":
    main()
