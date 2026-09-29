from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

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
DEFAULT_MEMORY_SIM_THRESHOLD = 0.90


def classify_gate(ranked, thresholds):
    top1 = ranked[0]
    top2 = ranked[1] if len(ranked) > 1 else None
    margin = top1.similarity - (top2.similarity if top2 else -1.0)
    if top1.similarity < float(thresholds["similarity_threshold"]):
        return "UNKNOWN_KNOWLEDGE", top1, margin
    if margin < float(thresholds["margin_threshold"]):
        return "GATE_REVIEW", top1, margin
    return "ACCEPT", top1, margin


def build_memory_vectors(router, memory_rows):
    return [(row, router._encode_tensor(row.text)) for row in memory_rows]


def nearest_memory(router, text, memory_vectors):
    if not memory_vectors:
        return None

    query = router._encode_tensor(text)
    ranked = []

    for row, vector in memory_vectors:
        similarity = float(F.cosine_similarity(query, vector, dim=0).item())
        ranked.append((similarity, row.label, row.text))

    ranked.sort(key=lambda x: x[0], reverse=True)
    best = ranked[0]
    second = ranked[1] if len(ranked) > 1 else None
    return best, second


def memory_margin_text(best_sim, second):
    if second is None:
        return "N/A"
    return f"{best_sim - second[0]:.6f}"


def main():
    p = argparse.ArgumentParser(
        description="LLM_SEM v0.3.3 Adaptive/Base Agreement Gate shell."
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
        help="Minimum cosine similarity for adaptive-memory consideration.",
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
        memory_vectors = build_memory_vectors(router, adaptive)
        return adaptive, thresholds, memory_vectors

    adaptive, thresholds, memory_vectors = rebuild()

    print()
    print("============================================================")
    print(" LLM_SEM v0.3.3 Adaptive/Base Agreement Gate")
    print("============================================================")
    print("Device             :", device)
    if device.type == "cuda":
        print("GPU                :", torch.cuda.get_device_name(0))
    print("Checkpoint loss    :", checkpoint.get("loss"))
    print("Adaptive samples   :", len(adaptive))
    print("Memory similarity  :", f"{args.memory_sim_th:.3f}")
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
            for row in adaptive:
                print(f"  {row.label}: {row.text}")
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
            adaptive, thresholds, memory_vectors = rebuild()
            print("Learned." if added else "Already learned.")
            continue

        last_text = text

        taught = exact_memory_label(memory_path, text)
        if taught is not None:
            print(
                f"SEM> ACCEPT  label={taught} "
                "sim=1.000000 margin=N/A"
            )
            print("SEM> Routed by exact adaptive memory.")
            continue

        ranked = router.route(text)
        base_gate, base_top1, base_margin = classify_gate(ranked, thresholds)

        memory_match = nearest_memory(router, text, memory_vectors)
        if memory_match is not None:
            best, second = memory_match
            best_sim, best_label, best_text = best

            if best_sim >= args.memory_sim_th:
                margin_text = memory_margin_text(best_sim, second)

                if best_label == base_top1.label:
                    print(
                        f"SEM> ACCEPT  label={best_label} "
                        f"mem_sim={best_sim:.6f} mem_margin={margin_text} "
                        f"base_sim={base_top1.similarity:.6f}"
                    )
                    print(
                        "SEM> Adaptive/Base agreement: "
                        f"{best_label} == {base_top1.label}"
                    )
                    print(
                        "SEM> Memory source: "
                        f"{best_text!r}"
                    )
                    continue

                print(
                    f"SEM> GATE_REVIEW  memory={best_label} "
                    f"mem_sim={best_sim:.6f} mem_margin={margin_text} "
                    f"base={base_top1.label} "
                    f"base_sim={base_top1.similarity:.6f}"
                )
                print(
                    "SEM> Adaptive/Base disagreement; "
                    "manual review or additional teaching recommended."
                )
                continue

            print(
                f"MEM> candidate={best_label} "
                f"sim={best_sim:.6f} below-threshold"
            )

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
