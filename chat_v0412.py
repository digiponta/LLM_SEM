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
    relabel_semantic_memory,
)
from adaptive_semantic_runtime import (
    build_multi_prototypes,
    decide_adaptive_route,
    encode_memory,
)
from semantic_contrastive_enrichment_policy import (
    load_enrichment_pool,
    suggest_contrastive_enrichment,
)
from semantic_structural_counterfactual import simulate_structural_candidate
from semantic_eval import LabeledSentence, load_benchmark
from model import LanguageModel
from semantic_router import SemanticRouter, get_policy_thresholds
from tokenizer import Tokenizer

DEFAULT_MODEL = "model/model-gpu-v0.4.pt"
DEFAULT_TOKENIZER = "model/tokenizer.json"
DEFAULT_BENCHMARK = "my_benchmark.csv"
DEFAULT_UNKNOWN_BENCHMARK = "unknown_benchmark.csv"
DEFAULT_MEMORY = "data/semantic_memory.jsonl"
DEFAULT_ENRICHMENT_POOL = "semantic_memory_enrichment_train_v048.csv"


def classify_base_gate(ranked, thresholds):
    top1 = ranked[0]
    top2 = ranked[1] if len(ranked) > 1 else None
    margin = top1.similarity - (top2.similarity if top2 else -1.0)
    if top1.similarity < float(thresholds["similarity_threshold"]):
        return "UNKNOWN_KNOWLEDGE", top1, margin
    if margin < float(thresholds["margin_threshold"]):
        return "GATE_REVIEW", top1, margin
    return "ACCEPT", top1, margin


def print_structural_counterfactuals(
    router,
    text,
    decision,
    encoded_memory,
    enrichment_pool,
    *,
    prototypes_per_label,
    memory_sim_th,
    override_sim_th,
    local_k,
    local_purity_th,
):
    suggestions = suggest_contrastive_enrichment(
        router,
        text,
        decision,
        encoded_memory,
        enrichment_pool,
    )

    if not suggestions:
        print("ENRICH> No structural counterfactual candidates found.")
        return

    evaluated = []
    for item in suggestions:
        result = simulate_structural_candidate(
            router,
            text,
            decision,
            encoded_memory,
            LabeledSentence(label=item.label, text=item.text),
            prototypes_per_label=prototypes_per_label,
            base_similarity_threshold=memory_sim_th,
            override_similarity_threshold=override_sim_th,
            local_k=local_k,
            local_purity_threshold=local_purity_th,
        )
        evaluated.append((result.structural_score, item, result))

    category_rank = {
        "STRUCTURAL_IMPROVEMENT": 3,
        "PARTIAL_STRUCTURE_IMPROVEMENT": 2,
        "DECISION_ONLY_ACCEPT": 1,
        "NO_IMPROVEMENT": 0,
    }

    evaluated.sort(
        key=lambda x: (
            category_rank[x[2].category],
            x[0],
            x[1].boundary_score,
        ),
        reverse=True,
    )

    print("ENRICH> Structural counterfactual simulation:")
    for i, (_, item, result) in enumerate(evaluated, 1):
        print(
            f"  {i}. [{result.category}] "
            f"label={item.label} "
            f"struct_score={result.structural_score:+.6f} "
            f"boundary_score={item.boundary_score:.6f}"
        )
        print(f"     text={item.text}")
        print(
            "     before: "
            f"action={result.before_action} "
            f"memory={result.before_memory_label} "
            f"local={result.before_local_label} "
            f"base={result.base_label} "
            f"purity={result.before_purity:.3f}"
        )
        print(
            "     after : "
            f"action={result.after_action} "
            f"memory={result.after_memory_label} "
            f"local={result.after_local_label} "
            f"base={result.base_label} "
            f"purity={result.after_purity:.3f}"
        )
        print(
            "     delta : "
            f"decision={result.decision_gain:+.1f} "
            f"proto-base={result.prototype_base_gain:+.1f} "
            f"local-base={result.local_base_gain:+.1f} "
            f"proto-local={result.prototype_local_gain:+.1f} "
            f"purity={result.purity_gain:+.3f}"
        )


def main():
    p = argparse.ArgumentParser(
        description="LLM_SEM v0.4.12 Structural Counterfactual Runtime."
    )
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    p.add_argument("--benchmark", default=DEFAULT_BENCHMARK)
    p.add_argument("--unknown-benchmark", default=DEFAULT_UNKNOWN_BENCHMARK)
    p.add_argument("--memory", default=DEFAULT_MEMORY)
    p.add_argument("--enrichment-pool", default=DEFAULT_ENRICHMENT_POOL)
    p.add_argument("--policy", default="balanced",
                   choices=["known-first", "balanced", "discovery-first"])
    p.add_argument("--alpha", type=float, default=0.35)
    p.add_argument("--memory-sim-th", type=float, default=0.80)
    p.add_argument("--override-sim-th", type=float, default=0.92)
    p.add_argument("--local-k", type=int, default=3)
    p.add_argument("--local-purity-th", type=float, default=1.00)
    p.add_argument("--prototypes-per-label", type=int, default=2)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer.load(args.tokenizer)
    model, checkpoint = LanguageModel.load_checkpoint(args.model, device=device)

    base = load_benchmark(args.benchmark)
    unknown = load_benchmark(args.unknown_benchmark)
    memory_path = Path(args.memory)
    enrichment_pool = load_enrichment_pool(Path(args.enrichment_pool))

    router = SemanticRouter(model, tokenizer, alpha=args.alpha)
    router.fit(base)
    thresholds = get_policy_thresholds(router, base, unknown, args.policy)

    last_text = None
    last_decision = None

    def rebuild_memory():
        adaptive = load_semantic_memory(memory_path)
        encoded = encode_memory(router, adaptive)
        prototypes = build_multi_prototypes(
            encoded,
            per_label=args.prototypes_per_label,
        )
        return adaptive, encoded, prototypes

    adaptive, encoded_memory, prototypes = rebuild_memory()

    print()
    print("============================================================")
    print(" LLM_SEM v0.4.12 Structural Counterfactual Runtime")
    print("============================================================")
    print("Device               :", device)
    if device.type == "cuda":
        print("GPU                  :", torch.cuda.get_device_name(0))
    print("Checkpoint loss      :", checkpoint.get("loss"))
    print("Base Router          : FIXED (base benchmark only)")
    print("Adaptive samples     :", len(adaptive))
    print("Memory labels        :", len(prototypes))
    print("Enrichment pool      :", len(enrichment_pool))
    print("Prototypes / label   :", args.prototypes_per_label)
    print("Override sim         :", f"{args.override_sim_th:.2f}")
    print("Local k              :", args.local_k)
    print("Local purity         :", f"{args.local_purity_th:.2f}")
    print("Enrichment ranking   : structural counterfactual improvement")
    print()
    print(
        "Commands: /teach <label>, /relabel <label>, /forget, "
        "/memory, /simulate, /quit"
    )
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

        if text == "/simulate":
            if last_text is None or last_decision is None:
                print("No previous review/unknown utterance is available.")
                continue
            print_structural_counterfactuals(
                router,
                last_text,
                last_decision,
                encoded_memory,
                enrichment_pool,
                prototypes_per_label=args.prototypes_per_label,
                memory_sim_th=args.memory_sim_th,
                override_sim_th=args.override_sim_th,
                local_k=args.local_k,
                local_purity_th=args.local_purity_th,
            )
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
            adaptive, encoded_memory, prototypes = rebuild_memory()
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
            adaptive, encoded_memory, prototypes = rebuild_memory()
            print("Relabeled." if changed else "No exact memory entry to relabel.")
            continue

        if text == "/forget":
            if last_text is None:
                print("No previous utterance is available to forget.")
                continue
            removed = forget_semantic_memory(memory_path, last_text)
            adaptive, encoded_memory, prototypes = rebuild_memory()
            print("Forgotten." if removed else "No exact memory entry to forget.")
            continue

        last_text = text
        last_decision = None

        taught = exact_memory_label(memory_path, text)
        if taught is not None:
            print(
                f"SEM> ACCEPT  label={taught} "
                "mem_sim=1.000000 local=EXACT"
            )
            print("SEM> Routed by exact adaptive memory.")
            continue

        decision = decide_adaptive_route(
            router,
            text,
            encoded_memory,
            prototypes,
            base_similarity_threshold=args.memory_sim_th,
            override_similarity_threshold=args.override_sim_th,
            local_k=args.local_k,
            local_purity_threshold=args.local_purity_th,
        )
        last_decision = decision

        if decision is not None:
            margin_text = (
                "N/A"
                if decision.memory_margin is None
                else f"{decision.memory_margin:.6f}"
            )
            print(
                f"MEM> label={decision.label} "
                f"sim={decision.memory_similarity:.6f} "
                f"margin={margin_text} "
                f"local_majority={decision.local_majority_label} "
                f"purity={decision.local_purity:.3f} "
                f"k={decision.local_k} "
                f"base={decision.base_label} "
                f"base_sim={decision.base_similarity:.6f}"
            )

            if decision.action == "ACCEPT":
                print(
                    f"SEM> ACCEPT  label={decision.label} "
                    "reason=adaptive-base-agreement"
                )
                continue

            if decision.action == "ADAPTIVE_OVERRIDE":
                print(
                    f"SEM> ADAPTIVE_OVERRIDE  label={decision.label} "
                    f"mem_sim={decision.memory_similarity:.6f} "
                    f"purity={decision.local_purity:.3f} "
                    f"k={decision.local_k}"
                )
                continue

            if decision.action == "GATE_REVIEW":
                print(
                    f"SEM> GATE_REVIEW  memory={decision.label} "
                    f"local_majority={decision.local_majority_label} "
                    f"purity={decision.local_purity:.3f} "
                    f"base={decision.base_label}"
                )
                print_structural_counterfactuals(
                    router,
                    text,
                    decision,
                    encoded_memory,
                    enrichment_pool,
                    prototypes_per_label=args.prototypes_per_label,
                    memory_sim_th=args.memory_sim_th,
                    override_sim_th=args.override_sim_th,
                    local_k=args.local_k,
                    local_purity_th=args.local_purity_th,
                )
                continue

        ranked = router.route(text)
        base_gate, base_top1, base_margin = classify_base_gate(
            ranked,
            thresholds,
        )
        print(
            f"SEM> {base_gate}  label={base_top1.label} "
            f"sim={base_top1.similarity:.6f} margin={base_margin:.6f}"
        )

        if base_gate == "ACCEPT":
            print(f"SEM> Routed to base semantic class: {base_top1.label}")
        elif base_gate == "UNKNOWN_KNOWLEDGE":
            print("SEM> Unknown semantic region.")
        else:
            print("SEM> Ambiguous base semantic region.")


if __name__ == "__main__":
    main()
