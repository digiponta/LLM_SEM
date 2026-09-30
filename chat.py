# chat.py
#
# LLM_SEM v0.3 Adaptive Semantic Learning interactive shell
# with native Semantic Data v2.0 generation.
#
# The base Transformer remains frozen. New semantic examples are stored in a
# persistent JSONL memory and centroids are immediately refit. Every normal
# user query is also converted into SemanticDataV2 and summarized on screen.

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
from semantic_router import (
    SemanticRouter,
    get_policy_thresholds,
)
from semantic_runtime_v2 import from_runtime_dict, runtime_summary
from tokenizer import Tokenizer


DEFAULT_MODEL = "model/model-gpu-v0.4.pt"
DEFAULT_TOKENIZER = "model/tokenizer.json"
DEFAULT_BENCHMARK = "my_benchmark.csv"
DEFAULT_UNKNOWN_BENCHMARK = "unknown_benchmark.csv"
DEFAULT_MEMORY = "data/semantic_memory.jsonl"
DEFAULT_POLICY = "balanced"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "LLM_SEM v0.3 Adaptive Semantic Learning shell "
            "with Semantic Data v2.0 runtime integration."
        )
    )
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    p.add_argument("--benchmark", default=DEFAULT_BENCHMARK)
    p.add_argument("--unknown-benchmark", default=DEFAULT_UNKNOWN_BENCHMARK)
    p.add_argument("--memory", default=DEFAULT_MEMORY)
    p.add_argument(
        "--policy",
        default=DEFAULT_POLICY,
        choices=["known-first", "balanced", "discovery-first"],
    )
    p.add_argument("--alpha", type=float, default=0.35)
    p.add_argument(
        "--learn",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument(
        "--semantic-v2",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Automatically generate/display SemanticDataV2 for normal queries.",
    )
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


def gate_reason(gate: str, top1_similarity: float, margin: float, thresholds) -> str:
    sim_th = float(thresholds["similarity_threshold"])
    margin_th = float(thresholds["margin_threshold"])

    if gate == "UNKNOWN_KNOWLEDGE":
        return f"similarity {top1_similarity:.6f} < threshold {sim_th:.6f}"
    if gate == "GATE_REVIEW":
        return f"margin {margin:.6f} < threshold {margin_th:.6f}"
    return (
        f"similarity {top1_similarity:.6f} >= {sim_th:.6f} and "
        f"margin {margin:.6f} >= {margin_th:.6f}"
    )


def print_semantic_v2(summary: dict[str, object], runtime: dict[str, object]) -> None:
    """Print SemanticDataV2 plus the runtime evidence behind the decision."""
    confidence = summary.get("confidence")
    uncertainty = summary.get("uncertainty")
    concepts = summary.get("concepts") or []
    purpose = summary.get("purpose")
    intent = summary.get("intent")
    context = summary.get("context") or {}

    conf_text = (
        f"{float(confidence):.6f}"
        if isinstance(confidence, (int, float))
        else "n/a"
    )
    unc_text = (
        f"{float(uncertainty):.6f}"
        if isinstance(uncertainty, (int, float))
        else "n/a"
    )

    print(
        f"V2> schema={summary.get('schema_version')} "
        f"confidence={conf_text} uncertainty={unc_text}"
    )
    print(
        "V2> concepts="
        + (", ".join(str(x) for x in concepts) if concepts else "(none)")
    )
    print(
        f"V2> purpose={purpose!r}"
        + (f" intent={intent}" if intent else "")
    )

    gate_state = context.get("gate_state") if isinstance(context, dict) else None
    decision_margin = (
        context.get("decision_margin") if isinstance(context, dict) else None
    )
    if gate_state is not None or decision_margin is not None:
        print(
            f"V2> context gate={gate_state or 'n/a'} "
            f"decision_margin={decision_margin or 'n/a'}"
        )

    candidates = runtime.get("candidate_labels") or []
    scores = runtime.get("candidate_scores") or {}
    if candidates:
        print("V2> candidates:")
        for index, label in enumerate(candidates, 1):
            score = scores.get(label) if isinstance(scores, dict) else None
            score_text = (
                f"{float(score):.6f}"
                if isinstance(score, (int, float))
                else "n/a"
            )
            print(f"    {index}. {label:<12} {score_text}")

    memory_label = runtime.get("memory_label")
    base_label = runtime.get("base_label")
    base_similarity = runtime.get("base_similarity")
    selected_label = runtime.get("selected_label")
    disagreement = bool(
        memory_label is not None
        and base_label is not None
        and memory_label != base_label
    )

    base_text = str(base_label) if base_label is not None else "(none)"
    if isinstance(base_similarity, (int, float)):
        base_text += f" ({float(base_similarity):.6f})"

    print(
        f"V2> memory={memory_label or '(none)'} "
        f"base={base_text} selected={selected_label or '(none)'}"
    )
    print(f"V2> disagreement={disagreement}")
    reason = runtime.get("gate_reason")
    if reason:
        print(f"V2> gate_reason={reason}")


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
    semantic_v2_enabled = bool(args.semantic_v2)
    last_text: str | None = None
    last_semantic_v2 = None

    router = SemanticRouter(model, tokenizer, alpha=args.alpha)
    base_router = SemanticRouter(model, tokenizer, alpha=args.alpha)
    base_router.fit(base_samples)

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
    print(" LLM_SEM v0.3.1 Semantic Data v2 Runtime Evidence")
    print("============================================================")
    print("Device          :", device)
    if device.type == "cuda":
        print("GPU             :", torch.cuda.get_device_name(0))
    print("Checkpoint loss :", checkpoint.get("loss"))
    print("Policy          :", args.policy)
    print("Base samples    :", len(base_samples))
    print("Adaptive samples:", len(adaptive))
    print("Learning        :", "ON" if learning_enabled else "OFF")
    print("Semantic Data v2:", "ON" if semantic_v2_enabled else "OFF")
    print()
    print("Commands:")
    print("  /learn on|off|status")
    print("  /semantic on|off|status")
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

        if text.startswith("/semantic"):
            parts = text.split()
            if len(parts) == 1 or parts[1] == "status":
                print(
                    "Semantic Data v2:",
                    "ON" if semantic_v2_enabled else "OFF",
                )
            elif parts[1] == "on":
                semantic_v2_enabled = True
                print("Semantic Data v2: ON")
            elif parts[1] == "off":
                semantic_v2_enabled = False
                print("Semantic Data v2: OFF")
            else:
                print("Usage: /semantic on|off|status")
            continue

        if text == "/memory":
            adaptive = load_semantic_memory(memory_path)
            labels = sorted({row.label for row in adaptive})
            print(f"Adaptive samples: {len(adaptive)} -> {memory_path}")
            print(
                "Adaptive labels :",
                ", ".join(labels) if labels else "(none)",
            )
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
        base_ranked = base_router.route(text)
        base_top1 = base_ranked[0]

        print(
            f"SEM> {gate}  label={top1.label} "
            f"sim={top1.similarity:.6f} margin={margin:.6f}"
        )

        if semantic_v2_enabled:
            adaptive = load_semantic_memory(memory_path)
            taught_label = exact_memory_label(memory_path, text)
            candidate_rows = ranked[: min(3, len(ranked))]
            candidate_labels = [row.label for row in candidate_rows]
            candidate_scores = {
                row.label: float(row.similarity)
                for row in candidate_rows
            }

            runtime = {
                "memory_label": taught_label,
                "memory_similarity": 1.0 if taught_label is not None else None,
                "base_label": base_top1.label,
                "base_similarity": float(base_top1.similarity),
                "gate_state": gate,
                "selected_label": top1.label,
                "selected_similarity": float(top1.similarity),
                "decision_margin": float(margin),
                "candidate_labels": candidate_labels,
                "candidate_scores": candidate_scores,
                # Cosine similarity is used only as a runtime heuristic here,
                # not as a calibrated probability.
                "confidence": float(top1.similarity),
                "adaptive_enabled": learning_enabled,
                "adaptive_samples": len(adaptive),
                "memory_labels": len({row.label for row in adaptive}),
                "metadata": {
                    "runtime": "v0.3.1-chat-native",
                    "router": "adaptive-centroid",
                    "policy": args.policy,
                },
                "gate_reason": gate_reason(
                    gate,
                    float(top1.similarity),
                    float(margin),
                    thresholds,
                ),
            }

            last_semantic_v2 = from_runtime_dict(
                model,
                tokenizer,
                text,
                runtime,
                concept_texts=[top1.label],
                purpose_text=text,
                intent=None,
            )
            print_semantic_v2(runtime_summary(last_semantic_v2), runtime)

        if gate == "UNKNOWN_KNOWLEDGE":
            print("SEM> Unknown semantic region. Teach with: /teach <label>")
        elif gate == "GATE_REVIEW":
            print("SEM> Ambiguous semantic region. Review or teach a better label.")
        else:
            print(f"SEM> Routed to semantic class: {top1.label}")


if __name__ == "__main__":
    main()
