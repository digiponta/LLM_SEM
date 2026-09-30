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
from dataclasses import dataclass
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
from semantic_intent_v034 import extract_purpose_intent
from semantic_relations_v035 import generate_semantic_relations
from semantic_proposition_v036 import (
    extract_propositions,
    proposition_concepts,
    refine_purpose,
)
from tokenizer import Tokenizer


DEFAULT_MODEL = "model/model-gpu-v0.4.pt"
DEFAULT_TOKENIZER = "model/tokenizer.json"
DEFAULT_BENCHMARK = "my_benchmark.csv"
DEFAULT_UNKNOWN_BENCHMARK = "unknown_benchmark.csv"
DEFAULT_MEMORY = "data/semantic_memory.jsonl"
DEFAULT_POLICY = "balanced"


@dataclass
class TeachingSnapshot:
    text: str
    gate: str
    selected_label: str
    similarity: float
    margin: float
    candidates: list[tuple[str, float]]
    memory_label: str | None
    base_label: str
    base_similarity: float
    disagreement: bool


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

    if gate == "ACCEPT_MEMORY":
        return "exact semantic-memory match overrides centroid thresholds"
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

    relations = summary.get("relations") or []
    if relations:
        print("V2> relations:")
        for rel in relations:
            if not isinstance(rel, dict):
                continue
            confidence = rel.get("confidence")
            conf_text = (
                f" [{float(confidence):.3f}]"
                if isinstance(confidence, (int, float))
                else ""
            )
            print(
                "    "
                f"{rel.get('subject')} --{rel.get('predicate')}--> "
                f"{rel.get('object')}{conf_text}"
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
    print(
        f"V2> memory_exact={bool(runtime.get('memory_exact'))} "
        f"decision_source={runtime.get('decision_source') or 'router'} "
        f"review_recommended={bool(runtime.get('review_recommended'))}"
    )
    extraction_rule = runtime.get("extraction_rule")
    extraction_confidence = runtime.get("extraction_confidence")
    if extraction_rule is not None:
        conf = (
            f"{float(extraction_confidence):.3f}"
            if isinstance(extraction_confidence, (int, float))
            else "n/a"
        )
        print(
            f"V2> extraction rule={extraction_rule} "
            f"confidence={conf}"
        )

    reason = runtime.get("gate_reason")
    if reason:
        print(f"V2> gate_reason={reason}")


def make_snapshot(
    text: str,
    ranked,
    thresholds,
    base_ranked,
    memory_path: Path,
) -> TeachingSnapshot:
    base_top1 = base_ranked[0]
    memory_label = exact_memory_label(memory_path, text)

    candidates = [
        (row.label, float(row.similarity))
        for row in ranked[: min(3, len(ranked))]
    ]

    normal_gate, normal_top1, margin = classify_gate(ranked, thresholds)

    if memory_label is not None:
        memory_score = next(
            (
                float(row.similarity)
                for row in ranked
                if row.label == memory_label
            ),
            1.0,
        )
        return TeachingSnapshot(
            text=text,
            gate="ACCEPT_MEMORY",
            selected_label=memory_label,
            similarity=memory_score,
            margin=float(margin),
            candidates=candidates,
            memory_label=memory_label,
            base_label=base_top1.label,
            base_similarity=float(base_top1.similarity),
            disagreement=(memory_label != base_top1.label),
        )

    return TeachingSnapshot(
        text=text,
        gate=normal_gate,
        selected_label=normal_top1.label,
        similarity=float(normal_top1.similarity),
        margin=float(margin),
        candidates=candidates,
        memory_label=None,
        base_label=base_top1.label,
        base_similarity=float(base_top1.similarity),
        disagreement=False,
    )


def print_teaching_effect(
    before: TeachingSnapshot,
    after: TeachingSnapshot,
    taught_label: str,
) -> None:
    print("TCH> --------------------------------------------------------")
    print("TCH> Teaching Effect Evaluation")
    print(f"TCH> text={before.text!r} taught_label={taught_label}")
    print(
        f"TCH> before gate={before.gate} selected={before.selected_label} "
        f"sim={before.similarity:.6f} margin={before.margin:.6f}"
    )
    print(
        f"TCH> after  gate={after.gate} selected={after.selected_label} "
        f"sim={after.similarity:.6f} margin={after.margin:.6f}"
    )
    print(
        f"TCH> delta_similarity={after.similarity - before.similarity:+.6f} "
        f"delta_margin={after.margin - before.margin:+.6f}"
    )
    print(
        f"TCH> gate_transition={before.gate}->{after.gate} "
        f"label_transition={before.selected_label}->{after.selected_label}"
    )
    print(
        f"TCH> memory_before={before.memory_label or '(none)'} "
        f"memory_after={after.memory_label or '(none)'}"
    )
    print(
        f"TCH> base={after.base_label} ({after.base_similarity:.6f}) "
        f"disagreement_after={after.disagreement}"
    )
    print("TCH> candidates_before:")
    for index, (label, score) in enumerate(before.candidates, 1):
        print(f"     {index}. {label:<12} {score:.6f}")
    print("TCH> candidates_after:")
    for index, (label, score) in enumerate(after.candidates, 1):
        print(f"     {index}. {label:<12} {score:.6f}")
    print("TCH> --------------------------------------------------------")


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
    last_snapshot: TeachingSnapshot | None = None

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
    print(" LLM_SEM v0.3.7 Proposition-Aware Concept Separation")
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
            before_snapshot = last_snapshot
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

                after_ranked = router.route(last_text)
                after_base_ranked = base_router.route(last_text)
                after_snapshot = make_snapshot(
                    last_text,
                    after_ranked,
                    thresholds,
                    after_base_ranked,
                    memory_path,
                )

                if before_snapshot is not None:
                    print_teaching_effect(
                        before_snapshot,
                        after_snapshot,
                        label,
                    )
                last_snapshot = after_snapshot

                if semantic_v2_enabled:
                    extracted = extract_purpose_intent(last_text)
                    propositions = extract_propositions(
                        extracted.concept_texts[0]
                        if extracted.concept_texts
                        else last_text
                    )
                    purpose_text = refine_purpose(
                        extracted.intent,
                        extracted.purpose_text,
                        propositions,
                    )
                    concept_texts = proposition_concepts(
                        extracted.concept_texts,
                        propositions,
                    )
                    semantic_relations = generate_semantic_relations(
                        extracted,
                        propositions,
                        purpose_text=purpose_text,
                        concept_texts=concept_texts,
                    )
                    after_runtime = {
                        "memory_label": after_snapshot.memory_label,
                        "memory_similarity": (
                            1.0
                            if after_snapshot.memory_label is not None
                            else None
                        ),
                        "memory_exact": after_snapshot.memory_label is not None,
                        "review_recommended": after_snapshot.disagreement,
                        "decision_source": (
                            "explicit_memory"
                            if after_snapshot.memory_label is not None
                            else "router"
                        ),
                        "base_label": after_snapshot.base_label,
                        "base_similarity": after_snapshot.base_similarity,
                        "gate_state": after_snapshot.gate,
                        "selected_label": after_snapshot.selected_label,
                        "selected_similarity": after_snapshot.similarity,
                        "decision_margin": after_snapshot.margin,
                        "candidate_labels": [
                            item[0] for item in after_snapshot.candidates
                        ],
                        "candidate_scores": {
                            item[0]: item[1]
                            for item in after_snapshot.candidates
                        },
                        "confidence": after_snapshot.similarity,
                        "adaptive_enabled": learning_enabled,
                        "adaptive_samples": len(adaptive),
                        "memory_labels": len({row.label for row in adaptive}),
                        "metadata": {
                            "runtime": "v0.3.7-teaching-effect",
                            "router": "adaptive-centroid",
                            "policy": args.policy,
                            "event": "after-teach",
                            "intent_rule": extracted.rule,
                        },
                        "extraction_rule": extracted.rule,
                        "extraction_confidence": extracted.confidence,
                        "gate_reason": gate_reason(
                            after_snapshot.gate,
                            after_snapshot.similarity,
                            after_snapshot.margin,
                            thresholds,
                        ),
                    }
                    last_semantic_v2 = from_runtime_dict(
                        model,
                        tokenizer,
                        last_text,
                        after_runtime,
                        concept_texts=concept_texts,
                        purpose_text=purpose_text,
                        intent=extracted.intent,
                        extra_relations=semantic_relations,
                    )
                    print("TCH> SemanticDataV2 after teaching:")
                    print_semantic_v2(
                        runtime_summary(last_semantic_v2),
                        after_runtime,
                    )
            else:
                print("Already learned.")
            continue

        last_text = text
        ranked = router.route(text)
        base_ranked = base_router.route(text)
        last_snapshot = make_snapshot(
            text,
            ranked,
            thresholds,
            base_ranked,
            memory_path,
        )
        gate = last_snapshot.gate
        margin = last_snapshot.margin
        base_top1 = base_ranked[0]

        print(
            f"SEM> {gate}  label={last_snapshot.selected_label} "
            f"sim={last_snapshot.similarity:.6f} margin={margin:.6f}"
        )

        if semantic_v2_enabled:
            extracted = extract_purpose_intent(text)
            propositions = extract_propositions(
                extracted.concept_texts[0]
                if extracted.concept_texts
                else text
            )
            purpose_text = refine_purpose(
                extracted.intent,
                extracted.purpose_text,
                propositions,
            )
            concept_texts = proposition_concepts(
                extracted.concept_texts,
                propositions,
            )
            semantic_relations = generate_semantic_relations(
                extracted,
                propositions,
                purpose_text=purpose_text,
                concept_texts=concept_texts,
            )
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
                "memory_exact": taught_label is not None,
                "review_recommended": bool(
                    taught_label is not None and taught_label != base_top1.label
                ),
                "decision_source": (
                    "explicit_memory" if taught_label is not None else "router"
                ),
                "base_label": base_top1.label,
                "base_similarity": float(base_top1.similarity),
                "gate_state": gate,
                "selected_label": last_snapshot.selected_label,
                "selected_similarity": last_snapshot.similarity,
                "decision_margin": float(margin),
                "candidate_labels": candidate_labels,
                "candidate_scores": candidate_scores,
                # Cosine similarity is used only as a runtime heuristic here,
                # not as a calibrated probability.
                "confidence": last_snapshot.similarity,
                "adaptive_enabled": learning_enabled,
                "adaptive_samples": len(adaptive),
                "memory_labels": len({row.label for row in adaptive}),
                "metadata": {
                    "runtime": "v0.3.7-chat-native",
                    "router": "adaptive-centroid",
                    "policy": args.policy,
                    "intent_rule": extracted.rule,
                },
                "extraction_rule": extracted.rule,
                "extraction_confidence": extracted.confidence,
                "gate_reason": gate_reason(
                    gate,
                    last_snapshot.similarity,
                    float(margin),
                    thresholds,
                ),
            }

            last_semantic_v2 = from_runtime_dict(
                model,
                tokenizer,
                text,
                runtime,
                concept_texts=concept_texts,
                purpose_text=purpose_text,
                intent=extracted.intent,
                extra_relations=semantic_relations,
            )
            print_semantic_v2(runtime_summary(last_semantic_v2), runtime)

        if gate == "ACCEPT_MEMORY":
            print(
                "SEM> Accepted by exact semantic memory: "
                f"{last_snapshot.selected_label}"
            )
            if last_snapshot.disagreement:
                print(
                    "SEM> Base/memory disagreement detected; "
                    "review is recommended."
                )
        elif gate == "UNKNOWN_KNOWLEDGE":
            print("SEM> Unknown semantic region. Teach with: /teach <label>")
        elif gate == "GATE_REVIEW":
            print("SEM> Ambiguous semantic region. Review or teach a better label.")
        else:
            print(
                "SEM> Routed to semantic class: "
                f"{last_snapshot.selected_label}"
            )


if __name__ == "__main__":
    main()
