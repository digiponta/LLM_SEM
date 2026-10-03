# chat.py
#
# LLM_SEM v0.10.8 Internal Knowledge Probe
#
# Integrates adaptive learning, Semantic Data v2.0, structural
# relation/proposition extraction, and the v0.4.6 local-evidence
# multi-prototype runtime. The base Transformer remains frozen.

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import json
import subprocess
import sys

import torch

from adaptive_semantic_learning import (
    append_semantic_memory,
    exact_memory_label,
    exact_truth_record,
    forget_semantic_memory,
    load_semantic_memory,
    memory_status_counts,
    merge_samples,
    truth_notice,
    truth_status_counts,
    update_memory_truth,
)
from adaptive_semantic_runtime import (
    build_multi_prototypes,
    decide_adaptive_route,
    encode_memory,
)
from adaptive_composition_runtime_v065 import (
    AdaptiveCompositionRuntime,
    enrich_proposition_specs,
    ROLE_CHECKPOINT as COMPOSITION_ROLE_CHECKPOINT,
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
    proposition_specs,
    refine_purpose,
)
from tokenizer import Tokenizer
from answer_aware_gate_v099 import apply_answer_aware_gate
from relation_memory_v0101 import (
    DEFAULT_RELATION_MEMORY,
    append_relation_fact,
    compose_subject_facts,
    parse_relation_fact,
)
from semantic_answer_memory_v098 import (
    DEFAULT_ANSWER_MEMORY,
    DEFAULT_LEARNED_ANSWER_MEMORY,
    DEFAULT_UNIFIED_ANSWER_MEMORY,
    SemanticAnswerMemory,
    truth_allows_answer_memory,
)


DEFAULT_MODEL = "model/model-gpu-v0.4.pt"
DEFAULT_ACTIVE_MODEL_MANIFEST = "model/active-model.json"
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
            "LLM_SEM v0.7.2 Unified Semantic Runtime: adaptive memory, "
            "Semantic Data v2.0, structural relations/propositions, and "
            "local-evidence routing."
        )
    )
    p.add_argument("--model", default=None, help="Explicit checkpoint override. If omitted, use model/active-model.json when available.")
    p.add_argument("--active-model-manifest", default=DEFAULT_ACTIVE_MODEL_MANIFEST)
    p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    p.add_argument("--benchmark", default=DEFAULT_BENCHMARK)
    p.add_argument("--unknown-benchmark", default=DEFAULT_UNKNOWN_BENCHMARK)
    p.add_argument("--memory", default=DEFAULT_MEMORY)
    p.add_argument("--answer-memory", default=DEFAULT_ANSWER_MEMORY)
    p.add_argument("--unified-answer-memory", default=DEFAULT_UNIFIED_ANSWER_MEMORY)
    p.add_argument("--learned-answer-memory", default=DEFAULT_LEARNED_ANSWER_MEMORY)
    p.add_argument("--relation-memory", default=DEFAULT_RELATION_MEMORY)
    p.add_argument("--answer-memory-min-score", type=float, default=7.0)
    p.add_argument("--answer-gate-min-score", type=float, default=12.0)
    p.add_argument("--sleep-candidate", default="model/model-sem-sleep-v0106.pt")
    p.add_argument("--sleep-semantic-candidate", default="model/model-sem-sleep-sem-v0106.pt")
    p.add_argument("--sleep-qa-epochs", type=int, default=80)
    p.add_argument("--sleep-epochs", type=int, default=80)
    p.add_argument(
        "--policy",
        default=DEFAULT_POLICY,
        choices=["known-first", "balanced", "discovery-first"],
    )
    p.add_argument("--alpha", type=float, default=0.35)
    p.add_argument("--max-new-tokens", type=int, default=80)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument(
        "--answer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generate a natural-language continuation after semantic/truth analysis.",
    )
    p.add_argument("--prototypes-per-label", type=int, default=2)
    p.add_argument("--local-k", type=int, default=3)
    p.add_argument("--local-purity", type=float, default=0.60)
    p.add_argument("--memory-sim", type=float, default=0.80)
    p.add_argument("--override-sim", type=float, default=0.92)
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
    p.add_argument(
        "--composition",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable v0.6.5 adaptive proposition composition.",
    )
    p.add_argument(
        "--composition-checkpoint",
        default=COMPOSITION_ROLE_CHECKPOINT,
        help="Frozen structural-role checkpoint used by adaptive composition.",
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


def evaluate_local_runtime(
    base_router,
    adaptive_samples,
    text: str,
    args: argparse.Namespace,
):
    """Evaluate v0.4.6 local-evidence adaptive routing without retraining base."""
    if not adaptive_samples:
        return None

    memory = encode_memory(base_router, adaptive_samples)
    prototypes = build_multi_prototypes(
        memory,
        per_label=max(1, int(args.prototypes_per_label)),
    )
    return decide_adaptive_route(
        base_router,
        text,
        memory,
        prototypes,
        base_similarity_threshold=float(args.memory_sim),
        override_similarity_threshold=float(args.override_sim),
        local_k=max(1, int(args.local_k)),
        local_purity_threshold=float(args.local_purity),
    )


def print_local_runtime(decision) -> None:
    if decision is None:
        print("LOC> adaptive memory unavailable")
        return
    margin = (
        f"{decision.memory_margin:.6f}"
        if decision.memory_margin is not None
        else "n/a"
    )
    print(
        f"LOC> action={decision.action} label={decision.label} "
        f"sim={decision.memory_similarity:.6f} margin={margin}"
    )
    print(
        f"LOC> majority={decision.local_majority_label or '(none)'} "
        f"purity={decision.local_purity:.3f} k={decision.local_k} "
        f"base={decision.base_label} ({decision.base_similarity:.6f})"
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
            vector_dimension = rel.get("vector_dimension")
            vec_text = (
                f" vec={vector_dimension}"
                if isinstance(vector_dimension, int)
                else ""
            )
            print(
                "    "
                f"{rel.get('subject')} --{rel.get('predicate')}--> "
                f"{rel.get('object')}{conf_text}{vec_text}"
            )

    propositions = summary.get("propositions") or []
    if propositions:
        print("V2> propositions:")
        for prop in propositions:
            if not isinstance(prop, dict):
                continue
            attrs = prop.get("attributes") or {}
            print(
                "    "
                f"{prop.get('proposition_id')}: "
                f"{prop.get('subject')} --{prop.get('predicate')}--> "
                f"{prop.get('object')} "
                f"vec={prop.get('vector_dimension')} "
                f"role={prop.get('vector_role') or 'proposition'}"
            )
            if isinstance(attrs, dict) and attrs.get("composition_mode"):
                print(
                    "      "
                    f"composition={attrs.get('composition_mode')} "
                    f"weights={attrs.get('composition_weights')} "
                    f"novelty=(R:{attrs.get('relation_seen')}, "
                    f"S:{attrs.get('subject_seen')}, "
                    f"O:{attrs.get('object_seen')}) "
                    f"conf={attrs.get('composition_confidence')}"
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


def build_semantic_generation_prompt(
    text: str,
    *,
    selected_label: str,
    gate: str,
    intent: str | None,
    concepts: list[str] | None,
    truth_record: dict | None,
) -> str:
    """Build a compact generation prompt from semantic runtime state.

    The model is small and not instruction-tuned, so the prompt deliberately
    stays short and regular rather than adding a long system-style instruction.
    """
    concept_text = "、".join(concepts or []) or text
    truth = (
        str(truth_record.get("truth_status", "UNVERIFIED"))
        if truth_record is not None
        else "UNKNOWN"
    )
    intent_text = intent or "general"
    return (
        f"質問:{text}\n"
        f"分類:{selected_label}\n"
        f"目的:{intent_text}\n"
        f"概念:{concept_text}\n"
        f"真偽:{truth}\n"
        "回答:"
    )


def generate_answer(
    model: LanguageModel,
    tokenizer: Tokenizer,
    text: str,
    args: argparse.Namespace,
    *,
    selected_label: str | None = None,
    gate: str | None = None,
    intent: str | None = None,
    concepts: list[str] | None = None,
    truth_record: dict | None = None,
) -> tuple[str, str]:
    """Generate a user-visible answer using semantic guidance when available."""
    if selected_label:
        prompt = build_semantic_generation_prompt(
            text,
            selected_label=selected_label,
            gate=gate or "UNKNOWN",
            intent=intent,
            concepts=concepts,
            truth_record=truth_record,
        )
        mode = "semantic-guided"
    else:
        prompt = text
        mode = "raw"

    prompt_ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
    generated = model.generate(
        prompt_ids,
        max_new_tokens=max(1, int(args.max_new_tokens)),
        eos_id=tokenizer.eos_id,
        temperature=float(args.temperature),
        top_k=max(1, int(args.top_k)),
        repetition_penalty=1.15,
    )
    continuation = generated[len(prompt_ids):]
    answer = tokenizer.decode(continuation, skip_special_tokens=True).strip()
    if answer:
        return answer, mode

    full = tokenizer.decode(generated, skip_special_tokens=True)
    if full.startswith(prompt):
        full = full[len(prompt):]
    return full.strip() or "(generation produced no visible tokens)", mode


def resolve_runtime_model(args: argparse.Namespace) -> str:
    if args.model:
        return args.model

    manifest_path = Path(args.active_model_manifest)
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            active = str(manifest.get("active_model", "")).strip()
            if active and Path(active).exists():
                return active
            if active:
                print(
                    "WARN> active-model manifest points to missing checkpoint: "
                    f"{active}; falling back to {DEFAULT_MODEL}"
                )
        except (json.JSONDecodeError, OSError) as exc:
            print(
                "WARN> failed to read active-model manifest "
                f"{manifest_path}: {exc}; falling back to {DEFAULT_MODEL}"
            )
    return DEFAULT_MODEL


def main() -> None:
    args = parse_args()
    args.model = resolve_runtime_model(args)

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
    answer_memory_paths = []
    if Path(args.learned_answer_memory).exists():
        answer_memory_paths.append(args.learned_answer_memory)
    if Path(args.unified_answer_memory).exists():
        answer_memory_paths.append(args.unified_answer_memory)
    answer_memory_paths.append(args.answer_memory)
    answer_memory = SemanticAnswerMemory.load_many(answer_memory_paths)
    learning_enabled = bool(args.learn)
    semantic_v2_enabled = bool(args.semantic_v2)
    composition_enabled = bool(args.composition)
    composition_runtime = (
        AdaptiveCompositionRuntime(
            device,
            checkpoint_path=args.composition_checkpoint,
        )
        if composition_enabled
        else None
    )
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
    print(" LLM_SEM v0.10.8 Internal Knowledge Probe")
    print("============================================================")
    print("Device          :", device)
    if device.type == "cuda":
        print("GPU             :", torch.cuda.get_device_name(0))
    print("Model           :", args.model)
    print("Checkpoint loss :", checkpoint.get("loss"))
    print("Policy          :", args.policy)
    print("Base samples    :", len(base_samples))
    print("Adaptive samples:", len(adaptive))
    print("Learning        :", "ON" if learning_enabled else "OFF")
    print("Semantic Data v2:", "ON" if semantic_v2_enabled else "OFF")
    print("Composition      :", "ON" if composition_enabled else "OFF")
    if composition_runtime is not None:
        print("Composition ckpt :", composition_runtime.checkpoint_path)
        print("Composition seed :", composition_runtime.seed)
    print("Local evidence   : ON")
    print("Prototypes/label :", args.prototypes_per_label)
    print("Local k          :", args.local_k)
    print("Local purity     :", args.local_purity)
    print("Memory sim       :", args.memory_sim)
    print("Override sim     :", args.override_sim)
    print("Answer generation:", "ON" if args.answer else "OFF")
    print("Max new tokens   :", args.max_new_tokens)
    print("Answer memory    :", args.answer_memory)
    print("Unified memory   :", args.unified_answer_memory if Path(args.unified_answer_memory).exists() else "(not built)")
    print("Learned memory   :", args.learned_answer_memory)
    print("Relation memory  :", args.relation_memory)
    print("Answer entries   :", len(answer_memory.rows))
    print("Answer gate min  :", args.answer_gate_min_score)
    print()
    print("Commands:")
    print("  /learn on|off|status")
    print("  /semantic on|off|status")
    print("  /teach <label|answer> teach a semantic label or trusted answer")
    print("  /forget              remove semantic teaching for previous utterance")
    print("  /train               reload learned memories (compatibility command)")
    print("  /sleep               consolidate external memories into the internal LLM")
    print("  /internal <query>    probe only the internal model; bypass external memories")
    print("  /teach-answer <text> persist a trusted answer for the previous utterance")
    print("  /truth <STATE>       mark previous utterance TRUE/FALSE/UNVERIFIED/CONTESTED/OUTDATED")
    print("  /memory               show adaptive sample count")
    print("  /runtime              show v0.10.8 runtime policy")
    print("  /quit")
    print()

    while True:
        text = input("You> ").strip()
        if not text:
            continue

        if text in ("/quit", "/exit", "quit", "exit"):
            break

        if text.startswith("/internal"):
            parts = text.split(maxsplit=1)
            if len(parts) != 2 or not parts[1].strip():
                print("Usage: /internal <query>")
                continue

            probe_text = parts[1].strip()
            ranked_internal = base_router.route(probe_text)
            if not ranked_internal:
                print("INTERNAL> no semantic candidate.")
                continue

            top = ranked_internal[0]
            second = ranked_internal[1] if len(ranked_internal) > 1 else None
            margin_internal = float(
                top.similarity - (second.similarity if second is not None else -1.0)
            )

            extracted_internal = extract_purpose_intent(probe_text)
            props_internal = extract_propositions(
                extracted_internal.concept_texts[0]
                if extracted_internal.concept_texts
                else probe_text
            )
            concepts_internal = proposition_concepts(
                extracted_internal.concept_texts,
                props_internal,
            )

            answer_internal, generation_mode = generate_answer(
                model,
                tokenizer,
                probe_text,
                args,
                selected_label=top.label,
                gate="INTERNAL_PROBE",
                intent=extracted_internal.intent,
                concepts=concepts_internal,
                truth_record=None,
            )

            print(
                "INTERNAL> bypass=SemanticMemory,AnswerMemory,RelationMemory "
                f"model={args.model}"
            )
            print(
                f"INTERNAL> label={top.label} sim={top.similarity:.6f} "
                f"margin={margin_internal:.6f} mode={generation_mode}"
            )
            print("AI-INTERNAL>", answer_internal)
            continue

        if text == "/forget":
            if last_text is None:
                print("No previous utterance is available to forget.")
            else:
                removed = forget_semantic_memory(memory_path, last_text)
                samples, adaptive, thresholds = rebuild()
                print(
                    f"Forgot semantic teaching for {last_text!r}."
                    if removed else
                    "No matching Semantic Memory record."
                )
                if removed:
                    last_snapshot = None
            continue

        if text == "/sleep":
            cmd = [
                sys.executable,
                "semantic_sleep_v0105.py",
                "--memory", str(memory_path),
                "--model", str(args.model),
                "--tokenizer", str(args.tokenizer),
                "--benchmark", str(args.benchmark),
                "--candidate", str(args.sleep_candidate),
                "--semantic-candidate", str(args.sleep_semantic_candidate),
                "--manifest", str(args.active_model_manifest),
                "--base-answer-memory", str(args.answer_memory),
                "--learned-answer-memory", str(args.learned_answer_memory),
                "--relation-memory", str(args.relation_memory),
                "--epochs", str(args.sleep_epochs),
                "--qa-epochs", str(args.sleep_qa_epochs),
            ]
            print("SLEEP> Semantic Memory -> internal LLM consolidation")
            print("SLEEP> source    :", args.model)
            print("SLEEP> candidate :", args.sleep_candidate)
            result = subprocess.run(cmd, check=False)
            if result.returncode == 0:
                samples, adaptive, thresholds = rebuild()
                print(
                    "SLEEP> pipeline completed. "
                    "Restart chat.py to load a newly promoted model if promotion occurred."
                )
            else:
                print(
                    f"SLEEP> pipeline failed with exit code {result.returncode}. "
                    "Semantic Memory remains authoritative for non-consolidated records."
                )
            continue

        if text == "/train":
            answer_memory_paths = []
            if Path(args.learned_answer_memory).exists():
                answer_memory_paths.append(args.learned_answer_memory)
            if Path(args.unified_answer_memory).exists():
                answer_memory_paths.append(args.unified_answer_memory)
            answer_memory_paths.append(args.answer_memory)
            answer_memory = SemanticAnswerMemory.load_many(answer_memory_paths)
            samples, adaptive, thresholds = rebuild()
            print(
                f"TRAIN> memories reloaded: adaptive={len(adaptive)} "
                f"answer_entries={len(answer_memory.rows)}"
            )
            print("TRAIN> no gradient update; use consolidation/fine-tuning scripts for model training.")
            continue

        if text.startswith("/teach "):
            teach_value = text.split(maxsplit=1)[1].strip()
            if (
                len(teach_value) >= 16
                or any(mark in teach_value for mark in ("。", "、", "です", "である", "は", "を", "する"))
            ):
                text = "/teach-answer " + teach_value
                print("TCH> interpreted /teach argument as a trusted answer.")

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
            counts = memory_status_counts(memory_path)
            print(f"Adaptive samples: {len(adaptive)} -> {memory_path}")
            print(
                "Adaptive labels :",
                ", ".join(labels) if labels else "(none)",
            )
            print(
                "Memory states   : "
                + ", ".join(
                    f"{state}={counts.get(state, 0)}"
                    for state in (
                        "ACTIVE",
                        "TRAINING",
                        "VALIDATING",
                        "FAILED",
                        "CONSOLIDATED",
                    )
                )
            )
            print(
                "Memory priority : ACTIVE/TRAINING/VALIDATING/FAILED "
                "remain authoritative until validation PASS"
            )
            truth_counts = truth_status_counts(memory_path)
            print(
                "Truth states    : "
                + ", ".join(
                    f"{state}={truth_counts.get(state, 0)}"
                    for state in ("TRUE", "FALSE", "UNVERIFIED", "CONTESTED", "OUTDATED")
                )
            )
            continue

        if text == "/runtime":
            print("Runtime        : LLM_SEM v0.10.8 Internal Knowledge Probe")
            print("Base router    : FIXED benchmark router")
            print("Adaptive memory: multi-prototype + local evidence")
            print(
                "Consolidation  : memory-primary until validation PASS; "
                "CONSOLIDATED falls back to internal/base model"
            )
            print("Semantic data  : v2.0")
            print("Structure      : relation + adaptive proposition vectors")
            print(
                "Composition    : "
                + ("adaptive confirmed gate" if composition_enabled else "OFF")
            )
            print(
                "Policy         : "
                f"mem>={args.memory_sim:.2f}, override>={args.override_sim:.2f}, "
                f"k={args.local_k}, purity>={args.local_purity:.2f}"
            )
            continue

        if text.startswith("/teach-answer"):
            parts = text.split(maxsplit=1)
            if len(parts) != 2 or not parts[1].strip():
                print("Usage: /teach-answer <trusted answer>")
                continue
            if last_text is None or last_snapshot is None:
                print("No previous utterance is available to teach.")
                continue
            if not learning_enabled:
                print("Learning is OFF. Use /learn on first.")
                continue

            trusted_answer = parts[1].strip()
            extracted_answer = extract_purpose_intent(last_text)
            answer_props = extract_propositions(
                extracted_answer.concept_texts[0]
                if extracted_answer.concept_texts
                else last_text
            )
            answer_concepts = proposition_concepts(
                extracted_answer.concept_texts,
                answer_props,
            )
            truth_row = exact_truth_record(memory_path, last_text)
            truth_state = (
                str(truth_row.get("truth_status", "UNVERIFIED"))
                if truth_row is not None
                else "UNVERIFIED"
            )
            added = answer_memory.append_persistent(
                args.learned_answer_memory,
                query=last_text,
                answer=trusted_answer,
                label=last_snapshot.selected_label,
                intent=extracted_answer.intent,
                concepts=answer_concepts,
                truth_status=truth_state,
            )
            parsed_fact = parse_relation_fact(trusted_answer)
            relation_added = append_relation_fact(
                args.relation_memory,
                trusted_answer,
            )

            merged_answer = ""
            if parsed_fact is not None:
                merged_answer = compose_subject_facts(
                    args.relation_memory,
                    str(parsed_fact.get("subject", "")),
                )

            if merged_answer:
                answer_memory.upsert_persistent(
                    args.learned_answer_memory,
                    query=last_text,
                    answer=merged_answer,
                    label=last_snapshot.selected_label,
                    intent=extracted_answer.intent,
                    concepts=answer_concepts,
                    truth_status=truth_state,
                    source="chat-fact-merge",
                )
                print(
                    "FACT> merged subject knowledge: "
                    f"{merged_answer}"
                )
                print(
                    "Learned canonical answer: "
                    f"query={last_text!r} -> {args.learned_answer_memory}"
                )
            elif added:
                print(
                    "Learned answer: "
                    f"query={last_text!r} label={last_snapshot.selected_label!r} "
                    f"-> {args.learned_answer_memory}"
                )
            else:
                print("Answer already learned or invalid.")

            if relation_added:
                print(
                    "Learned relation fact -> "
                    f"{args.relation_memory}"
                )
            continue

        if text.startswith("/truth"):
            parts = text.split(maxsplit=1)
            if len(parts) != 2:
                print("Usage: /truth TRUE|FALSE|UNVERIFIED|CONTESTED|OUTDATED")
                continue
            if last_text is None:
                print("No previous utterance is available.")
                continue
            state = parts[1].strip().upper()
            try:
                changed = update_memory_truth(
                    memory_path,
                    last_text,
                    state,
                )
            except ValueError as exc:
                print(exc)
                continue
            if changed:
                print(f"Truth state updated: {state} for {last_text!r}")
            else:
                print("No matching Semantic Memory record. Teach it first.")
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
                    prop_specs = proposition_specs(propositions)
                    if composition_runtime is not None:
                        prop_specs = enrich_proposition_specs(
                            composition_runtime,
                            model,
                            tokenizer,
                            prop_specs,
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
                        proposition_specs=prop_specs,
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

        adaptive = load_semantic_memory(memory_path)
        local_decision = evaluate_local_runtime(
            base_router,
            adaptive,
            text,
            args,
        )
        if (
            gate != "ACCEPT_MEMORY"
            and local_decision is not None
            and local_decision.action == "ADAPTIVE_OVERRIDE"
        ):
            gate = "ACCEPT_ADAPTIVE"
            last_snapshot.selected_label = local_decision.label
            last_snapshot.similarity = local_decision.memory_similarity

        print(
            f"SEM> {gate}  label={last_snapshot.selected_label} "
            f"sim={last_snapshot.similarity:.6f} margin={margin:.6f}"
        )
        print_local_runtime(local_decision)

        if semantic_v2_enabled:
            pre_extracted = extract_purpose_intent(text)
            pre_generation_intent = pre_extracted.intent
            pre_generation_concepts = proposition_concepts(
                pre_extracted.concept_texts,
                extract_propositions(
                    pre_extracted.concept_texts[0]
                    if pre_extracted.concept_texts
                    else text
                ),
            )
        else:
            pre_extracted = extract_purpose_intent(text)
            pre_generation_intent = pre_extracted.intent
            pre_generation_concepts = pre_extracted.concept_texts

        pre_resolution = answer_memory.resolve(
            text,
            label=last_snapshot.selected_label,
            intent=pre_generation_intent,
            concepts=pre_generation_concepts,
            min_score=args.answer_memory_min_score,
        )
        truth_record = exact_truth_record(memory_path, text)
        notice = truth_notice(truth_record)

        answer_gate_decision = apply_answer_aware_gate(
            gate,
            resolution=pre_resolution,
            truth_record=truth_record,
            min_promote_score=args.answer_gate_min_score,
        )
        if answer_gate_decision.promoted:
            gate = answer_gate_decision.gate
            last_snapshot.gate = gate
            print(
                "GATE> promoted to ACCEPT_ANSWER_MEMORY "
                f"({answer_gate_decision.reason})"
            )
        if truth_record is not None:
            print(
                "TRUTH> "
                f"state={truth_record.get('truth_status', 'UNVERIFIED')} "
                f"confidence={float(truth_record.get('truth_confidence', 0.0)):.3f}"
            )
            if notice:
                print("TRUTH> WARNING:", notice)

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
            prop_specs = proposition_specs(propositions)
            if composition_runtime is not None:
                prop_specs = enrich_proposition_specs(
                    composition_runtime,
                    model,
                    tokenizer,
                    prop_specs,
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
                    "runtime": "v0.7.2-unified-consolidation",
                    "router": "base-fixed + adaptive-local-evidence",
                    "policy": args.policy,
                    "intent_rule": extracted.rule,
                },
                "extraction_rule": extracted.rule,
                "extraction_confidence": extracted.confidence,
                "gate_reason": (
                    "local evidence accepted adaptive override"
                    if gate == "ACCEPT_ADAPTIVE"
                    else (
                        answer_gate_decision.reason
                        if gate == "ACCEPT_ANSWER_MEMORY"
                        else gate_reason(
                            gate,
                            last_snapshot.similarity,
                            float(margin),
                            thresholds,
                        )
                    )
                ),
            }
            if local_decision is not None:
                runtime["metadata"].update(
                    {
                        "local_action": local_decision.action,
                        "local_majority": local_decision.local_majority_label,
                        "local_purity": local_decision.local_purity,
                        "local_k": local_decision.local_k,
                        "memory_similarity": local_decision.memory_similarity,
                        "memory_margin": local_decision.memory_margin,
                    }
                )

            last_semantic_v2 = from_runtime_dict(
                model,
                tokenizer,
                text,
                runtime,
                concept_texts=concept_texts,
                purpose_text=purpose_text,
                intent=extracted.intent,
                extra_relations=semantic_relations,
                proposition_specs=prop_specs,
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
        elif gate == "ACCEPT_ADAPTIVE":
            print(
                "SEM> Accepted by local-evidence adaptive override: "
                f"{last_snapshot.selected_label}"
            )
        elif gate == "ACCEPT_ANSWER_MEMORY":
            print(
                "SEM> Accepted by high-confidence Semantic Answer Memory: "
                f"{last_snapshot.selected_label}"
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

        if args.answer:
            if semantic_v2_enabled:
                generation_intent = extracted.intent
                generation_concepts = concept_texts
            else:
                fallback_extracted = extract_purpose_intent(text)
                generation_intent = fallback_extracted.intent
                generation_concepts = fallback_extracted.concept_texts

            resolution = pre_resolution

            if resolution.matched and truth_allows_answer_memory(
                truth_record,
                resolution.candidate,
            ):
                answer = resolution.answer or ""
                generation_mode = "semantic-answer-memory"
                print(
                    "ANS> "
                    f"mode={generation_mode} "
                    f"score={resolution.score:.2f} "
                    f"reason={resolution.reason}"
                )
            else:
                answer, generation_mode = generate_answer(
                    model,
                    tokenizer,
                    text,
                    args,
                    selected_label=last_snapshot.selected_label,
                    gate=gate,
                    intent=generation_intent,
                    concepts=generation_concepts,
                    truth_record=truth_record,
                )
                print(
                    "GEN> "
                    f"mode={generation_mode} "
                    f"label={last_snapshot.selected_label} "
                    f"intent={generation_intent or 'general'} "
                    f"truth={truth_record.get('truth_status', 'UNKNOWN') if truth_record else 'UNKNOWN'}"
                )
                if resolution.candidate is not None:
                    print(
                        "ANS> fallback "
                        f"score={resolution.score:.2f} "
                        f"reason={resolution.reason}"
                    )
            print("AI>", answer)


if __name__ == "__main__":
    main()


