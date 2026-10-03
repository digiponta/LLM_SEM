# runtime_answer_retention_v01015.py
#
# LLM_SEM v0.10.15
# Validate answer retention through the actual /internal runtime path:
# SemanticRouter -> selected label -> semantic-guided prompt -> generation.

from __future__ import annotations

import argparse
from difflib import SequenceMatcher
from pathlib import Path

import torch

from chat import build_semantic_generation_prompt
from model import LanguageModel
from semantic_eval import load_benchmark
from semantic_router import SemanticRouter
from semantic_intent_v034 import extract_purpose_intent
from semantic_proposition_v036 import (
    extract_propositions,
    proposition_concepts,
)
from semantic_guided_answer_finetune_v097 import (
    generation_quality,
    load_dataset,
    stabilize_generated_answer,
)
from tokenizer import Tokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LLM_SEM v0.10.18 Improvement-Aware Runtime Answer Retention"
    )
    p.add_argument("--source", required=True)
    p.add_argument("--candidate", required=True)
    p.add_argument("--dataset", default="data/semantic_sleep_qa_v0106.json")
    p.add_argument("--benchmark", default="my_benchmark.csv")
    p.add_argument("--tokenizer", default="model/tokenizer.json")
    p.add_argument("--alpha", type=float, default=0.35)
    p.add_argument("--min-source-candidate-sim", type=float, default=0.70)
    p.add_argument("--max-canonical-drop", type=float, default=0.05)
    p.add_argument("--known-canonical-threshold", type=float, default=0.70)
    p.add_argument("--min-improvement-gain", type=float, default=0.10)
    p.add_argument("--min-improved-canonical", type=float, default=0.70)
    p.add_argument("--max-abnormal-ratio", type=float, default=0.02)
    p.add_argument("--max-repetition-ratio", type=float, default=0.20)
    p.add_argument("--allow-cpu", action="store_true")
    return p.parse_args()


def compact(text: str) -> str:
    return "".join(str(text).split())


def ratio(a: str, b: str) -> float:
    aa = compact(stabilize_generated_answer(a))
    bb = compact(stabilize_generated_answer(b))
    if not aa and not bb:
        return 1.0
    return SequenceMatcher(None, aa, bb).ratio()


@torch.no_grad()
def runtime_generate(
    model,
    tokenizer,
    router,
    query: str,
    *,
    max_new_tokens: int = 96,
) -> tuple[str, str, float]:
    ranked = router.route(query)
    if not ranked:
        return "", "(none)", -1.0

    top = ranked[0]
    second = ranked[1] if len(ranked) > 1 else None
    margin = float(
        top.similarity - (second.similarity if second is not None else -1.0)
    )

    extracted = extract_purpose_intent(query)
    props = extract_propositions(
        extracted.concept_texts[0]
        if extracted.concept_texts
        else query
    )
    concepts = proposition_concepts(
        extracted.concept_texts,
        props,
    )

    prompt = build_semantic_generation_prompt(
        query,
        selected_label=top.label,
        gate="INTERNAL_PROBE",
        intent=extracted.intent,
        concepts=concepts,
        truth_record=None,
    )
    prompt_ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
    generated = model.generate(
        prompt_ids,
        max_new_tokens=max_new_tokens,
        eos_id=tokenizer.eos_id,
        temperature=0.2,
        top_k=1,
        repetition_penalty=1.10,
    )
    continuation = generated[len(prompt_ids):]
    answer = tokenizer.decode(
        continuation,
        skip_special_tokens=True,
    ).strip()
    return stabilize_generated_answer(answer), str(top.label), margin


def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and not args.allow_cpu:
        raise RuntimeError("CUDA unavailable. Use --allow-cpu only for testing.")

    tokenizer = Tokenizer.load(args.tokenizer)
    benchmark = load_benchmark(args.benchmark)
    rows = [
        row for row in load_dataset(Path(args.dataset))
        if bool(row.get("must_train", False))
    ]
    if not rows:
        raise RuntimeError("No mandatory runtime probes found.")

    # One probe per unique query.
    by_query = {}
    for row in rows:
        by_query.setdefault(str(row["query"]), row)
    probes = list(by_query.values())

    source_model, _ = LanguageModel.load_checkpoint(args.source, device=device)
    candidate_model, _ = LanguageModel.load_checkpoint(
        args.candidate,
        device=device,
    )

    source_router = SemanticRouter(
        source_model,
        tokenizer,
        alpha=args.alpha,
    )
    candidate_router = SemanticRouter(
        candidate_model,
        tokenizer,
        alpha=args.alpha,
    )
    source_router.fit(benchmark)
    candidate_router.fit(benchmark)

    print("=" * 104)
    print(" LLM_SEM v0.10.18 Improvement-Aware Runtime /internal Retention")
    print("=" * 104)
    print("Source checkpoint   :", args.source)
    print("Candidate checkpoint:", args.candidate)
    print("Runtime probes      :", len(probes))
    print(
        "Min source/candidate similarity:",
        args.min_source_candidate_sim,
    )
    print("Max canonical drop  :", args.max_canonical_drop)
    print("Known canonical th  :", args.known_canonical_threshold)
    print("Min improvement gain:", args.min_improvement_gain)
    print("Min improved canon. :", args.min_improved_canonical)

    failures = 0
    source_canonical_vals = []
    candidate_canonical_vals = []
    pair_vals = []

    for index, row in enumerate(probes, 1):
        query = str(row["query"])
        canonical = str(row["answer"])

        source_answer, source_label, source_margin = runtime_generate(
            source_model,
            tokenizer,
            source_router,
            query,
        )
        candidate_answer, candidate_label, candidate_margin = runtime_generate(
            candidate_model,
            tokenizer,
            candidate_router,
            query,
        )

        pair_sim = ratio(source_answer, candidate_answer)
        source_canonical = ratio(source_answer, canonical)
        candidate_canonical = ratio(candidate_answer, canonical)
        canonical_drop = source_canonical - candidate_canonical
        canonical_gain = candidate_canonical - source_canonical
        quality = generation_quality(candidate_answer)

        pair_vals.append(pair_sim)
        source_canonical_vals.append(source_canonical)
        candidate_canonical_vals.append(candidate_canonical)

        quality_ok = (
            bool(quality["terminated"])
            and float(quality["abnormal_ratio"]) <= args.max_abnormal_ratio
            and float(quality["repetition_ratio"]) <= args.max_repetition_ratio
        )

        source_is_known = (
            source_canonical >= args.known_canonical_threshold
        )
        improvement_ok = (
            not source_is_known
            and canonical_gain >= args.min_improvement_gain
            and candidate_canonical >= args.min_improved_canonical
        )
        retention_ok = (
            source_is_known
            and pair_sim >= args.min_source_candidate_sim
            and canonical_drop <= args.max_canonical_drop
        )

        if improvement_ok:
            mode = "IMPROVEMENT"
            ok = quality_ok
        else:
            mode = "RETENTION"
            ok = retention_ok and quality_ok

        failures += int(not ok)

        print()
        print(
            f"{index:02d}. [{'PASS' if ok else 'FAIL'}] "
            f"mode={mode} query={query!r}"
        )
        print(
            f"    route      : {source_label} ({source_margin:+.6f})"
            f" -> {candidate_label} ({candidate_margin:+.6f})"
        )
        print(
            f"    source/cand: {pair_sim:.6f}  "
            f"canonical: {source_canonical:.6f}"
            f" -> {candidate_canonical:.6f} "
            f"drop={canonical_drop:+.6f} gain={canonical_gain:+.6f}"
        )
        print(
            "    quality    : "
            f"terminated={quality['terminated']} "
            f"abnormal={float(quality['abnormal_ratio']):.6f} "
            f"repetition={float(quality['repetition_ratio']):.6f}"
        )
        print("    source     :", source_answer)
        print("    candidate  :", candidate_answer)

    pair_mean = sum(pair_vals) / len(pair_vals)
    source_canonical_mean = (
        sum(source_canonical_vals) / len(source_canonical_vals)
    )
    candidate_canonical_mean = (
        sum(candidate_canonical_vals) / len(candidate_canonical_vals)
    )

    print()
    print("Runtime answer retention summary")
    print("--------------------------------")
    print("Source/candidate mean similarity:", f"{pair_mean:.6f}")
    print(
        "Canonical mean similarity       :",
        f"{source_canonical_mean:.6f} -> {candidate_canonical_mean:.6f}",
    )
    print("Per-probe failures               :", failures)
    print(
        "Policy                           :",
        "preserve known answers; allow canonical-improving new knowledge",
    )
    result = failures == 0
    print("RESULT                           :", "PASS" if result else "FAIL")

    raise SystemExit(0 if result else 1)


if __name__ == "__main__":
    main()
