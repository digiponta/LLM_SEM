# multi_knowledge_internalization_v01016.py
#
# LLM_SEM v0.10.16
# Candidate-only multi-knowledge internalization validation.
# External Semantic/Answer/Relation Memory is not consulted.

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from model import LanguageModel
from semantic_eval import load_benchmark
from semantic_router import SemanticRouter
from semantic_guided_answer_finetune_v097 import (
    generation_quality,
    load_dataset,
)
from runtime_answer_retention_v01015 import (
    ratio,
    runtime_generate,
)
from tokenizer import Tokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LLM_SEM v0.10.16 Multi-Knowledge Internalization"
    )
    p.add_argument("--model", required=True)
    p.add_argument("--dataset", default="data/semantic_sleep_qa_v0106.json")
    p.add_argument("--benchmark", default="my_benchmark.csv")
    p.add_argument("--tokenizer", default="model/tokenizer.json")
    p.add_argument("--alpha", type=float, default=0.35)
    p.add_argument("--min-concepts", type=int, default=2)
    p.add_argument("--min-similarity", type=float, default=0.70)
    p.add_argument("--min-concept-mean", type=float, default=0.75)
    p.add_argument("--max-abnormal-ratio", type=float, default=0.02)
    p.add_argument("--max-repetition-ratio", type=float, default=0.20)
    p.add_argument("--allow-cpu", action="store_true")
    return p.parse_args()


def concept_of(row: dict) -> str:
    concepts = [
        str(x).strip()
        for x in row.get("concepts", [])
        if str(x).strip()
    ]
    if concepts:
        return concepts[0]
    return str(row.get("query", "")).strip()


def probe_priority(query: str, concept: str) -> tuple[int, int]:
    q = "".join(query.split())
    c = "".join(concept.split())
    if q == c + "とは":
        return (0, len(q))
    if q == c:
        return (1, len(q))
    if q == c + "について教えて":
        return (2, len(q))
    if q == c + "を説明して":
        return (3, len(q))
    return (9, len(q))


def select_probes(rows: list[dict]) -> list[tuple[str, dict]]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        if not bool(row.get("must_train", False)):
            continue
        concept = concept_of(row)
        if not concept:
            continue
        grouped.setdefault(concept, []).append(row)

    selected = []
    for concept, candidates in grouped.items():
        best = sorted(
            candidates,
            key=lambda row: probe_priority(str(row["query"]), concept),
        )[0]
        selected.append((concept, best))
    return selected


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and not args.allow_cpu:
        raise RuntimeError("CUDA unavailable. Use --allow-cpu only for testing.")

    tokenizer = Tokenizer.load(args.tokenizer)
    rows = load_dataset(Path(args.dataset))
    probes = select_probes(rows)

    print("=" * 104)
    print(" LLM_SEM v0.10.16 Multi-Knowledge Internalization")
    print("=" * 104)
    print("Model              :", args.model)
    print("Dataset            :", args.dataset)
    print("Unique concepts    :", len(probes))
    print("Required concepts  :", args.min_concepts)
    print("Min similarity     :", args.min_similarity)
    print("Min concept mean   :", args.min_concept_mean)

    if len(probes) < args.min_concepts:
        print()
        print(
            "RESULT             : SKIP "
            f"(need >= {args.min_concepts} concepts, found {len(probes)})"
        )
        raise SystemExit(0)

    model, checkpoint = LanguageModel.load_checkpoint(args.model, device=device)
    benchmark = load_benchmark(args.benchmark)
    router = SemanticRouter(model, tokenizer, alpha=args.alpha)
    router.fit(benchmark)

    print("Checkpoint loss    :", checkpoint.get("loss"))

    scores = []
    failures = 0

    for index, (concept, row) in enumerate(probes, 1):
        query = str(row["query"])
        canonical = str(row["answer"])
        generated, label, margin = runtime_generate(
            model,
            tokenizer,
            router,
            query,
        )
        sim = ratio(generated, canonical)
        quality = generation_quality(generated)
        ok = (
            sim >= args.min_similarity
            and bool(quality["terminated"])
            and float(quality["abnormal_ratio"]) <= args.max_abnormal_ratio
            and float(quality["repetition_ratio"]) <= args.max_repetition_ratio
        )
        scores.append(sim)
        failures += int(not ok)

        print()
        print(
            f"{index:02d}. [{'PASS' if ok else 'FAIL'}] "
            f"concept={concept!r} query={query!r}"
        )
        print(
            f"    route      : label={label} margin={margin:+.6f}"
        )
        print(
            f"    similarity : {sim:.6f}"
        )
        print(
            "    quality    : "
            f"terminated={quality['terminated']} "
            f"abnormal={float(quality['abnormal_ratio']):.6f} "
            f"repetition={float(quality['repetition_ratio']):.6f}"
        )
        print("    expected   :", canonical)
        print("    internal   :", generated)

    mean_sim = sum(scores) / len(scores) if scores else 0.0
    result = failures == 0 and mean_sim >= args.min_concept_mean

    print()
    print("Multi-knowledge summary")
    print("-----------------------")
    print("Concepts tested       :", len(probes))
    print("Concepts failed       :", failures)
    print("Mean canonical sim    :", f"{mean_sim:.6f}")
    print("External memory used  : NO")
    print("RESULT                :", "PASS" if result else "FAIL")

    raise SystemExit(0 if result else 1)


if __name__ == "__main__":
    main()
