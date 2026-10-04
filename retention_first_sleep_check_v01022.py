# retention_first_sleep_check_v01022.py
#
# LLM_SEM v0.10.41
# Preflight check: decide whether balanced QA sleep is necessary.

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from model import LanguageModel
from semantic_eval import load_benchmark
from semantic_router import SemanticRouter
from semantic_guided_answer_finetune_v097 import generation_quality, load_dataset
from multi_knowledge_internalization_v01016 import select_probes
from runtime_answer_retention_v01015 import ratio, runtime_generate
from tokenizer import Tokenizer


def parse_args():
    p = argparse.ArgumentParser(
        description="LLM_SEM v0.10.41 Retention-First Sleep Precheck"
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
    p.add_argument("--result-json", default="")
    p.add_argument("--allow-cpu", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and not args.allow_cpu:
        raise RuntimeError("CUDA unavailable. Use --allow-cpu only for testing.")

    tokenizer = Tokenizer.load(args.tokenizer)
    rows = load_dataset(Path(args.dataset))
    probes = select_probes(rows)

    print("=" * 104)
    print(" LLM_SEM v0.10.41 Retention-First Sleep Precheck")
    print("=" * 104)
    print("Model             :", args.model)
    print("Dataset           :", args.dataset)
    print("Unique concepts   :", len(probes))
    print("Required concepts :", args.min_concepts)

    if len(probes) < args.min_concepts:
        result = {
            "status": "SKIP",
            "reason": "insufficient-concepts",
            "concepts": len(probes),
        }
        if args.result_json:
            Path(args.result_json).write_text(
                json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        print("RESULT            : SKIP")
        raise SystemExit(2)

    model, checkpoint = LanguageModel.load_checkpoint(args.model, device=device)
    benchmark = load_benchmark(args.benchmark)
    router = SemanticRouter(model, tokenizer, alpha=args.alpha)
    router.fit(benchmark)

    scores = []
    failures = 0
    details = []

    for index, (concept, row) in enumerate(probes, 1):
        query = str(row["query"])
        canonical = str(row["answer"])
        generated, label, margin = runtime_generate(
            model, tokenizer, router, query
        )
        sim = ratio(generated, canonical)
        quality = generation_quality(generated)
        canonical_complete = sim >= 0.999999
        termination_ok = bool(quality["terminated"]) or canonical_complete
        ok = (
            sim >= args.min_similarity
            and termination_ok
            and float(quality["abnormal_ratio"]) <= args.max_abnormal_ratio
            and float(quality["repetition_ratio"]) <= args.max_repetition_ratio
        )
        scores.append(sim)
        failures += int(not ok)
        details.append({
            "concept": concept,
            "query": query,
            "similarity": sim,
            "label": label,
            "margin": margin,
            "generated": generated,
            "passed": ok,
            "canonical_complete": canonical_complete,
            "termination_ok": termination_ok,
        })
        print(
            f"{index:02d}. [{'PASS' if ok else 'FAIL'}] "
            f"concept={concept!r} sim={sim:.6f} "
            f"route={label} margin={margin:+.6f}"
        )

    mean_sim = sum(scores) / len(scores) if scores else 0.0
    passed = failures == 0 and mean_sim >= args.min_concept_mean
    result = {
        "status": "PASS" if passed else "FAIL",
        "checkpoint_loss": checkpoint.get("loss"),
        "concepts": len(probes),
        "failures": failures,
        "mean_similarity": mean_sim,
        "details": details,
    }

    print()
    print("Retention-first summary")
    print("-----------------------")
    print("Concepts failed      :", failures)
    print("Mean canonical sim   :", f"{mean_sim:.6f}")
    print("Balanced QA required :", "NO" if passed else "YES")
    print("RESULT               :", result["status"])

    if args.result_json:
        Path(args.result_json).write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
