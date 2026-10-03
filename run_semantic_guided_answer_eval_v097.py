# run_semantic_guided_answer_eval_v097.py
#
# LLM_SEM v0.9.7
# Evaluate answer fine-tuning and semantic retention.

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from chat import generate_answer
from model import LanguageModel
from semantic_eval import load_benchmark
from semantic_router import SemanticRouter
from tokenizer import Tokenizer


def parse_args():
    p = argparse.ArgumentParser(
        description="LLM_SEM v0.9.7 semantic-guided answer evaluation"
    )
    p.add_argument("--source", default="model/model-sem-consolidation-v081.pt")
    p.add_argument("--candidate", default="model/model-sem-guided-answer-v097.pt")
    p.add_argument("--tokenizer", default="model/tokenizer.json")
    p.add_argument("--benchmark", default="my_benchmark.csv")
    p.add_argument("--dataset", default="data/semantic_guided_qa_v097.json")
    p.add_argument("--max-new-tokens", type=int, default=50)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--alpha", type=float, default=0.35)
    p.add_argument("--allow-cpu", action="store_true")
    return p.parse_args()


def route_accuracy(model, tokenizer, benchmark, alpha):
    router = SemanticRouter(model, tokenizer, alpha=alpha)
    router.fit(benchmark)
    correct = 0
    for row in benchmark:
        ranked = router.route(row.text)
        correct += int(bool(ranked) and ranked[0].label == row.label)
    return correct / len(benchmark)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and not args.allow_cpu:
        raise RuntimeError("CUDA unavailable. Use --allow-cpu only for testing.")

    tokenizer = Tokenizer.load(args.tokenizer)
    source, source_ckpt = LanguageModel.load_checkpoint(args.source, device=device)
    candidate, candidate_ckpt = LanguageModel.load_checkpoint(args.candidate, device=device)
    source.eval()
    candidate.eval()

    benchmark = load_benchmark(args.benchmark)
    dataset = json.loads(Path(args.dataset).read_text(encoding="utf-8"))["samples"]

    source_acc = route_accuracy(source, tokenizer, benchmark, args.alpha)
    candidate_acc = route_accuracy(candidate, tokenizer, benchmark, args.alpha)

    tests = []
    wanted = ["CPUとは", "GPUとは", "宇宙とは", "原子とは", "暗号とは", "電車とは"]
    by_query = {row["query"]: row for row in dataset}
    for q in wanted:
        if q in by_query:
            tests.append(by_query[q])

    print("=" * 96)
    print(" LLM_SEM v0.9.7 Semantic-Guided Answer Evaluation")
    print("=" * 96)
    print("Device              :", device)
    if device.type == "cuda":
        print("GPU                 :", torch.cuda.get_device_name(0))
    print("Source              :", args.source)
    print("Source loss         :", source_ckpt.get("loss"))
    print("Candidate           :", args.candidate)
    print("Candidate loss      :", candidate_ckpt.get("loss"))
    print("Source route acc    :", f"{source_acc*100:.2f}%")
    print("Candidate route acc :", f"{candidate_acc*100:.2f}%")
    print("Route delta         :", f"{(candidate_acc-source_acc)*100:+.2f} pp")
    print()

    args.answer = True
    for row in tests:
        truth_record = {"truth_status": row.get("truth_status", "UNVERIFIED")}
        kwargs = dict(
            selected_label=row["label"],
            gate="ACCEPT",
            intent=row.get("intent"),
            concepts=row.get("concepts") or [],
            truth_record=truth_record,
        )
        src_answer, _ = generate_answer(
            source, tokenizer, row["query"], args, **kwargs
        )
        cand_answer, _ = generate_answer(
            candidate, tokenizer, row["query"], args, **kwargs
        )
        print(f"[{row['query']}]")
        print("  target   :", row["answer"])
        print("  source   :", src_answer)
        print("  candidate:", cand_answer)
        print()

    route_ok = candidate_acc >= source_acc - 0.10
    print("Semantic retention :", "PASS" if route_ok else "FAIL")
    print(
        "RESULT             :",
        "PASS" if route_ok else "REVIEW",
    )
    print()
    print(
        "Use this script for side-by-side qualitative answer comparison. "
        "Promotion should also require consolidated_retention_v094.py PASS."
    )


if __name__ == "__main__":
    main()
