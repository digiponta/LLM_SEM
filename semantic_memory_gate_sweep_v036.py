# semantic_memory_gate_sweep_v036.py
#
# LLM_SEM v0.3.6
# Sweep evaluation for:
#   memory similarity x memory margin x Base agreement
#
# The model and semantic memory are encoded once.  Gate thresholds are then
# swept over the cached per-query scores so each configuration is compared on
# exactly the same semantic evidence.

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List

import torch

from adaptive_semantic_learning import load_semantic_memory, merge_samples
from model import LanguageModel
from semantic_eval import LabeledSentence, load_benchmark
from semantic_memory_prototype import (
    build_prototypes,
    prototype_margin,
    rank_prototypes,
)
from semantic_router import SemanticRouter
from tokenizer import Tokenizer


DEFAULT_MODEL = "model/model-gpu-v0.4.pt"
DEFAULT_TOKENIZER = "model/tokenizer.json"
DEFAULT_BENCHMARK = "my_benchmark.csv"
DEFAULT_MEMORY = "data/semantic_memory.jsonl"

SIM_THRESHOLDS = (0.85, 0.88, 0.90, 0.92, 0.94, 0.95, 0.96, 0.97)
MARGIN_THRESHOLDS = (0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08)

# These are deliberately NOT the exact teaching strings used in the current
# v0.3.5 experiment.
DEFAULT_CASES = (
    ("computer", True, "量子コンピュータの仕組みを教えて"),
    ("computer", True, "量子計算とは何ですか"),
    ("computer", True, "量子計算の仕組みを説明してください"),
    ("computer", True, "スーパーコンピュータとは何ですか"),
    ("computer", True, "GPUによる並列計算を説明して"),
    ("science", True, "量子力学を説明してください"),
    ("science", True, "量子暗号について説明して"),
    ("science", True, "量子力学の基本を教えて"),
    ("science", True, "物理学における量子とは何ですか"),
    ("science", True, "量子状態について説明して"),
    ("REJECT", False, "今日の天気を教えて"),
    ("REJECT", False, "寿司について教えて"),
    ("REJECT", False, "猫について説明して"),
    ("REJECT", False, "東京駅への行き方を教えて"),
)


@dataclass(frozen=True)
class EvalCase:
    expected_label: str
    should_accept: bool
    text: str


@dataclass(frozen=True)
class CachedScore:
    expected_label: str
    should_accept: bool
    text: str
    memory_label: str
    memory_similarity: float
    memory_margin: float
    base_label: str
    base_similarity: float
    agreement: bool


def parse_args():
    p = argparse.ArgumentParser(
        description="LLM_SEM v0.3.6 Semantic Memory Gate sweep."
    )
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    p.add_argument("--benchmark", default=DEFAULT_BENCHMARK)
    p.add_argument("--memory", default=DEFAULT_MEMORY)
    p.add_argument(
        "--cases",
        default=None,
        help="Optional CSV with expected_label,should_accept,text columns.",
    )
    p.add_argument(
        "--summary-csv",
        default="semantic_memory_gate_sweep_v036.csv",
    )
    p.add_argument(
        "--detail-csv",
        default="semantic_memory_gate_detail_v036.csv",
    )
    p.add_argument("--alpha", type=float, default=0.35)
    return p.parse_args()


def load_cases(filename: str | None) -> List[EvalCase]:
    if filename is None:
        return [
            EvalCase(label, accept, text)
            for label, accept, text in DEFAULT_CASES
        ]

    rows: List[EvalCase] = []
    with Path(filename).open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"expected_label", "should_accept", "text"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(
                "Case CSV must contain expected_label,should_accept,text"
            )
        for row in reader:
            raw = str(row["should_accept"]).strip().lower()
            should_accept = raw in ("1", "true", "yes", "y")
            rows.append(
                EvalCase(
                    expected_label=str(row["expected_label"]).strip(),
                    should_accept=should_accept,
                    text=str(row["text"]).strip(),
                )
            )
    return rows


def cache_scores(router, prototypes, cases):
    cached: List[CachedScore] = []

    for case in cases:
        proto_scores = rank_prototypes(router, case.text, prototypes)
        if len(proto_scores) < 2:
            raise ValueError(
                "At least two memory labels are required for margin sweep."
            )

        mem_top1 = proto_scores[0]
        mem_margin = prototype_margin(proto_scores)
        base_ranked = router.route(case.text)
        base_top1 = base_ranked[0]

        cached.append(
            CachedScore(
                expected_label=case.expected_label,
                should_accept=case.should_accept,
                text=case.text,
                memory_label=mem_top1.label,
                memory_similarity=mem_top1.similarity,
                memory_margin=float(mem_margin),
                base_label=base_top1.label,
                base_similarity=base_top1.similarity,
                agreement=(mem_top1.label == base_top1.label),
            )
        )

    return cached


def evaluate_config(cached, sim_th, margin_th, require_agreement):
    tp = fp = tn = fn = 0
    correct_accept = 0
    wrong_label_accept = 0
    accepted = 0

    for row in cached:
        gate_accept = (
            row.memory_similarity >= sim_th
            and row.memory_margin >= margin_th
            and (row.agreement or not require_agreement)
        )

        if gate_accept:
            accepted += 1

        correct_label = (
            row.expected_label != "REJECT"
            and row.memory_label == row.expected_label
        )

        if row.should_accept:
            if gate_accept and correct_label:
                tp += 1
                correct_accept += 1
            else:
                fn += 1
                if gate_accept and not correct_label:
                    wrong_label_accept += 1
        else:
            if gate_accept:
                fp += 1
            else:
                tn += 1

    total = len(cached)
    target_count = sum(int(x.should_accept) for x in cached)
    reject_count = total - target_count

    precision = tp / max(1, tp + fp + wrong_label_accept)
    recall = tp / max(1, target_count)
    specificity = tn / max(1, reject_count)
    balanced = (recall + specificity) / 2.0
    coverage = accepted / max(1, total)
    accepted_accuracy = correct_accept / max(1, accepted)
    false_accept_rate = (fp + wrong_label_accept) / max(1, total)

    return {
        "similarity_threshold": sim_th,
        "margin_threshold": margin_th,
        "require_base_agreement": int(require_agreement),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "wrong_label_accept": wrong_label_accept,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "balanced_accuracy": balanced,
        "coverage": coverage,
        "accepted_accuracy": accepted_accuracy,
        "false_accept_rate": false_accept_rate,
    }


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = Tokenizer.load(args.tokenizer)
    model, checkpoint = LanguageModel.load_checkpoint(args.model, device=device)

    base = load_benchmark(args.benchmark)
    memory = load_semantic_memory(Path(args.memory))
    if len({row.label for row in memory}) < 2:
        raise ValueError(
            "At least two labels are required in semantic memory for v0.3.6."
        )

    router = SemanticRouter(model, tokenizer, alpha=args.alpha)
    router.fit(merge_samples(base, memory))
    prototypes = build_prototypes(router, memory)
    cases = load_cases(args.cases)

    print()
    print("=" * 78)
    print(" LLM_SEM v0.3.6 Memory Sim x Margin x Base Agreement Sweep")
    print("=" * 78)
    print("Device           :", device)
    if device.type == "cuda":
        print("GPU              :", torch.cuda.get_device_name(0))
    print("Checkpoint loss  :", checkpoint.get("loss"))
    print("Memory samples   :", len(memory))
    print("Memory labels    :", ", ".join(sorted(prototypes)))
    print("Evaluation cases :", len(cases))
    print()

    cached = cache_scores(router, prototypes, cases)

    with Path(args.detail_csv).open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        fieldnames = list(CachedScore.__dataclass_fields__.keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in cached:
            writer.writerow(row.__dict__)

    print("Per-query semantic evidence")
    print("-" * 78)
    print(
        f"{'Expected':<10} {'Mem':<10} {'MemSim':>8} "
        f"{'Margin':>8} {'Base':<10} {'Agree':>5}  Text"
    )
    for row in cached:
        print(
            f"{row.expected_label:<10} {row.memory_label:<10} "
            f"{row.memory_similarity:>8.4f} "
            f"{row.memory_margin:>8.4f} "
            f"{row.base_label:<10} "
            f"{str(row.agreement):>5}  {row.text}"
        )

    summary = []
    for require_agreement in (False, True):
        for sim_th in SIM_THRESHOLDS:
            for margin_th in MARGIN_THRESHOLDS:
                summary.append(
                    evaluate_config(
                        cached,
                        sim_th,
                        margin_th,
                        require_agreement,
                    )
                )

    summary.sort(
        key=lambda x: (
            x["balanced_accuracy"],
            x["accepted_accuracy"],
            -x["false_accept_rate"],
            x["recall"],
            x["coverage"],
        ),
        reverse=True,
    )

    with Path(args.summary_csv).open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)

    print()
    print("Top sweep configurations")
    print("-" * 78)
    print(
        f"{'Sim':>5} {'Margin':>7} {'Agree':>5} "
        f"{'Prec':>7} {'Recall':>7} {'Spec':>7} "
        f"{'BalAcc':>7} {'Cover':>7} {'False':>7}"
    )
    for row in summary[:15]:
        print(
            f"{row['similarity_threshold']:>5.2f} "
            f"{row['margin_threshold']:>7.3f} "
            f"{row['require_base_agreement']:>5d} "
            f"{row['precision']*100:>6.1f}% "
            f"{row['recall']*100:>6.1f}% "
            f"{row['specificity']*100:>6.1f}% "
            f"{row['balanced_accuracy']*100:>6.1f}% "
            f"{row['coverage']*100:>6.1f}% "
            f"{row['false_accept_rate']*100:>6.1f}%"
        )

    print()
    print("Summary CSV:", args.summary_csv)
    print("Detail CSV :", args.detail_csv)


if __name__ == "__main__":
    main()
