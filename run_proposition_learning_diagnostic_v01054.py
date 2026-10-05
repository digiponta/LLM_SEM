from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch

from proposition_bootstrap_v01050 import (
    concept_of,
    proposition_generation_mean,
    proposition_rows,
    save_dataset,
    split_definition,
    tokenizer_unknown_chars,
)
from semantic_guided_answer_finetune_v097 import load_dataset
from tokenizer import Tokenizer


def parse_args():
    p = argparse.ArgumentParser(
        description="LLM_SEM v0.10.56 First-Divergence Proposition Diagnostic"
    )
    p.add_argument(
        "--source",
        default="model/model-sem-sleep-v0144.pt",
    )
    p.add_argument(
        "--dataset",
        default="data/semantic_sleep_qa_v0106.json",
    )
    p.add_argument("--concept", default="量子センサー")
    p.add_argument("--benchmark", default="my_benchmark.csv")
    p.add_argument("--tokenizer", default="model/tokenizer.json")
    p.add_argument(
        "--output",
        default="model/model-sem-diagnostic-v0156.pt",
    )
    p.add_argument("--epochs", type=int, default=140)
    p.add_argument("--learning-rate", type=float, default=5e-6)
    p.add_argument("--lm-head-lr", type=float, default=2.5e-5)
    p.add_argument("--preserve-weight", type=float, default=5.0)
    p.add_argument("--protected-distill-weight", type=float, default=4.0)
    p.add_argument("--new-knowledge-weight", type=float, default=8.0)
    p.add_argument("--train-blocks", type=int, default=1)
    p.add_argument("--allow-cpu", action="store_true")
    return p.parse_args()


def remove(path: Path) -> None:
    if path.exists():
        path.unlink()


def run(cmd: list[str]) -> int:
    print(">", " ".join(cmd))
    return subprocess.run(cmd, check=False).returncode


def mean_for_queries(metrics: dict, queries: set[str], field: str) -> float:
    vals = [
        float(row.get(field, 0.0))
        for row in metrics.get("details", [])
        if str(row.get("query", "")) in queries
    ]
    return sum(vals) / len(vals) if vals else 0.0


def classify(
    *,
    unk_count: int,
    nll_rel_drop: float,
    param_rel: float,
    proposition_gain: float,
    full_gain: float,
    known_failures: int,
) -> tuple[str, str]:
    if unk_count:
        return (
            "TOKENIZER_BLOCK",
            "Tokenizer cannot represent every target character.",
        )
    if nll_rel_drop < 0.01:
        return (
            "NO_LEARNING_SIGNAL",
            "Target NLL barely changed; inspect target construction, prompt, and optimizer inputs.",
        )
    if param_rel < 1.0e-7:
        return (
            "NO_PARAMETER_UPDATE",
            "NLL changed but trainable parameter delta is effectively zero; inspect optimizer/trainable configuration.",
        )
    if proposition_gain < 0.02:
        return (
            "LATENT_ONLY",
            "NLL and weights changed, but proposition generation did not move enough; the update remains below the decoding decision boundary.",
        )
    if known_failures > 0:
        return (
            "RETENTION_CONFLICT",
            "Proposition generation moved, but protected knowledge regressed.",
        )
    if full_gain < 0.02:
        return (
            "COMPOSITION_GAP",
            "Propositions improved safely, but the original full query did not improve; composition/generalization is the bottleneck.",
        )
    return (
        "END_TO_END_PROGRESS",
        "Learning signal, parameter update, proposition generation, and full runtime all moved in the expected direction.",
    )


def main():
    args = parse_args()
    source = Path(args.source)
    dataset = Path(args.dataset)
    output = Path(args.output)

    if not source.exists():
        raise FileNotFoundError(source)
    if not dataset.exists():
        raise FileNotFoundError(dataset)

    rows = load_dataset(dataset)
    targets = [
        dict(row)
        for row in rows
        if bool(row.get("must_train", False))
        and concept_of(row) == args.concept
    ]
    if not targets:
        raise RuntimeError(
            f"No mandatory rows found for concept={args.concept!r}"
        )

    protected = [
        dict(row)
        for row in rows
        if bool(row.get("must_train", False))
        and concept_of(row) != args.concept
    ]

    canonical = str(targets[0].get("answer", "")).strip()
    parts = split_definition(args.concept, canonical)
    if len(parts) < 2:
        raise RuntimeError("Target definition could not be decomposed.")

    tokenizer = Tokenizer.load(args.tokenizer)
    target_text = (
        args.concept
        + canonical
        + "\n".join(str(row.get("query", "")) for row in targets)
        + "\n".join(parts)
    )
    unknown = tokenizer_unknown_chars(tokenizer, target_text)

    print("=" * 112)
    print(" LLM_SEM v0.10.54 Proposition Learning Diagnostic")
    print("=" * 112)
    print("Source             :", source)
    print("Dataset            :", dataset)
    print("Concept            :", args.concept)
    print("Target rows        :", len(targets))
    print("Protected rows     :", len(protected))
    print("Canonical chars    :", len(canonical))
    print("Tokenizer UNK      :", unknown or "(none)")
    print("Prefix focus       :", f"{min(12, max(1, len(args.concept) + 2))} tokens x4.0")
    for i, part in enumerate(parts, 1):
        print(f"P{i}                 :", part)

    if unknown:
        state, reason = classify(
            unk_count=len(unknown),
            nll_rel_drop=0.0,
            param_rel=0.0,
            proposition_gain=0.0,
            full_gain=0.0,
            known_failures=0,
        )
        print()
        print("DIAGNOSIS          :", state)
        print("Reason             :", reason)
        raise SystemExit(2)

    aux_rows = proposition_rows(args.concept, targets, parts)
    diag_dataset = output.with_name(
        f"{output.stem}.dataset.json"
    )
    train_json = output.with_name(
        f"{output.stem}.train.json"
    )
    runtime_json = output.with_name(
        f"{output.stem}.runtime.json"
    )

    save_dataset(
        diag_dataset,
        aux_rows,
        protected,
        mode="v0.10.54-proposition-diagnostic",
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    if device.type != "cuda" and not args.allow_cpu:
        raise RuntimeError("CUDA unavailable. Use --allow-cpu only for testing.")

    before_prop, before_details = proposition_generation_mean(
        source,
        args.tokenizer,
        args.benchmark,
        aux_rows,
        device,
    )

    remove(output)
    remove(train_json)
    train_cmd = [
        sys.executable,
        "semantic_guided_answer_finetune_v097.py",
        "--model", str(source),
        "--tokenizer", args.tokenizer,
        "--dataset", str(diag_dataset),
        "--benchmark", args.benchmark,
        "--output", str(output),
        "--epochs", str(args.epochs),
        "--learning-rate", str(args.learning_rate),
        "--lm-head-lr", str(args.lm_head_lr),
        "--preserve-weight", str(args.preserve_weight),
        "--protected-distill-weight",
        str(args.protected_distill_weight),
        "--new-knowledge-weight",
        str(args.new_knowledge_weight),
        "--first-divergence-weight", "2.0",
        "--first-divergence-margin", "1.0",
        "--train-blocks", str(args.train_blocks),
        "--min-generation-sim", "0.0",
        "--result-json", str(train_json),
        "--prefer-final-state",
        "--concept-balanced",
    ]
    if args.allow_cpu:
        train_cmd.append("--allow-cpu")

    print()
    print("DIAG STEP 1: controlled proposition fine-tuning")
    code = run(train_cmd)
    if code != 0:
        raise RuntimeError(f"fine-tuning failed with exit code {code}")
    if not output.exists():
        raise RuntimeError(f"diagnostic checkpoint missing: {output}")
    if not train_json.exists():
        raise RuntimeError(f"diagnostic metrics missing: {train_json}")

    train_metrics = json.loads(
        train_json.read_text(encoding="utf-8")
    )
    nll_before = float(train_metrics.get("target_nll_before", 0.0))
    nll_after = float(
        train_metrics.get("target_nll_after", nll_before)
    )
    nll_drop = nll_before - nll_after
    nll_rel_drop = (
        nll_drop / nll_before if nll_before > 0.0 else 0.0
    )
    param_l2 = float(
        train_metrics.get("trainable_param_delta_l2", 0.0)
    )
    param_rel = float(
        train_metrics.get("trainable_param_relative_delta", 0.0)
    )
    semantic_cos = float(
        train_metrics.get("semantic_cosine", 0.0)
    )

    after_prop, after_details = proposition_generation_mean(
        output,
        args.tokenizer,
        args.benchmark,
        aux_rows,
        device,
    )
    proposition_gain = after_prop - before_prop

    remove(runtime_json)
    runtime_cmd = [
        sys.executable,
        "runtime_answer_retention_v01015.py",
        "--source", str(source),
        "--candidate", str(output),
        "--dataset", str(dataset),
        "--benchmark", args.benchmark,
        "--tokenizer", args.tokenizer,
        "--result-json", str(runtime_json),
    ]
    if args.allow_cpu:
        runtime_cmd.append("--allow-cpu")

    print()
    print("DIAG STEP 2: full runtime retention/progress")
    runtime_code = run(runtime_cmd)
    if not runtime_json.exists():
        raise RuntimeError(
            f"runtime diagnostic metrics missing: {runtime_json}"
        )
    runtime = json.loads(
        runtime_json.read_text(encoding="utf-8")
    )

    queries = {
        str(row.get("query", ""))
        for row in targets
    }
    full_before = mean_for_queries(
        runtime,
        queries,
        "source_canonical_similarity",
    )
    full_after = mean_for_queries(
        runtime,
        queries,
        "candidate_canonical_similarity",
    )
    full_gain = full_after - full_before
    known_failures = int(runtime.get("known_failures", 10**9))

    state, reason = classify(
        unk_count=0,
        nll_rel_drop=nll_rel_drop,
        param_rel=param_rel,
        proposition_gain=proposition_gain,
        full_gain=full_gain,
        known_failures=known_failures,
    )

    print()
    print("=" * 112)
    print(" DIAGNOSTIC MATRIX")
    print("=" * 112)
    print("Tokenizer coverage :", "PASS")
    print(
        "Target NLL         : "
        f"{nll_before:.6f} -> {nll_after:.6f} "
        f"drop={nll_drop:+.6f} rel={nll_rel_drop:+.2%}"
    )
    print(
        "Parameter delta    : "
        f"L2={param_l2:.6f} rel={param_rel:.9f}"
    )
    print("Semantic cosine    :", f"{semantic_cos:.6f}")
    print(
        "Proposition runtime: "
        f"{before_prop:.6f} -> {after_prop:.6f} "
        f"gain={proposition_gain:+.6f}"
    )
    print(
        "Full-query runtime : "
        f"{full_before:.6f} -> {full_after:.6f} "
        f"gain={full_gain:+.6f}"
    )
    print("Known failures     :", known_failures)
    print("Runtime exit code  :", runtime_code)

    print()
    print("Proposition details")
    print("-------------------")
    before_by_query = {
        query: (sim, expected, generated)
        for query, sim, expected, generated in before_details
    }
    for query, sim, expected, generated in after_details:
        bsim, _, bgenerated = before_by_query.get(
            query,
            (0.0, expected, ""),
        )
        print(
            f"{query!r}: {bsim:.6f} -> {sim:.6f} "
            f"gain={sim - bsim:+.6f}"
        )
        print("  expected :", expected)
        print("  before   :", bgenerated)
        print("  after    :", generated)

    print()
    print("=" * 112)
    print(" DIAGNOSIS")
    print("=" * 112)
    print("State  :", state)
    print("Reason :", reason)
    print("NOTE   : diagnostic checkpoint is NOT promoted.")

    summary = output.with_name(
        f"{output.stem}.diagnostic.json"
    )
    summary.write_text(
        json.dumps(
            {
                "version": "v0.10.56",
                "concept": args.concept,
                "source": str(source),
                "candidate": str(output),
                "tokenizer_unknown": unknown,
                "target_nll_before": nll_before,
                "target_nll_after": nll_after,
                "target_nll_relative_drop": nll_rel_drop,
                "trainable_param_delta_l2": param_l2,
                "trainable_param_relative_delta": param_rel,
                "semantic_cosine": semantic_cos,
                "proposition_before": before_prop,
                "proposition_after": after_prop,
                "proposition_gain": proposition_gain,
                "full_before": full_before,
                "full_after": full_after,
                "full_gain": full_gain,
                "known_failures": known_failures,
                "runtime_exit_code": runtime_code,
                "diagnosis": state,
                "reason": reason,
            },
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    print("Summary:", summary)


if __name__ == "__main__":
    main()
