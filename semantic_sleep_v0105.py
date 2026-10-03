# semantic_sleep_v0105.py
#
# LLM_SEM v0.10.11
# Iterative /sleep consolidation orchestrator.

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from adaptive_semantic_learning import (
    load_semantic_memory_records,
    update_memory_status,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LLM_SEM v0.10.11 iterative full sleep consolidation"
    )
    p.add_argument("--memory", default="data/semantic_memory.jsonl")
    p.add_argument("--model", required=True)
    p.add_argument("--tokenizer", default="model/tokenizer.json")
    p.add_argument("--benchmark", default="my_benchmark.csv")
    p.add_argument("--candidate", default="model/model-sem-sleep-v0111.pt")
    p.add_argument("--semantic-candidate", default="model/model-sem-sleep-sem-v0111.pt")
    p.add_argument("--manifest", default="model/active-model.json")
    p.add_argument("--base-answer-memory", default="data/semantic_guided_qa_v097.json")
    p.add_argument("--learned-answer-memory", default="data/semantic_answer_memory_learned.jsonl")
    p.add_argument("--relation-memory", default="data/relation_memory_v0101.jsonl")
    p.add_argument("--sleep-dataset", default="data/semantic_sleep_qa_v0106.json")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--qa-epochs", type=int, default=240)
    p.add_argument("--qa-learning-rate", type=float, default=1e-5)
    p.add_argument("--qa-lm-head-lr", type=float, default=5e-5)
    p.add_argument("--qa-preserve-weight", type=float, default=5.0)
    p.add_argument("--qa-train-blocks", type=int, default=2)
    p.add_argument("--sleep-max-rounds", type=int, default=5)
    p.add_argument("--sleep-target-mean", type=float, default=0.80)
    p.add_argument("--sleep-target-min", type=float, default=0.60)
    p.add_argument("--sleep-min-improvement", type=float, default=0.01)
    p.add_argument("--sleep-max-stall-rounds", type=int, default=2)
    p.add_argument("--allow-cpu", action="store_true")
    return p.parse_args()


def run_step(command: list[str], name: str) -> None:
    print()
    print("=" * 96)
    print(" SLEEP STEP:", name)
    print("=" * 96)
    print(">", " ".join(command))
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"{name} failed with exit code {result.returncode}")


def active_records(memory: Path) -> list[dict]:
    return [
        row
        for row in load_semantic_memory_records(memory)
        if row.get("status") == "ACTIVE"
    ]


def records_for_texts(memory: Path, texts: set[str]) -> list[dict]:
    return [
        row
        for row in load_semantic_memory_records(memory)
        if str(row.get("text", "")) in texts
    ]


def consolidated_records(memory: Path) -> list[dict]:
    return [
        row
        for row in load_semantic_memory_records(memory)
        if row.get("status") == "CONSOLIDATED"
    ]


def has_nonempty_file(path: str) -> bool:
    p = Path(path)
    return p.exists() and bool(p.read_text(encoding="utf-8").strip())


def restore_targets_to_active(
    memory: Path,
    target_texts: set[str],
    *,
    model_version: str,
) -> None:
    for row in records_for_texts(memory, target_texts):
        if row.get("status") in {"TRAINING", "VALIDATING"}:
            update_memory_status(
                memory,
                str(row["text"]),
                "ACTIVE",
                model_version=model_version,
                verified=False,
            )


def round_checkpoint(final_candidate: Path, round_index: int) -> Path:
    return final_candidate.with_name(
        f"{final_candidate.stem}.round{round_index}{final_candidate.suffix}"
    )


def round_result(final_candidate: Path, round_index: int) -> Path:
    return final_candidate.with_name(
        f"{final_candidate.stem}.round{round_index}.json"
    )


def main() -> None:
    args = parse_args()

    memory = Path(args.memory)
    source = Path(args.model)
    final_candidate = Path(args.candidate)
    semantic_candidate = Path(args.semantic_candidate)

    if not memory.exists():
        raise FileNotFoundError(memory)
    if not source.exists():
        raise FileNotFoundError(source)
    if final_candidate.resolve() == source.resolve():
        raise ValueError("candidate checkpoint must differ from source model")

    pending = active_records(memory)
    target_texts = {str(row["text"]) for row in pending}
    has_answer_learning = (
        has_nonempty_file(args.learned_answer_memory)
        or has_nonempty_file(args.relation_memory)
    )

    if not pending and not has_answer_learning:
        print("SLEEP> nothing pending in Semantic/Answer/Relation Memory.")
        return

    print("=" * 96)
    print(" LLM_SEM v0.10.11 Iterative Sleep Consolidation")
    print("=" * 96)
    print("Source model       :", source)
    print("Final candidate    :", final_candidate)
    print("Semantic ACTIVE    :", len(pending))
    print("Answer learning    :", "YES" if has_answer_learning else "NO")
    print("Max QA rounds      :", args.sleep_max_rounds)
    print("Target mean sim    :", args.sleep_target_mean)
    print("Target min sim     :", args.sleep_target_min)
    print("Min improvement    :", args.sleep_min_improvement)
    print("Max stall rounds   :", args.sleep_max_stall_rounds)

    source_for_qa = source

    # Stage A: Semantic Memory -> internal semantic representation.
    if pending:
        for row in pending:
            update_memory_status(
                memory,
                str(row["text"]),
                "TRAINING",
                model_version=source.name,
                verified=False,
            )
        print("SLEEP> lifecycle ACTIVE -> TRAINING")

        semantic_cmd = [
            sys.executable,
            "semantic_memory_semantic_preserve_train_v078.py",
            "--memory", str(memory),
            "--model", str(source),
            "--tokenizer", args.tokenizer,
            "--benchmark", args.benchmark,
            "--output", str(semantic_candidate),
            "--epochs", str(args.epochs),
        ]
        if args.allow_cpu:
            semantic_cmd.append("--allow-cpu")

        try:
            run_step(semantic_cmd, "Semantic Memory -> internal semantic training")
        except Exception:
            restore_targets_to_active(
                memory,
                target_texts,
                model_version=source.name,
            )
            raise
        source_for_qa = semantic_candidate

    # Stage B: Build canonical QA dataset once.
    if has_answer_learning:
        build_cmd = [
            sys.executable,
            "build_sleep_qa_dataset_v0106.py",
            "--base", args.base_answer_memory,
            "--learned", args.learned_answer_memory,
            "--relation", args.relation_memory,
            "--output", args.sleep_dataset,
        ]
        run_step(build_cmd, "build canonical sleep QA dataset")

        current_source = Path(source_for_qa)
        completed = False
        previous_mean = -1.0
        stall_rounds = 0
        last_metrics: dict = {}
        last_round_checkpoint: Path | None = None

        for round_index in range(1, max(1, args.sleep_max_rounds) + 1):
            out = round_checkpoint(final_candidate, round_index)
            result_json = round_result(final_candidate, round_index)

            print()
            print("=" * 96)
            print(
                f" SLEEP QA ROUND {round_index}/{args.sleep_max_rounds} "
                f"source={current_source.name}"
            )
            print("=" * 96)

            qa_cmd = [
                sys.executable,
                "semantic_guided_answer_finetune_v097.py",
                "--model", str(current_source),
                "--tokenizer", args.tokenizer,
                "--dataset", args.sleep_dataset,
                "--benchmark", args.benchmark,
                "--output", str(out),
                "--epochs", str(args.qa_epochs),
                "--learning-rate", str(args.qa_learning_rate),
                "--lm-head-lr", str(args.qa_lm_head_lr),
                "--preserve-weight", str(args.qa_preserve_weight),
                "--train-blocks", str(args.qa_train_blocks),
                "--min-generation-sim", "0.0",
                "--result-json", str(result_json),
            ]
            if args.allow_cpu:
                qa_cmd.append("--allow-cpu")

            try:
                run_step(
                    qa_cmd,
                    f"iterative Answer/Relation internal training round {round_index}",
                )
            except Exception:
                if pending:
                    restore_targets_to_active(
                        memory,
                        target_texts,
                        model_version=source.name,
                    )
                raise

            if not result_json.exists():
                raise RuntimeError(
                    f"Sleep round did not create metrics: {result_json}"
                )
            metrics = json.loads(result_json.read_text(encoding="utf-8"))
            last_metrics = metrics
            mean_sim = float(metrics.get("generation_similarity_mean", 0.0))
            min_sim = float(metrics.get("generation_similarity_min", 0.0))
            sem_cos = float(metrics.get("semantic_cosine", 0.0))
            improvement = (
                mean_sim - previous_mean if previous_mean >= 0.0 else mean_sim
            )

            print(
                "SLEEP> round result: "
                f"mean={mean_sim:.6f} min={min_sim:.6f} "
                f"semantic_cosine={sem_cos:.6f} "
                f"improvement={improvement:+.6f}"
            )

            if sem_cos < 0.98:
                print(
                    "SLEEP> STOP: semantic preservation fell below 0.98. "
                    "Candidate will not be promoted."
                )
                return

            if (
                mean_sim >= args.sleep_target_mean
                and min_sim >= args.sleep_target_min
            ):
                completed = True
                last_round_checkpoint = out
                print(
                    f"SLEEP> INTERNAL LEARNING COMPLETE at round {round_index}: "
                    f"mean={mean_sim:.6f}, min={min_sim:.6f}"
                )
                break

            if (
                previous_mean >= 0.0
                and improvement < args.sleep_min_improvement
            ):
                stall_rounds += 1
            else:
                stall_rounds = 0

            if stall_rounds >= args.sleep_max_stall_rounds:
                print(
                    "SLEEP> STOP: generation similarity improvement stalled "
                    f"for {stall_rounds} rounds."
                )
                return

            previous_mean = mean_sim
            current_source = out
            last_round_checkpoint = out

        if not completed:
            print(
                "SLEEP> STOP: maximum rounds reached before internal-learning "
                "completion criteria were satisfied."
            )
            if last_metrics:
                print(
                    "SLEEP> final metrics: "
                    f"mean={float(last_metrics.get('generation_similarity_mean', 0.0)):.6f} "
                    f"min={float(last_metrics.get('generation_similarity_min', 0.0)):.6f}"
                )
            return

        assert last_round_checkpoint is not None
        final_candidate.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(last_round_checkpoint, final_candidate)
        print("SLEEP> selected completed checkpoint:", last_round_checkpoint)
        print("SLEEP> copied final candidate       :", final_candidate)
    else:
        final_candidate.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(semantic_candidate, final_candidate)

    # Stage C: Validate newly consolidated Semantic Memory against final model.
    if pending:
        validate_cmd = [
            sys.executable,
            "semantic_memory_batch_validate_v080.py",
            "--memory", str(memory),
            "--source", str(source),
            "--candidate", str(final_candidate),
            "--tokenizer", args.tokenizer,
            "--benchmark", args.benchmark,
        ]
        if args.allow_cpu:
            validate_cmd.append("--allow-cpu")
        run_step(validate_cmd, "final semantic validation")

        after_validation = records_for_texts(memory, target_texts)
        failed = [
            row for row in after_validation
            if row.get("status") != "CONSOLIDATED"
        ]
        if failed:
            print("SLEEP> promotion blocked: target Semantic Memory validation failed.")
            for row in failed:
                print("  -", row.get("status"), row.get("label"), repr(row.get("text")))
            return

    # Stage D: Retention/regression against consolidated knowledge.
    consolidated = consolidated_records(memory)
    if consolidated:
        retention_cmd = [
            sys.executable,
            "consolidated_retention_v094.py",
            "--source", str(source),
            "--candidate", str(final_candidate),
            "--tokenizer", args.tokenizer,
            "--benchmark", args.benchmark,
            "--memory", str(memory),
        ]
        if args.allow_cpu:
            retention_cmd.append("--allow-cpu")
        run_step(retention_cmd, "consolidated retention validation")

    # Stage E: Promote only after iterative completion + validation.
    promote_cmd = [
        sys.executable,
        "promote_active_model_v094.py",
        "--candidate", str(final_candidate),
        "--manifest", args.manifest,
        "--retention-pass",
        "--note", "v0.10.11 iterative /sleep consolidation",
    ]
    if args.allow_cpu:
        promote_cmd.append("--allow-cpu")
    run_step(promote_cmd, "active model promotion")

    print()
    print("=" * 96)
    print(" ITERATIVE SLEEP COMPLETE")
    print("=" * 96)
    print("Promoted model       :", final_candidate)
    print("Manifest             :", args.manifest)
    print("Semantic consolidated:", len(target_texts))
    print("Answer/Relation QA    :", "internally learned" if has_answer_learning else "not present")
    print("Restart chat.py to load the promoted internal model.")


if __name__ == "__main__":
    main()
