# semantic_sleep_v0105.py
#
# LLM_SEM v0.10.11
# Iterative /sleep consolidation orchestrator.

from __future__ import annotations

import argparse
import filecmp
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
    p.add_argument("--sleep-concept-target-mean", type=float, default=0.75)
    p.add_argument("--sleep-concept-target-min", type=float, default=0.70)
    p.add_argument("--sleep-min-improvement", type=float, default=0.01)
    p.add_argument("--sleep-max-stall-rounds", type=int, default=2)
    p.add_argument("--sleep-min-termination-rate", type=float, default=1.0)
    p.add_argument("--sleep-max-abnormal-ratio", type=float, default=0.02)
    p.add_argument("--sleep-max-repetition-ratio", type=float, default=0.20)
    p.add_argument("--incremental-distill-weight", type=float, default=8.0)
    p.add_argument("--incremental-lr-scale", type=float, default=0.5)
    p.add_argument("--incremental-train-blocks", type=int, default=1)
    p.add_argument("--incremental-chunk-epochs", type=int, default=40)
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


def run_step_code(command: list[str], name: str) -> int:
    print()
    print("=" * 96)
    print(" SLEEP STEP:", name)
    print("=" * 96)
    print(">", " ".join(command))
    result = subprocess.run(command, check=False)
    return int(result.returncode)


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


def internal_learning_complete(
    mean_similarity: float,
    min_similarity: float,
    *,
    target_mean: float,
    target_min: float,
) -> bool:
    return (
        mean_similarity >= target_mean
        and min_similarity >= target_min
    )


def next_stall_count(
    previous_mean: float,
    current_mean: float,
    current_stall: int,
    *,
    min_improvement: float,
) -> int:
    if previous_mean < 0.0:
        return 0
    improvement = current_mean - previous_mean
    return current_stall + 1 if improvement < min_improvement else 0


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
        final_candidate = final_candidate.with_name(
            f"{final_candidate.stem}.next{final_candidate.suffix}"
        )
        print(
            "SLEEP> candidate matched active source; "
            "using next candidate:",
            final_candidate,
        )
    if semantic_candidate.resolve() == source.resolve():
        semantic_candidate = semantic_candidate.with_name(
            f"{semantic_candidate.stem}.next{semantic_candidate.suffix}"
        )

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
    print(" LLM_SEM v0.10.38 Iterative Protected-Repair Partial Commit Sleep")
    print("=" * 96)
    print("Source model       :", source)
    print("Final candidate    :", final_candidate)
    print("Semantic ACTIVE    :", len(pending))
    print("Answer learning    :", "YES" if has_answer_learning else "NO")
    print("Max QA rounds      :", args.sleep_max_rounds)
    print("Target mean sim    :", args.sleep_target_mean)
    print("Target min sim     :", args.sleep_target_min)
    print("Concept mean target:", args.sleep_concept_target_mean)
    print("Concept min target :", args.sleep_concept_target_min)
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
        qa_dataset_path = Path(args.sleep_dataset)
        incremental_mode = False
        completed = False
        previous_mean = -1.0
        stall_rounds = 0
        last_metrics: dict = {}
        last_round_checkpoint: Path | None = None
        best_safe_checkpoint: Path | None = None
        best_safe_new_failures = 10**9
        best_safe_canonical_mean = -1.0

        precheck_json = final_candidate.with_name(
            f"{final_candidate.stem}.precheck.json"
        )
        precheck_cmd = [
            sys.executable,
            "retention_first_sleep_check_v01022.py",
            "--model", str(current_source),
            "--dataset", args.sleep_dataset,
            "--benchmark", args.benchmark,
            "--tokenizer", args.tokenizer,
            "--result-json", str(precheck_json),
        ]
        if args.allow_cpu:
            precheck_cmd.append("--allow-cpu")

        precheck_code = run_step_code(
            precheck_cmd,
            "retention-first internalization precheck",
        )

        if precheck_code == 0:
            final_candidate.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(current_source, final_candidate)
            completed = True
            last_round_checkpoint = final_candidate
            print(
                "SLEEP> RETENTION-FIRST PASS: all mandatory concepts "
                "already internalized; Balanced QA sleep SKIPPED."
            )
            print("SLEEP> preserved source candidate:", final_candidate)
        elif precheck_code == 2:
            print(
                "SLEEP> retention-first precheck SKIP: insufficient concepts; "
                "continue with Balanced QA sleep."
            )
        elif precheck_code == 1:
            print(
                "SLEEP> retention-first precheck FAIL: at least one concept "
                "needs learning; build incremental-only QA dataset."
            )
            incremental_dataset = final_candidate.with_name(
                f"{final_candidate.stem}.incremental.json"
            )
            incremental_cmd = [
                sys.executable,
                "build_incremental_sleep_dataset_v01023.py",
                "--dataset", args.sleep_dataset,
                "--precheck", str(precheck_json),
                "--output", str(incremental_dataset),
            ]
            incremental_code = run_step_code(
                incremental_cmd,
                "build incremental new-knowledge QA dataset",
            )
            if incremental_code == 0:
                qa_dataset_path = incremental_dataset
                incremental_mode = True
                print("SLEEP> incremental QA dataset:", qa_dataset_path)
                print("SLEEP> protected distill weight:", args.incremental_distill_weight)
                print("SLEEP> incremental LR scale    :", args.incremental_lr_scale)
                print("SLEEP> incremental train blocks:", args.incremental_train_blocks)
            else:
                raise RuntimeError(
                    "incremental sleep dataset could not be constructed "
                    f"(exit code {incremental_code})"
                )
        else:
            raise RuntimeError(
                f"retention-first precheck failed with exit code {precheck_code}"
            )

        if incremental_mode and not completed:
            one_by_one_candidate = final_candidate.with_name(
                f"{final_candidate.stem}.onebyone{final_candidate.suffix}"
            )
            one_by_one_cmd = [
                sys.executable,
                "one_by_one_sleep_v01030.py",
                "--source", str(current_source),
                "--incremental-dataset", str(qa_dataset_path),
                "--full-dataset", args.sleep_dataset,
                "--precheck", str(precheck_json),
                "--benchmark", args.benchmark,
                "--tokenizer", args.tokenizer,
                "--output", str(one_by_one_candidate),
                "--epochs", str(args.qa_epochs),
                "--learning-rate", str(
                    args.qa_learning_rate * args.incremental_lr_scale
                ),
                "--lm-head-lr", str(
                    args.qa_lm_head_lr * args.incremental_lr_scale
                ),
                "--preserve-weight", str(args.qa_preserve_weight),
                "--replay-weight", str(args.incremental_distill_weight),
                "--new-weight", "3.0",
                "--train-blocks", str(args.incremental_train_blocks),
            ]
            if args.allow_cpu:
                one_by_one_cmd.append("--allow-cpu")

            one_by_one_code = run_step_code(
                one_by_one_cmd,
                "one-by-one incremental consolidation",
            )
            if one_by_one_code == 0:
                final_candidate.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(one_by_one_candidate, final_candidate)
                completed = True
                last_round_checkpoint = final_candidate
                print(
                    "SLEEP> one-by-one consolidation reached COMPLETE:",
                    final_candidate,
                )
            elif one_by_one_code == 3:
                partial_state = one_by_one_candidate.with_name(
                    f"{one_by_one_candidate.stem}.partial-state.json"
                )
                if not partial_state.exists():
                    raise RuntimeError(
                        f"partial commit state missing: {partial_state}"
                    )
                partial = json.loads(
                    partial_state.read_text(encoding="utf-8")
                )
                accepted_rows = int(partial.get("accepted_rows", 0))
                runtime = dict(partial.get("runtime", {}))
                known_failures = int(
                    runtime.get("known_failures", 10**9)
                )
                if (
                    partial.get("state") != "PARTIAL"
                    or accepted_rows <= 0
                    or known_failures != 0
                ):
                    raise RuntimeError(
                        "invalid PARTIAL commit: expected accepted_rows > 0 "
                        "and known_failures == 0"
                    )

                final_candidate.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(one_by_one_candidate, final_candidate)
                final_partial_state = final_candidate.with_name(
                    f"{final_candidate.stem}.partial-state.json"
                )
                shutil.copy2(partial_state, final_partial_state)

                if pending:
                    restore_targets_to_active(
                        memory,
                        target_texts,
                        model_version=final_candidate.name,
                    )

                partial_promote_cmd = [
                    sys.executable,
                    "promote_active_model_v094.py",
                    "--candidate", str(final_candidate),
                    "--manifest", args.manifest,
                    "--retention-pass",
                    "--note",
                    "v0.10.38 ITERATIVE PROTECTED-REPAIR PARTIAL sleep commit",
                ]
                if args.allow_cpu:
                    partial_promote_cmd.append("--allow-cpu")
                run_step(
                    partial_promote_cmd,
                    "PARTIAL active model promotion",
                )

                print()
                print("=" * 96)
                print(" PARTIAL SLEEP COMMIT")
                print("=" * 96)
                print("State               : PARTIAL")
                print("Promoted model      :", final_candidate)
                print("Accepted rows       :", accepted_rows)
                print("Remaining failures  :", runtime.get("new_failures"))
                print("Known failures      :", known_failures)
                print("Manifest            :", args.manifest)
                print(
                    "Next /sleep will resume from this PARTIAL model "
                    "and retry only rows that are still below target."
                )
                return
            else:
                print(
                    "SLEEP> no safe row-level progress was found; "
                    "source model remains active."
                )
                return

        if not completed:
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

                effective_lr = (
                    args.qa_learning_rate * args.incremental_lr_scale
                    if incremental_mode else args.qa_learning_rate
                )
                effective_lm_head_lr = (
                    args.qa_lm_head_lr * args.incremental_lr_scale
                    if incremental_mode else args.qa_lm_head_lr
                )
                effective_train_blocks = (
                    args.incremental_train_blocks
                    if incremental_mode else args.qa_train_blocks
                )
                effective_epochs = (
                    args.incremental_chunk_epochs
                    if incremental_mode else args.qa_epochs
                )

                qa_cmd = [
                    sys.executable,
                    "semantic_guided_answer_finetune_v097.py",
                    "--model", str(current_source),
                    "--tokenizer", args.tokenizer,
                    "--dataset", str(qa_dataset_path),
                    "--benchmark", args.benchmark,
                    "--output", str(out),
                    "--epochs", str(effective_epochs),
                    "--learning-rate", str(effective_lr),
                    "--lm-head-lr", str(effective_lm_head_lr),
                    "--preserve-weight", str(args.qa_preserve_weight),
                    "--train-blocks", str(effective_train_blocks),
                    "--min-generation-sim", "0.0",
                    "--result-json", str(result_json),
                    "--prefer-final-state",
                    "--concept-balanced",
                ]
                if incremental_mode:
                    qa_cmd.extend([
                        "--protected-distill-weight",
                        str(args.incremental_distill_weight),
                    ])
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
                termination_rate = float(metrics.get("termination_rate", 0.0))
                abnormal_ratio_max = float(metrics.get("abnormal_ratio_max", 1.0))
                repetition_ratio_max = float(metrics.get("repetition_ratio_max", 1.0))
                concept_mean = float(metrics.get("concept_generation_mean", mean_sim))
                concept_min = float(metrics.get("concept_generation_min", min_sim))
                improvement = (
                    mean_sim - previous_mean if previous_mean >= 0.0 else mean_sim
                )

                print(
                    "SLEEP> round result: "
                    f"mean={mean_sim:.6f} min={min_sim:.6f} "
                    f"semantic_cosine={sem_cos:.6f} "
                    f"termination={termination_rate:.3f} "
                    f"abnormal={abnormal_ratio_max:.3f} "
                    f"repetition={repetition_ratio_max:.3f} "
                    f"concept_mean={concept_mean:.6f} concept_min={concept_min:.6f} "
                    f"improvement={improvement:+.6f}"
                )

                if sem_cos < 0.98:
                    print(
                        "SLEEP> STOP: semantic preservation fell below 0.98. "
                        "Candidate will not be promoted."
                    )
                    return

                if incremental_mode:
                    runtime_json = final_candidate.with_name(
                        f"{final_candidate.stem}.round{round_index}.runtime.json"
                    )
                    runtime_cmd = [
                        sys.executable,
                        "runtime_answer_retention_v01015.py",
                        "--source", str(source),
                        "--candidate", str(out),
                        "--dataset", args.sleep_dataset,
                        "--benchmark", args.benchmark,
                        "--tokenizer", args.tokenizer,
                        "--result-json", str(runtime_json),
                    ]
                    if args.allow_cpu:
                        runtime_cmd.append("--allow-cpu")

                    runtime_code = run_step_code(
                        runtime_cmd,
                        f"full runtime checkpoint validation round {round_index}",
                    )
                    if not runtime_json.exists():
                        raise RuntimeError(
                            f"runtime checkpoint validation did not create {runtime_json}"
                        )
                    runtime_metrics = json.loads(
                        runtime_json.read_text(encoding="utf-8")
                    )
                    known_failures = int(
                        runtime_metrics.get("known_failures", 10**9)
                    )
                    new_failures = int(
                        runtime_metrics.get("new_failures", 10**9)
                    )
                    runtime_mean = float(
                        runtime_metrics.get("candidate_canonical_mean", 0.0)
                    )

                    print(
                        "SLEEP> runtime checkpoint: "
                        f"known_failures={known_failures} "
                        f"new_failures={new_failures} "
                        f"canonical_mean={runtime_mean:.6f}"
                    )

                    safe = known_failures == 0
                    better_safe = (
                        safe
                        and (
                            new_failures < best_safe_new_failures
                            or (
                                new_failures == best_safe_new_failures
                                and runtime_mean > best_safe_canonical_mean
                            )
                        )
                    )
                    if better_safe:
                        best_safe_checkpoint = out
                        best_safe_new_failures = new_failures
                        best_safe_canonical_mean = runtime_mean
                        print(
                            "SLEEP> BEST SAFE CHECKPOINT:",
                            out,
                            f"(new_failures={new_failures}, "
                            f"canonical_mean={runtime_mean:.6f})",
                        )

                    if runtime_code == 0:
                        completed = True
                        last_round_checkpoint = out
                        print(
                            "SLEEP> FULL RUNTIME PASS at incremental round "
                            f"{round_index}; checkpoint selected."
                        )
                        break

                    if known_failures > 0:
                        print(
                            "SLEEP> protected knowledge regression detected; "
                            "this checkpoint will NOT become the next training source."
                        )
                        if best_safe_checkpoint is not None:
                            current_source = best_safe_checkpoint
                        else:
                            current_source = Path(source_for_qa)
                        continue

                    # Safe checkpoint: advance from it even if the new concept
                    # still needs more learning.
                    current_source = out
                    last_round_checkpoint = out

                quality_ok = (
                    termination_rate >= args.sleep_min_termination_rate
                    and abnormal_ratio_max <= args.sleep_max_abnormal_ratio
                    and repetition_ratio_max <= args.sleep_max_repetition_ratio
                )

                if (
                    not incremental_mode
                    and internal_learning_complete(
                        mean_sim,
                        min_sim,
                        target_mean=args.sleep_target_mean,
                        target_min=args.sleep_target_min,
                    )
                    and concept_mean >= args.sleep_concept_target_mean
                    and concept_min >= args.sleep_concept_target_min
                    and quality_ok
                ):
                    completed = True
                    last_round_checkpoint = out
                    print(
                        f"SLEEP> INTERNAL LEARNING COMPLETE at round {round_index}: "
                        f"mean={mean_sim:.6f}, min={min_sim:.6f}, "
                        f"concept_mean={concept_mean:.6f}, concept_min={concept_min:.6f}"
                    )
                    break

                if not incremental_mode:
                    stall_rounds = next_stall_count(
                        previous_mean,
                        mean_sim,
                        stall_rounds,
                        min_improvement=args.sleep_min_improvement,
                    )

                    if stall_rounds >= args.sleep_max_stall_rounds:
                        print(
                            "SLEEP> STOP: generation similarity improvement stalled "
                            f"for {stall_rounds} rounds."
                        )
                        return

                    previous_mean = mean_sim
                    current_source = out
                    last_round_checkpoint = out
                else:
                    previous_mean = mean_sim

        if not completed:
            if incremental_mode and best_safe_checkpoint is not None:
                print(
                    "SLEEP> STOP: no checkpoint passed the full runtime gate "
                    "within the incremental search budget."
                )
                print(
                    "SLEEP> best safe checkpoint retained for diagnostics:",
                    best_safe_checkpoint,
                    f"new_failures={best_safe_new_failures}",
                    f"canonical_mean={best_safe_canonical_mean:.6f}",
                )
            else:
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
        if last_round_checkpoint.resolve() != final_candidate.resolve():
            shutil.copy2(last_round_checkpoint, final_candidate)
        print("SLEEP> selected completed checkpoint:", last_round_checkpoint)
        print("SLEEP> copied final candidate       :", final_candidate)
    else:
        final_candidate.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(semantic_candidate, final_candidate)

    # Stage C: Optional selective surface repair with automatic rollback.
    surface_candidate = final_candidate.with_name(
        f"{final_candidate.stem}.surface{final_candidate.suffix}"
    )
    surface_cmd = [
        sys.executable,
        "selective_surface_repair_v01020.py",
        "--model", str(final_candidate),
        "--output", str(surface_candidate),
        "--dataset", args.sleep_dataset,
        "--benchmark", args.benchmark,
        "--tokenizer", args.tokenizer,
    ]
    if args.allow_cpu:
        surface_cmd.append("--allow-cpu")
    run_step(surface_cmd, "selective surface repair")
    shutil.copy2(surface_candidate, final_candidate)
    print("SLEEP> selected surface candidate:", final_candidate)

    if (
        not pending
        and source.exists()
        and final_candidate.exists()
        and filecmp.cmp(source, final_candidate, shallow=False)
    ):
        print()
        print("=" * 96)
        print(" RETENTION-FIRST NO-OP SLEEP")
        print("=" * 96)
        print("SLEEP> active model already satisfies mandatory internal knowledge.")
        print("SLEEP> selective surface repair produced no safe improvement.")
        print("SLEEP> KEEP_SOURCE:", source)
        print("SLEEP> promotion skipped; active-model manifest unchanged.")
        return

    # Stage D: Validate newly consolidated Semantic Memory against final model.
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

    # Stage E: Retention/regression against consolidated knowledge.
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

    # Stage F: Answer/runtime/multi-knowledge retention gates.
    answer_retention_cmd = [
        sys.executable,
        "answer_retention_v01014.py",
        "--source", str(source),
        "--candidate", str(final_candidate),
        "--dataset", args.sleep_dataset,
        "--tokenizer", args.tokenizer,
    ]
    if args.allow_cpu:
        answer_retention_cmd.append("--allow-cpu")
    run_step(answer_retention_cmd, "dataset answer retention")

    runtime_retention_cmd = [
        sys.executable,
        "runtime_answer_retention_v01015.py",
        "--source", str(source),
        "--candidate", str(final_candidate),
        "--dataset", args.sleep_dataset,
        "--benchmark", args.benchmark,
        "--tokenizer", args.tokenizer,
    ]
    if args.allow_cpu:
        runtime_retention_cmd.append("--allow-cpu")
    run_step(runtime_retention_cmd, "runtime /internal answer retention")

    multi_knowledge_cmd = [
        sys.executable,
        "multi_knowledge_internalization_v01016.py",
        "--model", str(final_candidate),
        "--dataset", args.sleep_dataset,
        "--benchmark", args.benchmark,
        "--tokenizer", args.tokenizer,
    ]
    if args.allow_cpu:
        multi_knowledge_cmd.append("--allow-cpu")
    run_step(multi_knowledge_cmd, "multi-knowledge internalization")

    # Stage G: Promote only after every retention/internalization gate.
    promote_cmd = [
        sys.executable,
        "promote_active_model_v094.py",
        "--candidate", str(final_candidate),
        "--manifest", args.manifest,
        "--retention-pass",
        "--note", "v0.10.38 complete iterative protected-repair sleep",
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
