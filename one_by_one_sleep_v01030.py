# one_by_one_sleep_v01030.py
#
# LLM_SEM v0.10.34
# Learn exactly one new QA row at a time.
# After each row:
#   - protect previously known/accepted rows with runtime replay,
#   - validate protected rows,
#   - require the current target row to improve,
#   - ACCEPT -> candidate becomes next source,
#   - REJECT -> rollback and continue from previous source.

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(
        description="LLM_SEM v0.10.34 Partial Commit One-by-One Sleep"
    )
    p.add_argument("--source", required=True)
    p.add_argument("--incremental-dataset", required=True)
    p.add_argument("--full-dataset", required=True)
    p.add_argument("--precheck", required=True)
    p.add_argument("--benchmark", default="my_benchmark.csv")
    p.add_argument("--tokenizer", default="model/tokenizer.json")
    p.add_argument("--output", required=True)
    p.add_argument("--epochs", type=int, default=160)
    p.add_argument("--learning-rate", type=float, default=5e-6)
    p.add_argument("--lm-head-lr", type=float, default=2.5e-5)
    p.add_argument("--preserve-weight", type=float, default=5.0)
    p.add_argument("--replay-weight", type=float, default=2.0)
    p.add_argument("--new-weight", type=float, default=3.0)
    p.add_argument("--train-blocks", type=int, default=1)
    p.add_argument("--min-target-sim", type=float, default=0.70)
    p.add_argument("--min-target-gain", type=float, default=0.10)
    p.add_argument("--max-protected-drop", type=float, default=0.05)
    p.add_argument("--allow-cpu", action="store_true")
    return p.parse_args()


def run(cmd):
    return subprocess.run(cmd, check=False).returncode


def concept_of(row: dict) -> str:
    concepts = [
        str(x).strip()
        for x in row.get("concepts", [])
        if str(x).strip()
    ]
    return concepts[0] if concepts else str(row.get("query", "")).strip()


def load_samples(path: Path) -> list[dict]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    return [dict(row) for row in obj.get("samples", [])]


def save_step_dataset(
    path: Path,
    target: dict,
    protected_rows: list[dict],
):
    rows = []
    for raw in protected_rows:
        row = dict(raw)
        row["must_train"] = False
        row["protected"] = True
        rows.append(row)

    target_row = dict(target)
    target_row["must_train"] = True
    target_row["protected"] = False
    rows.append(target_row)

    path.write_text(
        json.dumps(
            {
                "version": "v0.10.34",
                "mode": "one-by-one-partial-commit",
                "samples": rows,
            },
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )


def runtime_result(
    *,
    source: Path,
    candidate: Path,
    full_dataset: str,
    benchmark: str,
    tokenizer: str,
    result_json: Path,
    allow_cpu: bool,
) -> tuple[int, dict]:
    cmd = [
        sys.executable,
        "runtime_answer_retention_v01015.py",
        "--source", str(source),
        "--candidate", str(candidate),
        "--dataset", full_dataset,
        "--benchmark", benchmark,
        "--tokenizer", tokenizer,
        "--result-json", str(result_json),
    ]
    if allow_cpu:
        cmd.append("--allow-cpu")
    code = run(cmd)
    if not result_json.exists():
        raise RuntimeError(f"runtime result not created: {result_json}")
    return code, json.loads(result_json.read_text(encoding="utf-8"))


def find_detail(metrics: dict, query: str) -> dict | None:
    for item in metrics.get("details", []):
        if str(item.get("query", "")) == query:
            return item
    return None


def main():
    args = parse_args()
    source = Path(args.source)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    incremental_rows = load_samples(Path(args.incremental_dataset))
    full_rows = load_samples(Path(args.full_dataset))
    precheck = json.loads(Path(args.precheck).read_text(encoding="utf-8"))

    new_rows = [
        row for row in incremental_rows
        if bool(row.get("must_train", False))
    ]
    protected_concepts = {
        str(item.get("concept", "")).strip()
        for item in precheck.get("details", [])
        if bool(item.get("passed", False))
        and str(item.get("concept", "")).strip()
    }

    # Protect every mandatory runtime row belonging to an already-passing
    # concept, not only the representative precheck query.
    protected_rows = [
        dict(row)
        for row in full_rows
        if bool(row.get("must_train", False))
        and concept_of(row) in protected_concepts
    ]

    # Recover row-level PARTIAL progress from the current source model.
    # A previously accepted query may belong to a concept that is still
    # incomplete overall, so concept-level precheck alone cannot protect it.
    baseline_json = output.with_name(f"{output.stem}.baseline.runtime.json")
    _, baseline_metrics = runtime_result(
        source=source,
        candidate=source,
        full_dataset=args.full_dataset,
        benchmark=args.benchmark,
        tokenizer=args.tokenizer,
        result_json=baseline_json,
        allow_cpu=args.allow_cpu,
    )
    baseline_by_query = {
        str(item.get("query", "")): item
        for item in baseline_metrics.get("details", [])
    }

    recovered_partial_rows = []
    remaining_new_rows = []
    protected_query_set = {
        str(row.get("query", ""))
        for row in protected_rows
    }

    for row in new_rows:
        query = str(row.get("query", ""))
        detail = baseline_by_query.get(query)
        current_sim = (
            float(detail.get("source_canonical_similarity", 0.0))
            if detail is not None
            else 0.0
        )
        if current_sim >= args.min_target_sim:
            recovered = dict(row)
            recovered["must_train"] = False
            recovered["protected"] = True
            if query not in protected_query_set:
                protected_rows.append(recovered)
                protected_query_set.add(query)
            recovered_partial_rows.append(recovered)
        else:
            remaining_new_rows.append(row)

    new_rows = remaining_new_rows

    print("=" * 108)
    print(" LLM_SEM v0.10.34 Partial Commit One-by-One Sleep")
    print("=" * 108)
    print("Source model       :", source)
    print("New training rows  :", len(new_rows))
    print("Initial protected  :", len(protected_rows))
    print("Recovered PARTIAL  :", len(recovered_partial_rows))
    print("Epochs / row       :", args.epochs)
    print("Replay weight      :", args.replay_weight)
    print("New knowledge wt   :", args.new_weight)
    print()

    if not new_rows:
        print("ONE-BY-ONE> no new rows.")
        shutil.copy2(source, output)
        raise SystemExit(0)

    current_source = source
    accepted_rows: list[dict] = []
    accepted = 0
    rejected = 0
    accepted_details = []
    rejected_details = []

    for index, target in enumerate(new_rows, 1):
        query = str(target.get("query", ""))
        answer = str(target.get("answer", ""))
        concept = concept_of(target)

        step_dataset = output.with_name(
            f"{output.stem}.step{index}.json"
        )
        step_candidate = output.with_name(
            f"{output.stem}.step{index}{output.suffix}"
        )
        train_json = output.with_name(
            f"{output.stem}.step{index}.train.json"
        )
        runtime_json = output.with_name(
            f"{output.stem}.step{index}.runtime.json"
        )

        anchors = protected_rows + accepted_rows
        save_step_dataset(step_dataset, target, anchors)

        print()
        print("=" * 108)
        print(f" ONE-BY-ONE STEP {index}/{len(new_rows)}")
        print("=" * 108)
        print("SOURCE :", current_source)
        print("CONCEPT:", repr(concept))
        print("QUERY  :", query)
        print("ANSWER :", answer)
        print("PROTECTED ANCHORS:", len(anchors))

        # Each target row is still learned independently, but several safe
        # hyperparameter attempts are tried from the SAME committed source.
        # Only the best protected-safe attempt may become the next source.
        attempts = [
            # epochs, lr_scale, replay_weight, new_weight
            (120, 0.60, 2.0, 4.0),
            (160, 0.80, 2.0, 5.0),
            (200, 1.00, 2.0, 6.0),
            (240, 1.00, 1.0, 6.0),
            (240, 1.20, 1.0, 8.0),
        ]

        best = None
        anchor_queries = {
            str(r.get("query", ""))
            for r in anchors
        }

        for attempt_index, (epochs, lr_scale, replay_weight, new_weight) in enumerate(attempts, 1):
            attempt_candidate = output.with_name(
                f"{output.stem}.step{index}.try{attempt_index}{output.suffix}"
            )
            train_json = output.with_name(
                f"{output.stem}.step{index}.try{attempt_index}.train.json"
            )
            runtime_json = output.with_name(
                f"{output.stem}.step{index}.try{attempt_index}.runtime.json"
            )

            print()
            print(
                f"TRY {attempt_index}/{len(attempts)}: "
                f"epochs={epochs} lr_scale={lr_scale:.2f} "
                f"replay={replay_weight:.1f} new={new_weight:.1f}"
            )

            train_cmd = [
                sys.executable,
                "semantic_guided_answer_finetune_v097.py",
                "--model", str(current_source),
                "--tokenizer", args.tokenizer,
                "--dataset", str(step_dataset),
                "--benchmark", args.benchmark,
                "--output", str(attempt_candidate),
                "--epochs", str(epochs),
                "--learning-rate", str(args.learning_rate * lr_scale),
                "--lm-head-lr", str(args.lm_head_lr * lr_scale),
                "--preserve-weight", str(args.preserve_weight),
                "--train-blocks", str(args.train_blocks),
                "--protected-distill-weight", str(replay_weight),
                "--new-knowledge-weight", str(new_weight),
                "--min-generation-sim", "0.0",
                "--result-json", str(train_json),
                "--prefer-final-state",
                "--concept-balanced",
            ]
            if args.allow_cpu:
                train_cmd.append("--allow-cpu")

            if run(train_cmd) != 0:
                print("TRY RESULT: TRAINING FAILURE")
                continue

            _, metrics = runtime_result(
                source=current_source,
                candidate=attempt_candidate,
                full_dataset=args.full_dataset,
                benchmark=args.benchmark,
                tokenizer=args.tokenizer,
                result_json=runtime_json,
                allow_cpu=args.allow_cpu,
            )

            detail = find_detail(metrics, query)
            if detail is None:
                print("TRY RESULT: target query missing from runtime report")
                continue

            target_before = float(
                detail.get("source_canonical_similarity", 0.0)
            )
            target_after = float(
                detail.get("candidate_canonical_similarity", 0.0)
            )
            target_gain = target_after - target_before

            protected_failures = []
            max_drop = 0.0
            min_pair = 1.0
            for item in metrics.get("details", []):
                q = str(item.get("query", ""))
                if q not in anchor_queries:
                    continue
                before = float(
                    item.get("source_canonical_similarity", 0.0)
                )
                after = float(
                    item.get("candidate_canonical_similarity", 0.0)
                )
                drop = before - after
                pair = float(
                    item.get("source_candidate_similarity", 0.0)
                )
                max_drop = max(max_drop, drop)
                min_pair = min(min_pair, pair)
                passed = (
                    drop <= args.max_protected_drop
                    and pair >= 0.70
                )
                if not passed:
                    protected_failures.append({
                        "query": q,
                        "drop": drop,
                        "pair": pair,
                    })

            target_ok = (
                target_after >= args.min_target_sim
                and (
                    target_gain >= args.min_target_gain
                    or target_after >= 0.999999
                )
            )
            protected_ok = not protected_failures

            print(
                "TRY TARGET: "
                f"{target_before:.6f}->{target_after:.6f} "
                f"gain={target_gain:+.6f}"
            )
            print(
                "TRY PROTECTION: "
                f"failures={len(protected_failures)} "
                f"max_drop={max_drop:+.6f} "
                f"min_pair={min_pair:.6f}"
            )

            if protected_failures:
                for item in protected_failures:
                    print(
                        "  -",
                        item["query"],
                        f"drop={item['drop']:+.6f}",
                        f"pair={item['pair']:.6f}",
                    )

            if not (target_ok and protected_ok):
                print("TRY RESULT: REJECT")
                continue

            score = (
                target_after,
                target_gain,
                -max_drop,
                min_pair,
            )
            if best is None or score > best["score"]:
                best = {
                    "score": score,
                    "candidate": attempt_candidate,
                    "target_before": target_before,
                    "target_after": target_after,
                    "target_gain": target_gain,
                    "attempt": attempt_index,
                    "epochs": epochs,
                    "lr_scale": lr_scale,
                    "replay_weight": replay_weight,
                    "new_weight": new_weight,
                }
                print("TRY RESULT: SAFE CANDIDATE")
            else:
                print("TRY RESULT: SAFE, but not best")

        if best is not None:
            print()
            print(
                "DECISION: ACCEPT "
                f"try={best['attempt']} "
                f"canonical={best['target_before']:.6f}"
                f"->{best['target_after']:.6f} "
                f"gain={best['target_gain']:+.6f}"
            )
            current_source = Path(best["candidate"])
            accepted_row = dict(target)
            accepted_row["must_train"] = False
            accepted_row["protected"] = True
            accepted_rows.append(accepted_row)
            accepted += 1
            accepted_details.append({
                "query": query,
                "concept": concept,
                "attempt": int(best["attempt"]),
                "target_before": float(best["target_before"]),
                "target_after": float(best["target_after"]),
                "target_gain": float(best["target_gain"]),
                "candidate": str(best["candidate"]),
            })
        else:
            print()
            print("DECISION: REJECT / ROLLBACK (all retries unsafe or insufficient)")
            print("ROLLBACK SOURCE:", current_source)
            rejected += 1
            rejected_details.append({
                "query": query,
                "concept": concept,
                "reason": "all retries unsafe or insufficient",
            })

    print()
    print("=" * 108)
    print(" ONE-BY-ONE LEARNING SUMMARY")
    print("=" * 108)
    print("Accepted rows:", accepted)
    print("Rejected rows:", rejected)
    print("Final source :", current_source)

    # Final promotion candidate is the accumulated accepted model.
    shutil.copy2(current_source, output)

    # Full gate remains authoritative. Returning nonzero means /sleep will not promote.
    final_json = output.with_name(f"{output.stem}.final.runtime.json")
    final_code, final_metrics = runtime_result(
        source=source,
        candidate=output,
        full_dataset=args.full_dataset,
        benchmark=args.benchmark,
        tokenizer=args.tokenizer,
        result_json=final_json,
        allow_cpu=args.allow_cpu,
    )
    final_failures = int(final_metrics.get("failures", 10**9))
    final_known = int(final_metrics.get("known_failures", 10**9))
    final_new = int(final_metrics.get("new_failures", 10**9))
    final_mean = float(final_metrics.get("candidate_canonical_mean", 0.0))

    if final_code == 0:
        state = "COMPLETE"
        exit_code = 0
    elif accepted > 0 and final_known == 0:
        state = "PARTIAL"
        exit_code = 3
    else:
        state = "UNLEARNED"
        exit_code = 1

    state_json = output.with_name(f"{output.stem}.partial-state.json")
    state_json.write_text(
        json.dumps(
            {
                "version": "v0.10.34",
                "state": state,
                "source": str(source),
                "output": str(output),
                "accepted_rows": accepted,
                "rejected_rows": rejected,
                "accepted_details": accepted_details,
                "rejected_details": rejected_details,
                "runtime": {
                    "failures": final_failures,
                    "known_failures": final_known,
                    "new_failures": final_new,
                    "canonical_mean": final_mean,
                },
            },
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    print(
        "FINAL RUNTIME: "
        f"failures={final_failures} "
        f"known={final_known} "
        f"new={final_new} "
        f"canonical_mean={final_mean:.6f}"
    )
    print("INTERNALIZATION STATE:", state)
    print("Accepted rows:", accepted)
    print("Rejected rows:", rejected)
    print("State file   :", state_json)

    if state == "COMPLETE":
        print("RESULT: COMPLETE - all runtime probes passed.")
    elif state == "PARTIAL":
        print(
            "RESULT: PARTIAL - accepted rows are retained; "
            "remaining rows will be retried by a later /sleep."
        )
    else:
        print("RESULT: UNLEARNED - no safe partial progress to commit.")

    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
