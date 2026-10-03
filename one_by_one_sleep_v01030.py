# one_by_one_sleep_v01030.py
#
# LLM_SEM v0.10.30
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
        description="LLM_SEM v0.10.30 One-by-One Sleep Consolidation"
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
                "version": "v0.10.30",
                "mode": "one-by-one",
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
    protected_queries = {
        str(item.get("query", ""))
        for item in precheck.get("details", [])
        if bool(item.get("passed", False))
    }

    # Use full-dataset rows for protected anchors so the original metadata is kept.
    by_query = {str(row.get("query", "")): row for row in full_rows}
    protected_rows = [
        dict(by_query[q])
        for q in protected_queries
        if q in by_query
    ]

    print("=" * 108)
    print(" LLM_SEM v0.10.30 One-by-One Sleep Consolidation")
    print("=" * 108)
    print("Source model       :", source)
    print("New training rows  :", len(new_rows))
    print("Initial protected  :", len(protected_rows))
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

        train_cmd = [
            sys.executable,
            "semantic_guided_answer_finetune_v097.py",
            "--model", str(current_source),
            "--tokenizer", args.tokenizer,
            "--dataset", str(step_dataset),
            "--benchmark", args.benchmark,
            "--output", str(step_candidate),
            "--epochs", str(args.epochs),
            "--learning-rate", str(args.learning_rate),
            "--lm-head-lr", str(args.lm_head_lr),
            "--preserve-weight", str(args.preserve_weight),
            "--train-blocks", str(args.train_blocks),
            "--protected-distill-weight", str(args.replay_weight),
            "--new-knowledge-weight", str(args.new_weight),
            "--min-generation-sim", "0.0",
            "--result-json", str(train_json),
            "--prefer-final-state",
            "--concept-balanced",
        ]
        if args.allow_cpu:
            train_cmd.append("--allow-cpu")

        if run(train_cmd) != 0:
            print("DECISION: REJECT / ROLLBACK (training failure)")
            rejected += 1
            continue

        _, metrics = runtime_result(
            source=current_source,
            candidate=step_candidate,
            full_dataset=args.full_dataset,
            benchmark=args.benchmark,
            tokenizer=args.tokenizer,
            result_json=runtime_json,
            allow_cpu=args.allow_cpu,
        )

        detail = find_detail(metrics, query)
        if detail is None:
            print("DECISION: REJECT / ROLLBACK (target query missing from runtime report)")
            rejected += 1
            continue

        target_before = float(detail.get("source_canonical_similarity", 0.0))
        target_after = float(detail.get("candidate_canonical_similarity", 0.0))
        target_gain = target_after - target_before

        # Protected set is the knowledge already committed before this step.
        protected_failures = []
        for item in metrics.get("details", []):
            q = str(item.get("query", ""))
            if q not in {str(r.get("query", "")) for r in anchors}:
                continue
            before = float(item.get("source_canonical_similarity", 0.0))
            after = float(item.get("candidate_canonical_similarity", 0.0))
            drop = before - after
            pair = float(item.get("source_candidate_similarity", 0.0))
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
            "TARGET : "
            f"canonical={target_before:.6f}->{target_after:.6f} "
            f"gain={target_gain:+.6f}"
        )
        print("PROTECTED FAILURES:", len(protected_failures))
        for item in protected_failures:
            print(
                "  -",
                item["query"],
                f"drop={item['drop']:+.6f}",
                f"pair={item['pair']:.6f}",
            )

        if target_ok and protected_ok:
            print("DECISION: ACCEPT")
            current_source = step_candidate
            accepted_row = dict(target)
            accepted_row["must_train"] = False
            accepted_row["protected"] = True
            accepted_rows.append(accepted_row)
            accepted += 1
        else:
            print("DECISION: REJECT / ROLLBACK")
            print("ROLLBACK SOURCE:", current_source)
            rejected += 1

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
    print(
        "FINAL RUNTIME: "
        f"failures={final_metrics.get('failures')} "
        f"known={final_metrics.get('known_failures')} "
        f"new={final_metrics.get('new_failures')} "
        f"canonical_mean={float(final_metrics.get('candidate_canonical_mean', 0.0)):.6f}"
    )
    print("RESULT:", "PASS" if final_code == 0 else "FAIL")
    raise SystemExit(final_code)


if __name__ == "__main__":
    main()
