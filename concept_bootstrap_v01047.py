from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(
        description="LLM_SEM v0.10.47 Concept Bootstrap"
    )
    p.add_argument("--source", required=True)
    p.add_argument("--incremental-dataset", required=True)
    p.add_argument("--full-dataset", required=True)
    p.add_argument("--benchmark", default="my_benchmark.csv")
    p.add_argument("--tokenizer", default="model/tokenizer.json")
    p.add_argument("--output", required=True)
    p.add_argument("--epochs", type=int, default=180)
    p.add_argument("--learning-rate", type=float, default=5e-6)
    p.add_argument("--lm-head-lr", type=float, default=2.5e-5)
    p.add_argument("--preserve-weight", type=float, default=5.0)
    p.add_argument("--protected-distill-weight", type=float, default=4.0)
    p.add_argument("--new-knowledge-weight", type=float, default=6.0)
    p.add_argument("--train-blocks", type=int, default=1)
    p.add_argument("--min-group-rows", type=int, default=2)
    p.add_argument("--min-concept-gain", type=float, default=0.05)
    p.add_argument("--allow-cpu", action="store_true")
    return p.parse_args()


def concept_of(row: dict) -> str:
    values = [
        str(x).strip()
        for x in row.get("concepts", [])
        if str(x).strip()
    ]
    return values[0] if values else str(row.get("query", "")).strip()


def load_samples(path: Path) -> list[dict]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    return [dict(x) for x in obj.get("samples", [])]


def save_dataset(path: Path, targets: list[dict], protected: list[dict]):
    rows = []
    target_queries = {str(x.get("query", "")) for x in targets}

    for raw in protected:
        if str(raw.get("query", "")) in target_queries:
            continue
        row = dict(raw)
        row["must_train"] = False
        row["protected"] = True
        rows.append(row)

    for raw in targets:
        row = dict(raw)
        row["must_train"] = True
        row["protected"] = False
        rows.append(row)

    path.write_text(
        json.dumps(
            {
                "version": "v0.10.47",
                "mode": "concept-bootstrap",
                "samples": rows,
            },
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )


def run(cmd: list[str]) -> int:
    print(">", " ".join(cmd))
    return subprocess.run(cmd, check=False).returncode


def runtime_validate(args, source: Path, candidate: Path, result_json: Path):
    cmd = [
        sys.executable,
        "runtime_answer_retention_v01015.py",
        "--source", str(source),
        "--candidate", str(candidate),
        "--dataset", args.full_dataset,
        "--benchmark", args.benchmark,
        "--tokenizer", args.tokenizer,
        "--result-json", str(result_json),
    ]
    if args.allow_cpu:
        cmd.append("--allow-cpu")
    code = run(cmd)
    if not result_json.exists():
        raise RuntimeError(f"runtime result missing: {result_json}")
    return code, json.loads(result_json.read_text(encoding="utf-8"))


def main():
    args = parse_args()
    source = Path(args.source)
    output = Path(args.output)
    rows = load_samples(Path(args.incremental_dataset))

    mandatory = [x for x in rows if bool(x.get("must_train", False))]
    protected = [x for x in rows if bool(x.get("protected", False))]

    grouped: dict[str, list[dict]] = {}
    for row in mandatory:
        grouped.setdefault(concept_of(row), []).append(row)

    eligible = {
        concept: items
        for concept, items in grouped.items()
        if len(items) >= args.min_group_rows
    }

    print("=" * 104)
    print(" LLM_SEM v0.10.47 Concept Bootstrap")
    print("=" * 104)
    print("Source model       :", source)
    print("Mandatory rows     :", len(mandatory))
    print("Protected rows     :", len(protected))
    print("Eligible concepts  :", len(eligible))
    for concept, items in eligible.items():
        print(f"  - {concept!r}: {len(items)} rows")

    if not eligible:
        shutil.copy2(source, output)
        print("RESULT: SKIP - no multi-row unseen concept.")
        raise SystemExit(2)

    current = source
    accepted = 0

    for index, (concept, targets) in enumerate(eligible.items(), 1):
        dataset = output.with_name(
            f"{output.stem}.bootstrap{index}.json"
        )
        candidate = output.with_name(
            f"{output.stem}.bootstrap{index}{output.suffix}"
        )
        train_json = output.with_name(
            f"{output.stem}.bootstrap{index}.train.json"
        )
        runtime_json = output.with_name(
            f"{output.stem}.bootstrap{index}.runtime.json"
        )
        save_dataset(dataset, targets, protected)

        print()
        print("-" * 104)
        print(
            f"BOOTSTRAP {index}/{len(eligible)} concept={concept!r} "
            f"rows={len(targets)}"
        )

        train_cmd = [
            sys.executable,
            "semantic_guided_answer_finetune_v097.py",
            "--model", str(current),
            "--tokenizer", args.tokenizer,
            "--dataset", str(dataset),
            "--benchmark", args.benchmark,
            "--output", str(candidate),
            "--epochs", str(args.epochs),
            "--learning-rate", str(args.learning_rate),
            "--lm-head-lr", str(args.lm_head_lr),
            "--preserve-weight", str(args.preserve_weight),
            "--protected-distill-weight",
            str(args.protected_distill_weight),
            "--new-knowledge-weight",
            str(args.new_knowledge_weight),
            "--train-blocks", str(args.train_blocks),
            "--min-generation-sim", "0.0",
            "--result-json", str(train_json),
            "--prefer-final-state",
            "--concept-balanced",
        ]
        if args.allow_cpu:
            train_cmd.append("--allow-cpu")

        if run(train_cmd) != 0:
            print("BOOTSTRAP RESULT: training failed; rollback.")
            continue

        _, metrics = runtime_validate(
            args,
            current,
            candidate,
            runtime_json,
        )

        target_queries = {
            str(x.get("query", ""))
            for x in targets
        }
        details = [
            x for x in metrics.get("details", [])
            if str(x.get("query", "")) in target_queries
        ]
        before = [
            float(x.get("source_canonical_similarity", 0.0))
            for x in details
        ]
        after = [
            float(x.get("candidate_canonical_similarity", 0.0))
            for x in details
        ]
        before_mean = sum(before) / len(before) if before else 0.0
        after_mean = sum(after) / len(after) if after else 0.0
        gain = after_mean - before_mean
        known_failures = int(metrics.get("known_failures", 10**9))

        safe = known_failures == 0
        improved = gain >= args.min_concept_gain

        print(
            "BOOTSTRAP RUNTIME: "
            f"concept_mean={before_mean:.6f}->{after_mean:.6f} "
            f"gain={gain:+.6f} known_failures={known_failures}"
        )

        if safe and improved:
            current = candidate
            accepted += 1
            print("BOOTSTRAP RESULT: ACCEPT")
        else:
            print(
                "BOOTSTRAP RESULT: REJECT "
                f"(safe={safe}, improved={improved})"
            )

    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(current, output)

    print()
    print("=" * 104)
    print("Concept Bootstrap summary")
    print("=" * 104)
    print("Accepted concepts :", accepted)
    print("Selected model    :", output)

    if accepted:
        print("RESULT: PROGRESS")
        raise SystemExit(0)

    print("RESULT: NO_PROGRESS")
    raise SystemExit(2)


if __name__ == "__main__":
    main()
