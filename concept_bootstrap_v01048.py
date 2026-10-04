from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(
        description="LLM_SEM v0.10.48 Anchor-First Concept Bootstrap"
    )
    p.add_argument("--source", required=True)
    p.add_argument("--incremental-dataset", required=True)
    p.add_argument("--full-dataset", required=True)
    p.add_argument("--benchmark", default="my_benchmark.csv")
    p.add_argument("--tokenizer", default="model/tokenizer.json")
    p.add_argument("--output", required=True)
    p.add_argument("--anchor-epochs", type=int, default=120)
    p.add_argument("--full-epochs", type=int, default=180)
    p.add_argument("--learning-rate", type=float, default=5e-6)
    p.add_argument("--lm-head-lr", type=float, default=2.5e-5)
    p.add_argument("--preserve-weight", type=float, default=5.0)
    p.add_argument("--protected-distill-weight", type=float, default=4.0)
    p.add_argument("--anchor-weight", type=float, default=8.0)
    p.add_argument("--full-weight", type=float, default=6.0)
    p.add_argument("--train-blocks", type=int, default=1)
    p.add_argument("--min-group-rows", type=int, default=2)
    p.add_argument("--min-anchor-gain", type=float, default=0.02)
    p.add_argument("--min-full-gain", type=float, default=0.05)
    p.add_argument("--anchor-max-chars", type=int, default=48)
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


def make_short_anchor(concept: str, answer: str, max_chars: int = 48) -> str:
    text = str(answer).strip()
    if len(text) <= max_chars:
        return text

    terminal = "。" if text.endswith("。") else ""
    body = text[:-1] if terminal else text
    parts = [x.strip() for x in re.split(r"[、，,]", body) if x.strip()]
    if not parts:
        return text[:max_chars].rstrip("、，,") + terminal

    first = parts[0]
    if concept and not first.startswith(concept):
        first = f"{concept}は"

    chosen = [first]
    used = len(first) + len(terminal)

    # Prefer the semantic predicate/end of the definition. Build backwards
    # from the final clauses so the anchor remains a grammatical definition.
    suffix = []
    for part in reversed(parts[1:]):
        extra = len(part) + 1
        if used + sum(len(x) + 1 for x in suffix) + extra > max_chars:
            continue
        suffix.insert(0, part)

    if suffix:
        chosen.extend(suffix)

    anchor = "、".join(chosen)
    if terminal and not anchor.endswith(terminal):
        anchor += terminal

    if len(anchor) > max_chars:
        anchor = anchor[: max(1, max_chars - len(terminal))].rstrip("、，,")
        anchor += terminal

    return anchor


def save_dataset(
    path: Path,
    targets: list[dict],
    protected: list[dict],
    *,
    anchor_answer: str | None = None,
    mode: str,
):
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
        if anchor_answer is not None:
            row["answer"] = anchor_answer
            row["sleep_source"] = "concept_anchor"
        rows.append(row)

    path.write_text(
        json.dumps(
            {
                "version": "v0.10.48",
                "mode": mode,
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


def train(
    args,
    source: Path,
    dataset: Path,
    output: Path,
    result_json: Path,
    *,
    epochs: int,
    new_weight: float,
) -> bool:
    cmd = [
        sys.executable,
        "semantic_guided_answer_finetune_v097.py",
        "--model", str(source),
        "--tokenizer", args.tokenizer,
        "--dataset", str(dataset),
        "--benchmark", args.benchmark,
        "--output", str(output),
        "--epochs", str(epochs),
        "--learning-rate", str(args.learning_rate),
        "--lm-head-lr", str(args.lm_head_lr),
        "--preserve-weight", str(args.preserve_weight),
        "--protected-distill-weight",
        str(args.protected_distill_weight),
        "--new-knowledge-weight", str(new_weight),
        "--train-blocks", str(args.train_blocks),
        "--min-generation-sim", "0.0",
        "--result-json", str(result_json),
        "--prefer-final-state",
        "--concept-balanced",
    ]
    if args.allow_cpu:
        cmd.append("--allow-cpu")
    return run(cmd) == 0


def runtime_metrics(args, source: Path, candidate: Path, result_json: Path):
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
    run(cmd)
    if not result_json.exists():
        raise RuntimeError(f"runtime result missing: {result_json}")
    return json.loads(result_json.read_text(encoding="utf-8"))


def concept_mean(metrics: dict, queries: set[str], field: str) -> float:
    vals = [
        float(x.get(field, 0.0))
        for x in metrics.get("details", [])
        if str(x.get("query", "")) in queries
    ]
    return sum(vals) / len(vals) if vals else 0.0


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

    print("=" * 108)
    print(" LLM_SEM v0.10.48 Anchor-First Concept Bootstrap")
    print("=" * 108)
    print("Source model      :", source)
    print("Eligible concepts :", len(eligible))

    if not eligible:
        shutil.copy2(source, output)
        print("RESULT: SKIP")
        raise SystemExit(2)

    current = source
    accepted = 0

    for index, (concept, targets) in enumerate(eligible.items(), 1):
        canonical = str(targets[0].get("answer", "")).strip()
        anchor = make_short_anchor(
            concept,
            canonical,
            max_chars=max(16, int(args.anchor_max_chars)),
        )
        queries = {str(x.get("query", "")) for x in targets}

        print()
        print("-" * 108)
        print(f"CONCEPT {index}/{len(eligible)}: {concept!r}")
        print("Canonical chars :", len(canonical))
        print("Anchor chars    :", len(anchor))
        print("Anchor          :", anchor)

        anchor_dataset = output.with_name(
            f"{output.stem}.anchor{index}.json"
        )
        anchor_candidate = output.with_name(
            f"{output.stem}.anchor{index}{output.suffix}"
        )
        anchor_train_json = output.with_name(
            f"{output.stem}.anchor{index}.train.json"
        )
        anchor_runtime_json = output.with_name(
            f"{output.stem}.anchor{index}.runtime.json"
        )

        save_dataset(
            anchor_dataset,
            targets,
            protected,
            anchor_answer=anchor,
            mode="concept-anchor",
        )

        if not train(
            args,
            current,
            anchor_dataset,
            anchor_candidate,
            anchor_train_json,
            epochs=args.anchor_epochs,
            new_weight=args.anchor_weight,
        ):
            print("ANCHOR RESULT: TRAINING FAILURE")
            continue

        anchor_metrics = runtime_metrics(
            args,
            current,
            anchor_candidate,
            anchor_runtime_json,
        )
        # Runtime validator still compares against the full canonical target.
        # Even a small positive movement proves that the new concept identity
        # reached generation and is therefore a useful bootstrap source.
        anchor_before = concept_mean(
            anchor_metrics,
            queries,
            "source_canonical_similarity",
        )
        anchor_after = concept_mean(
            anchor_metrics,
            queries,
            "candidate_canonical_similarity",
        )
        anchor_gain = anchor_after - anchor_before
        anchor_safe = int(anchor_metrics.get("known_failures", 10**9)) == 0

        print(
            "ANCHOR RUNTIME: "
            f"{anchor_before:.6f}->{anchor_after:.6f} "
            f"gain={anchor_gain:+.6f} "
            f"known_failures={anchor_metrics.get('known_failures')}"
        )

        if not (anchor_safe and anchor_gain >= args.min_anchor_gain):
            print("ANCHOR RESULT: REJECT")
            continue

        print("ANCHOR RESULT: ACCEPT")

        full_dataset = output.with_name(
            f"{output.stem}.full{index}.json"
        )
        full_candidate = output.with_name(
            f"{output.stem}.full{index}{output.suffix}"
        )
        full_train_json = output.with_name(
            f"{output.stem}.full{index}.train.json"
        )
        full_runtime_json = output.with_name(
            f"{output.stem}.full{index}.runtime.json"
        )
        save_dataset(
            full_dataset,
            targets,
            protected,
            anchor_answer=None,
            mode="concept-full",
        )

        if not train(
            args,
            anchor_candidate,
            full_dataset,
            full_candidate,
            full_train_json,
            epochs=args.full_epochs,
            new_weight=args.full_weight,
        ):
            print("FULL RESULT: TRAINING FAILURE; keep anchor candidate.")
            current = anchor_candidate
            accepted += 1
            continue

        full_metrics = runtime_metrics(
            args,
            current,
            full_candidate,
            full_runtime_json,
        )
        full_before = concept_mean(
            full_metrics,
            queries,
            "source_canonical_similarity",
        )
        full_after = concept_mean(
            full_metrics,
            queries,
            "candidate_canonical_similarity",
        )
        full_gain = full_after - full_before
        full_safe = int(full_metrics.get("known_failures", 10**9)) == 0

        print(
            "FULL RUNTIME: "
            f"{full_before:.6f}->{full_after:.6f} "
            f"gain={full_gain:+.6f} "
            f"known_failures={full_metrics.get('known_failures')}"
        )

        if full_safe and full_gain >= args.min_full_gain:
            current = full_candidate
            print("FULL RESULT: ACCEPT")
        else:
            current = anchor_candidate
            print("FULL RESULT: REJECT; keep accepted anchor.")

        accepted += 1

    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(current, output)

    print()
    print("=" * 108)
    print("Anchor-First Concept Bootstrap summary")
    print("=" * 108)
    print("Accepted concepts :", accepted)
    print("Selected model    :", output)

    if accepted:
        print("RESULT: PROGRESS")
        raise SystemExit(0)

    print("RESULT: NO_PROGRESS")
    raise SystemExit(2)


if __name__ == "__main__":
    main()
