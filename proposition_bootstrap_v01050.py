from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import torch

from model import LanguageModel
from runtime_answer_retention_v01015 import ratio, runtime_generate
from semantic_eval import load_benchmark
from semantic_router import SemanticRouter
from tokenizer import Tokenizer


def parse_args():
    p = argparse.ArgumentParser(
        description="LLM_SEM v0.10.56 First-Divergence Proposition Bootstrap"
    )
    p.add_argument("--source", required=True)
    p.add_argument("--incremental-dataset", required=True)
    p.add_argument("--full-dataset", required=True)
    p.add_argument("--benchmark", default="my_benchmark.csv")
    p.add_argument("--tokenizer", default="model/tokenizer.json")
    p.add_argument("--output", required=True)
    p.add_argument("--proposition-epochs", type=int, default=140)
    p.add_argument("--full-epochs", type=int, default=180)
    p.add_argument("--learning-rate", type=float, default=5e-6)
    p.add_argument("--lm-head-lr", type=float, default=2.5e-5)
    p.add_argument("--preserve-weight", type=float, default=5.0)
    p.add_argument("--protected-distill-weight", type=float, default=4.0)
    p.add_argument("--proposition-weight", type=float, default=8.0)
    p.add_argument("--full-weight", type=float, default=6.0)
    p.add_argument("--first-divergence-weight", type=float, default=2.0)
    p.add_argument("--first-divergence-margin", type=float, default=1.0)
    p.add_argument("--train-blocks", type=int, default=1)
    p.add_argument("--min-proposition-gain", type=float, default=0.05)
    p.add_argument("--min-full-gain", type=float, default=0.05)
    p.add_argument("--min-group-rows", type=int, default=2)
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


def normalize_sentence(text: str) -> str:
    out = re.sub(r"^[、，,\s]+", "", str(text).strip())
    out = re.sub(r"[、，,\s]+$", "", out)
    if not out.endswith("。"):
        out += "。"
    return out


def split_definition(concept: str, answer: str) -> list[str]:
    """Split a long Japanese definition into two independently learnable facts.

    Prefer connective boundaries such as 利用して、 so the first proposition
    can be normalized to 利用する。 and the second receives the subject again.
    """
    text = str(answer).strip()
    body = text[:-1] if text.endswith("。") else text

    patterns = [
        ("利用して、", "利用する。"),
        ("利用し、", "利用する。"),
        ("用いて、", "用いる。"),
        ("用い、", "用いる。"),
    ]
    for marker, left_ending in patterns:
        pos = body.find(marker)
        if pos > 0:
            left_raw = body[:pos].rstrip("、，, ")
            right_raw = body[pos + len(marker):].lstrip("、，, ")
            if left_raw and right_raw:
                left = normalize_sentence(left_raw + left_ending[:-1])
                right = normalize_sentence(f"{concept}は、{right_raw}")
                return [left, right]

    commas = [m.start() for m in re.finditer("、", body)]
    if commas:
        midpoint = len(body) / 2.0
        pos = min(commas, key=lambda x: abs(x - midpoint))
        left_raw = body[:pos].rstrip("、，, ")
        right_raw = body[pos + 1:].lstrip("、，, ")
        if left_raw and right_raw:
            return [
                normalize_sentence(left_raw),
                normalize_sentence(f"{concept}は、{right_raw}"),
            ]

    return [normalize_sentence(text)]


def tokenizer_unknown_chars(tokenizer: Tokenizer, text: str) -> list[str]:
    return sorted({
        ch for ch in str(text)
        if tokenizer.token_to_id.get(ch, tokenizer.unk_id) == tokenizer.unk_id
    })


def proposition_rows(concept: str, targets: list[dict], parts: list[str]) -> list[dict]:
    """Assign natural runtime query variants to decomposed propositions.

    Avoid synthetic queries such as '命題1/2'; they may introduce tokenizer
    OOV characters and do not match the real runtime query distribution.
    """
    rows = []
    if not targets or not parts:
        return rows

    for index, raw in enumerate(targets):
        row = dict(raw)
        part = parts[index % len(parts)]
        row["answer"] = part
        row["must_train"] = True
        row["protected"] = False
        row["sleep_source"] = "proposition_decomposition"
        row["concepts"] = [concept]
        row["proposition_index"] = (index % len(parts)) + 1
        # Character tokenizer: emphasize "concept + は、" so a new concept
        # can cross the greedy-decoding boundary before learning the tail.
        row["loss_prefix_tokens"] = min(12, max(1, len(concept) + 2))
        row["loss_prefix_weight"] = 4.0
        rows.append(row)
    return rows


def save_dataset(
    path: Path,
    targets: list[dict],
    protected: list[dict],
    *,
    mode: str,
):
    target_queries = {str(x.get("query", "")) for x in targets}
    rows = []
    for raw in protected:
        if str(raw.get("query", "")) in target_queries:
            continue
        row = dict(raw)
        row["must_train"] = False
        row["protected"] = True
        rows.append(row)
    rows.extend(dict(x) for x in targets)

    path.write_text(
        json.dumps(
            {
                "version": "v0.10.56",
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


def remove_stale(path: Path) -> None:
    if path.exists():
        path.unlink()


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
    remove_stale(result_json)
    if output.exists():
        output.unlink()

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
        "--first-divergence-weight", str(args.first_divergence_weight),
        "--first-divergence-margin", str(args.first_divergence_margin),
        "--train-blocks", str(args.train_blocks),
        "--min-generation-sim", "0.0",
        "--result-json", str(result_json),
        "--prefer-final-state",
        "--concept-balanced",
    ]
    if args.allow_cpu:
        cmd.append("--allow-cpu")
    code = run(cmd)
    if code != 0:
        return False
    if not result_json.exists():
        raise RuntimeError(f"training result missing: {result_json}")
    if not output.exists():
        raise RuntimeError(f"training checkpoint missing: {output}")
    return True


def full_runtime_metrics(
    args,
    source: Path,
    candidate: Path,
    result_json: Path,
) -> dict:
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
    remove_stale(result_json)
    run(cmd)
    if not result_json.exists():
        raise RuntimeError(f"runtime result missing: {result_json}")
    return json.loads(result_json.read_text(encoding="utf-8"))


@torch.no_grad()
def proposition_generation_mean(
    model_path: Path,
    tokenizer_path: str,
    benchmark_path: str,
    rows: list[dict],
    device: torch.device,
) -> tuple[float, list[tuple[str, float, str, str]]]:
    """Evaluate proposition targets on the exact /internal runtime prompt."""
    tokenizer = Tokenizer.load(tokenizer_path)
    model, _ = LanguageModel.load_checkpoint(model_path, device=device)
    benchmark = load_benchmark(benchmark_path)
    router = SemanticRouter(model, tokenizer, alpha=0.35)
    router.fit(benchmark)

    values = []
    details = []
    for row in rows:
        query = str(row["query"])
        expected = str(row["answer"])
        generated, _, _ = runtime_generate(
            model,
            tokenizer,
            router,
            query,
        )
        sim = ratio(generated, expected)
        values.append(sim)
        details.append(
            (
                query,
                sim,
                expected,
                generated,
            )
        )
    return (
        sum(values) / len(values) if values else 0.0,
        details,
    )


def concept_canonical_mean(metrics: dict, queries: set[str], field: str) -> float:
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
    tokenizer = Tokenizer.load(args.tokenizer)
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

    print("=" * 112)
    print(" LLM_SEM v0.10.50 Proposition-Decomposed Bootstrap")
    print("=" * 112)
    print("Source model      :", source)
    print("Eligible concepts :", len(eligible))

    if not eligible:
        shutil.copy2(source, output)
        print("RESULT: SKIP")
        raise SystemExit(2)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    current = source
    accepted = 0

    for index, (concept, targets) in enumerate(eligible.items(), 1):
        canonical = str(targets[0].get("answer", "")).strip()
        parts = split_definition(concept, canonical)

        print()
        print("-" * 112)
        print(f"CONCEPT {index}/{len(eligible)}: {concept!r}")
        print("Canonical chars :", len(canonical))
        print("Propositions    :", len(parts))
        for pidx, part in enumerate(parts, 1):
            print(f"  P{pidx}: {part}")

        if len(parts) < 2:
            print("PROPOSITION RESULT: SKIP - definition not decomposable.")
            continue

        unknown = tokenizer_unknown_chars(
            tokenizer,
            concept + canonical + "\n".join(
                str(x.get("query", "")) for x in targets
            ),
        )
        print("Tokenizer UNK   :", unknown or "(none)")
        if unknown:
            print(
                "PROPOSITION RESULT: SKIP - tokenizer cannot represent "
                + ", ".join(repr(ch) for ch in unknown)
            )
            continue

        aux_rows = proposition_rows(concept, targets, parts)
        prop_dataset = output.with_name(
            f"{output.stem}.proposition{index}.json"
        )
        prop_candidate = output.with_name(
            f"{output.stem}.proposition{index}{output.suffix}"
        )
        prop_train_json = output.with_name(
            f"{output.stem}.proposition{index}.train.json"
        )
        prop_runtime_json = output.with_name(
            f"{output.stem}.proposition{index}.runtime.json"
        )

        save_dataset(
            prop_dataset,
            aux_rows,
            protected,
            mode="proposition-bootstrap",
        )

        before_prop, _ = proposition_generation_mean(
            current,
            args.tokenizer,
            args.benchmark,
            aux_rows,
            device,
        )

        if not train(
            args,
            current,
            prop_dataset,
            prop_candidate,
            prop_train_json,
            epochs=args.proposition_epochs,
            new_weight=args.proposition_weight,
        ):
            print("PROPOSITION RESULT: TRAINING FAILURE")
            continue

        after_prop, prop_details = proposition_generation_mean(
            prop_candidate,
            args.tokenizer,
            args.benchmark,
            aux_rows,
            device,
        )
        prop_gain = after_prop - before_prop

        retention = full_runtime_metrics(
            args,
            current,
            prop_candidate,
            prop_runtime_json,
        )
        safe = int(retention.get("known_failures", 10**9)) == 0

        print(
            "PROPOSITION TARGET: "
            f"{before_prop:.6f}->{after_prop:.6f} "
            f"gain={prop_gain:+.6f} "
            f"known_failures={retention.get('known_failures')}"
        )
        for query, sim, expected, generated in prop_details:
            print(f"  query={query!r} sim={sim:.6f}")
            print("    expected :", expected)
            print("    generated:", generated)

        if not (safe and prop_gain >= args.min_proposition_gain):
            print("PROPOSITION RESULT: REJECT")
            continue

        print("PROPOSITION RESULT: ACCEPT")

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
            mode="proposition-to-full",
        )

        if not train(
            args,
            prop_candidate,
            full_dataset,
            full_candidate,
            full_train_json,
            epochs=args.full_epochs,
            new_weight=args.full_weight,
        ):
            current = prop_candidate
            accepted += 1
            print("FULL RESULT: TRAINING FAILURE; keep proposition candidate.")
            continue

        metrics = full_runtime_metrics(
            args,
            current,
            full_candidate,
            full_runtime_json,
        )
        queries = {str(x.get("query", "")) for x in targets}
        before_full = concept_canonical_mean(
            metrics,
            queries,
            "source_canonical_similarity",
        )
        after_full = concept_canonical_mean(
            metrics,
            queries,
            "candidate_canonical_similarity",
        )
        full_gain = after_full - before_full
        full_safe = int(metrics.get("known_failures", 10**9)) == 0

        print(
            "FULL TARGET: "
            f"{before_full:.6f}->{after_full:.6f} "
            f"gain={full_gain:+.6f} "
            f"known_failures={metrics.get('known_failures')}"
        )

        if full_safe and full_gain >= args.min_full_gain:
            current = full_candidate
            print("FULL RESULT: ACCEPT")
        else:
            current = prop_candidate
            print("FULL RESULT: REJECT; keep proposition candidate.")

        accepted += 1

    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(current, output)

    print()
    print("=" * 112)
    print("Proposition-Decomposed Bootstrap summary")
    print("=" * 112)
    print("Accepted concepts :", accepted)
    print("Selected model    :", output)

    if accepted:
        print("RESULT: PROGRESS")
        raise SystemExit(0)

    print("RESULT: NO_PROGRESS")
    raise SystemExit(2)


if __name__ == "__main__":
    main()
