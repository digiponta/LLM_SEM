# semantic_projection_diagnose.py
#
# Diagnose whether Semantic Dataset v2 errors are mainly caused by
# dataset composition / wording or by the learned semantic projection.
#
# Reports:
#   - overall known routing accuracy before/after projection
#   - per-label accuracy
#   - confusion matrices
#   - pattern x label accuracy
#   - difficulty x label accuracy
#   - prediction-change summary before -> after projection
#
# This tool is intended for LLM_SEM v0.2 and Semantic Dataset v2.

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F

from model import LanguageModel
from semantic_eval import LabeledSentence, load_benchmark
from semantic_projection import DEFAULT_PROJECTION, SemanticProjectionHead
from semantic_router import (
    DEFAULT_ALPHA,
    DEFAULT_BENCHMARK,
    DEFAULT_MODEL,
    DEFAULT_TOKENIZER,
    SemanticRouter,
)
from tokenizer import Tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose Semantic Dataset v2 and projection behavior."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--benchmark", default=DEFAULT_BENCHMARK)
    parser.add_argument("--known-holdout", required=True)
    parser.add_argument("--projection", default=DEFAULT_PROJECTION)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    return parser.parse_args()


def cosine_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    return 1.0 - float(F.cosine_similarity(a, b, dim=0).item())


def centroids(
    samples: Sequence[LabeledSentence],
    vectors: Sequence[torch.Tensor],
) -> Dict[str, torch.Tensor]:
    grouped: Dict[str, List[torch.Tensor]] = defaultdict(list)
    for sample, vector in zip(samples, vectors):
        grouped[sample.label].append(vector)
    return {
        label: torch.stack(items).mean(dim=0)
        for label, items in grouped.items()
    }


def predict(
    vector: torch.Tensor,
    centers: Dict[str, torch.Tensor],
) -> Tuple[str, float]:
    ranked = sorted(
        (
            (label, cosine_distance(vector, center))
            for label, center in centers.items()
        ),
        key=lambda item: item[1],
    )
    return ranked[0]


def project_all(
    head: SemanticProjectionHead,
    vectors: Sequence[torch.Tensor],
    device: torch.device,
) -> List[torch.Tensor]:
    batch = torch.stack(vectors).to(device)
    with torch.no_grad():
        projected = head(batch, normalize=True).cpu()
    return [row for row in projected]


def evaluate_predictions(
    samples: Sequence[LabeledSentence],
    vectors: Sequence[torch.Tensor],
    centers: Dict[str, torch.Tensor],
) -> List[Tuple[LabeledSentence, str, float, bool]]:
    result = []
    for sample, vector in zip(samples, vectors):
        predicted, distance = predict(vector, centers)
        result.append(
            (sample, predicted, distance, predicted == sample.label)
        )
    return result


def accuracy(rows) -> float:
    if not rows:
        return float("nan")
    return sum(int(row[3]) for row in rows) / len(rows)


def grouped_accuracy(rows, attribute: str) -> Dict[str, Tuple[int, float]]:
    grouped = defaultdict(list)
    for row in rows:
        sample = row[0]
        key = str(getattr(sample, attribute, "") or "").strip()
        if key:
            grouped[key].append(row)
    return {
        key: (len(items), accuracy(items))
        for key, items in grouped.items()
    }


def label_accuracy(rows) -> Dict[str, Tuple[int, float]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row[0].label].append(row)
    return {
        label: (len(items), accuracy(items))
        for label, items in grouped.items()
    }


def cross_accuracy(
    rows,
    outer_attribute: str,
) -> Dict[str, Dict[str, Tuple[int, float]]]:
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        sample = row[0]
        outer = str(getattr(sample, outer_attribute, "") or "").strip()
        if not outer:
            continue
        grouped[outer][sample.label].append(row)

    result = {}
    for outer, by_label in grouped.items():
        result[outer] = {
            label: (len(items), accuracy(items))
            for label, items in by_label.items()
        }
    return result


def confusion_matrix(rows, labels: Sequence[str]) -> Dict[str, Dict[str, int]]:
    matrix = {
        actual: {predicted: 0 for predicted in labels}
        for actual in labels
    }
    for sample, predicted, _, _ in rows:
        matrix[sample.label][predicted] += 1
    return matrix


def print_accuracy_table(
    title: str,
    before: Dict[str, Tuple[int, float]],
    after: Dict[str, Tuple[int, float]],
) -> None:
    keys = sorted(set(before) | set(after))
    print(title)
    print("=" * len(title))
    print()
    print(f"{'Group':<18} {'N':>4} {'Before':>10} {'After':>10} {'Delta':>10}")
    print("-" * 58)
    for key in keys:
        n = int((after.get(key) or before.get(key) or (0, 0.0))[0])
        b = before.get(key, (0, float("nan")))[1]
        a = after.get(key, (0, float("nan")))[1]
        print(
            f"{key:<18} {n:>4d} "
            f"{b * 100:>9.2f}% "
            f"{a * 100:>9.2f}% "
            f"{(a - b) * 100:>+9.2f}"
        )
    print()


def print_confusion(
    title: str,
    matrix: Dict[str, Dict[str, int]],
    labels: Sequence[str],
) -> None:
    print(title)
    print("=" * len(title))
    print()
    width = max(9, max(len(label) for label in labels) + 1)
    print("Actual\\Pred".ljust(width), end="")
    for label in labels:
        print(label[:8].rjust(9), end="")
    print()
    print("-" * (width + 9 * len(labels)))
    for actual in labels:
        print(actual.ljust(width), end="")
        for predicted in labels:
            print(str(matrix[actual][predicted]).rjust(9), end="")
        print()
    print()


def print_cross_table(
    title: str,
    before: Dict[str, Dict[str, Tuple[int, float]]],
    after: Dict[str, Dict[str, Tuple[int, float]]],
    labels: Sequence[str],
) -> None:
    groups = sorted(set(before) | set(after))
    print(title)
    print("=" * len(title))
    print()
    for group in groups:
        print(f"[{group}]")
        print(
            f"{'Label':<12} {'N':>4} {'Before':>10} "
            f"{'After':>10} {'Delta':>10}"
        )
        print("-" * 50)
        for label in labels:
            b_entry = before.get(group, {}).get(label)
            a_entry = after.get(group, {}).get(label)
            if b_entry is None and a_entry is None:
                continue
            n = int((a_entry or b_entry)[0])
            b = (b_entry or (0, float("nan")))[1]
            a = (a_entry or (0, float("nan")))[1]
            print(
                f"{label:<12} {n:>4d} "
                f"{b * 100:>9.2f}% "
                f"{a * 100:>9.2f}% "
                f"{(a - b) * 100:>+9.2f}"
            )
        print()


def print_prediction_changes(before_rows, after_rows) -> None:
    transitions = Counter()
    improved = 0
    degraded = 0
    unchanged_correct = 0
    unchanged_wrong = 0

    for before, after in zip(before_rows, after_rows):
        b_correct = before[3]
        a_correct = after[3]
        transitions[(before[1], after[1])] += 1

        if not b_correct and a_correct:
            improved += 1
        elif b_correct and not a_correct:
            degraded += 1
        elif b_correct and a_correct:
            unchanged_correct += 1
        else:
            unchanged_wrong += 1

    print("Prediction changes caused by projection")
    print("=======================================")
    print()
    print("Improved samples        :", improved)
    print("Degraded samples        :", degraded)
    print("Still correct           :", unchanged_correct)
    print("Still incorrect         :", unchanged_wrong)
    print()

    changed = [
        (pair, count)
        for pair, count in transitions.items()
        if pair[0] != pair[1]
    ]
    changed.sort(key=lambda item: (-item[1], item[0]))
    if changed:
        print("Most common label changes")
        print("-------------------------")
        for (before_label, after_label), count in changed[:15]:
            print(
                f"{before_label:<12} -> {after_label:<12} : {count}"
            )
        print()


def print_diagnosis(before_rows, after_rows) -> None:
    before_acc = accuracy(before_rows)
    after_acc = accuracy(after_rows)
    improved = sum(
        1 for b, a in zip(before_rows, after_rows)
        if not b[3] and a[3]
    )
    degraded = sum(
        1 for b, a in zip(before_rows, after_rows)
        if b[3] and not a[3]
    )
    changed_predictions = sum(
        1 for b, a in zip(before_rows, after_rows)
        if b[1] != a[1]
    )

    print("Diagnostic summary")
    print("==================")
    print()
    print("Known accuracy before :", f"{before_acc * 100:.2f}%")
    print("Known accuracy after  :", f"{after_acc * 100:.2f}%")
    print("Prediction changes    :", changed_predictions)
    print("Improved samples      :", improved)
    print("Degraded samples      :", degraded)
    print()

    if changed_predictions == 0:
        print(
            "Interpretation: projection does not change known-class decisions. "
            "Investigate the base semantic representation and dataset wording "
            "before increasing projection training epochs."
        )
    elif improved <= degraded:
        print(
            "Interpretation: projection changes decisions but does not produce "
            "a net known-class gain. Inspect class confusion and consider "
            "stronger inter-class separation in the projection objective."
        )
    else:
        print(
            "Interpretation: projection improves more samples than it degrades. "
            "Use the cross tables to identify which classes and utterance "
            "styles benefit most."
        )
    print()


def main() -> None:
    args = parse_args()

    for filename, label in (
        (args.model, "Model checkpoint"),
        (args.tokenizer, "Tokenizer"),
        (args.benchmark, "Development benchmark"),
        (args.known_holdout, "Known holdout"),
        (args.projection, "Projection checkpoint"),
    ):
        if not Path(filename).exists():
            raise FileNotFoundError(f"{label} not found: {filename}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = Tokenizer.load(args.tokenizer)
    model, _ = LanguageModel.load_checkpoint(args.model, device=device)
    model.eval()

    head, _ = SemanticProjectionHead.load_checkpoint(
        args.projection,
        device=device,
    )

    development = load_benchmark(args.benchmark)
    known = load_benchmark(args.known_holdout)

    labels = sorted({sample.label for sample in development})

    router = SemanticRouter(
        model=model,
        tokenizer=tokenizer,
        alpha=args.alpha,
    )

    print()
    print("Encoding semantic vectors...")

    development_raw = [
        router._encode_tensor(sample.text)
        for sample in development
    ]
    known_raw = [
        router._encode_tensor(sample.text)
        for sample in known
    ]

    development_projected = project_all(
        head,
        development_raw,
        device,
    )
    known_projected = project_all(
        head,
        known_raw,
        device,
    )

    raw_centers = centroids(development, development_raw)
    projected_centers = centroids(
        development,
        development_projected,
    )

    before_rows = evaluate_predictions(
        known,
        known_raw,
        raw_centers,
    )
    after_rows = evaluate_predictions(
        known,
        known_projected,
        projected_centers,
    )

    print()
    print("============================================================")
    print(" LLM_SEM Semantic Projection Diagnostic")
    print("============================================================")
    print()
    print("Device            :", device)
    if device.type == "cuda":
        print("GPU               :", torch.cuda.get_device_name(0))
    print("Development       :", args.benchmark)
    print("Known holdout     :", args.known_holdout)
    print("Samples           :", len(known))
    print("Classes           :", len(labels))
    print()

    print_accuracy_table(
        "Per-label routing accuracy",
        label_accuracy(before_rows),
        label_accuracy(after_rows),
    )

    print_confusion(
        "Confusion matrix before projection",
        confusion_matrix(before_rows, labels),
        labels,
    )
    print_confusion(
        "Confusion matrix after projection",
        confusion_matrix(after_rows, labels),
        labels,
    )

    print_cross_table(
        "Pattern x label routing accuracy",
        cross_accuracy(before_rows, "pattern"),
        cross_accuracy(after_rows, "pattern"),
        labels,
    )

    print_cross_table(
        "Difficulty x label routing accuracy",
        cross_accuracy(before_rows, "difficulty"),
        cross_accuracy(after_rows, "difficulty"),
        labels,
    )

    print_prediction_changes(before_rows, after_rows)
    print_diagnosis(before_rows, after_rows)


if __name__ == "__main__":
    main()
