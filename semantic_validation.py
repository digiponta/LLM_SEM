# semantic_validation.py
#
# Independent holdout validation for LLM_SEM semantic routing.
#
# Policy thresholds are selected only from development data:
#   my_benchmark.csv + unknown_benchmark.csv
#
# Final metrics are then measured on independent holdout data:
#   holdout_benchmark.csv + unknown_holdout.csv
#
# Holdout data are never used for threshold optimization.

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import torch

from model import LanguageModel
from semantic_eval import LabeledSentence, load_benchmark
from semantic_radius import (
    DEFAULT_RADIUS_QUANTILE,
    DEFAULT_RADIUS_SCALE,
    classify_with_class_radius,
    fit_class_radii,
)
from semantic_router import (
    DEFAULT_ALPHA,
    DEFAULT_BALANCED_MIN_KNOWN_RECALL,
    DEFAULT_BENCHMARK,
    DEFAULT_MODEL,
    DEFAULT_TOKENIZER,
    DEFAULT_UNKNOWN_BENCHMARK,
    SemanticRouter,
    _unknown_decision,
    get_policy_thresholds,
)
from tokenizer import Tokenizer


DEFAULT_KNOWN_HOLDOUT = "holdout_benchmark.csv"
DEFAULT_UNKNOWN_HOLDOUT = "unknown_holdout.csv"
DEFAULT_OUTPUT = "semantic_validation_results.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate LLM_SEM routing on independent holdout benchmarks "
            "without re-optimizing thresholds on the holdout set."
        )
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--benchmark", default=DEFAULT_BENCHMARK)
    parser.add_argument(
        "--unknown-benchmark",
        default=DEFAULT_UNKNOWN_BENCHMARK,
    )
    parser.add_argument(
        "--known-holdout",
        default=DEFAULT_KNOWN_HOLDOUT,
    )
    parser.add_argument(
        "--unknown-holdout",
        default=DEFAULT_UNKNOWN_HOLDOUT,
    )
    parser.add_argument(
        "--detector",
        choices=["global", "class-radius"],
        default="class-radius",
        help="Unknown detector used for final holdout validation.",
    )
    parser.add_argument(
        "--policy",
        choices=["known-first", "balanced", "discovery-first"],
        default="balanced",
        help="Global-threshold policy when --detector global is used.",
    )
    parser.add_argument(
        "--radius-quantile",
        type=float,
        default=DEFAULT_RADIUS_QUANTILE,
    )
    parser.add_argument(
        "--radius-scale",
        type=float,
        default=DEFAULT_RADIUS_SCALE,
    )
    parser.add_argument(
        "--balanced-min-known-recall",
        type=float,
        default=DEFAULT_BALANCED_MIN_KNOWN_RECALL,
    )
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    return parser.parse_args()


def validate(
    router: SemanticRouter,
    development_known: List[LabeledSentence],
    development_unknown: List[LabeledSentence],
    known_holdout: List[LabeledSentence],
    unknown_holdout: List[LabeledSentence],
    detector: str,
    policy: str,
    balanced_min_known_recall: float,
    radius_quantile: float,
    radius_scale: float,
    output_filename: str,
) -> None:
    policy_metrics = None
    similarity_threshold = None
    margin_threshold = None
    class_radii = None

    if detector == "global":
        policy_metrics = get_policy_thresholds(
            router,
            development_known,
            development_unknown,
            policy,
            balanced_min_known_recall=balanced_min_known_recall,
        )
        similarity_threshold = policy_metrics["similarity_threshold"]
        margin_threshold = policy_metrics["margin_threshold"]
    else:
        class_radii = fit_class_radii(
            router,
            development_known,
            quantile=radius_quantile,
            scale=radius_scale,
        )

    # Fit known centroids using development known data only.
    router.fit(development_known)

    rows: List[Dict[str, object]] = []

    known_total = len(known_holdout)
    known_base_correct = 0
    known_routed_correctly = 0
    known_accepted = 0
    false_unknown = 0

    known_class_total: Dict[str, int] = defaultdict(int)
    known_class_correct: Dict[str, int] = defaultdict(int)

    for sample in known_holdout:
        ranked = router.route(sample.text)
        top1 = ranked[0]
        top2 = ranked[1]
        margin = top1.similarity - top2.similarity
        if detector == "global":
            rejected = _unknown_decision(
                top1.similarity,
                margin,
                float(similarity_threshold),
                float(margin_threshold),
            )
            radius = None
        else:
            rejected, radius = classify_with_class_radius(
                ranked,
                class_radii,
            )

        base_correct = top1.label == sample.label
        if base_correct:
            known_base_correct += 1

        known_class_total[sample.label] += 1

        if rejected:
            false_unknown += 1
            predicted = "unknown"
        else:
            known_accepted += 1
            predicted = top1.label
            if base_correct:
                known_routed_correctly += 1
                known_class_correct[sample.label] += 1

        rows.append(
            {
                "source": "known_holdout",
                "expected": sample.label,
                "predicted": predicted,
                "top1_label": top1.label,
                "top1_similarity": top1.similarity,
                "top2_similarity": top2.similarity,
                "top1_top2_margin": margin,
                "unknown": rejected,
                "class_radius": radius,
                "correct": predicted == sample.label,
                "text": sample.text,
            }
        )

    unknown_total = len(unknown_holdout)
    unknown_detected = 0
    false_known = 0

    unknown_class_total: Dict[str, int] = defaultdict(int)
    unknown_class_detected: Dict[str, int] = defaultdict(int)

    for sample in unknown_holdout:
        ranked = router.route(sample.text)
        top1 = ranked[0]
        top2 = ranked[1]
        margin = top1.similarity - top2.similarity
        if detector == "global":
            rejected = _unknown_decision(
                top1.similarity,
                margin,
                float(similarity_threshold),
                float(margin_threshold),
            )
            radius = None
        else:
            rejected, radius = classify_with_class_radius(
                ranked,
                class_radii,
            )

        unknown_class_total[sample.label] += 1

        if rejected:
            unknown_detected += 1
            unknown_class_detected[sample.label] += 1
            predicted = "unknown"
        else:
            false_known += 1
            predicted = top1.label

        rows.append(
            {
                "source": "unknown_holdout",
                "expected": sample.label,
                "predicted": predicted,
                "top1_label": top1.label,
                "top1_similarity": top1.similarity,
                "top2_similarity": top2.similarity,
                "top1_top2_margin": margin,
                "unknown": rejected,
                "class_radius": radius,
                "correct": rejected,
                "text": sample.text,
            }
        )

    base_known_accuracy = (
        known_base_correct / known_total if known_total else 0.0
    )
    known_recall = (
        known_routed_correctly / known_total if known_total else 0.0
    )
    known_accept_rate = (
        known_accepted / known_total if known_total else 0.0
    )
    false_unknown_rate = (
        false_unknown / known_total if known_total else 0.0
    )
    unknown_detection_rate = (
        unknown_detected / unknown_total if unknown_total else 0.0
    )
    false_known_rate = (
        false_known / unknown_total if unknown_total else 0.0
    )
    balanced_accuracy = (
        known_recall + unknown_detection_rate
    ) / 2.0

    output_path = Path(output_filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "source",
                "expected",
                "predicted",
                "top1_label",
                "top1_similarity",
                "top2_similarity",
                "top1_top2_margin",
                "unknown",
                "class_radius",
                "correct",
                "text",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print()
    print("============================================================")
    print(" Independent Holdout Semantic Validation")
    print("============================================================")
    print()
    print("Detector                 :", detector)
    if detector == "global":
        print("Policy                   :", policy)
        if policy == "balanced":
            print(
                "Min development Known Recall:",
                f"{balanced_min_known_recall * 100.0:.2f}%",
            )
        print(
            "Fixed similarity threshold:",
            f"{float(similarity_threshold):.6f}",
        )
        print(
            "Fixed margin threshold    :",
            f"{float(margin_threshold):.6f}",
        )
        print()
        print("Development policy metrics")
        print("--------------------------")
        print(
            "Expected Known Recall     :",
            f"{policy_metrics['known_recall'] * 100.0:.2f}%",
        )
        print(
            "Expected Unknown Detection:",
            f"{policy_metrics['unknown_detection_rate'] * 100.0:.2f}%",
        )
        print(
            "Expected Balanced Accuracy:",
            f"{policy_metrics['balanced_accuracy'] * 100.0:.2f}%",
        )
    else:
        print("Radius quantile          :", f"{radius_quantile:.2f}")
        print("Radius scale             :", f"{radius_scale:.2f}")
        print()
        print("Class semantic radii")
        print("--------------------")
        for label in sorted(class_radii):
            stats = class_radii[label]
            print(
                f"{label:<12} radius={stats.radius:.6f} "
                f"mean={stats.mean_distance:.6f} "
                f"max={stats.max_distance:.6f}"
            )
    print()
    print("Independent holdout metrics")
    print("---------------------------")
    print("Known holdout samples      :", known_total)
    print("Unknown holdout samples    :", unknown_total)
    print(
        "Known routing accuracy    :",
        f"{base_known_accuracy * 100.0:.2f}%",
    )
    print(
        "Known recall              :",
        f"{known_recall * 100.0:.2f}%",
    )
    print(
        "Known accept rate         :",
        f"{known_accept_rate * 100.0:.2f}%",
    )
    print(
        "Unknown detection rate    :",
        f"{unknown_detection_rate * 100.0:.2f}%",
    )
    print(
        "False Unknown rate        :",
        f"{false_unknown_rate * 100.0:.2f}%",
    )
    print(
        "False Known rate          :",
        f"{false_known_rate * 100.0:.2f}%",
    )
    print(
        "Balanced accuracy         :",
        f"{balanced_accuracy * 100.0:.2f}%",
    )

    print()
    print("Known category breakdown")
    print("------------------------")
    for label in sorted(known_class_total):
        total = known_class_total[label]
        correct = known_class_correct[label]
        print(
            f"{label:<14} {correct:>2}/{total:<2} "
            f"{correct / total * 100.0:>6.2f}%"
        )

    print()
    print("Unknown category breakdown")
    print("--------------------------")
    for label in sorted(unknown_class_total):
        total = unknown_class_total[label]
        detected = unknown_class_detected[label]
        print(
            f"{label:<14} {detected:>2}/{total:<2} "
            f"{detected / total * 100.0:>6.2f}%"
        )

    print()
    print("Validation CSV saved:", output_filename)


def main() -> None:
    args = parse_args()

    for filename, label in (
        (args.model, "Model checkpoint"),
        (args.tokenizer, "Tokenizer"),
        (args.benchmark, "Development known benchmark"),
        (args.unknown_benchmark, "Development unknown benchmark"),
        (args.known_holdout, "Known holdout benchmark"),
        (args.unknown_holdout, "Unknown holdout benchmark"),
    ):
        if not Path(filename).exists():
            raise FileNotFoundError(f"{label} not found: {filename}")

    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("alpha must be between 0.0 and 1.0.")
    if not 0.0 <= args.balanced_min_known_recall <= 1.0:
        raise ValueError(
            "balanced-min-known-recall must be between 0.0 and 1.0."
        )
    if not 0.0 <= args.radius_quantile <= 1.0:
        raise ValueError("radius-quantile must be between 0.0 and 1.0.")
    if args.radius_scale <= 0.0:
        raise ValueError("radius-scale must be > 0.")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    tokenizer = Tokenizer.load(args.tokenizer)
    model, checkpoint = LanguageModel.load_checkpoint(
        args.model,
        device=device,
    )

    development_known = load_benchmark(args.benchmark)
    development_unknown = load_benchmark(args.unknown_benchmark)
    known_holdout = load_benchmark(args.known_holdout)
    unknown_holdout = load_benchmark(args.unknown_holdout)

    print()
    print("LLM_SEM Independent Semantic Validation")
    print("---------------------------------------")
    print("Device              :", device)
    if device.type == "cuda":
        print("GPU                 :", torch.cuda.get_device_name(0))
    print("Model               :", args.model)
    print("Development known   :", args.benchmark)
    print("Development unknown :", args.unknown_benchmark)
    print("Known holdout       :", args.known_holdout)
    print("Unknown holdout     :", args.unknown_holdout)
    print("Pooling             : raw hybrid")
    print("Alpha               :", args.alpha)
    print("Checkpoint loss     :", checkpoint.get("loss"))

    router = SemanticRouter(
        model=model,
        tokenizer=tokenizer,
        alpha=args.alpha,
    )

    validate(
        router=router,
        development_known=development_known,
        development_unknown=development_unknown,
        known_holdout=known_holdout,
        unknown_holdout=unknown_holdout,
        detector=args.detector,
        policy=args.policy,
        balanced_min_known_recall=args.balanced_min_known_recall,
        radius_quantile=args.radius_quantile,
        radius_scale=args.radius_scale,
        output_filename=args.output,
    )


if __name__ == "__main__":
    main()
