# truth_aware_semantic_memory.py
#
# LLM_SEM v0.9.0
# Truth-Aware Semantic Memory CLI.
#
# Truth state is independent from lifecycle state.
# A record may therefore be, for example:
#   CONSOLIDATED + FALSE
# meaning that the model has internalized the proposition while also retaining
# that the proposition is known to be incorrect.

from __future__ import annotations

import argparse
from pathlib import Path

from adaptive_semantic_learning import (
    TRUTH_STATES,
    append_semantic_memory,
    load_semantic_memory_records,
    truth_notice,
    truth_status_counts,
    update_memory_truth,
)


DEFAULT_MEMORY = "data/semantic_memory.jsonl"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LLM_SEM v0.9.0 Truth-Aware Semantic Memory"
    )
    p.add_argument("--memory", default=DEFAULT_MEMORY)

    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("status")

    show = sub.add_parser("show")
    show.add_argument("text")

    teach = sub.add_parser("teach")
    teach.add_argument("label")
    teach.add_argument("text")
    teach.add_argument(
        "--truth",
        default="UNVERIFIED",
        choices=sorted(TRUTH_STATES),
    )
    teach.add_argument("--confidence", type=float, default=0.0)
    teach.add_argument("--provenance", default="manual")
    teach.add_argument("--correction", default="")

    mark = sub.add_parser("mark")
    mark.add_argument("text")
    mark.add_argument("truth", choices=sorted(TRUTH_STATES))
    mark.add_argument("--confidence", type=float)
    mark.add_argument("--provenance")
    mark.add_argument("--correction")

    return p.parse_args()


def print_record(row: dict) -> None:
    print("text             :", row.get("text"))
    print("label            :", row.get("label"))
    print("lifecycle        :", row.get("status"))
    print("truth_status     :", row.get("truth_status"))
    print("truth_confidence :", row.get("truth_confidence"))
    print("provenance       :", row.get("provenance"))
    print("correction       :", row.get("correction_target"))
    print("model_version    :", row.get("model_version"))
    print("verified         :", row.get("verified"))
    notice = truth_notice(row)
    if notice:
        print("notice           :", notice)


def main() -> None:
    args = parse_args()
    path = Path(args.memory)

    if args.command == "status":
        counts = truth_status_counts(path)
        print("Truth-Aware Semantic Memory")
        print("---------------------------")
        for state in sorted(TRUTH_STATES):
            print(f"{state:<11}: {counts.get(state, 0)}")
        return

    if args.command == "show":
        found = False
        for row in load_semantic_memory_records(path):
            if str(row.get("text", "")).strip() == args.text.strip():
                print_record(row)
                found = True
                print()
        if not found:
            raise RuntimeError("Semantic Memory record not found.")
        return

    if args.command == "teach":
        added = append_semantic_memory(
            path,
            args.label,
            args.text,
            source="truth-aware-teach",
            truth_status=args.truth,
            truth_confidence=args.confidence,
            provenance=args.provenance,
            correction_target=args.correction,
        )
        print("RESULT :", "ADDED" if added else "ALREADY_EXISTS")
        if added:
            print("truth  :", args.truth)
        return

    if args.command == "mark":
        changed = update_memory_truth(
            path,
            args.text,
            args.truth,
            truth_confidence=args.confidence,
            provenance=args.provenance,
            correction_target=args.correction,
        )
        print("RESULT :", "UPDATED" if changed else "NOT_FOUND")
        if changed:
            print("truth  :", args.truth)
        return


if __name__ == "__main__":
    main()
