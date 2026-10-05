#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LLM_SEM v0.15.7.5 Final Promotion Gate

Promotes exactly one pre-qualified decoder-crossing candidate only when:
  1) v0.15.7.4 trajectory preservation reports QUALIFIED_CANDIDATE_FOUND
  2) the selected candidate matches the qualified best candidate
  3) prompt-aware boundary result is PASS
  4) trajectory preservation result is PASS

Promotion is implemented as a local checkpoint copy with provenance metadata.
The original candidate is never overwritten.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch


DEFAULT_VALIDATION = "results/trajectory_preservation_v01574.json"
DEFAULT_OUTPUT = "model/model-sem-internalized-v01575.pt"
DEFAULT_MANIFEST = "results/promotion_manifest_v01575.json"


def parse_args():
    p = argparse.ArgumentParser(
        description="LLM_SEM v0.15.7.5 Final Promotion Gate"
    )
    p.add_argument(
        "--validation",
        default=DEFAULT_VALIDATION,
    )
    p.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
    )
    p.add_argument(
        "--manifest",
        default=DEFAULT_MANIFEST,
    )
    p.add_argument(
        "--expected-source",
        default="model/model-sem-diagnostic-v0154.pt",
    )
    p.add_argument(
        "--expected-candidate-fragment",
        default="fn5em04_head2em04",
    )
    return p.parse_args()


def normalized_path_text(value: str) -> str:
    """
    Normalize path spelling for cross-platform provenance comparison.

    Validation JSON may contain Windows backslashes while CLI defaults use
    forward slashes.  Comparing raw strings would incorrectly block a valid
    promotion.
    """
    return str(Path(value))


def main():
    args = parse_args()

    validation_path = Path(args.validation)
    if not validation_path.exists():
        raise FileNotFoundError(validation_path)

    payload = json.loads(
        validation_path.read_text(encoding="utf-8")
    )

    if payload.get("result") != "QUALIFIED_CANDIDATE_FOUND":
        raise RuntimeError(
            "Promotion blocked: validation result is not "
            "QUALIFIED_CANDIDATE_FOUND"
        )

    best = payload.get("best")
    if not isinstance(best, dict):
        raise RuntimeError("Promotion blocked: best candidate missing")

    candidate = Path(best.get("candidate", ""))
    if not candidate.exists():
        raise FileNotFoundError(candidate)

    source = payload.get("source")
    if not isinstance(source, str) or not source:
        raise RuntimeError("Promotion blocked: source checkpoint missing")

    source_normalized = normalized_path_text(source)
    expected_source_normalized = normalized_path_text(
        args.expected_source
    )

    if source_normalized != expected_source_normalized:
        raise RuntimeError(
            "Promotion blocked: unexpected source "
            f"{source!r} (normalized={source_normalized!r}, "
            f"expected={expected_source_normalized!r})"
        )

    if args.expected_candidate_fragment not in candidate.name:
        raise RuntimeError(
            "Promotion blocked: qualified candidate does not match "
            f"expected fragment {args.expected_candidate_fragment!r}"
        )

    boundary_ok = bool(best.get("boundary_ok"))
    preservation_ok = bool(best.get("preservation_ok"))
    qualified = bool(best.get("qualified"))

    if not (boundary_ok and preservation_ok and qualified):
        raise RuntimeError(
            "Promotion blocked: qualified/boundary/preservation gate failed"
        )

    boundary = best.get("boundary", {})
    if boundary.get("result") != "PROMPT_AWARE_PASS":
        raise RuntimeError(
            "Promotion blocked: prompt-aware boundary is not PASS"
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite promoted checkpoint: {output_path}"
        )

    shutil.copy2(candidate, output_path)

    checkpoint = torch.load(
        output_path,
        map_location="cpu",
    )

    checkpoint["promotion"] = {
        "version": "v0.15.7.5",
        "status": "PROMOTED",
        "source_checkpoint": source_normalized,
        "candidate_checkpoint": str(candidate),
        "validation_report": str(validation_path),
        "boundary_result": boundary.get("result"),
        "mean_prompt_js": best.get("mean_prompt_js"),
        "prompt_top1_retention": best.get("prompt_top1_retention"),
        "mean_trajectory_js": best.get("mean_trajectory_js"),
        "mean_trajectory_top1": best.get("mean_trajectory_top1"),
        "mean_source_nll_delta": best.get("mean_source_nll_delta"),
        "policy": (
            "boundary crossed and source trajectory preserved; "
            "manual runtime review completed before promotion gate"
        ),
    }

    torch.save(checkpoint, output_path)

    manifest = {
        "version": "v0.15.7.5",
        "result": "PROMOTED",
        "source_checkpoint": source_normalized,
        "candidate_checkpoint": str(candidate),
        "promoted_checkpoint": str(output_path),
        "validation_report": str(validation_path),
        "boundary_result": boundary.get("result"),
        "metrics": {
            "mean_prompt_js": best.get("mean_prompt_js"),
            "prompt_top1_retention": best.get("prompt_top1_retention"),
            "mean_trajectory_js": best.get("mean_trajectory_js"),
            "mean_trajectory_top1": best.get("mean_trajectory_top1"),
            "mean_source_nll_delta": best.get("mean_source_nll_delta"),
        },
        "promotion_conditions": {
            "qualified": qualified,
            "boundary_ok": boundary_ok,
            "preservation_ok": preservation_ok,
            "expected_source": expected_source_normalized,
            "expected_candidate_fragment": args.expected_candidate_fragment,
        },
    }

    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("=" * 100)
    print(" LLM_SEM v0.15.7.5 Final Promotion Gate")
    print("=" * 100)
    print("Validation           :", validation_path)
    print("Source               :", source_normalized)
    print("Qualified candidate  :", candidate)
    print("Boundary             :", boundary.get("result"))
    print(
        "Trajectory JS        :",
        f"{best.get('mean_trajectory_js'):.6f}",
    )
    print(
        "Trajectory top1      :",
        f"{best.get('mean_trajectory_top1'):.1%}",
    )
    print(
        "Source NLL delta     :",
        f"{best.get('mean_source_nll_delta'):+.6f}",
    )
    print("Promoted checkpoint  :", output_path)
    print("Manifest             :", manifest_path)
    print("RESULT               : PROMOTED")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
