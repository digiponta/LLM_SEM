# promote_active_model_v094.py
#
# LLM_SEM v0.9.4
# Promote a validated semantic-consolidation checkpoint to active runtime model.
#
# This script intentionally requires an explicit --retention-pass marker.
# Typical flow:
#   1. run consolidated_retention_v094.py
#   2. if PASS, run this script with --retention-pass

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from model import LanguageModel


DEFAULT_CANDIDATE = "model/model-sem-consolidation-v081.pt"
DEFAULT_MANIFEST = "model/active-model.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LLM_SEM v0.9.4 active-model promotion"
    )
    p.add_argument("--candidate", default=DEFAULT_CANDIDATE)
    p.add_argument("--manifest", default=DEFAULT_MANIFEST)
    p.add_argument("--retention-pass", action="store_true")
    p.add_argument("--note", default="semantic-consolidation retention validated")
    p.add_argument("--allow-cpu", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.retention_pass:
        raise RuntimeError(
            "Promotion blocked. Run consolidated_retention_v094.py first, "
            "then pass --retention-pass only after RESULT: PASS."
        )

    candidate = Path(args.candidate)
    if not candidate.exists():
        raise FileNotFoundError(f"Candidate checkpoint not found: {candidate}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and not args.allow_cpu:
        raise RuntimeError("CUDA unavailable. Use --allow-cpu only for testing.")

    _, checkpoint = LanguageModel.load_checkpoint(str(candidate), device=device)

    manifest = {
        "schema_version": "0.9.4",
        "active_model": str(candidate).replace("\\", "/"),
        "promoted_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "checkpoint_loss": checkpoint.get("loss"),
        "promotion_reason": "consolidated_retention_pass",
        "note": args.note,
    }

    path = Path(args.manifest)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temp.replace(path)

    print("=" * 82)
    print(" LLM_SEM v0.9.4 Active Model Promotion")
    print("=" * 82)
    print("Candidate :", manifest["active_model"])
    print("Loss      :", manifest["checkpoint_loss"])
    print("Manifest  :", path)
    print("RESULT    : PROMOTED")


if __name__ == "__main__":
    main()
