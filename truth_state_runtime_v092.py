# truth_state_runtime_v092.py
#
# LLM_SEM v0.9.2
# Runtime inference for Contrastive Truth-State Representation.

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

from model import LanguageModel
from tokenizer import Tokenizer
from truth_state_contrastive_v092 import TruthProjector


DEFAULT_MODEL = "model/model-sem-consolidation-v081.pt"
DEFAULT_TOKENIZER = "model/tokenizer.json"
DEFAULT_PROJECTION = "model/truth-state-contrastive-v092.pt"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LLM_SEM v0.9.2 contrastive truth-state runtime"
    )
    p.add_argument("text")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    p.add_argument("--projection", default=DEFAULT_PROJECTION)
    p.add_argument("--allow-cpu", action="store_true")
    return p.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and not args.allow_cpu:
        raise RuntimeError("CUDA unavailable. Use --allow-cpu only for testing.")

    tokenizer = Tokenizer.load(args.tokenizer)
    model, _ = LanguageModel.load_checkpoint(args.model, device=device)
    model.eval()

    checkpoint = torch.load(args.projection, map_location=device)
    labels = list(checkpoint["labels"])

    head = TruthProjector(
        input_dim=int(checkpoint["input_dim"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
        projection_dim=int(checkpoint["projection_dim"]),
        num_classes=len(labels),
    ).to(device)
    head.load_state_dict(checkpoint["state_dict"])
    head.eval()

    ids = tokenizer.encode(args.text, add_bos=True, add_eos=True)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    vec = model.encode_semantic(
        x,
        pooling="hybrid",
        hybrid_alpha=float(checkpoint.get("alpha", 0.35)),
        normalize_hybrid=False,
    )
    vec = F.normalize(vec, dim=-1)

    z, logits = head(vec)
    probs = F.softmax(logits[0], dim=-1)
    order = torch.argsort(probs, descending=True)

    top_id = int(order[0].item())
    top = labels[top_id]

    print("=" * 82)
    print(" LLM_SEM v0.9.2 Contrastive Truth-State Runtime")
    print("=" * 82)
    print("Text            :", args.text)
    print("Prediction      :", top)
    print("Confidence      :", f"{float(probs[top_id].item()):.6f}")
    print("Truth vector dim:", z.size(-1))
    print("Ranking:")
    for rank, idx in enumerate(order.tolist(), 1):
        print(f"  {rank}. {labels[idx]:<11} {float(probs[idx].item()):.6f}")

    if top == "FALSE":
        print("NOTICE          : predicted as incorrect information")
    elif top == "CONTESTED":
        print("NOTICE          : predicted as contested information")
    elif top == "OUTDATED":
        print("NOTICE          : predicted as outdated information")
    elif top == "UNVERIFIED":
        print("NOTICE          : predicted as unverified information")
    else:
        print("NOTICE          : predicted as TRUE")


if __name__ == "__main__":
    main()
