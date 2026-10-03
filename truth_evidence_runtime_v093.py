# truth_evidence_runtime_v093.py
#
# LLM_SEM v0.9.3
# Evidence-aware truth-state runtime.

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

from model import LanguageModel
from tokenizer import Tokenizer
from truth_evidence_projection_v093 import EvidenceTruthHead, feature_vector


DEFAULT_MODEL = "model/model-sem-consolidation-v081.pt"
DEFAULT_TOKENIZER = "model/tokenizer.json"
DEFAULT_PROJECTION = "model/truth-evidence-projection-v093.pt"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LLM_SEM v0.9.3 evidence-aware truth runtime"
    )
    p.add_argument("query")
    p.add_argument("reference")
    p.add_argument(
        "relation",
        choices=["SUPPORTS", "CONTRADICTS", "UNCERTAIN", "SUPERSEDED", "DISPUTED"],
    )
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    p.add_argument("--projection", default=DEFAULT_PROJECTION)
    p.add_argument("--allow-cpu", action="store_true")
    return p.parse_args()


@torch.no_grad()
def encode(model, tokenizer, text, device, alpha):
    ids = tokenizer.encode(text, add_bos=True, add_eos=True)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    v = model.encode_semantic(
        x,
        pooling="hybrid",
        hybrid_alpha=alpha,
        normalize_hybrid=False,
    )[0]
    return F.normalize(v, dim=0)


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
    relations = list(checkpoint["relation_types"])
    labels = list(checkpoint["labels"])
    relation_to_id = {name: i for i, name in enumerate(relations)}

    head = EvidenceTruthHead(
        semantic_dim=int(checkpoint["semantic_dim"]),
        relation_dim=len(relations),
        hidden_dim=int(checkpoint["hidden_dim"]),
        num_classes=len(labels),
    ).to(device)
    head.load_state_dict(checkpoint["state_dict"])
    head.eval()

    q = encode(
        model, tokenizer, args.query, device, float(checkpoint.get("alpha", 0.35))
    )
    r = encode(
        model, tokenizer, args.reference, device, float(checkpoint.get("alpha", 0.35))
    )
    feat = feature_vector(q, r, args.relation, relation_to_id).unsqueeze(0)

    probs = F.softmax(head(feat)[0], dim=-1)
    order = torch.argsort(probs, descending=True)
    top_id = int(order[0].item())
    top = labels[top_id]

    print("=" * 86)
    print(" LLM_SEM v0.9.3 Evidence-Aware Truth Runtime")
    print("=" * 86)
    print("Query      :", args.query)
    print("Reference  :", args.reference)
    print("Relation   :", args.relation)
    print("Prediction :", top)
    print("Confidence :", f"{float(probs[top_id].item()):.6f}")
    print("Ranking:")
    for rank, idx in enumerate(order.tolist(), 1):
        print(f"  {rank}. {labels[idx]:<11} {float(probs[idx].item()):.6f}")


if __name__ == "__main__":
    main()
