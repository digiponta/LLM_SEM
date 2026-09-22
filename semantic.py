# semantic.py
#
# Semantic data utilities for LLM_SEM.

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn.functional as F

from model import LanguageModel
from tokenizer import Tokenizer


@dataclass
class SemanticData:
    """Explicit semantic representation exported by LLM_SEM."""

    text: str
    vector: List[float]
    dimension: int
    token_count: int
    model_type: str = "transformer-hidden"
    pooling: str = "mean"
    confidence: Optional[float] = None


@torch.no_grad()
def encode_text(
    model: LanguageModel,
    tokenizer: Tokenizer,
    text: str,
    pooling: str = "mean",
) -> SemanticData:
    """Convert text into one contextual semantic vector."""
    if not text:
        raise ValueError("text must not be empty.")

    model.eval()
    device = next(model.parameters()).device

    token_ids = tokenizer.encode(text, add_bos=True, add_eos=True)
    tensor = torch.tensor(
        [token_ids],
        dtype=torch.long,
        device=device,
    )

    semantic = model.encode_semantic(
        tensor,
        pooling=pooling,
    )[0]
    vector = semantic.detach().cpu().tolist()

    return SemanticData(
        text=text,
        vector=vector,
        dimension=len(vector),
        token_count=len(token_ids),
        pooling=pooling,
    )


def cosine_similarity(
    left: SemanticData,
    right: SemanticData,
) -> float:
    """Return cosine similarity in [-1, 1]."""
    if left.dimension != right.dimension:
        raise ValueError(
            "Semantic vector dimensions do not match: "
            f"{left.dimension} != {right.dimension}"
        )

    a = torch.tensor(left.vector, dtype=torch.float32)
    b = torch.tensor(right.vector, dtype=torch.float32)
    return float(F.cosine_similarity(a, b, dim=0).item())


def semantic_distance(
    left: SemanticData,
    right: SemanticData,
) -> float:
    """Cosine distance: 1 - cosine similarity."""
    return 1.0 - cosine_similarity(left, right)
