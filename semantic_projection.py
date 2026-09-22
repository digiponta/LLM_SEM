# semantic_projection.py
#
# Trainable semantic projection head for LLM_SEM.
#
# The base language model remains frozen. This module learns a small mapping
# from the existing 64-D semantic vector into a better separated semantic space.

from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_PROJECTION = "model/semantic-projection-v0.1.pt"


class SemanticProjectionHead(nn.Module):
    def __init__(
        self,
        input_dim: int = 64,
        hidden_dim: int = 128,
        output_dim: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        if input_dim <= 0 or hidden_dim <= 0 or output_dim <= 0:
            raise ValueError("Projection dimensions must be > 0.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout = dropout

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

        self.residual = (
            nn.Identity()
            if input_dim == output_dim
            else nn.Linear(input_dim, output_dim, bias=False)
        )
        self.norm = nn.LayerNorm(output_dim)

    def config(self) -> Dict[str, object]:
        return {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
            "dropout": self.dropout,
        }

    def forward(
        self,
        semantic_vectors: torch.Tensor,
        normalize: bool = True,
    ) -> torch.Tensor:
        projected = self.norm(
            self.residual(semantic_vectors) + self.net(semantic_vectors)
        )
        if normalize:
            projected = F.normalize(projected, p=2, dim=-1)
        return projected

    def save_checkpoint(
        self,
        filename: str = DEFAULT_PROJECTION,
        **metadata,
    ) -> None:
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format": "llm-semantic-projection-v0.1",
                "config": self.config(),
                "state_dict": self.state_dict(),
                **metadata,
            },
            path,
        )

    @classmethod
    def load_checkpoint(
        cls,
        filename: str,
        device: torch.device,
    ):
        checkpoint = torch.load(filename, map_location=device)
        config = checkpoint["config"]
        model = cls(
            input_dim=int(config["input_dim"]),
            hidden_dim=int(config["hidden_dim"]),
            output_dim=int(config["output_dim"]),
            dropout=float(config.get("dropout", 0.0)),
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
        model.eval()
        return model, checkpoint


@torch.no_grad()
def project_vector(
    head: SemanticProjectionHead,
    vector: torch.Tensor,
) -> torch.Tensor:
    head.eval()
    if vector.dim() == 1:
        vector = vector.unsqueeze(0)
    return head(vector, normalize=True)
