#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LLM_SEM v0.18.5 Regularized near-identity NDC projection.

v0.18.4 overfit 60 training examples: training accuracy reached 100%, while
held-out routing fell below the v0.18.3 baseline. This module therefore keeps
the projected space close to the original semantic geometry.

Architecture:
    z = normalize(x + alpha * delta(x))
    delta: Linear(64, 32) -> GELU -> Linear(32, 64)

The final delta layer is zero-initialized, so the projection starts as exact
identity. Only a small residual deformation is learned.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualNDCProjection(nn.Module):
    def __init__(
        self,
        input_dim: int,
        bottleneck_dim: int = 32,
        alpha: float = 0.25,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(input_dim)
        self.bottleneck_dim = int(bottleneck_dim)
        self.alpha = float(alpha)

        self.fc1 = nn.Linear(self.input_dim, self.bottleneck_dim)
        self.fc2 = nn.Linear(self.bottleneck_dim, self.input_dim)

        nn.init.normal_(self.fc1.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.normalize(x, p=2, dim=-1)
        delta = self.fc2(F.gelu(self.fc1(base)))
        return F.normalize(base + self.alpha * delta, p=2, dim=-1)


@torch.no_grad()
def project_vector(
    head: ResidualNDCProjection,
    vector: torch.Tensor,
) -> torch.Tensor:
    if vector.dim() == 1:
        return head(vector.unsqueeze(0))[0]
    return head(vector)


@torch.no_grad()
def build_projected_prototypes(
    head: ResidualNDCProjection,
    base_vectors: Dict[str, Tuple[torch.Tensor, ...]],
) -> Dict[str, Tuple[torch.Tensor, ...]]:
    return {
        main: tuple(project_vector(head, vector) for vector in vectors)
        for main, vectors in base_vectors.items()
    }


def save_projection_checkpoint(
    path: str,
    head: ResidualNDCProjection,
    *,
    metadata: Dict[str, object],
) -> None:
    torch.save(
        {
            "version": "v0.18.5",
            "input_dim": head.input_dim,
            "bottleneck_dim": head.bottleneck_dim,
            "alpha": head.alpha,
            "state_dict": head.state_dict(),
            "metadata": metadata,
        },
        path,
    )


def load_projection_checkpoint(
    path: str,
    *,
    device: torch.device,
):
    payload = torch.load(path, map_location=device)
    head = ResidualNDCProjection(
        input_dim=int(payload["input_dim"]),
        bottleneck_dim=int(payload["bottleneck_dim"]),
        alpha=float(payload["alpha"]),
    ).to(device)
    head.load_state_dict(payload["state_dict"])
    head.eval()
    return head, dict(payload.get("metadata", {}))
