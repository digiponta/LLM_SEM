#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LLM_SEM v0.18.4 NDC-specific semantic projection head.

The base LLM remains frozen. Only this small projection module is trained.

Architecture:
    base semantic vector (d_model)
        -> Linear(d_model, 128)
        -> GELU
        -> LayerNorm(128)
        -> Linear(128, 64)
        -> L2 normalize

Training uses a cosine-prototype classifier. The classifier prototypes are
trainable during projection learning, but runtime routing can continue to use
projected NDC semantic prototypes built from representative texts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class NDCProjectionHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        output_dim: int = 64,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim)

        self.net = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        return F.normalize(z, p=2, dim=-1)


class CosinePrototypeClassifier(nn.Module):
    def __init__(
        self,
        projection: NDCProjectionHead,
        num_classes: int = 10,
        temperature: float = 0.08,
    ) -> None:
        super().__init__()
        self.projection = projection
        self.num_classes = int(num_classes)
        self.temperature = float(temperature)
        self.class_prototypes = nn.Parameter(
            torch.empty(self.num_classes, projection.output_dim)
        )
        nn.init.normal_(self.class_prototypes, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.projection(x)
        prototypes = F.normalize(self.class_prototypes, p=2, dim=-1)
        logits = z @ prototypes.t()
        logits = logits / self.temperature
        return logits, z


@torch.no_grad()
def project_vector(
    head: NDCProjectionHead,
    vector: torch.Tensor,
) -> torch.Tensor:
    if vector.dim() == 1:
        vector = vector.unsqueeze(0)
        result = head(vector)[0]
    else:
        result = head(vector)
    return result


@torch.no_grad()
def build_projected_prototypes(
    head: NDCProjectionHead,
    base_vectors: Dict[str, Tuple[torch.Tensor, ...]],
) -> Dict[str, Tuple[torch.Tensor, ...]]:
    projected: Dict[str, Tuple[torch.Tensor, ...]] = {}
    for main, vectors in base_vectors.items():
        projected[main] = tuple(
            project_vector(head, vector)
            for vector in vectors
        )
    return projected


def save_projection_checkpoint(
    path: str,
    head: NDCProjectionHead,
    *,
    metadata: Dict[str, object],
) -> None:
    torch.save(
        {
            "version": "v0.18.4",
            "input_dim": head.input_dim,
            "hidden_dim": head.hidden_dim,
            "output_dim": head.output_dim,
            "state_dict": head.state_dict(),
            "metadata": metadata,
        },
        path,
    )


def load_projection_checkpoint(
    path: str,
    *,
    device: torch.device,
) -> Tuple[NDCProjectionHead, Dict[str, object]]:
    payload = torch.load(path, map_location=device)
    head = NDCProjectionHead(
        input_dim=int(payload["input_dim"]),
        hidden_dim=int(payload["hidden_dim"]),
        output_dim=int(payload["output_dim"]),
    ).to(device)
    head.load_state_dict(payload["state_dict"])
    head.eval()
    return head, dict(payload.get("metadata", {}))
