# adaptive_composition_runtime_v065.py
#
# LLM_SEM v0.6.5 Adaptive Composition Runtime
#
# Runtime promotion of the v0.6.3 policy confirmed by v0.6.4.
#
# Frozen policy:
#   known relation -> balanced
#   unseen relation + subject/object both seen in TRAIN -> relation_aware
#   otherwise -> balanced
#
# The module returns both a composed proposition vector and provenance.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F

from semantic import encode_text
from run_relation_unseen_generalization_v043 import (
    TRAIN_CASES,
    StructuralRoleProjection,
)


ROLE_CHECKPOINT = "model/structural-role-projection-v043.pt"

BALANCED = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
RELATION_AWARE = (0.25, 0.50, 0.25)


def _training_vocabulary():
    relations = {case.predicate for case in TRAIN_CASES}
    concepts = set()
    for case in TRAIN_CASES:
        concepts.add(case.subject)
        concepts.add(case.object)
    return relations, concepts


TRAIN_RELATIONS, TRAIN_CONCEPTS = _training_vocabulary()


@dataclass(frozen=True)
class CompositionDecision:
    mode: str
    weights: tuple[float, float, float]
    relation_seen: bool
    subject_seen: bool
    object_seen: bool
    reason: str
    confidence: float


class AdaptiveCompositionRuntime:
    def __init__(self, device, checkpoint_path: str = ROLE_CHECKPOINT):
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(checkpoint_path)

        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
        dim = int(checkpoint["dimension"])
        seed = int(checkpoint["best_seed"])

        self.device = device
        self.dimension = dim
        self.seed = seed
        self.checkpoint_path = checkpoint_path

        self.role_model = StructuralRoleProjection(dim, seed=seed).to(device)
        self.role_model.load_state_dict(checkpoint["state_dict"])
        self.role_model.eval()

        for parameter in self.role_model.parameters():
            parameter.requires_grad_(False)

    def decision(self, subject: str, predicate: str, object_text: str):
        relation_seen = predicate in TRAIN_RELATIONS
        subject_seen = subject in TRAIN_CONCEPTS
        object_seen = object_text in TRAIN_CONCEPTS

        if relation_seen:
            mode = "balanced"
            weights = BALANCED
            reason = "known_relation"
            confidence = 0.90
        elif subject_seen and object_seen:
            mode = "relation_aware"
            weights = RELATION_AWARE
            reason = "unseen_relation_seen_concepts"
            confidence = 0.90
        else:
            mode = "balanced"
            weights = BALANCED
            reason = "otherwise_balanced"
            confidence = 0.80

        return CompositionDecision(
            mode=mode,
            weights=weights,
            relation_seen=relation_seen,
            subject_seen=subject_seen,
            object_seen=object_seen,
            reason=reason,
            confidence=confidence,
        )

    def _encode(self, model, tokenizer, text: str):
        item = encode_text(model, tokenizer, text)
        return torch.tensor(
            item.vector,
            dtype=torch.float32,
            device=self.device,
        )

    @torch.no_grad()
    def compose(
        self,
        model,
        tokenizer,
        subject: str,
        predicate: str,
        object_text: str,
    ):
        decision = self.decision(subject, predicate, object_text)

        s = self._encode(model, tokenizer, subject)
        p = self._encode(model, tokenizer, predicate)
        o = self._encode(model, tokenizer, object_text)

        rs = self.role_model.subject(s)
        rp = self.role_model.predicate(p)
        ro = self.role_model.object(o)

        ws, wp, wo = decision.weights
        vector = F.normalize(ws * rs + wp * rp + wo * ro, dim=0)

        return vector.detach().cpu().tolist(), decision


def enrich_proposition_specs(
    runtime: AdaptiveCompositionRuntime,
    model,
    tokenizer,
    specs: Iterable[dict[str, object]],
):
    enriched = []

    for spec in specs:
        item = dict(spec)
        subject = str(item.get("subject", ""))
        predicate = str(item.get("predicate", ""))
        object_text = str(item.get("object", ""))

        if not subject or not predicate or not object_text:
            enriched.append(item)
            continue

        vector, decision = runtime.compose(
            model,
            tokenizer,
            subject,
            predicate,
            object_text,
        )

        attrs = dict(item.get("attributes") or {})
        ws, wp, wo = decision.weights
        attrs.update(
            {
                "composition_version": "v0.6.5",
                "composition_mode": decision.mode,
                "composition_weights": (
                    f"S:{ws:.3f},P:{wp:.3f},O:{wo:.3f}"
                ),
                "composition_reason": decision.reason,
                "relation_seen": str(decision.relation_seen),
                "subject_seen": str(decision.subject_seen),
                "object_seen": str(decision.object_seen),
                "composition_confidence": f"{decision.confidence:.3f}",
                "role_checkpoint": runtime.checkpoint_path,
            }
        )

        item["attributes"] = attrs
        item["vector"] = vector
        item["vector_role"] = "adaptive_proposition"
        enriched.append(item)

    return enriched
