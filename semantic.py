# semantic.py
#
# Semantic data utilities for LLM_SEM.
#
# Semantic Data v2.0 keeps the original one-vector API for compatibility while
# adding a structured semantic object:
#
#   Global Vector + Concept Vectors[] + Purpose Vector
#   + Relations + Context + Confidence/Uncertainty
#
# Existing callers can continue using SemanticData/encode_text(). New code
# should use SemanticDataV2/encode_semantic_v2().

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F

from model import LanguageModel
from tokenizer import Tokenizer


@dataclass
class SemanticData:
    """Legacy Semantic Data v1 representation.

    Retained for backward compatibility with the current router/evaluation
    scripts. Semantic Data v2.0 exposes this representation through
    SemanticDataV2.as_legacy().
    """

    text: str
    vector: List[float]
    dimension: int
    token_count: int
    model_type: str = "transformer-hidden"
    pooling: str = "hybrid"
    hybrid_alpha: Optional[float] = None
    normalized: bool = False
    confidence: Optional[float] = None


@dataclass
class SemanticVector:
    """Named semantic vector with provenance metadata."""

    vector: List[float]
    dimension: int
    source_text: str
    role: str
    model_type: str = "transformer-hidden"
    pooling: str = "hybrid"
    confidence: Optional[float] = None

    def as_legacy(
        self,
        token_count: int = 0,
        hybrid_alpha: Optional[float] = None,
        normalized: bool = False,
    ) -> SemanticData:
        return SemanticData(
            text=self.source_text,
            vector=list(self.vector),
            dimension=self.dimension,
            token_count=token_count,
            model_type=self.model_type,
            pooling=self.pooling,
            hybrid_alpha=hybrid_alpha,
            normalized=normalized,
            confidence=self.confidence,
        )


@dataclass
class SemanticConcept:
    """Concept/entity representation inside one semantic object."""

    name: str
    vector: SemanticVector
    role: Optional[str] = None
    attributes: Dict[str, str] = field(default_factory=dict)
    confidence: Optional[float] = None


@dataclass
class SemanticRelation:
    """Explicit relation/binding between concepts with an optional vector."""

    subject: str
    predicate: str
    object: str
    confidence: Optional[float] = None
    attributes: Dict[str, str] = field(default_factory=dict)
    vector: Optional[SemanticVector] = None


@dataclass
class SemanticProposition:
    """Vectorized proposition node binding subject, predicate, and object."""

    proposition_id: str
    subject: str
    predicate: str
    object: str
    vector: SemanticVector
    confidence: Optional[float] = None
    attributes: Dict[str, str] = field(default_factory=dict)


@dataclass
class SemanticContext:
    """Structured context attached to the semantic object."""

    text: Optional[str] = None
    vector: Optional[SemanticVector] = None
    attributes: Dict[str, str] = field(default_factory=dict)


@dataclass
class SemanticPurpose:
    """Purpose/goal representation.

    Purpose has its own vector because routing by topic and routing by intended
    action are not always equivalent.
    """

    text: str
    vector: SemanticVector
    intent: Optional[str] = None
    confidence: Optional[float] = None


@dataclass
class SemanticDataV2:
    """Semantic Data v2.0 exported by LLM_SEM.

    Required:
      - global_vector
      - concept_vectors
      - purpose

    Structural:
      - relations
      - context

    Quality:
      - confidence
      - uncertainty

    The legacy router can use primary_vector() / as_legacy() without knowing
    about the additional structure.
    """

    text: str
    global_vector: SemanticVector
    concept_vectors: List[SemanticConcept]
    purpose: SemanticPurpose
    relations: List[SemanticRelation] = field(default_factory=list)
    propositions: List[SemanticProposition] = field(default_factory=list)
    context: Optional[SemanticContext] = None
    confidence: Optional[float] = None
    uncertainty: Optional[float] = None
    metadata: Dict[str, str] = field(default_factory=dict)
    schema_version: str = "2.0"

    def primary_vector(self) -> List[float]:
        """Return the legacy-compatible routing vector."""
        return self.global_vector.vector

    @property
    def dimension(self) -> int:
        return self.global_vector.dimension

    def as_legacy(self, token_count: int = 0) -> SemanticData:
        """Expose the global vector through the Semantic Data v1 interface."""
        return self.global_vector.as_legacy(token_count=token_count)


@torch.no_grad()
def encode_text(
    model: LanguageModel,
    tokenizer: Tokenizer,
    text: str,
    pooling: str = "hybrid",
    hybrid_alpha: float = 0.35,
    normalize_hybrid: bool = False,
) -> SemanticData:
    """Convert text into one contextual semantic vector (legacy v1 API)."""
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
        hybrid_alpha=hybrid_alpha,
        normalize_hybrid=normalize_hybrid,
    )[0]
    vector = semantic.detach().cpu().tolist()

    return SemanticData(
        text=text,
        vector=vector,
        dimension=len(vector),
        token_count=len(token_ids),
        pooling=pooling,
        hybrid_alpha=(hybrid_alpha if pooling == "hybrid" else None),
        normalized=(normalize_hybrid if pooling == "hybrid" else False),
    )


def _encode_named_vector(
    model: LanguageModel,
    tokenizer: Tokenizer,
    text: str,
    role: str,
    pooling: str,
    hybrid_alpha: float,
    normalize_hybrid: bool,
    confidence: Optional[float] = None,
) -> SemanticVector:
    encoded = encode_text(
        model=model,
        tokenizer=tokenizer,
        text=text,
        pooling=pooling,
        hybrid_alpha=hybrid_alpha,
        normalize_hybrid=normalize_hybrid,
    )
    return SemanticVector(
        vector=encoded.vector,
        dimension=encoded.dimension,
        source_text=text,
        role=role,
        model_type=encoded.model_type,
        pooling=encoded.pooling,
        confidence=confidence,
    )


@torch.no_grad()
def encode_semantic_v2(
    model: LanguageModel,
    tokenizer: Tokenizer,
    text: str,
    *,
    concepts: Optional[Sequence[str]] = None,
    purpose_text: Optional[str] = None,
    intent: Optional[str] = None,
    relations: Optional[Sequence[SemanticRelation]] = None,
    proposition_specs: Optional[Sequence[Dict[str, object]]] = None,
    context_text: Optional[str] = None,
    context_attributes: Optional[Dict[str, str]] = None,
    confidence: Optional[float] = None,
    uncertainty: Optional[float] = None,
    metadata: Optional[Dict[str, str]] = None,
    pooling: str = "hybrid",
    hybrid_alpha: float = 0.35,
    normalize_hybrid: bool = False,
) -> SemanticDataV2:
    """Encode text as Semantic Data v2.0.

    This function is deliberately extraction-agnostic. Callers may supply
    concepts, purpose, relations and context from a rule-based extractor,
    LLM_SEM runtime, teaching memory, or a future learned semantic parser.

    When no concepts are supplied, the full text is used as one fallback
    concept so every v2 object has at least one concept vector.

    When purpose_text is omitted, the full text is used as a conservative
    purpose fallback. This keeps the schema valid while allowing purpose
    extraction to evolve independently.
    """
    if not text:
        raise ValueError("text must not be empty.")

    global_vector = _encode_named_vector(
        model,
        tokenizer,
        text,
        role="global",
        pooling=pooling,
        hybrid_alpha=hybrid_alpha,
        normalize_hybrid=normalize_hybrid,
        confidence=confidence,
    )

    concept_texts = [c for c in (concepts or []) if c]
    if not concept_texts:
        concept_texts = [text]

    concept_vectors: List[SemanticConcept] = []
    for concept_text in concept_texts:
        concept_vectors.append(
            SemanticConcept(
                name=concept_text,
                vector=_encode_named_vector(
                    model,
                    tokenizer,
                    concept_text,
                    role="concept",
                    pooling=pooling,
                    hybrid_alpha=hybrid_alpha,
                    normalize_hybrid=normalize_hybrid,
                ),
            )
        )

    purpose_source = purpose_text or text
    purpose = SemanticPurpose(
        text=purpose_source,
        intent=intent,
        confidence=confidence,
        vector=_encode_named_vector(
            model,
            tokenizer,
            purpose_source,
            role="purpose",
            pooling=pooling,
            hybrid_alpha=hybrid_alpha,
            normalize_hybrid=normalize_hybrid,
            confidence=confidence,
        ),
    )

    relation_vectors: List[SemanticRelation] = []
    for relation in relations or []:
        relation_source = f"{relation.subject} {relation.predicate} {relation.object}"
        relation_vectors.append(
            SemanticRelation(
                subject=relation.subject,
                predicate=relation.predicate,
                object=relation.object,
                confidence=relation.confidence,
                attributes=dict(relation.attributes),
                vector=_encode_named_vector(
                    model,
                    tokenizer,
                    relation_source,
                    role="relation",
                    pooling=pooling,
                    hybrid_alpha=hybrid_alpha,
                    normalize_hybrid=normalize_hybrid,
                    confidence=relation.confidence,
                ),
            )
        )

    proposition_vectors: List[SemanticProposition] = []
    for spec in proposition_specs or []:
        proposition_id = str(spec.get("proposition_id", "proposition"))
        subject = str(spec.get("subject", ""))
        predicate = str(spec.get("predicate", ""))
        object_text = str(spec.get("object", ""))
        prop_conf = spec.get("confidence")
        attributes = dict(spec.get("attributes") or {})
        proposition_source = f"{subject} {predicate} {object_text}".strip()
        if not proposition_source:
            continue
        proposition_vectors.append(
            SemanticProposition(
                proposition_id=proposition_id,
                subject=subject,
                predicate=predicate,
                object=object_text,
                confidence=(
                    float(prop_conf)
                    if isinstance(prop_conf, (int, float))
                    else None
                ),
                attributes=attributes,
                vector=_encode_named_vector(
                    model,
                    tokenizer,
                    proposition_source,
                    role="proposition",
                    pooling=pooling,
                    hybrid_alpha=hybrid_alpha,
                    normalize_hybrid=normalize_hybrid,
                    confidence=(
                        float(prop_conf)
                        if isinstance(prop_conf, (int, float))
                        else None
                    ),
                ),
            )
        )

    context: Optional[SemanticContext] = None
    if context_text or context_attributes:
        context = SemanticContext(
            text=context_text,
            attributes=dict(context_attributes or {}),
            vector=(
                _encode_named_vector(
                    model,
                    tokenizer,
                    context_text,
                    role="context",
                    pooling=pooling,
                    hybrid_alpha=hybrid_alpha,
                    normalize_hybrid=normalize_hybrid,
                )
                if context_text
                else None
            ),
        )

    return SemanticDataV2(
        text=text,
        global_vector=global_vector,
        concept_vectors=concept_vectors,
        purpose=purpose,
        relations=relation_vectors,
        propositions=proposition_vectors,
        context=context,
        confidence=confidence,
        uncertainty=uncertainty,
        metadata=dict(metadata or {}),
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


def cosine_similarity_v2(
    left: SemanticDataV2,
    right: SemanticDataV2,
) -> float:
    """Backward-compatible similarity using each object's global vector."""
    return cosine_similarity(left.as_legacy(), right.as_legacy())


def semantic_distance_v2(
    left: SemanticDataV2,
    right: SemanticDataV2,
) -> float:
    """Backward-compatible distance using each object's global vector."""
    return 1.0 - cosine_similarity_v2(left, right)
