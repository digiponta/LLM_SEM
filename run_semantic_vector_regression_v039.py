# run_semantic_vector_regression_v039.py
#
# Regression for v0.3.9 Proposition Vector / Relation Vector.

from pathlib import Path

import torch

from model import LanguageModel
from tokenizer import Tokenizer
from semantic import SemanticRelation, encode_semantic_v2


MODEL = "model/model-gpu-v0.4.pt"
TOKENIZER = "model/tokenizer.json"


def main() -> None:
    if not Path(MODEL).exists():
        raise FileNotFoundError(MODEL)
    if not Path(TOKENIZER).exists():
        raise FileNotFoundError(TOKENIZER)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer.load(TOKENIZER)
    model, checkpoint = LanguageModel.load_checkpoint(MODEL, device=device)

    relations = [
        SemanticRelation(
            subject="query",
            predicate="requests_reason_for",
            object="proposition_1",
            confidence=0.90,
        ),
        SemanticRelation(
            subject="GPU",
            predicate="has_property",
            object="高速",
            confidence=0.92,
        ),
    ]
    proposition_specs = [
        {
            "proposition_id": "proposition_1",
            "subject": "GPU",
            "predicate": "has_property",
            "object": "高速",
            "confidence": 0.92,
        }
    ]

    semantic = encode_semantic_v2(
        model,
        tokenizer,
        "なぜGPUは高速ですか",
        concepts=["GPU", "高速"],
        purpose_text="explain_reason(GPU, 高速)",
        intent="why",
        relations=relations,
        proposition_specs=proposition_specs,
    )

    dim = semantic.dimension

    checks = [
        ("global-vector", semantic.global_vector.dimension == dim),
        (
            "concept-vectors",
            len(semantic.concept_vectors) == 2
            and all(c.vector.dimension == dim for c in semantic.concept_vectors),
        ),
        ("purpose-vector", semantic.purpose.vector.dimension == dim),
        (
            "relation-vectors",
            len(semantic.relations) == 2
            and all(
                r.vector is not None and r.vector.dimension == dim
                for r in semantic.relations
            ),
        ),
        (
            "proposition-vector",
            len(semantic.propositions) == 1
            and semantic.propositions[0].vector.dimension == dim,
        ),
    ]

    print("=" * 78)
    print(" LLM_SEM v0.3.9 Proposition / Relation Vector Regression")
    print("=" * 78)
    print("Device          :", device)
    if device.type == "cuda":
        print("GPU             :", torch.cuda.get_device_name(0))
    print("Checkpoint loss :", checkpoint.get("loss"))
    print("Vector dimension:", dim)
    print()

    passed = 0
    for name, ok in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
        passed += int(ok)

    print("-" * 78)
    print(f"Result: {passed}/{len(checks)} passed")

    if passed != len(checks):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
