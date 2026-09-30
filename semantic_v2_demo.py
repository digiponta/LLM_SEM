# semantic_v2_demo.py
#
# Minimal Semantic Data v2.0 demonstration.

from semantic import (
    SemanticRelation,
    encode_semantic_v2,
)


def build_example(model, tokenizer):
    """Build one structured semantic object.

    The caller supplies a loaded LLM_SEM model and tokenizer.
    """

    semantic = encode_semantic_v2(
        model,
        tokenizer,
        "GPUでニューラルネットワークを高速化する方法を教えて",
        concepts=["GPU", "ニューラルネットワーク"],
        purpose_text="高速化する方法を教えて",
        intent="explain",
        relations=[
            SemanticRelation(
                subject="GPU",
                predicate="accelerates",
                object="ニューラルネットワーク",
                confidence=0.95,
            )
        ],
        context_text="CUDAを利用する計算環境",
        context_attributes={
            "runtime": "CUDA",
            "domain": "computer",
        },
        confidence=0.90,
        uncertainty=0.10,
        metadata={
            "producer": "LLM_SEM",
            "schema": "Semantic Data v2.0",
        },
    )

    print("schema_version :", semantic.schema_version)
    print("dimension      :", semantic.dimension)
    print("concepts       :", [c.name for c in semantic.concept_vectors])
    print("purpose        :", semantic.purpose.text)
    print("intent         :", semantic.purpose.intent)
    print(
        "relations      :",
        [
            (r.subject, r.predicate, r.object)
            for r in semantic.relations
        ],
    )
    print("context        :", semantic.context)
    print("confidence     :", semantic.confidence)
    print("uncertainty    :", semantic.uncertainty)

    # Existing SemanticData-v1 routers can continue to consume the global
    # vector through the compatibility adapter.
    legacy = semantic.as_legacy()
    print("legacy vector  :", len(legacy.vector))

    return semantic
