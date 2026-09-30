# semantic_runtime_v2_demo.py
#
# v0.5.1 runtime integration smoke test.
#
# This file shows how values already produced by chat_v0412.py can be mapped
# into Semantic Data v2.0. It assumes model/tokenizer are loaded by the caller.

from semantic_runtime_v2 import from_runtime_dict, runtime_summary


def build_example(model, tokenizer):
    runtime = {
        "memory_label": "computer",
        "memory_similarity": 0.970126,
        "memory_margin": 0.008111,
        "local_majority": "computer",
        "local_purity": 0.667,
        "local_k": 3,
        "base_label": "science",
        "base_similarity": 0.883911,
        "gate_state": "GATE_REVIEW",
        "selected_label": "computer",
        "adaptive_enabled": True,
        "adaptive_samples": 5,
        "memory_labels": 2,
        "prototype_labels": ["computer", "science"],
    }

    semantic = from_runtime_dict(
        model,
        tokenizer,
        "量子状態の意味を教えて",
        runtime,
        concept_texts=["量子状態"],
        purpose_text="意味を教えて",
        intent="explain",
    )

    print(runtime_summary(semantic))
    return semantic
