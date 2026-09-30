# run_composition_runtime_validation_v066.py
#
# LLM_SEM v0.6.6 Proposition Runtime Validation
#
# Exercise proposition extraction -> adaptive composition -> SemanticDataV2.
#
# Run:
#   python run_composition_runtime_validation_v066.py

from __future__ import annotations

from pathlib import Path
import torch

from adaptive_composition_runtime_v065 import AdaptiveCompositionRuntime, enrich_proposition_specs
from model import LanguageModel
from tokenizer import Tokenizer
from semantic_intent_v034 import extract_purpose_intent
from semantic_proposition_v036 import extract_propositions, proposition_concepts, proposition_specs, refine_purpose
from semantic_relations_v035 import generate_semantic_relations
from semantic_runtime_v2 import from_runtime_dict, runtime_summary

MODEL = "model/model-gpu-v0.4.pt"
TOKENIZER = "model/tokenizer.json"

CASES = [
    ("known-relation-balanced", "GPUは高速", "has_property", "balanced"),
    ("unseen-relation-seen-concepts", "GPUはcomputer", "has_predicate", "relation_aware"),
    ("unseen-relation-novel-object", "GPUは計算装置", "has_predicate", "balanced"),
]

def check(name, condition, detail=""):
    if not condition:
        raise AssertionError(f"{name}: {detail}" if detail else name)
    print(f"[PASS] {name}")

def main():
    for path in (MODEL, TOKENIZER):
        if not Path(path).exists():
            raise FileNotFoundError(path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer.load(TOKENIZER)
    model, checkpoint = LanguageModel.load_checkpoint(MODEL, device=device)
    model.eval()

    composition = AdaptiveCompositionRuntime(device)

    print("=" * 86)
    print(" LLM_SEM v0.6.6 Proposition Runtime Validation")
    print("=" * 86)
    print("Device               :", device)
    if device.type == "cuda":
        print("GPU                  :", torch.cuda.get_device_name(0))
    print("Checkpoint loss      :", checkpoint.get("loss"))
    print("Composition checkpoint:", composition.checkpoint_path)
    print("Composition seed     :", composition.seed)
    print()

    for name, text, expect_predicate, expect_mode in CASES:
        print(f"[CASE] {name}: {text}")

        extracted = extract_purpose_intent(text)
        propositions = extract_propositions(
            extracted.concept_texts[0] if extracted.concept_texts else text
        )
        check(f"{name}:proposition-extracted", len(propositions) == 1)

        proposition = propositions[0]
        check(f"{name}:predicate", proposition.predicate == expect_predicate, proposition.predicate)

        purpose_text = refine_purpose(extracted.intent, extracted.purpose_text, propositions)
        concepts = proposition_concepts(extracted.concept_texts, propositions)
        relations = generate_semantic_relations(
            extracted,
            propositions,
            purpose_text=purpose_text,
            concept_texts=concepts,
        )

        specs = enrich_proposition_specs(
            composition,
            model,
            tokenizer,
            proposition_specs(propositions),
        )

        attrs = specs[0]["attributes"]
        check(f"{name}:composition-mode", attrs["composition_mode"] == expect_mode, attrs["composition_mode"])
        check(
            f"{name}:vector-present",
            isinstance(specs[0].get("vector"), list) and len(specs[0]["vector"]) == composition.dimension,
        )

        runtime = {
            "gate_state": "VALIDATION",
            "selected_label": "validation",
            "selected_similarity": 1.0,
            "confidence": 1.0,
            "metadata": {"runtime": "v0.6.6-validation", "event": name},
        }

        semantic = from_runtime_dict(
            model,
            tokenizer,
            text,
            runtime,
            concept_texts=concepts,
            purpose_text=purpose_text,
            intent=extracted.intent,
            extra_relations=relations,
            proposition_specs=specs,
        )
        summary = runtime_summary(semantic)
        check(f"{name}:semantic-proposition", len(summary["propositions"]) == 1)

        prop = summary["propositions"][0]
        check(f"{name}:vector-role", prop["vector_role"] == "adaptive_proposition", str(prop["vector_role"]))
        check(f"{name}:model-type", prop["vector_model_type"] == "adaptive-composition", str(prop["vector_model_type"]))
        check(
            f"{name}:provenance-mode",
            prop["attributes"].get("composition_mode") == expect_mode,
        )

        print(
            "      "
            f"predicate={proposition.predicate} "
            f"mode={attrs['composition_mode']} "
            f"weights={attrs['composition_weights']} "
            f"novelty=(R:{attrs['relation_seen']}, "
            f"S:{attrs['subject_seen']}, O:{attrs['object_seen']})"
        )
        print()

    print("-" * 86)
    print("RESULT: PASS")
    print("Actual proposition extraction -> adaptive composition -> SemanticDataV2 path is working.")

if __name__ == "__main__":
    main()
