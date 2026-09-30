# run_semantic_proposition_regression_v036.py
#
# Regression tests for v0.3.6 proposition extraction.

from semantic_intent_v034 import extract_purpose_intent
from semantic_relations_v035 import generate_semantic_relations
from semantic_proposition_v036 import (
    extract_propositions,
    proposition_concepts,
    refine_purpose,
)


CASES = [
    (
        "なぜGPUは高速ですか",
        "GPU",
        "has_property",
        "高速",
        "explain_reason(GPU, 高速)",
    ),
    (
        "CPUは命令を実行する",
        "CPU",
        "has_predicate",
        "命令を実行する",
        "CPUは命令を実行する",
    ),
    (
        "GPUが高速",
        "GPU",
        "has_property",
        "高速",
        "GPUが高速",
    ),
]


def main() -> None:
    print("=" * 78)
    print(" LLM_SEM v0.3.6 Proposition Regression")
    print("=" * 78)

    passed = 0
    for index, (text, subj, pred, obj, expected_purpose) in enumerate(CASES, 1):
        extracted = extract_purpose_intent(text)
        source = extracted.concept_texts[0] if extracted.concept_texts else text
        propositions = extract_propositions(source)
        purpose = refine_purpose(
            extracted.intent,
            extracted.purpose_text,
            propositions,
        )
        concepts = proposition_concepts(
            extracted.concept_texts,
            propositions,
        )
        relations = generate_semantic_relations(
            extracted,
            propositions,
            purpose_text=purpose,
            concept_texts=concepts,
        )

        ok = bool(propositions)
        if ok:
            p = propositions[0]
            ok = (
                p.subject == subj
                and p.predicate == pred
                and p.object == obj
                and purpose == expected_purpose
            )
            if text == "なぜGPUは高速ですか":
                purpose_objects = [
                    rel.object
                    for rel in relations
                    if rel.predicate == "has_purpose"
                ]
                purpose_subjects = [
                    rel.subject
                    for rel in relations
                    if rel.predicate == "targets_concept"
                ]
                ok = (
                    ok
                    and concepts == ["GPU", "高速"]
                    and purpose_objects == ["explain_reason(GPU, 高速)"]
                    and all(
                        subject == "explain_reason(GPU, 高速)"
                        for subject in purpose_subjects
                    )
                )
            elif text == "CPUは命令を実行する":
                ok = ok and concepts == ["CPU", "命令を実行する"]
            elif text == "GPUが高速":
                ok = ok and concepts == ["GPU", "高速"]

        status = "PASS" if ok else "FAIL"
        if propositions:
            p = propositions[0]
            print(
                f"{index:02d}. [{status}] {text!r} -> "
                f"{p.subject} --{p.predicate}--> {p.object} "
                f"purpose={purpose} concepts={concepts}"
            )
        else:
            print(f"{index:02d}. [{status}] {text!r} -> no proposition")

        passed += int(ok)

    print("-" * 78)
    print(f"Result: {passed}/{len(CASES)} passed")

    if passed != len(CASES):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
