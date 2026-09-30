# run_semantic_relations_regression_v035.py
#
# Lightweight regression for v0.3.5 automatic relation generation.

from semantic_intent_v034 import extract_purpose_intent
from semantic_relations_v035 import generate_semantic_relations


CASES = [
    (
        "CPUとは",
        "definition",
        "CPU",
        "requests_definition_of",
    ),
    (
        "GPUについて教えて",
        "explain",
        "GPU",
        "requests_explanation_of",
    ),
    (
        "Pythonの使い方を教えて",
        "how_to",
        "Pythonの使い",
        "requests_how_to_for",
    ),
    (
        "暗号",
        "unspecified",
        "暗号",
        "mentions",
    ),
]


def main() -> None:
    print("=" * 78)
    print(" LLM_SEM v0.3.5 Semantic Relation Regression")
    print("=" * 78)

    passed = 0
    for index, (text, expected_intent, expected_concept, expected_predicate) in enumerate(
        CASES,
        1,
    ):
        extracted = extract_purpose_intent(text)
        relations = generate_semantic_relations(extracted)

        first = relations[0]
        ok = (
            extracted.intent == expected_intent
            and extracted.concept_texts[0] == expected_concept
            and first.predicate == expected_predicate
            and first.object == expected_concept
        )

        status = "PASS" if ok else "FAIL"
        print(
            f"{index:02d}. [{status}] {text!r} -> "
            f"intent={extracted.intent} concept={extracted.concept_texts[0]!r} "
            f"relation={first.subject} --{first.predicate}--> {first.object}"
        )
        passed += int(ok)

    print("-" * 78)
    print(f"Result: {passed}/{len(CASES)} passed")

    if passed != len(CASES):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
