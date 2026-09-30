# run_semantic_structural_consistency_v040.py
#
# LLM_SEM v0.4.0 Semantic Structural Consistency Evaluation
#
# Goal:
#   Test whether a proposition vector is more consistent with the independently
#   encoded vectors of its own subject / predicate / object than with simple
#   counterfactual propositions in which one structural element is replaced.
#
# Important:
#   This is a diagnostic evaluation, not a calibrated semantic probability.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from model import LanguageModel
from tokenizer import Tokenizer
from semantic import encode_text


MODEL = "model/model-gpu-v0.4.pt"
TOKENIZER = "model/tokenizer.json"


@dataclass
class Case:
    name: str
    subject: str
    predicate: str
    object: str
    subject_cf: str
    predicate_cf: str
    object_cf: str


CASES = [
    Case(
        name="gpu-fast",
        subject="GPU",
        predicate="has_property",
        object="高速",
        subject_cf="CPU",
        predicate_cf="has_predicate",
        object_cf="低速",
    ),
    Case(
        name="cpu-executes",
        subject="CPU",
        predicate="has_predicate",
        object="命令を実行する",
        subject_cf="GPU",
        predicate_cf="has_property",
        object_cf="画像を表示する",
    ),
    Case(
        name="cuda-gpu",
        subject="CUDA",
        predicate="targets",
        object="GPU",
        subject_cf="Python",
        predicate_cf="has_property",
        object_cf="weather",
    ),
    Case(
        name="python-computer",
        subject="Python",
        predicate="related_with",
        object="computer",
        subject_cf="weather",
        predicate_cf="has_property",
        object_cf="food",
    ),
]


def vec(model, tokenizer, text: str) -> torch.Tensor:
    encoded = encode_text(model, tokenizer, text)
    return torch.tensor(encoded.vector, dtype=torch.float32)


def normalized_mean(*vectors: torch.Tensor) -> torch.Tensor:
    stacked = torch.stack([F.normalize(v, dim=0) for v in vectors])
    return F.normalize(stacked.mean(dim=0), dim=0)


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(a, b, dim=0).item())


def proposition_text(subject: str, predicate: str, object_: str) -> str:
    return f"{subject} {predicate} {object_}"


def main() -> None:
    if not Path(MODEL).exists():
        raise FileNotFoundError(MODEL)
    if not Path(TOKENIZER).exists():
        raise FileNotFoundError(TOKENIZER)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer.load(TOKENIZER)
    model, checkpoint = LanguageModel.load_checkpoint(MODEL, device=device)

    print("=" * 100)
    print(" LLM_SEM v0.4.0 Semantic Structural Consistency Evaluation")
    print("=" * 100)
    print("Device          :", device)
    if device.type == "cuda":
        print("GPU             :", torch.cuda.get_device_name(0))
    print("Checkpoint loss :", checkpoint.get("loss"))
    print("Cases           :", len(CASES))
    print()

    rows = []
    passed = 0

    for case in CASES:
        subject_v = vec(model, tokenizer, case.subject)
        predicate_v = vec(model, tokenizer, case.predicate)
        object_v = vec(model, tokenizer, case.object)
        composed = normalized_mean(subject_v, predicate_v, object_v)

        positive_v = vec(
            model,
            tokenizer,
            proposition_text(case.subject, case.predicate, case.object),
        )

        cf_subject_v = vec(
            model,
            tokenizer,
            proposition_text(case.subject_cf, case.predicate, case.object),
        )
        cf_predicate_v = vec(
            model,
            tokenizer,
            proposition_text(case.subject, case.predicate_cf, case.object),
        )
        cf_object_v = vec(
            model,
            tokenizer,
            proposition_text(case.subject, case.predicate, case.object_cf),
        )

        pos = cosine(composed, positive_v)
        cf_subject = cosine(composed, cf_subject_v)
        cf_predicate = cosine(composed, cf_predicate_v)
        cf_object = cosine(composed, cf_object_v)
        cf_best = max(cf_subject, cf_predicate, cf_object)
        margin = pos - cf_best
        ok = margin > 0.0
        passed += int(ok)

        rows.append(
            (
                case.name,
                pos,
                cf_subject,
                cf_predicate,
                cf_object,
                cf_best,
                margin,
                ok,
            )
        )

    header = (
        f"{'Case':<18} {'Positive':>9} {'CF-subj':>9} {'CF-pred':>9} "
        f"{'CF-obj':>9} {'CF-best':>9} {'Margin':>9} {'Result':>8}"
    )
    print(header)
    print("-" * len(header))

    margins = []
    positives = []
    cf_bests = []

    for row in rows:
        name, pos, cfs, cfp, cfo, cfb, margin, ok = row
        positives.append(pos)
        cf_bests.append(cfb)
        margins.append(margin)
        print(
            f"{name:<18} {pos:9.6f} {cfs:9.6f} {cfp:9.6f} "
            f"{cfo:9.6f} {cfb:9.6f} {margin:+9.6f} "
            f"{'PASS' if ok else 'FAIL':>8}"
        )

    mean_positive = sum(positives) / len(positives)
    mean_cf_best = sum(cf_bests) / len(cf_bests)
    mean_margin = sum(margins) / len(margins)

    print("-" * len(header))
    print(f"Mean positive similarity : {mean_positive:.6f}")
    print(f"Mean best CF similarity  : {mean_cf_best:.6f}")
    print(f"Mean structural margin   : {mean_margin:+.6f}")
    print(f"Positive-margin cases    : {passed}/{len(CASES)}")

    # Separate diagnostics: how much each component vector aligns with the
    # positive proposition vector. This helps identify which element dominates.
    print()
    print("Component alignment diagnostics")
    print("-" * 100)
    print(
        f"{'Case':<18} {'Subject':>9} {'Predicate':>10} {'Object':>9}"
    )
    print("-" * 100)

    for case in CASES:
        positive_v = vec(
            model,
            tokenizer,
            proposition_text(case.subject, case.predicate, case.object),
        )
        s = cosine(vec(model, tokenizer, case.subject), positive_v)
        p = cosine(vec(model, tokenizer, case.predicate), positive_v)
        o = cosine(vec(model, tokenizer, case.object), positive_v)
        print(f"{case.name:<18} {s:9.6f} {p:10.6f} {o:9.6f}")

    print()
    print("Interpretation:")
    print(
        "  PASS means the independently composed subject/predicate/object vector "
        "is closer to the correct proposition than to all one-element "
        "counterfactuals."
    )
    print(
        "  A negative margin means the current frozen encoder does not preserve "
        "that structural distinction strongly enough for this case."
    )


if __name__ == "__main__":
    main()
