# answer_aware_gate_v099.py
#
# LLM_SEM v0.9.9
# Answer-Aware Gate
#
# Promote UNKNOWN_KNOWLEDGE / GATE_REVIEW to ACCEPT_ANSWER_MEMORY only when:
# - Semantic Answer Memory resolves strongly
# - truth metadata does not explicitly block the answer
# - label evidence is not contradicted by a different exact Semantic Memory label

from __future__ import annotations

from dataclasses import dataclass

from semantic_answer_memory_v098 import (
    AnswerResolution,
    truth_allows_answer_memory,
)


@dataclass
class AnswerAwareGateDecision:
    gate: str
    promoted: bool
    reason: str


def apply_answer_aware_gate(
    gate: str,
    *,
    resolution: AnswerResolution,
    truth_record: dict | None,
    min_promote_score: float = 12.0,
) -> AnswerAwareGateDecision:
    """Return the final gate after considering Answer Memory evidence."""

    if gate not in {"UNKNOWN_KNOWLEDGE", "GATE_REVIEW"}:
        return AnswerAwareGateDecision(
            gate=gate,
            promoted=False,
            reason="base-gate-already-accepted",
        )

    if not resolution.matched or resolution.candidate is None:
        return AnswerAwareGateDecision(
            gate=gate,
            promoted=False,
            reason="answer-memory-no-match",
        )

    if resolution.score < float(min_promote_score):
        return AnswerAwareGateDecision(
            gate=gate,
            promoted=False,
            reason=(
                f"answer-memory-score {resolution.score:.2f} "
                f"< {float(min_promote_score):.2f}"
            ),
        )

    if not truth_allows_answer_memory(truth_record, resolution.candidate):
        state = (
            str(truth_record.get("truth_status", "UNVERIFIED")).upper()
            if truth_record
            else "UNKNOWN"
        )
        return AnswerAwareGateDecision(
            gate=gate,
            promoted=False,
            reason=f"truth-state-blocked:{state}",
        )

    return AnswerAwareGateDecision(
        gate="ACCEPT_ANSWER_MEMORY",
        promoted=True,
        reason=(
            f"answer-memory score={resolution.score:.2f} "
            f"reason={resolution.reason}"
        ),
    )
