from __future__ import annotations

from consultation_runtime import ConsultationResult


def compute_aqt(results: list[ConsultationResult]) -> float:
    """Average Question Turns across all cases."""
    if not results:
        return 0.0
    return sum(r.question_turns for r in results) / len(results)
