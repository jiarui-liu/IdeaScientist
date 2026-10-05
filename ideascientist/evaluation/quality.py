"""Proposal quality: the six reported dimensions.

Clarity, specificity, actionability, soundness, and impact are judged from the
proposal alone in a single call — they ask whether the proposal is well formed,
which needs no reference. Relevance is judged separately because it is the one
quality dimension that needs the target problem and challenge: it asks whether
the proposal addresses *this* problem rather than drifting to another.
"""

from __future__ import annotations

from typing import Any, Optional

from ideascientist.utils.llm import LLMSettings

from . import prompts
from .llm_judge import clamp_score, judge_json

DIMENSIONS = prompts.PROPOSAL_QUALITY_DIMENSIONS


def score_proposal_quality(
    report_md: str, *, model: Optional[str] = None,
    settings: Optional[LLMSettings] = None, few_shot: str = "",
) -> dict[str, Any]:
    """Return ``{dimension: {score, reasoning}, ..., "average": float}``."""
    raw = judge_json(
        prompts.proposal_quality_system(few_shot),
        prompts.proposal_quality_user(report_md),
        model=model,
        settings=settings,
        max_tokens=1200,
    )
    out: dict[str, Any] = {}
    scores: list[float] = []
    for dim in DIMENSIONS:
        entry = raw.get(dim) or {}
        if not isinstance(entry, dict):
            entry = {"score": entry, "reasoning": ""}
        sc = clamp_score(entry.get("score"))
        out[dim] = {"score": sc, "reasoning": str(entry.get("reasoning", "")).strip()}
        if sc is not None:
            scores.append(sc)
    out["average"] = round(sum(scores) / len(scores), 3) if scores else None
    return out


def score_relevance(
    problem_definition: str,
    challenge: str,
    report_md: str,
    *,
    model: Optional[str] = None,
    settings: Optional[LLMSettings] = None,
    few_shot: str = "",
) -> dict[str, Any]:
    raw = judge_json(
        prompts.relevance_system(few_shot),
        prompts.relevance_user(problem_definition, challenge, report_md),
        model=model,
        settings=settings,
        max_tokens=600,
    )
    entry = raw.get("relevance") or {}
    if not isinstance(entry, dict):
        entry = {"score": entry, "reasoning": ""}
    return {
        "score": clamp_score(entry.get("score")),
        "reasoning": str(entry.get("reasoning", "")).strip(),
    }


# Fine-grained, per-field rubric over the results-masked (results-stripped) content
# fields — the proposal and the references share this exact field set, so the
# judge matches field-by-field. Grouped only for reporting convenience.
