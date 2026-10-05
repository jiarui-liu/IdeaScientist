"""Reference-grounded similarity: the proposal against the target paper.

Scores the proposal field by field against the results-masked record of the
held-out target paper and its closest prior work. A field marked not applicable
is excluded from the average rather than scored zero, so a proposal is not
penalized for a field that genuinely does not apply to its kind of work.

This is the reference-grounded score reported in the checkpoint-selection
table; the prompt is in :mod:`ideascientist.evaluation.prompts`.
"""

from __future__ import annotations

from typing import Any, Optional

from ideascientist.utils.llm import LLMSettings

from . import prompts
from .blocks import TARGET_RECORD_CAP, record_block, reference_block
from .llm_judge import clamp_score, judge_json
from .loaders import SourcePaper
from .similar_papers import SimilarPaper


DIMENSIONS = (
    "core_problem",
    "key_novelty",
    "problem_definition",
    "related_work",
    "method",
    "model_details",
    "comparison_to_sota",
    "proposed_evaluation",
    "falsifiable_predictions",
    "limitations",
)

_GROUPS = {
    "idea": ("core_problem", "key_novelty", "problem_definition", "related_work"),
    "method": ("method", "model_details", "comparison_to_sota"),
    "evaluation": ("proposed_evaluation", "falsifiable_predictions", "limitations"),
}

# results-masked content fields serialized into the judge prompt (in schema order).


def score_reference_grounded(
    report_obj: dict,
    source: SourcePaper,
    sims: list[SimilarPaper],
    *,
    model: Optional[str] = None,
    settings: Optional[LLMSettings] = None,
) -> dict[str, Any]:
    """Per-field rubric similarity of the proposal (results-masked object) vs. the
    reference paper(s) (results-masked objects, full-text fallback).

    Each of the content fields is scored independently (1–10) with an
    ``applicable`` flag; fields marked not-applicable (``score: null``) are
    excluded from the averages so empirical papers aren't penalized.
    """
    report_block = record_block(report_obj, TARGET_RECORD_CAP) if isinstance(report_obj, dict) else str(report_obj)
    raw = judge_json(
        prompts.REFERENCE_GROUNDED_SYSTEM,
        prompts.reference_grounded_user(report_block, reference_block(source, sims)),
        model=model,
        settings=settings,
        max_tokens=2000,
    )
    out: dict[str, Any] = {}
    applicable_scores: list[float] = []
    for dim in DIMENSIONS:
        entry = raw.get(dim) or {}
        if not isinstance(entry, dict):
            entry = {"score": entry, "reasoning": "", "applicable": True}
        applicable = entry.get("applicable", True)
        if isinstance(applicable, str):
            applicable = applicable.strip().lower() not in ("false", "no", "0", "n/a")
        sc = clamp_score(entry.get("score")) if applicable else None
        out[dim] = {
            "score": sc,
            "applicable": bool(applicable),
            "reasoning": str(entry.get("reasoning", "")).strip(),
        }
        if sc is not None:
            applicable_scores.append(sc)

    # Group averages (over applicable fields only).
    group_avgs: dict[str, Optional[float]] = {}
    for g, dims in _GROUPS.items():
        vals = [out[d]["score"] for d in dims if out[d]["score"] is not None]
        group_avgs[g] = round(sum(vals) / len(vals), 3) if vals else None
    out["group_averages"] = group_avgs
    out["average"] = (
        round(sum(applicable_scores) / len(applicable_scores), 3)
        if applicable_scores else None
    )
    out["num_applicable"] = len(applicable_scores)
    return out
