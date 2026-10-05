"""Novelty against a comparison set: the All and Cutoff scores.

Both use the same prompt. The scope is set entirely by which papers the caller
retrieves into the comparison set — the whole corpus for All, only pre-cutoff
work for Cutoff — so the two numbers are directly comparable.

The binary In-domain transfer judgment and Mechanism non-obviousness live in
:mod:`ideascientist.evaluation.novelty_aspects`; they compare against derived
candidate mechanisms rather than a literature set.
"""

from __future__ import annotations

from typing import Any, Optional

from ideascientist.utils.llm import LLMSettings

from . import prompts
from .blocks import MAX_REFERENCE_PAPERS, comparison_set_block
from .llm_judge import clamp_score, judge_json
from .similar_papers import SimilarPaper


def score_novelty(
    report_md: str, sims: list[SimilarPaper], *, model: Optional[str] = None,
    settings: Optional[LLMSettings] = None, few_shot: str = "",
) -> dict[str, Any]:
    """Novelty against a supplied comparison set.

    The scope reported as All or Cutoff is set by which papers the caller puts
    in ``sims``, not by a different prompt.
    """
    # An empty reference list silently turns this into a reference-free
    # judgement -- the prompt still promises the model prior work -- so the
    # resulting score would be invalid rather than merely noisy. Fail loudly.
    if not sims:
        raise ValueError(
            "score_novelty called with no reference papers; this would produce a "
            "reference-free score under a prompt that claims to supply prior work. "
            "Check the retrieval."
        )
    raw = judge_json(
        prompts.novelty_system(few_shot),
        prompts.novelty_user(report_md, comparison_set_block(sims)),
        model=model,
        settings=settings,
        max_tokens=1000,
    )
    entry = raw.get("novelty") or {}
    if not isinstance(entry, dict):
        entry = {"score": entry, "reasoning": ""}
    overlap = raw.get("overlapping_prior_work") or []
    if not isinstance(overlap, list):
        overlap = [str(overlap)]
    return {
        "score": clamp_score(entry.get("score")),
        "reasoning": str(entry.get("reasoning", "")).strip(),
        "overlapping_prior_work": [str(x) for x in overlap],
        "num_reference_papers": min(len(sims), MAX_REFERENCE_PAPERS),
    }
