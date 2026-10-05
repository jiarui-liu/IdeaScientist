"""The reference-anchored half of the ``innovator`` reward.

Mirrors ``gap_finder/reference_rubric.py``: the human-validated candidate
becomes a scoring anchor, decomposed into one sub-item per candidate field so
the judge grades one field at a time, and the whole reward is

    reward = (base_quality + reference_quality + citation_f1) / 3

with ``reference_quality`` a tier-weighted mean over the match sub-items and
``citation_f1`` a deterministic set comparison. With no reference citation set
the scorer falls back to the two-way mean.

Gating is global and lives on the base side: a failed deterministic gate or an
unmet ``methodological_contribution`` zeroes both halves, and the anti-hacking
negatives still subtract from the base half — so the reference-match direction
cannot be farmed with fabricated or padded candidates.
"""

from __future__ import annotations

from ideascientist.rewards.common import (
    SATISFACTION_FRACTION,
    TIER_WEIGHT,
    gate_check_ids,
)

from .rubric import RESEARCH_INTUITION_RUBRIC, score_candidate as base_score_candidate
from ideascientist.rewards.common import citation_set_f1, reference_quality_tiered


# The reference pillar, DECOMPOSED into one sub-item per required candidate field
# so the judge scores one field at a time (denser, more reliable per-call
# judgments than one holistic verdict). ``matches_reference_<field>`` is per-field
# recall of the validated label. Field IMPORTANCE encodes what actually carries
# the methodological signal of a research idea: the intuition itself, the gap it
# attacks, the causal mapping into the target problem, the why-it-works mechanism,
# the named novelty delta, and the borrowed source inspiration (THE CENTERPIECE of
# a cross-domain transfer) are ``important``; only the main-risk statement is
# supporting context (``optional``). The reference half's quality is a WEIGHTED
# ``mean(match fields)`` by these tiers (see ``_reference_quality``).
_REF_FIELDS: tuple[tuple[str, str, str, str], ...] = (
    (
        "intuition",
        "Intuition",
        "the core research intuition (the one central transferable idea)",
        "important",
    ),
    (
        "gap",
        "Gap it attacks",
        "which specific gap the intuition attacks and its concrete failure mode",
        "important",
    ),
    (
        "mapping",
        "How it maps to this problem",
        "the concrete mapping of the source mechanism onto the target problem",
        "important",
    ),
    (
        "mechanism",
        "Why it could work / feasibility",
        "the causal reason the transferred mechanism could work here",
        "important",
    ),
    (
        "novelty",
        "Novelty vs. closest in-domain work",
        "the checkable difference versus the named closest in-domain prior work",
        "important",
    ),
    (
        "source",
        "Source inspiration",
        "the borrowed source mechanism / inspiration and its origin paper(s)",
        "important",
    ),
    (
        "risk",
        "Main risk",
        "the most likely reason the intuition or transfer would fail",
        "optional",
    ),
)


def _match_item(slug: str, field: str, desc: str, tier: str) -> dict:
    return {
        "id": f"matches_reference_{slug}",
        "gate": False,
        "scope": "candidate",
        "tier": tier,
        "weight": TIER_WEIGHT[tier],
        "criterion": (
            f"The '{field}' field captures the SAME content as one reference "
            f"candidate's '{field}'."
        ),
        "explanation": (
            f"Per-field recall of the validated label for {desc}. Score 2 when the "
            f"generated candidate's '{field}' identifies the same {desc} as a "
            f"reference candidate, 1 when it partially overlaps (same area, misses "
            f"the specific point), 0 when it addresses something the reference "
            f"candidates do not. Different wording or citations are fine."
        ),
    }


_MATCH_ITEMS: list[dict] = [_match_item(s, f, d, t) for s, f, d, t in _REF_FIELDS]

# Sub-item ids for pillar-level aggregation in ``_reference_quality`` (weighted
# mean across the per-field match sub-items).
MATCH_IDS: tuple[str, ...] = tuple(r["id"] for r in _MATCH_ITEMS)
_PILLAR_IDS: tuple[str, ...] = MATCH_IDS


# --------------------------------------------------------------------------- #
# Judge prompt: reference-anchored. Adds a <<reference_candidate>> block (the
# validated reference candidates) so the match pillar can be scored. The base
# intrinsic-quality items and negatives are judged exactly as in the base rubric;
# they simply ignore the reference block. BATCHED keyed-JSON output (one call per
# group).
# --------------------------------------------------------------------------- #
JUDGE_PROMPT_TEMPLATE = """\
You are scoring ONE research intuition written by an innovator agent against a GROUP
of rubric items. The candidate is judged RELATIVE TO the reference candidate
analysis (a human-validated label).

# Target gap the intuition must attack
<<gap>>

# Reference candidate analysis for this gap (human-validated label to compare against)
<<reference_candidate>>

# Papers the agent read this run (id -> one-line finding it can rely on)
<<available_papers>>

# The generated candidate being scored
<<candidate>>

# Rubric items to score (each has an id, a kind, and a criterion)
<<rubric_items>>

# Instructions
- This reward is for a CLEAR, WELL-SOURCED RESEARCH INTUITION, not a full method
  writeup. Do NOT penalize missing implementation details (pseudocode, equations,
  architecture, hyperparameters, an evaluation plan, expected margins, or
  falsifiable predictions); those are downstream report-writer work.
- The candidate must be a methodological contribution. Analysis, benchmark,
  dataset, survey, and position-paper ideas get no positive credit on
  methodological-contribution items.
- Score EVERY rubric item in the group above, independently.
- For a "matches_reference_<field>" item: judge whether the generated candidate's
  named field captures the SAME content as one reference candidate's corresponding
  field (2 = same, 1 = partial overlap, 0 = different). Different wording or
  citations are fine.
- For any other POSITIVE rubric item, score 0 (absent/contradicted), 1 (partial),
  or 2 (clearly satisfied), judging ONLY that item.
- For a NEGATIVE rubric item, score 1 if the flaw is PRESENT and 0 if absent.
  Judge whether the flaw is present, not overall quality.
- Ground your judgment in the read papers above. Do not credit a claim about a
  paper that its listed finding does not support.
- Return ONLY a JSON object keyed by rubric item id, each value an object with an
  integer "score" and a 1-sentence "explanation". Score EXACTLY the ids listed,
  no more and no fewer. Example shape:
  {"<id_a>": {"score": 2, "explanation": "..."}, "<id_b>": {"score": 0, "explanation": "..."}}
"""


# --------------------------------------------------------------------------- #
# Judge grouping: the COMBINED rubric's items are scored in THREE batched judge
# calls (instead of one call per item) — the base intrinsic positives, the
# per-field matches_reference sub-items, and the negatives. ``judge_groups``
# returns the ordered (group_name, kind, [items]) tuples the driver iterates over.
# --------------------------------------------------------------------------- #
_BASE_POSITIVE_IDS: frozenset[str] = frozenset(
    r["id"] for r in RESEARCH_INTUITION_RUBRIC["positive_rubrics"]
)




# --------------------------------------------------------------------------- #
# COMBINED rubric (drives the JUDGE) + scorer.
# --------------------------------------------------------------------------- #
RESEARCH_INTUITION_RUBRIC_COMBINED: dict = {
    "name": "research_intuition_combined",
    "source_documents": RESEARCH_INTUITION_RUBRIC["source_documents"],
    "unit_of_scoring": "candidate",
    "required_candidate_fields": RESEARCH_INTUITION_RUBRIC["required_candidate_fields"],
    "optional_candidate_fields": RESEARCH_INTUITION_RUBRIC.get(
        "optional_candidate_fields", []
    ),
    "scoring_hint": (
        "Judges the whole candidate against the reference. Global gating first: "
        "HARD-REJECTED (reward 0) if any deterministic gate fails or "
        "methodological_contribution scores below 2. Otherwise reward = "
        "(base_quality + reference_quality + citation_f1) / 3, each already "
        "normalized to [0, 1]. The reference quality is a tier-weighted "
        "mean(matches_reference): per-field recall of the validated label. "
        "citation_f1 is the set-F1 of the generated vs reference cited-paper ids "
        "(a pure deterministic verifier)."
    ),
    "verifiable_checks": RESEARCH_INTUITION_RUBRIC["verifiable_checks"],
    "positive_rubrics": (
        RESEARCH_INTUITION_RUBRIC["positive_rubrics"]
        + _MATCH_ITEMS
    ),
    "negative_rubrics": RESEARCH_INTUITION_RUBRIC["negative_rubrics"],
}






def score_document_combined(
    n_candidates: int,
    judgments: dict[str, int],
    *,
    det: dict[str, bool] | None = None,
    citation_f1: float | None = None,
    component_weights: dict[str, float] | None = None,
    rubric: dict = RESEARCH_INTUITION_RUBRIC_COMBINED,
) -> float:
    """Combined reward for the WHOLE ``candidates.md`` in [0, 1], doc-to-doc.

    Global gating first (all components zeroed together): reward 0 if any
    deterministic gate fails or the methodological_contribution gate scores below
    2. Otherwise the reward is a WEIGHTED mean of up to THREE components, each
    already in [0, 1]: ``base_quality`` (``rubrics.score_candidate``),
    ``reference_quality`` (tier-weighted ``mean(match)``), and ``citation_f1``
    (set-F1 of the generated vs reference cited-paper ids).

    ``component_weights`` gives the RELATIVE weight of each present component
    (``{"base", "reference", "citation"}``); only ratios matter. Defaults to equal
    weights (all-present -> ``(base + ref + cit) / 3``). A weight-0 component is
    dropped. ``citation_f1`` None -> the citation component is dropped and the
    reward is the weighted mean of base + reference.
    """
    if n_candidates <= 0:
        return 0.0
    if det is None:
        det = {cid: True for cid in gate_check_ids(rubric)}
    base_q = base_score_candidate(det, judgments, rubric=RESEARCH_INTUITION_RUBRIC)
    if base_q <= 0.0 and _is_gated(det, judgments):
        return 0.0
    ref_q = reference_quality_tiered(judgments, _MATCH_ITEMS, MATCH_IDS)

    w = component_weights or {"base": 1.0, "reference": 1.0, "citation": 1.0}
    parts: list[tuple[float, float]] = [  # (weight, value)
        (float(w.get("base", 0.0)), base_q),
        (float(w.get("reference", 0.0)), ref_q),
    ]
    if citation_f1 is not None:
        cit_q = min(max(float(citation_f1), 0.0), 1.0)
        parts.append((float(w.get("citation", 0.0)), cit_q))

    total_w = sum(weight for weight, _ in parts if weight > 0.0)
    if total_w <= 0.0:
        return 0.0
    return sum(weight * value for weight, value in parts if weight > 0.0) / total_w


def _is_gated(det: dict[str, bool], judgments: dict[str, int]) -> bool:
    """True if a global gate is unmet (any deterministic gate or methodological_contribution).

    The innovator base rubric HAS a gate positive (``methodological_contribution``),
    so it is part of the global gate — unlike report_writer, whose _is_gated is
    deterministic-gates-only.
    """
    for cid in gate_check_ids(RESEARCH_INTUITION_RUBRIC):
        if not det.get(cid, False):
            return True
    return judgments.get("methodological_contribution", 0) < 2
