"""The reference-anchored half of the ``gap_finder`` reward.

``rubric.py`` scores a gap analysis on its intrinsic qualities. This module
adds the human label as a scoring anchor, decomposed into one sub-item per gap
field so the judge grades one field at a time, and composes the whole reward:

    reward = (base_quality + reference_quality + citation_f1) / 3

``citation_f1`` is a deterministic set comparison, no judge. With no reference
citation set supplied the scorer falls back to the two-way mean.

There is deliberately no second "at least as good as the reference" pillar. A
pairwise dominance judge saturates near the top of its scale, so it contributes
almost no advantage variance, and combining the two as ``max`` would hide the
discriminative ``matches_reference`` signal underneath a near-constant term.

Gating is global and lives on the base side: a failed deterministic gate or an
unmet ``methodological_focus`` zeroes both halves, and the anti-hacking
negatives still subtract from the base half — so the reference-match direction
cannot be farmed with fabricated or padded gaps.

Reference-anchored recall judging follows HealthBench and DR Tulu, kept gated
and verifier-grounded so the judge is never trusted for the hackable parts.
"""

from __future__ import annotations

from ideascientist.rewards.common import (
    SATISFACTION_FRACTION,
    TIER_WEIGHT,
    gate_check_ids,
)

from .rubric import RESEARCH_GAP_RUBRIC, score_gap as base_score_gap
from ideascientist.rewards.common import citation_set_f1


# The reference pillar, DECOMPOSED into one sub-item per required gap field so the
# judge scores one field at a time (denser, more reliable per-call judgments than
# one holistic verdict). ``matches_reference_<field>`` is per-field recall of the
# validated label. The reference half's quality is ``mean(match fields)`` (see
# ``_reference_quality``).
_REF_FIELDS: tuple[tuple[str, str, str], ...] = (
    ("gap", "Gap", "the core gap statement"),
    (
        "nearmiss",
        "Near-miss methods and coverage",
        "which prior near-miss methods are named and what they cover vs. leave open",
    ),
    (
        "deficiency",
        "Methodological deficiency",
        "the precise method-level deficiency that blocks the near-miss work",
    ),
    ("why", "Why it matters", "the significance / 'so what?' stakes"),
    (
        "route",
        "Evidence route",
        "the observable route that would show a future method closes the gap",
    ),
)


def _match_item(slug: str, field: str, desc: str) -> dict:
    return {
        "id": f"matches_reference_{slug}",
        "gate": False,
        "scope": "gap",
        "tier": "important",
        "weight": TIER_WEIGHT["important"],
        "criterion": (
            f"The '{field}' field captures the SAME content as one reference "
            f"gap's '{field}'."
        ),
        "explanation": (
            f"Per-field recall of the validated label for {desc}. Score 2 when the "
            f"generated gap's '{field}' identifies the same {desc} as a reference "
            f"gap, 1 when it partially overlaps (same area, misses the specific "
            f"point), 0 when it addresses something the reference gaps do not. "
            f"Different wording or citations are fine."
        ),
    }


_MATCH_ITEMS: list[dict] = [_match_item(s, f, d) for s, f, d in _REF_FIELDS]

# Sub-item ids for pillar-level aggregation in ``_reference_quality`` (mean across
# the per-field match sub-items).
MATCH_IDS: tuple[str, ...] = tuple(r["id"] for r in _MATCH_ITEMS)
_PILLAR_IDS: tuple[str, ...] = MATCH_IDS


# --------------------------------------------------------------------------- #
# Judge prompt: reference-anchored. Adds a <<reference_gap>> block (the axis's
# validated reference gaps) so the match pillar can be scored. The base
# intrinsic-quality items and negatives are judged exactly as in the base rubric;
# they simply ignore the reference block.
# --------------------------------------------------------------------------- #
JUDGE_PROMPT_TEMPLATE = """\
You are scoring ONE research gap written by a gap-finding agent against a GROUP of
rubric items. The gap is judged RELATIVE TO the reference gap analysis (a
human-validated label).

# Challenge axis the gap must address
<<axis>>

# Reference gap analysis for this axis (human-validated label to compare against)
<<reference_gap>>

# Papers the agent read this run (id -> one-line finding it can rely on)
<<available_papers>>

# The generated gap being scored
<<gap>>

# Rubric items to score (each has an id, a kind, and a criterion)
<<rubric_items>>

# Instructions
- This reward is for METHODOLOGICAL research gaps only. A benchmark/dataset,
  empirical-only, evaluation-only, or application-setting gap gets no positive
  credit unless it identifies a concrete deficiency in the prior method,
  mechanism, objective, representation, or training procedure.
- Score EVERY rubric item in the group above, independently.
- For a "matches_reference_<field>" item: judge whether the generated gap's named
  field captures the SAME content as one reference gap's corresponding field (2 =
  same, 1 = partial overlap, 0 = different). Different wording or citations are
  fine.
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
# per-field matches_reference sub-items, and the negatives. Each group is a
# homogeneous set the judge scores in one pass, returning a JSON object keyed by
# item id. ``judge_groups`` returns the ordered (group_name, kind, [items]) tuples
# the driver iterates over.
# --------------------------------------------------------------------------- #
_BASE_POSITIVE_IDS: frozenset[str] = frozenset(
    r["id"] for r in RESEARCH_GAP_RUBRIC["positive_rubrics"]
)




# --------------------------------------------------------------------------- #
# COMBINED rubric (drives the JUDGE) + scorer.
#
# ``RESEARCH_GAP_RUBRIC_COMBINED`` exposes the UNION of the base rubric's scored
# items and the reference match pillar, so ONE judging pass fills every id both
# halves need. The scoring is done by ``score_document_combined``: it applies the
# global gates once, then averages the base half (``rubrics.score_gap``, which
# normalizes the intrinsic positives and subtracts the negatives) with the
# reference half (``mean(match)``). Both are in [0, 1] so the average is in [0, 1].
# --------------------------------------------------------------------------- #
RESEARCH_GAP_RUBRIC_COMBINED: dict = {
    "name": "research_gap_combined",
    "source_documents": RESEARCH_GAP_RUBRIC["source_documents"],
    "unit_of_scoring": "gap",
    "required_gap_fields": RESEARCH_GAP_RUBRIC["required_gap_fields"],
    "scoring_hint": (
        "Judges the whole gap analysis against the reference. Global gating "
        "first: HARD-REJECTED (reward 0) if any deterministic gate fails or "
        "methodological_focus scores below 2. Otherwise reward = (base_quality + "
        "reference_quality + citation_f1) / 3, each already normalized to [0, 1]. "
        "The reference quality is mean(matches_reference): per-field recall of the "
        "validated label. citation_f1 is the set-F1 of the generated vs "
        "reference cited-paper ids (a pure deterministic verifier)."
    ),
    "verifiable_checks": RESEARCH_GAP_RUBRIC["verifiable_checks"],
    # Base positives (incl. the methodological_focus gate) + the per-field
    # reference match sub-items, so ONE judging pass fills every id both halves
    # read.
    "positive_rubrics": (
        RESEARCH_GAP_RUBRIC["positive_rubrics"]
        + _MATCH_ITEMS
    ),
    "negative_rubrics": RESEARCH_GAP_RUBRIC["negative_rubrics"],
}




def _reference_quality(judgments: dict[str, int]) -> float:
    """Reference half in [0, 1]: ``mean(match fields)``.

    The single ``matches_reference`` pillar is decomposed into one sub-item per
    required gap field, each judged 0/1/2 and mapped to 0/0.5/1. We average the
    per-field scores — so a rollout earns full reference credit by reproducing the
    validated label field-by-field. No gates or negatives here — gating is applied
    once, globally, by ``score_document_combined``; negatives subtract from the
    base half only.
    """
    def _mean(ids: tuple[str, ...]) -> float:
        if not ids:
            return 0.0
        return sum(
            SATISFACTION_FRACTION.get(judgments.get(i, 0), 0.0) for i in ids
        ) / len(ids)

    return _mean(MATCH_IDS)


def score_document_combined(
    n_gaps: int,
    judgments: dict[str, int],
    *,
    det: dict[str, bool] | None = None,
    citation_f1: float | None = None,
    component_weights: dict[str, float] | None = None,
    rubric: dict = RESEARCH_GAP_RUBRIC_COMBINED,
) -> float:
    """The reward for the whole ``gaps.md``, in [0, 1].

    Gating first: a failed deterministic gate or a ``methodological_focus``
    below 2 zeroes everything. Otherwise a weighted mean of base quality,
    reference recall, and citation F1, normalized by the weights of the
    components actually present — so dropping one (weight 0, or no reference
    citation set) rescales rather than penalises.

    The driver has already enforced the gates across all gaps and the gap-count
    cap, so ``det`` defaults to all-pass; ``methodological_focus`` is applied
    here through ``base_score_gap``.
    """
    if n_gaps <= 0:
        return 0.0
    if det is None:
        det = {cid: True for cid in gate_check_ids(rubric)}
    # base_score_gap applies the deterministic gates AND the methodological_focus
    # gate, returning 0 on any failure. When it gates to 0 we must zero the whole
    # reward (all components), not just the base half — otherwise a positive
    # reference/citation component could leak through a gate failure.
    base_q = base_score_gap(det, judgments, rubric=RESEARCH_GAP_RUBRIC)
    if base_q <= 0.0 and _is_gated(det, judgments):
        return 0.0
    ref_q = _reference_quality(judgments)

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
    """True if a global gate is unmet (any deterministic gate or methodological_focus)."""
    for cid in gate_check_ids(RESEARCH_GAP_RUBRIC):
        if not det.get(cid, False):
            return True
    return judgments.get("methodological_focus", 0) < 2
