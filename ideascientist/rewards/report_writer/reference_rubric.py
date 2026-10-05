"""The reference-anchored half of the ``report_writer`` reward.

Mirrors ``gap_finder/reference_rubric.py``, with one sub-item per scored
results-masked field so the judge grades one field at a time:

    reward = (base_quality + reference_quality + citation_f1) / 3

Not every field is scored. The framing fields — title, topic_relevance,
one_sentence_thesis, key_terms — are omitted entirely: no sub-item is emitted,
the judge is never asked about them, and they contribute nothing. Reproducing a
title is not evidence of having reproduced a proposal. Everything that is
scored carries the ``important`` tier.

Gating is global and deterministic-only here: unlike the other two roles,
report_writer's base rubric defines no gate positive. The anti-hacking
negatives still subtract from the base half, so the reference-match direction
cannot be farmed with overclaiming or vague proposals.
"""

from __future__ import annotations

from ideascientist.rewards.common import (
    SATISFACTION_FRACTION,
    TIER_WEIGHT,
    gate_check_ids,
)

from .rubric import SOLUTION_PROPOSAL_RUBRIC, score_proposal as base_score_proposal
from ideascientist.rewards.common import citation_set_f1, reference_quality_tiered


# The reference pillar, DECOMPOSED into one sub-item per SCORED results-masked schema
# field so the judge scores one field at a time. ``matches_reference_<field>`` is
# per-field recall of the reference proposal. Field importance: the
# proposed method / pipeline / eval / setup, the substantive scientific content
# (core_problem, key_novelty, problem_definition, falsifiable_predictions), and
# related_work + limitations are ALL ``important``. The pure framing fields (title,
# topic_relevance, one_sentence_thesis, key_terms) are OMITTED entirely — no match
# sub-item, so the judge is not called on them and they do not enter the reward.
# ``references`` is likewise OMITTED as a match sub-item — citation overlap is
# scored deterministically by the citation_f1 component instead.
_REF_FIELDS: tuple[tuple[str, str, str, str], ...] = (
    (
        "method",
        "method",
        "the proposed method / pipeline: modules, data flow, objectives, design choices",
        "important",
    ),
    (
        "model_details",
        "model_details",
        "the implementation setup: base model, architecture/adaptation, training, hyperparameters",
        "important",
    ),
    (
        "proposed_evaluation",
        "proposed_evaluation",
        "the evaluation plan: setting, benchmarks, baselines, metrics that test the claim",
        "important",
    ),
    (
        "comparison_to_sota",
        "comparison_to_sota",
        "the planned SOTA comparison: which strong baselines and how the method differs",
        "important",
    ),
    (
        "core_problem",
        "core_problem",
        "the core problem: what fails now, why it is hard, and the stakes",
        "important",
    ),
    (
        "key_novelty",
        "key_novelty",
        "the reusable novelty insight and why prior work does not already cover it",
        "important",
    ),
    (
        "problem_definition",
        "problem_definition",
        "the formal problem setting: task, inputs, outputs, evaluation target, scope",
        "important",
    ),
    (
        "falsifiable_predictions",
        "falsifiable_predictions",
        "the risky predictions and kill conditions for the major design pillars",
        "important",
    ),
    (
        "related_work",
        "related_work",
        "how prior work is organized and how the proposal differs from it",
        "important",
    ),
    (
        "limitations",
        "limitations",
        "the stated assumptions, scope bounds, and realistic failure modes",
        "important",
    ),
)


def _match_item(slug: str, field: str, desc: str, tier: str) -> dict:
    return {
        "id": f"matches_reference_{slug}",
        "gate": False,
        "scope": "proposal",
        "schema_key": field,
        "tier": tier,
        "weight": TIER_WEIGHT[tier],
        "criterion": (
            f"The '{field}' field captures the SAME content as the reference "
            f"proposal's '{field}'."
        ),
        "explanation": (
            f"Per-field recall of the validated label for {desc}. Score 2 when the "
            f"generated proposal's '{field}' identifies the same {desc} as the "
            f"reference proposal, 1 when it partially overlaps (same area, misses "
            f"the specific point), 0 when it addresses something the reference "
            f"proposal does not. Different wording or citations are fine."
        ),
    }


_MATCH_ITEMS: list[dict] = [_match_item(s, f, d, t) for s, f, d, t in _REF_FIELDS]

# Sub-item ids for pillar-level aggregation in ``_reference_quality`` (weighted
# mean across the per-field match sub-items).
MATCH_IDS: tuple[str, ...] = tuple(r["id"] for r in _MATCH_ITEMS)
_PILLAR_IDS: tuple[str, ...] = MATCH_IDS


# --------------------------------------------------------------------------- #
# Judge prompt: reference-anchored. Adds a <<reference_proposal>> block (the
# validated reference proposal) so the match pillar can be scored. The base
# intrinsic-quality items and negatives are judged exactly as in the base rubric;
# they simply ignore the reference block. BATCHED keyed-JSON output (one call per
# group).
# --------------------------------------------------------------------------- #
JUDGE_PROMPT_TEMPLATE = """\
You are scoring ONE research solution proposal written by a report-writer agent
against a GROUP of rubric items. The proposal is judged RELATIVE TO the reference
proposal (a human-validated label).

# Research problem the proposal must address
<<problem>>

# Research gap the proposal was assigned to close (from the upstream gap_finder)
<<gap>>

# Research intuition the proposal was told to develop (from the upstream innovator)
<<intuition>>

# Reference proposal for this problem (human-validated label to compare against)
<<reference_proposal>>

# Papers the agent read this run (id -> one-line finding it can rely on)
<<available_papers>>

# The generated proposal being scored (report.json)
<<proposal>>

# Rubric items to score (each has an id, a kind, and a criterion)
<<rubric_items>>

# Instructions
- This is a research PROPOSAL: the experiments have NOT been run. Do NOT penalize
  the absence of measured results; DO penalize expected outcomes presented as
  measured, and reward clearly separated predictions with kill conditions.
- Score EVERY rubric item in the group above, independently.
- For a "matches_reference_<field>" item: judge whether the generated proposal's
  named field captures the SAME content as the reference proposal's corresponding
  field (2 = same, 1 = partial overlap, 0 = different). Different wording or
  citations are fine.
- If a rubric item names a schema_key, judge that field of the proposal (and any
  inline claims that depend on it), not the whole document.
- For any other POSITIVE rubric item, score 0 (absent/contradicted), 1 (partial),
  or 2 (clearly satisfied), judging ONLY that item.
- For a NEGATIVE rubric item, score 1 if the flaw is PRESENT and 0 if absent.
  Judge whether the flaw is present, not overall quality.
- Ground your judgment in the read papers above. Do not credit (or fault) a claim
  about a paper that its listed finding does not support.
- Return ONLY a JSON object keyed by rubric item id, each value an object with an
  integer "score" and a 1-sentence "explanation". Score EXACTLY the ids listed,
  no more and no fewer. Example shape:
  {"<id_a>": {"score": 2, "explanation": "..."}, "<id_b>": {"score": 0, "explanation": "..."}}
"""


# --------------------------------------------------------------------------- #
# Judge grouping: THREE batched judge calls — the base intrinsic positives, the
# per-field matches_reference sub-items, and the negatives.
# --------------------------------------------------------------------------- #
_BASE_POSITIVE_IDS: frozenset[str] = frozenset(
    r["id"] for r in SOLUTION_PROPOSAL_RUBRIC["positive_rubrics"]
)




# --------------------------------------------------------------------------- #
# COMBINED rubric (drives the JUDGE) + scorer.
# --------------------------------------------------------------------------- #
SOLUTION_PROPOSAL_RUBRIC_COMBINED: dict = {
    "name": "solution_proposal_combined",
    "source_documents": SOLUTION_PROPOSAL_RUBRIC["source_documents"],
    "unit_of_scoring": "proposal",
    "required_schema_fields": SOLUTION_PROPOSAL_RUBRIC["required_schema_fields"],
    "scoring_hint": (
        "Judges the whole proposal against the reference. Global gating first: "
        "HARD-REJECTED (reward 0) if any deterministic gate fails (report_writer "
        "has no gate positive). Otherwise reward = (base_quality + "
        "reference_quality + citation_f1) / 3, each already normalized to [0, 1]. "
        "The reference quality is the mean(matches_reference) over the substantive "
        "fields only — method / model_details / proposed_evaluation / "
        "comparison_to_sota, the scientific content (core_problem / key_novelty / "
        "problem_definition / falsifiable_predictions), and related_work / "
        "limitations; the pure framing fields are omitted and never judged. "
        "citation_f1 is the set-F1 of the generated vs "
        "reference cited-paper anchor ids (a pure deterministic verifier)."
    ),
    "verifiable_checks": SOLUTION_PROPOSAL_RUBRIC["verifiable_checks"],
    "positive_rubrics": (
        SOLUTION_PROPOSAL_RUBRIC["positive_rubrics"]
        + _MATCH_ITEMS
    ),
    "negative_rubrics": SOLUTION_PROPOSAL_RUBRIC["negative_rubrics"],
}






def score_document_combined(
    n_proposals: int,
    judgments: dict[str, int],
    *,
    det: dict[str, bool] | None = None,
    citation_f1: float | None = None,
    component_weights: dict[str, float] | None = None,
    rubric: dict = SOLUTION_PROPOSAL_RUBRIC_COMBINED,
) -> float:
    """Combined reward for the WHOLE ``report.json`` in [0, 1], doc-to-doc.

    Global gating first (all components zeroed together): reward 0 if any
    deterministic gate fails. (report_writer defines no gate positive, so the
    global gate is deterministic-gates-only.) Otherwise the reward is a WEIGHTED
    mean of up to THREE components, each already in [0, 1]: ``base_quality``
    (``rubrics.score_proposal``), ``reference_quality`` (tier-weighted
    ``mean(match)``), and ``citation_f1`` (set-F1 of the generated vs reference
    cited-paper anchor ids).

    ``component_weights`` gives the RELATIVE weight of each present component
    (``{"base", "reference", "citation"}``); only ratios matter. Defaults to equal
    weights (all-present -> ``(base + ref + cit) / 3``). A weight-0 component is
    dropped. ``citation_f1`` None -> the citation component is dropped and the
    reward is the weighted mean of base + reference.
    """
    if n_proposals <= 0:
        return 0.0
    if det is None:
        det = {cid: True for cid in gate_check_ids(rubric)}
    base_q = base_score_proposal(det, judgments, rubric=SOLUTION_PROPOSAL_RUBRIC)
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
    """True if a global gate is unmet.

    DETERMINISTIC-GATES-ONLY: the report_writer base rubric defines NO gate
    positive, so ``judgments`` plays no role here — only the deterministic
    verifier gates can hard-reject the whole reward. (Contrast gap_finder /
    innovator, which also gate on a positive.)
    """
    for cid in gate_check_ids(SOLUTION_PROPOSAL_RUBRIC):
        if not det.get(cid, False):
            return True
    return False
