"""Rubric-based reward for training the ``gap_finder`` sub-agent.

The ``gap_finder`` first-tier sub-agent reads same-problem prior work on ONE challenge
axis and writes 1-2 grounded research gaps to ``gaps.md``. This module defines
the rubric that turns a gap analysis into a scalar reward for on-policy RL, plus
the deterministic verifier checks, the judge prompt, and the aggregation
function used to compute the reward.

Grounded in:
- ``docs/literature_reviews/what_is_a_good_research/research_gaps.md``
  (what makes a defensible research gap).
- ``docs/literature_reviews/rubric_based_rl.md`` (how to design a rubric reward
  that resists reward hacking).

Design choices, and the survey finding each one comes from
(``docs/literature_reviews/rubric_based_rl.md``):

1. **Close the presence/absence asymmetry without duplicating criteria.**
   Mahmoud et al. (2605.12474) show rubrics hold ~90% of weight in "reward
   containing X" criteria and only ~9% in "penalize doing Y" criteria, and RL
   pushes hardest on the under-specified axis. We keep explicit negative
   rubrics, but only for distinct failure modes rather than the inverse of every
   positive criterion.
2. **Gatekeeper / veto.** OpenRubrics/Rubicon/AdvancedIF: enforce hard,
   objective constraints before scoring soft qualities. Gate rubrics (and the
   deterministic gate checks) zero the reward for a gap when any is unmet,
   exactly like Rubicon's veto and AdvancedIF's all-or-nothing. The
   ``methodological_focus`` positive gate is the single source of truth for
   rejecting benchmark/data/evaluation-only gaps.
3. **Deterministic verifier grounding.** RLCF/Rubicon score objectively
   checkable items with programs, not a gameable judge; DR Tulu adds a
   citation-faithfulness reward. The single biggest gap-finder hack is
   fabricating near-miss paper IDs — so ``VERIFIABLE_CHECKS`` deterministically
   verify that every cited ID resolves in the corpus AND was actually read via
   ``read_paper`` this run, before the LLM judge is ever consulted.
4. **Per-gap credit + depth-over-breadth.** SRaR/DR Tulu score per item, not one
   blob. We score each open gap independently, then combine per-axis with a
   padding penalty so listing many shallow gaps cannot beat one deep one.
5. **Anti-parroting / anti-artifact.** RuscaRL hides the rubric from the policy;
   Rubicon/AdvancedIF penalize self-praise and "this response is perfect"
   meta-commentary. The ``self_evaluation_artifacts`` negative + the format gate
   suppress those.
6. **Normalized weighted aggregation.** HealthBench: score =
   achieved_points / total_positive_points, clipped to [0, 1].
"""

from __future__ import annotations

from ideascientist.rewards.common import (
    TIER_WEIGHT,
    score_unit,
)


RESEARCH_GAP_RUBRIC: dict = {
    "name": "research_gap",
    "source_documents": [
        "docs/literature_reviews/what_is_a_good_research/research_gaps.md",
        "docs/literature_reviews/rubric_based_rl.md",
    ],
    # Unit of scoring: the whole ``gaps.md`` for the axis is judged doc-to-doc
    # (one score per rubric item; see ``score_document``). "gap" scope items are
    # read as document-level qualities; "axis" scope items apply to the whole
    # ``## Axis`` block.
    "unit_of_scoring": "gap",
    "required_gap_fields": [
        "Gap",
        "Near-miss methods and coverage",
        "Methodological deficiency",
        "Why it matters",
        "Evidence route",
    ],
    "scoring_hint": (
        "Score each positive item 0 = absent or contradicted, 1 = partially "
        "present, 2 = clearly satisfied. Score each negative item 0 = flaw "
        "absent, 1 = flaw present (a negative scored 1 subtracts its weight). "
        "A gap is HARD-REJECTED (reward 0) if any gate rubric scores below 2 or "
        "any deterministic gate check in VERIFIABLE_CHECKS fails. Judge the soft "
        "qualities only for methodological gaps that clear every gate."
    ),
    # ------------------------------------------------------------------ #
    # Deterministic checks (RLCF/Rubicon verifier-programs + DR Tulu citation
    # faithfulness). Computed in code by the RL environment BEFORE the LLM judge;
    # all are gates that hard-reject on failure. They make the most hackable
    # requirement (near-miss grounding) verifiable, so the policy cannot farm
    # reward with fabricated or unread citations.
    # ------------------------------------------------------------------ #
    "verifiable_checks": [
        {
            "id": "schema_valid",
            "gate": True,
            "check": (
                "The gap parses under the required schema: a '- **Gap:**' entry "
                "with non-empty Near-miss methods and coverage (>=1 paper id "
                "plus method/coverage statement), Methodological deficiency, "
                "Why it matters, and Evidence route fields, and no truncated "
                "final field."
            ),
            "rationale": (
                "AdvancedIF's completeness/no-artifact reward shaping: an "
                "unparseable or cut-off gap gets no reward and cannot be scored."
            ),
        },
        {
            "id": "cited_ids_resolve",
            "gate": True,
            "check": (
                "Every paper id cited by the gap resolves to a real paper in the "
                "corpus (get_paper_biblio succeeds). No fabricated ids."
            ),
            "rationale": (
                "Directly kills the fabricated-citation hack (DR Tulu). A gap "
                "citing an id that does not exist is worthless and gameable."
            ),
        },
        {
            "id": "cited_ids_were_read",
            "gate": True,
            "check": (
                "Every near-miss id the gap leans on appears in this run's "
                "read_paper digests (papers/<id>.md exists with answered goals). "
                "The gap is grounded in reading, not in titles/abstracts."
            ),
            "rationale": (
                "Enforces the prompt rule 'ground each gap in a read_paper "
                "digest'. Prevents crediting near-miss claims the policy never "
                "actually verified."
            ),
        },
    ],
    "positive_rubrics": [
        {
            "id": "methodological_focus",
            "gate": True,
            "scope": "gap",
            "tier": "important",
            "weight": TIER_WEIGHT["important"],
            "criterion": "Is a METHODOLOGICAL gap, closable by a new or improved method/mechanism.",
            "explanation": (
                "This pipeline finds ONLY methodological gaps: the deficiency "
                "is in the approach itself (algorithm, model mechanism, "
                "training procedure, objective, inductive bias, credit "
                "assignment, representation) and closing it requires proposing "
                "or improving a METHOD. Gaps whose only fix is a new "
                "benchmark/dataset, more experiments, or a missing evaluation "
                "are OUT OF SCOPE and must hard-fail."
            ),
            "examples": [
                "Good: 'Existing code-RL methods assign one trajectory-level "
                "reward, so per-step credit is lost — a methodological "
                "credit-assignment gap.'",
                "Out of scope: 'There is no benchmark for multilingual code "
                "generation.' (benchmark gap)",
            ],
        },
        {
            "id": "concrete_deficiency",
            "gate": False,
            "scope": "gap",
            "tier": "important",
            "weight": TIER_WEIGHT["important"],
            "criterion": "Explains the ONE precise METHODOLOGICAL deficiency that keeps the near-miss work from closing the gap.",
            "explanation": (
                "The gap should name a single precise deficiency in the "
                "APPROACH itself — why the mechanism / algorithm / training "
                "procedure / objective / inductive bias / credit assignment / "
                "representation falls short — and tie it to the cited work. "
                "Empirical evidence may support the claim, but the deficiency "
                "must be about the method, not merely 'nobody ran the "
                "experiment'. State ONE deficiency, not a bundle."
            ),
            "examples": [
                "Good: 'Both methods treat provenance as static metadata, so "
                "the generator cannot propagate credit when retrieved evidence "
                "is revised after the answer is formed.'",
                "Weak: 'The prior work is limited in several ways.'",
            ],
        },
        {
            "id": "so_what_stakes",
            "gate": False,
            "scope": "gap",
            "tier": "important",
            "weight": TIER_WEIGHT["important"],
            "criterion": "Passes the 'so what?' test: states who is blocked and what closing it unlocks.",
            "explanation": (
                "The gap should explain who benefits, what intellectual or "
                "practical cost remains if it stays open, and what new "
                "understanding or capability becomes possible if it is closed. "
                "Its significance cannot be only leaderboard/SOTA improvement "
                "or technical difficulty."
            ),
            "examples": [
                "Good: 'Without a method that preserves provenance through "
                "document revisions, assistant builders cannot decide whether "
                "an answer remains grounded after source updates, so trusted "
                "enterprise deployment remains blocked.'",
            ],
        },
        {
            "id": "specific_contrastive_formulation",
            "gate": False,
            "scope": "gap",
            "tier": "important",
            "weight": TIER_WEIGHT["important"],
            "criterion": "States a concrete prior-method-vs-gap contrast.",
            "explanation": (
                "A strong formulation identifies the affected task/setting, "
                "method component, failure mode, and assumption being challenged "
                "in a contrast like: prior work addresses X through method Y, "
                "but Y leaves method-level issue Z unresolved."
            ),
            "examples": [
                "Good: 'Prior memory methods assume independent turns are "
                "sufficient state, but user corrections create persistent "
                "constraints that the update mechanism must retain across "
                "sessions.'",
            ],
        },
        {
            "id": "feasible_route_to_progress",
            "gate": False,
            "scope": "gap",
            "tier": "optional",
            "weight": TIER_WEIGHT["optional"],
            "criterion": "Identifies a plausible evidence route to progress.",
            "explanation": (
                "A gap should be researchable with available data, methods, "
                "compute, expertise, and time. The field should state what "
                "observable comparison or evidence would show that a future "
                "method closes the gap, without requiring the gap finder to "
                "invent that method."
            ),
            "examples": [
                "Good: 'Progress can be checked by holding the task data fixed "
                "and comparing whether a method preserves source-version state "
                "through retrieval, generation, and citation after document "
                "updates.'",
            ],
        },
    ],
    "negative_rubrics": [
        {
            "id": "unsupported_evidence",
            "scope": "gap",
            "weight": 4.0,
            "criterion": "Uses fabricated, unread, or unsupported evidence.",
            "explanation": (
                "This is the core gap-finder hack. It includes citing an id "
                "that does not exist, citing a paper that was not read this run, "
                "or using a real read-paper digest to support a claim the digest "
                "does not actually contain. The first two are also caught by "
                "deterministic gates; this negative catches unsupported use of "
                "otherwise valid ids."
            ),
            "examples": [
                "Bad: 'As shown in [p_9999]...' where [p_9999] is not in the "
                "corpus, or citing a real id whose digest never mentions the "
                "claimed method limitation.",
            ],
        },
        {
            "id": "parroted_or_generic_gap",
            "scope": "gap",
            "weight": 3.0,
            "criterion": "Restates the axis or claims generic absence without near-miss evidence.",
            "explanation": (
                "A gap must add a specific, grounded deficiency beyond the "
                "assigned axis. Claims like 'no one has done X' are unreliable "
                "unless tied to named near-miss methods and a search/citation "
                "trail."
            ),
            "examples": [
                "Bad: Axis = 'long-horizon credit assignment' -> gap = 'prior "
                "work does not solve long-horizon credit assignment.'",
                "Bad: 'There is no work on assistant memory.'",
            ],
        },
        {
            "id": "overclaimed_scope",
            "scope": "gap",
            "weight": 3.0,
            "criterion": "Inflates the evidence into a broader gap than it supports.",
            "explanation": (
                "This covers single-paper limitation inflation, arbitrary "
                "domain/dataset changes with no method-level reason, ignoring "
                "same-axis work that already closes the gap, or turning a "
                "narrow missing comparison into a claim about an entire field."
            ),
            "examples": [
                "Bad: 'Because [p_1] uses a static memory encoder, the field "
                "lacks update-aware memory methods,' while [p_2] and [p_3] "
                "already model updates.",
                "Bad: 'No one tried the same classifier on Dataset B,' with no "
                "method-level reason Dataset B changes the claim.",
            ],
        },
        {
            "id": "self_evaluation_artifacts",
            "scope": "gap",
            "weight": 2.0,
            "criterion": "Adds self-praise, meta-commentary, or claims the gap satisfies the rubric.",
            "explanation": (
                "Rubicon/AdvancedIF self-evaluation hack: text like 'this is a "
                "strong, well-grounded gap' or references to being scored. The "
                "gap should read as a natural finding with no meta-commentary."
            ),
            "examples": [
                "Bad: 'This gap is highly novel and clearly passes the so-what "
                "test.'",
            ],
        },
        {
            "id": "laundry_list_padding",
            "scope": "gap",
            "weight": 2.0,
            "criterion": "Pads the axis with many shallow gaps to farm coverage.",
            "explanation": (
                "Depth-over-breadth hack: emitting many weakly-grounded gaps to "
                "increase the chance some score. This negative subtracts its weight "
                "from the document score; the driver ALSO hard-gates any document "
                "that exceeds the gap-count cap (reward 0)."
            ),
            "examples": [
                "Bad: Six one-line 'open questions' with no cited near-miss work.",
            ],
        },
    ],
}


# --------------------------------------------------------------------------- #
# Judge prompt (HealthBench GRADER_TEMPLATE style): one gap, one rubric item,
# with the axis and the read papers supplied for grounding. The RL environment
# fills the placeholders and calls the judge once per (gap, rubric item), then
# feeds the parsed scores to ``score_gap``.
# --------------------------------------------------------------------------- #
JUDGE_PROMPT_TEMPLATE = """\
You are scoring ONE research gap written by a gap-finding agent against ONE rubric item.

# Challenge axis the gap must address
<<axis>>

# Papers the agent read this run (id -> one-line finding it can rely on)
<<available_papers>>

# The gap being scored
<<gap>>

# Rubric item
<<rubric_item>>

# Instructions
- This reward is for METHODOLOGICAL research gaps only. A benchmark/dataset,
  empirical-only, evaluation-only, or application-setting gap gets no positive
  credit unless the text identifies a concrete deficiency in the prior method,
  mechanism, objective, representation, or training procedure.
- If this is a POSITIVE rubric item, return "score" as 0 (absent/contradicted),
  1 (partially present), or 2 (clearly satisfied), judging ONLY this item.
- If this is a NEGATIVE rubric item, return "score" as 1 if the flaw is PRESENT
  and 0 if it is absent. Judge whether the flaw is present, not whether the gap
  is good overall.
- Ground your judgment in the read papers above. Do not credit a claim about a
  paper that its listed finding does not support.
- Return ONLY a JSON object: {"explanation": "<1-2 sentences>", "score": <int>}.
"""


# --------------------------------------------------------------------------- #
# Reward aggregation (pure, dependency-free, testable). The environment supplies:
#   det:        {check_id: bool}   results of VERIFIABLE_CHECKS for this gap
#   judgments:  {rubric_id: int}   positive items 0/1/2, negative items 0/1
# and receives a scalar reward in [0, 1].
# --------------------------------------------------------------------------- #


def score_gap(
    det: dict[str, bool],
    judgments: dict[str, int],
    *,
    rubric: dict = RESEARCH_GAP_RUBRIC,
) -> float:
    """Reward for a single gap in [0, 1].

    Gated (reward 0) if any deterministic gate fails or any GATE positive rubric
    scores below 2 (Rubicon veto / AdvancedIF all-or-nothing). Non-methodological
    gaps are rejected here by the ``methodological_focus`` positive gate — there
    are no negative gates; negatives only subtract weight. Otherwise normalized
    weighted positive credit minus triggered negative penalties. See
    ``ideascientist.rewards.common.score_unit``.
    """
    return score_unit(det, judgments, rubric=rubric, scope="gap")


