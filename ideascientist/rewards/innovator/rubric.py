"""Rubric-based reward for training the ``innovator`` sub-agent.

The ``innovator`` first-tier sub-agent reads ``gaps.md``, searches cross-domain source
papers, and writes ONE clearly stated research intuition to ``candidates.md``.
It does not write a full method, equations, pseudocode, hyperparameters, or an
evaluation plan; those belong to the downstream report writer. This module
defines the rubric, deterministic verifier checks, judge prompt, and aggregation
function for rewarding that intuition.

Grounded in:
- ``docs/literature_reviews/what_is_a_good_research/research_intuition_and_taste.md``
  (what makes a useful research intuition).
- ``docs/literature_reviews/rubric_based_rl.md`` (how to design rubric rewards
  that resist reward hacking).
"""

from __future__ import annotations

from ideascientist.rewards.common import TIER_WEIGHT, score_unit


RESEARCH_INTUITION_RUBRIC: dict = {
    "name": "research_intuition",
    "source_documents": [
        "docs/literature_reviews/what_is_a_good_research/research_intuition_and_taste.md",
        "docs/literature_reviews/rubric_based_rl.md",
    ],
    "unit_of_scoring": "candidate",
    "required_candidate_fields": [
        "Intuition",
        "Gap it attacks",
        "Source inspiration",
        "How it maps to this problem",
        "Why it could work / feasibility",
        "Novelty vs. closest in-domain work",
        "Main risk",
    ],
    "optional_candidate_fields": [
        "If improving prior candidate",
    ],
    "scoring_hint": (
        "Score each positive item 0 = absent or contradicted, 1 = partially "
        "present, 2 = clearly satisfied. Score each negative item 0 = flaw "
        "absent, 1 = flaw present. A candidate is HARD-REJECTED (reward 0) if "
        "any deterministic gate check fails or any gate positive rubric scores "
        "below 2. Do NOT penalize the absence of pseudocode, equations, "
        "architecture, hyperparameters, an evaluation plan, expected margins, "
        "or a falsifiable prediction; those are downstream report-writer work."
    ),
    "verifiable_checks": [
        {
            "id": "schema_valid",
            "gate": True,
            "check": (
                "The candidate parses as one '### C<n> — <short title>' block "
                "with the required fields: Intuition, Gap it attacks, "
                "Source inspiration, How it maps to this problem, Why it "
                "could work / feasibility, Novelty vs. closest in-domain work, "
                "and Main risk. The final field is not truncated."
            ),
            "rationale": (
                "A malformed or cut-off candidate cannot be scored reliably and "
                "should not receive reward."
            ),
        },
        {
            "id": "source_ids_resolve",
            "gate": True,
            "check": (
                "Every source-domain and closest in-domain paper id cited by "
                "the candidate resolves to a real paper in the corpus."
            ),
            "rationale": (
                "Prevents fabricated source or competitor ids from becoming a "
                "cheap way to claim grounding or novelty."
            ),
        },
        {
            "id": "source_ids_were_read",
            "gate": True,
            "check": (
                "Every paper id that carries a load-bearing source-mechanism or "
                "closest-prior novelty claim appears in this run's read_paper "
                "digests."
            ),
            "rationale": (
                "The innovator must ground the source mechanism and novelty delta "
                "in paper reading, not titles, abstracts, or search snippets."
            ),
        },
        {
            "id": "gap_exists",
            "gate": True,
            "check": (
                "The candidate names one specific gap or more from gaps.md by axis + gap, "
                "rather than attacking the whole problem or inventing a new gap."
            ),
            "rationale": (
                "Keeps ideation anchored to the prior gap-finder's evidence "
                "instead of rewarding generic topic brainstorming."
            ),
        },
    ],
    "positive_rubrics": [
        {
            "id": "methodological_contribution",
            "gate": True,
            "scope": "candidate",
            "tier": "important",
            "weight": TIER_WEIGHT["important"],
            "criterion": "Is a methodological research intuition, not an analysis/evaluation artifact.",
            "explanation": (
                "The candidate must plausibly become a method, technique, "
                "algorithm, model, training procedure, objective, representation, "
                "or system design. Analysis, benchmark, dataset, survey, and "
                "position-paper ideas hard-fail even if well written."
            ),
            "examples": [
                "Good: 'Use tree-search backup to assign credit to executable "
                "intermediate code states.'",
                "Out of scope: 'Build a benchmark for long-horizon code repair.'",
            ],
        },
        {
            "id": "grounded_gap_and_stakes",
            "gate": False,
            "scope": "candidate",
            "tier": "important",
            "weight": TIER_WEIGHT["important"],
            "criterion": "Explains the named gap's concrete current failure mode and what closing it would unlock (stakes).",
            "explanation": (
                "Given that the candidate names one or more real gaps from gaps.md (a "
                "prerequisite already enforced by the gap_exists gate check), this "
                "item judges QUALITY only: does it name the concrete current "
                "failure mode and state what closing the gap would unlock? It "
                "should earn little or no credit if it relies on broad importance "
                "language ('agents are important', 'LLMs need better reasoning') "
                "without saying what current approaches cannot do, who or what is "
                "blocked, and what new capability or understanding the intuition "
                "would unlock. Do not credit mere absence claims like 'no one has "
                "done X' unless they are tied to the named gap's concrete failure "
                "mode. Do not re-judge whether the gap exists in gaps.md here."
            ),
            "examples": [
                "Good: 'This attacks the Axis: credit assignment gap where "
                "trajectory-level reward loses which code edit caused failure; "
                "closing it would let code-RL methods learn from long partial "
                "solutions rather than only final pass/fail.'",
                "Weak: 'Code RL is important and underexplored.'",
            ],
        },
        {
            "id": "single_clear_intuition",
            "gate": False,
            "scope": "candidate",
            "tier": "important",
            "weight": TIER_WEIGHT["important"],
            "criterion": "States one central intuition clearly in one or two sentences.",
            "explanation": (
                "The innovator's deliverable is a crisp idea, not a bag of "
                "components. A reader should be able to repeat the central "
                "transfer in two sentences. Give little or no credit to a "
                "bundle of features, modules, or buzzwords that does not expose "
                "one reusable insight, even if the individual components sound "
                "plausible."
            ),
            "examples": [
                "Good: 'Treat partial programs like game-tree states: back up "
                "success/failure value through executable intermediates so the "
                "policy learns which edit moved the solution closer.'",
                "Weak: 'Combine search, RL, verification, and preference tuning.'",
            ],
        },
        {
            "id": "source_mechanism_grounded",
            "gate": False,
            "scope": "candidate",
            "tier": "important",
            "weight": TIER_WEIGHT["important"],
            "criterion": "Accurately explains the source mechanism it borrows and why it works in its source setting.",
            "explanation": (
                "The inspiration is the centerpiece. The candidate "
                "must name the source paper id(s), identify the borrowed "
                "mechanism/principle, and explain the condition under which it "
                "works in its source setting, grounded in what the innovator read. "
                "A CROSS-DOMAIN source (a mechanism from a distant sub-field) is "
                "preferred and should be attempted, but is NOT required: an "
                "in-domain source is acceptable as long as the borrowed mechanism "
                "is accurately grounded and its transfer is non-trivial. Do not "
                "penalize an in-domain source here on cross-domain grounds; the "
                "novelty of the transfer is judged separately by "
                "named_novelty_delta."
            ),
            "examples": [
                "Good: 'AlphaZero-style backup propagates value from evaluated "
                "leaf positions to earlier decisions because game states have "
                "well-defined successor values.'",
                "Weak: 'Tree search is powerful in games, so use it here.'",
            ],
        },
        {
            "id": "named_novelty_delta",
            "gate": False,
            "scope": "candidate",
            "tier": "important",
            "weight": TIER_WEIGHT["important"],
            "criterion": "Compares against named closest in-domain prior work and states a checkable difference.",
            "explanation": (
                "Novelty must be relative to a named competitor, not a vibe. The "
                "candidate should state what the closest in-domain paper already "
                "does and the specific difference that makes this transfer not "
                "a rediscovery."
            ),
            "examples": [
                "Good: 'Unlike [123], which assigns one final trajectory reward, "
                "this intuition backs value through executable intermediate "
                "states.'",
                "Weak: 'This is a new way to improve code RL.'",
            ],
        },
        {
            "id": "calibrated_main_risk",
            "gate": False,
            "scope": "candidate",
            "tier": "optional",
            "weight": TIER_WEIGHT["optional"],
            "criterion": "Names the most likely reason the intuition or transfer would fail.",
            "explanation": (
                "A useful intuition is calibrated. The candidate should state "
                "the main risk or false assumption directly, instead of using "
                "confidence as a substitute for evidence."
            ),
            "examples": [
                "Good: 'Main risk: partial programs may not yield stable enough "
                "intermediate signals for backup to be meaningful.'",
            ],
        },
    ],
    "negative_rubrics": [
        {
            "id": "metaphor_without_mapping",
            "scope": "candidate",
            "weight": 3.0,
            "criterion": "Relies on analogy or metaphor without a concrete conceptual mapping.",
            "explanation": (
                "Analogies can generate ideas, but the proposal must explain "
                "what in the target problem plays the role of what in the source "
                "domain."
            ),
            "examples": [
                "Bad: 'The model should reason like a scientist,' with no account "
                "of what corresponds to hypotheses, experiments, or evidence.",
            ],
        },
        {
            "id": "unsupported_transfer",
            "scope": "candidate",
            "weight": 3.0,
            "criterion": "Asserts the transfer will work without support from source literature.",
            "explanation": (
                "The candidate must ground transfer plausibility in what the "
                "source mechanism relies on. Ignoring an assumption that clearly "
                "fails in the target problem is a serious flaw."
            ),
            "examples": [
                "Bad: 'Tree search works in Go, so it will work for code "
                "generation' with no account of value signals or evaluable "
                "intermediate states.",
            ],
        },
        {
            "id": "prior_work_strawman",
            "scope": "candidate",
            "weight": 3.0,
            "criterion": "Misstates or underspecifies the closest in-domain prior work.",
            "explanation": (
                "A candidate should not get novelty credit by attacking a weak "
                "or inaccurate version of prior work, or by skipping an obvious "
                "nearer competitor."
            ),
            "examples": [
                "Bad: 'No prior code-RL method uses intermediate feedback' when "
                "a read paper already uses process-level reward.",
            ],
        },
        {
            "id": "self_evaluation_artifacts",
            "scope": "candidate",
            "weight": 2.0,
            "criterion": "Adds self-praise, meta-commentary, or claims the candidate satisfies the rubric.",
            "explanation": (
                "The candidate should read as a natural research intuition, not "
                "as a response trying to game a reward function."
            ),
            "examples": [
                "Bad: 'This is a strong, well-grounded intuition that clearly "
                "passes the novelty rubric.'",
            ],
        },
    ],
}


JUDGE_PROMPT_TEMPLATE = """\
You are scoring ONE research intuition written by an innovator agent against ONE rubric item.

# Target gap the intuition must attack
<<gap>>

# Papers the agent read this run (id -> one-line finding it can rely on)
<<available_papers>>

# Candidate intuition being scored
<<candidate>>

# Rubric item
<<rubric_item>>

# Instructions
- This reward is for a CLEAR, WELL-SOURCED RESEARCH INTUITION, not a full method
  writeup. Do NOT penalize missing implementation details of the research intuition, such as pseudocode, equations, architecture diagrams,
  hyperparameters, an evaluation plan, expected margins, or falsifiable
  predictions.
- The candidate must be a methodological contribution. Analysis,
  benchmark, dataset, survey, and position-paper ideas get no positive credit
  on methodological-contribution items.
- If this is a POSITIVE rubric item, return "score" as 0
  (absent/contradicted), 1 (partially present), or 2 (clearly satisfied),
  judging ONLY this item.
- If this is a NEGATIVE rubric item, return "score" as 1 if the flaw is PRESENT
  and 0 if it is absent. Judge whether the flaw is present, not whether the
  candidate is good overall.
- Ground your judgment in the read papers above. Do not credit a claim about a
  paper that its listed finding does not support.
- Return ONLY a JSON object: {"explanation": "<1-2 sentences>", "score": <int>}.
"""


def score_candidate(
    det: dict[str, bool],
    judgments: dict[str, int],
    *,
    rubric: dict = RESEARCH_INTUITION_RUBRIC,
) -> float:
    """Reward for one innovator candidate in [0, 1].

    Deterministic gate checks and gate-positive rubrics hard-reject the
    candidate. Passing candidates receive normalized weighted positive credit
    minus triggered negative penalties, clipped to [0, 1]. See
    ``ideascientist.rewards.common.score_unit``.
    """
    return score_unit(det, judgments, rubric=rubric, scope="candidate")


