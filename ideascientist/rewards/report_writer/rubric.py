"""Rubric-based reward for training the ``report_writer`` sub-agent.

The ``report_writer`` first-tier sub-agent reads ``gaps.md`` / ``candidates.md`` /
``scores.md``, grounds every claim in cited prior work, and emits ONE structured
JSON object to ``report.json`` following the results-masked schema
(shared with the offline metadata extractor via
``ideascientist.vault.schema.RESULTS_MASKED_SCHEMA``). It writes a solution PROPOSAL —
the experiments have NOT been run — so measured results, fabricated numbers, and
uncited (non-DB) papers are the signature hacks. This module defines the rubric
that turns a proposal into a scalar reward for on-policy RL, plus the
deterministic verifier checks, the judge prompt, and the aggregation function.

Grounded in:
- ``docs/literature_reviews/what_is_a_good_research/solution_proposal.md``
  (what makes a clear, credible, falsifiable, well-positioned proposal).
- ``docs/literature_reviews/rubric_based_rl.md`` (how to design rubric rewards
  that resist reward hacking).

Design choices mirror the gap_finder / innovator rubrics (same survey findings):

1. **Deterministic verifier grounding first.** Because ``report.json`` is a
   single structured object, the most hackable requirements are code-checkable
   BEFORE the judge is consulted: valid JSON, required fields present, EVERY
   cited ``[id]`` is a real DB paper id (RLCF/Rubicon verifier programs +
   DR Tulu citation faithfulness), and every cited id is grounded (read via
   read_paper OR present in the seeded gaps.md / candidates.md inputs). These are
   gates: any failure zeroes the reward.
2. **Core-coherence positives.** Four high-weight positive rubrics encode "is
   this even a coherent, testable proposal": one central claim, problem
   separated from solution, faithful development of the assigned gap/intuition,
   and evaluation aligned to the claim. They are scored 0/1/2 and weighted like
   any other positive (NOT vetoes) — a proposal that is incoherent on these
   simply loses their large share of the positive credit. Hard rejection
   (reward 0) comes only from the deterministic gates in #1.
3. **Close the presence/absence asymmetry without duplicating criteria.**
   Every negative is checked to be ORTHOGONAL to the positives: a negative
   survives only if a proposal can satisfy every positive field-quality
   criterion and STILL commit it. Negatives that were merely the inverse of a
   positive are folded into that positive as graded credit — the field-quality
   positives absorb missing baselines, non-reproducibility, strawman related
   work, mechanism-only method text, and harmless-only limitations; the
   core-coherence positives absorb no-clear-claim / solution-as-problem /
   gap-or-intuition drift / evaluation-mismatch. Reporting expected outcomes as
   measured results (qualitative OR numeric) is caught by the graded
   ``predictions_as_results`` negative. Only three orthogonal negatives remain
   (predictions_as_results, vague_terms, self_evaluation_artifacts).
4. **Per-field credit.** A proposal is scored as one artifact, but each positive
   maps to a results-masked ``schema_key`` so a strong method section is not zeroed by
   a weak limitations section (SRaR/DR Tulu per-item style).
5. **Anti-parroting / anti-artifact.** The ``self_evaluation_artifacts``
   negative + the ``json_parses`` deterministic gate suppress self-praise /
   rubric-gaming meta-commentary.
6. **Normalized weighted aggregation.** HealthBench: score =
   achieved_points / total_positive_points, minus triggered negatives, clipped
   to [0, 1].
"""

from __future__ import annotations

from ideascientist.rewards.common import TIER_WEIGHT, score_unit


SOLUTION_PROPOSAL_RUBRIC: dict = {
    "name": "solution_proposal",
    "source_documents": [
        "docs/literature_reviews/what_is_a_good_research/solution_proposal.md",
        "docs/literature_reviews/rubric_based_rl.md",
    ],
    # Unit of scoring: the whole report.json proposal (one artifact). Positives
    # carry a ``schema_key`` naming the results-masked field they judge, so per-field
    # credit is preserved.
    "unit_of_scoring": "proposal",
    "required_schema_fields": [
        "title",
        "topic_relevance",
        "one_sentence_thesis",
        "core_problem",
        "key_novelty",
        "problem_definition",
        "key_terms",
        "related_work",
        "method",
        "model_details",
        "comparison_to_sota",
        "proposed_evaluation",
        "falsifiable_predictions",
        "limitations",
    ],
    "scoring_hint": (
        "Score each positive item 0 = absent or contradicted, 1 = partially "
        "present, 2 = clearly satisfied. Score each negative item 0 = flaw "
        "absent, 1 = flaw present (a negative scored 1 subtracts its weight). "
        "The proposal is HARD-REJECTED (reward 0) only if a deterministic gate "
        "check in VERIFIABLE_CHECKS fails; positive rubrics (including the "
        "core-coherence ones) are graded 0/1/2 and weighted, never vetoing. "
        "This is a research PROPOSAL (the experiments have NOT been "
        "run): do NOT penalize the absence "
        "of measured results; DO penalize presenting expected outcomes as "
        "measured, and reward clearly separated predictions with kill "
        "conditions."
    ),
    # ------------------------------------------------------------------ #
    # Deterministic checks (RLCF/Rubicon verifier-programs + DR Tulu citation
    # faithfulness). Computed in code by the RL environment BEFORE the LLM judge;
    # all are gates that hard-reject on failure. report.json being a structured
    # object makes these stronger than for the markdown-emitting sub-agents: the
    # citation contract from harness/tools.py (bare [id] citations, DB-resolvable
    # and read-grounded) is fully verifiable here.
    # ------------------------------------------------------------------ #
    "verifiable_checks": [
        {
            "id": "json_parses",
            "gate": True,
            "check": (
                "report.json parses as EXACTLY one valid JSON object — no "
                "markdown code fences, no prose before or after."
            ),
            "rationale": (
                "STEP 2/3 of the report_writer prompt require a single valid "
                "JSON object. An unparseable file cannot be scored."
            ),
        },
        {
            "id": "schema_valid",
            "gate": True,
            "check": (
                "EVERY field in required_schema_fields is present and non-empty: "
                "title, topic_relevance, one_sentence_thesis, core_problem, "
                "key_novelty, problem_definition, key_terms, related_work, "
                "method, model_details, comparison_to_sota, proposed_evaluation, "
                "falsifiable_predictions, limitations. A missing or "
                "empty field fails the gate."
            ),
            "rationale": (
                "AdvancedIF completeness shaping: a scorable solution proposal "
                "must fill every results-masked field. A proposal missing any field "
                "is incomplete and hard-rejected."
            ),
        },
        {
            "id": "cited_ids_resolve",
            "gate": True,
            "check": (
                "EVERY cited paper id — every bare [id] used inline in any string "
                "value and every related_work[].citations id — resolves to a real "
                "paper in our DB (get_paper_biblio succeeds on it as a local paper "
                "id). No arxiv-id-only ids, no descriptive slugs, no fabricated "
                "ids — matching the CITATION RULE in harness/tools.py."
            ),
            "rationale": (
                "Directly enforces the DB-only citation contract and kills the "
                "fabricated / external-uncited-paper hack (DR Tulu). A citation "
                "the corpus cannot verify is worthless and gameable."
            ),
        },
        {
            "id": "cited_ids_were_read",
            "gate": True,
            "check": (
                "Every cited paper id is grounded: it appears in this run's "
                "read_paper digests (papers/<id>.md) OR is already cited in the "
                "seeded gaps.md / candidates.md inputs the report_writer was handed. "
                "An id that is neither read nor present in those inputs fails."
            ),
            "rationale": (
                "The report_writer must ground cited prior work in paper reading, "
                "not titles or snippets — but papers already surfaced in its "
                "gaps.md / candidates.md inputs are pre-grounded context and need no "
                "re-read (mirrors innovator's relaxed source_ids_were_read)."
            ),
        },
    ],
    "positive_rubrics": [
        # ---- Core-coherence positives: "is this a coherent, testable proposal" (graded, high-weight, NOT vetoes) ----
        {
            "id": "one_central_claim",
            "gate": False,
            "scope": "proposal",
            "schema_key": "title,one_sentence_thesis",
            "tier": "important",
            "weight": TIER_WEIGHT["important"],
            "criterion": "Communicates ONE central research idea in a plain title + one-sentence thesis, without unsupported performance claims.",
            "explanation": (
                "The proposal must reduce to a clean thesis: propose X to address "
                "Y because current approaches fail for Z. The title and "
                "one_sentence_thesis describe the contribution in plain language. "
                "A method name is acceptable only when it aids memory, not when it "
                "hides what the method does or implies unearned results. Score 0 "
                "when the proposal is a bundle of loosely connected contributions "
                "with no single idea, or the title is a branded, SOTA-implying "
                "label. Judge ONLY title+thesis clarity and single-idea focus — "
                "the novelty CONTENT is judged separately by "
                "novelty_as_reusable_insight."
            ),
            "examples": [
                "Good: 'Versioned Assistant Memory: testing whether models "
                "recover from user corrections.'",
                "Weak: 'MemBoost: A Revolutionary SOTA Memory Optimizer.'",
            ],
        },
        {
            "id": "faithful_to_gap_and_intuition",
            "gate": False,
            "scope": "proposal",
            "schema_key": "title,one_sentence_thesis,core_problem,key_novelty,method,proposed_evaluation",
            "tier": "important",
            "weight": TIER_WEIGHT["important"],
            "criterion": "Addresses the assigned research gap and develops the chosen intuition instead of drifting to an unrelated solution.",
            "explanation": (
                "The report_writer is downstream of gap_finder and innovator: the "
                "proposal should solve the provided gap using the selected "
                "intuition as its mechanism or organizing idea. Refining, "
                "narrowing, or making the intuition more concrete is good; "
                "silently replacing it with a different problem, different "
                "failure mode, or unrelated method scores low even if the new "
                "proposal is internally coherent."
            ),
            "examples": [
                "Good: 'For a gap about stale assistant memory after user "
                "corrections, develops the chosen versioned-memory intuition into "
                "a method and evaluation for correction recovery.'",
                "Weak: 'Given that same gap and intuition, instead proposes a "
                "generic long-context summarizer for conversation history.'",
            ],
        },
        {
            "id": "separate_problem_from_solution",
            "gate": False,
            "scope": "proposal",
            "schema_key": "core_problem",
            "tier": "important",
            "weight": TIER_WEIGHT["important"],
            "criterion": "Defines the problem (gap between current and desired state) BEFORE selling the method.",
            "explanation": (
                "The problem statement specifies what fails and why, and does "
                "NOT smuggle in the proposed solution as if it were part of the "
                "problem definition. Score 0 when the 'problem' is really a "
                "description of the method (e.g. 'the problem is the lack of our "
                "versioned memory module')."
            ),
            "examples": [
                "Good: 'Assistants often preserve outdated preferences after a "
                "user correction.'",
                "Weak: 'The problem is that assistants lack our versioned memory "
                "module.'",
            ],
        },
        {
            "id": "claim_aligned_evaluation",
            "gate": False,
            "scope": "proposal",
            "schema_key": "proposed_evaluation",
            "tier": "important",
            "weight": TIER_WEIGHT["important"],
            "criterion": "Designs proposed_evaluation to test the CENTRAL claim directly.",
            "explanation": (
                "A claim about robustness, generalization, efficiency, safety, or "
                "user value needs an evaluation that measures THAT property; "
                "benchmark choices are justified as research-design decisions. "
                "Score 0 when the evaluation measures something unrelated to the "
                "central claim. Judge alignment of the PROPOSED evaluation, not "
                "measured results."
            ),
            "examples": [
                "Good: 'Because the claim is correction robustness, report "
                "failure rate after contradicted memory updates, not only "
                "average QA accuracy.'",
                "Weak: 'Claims lower hallucination under changing evidence but "
                "evaluates only static multiple-choice accuracy.'",
            ],
        },
        # ---- Soft quality positives (per results-masked field) ----------------- #
        {
            "id": "novelty_as_reusable_insight",
            "gate": False,
            "scope": "proposal",
            "schema_key": "key_novelty",
            "tier": "important",
            "weight": TIER_WEIGHT["important"],
            "criterion": "States the reusable insight and why prior work does not already cover it.",
            "explanation": (
                "Novelty is a checkable claim about the contribution, not a "
                "label. Another researcher should understand what idea could "
                "transfer to related problems, and why named prior work does not "
                "already do it (not merely 'novel' / 'first')."
            ),
            "examples": [
                "Good: 'The reusable insight is to evaluate memory by "
                "state-transition correctness rather than static recall.'",
            ],
        },
        {
            "id": "specific_motivation_and_hardness",
            "gate": False,
            "scope": "proposal",
            "schema_key": "core_problem",
            "tier": "important",
            "weight": TIER_WEIGHT["important"],
            "criterion": "Explains why the problem matters and why naive/current methods are insufficient.",
            "explanation": (
                "Motivation answers who cares and why the issue is hard with "
                "concrete stakes, not only benchmark-score language or generic "
                "'this topic is important' claims."
            ),
            "examples": [
                "Good: 'A wrong persistent preference repeatedly affects future "
                "assistant actions, so one-off answer accuracy misses a "
                "deployment-relevant failure.'",
            ],
        },
        {
            "id": "concrete_example",
            "gate": False,
            "scope": "proposal",
            "schema_key": "core_problem",
            "tier": "optional",
            "weight": TIER_WEIGHT["optional"],
            "criterion": "Includes a tangible input-output example where current methods fail.",
            "explanation": (
                "A running example makes the problem concrete and lets a reader "
                "check whether method and evaluation address the same claim."
            ),
            "examples": [
                "Good: 'User: stop booking morning flights. Later the assistant "
                "recommends a 7 a.m. flight from a stale preference.'",
            ],
        },
        {
            "id": "formal_scope",
            "gate": False,
            "scope": "proposal",
            "schema_key": "problem_definition",
            "tier": "important",
            "weight": TIER_WEIGHT["important"],
            "criterion": "Defines task, inputs, outputs, evaluation target, and scoping assumptions.",
            "explanation": (
                "The formal problem definition is narrow enough to test "
                "rigorously and states what domains, datasets, users, or "
                "settings are inside vs. outside the claim."
            ),
            "examples": [
                "Good: 'Input: dialogue history + memory-store versions; output: "
                "action + cited memory version; metric: correction-regression "
                "rate.'",
            ],
        },
        {
            "id": "organized_related_work",
            "gate": False,
            "scope": "proposal",
            "schema_key": "related_work",
            "tier": "important",
            "weight": TIER_WEIGHT["important"],
            "criterion": "Compares prior work by theme/mechanism/question, accurately and generously.",
            "explanation": (
                "Related work explains what each line accomplishes, what gap "
                "remains, and how the proposal differs — representing competitors "
                "fairly, grounded in the cited prior work."
            ),
            "examples": [
                "Good: 'Memory benchmarks test static recall; agent benchmarks "
                "test planning; neither isolates correction after state change.'",
            ],
        },
        {
            "id": "fair_sota_comparison",
            "gate": False,
            "scope": "proposal",
            "schema_key": "comparison_to_sota",
            "tier": "important",
            "weight": TIER_WEIGHT["important"],
            "criterion": "Plans relevant, strong, fairly tuned baselines and attributes gains correctly.",
            "explanation": (
                "Compares against methods that actually test the claim under "
                "comparable data, compute, tuning, and protocols. Score low when "
                "the proposal omits an obvious strong baseline a reviewer would "
                "expect it to beat or distinguish itself from."
            ),
            "examples": [
                "Good: 'Compare latest-profile retrieval, full-history "
                "retrieval, and versioned retrieval at equal context budget.'",
            ],
        },
        {
            "id": "top_down_method",
            "gate": False,
            "scope": "proposal",
            "schema_key": "method",
            "tier": "important",
            "weight": TIER_WEIGHT["important"],
            "criterion": "Explains the method top-down with justified design choices.",
            "explanation": (
                "The main contribution appears early; modules, data flow, "
                "objectives, and interactions are clear; each design choice has a "
                "reason rather than being merely listed."
            ),
            "examples": [
                "Good: 'Three stages: detect correction, write a versioned diff, "
                "retrieve the active version — versioning lets evaluation "
                "distinguish stale from current memory.'",
            ],
        },
        {
            "id": "reproducible_model_details",
            "gate": False,
            "scope": "proposal",
            "schema_key": "model_details",
            "tier": "optional",
            "weight": TIER_WEIGHT["optional"],
            "criterion": "Specifies implementation details needed to rebuild or audit the method.",
            "explanation": (
                "If a model is used, state base model, architecture/adaptation, "
                "objective, data, hyperparameters, and selection plan. Genuinely "
                "open details may be marked as open, but the core method must be "
                "concrete enough to execute; a method too vague to rebuild even in "
                "principle scores low here."
            ),
            "examples": [
                "Good: 'Llama-3.1-8B-Instruct + LoRA rank 16, train on 20k "
                "correction dialogues, tune thresholds on a held-out split.'",
            ],
        },
        {
            "id": "statistical_and_ablation_plan",
            "gate": False,
            "scope": "proposal",
            "schema_key": "proposed_evaluation",
            "tier": "optional",
            "weight": TIER_WEIGHT["optional"],
            "criterion": "Plans uncertainty estimates, comparable runs, and ablations that isolate claims.",
            "explanation": (
                "Results are planned as distributions across seeds/splits where "
                "relevant, and ablations identify which design choice causes any "
                "gain. (Plans, not measured numbers.)"
            ),
            "examples": [
                "Good: 'Five seeds, bootstrap CIs, ablate correction detection, "
                "version storage, and retrieval policy separately.'",
            ],
        },
        {
            "id": "falsifiable_predictions",
            "gate": False,
            "scope": "proposal",
            "schema_key": "falsifiable_predictions",
            "tier": "important",
            "weight": TIER_WEIGHT["important"],
            "criterion": "States risky predictions and kill conditions for major design pillars.",
            "explanation": (
                "Distinguishes expected outcomes from measured results, names "
                "what observation would count AGAINST the mechanism, and "
                "considers alternative explanations."
            ),
            "examples": [
                "Good: 'If gains vanish when context length is matched, the "
                "versioning claim is unsupported and improvement is likely just "
                "extra context.'",
            ],
        },
        {
            "id": "real_limitations",
            "gate": False,
            "scope": "proposal",
            "schema_key": "limitations",
            "tier": "optional",
            "weight": TIER_WEIGHT["optional"],
            "criterion": "States assumptions, scope bounds, and realistic failure modes.",
            "explanation": (
                "Limitations include weaknesses a reviewer could reasonably "
                "identify and how evaluation will expose them — not only harmless "
                "caveats like 'more compute would help'."
            ),
            "examples": [
                "Good: 'Synthetic corrections may understate real ambiguity; the "
                "pilot inspects ambiguous cases and reports separate error "
                "categories.'",
            ],
        },
        {
            "id": "citation_integrity",
            "gate": False,
            "scope": "proposal",
            "schema_key": "related_work,comparison_to_sota",
            "tier": "important",
            "weight": TIER_WEIGHT["important"],
            "criterion": "Load-bearing claims cite the paper that ACTUALLY supports them (right paper, right contribution).",
            "explanation": (
                "Judge citation ACCURACY: the cited paper genuinely supports the "
                "specific claim it is attached to, primary sources are used for "
                "methods/benchmarks/datasets, and one citation is not stretched to "
                "cover two distinct claims. Ground your judgment on the read "
                "papers supplied to you."
            ),
            "examples": [
                "Good: 'Cites the benchmark paper for the protocol and the model "
                "paper for the architecture, not one citation for both.'",
            ],
        },
    ],
    "negative_rubrics": [
        # Only failure modes that are NOT the mere inverse of a positive rubric
        # survive here (design principle #3): a proposal can satisfy every
        # positive field-quality criterion and STILL commit one of these, so
        # they add orthogonal signal rather than double-penalizing a low positive.
        {
            "id": "predictions_as_results",
            "scope": "proposal",
            "weight": 2.0,
            "criterion": "Anywhere in the proposal, presents expected outcomes as measured results — qualitative ('outperforms/solves') OR numeric ('achieves 92% accuracy').",
            "explanation": (
                "This is a PROPOSAL — no experiments have been run, so the report "
                "must not contain results. The flaw is stating expected outcomes as "
                "if they were measured findings — asserting the method "
                "'outperforms', 'solves', or 'eliminates' something, OR reporting a "
                "specific achieved metric ('reaches 92.3 F1', 'improves accuracy by "
                "15%'), as though experiments had already been run. Read the whole "
                "document's tense and stance, not one field. CRUCIAL DISTINCTION: a "
                "falsifiable PREDICTION or target stated as a hypothesis ('we "
                "predict/expect accuracy >= 90%', 'if H holds, EV gap < 10%') is "
                "CORRECT and must NOT fire — the falsifiable_predictions field is "
                "meant to hold quantitative predictions. Only fire when a number or "
                "outcome is asserted as an ACHIEVED result rather than a "
                "to-be-tested prediction."
            ),
            "examples": [
                "Bad: 'Our method eliminates stale-preference errors' stated as "
                "fact before any experiment.",
                "Bad: 'The approach achieves 92.3 F1 on the benchmark' (asserted "
                "achieved result in a not-yet-run proposal).",
                "OK: 'We predict the method will reach >= 90% accuracy' "
                "(a falsifiable prediction / target, not a result).",
            ],
        },
        {
            "id": "vague_terms",
            "scope": "proposal",
            "weight": 2.0,
            "criterion": "Anywhere in the proposal, uses terms like better/robust/efficient/aligned without operational definitions.",
            "explanation": (
                "Scan the WHOLE document for load-bearing terms left without an "
                "operational definition, so a claim stays untestable. This fires "
                "even when the formal problem_definition itself is well specified: "
                "the flaw is using words like better/robust/efficient/aligned in "
                "claims elsewhere without tying them to a task, metric, "
                "assumption, or example."
            ),
            "examples": [
                "Bad: 'The method improves robustness and usefulness.'",
            ],
        },
        {
            "id": "self_evaluation_artifacts",
            "scope": "proposal",
            "weight": 2.0,
            "criterion": "Adds self-praise, meta-commentary, or claims the proposal satisfies the rubric.",
            "explanation": (
                "The flaw is self-evaluation text: self-praise like 'this is a "
                "strong, well-grounded proposal', or any reference to being scored "
                "or to satisfying the criteria. The proposal should read as a "
                "natural research document, not a pitch about its own quality."
            ),
            "examples": [
                "Bad: 'This proposal clearly passes the novelty and evaluation "
                "criteria.'",
            ],
        },
    ],
}


# --------------------------------------------------------------------------- #
# Judge prompt (HealthBench GRADER_TEMPLATE style): one proposal, one rubric
# item, with the upstream framing (problem -> assigned gap -> chosen intuition)
# and the read papers supplied for grounding. The RL environment fills the
# placeholders and calls the judge once per rubric item, then feeds the parsed
# scores to ``score_proposal``.
# --------------------------------------------------------------------------- #
JUDGE_PROMPT_TEMPLATE = """\
You are scoring ONE research solution proposal written by a report-writer agent against ONE rubric item.

# Research problem the proposal must address
<<problem>>

# Research gap the proposal was assigned to close (from the upstream gap_finder)
<<gap>>

# Research intuition the proposal was told to develop (from the upstream innovator)
<<intuition>>

# Papers the agent read this run (id -> one-line finding it can rely on)
<<available_papers>>

# The proposal being scored (report.json)
<<proposal>>

# Rubric item
<<rubric_item>>

# Instructions
- This is a research PROPOSAL: the experiments have NOT been run. Do NOT
  penalize the absence of measured results; DO penalize expected outcomes
  presented as measured, and reward clearly separated predictions with kill
  conditions.
- The proposal should faithfully develop the assigned gap and chosen intuition
  above: a proposal that ignores them or silently drifts to an unrelated idea
  does NOT deserve credit for the faithful-to-gap item and should also lose
  credit on central-claim or evaluation-alignment items when the drift affects
  those criteria. (Refining or sharpening the intuition is fine; abandoning it
  is not.)
- If a rubric item names a schema_key, judge that field of the proposal (and any
  inline claims that depend on it), not the whole document.
- If this is a POSITIVE rubric item, return "score" as 0 (absent/contradicted),
  1 (partially present), or 2 (clearly satisfied), judging ONLY this item.
- If this is a NEGATIVE rubric item, return "score" as 1 if the flaw is PRESENT
  and 0 if it is absent. Judge whether the flaw is present, not whether the
  proposal is good overall.
- Ground your judgment in the read papers above. Do not credit (or fault) a claim
  about a paper that its listed finding does not support.
- Return ONLY a JSON object: {"explanation": "<1-2 sentences>", "score": <int>}.
"""


# --------------------------------------------------------------------------- #
# Reward aggregation (pure, dependency-free, testable). The environment supplies:
#   det:        {check_id: bool}   results of VERIFIABLE_CHECKS for the proposal
#   judgments:  {rubric_id: int}   positive items 0/1/2, negative items 0/1
# and receives a scalar reward in [0, 1].
# --------------------------------------------------------------------------- #


def score_proposal(
    det: dict[str, bool],
    judgments: dict[str, int],
    *,
    rubric: dict = SOLUTION_PROPOSAL_RUBRIC,
) -> float:
    """Reward for one report_writer proposal in [0, 1].

    Hard-rejected (reward 0) if any deterministic gate in VERIFIABLE_CHECKS
    fails. This rubric defines no gate positives, so all positives are graded
    0/1/2, weighted, summed, and reduced by triggered negative penalties, then
    normalized to [0, 1]. See ``ideascientist.rewards.common.score_unit``.
    """
    return score_unit(det, judgments, rubric=rubric, scope="proposal")
