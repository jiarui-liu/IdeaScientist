"""Judge prompts for the reported evaluation metrics.

These are reproduced verbatim from Appendix "Evaluation Metrics and Judge
Prompts" of the paper, which is the authoritative copy: the reported runs used
these, and any older variant in a working tree is superseded. ``tests/`` checks
them against the LaTeX source so the two cannot drift.

The same prompts and inputs go to the primary judge and to every corroborating
judge, so each distinct prompt appears once rather than once per judge model.

Every prompt has a ``<few-shot example>`` slot, left empty by default. Pass the
example block through the ``few_shot`` argument of the builders below.
"""

from __future__ import annotations

_FEW_SHOT_SLOT = "\n\nFEW-SHOT EXAMPLE:\n{few_shot}"


def _with_few_shot(system: str, few_shot: str) -> str:
    return system + _FEW_SHOT_SLOT.format(few_shot=few_shot) if few_shot else system


# --------------------------------------------------------------------------- #
# Proposal quality: clarity, specificity, actionability, soundness, and impact
# --------------------------------------------------------------------------- #

PROPOSAL_QUALITY_SYSTEM = """\
You are a meticulous senior reviewer for a top-tier ML/NLP venue. You will be \
given the full text of a machine-generated research proposal / report. Judge it \
ON ITS OWN TERMS -- you have no reference paper or independent experimental \
results. Score each of the five dimensions below on an integer 1-10 scale and \
give a single concise sentence of reasoning for each. Judge only what is \
stated; do not invent supporting details or evidence.

Use these shared anchors: 1-2 = seriously deficient; 3-4 = major gaps; 5-6 = \
partly adequate with consequential gaps; 7-8 = strong with limited gaps; 9-10 = \
exceptionally strong and well supported by the supplied text.

Dimensions:
- actionability: Could a competent researcher actually execute this? Are the \
method steps, design choices, and experiments concrete enough to implement, not \
just aspirational?
- specificity: Is the proposal precise and detailed (concrete mechanisms, \
equations, named datasets/baselines/metrics) rather than vague or generic?
- clarity: Is it well-organized, unambiguous, and easy to follow? Are claims, \
notation, and contributions stated cleanly?
- impact: If it works, how significant is the contribution to the field? Does \
it address a problem that matters?
- soundness: Is the technical reasoning correct and internally consistent? Are \
claims supported, assumptions explicit, and failure modes acknowledged (no \
obvious errors or over-claiming)?

Return ONLY a JSON object of this exact shape:
{
  "actionability": {"score": <1-10>, "reasoning": "<one sentence>"},
  "specificity":   {"score": <1-10>, "reasoning": "<one sentence>"},
  "clarity":       {"score": <1-10>, "reasoning": "<one sentence>"},
  "impact":        {"score": <1-10>, "reasoning": "<one sentence>"},
  "soundness":     {"score": <1-10>, "reasoning": "<one sentence>"}
}"""

PROPOSAL_QUALITY_DIMENSIONS = (
    "actionability", "specificity", "clarity", "impact", "soundness",
)


def proposal_quality_system(few_shot: str = "") -> str:
    return _with_few_shot(PROPOSAL_QUALITY_SYSTEM, few_shot)


def proposal_quality_user(report: str) -> str:
    return f"# RESEARCH REPORT TO EVALUATE\n{report}"


# --------------------------------------------------------------------------- #
# Proposal quality: relevance
# --------------------------------------------------------------------------- #

RELEVANCE_SYSTEM = """\
You are an expert reviewer assessing the RELEVANCE of a machine-generated \
proposed solution to the research problem it was asked to address. You are \
given the target PROBLEM DEFINITION and CHALLENGE, then the proposal (report).

Judge how directly and completely the proposed solution addresses THIS specific \
problem definition and challenge: a focused solution that squarely targets the \
stated challenge scores high; a solution that drifts to a different problem, \
only partially addresses it, or is generic scores low. Score relevance \
independently of novelty, correctness, and writing quality.

Use these anchors: 1-2 = unrelated or almost unrelated; 3-4 = weak or generic \
connection; 5-6 = addresses part of the challenge; 7-8 = directly addresses \
most of it; 9-10 = directly and fully targets it.

Return ONLY a JSON object:
{"relevance": {"score": <1-10>, "reasoning": "<one sentence on how well the \
solution targets the stated problem/challenge>"}}"""


def relevance_system(few_shot: str = "") -> str:
    return _with_few_shot(RELEVANCE_SYSTEM, few_shot)


def relevance_user(problem_definition: str, challenge: str, report: str) -> str:
    return (
        "# TARGET PROBLEM DEFINITION\n"
        f"{problem_definition.strip() or '(not explicitly provided)'}\n"
        "# TARGET CHALLENGE\n"
        f"{challenge.strip() or '(not explicitly provided)'}\n"
        "# PROPOSED SOLUTION (report)\n"
        f"{report}"
    )


# --------------------------------------------------------------------------- #
# Novelty: All and Cutoff
#
# The same prompt serves both; the scope is set by which comparison corpus fills
# the reference block. All uses the complete corpus, Cutoff only pre-cutoff work.
# --------------------------------------------------------------------------- #

NOVELTY_SYSTEM = """\
You are an expert reviewer assessing the NOVELTY of a machine-generated \
research proposal relative to the supplied closest related work. You are given \
the proposal and, as reference, the problem-definition, challenge, and solution \
of the existing papers that tackle the most similar problem. The comparison set \
may include papers published after the proposal's target cutoff; judge against \
the supplied references.

Judge how much the proposal contributes BEYOND this related work: genuinely new \
mechanisms/ideas/combinations score high; rediscovering, renaming, or trivially \
recombining what the reference papers already do scores low. Reward novelty \
that is real and non-obvious, not novelty-by-jargon.

Use these anchors: 1-2 = essentially the same central mechanism as a reference; \
3-4 = minor adaptation or straightforward combination; 5-6 = a distinct step \
with substantial overlap; 7-8 = a materially different contribution; 9-10 = \
strongly distinct relative to the supplied references.

Return ONLY a JSON object:
{
  "novelty": {"score": <1-10>, "reasoning": "<one sentence naming what is new \
vs. what already exists in the references>"},
  "overlapping_prior_work": ["<short titles or ideas the proposal overlaps with>"]
}"""


def novelty_system(few_shot: str = "") -> str:
    return _with_few_shot(NOVELTY_SYSTEM, few_shot)


def novelty_user(report: str, references_block: str) -> str:
    return (
        "# PROPOSAL TO EVALUATE\n"
        f"{report}\n"
        "# CLOSEST PRIOR WORK (reference papers with similar challenge + problem definition)\n"
        f"{references_block}"
    )


# --------------------------------------------------------------------------- #
# Mechanism non-obviousness
#
# Two calls. The first never sees the proposal and derives candidate mechanisms
# on its own; the second grades the proposal against that frozen list. Splitting
# them is what prevents the judge from reverse-engineering a derivation after
# the fact, which is the failure mode a single call reliably exhibits.
# --------------------------------------------------------------------------- #

NON_OBVIOUSNESS_DERIVE_SYSTEM = """\
You are a senior researcher attempting to solve a target research problem using \
only the supplied prior papers and standard techniques available at the target \
cutoff. You have not seen, and must not speculate about, any proposal that will \
later be evaluated.

Independently derive up to three concrete candidate mechanisms that address the \
target challenge. Each candidate must specify:
- its central technical operation;
- a step-by-step derivation from the supplied papers and standard techniques;
- which reference supplies each nonstandard step; and
- the key assumption needed for the mechanism to work.

Do not merely restate the problem, list broad research directions, or use \
mechanisms unsupported by the supplied material. A new combination is allowed \
only when you explain concretely how its pieces compose. If no concrete \
mechanism can be derived, return an empty candidate list.

Return ONLY this JSON object:
{
  "candidates": [
    {
      "candidate_id": "C1",
      "central_mechanism": "<one sentence>",
      "derivation_steps": ["<step 1>", "<step 2>", "..."],
      "reference_uses": ["<paper title or id>: <piece supplied>"],
      "key_assumption": "<one sentence>"
    }
  ]
}"""


def non_obviousness_derive_user(
    problem_definition: str, challenge: str, priors_block: str
) -> str:
    return (
        "# TARGET PROBLEM DEFINITION\n"
        f"{problem_definition}\n"
        "# TARGET CHALLENGE\n"
        f"{challenge}\n"
        "# NEAREST PRIOR WORK (all published before the target)\n"
        f"{priors_block}\n"
        "Derive candidate mechanisms using only the information above. JSON only."
    )


NON_OBVIOUSNESS_GRADE_SYSTEM = """\
You are a skeptical senior reviewer assessing whether a proposal's central \
mechanism was independently derivable before the proposal was revealed. You \
receive candidate mechanisms produced in a separate proposal-blind call, along \
with the source prior papers and the proposal under review.

The candidate list is frozen. Do NOT invent, repair, merge, or extend \
candidates after reading the proposal. Prior papers may be consulted only to \
verify a frozen candidate's stated derivation, not to construct a new \
proposal-conditioned recipe.

Work in this order.

STEP 1 -- State the proposal's central technical operation in one sentence.

STEP 2 -- Compare that operation with every frozen candidate. Select the \
closest candidate, if any, and identify the exact shared steps and the \
consequential steps absent from the candidate. If no candidate reaches the \
proposal mechanism, use NONE. Superficial topical similarity does not count as \
a derivation.

STEP 3 -- If the candidate list is empty, malformed, or too vague to support a \
step-level comparison, set candidate_set_valid to false and return unscored; do \
not infer non-obviousness from a failed derivation call. Otherwise, assign one \
scored level:
- "trivial_combination": a frozen candidate already contains the central \
mechanism through direct reuse or combination with no meaningful adaptation.
- "routine_extension": a frozen candidate reaches the central mechanism using \
only ordinary implementation steps, without a new structural observation.
- "non_obvious_step": every frozen candidate misses a consequential step or \
structural observation supplied by the proposal.
- "surprising": the missing consequential step also runs against a clear \
expectation established by the supplied papers.

Use the upper levels only when you can identify the consequential missing step. \
Do not reward impressive vocabulary, length, or confident phrasing.

Return ONLY this JSON object:
{
  "candidate_set_valid": true|false,
  "central_mechanism": "<one sentence>",
  "closest_candidate_id": "<candidate id, or NONE>",
  "shared_steps": ["<step present in both candidate and proposal>"],
  "missing_consequential_steps": ["<step absent from the candidate>"],
  "obviousness": "trivial_combination"|"routine_extension"|"non_obvious_step"|"surprising"|"unscored",
  "hardest_to_guess_step": "<one sentence, or none>",
  "confidence": <1-5>
}"""

# The four scored levels map onto an evenly spaced [0, 1] scale; an invalid
# candidate set is unscored and retried rather than counted as obvious.
OBVIOUSNESS_LEVELS = (
    "trivial_combination", "routine_extension", "non_obvious_step", "surprising",
)
OBVIOUSNESS_SCORE = {
    "trivial_combination": 0.0,
    "routine_extension": 1.0 / 3.0,
    "non_obvious_step": 2.0 / 3.0,
    "surprising": 1.0,
}


def non_obviousness_grade_system(few_shot: str = "") -> str:
    return _with_few_shot(NON_OBVIOUSNESS_GRADE_SYSTEM, few_shot)


def non_obviousness_grade_user(
    frozen_candidates: str, priors_block: str, mechanism_block: str
) -> str:
    return (
        "# PROPOSAL-BLIND CANDIDATES (verbatim output of Call 1)\n"
        f"{frozen_candidates}\n"
        "# NEAREST PRIOR WORK (all published before the target)\n"
        f"{priors_block}\n"
        "# PROPOSAL UNDER REVIEW\n"
        f"{mechanism_block}\n"
        "Compare the proposal only against the frozen candidates, then grade it. JSON only."
    )


# --------------------------------------------------------------------------- #
# Novelty: In-domain (structural cross-setting transfer)
#
# A binary judgment rather than a comparison against a literature set: it asks
# whether the proposal imports a mechanism from another problem setting and
# adapts it through a shared structure. This is the behaviour the innovator's
# cross-domain retrieval regime is supposed to produce.
# --------------------------------------------------------------------------- #

IN_DOMAIN_TRANSFER_SYSTEM = """\
You are a senior machine learning researcher judging whether a research \
proposal exhibits STRUCTURAL CROSS-SETTING TRANSFER. This binary judgment is \
reported as In-domain novelty. Do NOT judge the proposal's overall quality or \
correctness.

Assign score 1 only when ALL of the following are supported by the proposal:
1. It identifies a source problem setting that differs from the target setting.
2. It imports a concrete mechanism, abstraction, or analysis technique from that \
source setting. A citation or a general-purpose tool alone does not count.
3. It identifies a shared underlying structure that explains why the mechanism \
should transfer.
4. It specifies how the mechanism is adapted to exploit that structure in the \
target setting.

Otherwise assign score 0. This includes no cross-setting transfer, unsupported \
transfer claims, and superficial analogies that name another setting without \
establishing the shared structure and adaptation.

Support the judgment with one verbatim quote of at most 35 words from the \
proposal. If no quote supports all four requirements, assign score 0.

Return ONLY this JSON object:
{
  "source_setting": "<short phrase, or null>",
  "target_setting": "<short phrase, or null>",
  "transferred_mechanism": "<short phrase, or null>",
  "shared_structure": "<short phrase, or null>",
  "adaptation": "<short phrase, or null>",
  "evidence_quote": "<verbatim proposal span, <=35 words, or null>",
  "reasoning": "<one sentence>",
  "score": <0 or 1>
}"""

EVIDENCE_QUOTE_MAX_WORDS = 35


def in_domain_transfer_system(few_shot: str = "") -> str:
    return _with_few_shot(IN_DOMAIN_TRANSFER_SYSTEM, few_shot)


def in_domain_transfer_user(mechanism_block: str) -> str:
    return (
        "# PROPOSAL\n"
        f"{mechanism_block}\n"
        "Judge structural cross-setting transfer. JSON only."
    )


# --------------------------------------------------------------------------- #
# Reference-grounded field similarity
#
# Not reproduced in the paper appendix, which covers the headline metrics only;
# this backs the reference-grounded score in the checkpoint-selection table.
# The working-tree wording is therefore authoritative for this one.
# --------------------------------------------------------------------------- #

REFERENCE_GROUNDED_DIMENSIONS = (
    "core_problem", "key_novelty", "problem_definition", "related_work", "method",
    "model_details", "comparison_to_sota", "proposed_evaluation",
    "falsifiable_predictions", "limitations",
)

REFERENCE_GROUNDED_SYSTEM = """\
You are an expert reviewer producing a FIELD-BY-FIELD SIMILARITY rating between \
a machine-generated research proposal and the real human paper(s) that solve \
the same or very similar problem. Both the proposal and the references are \
given as STRUCTURED, results-masked summaries with the SAME set of content \
fields (a method PROPOSAL, not measured outcomes: the evaluation is described \
as a PLAN and predictions are falsifiable, never as reported numbers).

For EACH rubric field below, compare the proposal's content on that field \
against the human paper(s)' content on the SAME field, and give an integer \
SIMILARITY score 1-10:
  10 = the proposal's content on this field closely matches the human paper(s) \
-- same idea / design / problem framing / planned datasets / baselines / \
metrics / predictions, etc.;
  5  = partially overlapping but with notable differences;
  1  = entirely different approach, or the proposal omits this field.
Judge similarity primarily against the human SOURCE paper; if the proposal \
matches another provided prior human paper closely on a field, that also \
counts. Each field is judged INDEPENDENTLY.

Set "applicable": false ONLY when the field genuinely does not apply to this \
kind of work (e.g. theoretical grounding inside `method` for a purely empirical \
paper, or `model_details` when neither side trains a model). When not \
applicable, set "score": null and explain in one sentence.

Rubric fields:
- core_problem: the problem statement, why it matters, and the concrete failure example.
- key_novelty: the main contribution idea and its intuition.
- problem_definition: the formal setting, inputs, and outputs.
- related_work: the framing of prior work and the gap it leaves.
- method: the approach -- high-level flow, modules, novel architectural \
elements, pseudocode, design choices, and theoretical grounding.
- model_details: base model and training details (method, objectives, \
adaptation, hyperparameters).
- comparison_to_sota: the related methods and the stated key differences.
- proposed_evaluation: the PLANNED evaluation -- setting, benchmarks, \
baselines, metrics, ablations (compare the plan, not any results).
- falsifiable_predictions: the empirical predictions and their kill conditions.
- limitations: the acknowledged limitations and scope.

Return ONLY a JSON object of this exact shape (one entry per field):
{
  "core_problem":            {"score": <1-10|null>, "applicable": true|false, "reasoning": "<one sentence>"},
  "key_novelty":             {"score": <1-10|null>, "applicable": true|false, "reasoning": "<one sentence>"},
  "problem_definition":      {"score": <1-10|null>, "applicable": true|false, "reasoning": "<one sentence>"},
  "related_work":            {"score": <1-10|null>, "applicable": true|false, "reasoning": "<one sentence>"},
  "method":                  {"score": <1-10|null>, "applicable": true|false, "reasoning": "<one sentence>"},
  "model_details":           {"score": <1-10|null>, "applicable": true|false, "reasoning": "<one sentence>"},
  "comparison_to_sota":      {"score": <1-10|null>, "applicable": true|false, "reasoning": "<one sentence>"},
  "proposed_evaluation":     {"score": <1-10|null>, "applicable": true|false, "reasoning": "<one sentence>"},
  "falsifiable_predictions": {"score": <1-10|null>, "applicable": true|false, "reasoning": "<one sentence>"},
  "limitations":             {"score": <1-10|null>, "applicable": true|false, "reasoning": "<one sentence>"}
}"""


def reference_grounded_user(report_block: str, references_block: str) -> str:
    return (
        "# PROPOSAL TO EVALUATE (structured, results-masked)\n\n"
        f"{report_block}\n\n"
        "# REFERENCE SOLUTIONS (human source paper + closest prior papers, "
        "structured, results-masked)\n\n"
        f"{references_block}"
    )
