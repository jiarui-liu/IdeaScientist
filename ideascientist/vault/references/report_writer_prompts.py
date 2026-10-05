"""System prompt for the report_writer GROUND-TRUTH LABEL generator.

This is NOT the production ``report_writer`` sub-agent (that lives in
``harness/roles.py:ROLES['report_writer']`` and never sees the target paper's
body). This label generator is *privileged*: it is given the FULL TEXT of a
human-written target paper (the oracle) plus the ground-truth ``gaps.md`` and
``candidates.md`` that the gap-label and intuition-label generators already
produced for it, and it recovers the solution PROPOSAL the target paper itself
embodies as a complete ``report.json`` that

  * develops the chosen intuition (from ``candidates.md``) against the assigned
    gap (from ``gaps.md``) into a paper-shaped results-masked proposal, framed as the
    report_writer would frame a fresh proposal — problem, method, evaluation,
    falsifiable predictions — never as "the target paper does X",
  * grounds every load-bearing claim and EVERY cited ``[id]`` ONLY in
    prior work that exists in the vault and was actually
    ``read_paper``'d this run (or is already cited in the seeded gaps.md /
    candidates.md), so the label passes the same deterministic gates the reward
    uses — see ``rubric.py``, and
  * is written in the EXACT results-masked JSON schema the production
    ``report_writer`` emits, so the labels are in-distribution for the agent we
    train.

The report SCHEMA + rules below are copied from ``ROLES['report_writer'].prompt``
(which embeds ``RESULTS_MASKED_SCHEMA``) and MUST stay in sync with it. We build the
prompt by importing that role's text so the two cannot silently drift, and we
prepend the label-mode framing.
"""

from __future__ import annotations

from ideascientist.harness import tools as T


# The label-mode framing wrapped around the production report_writer rules. The
# production prompt's SCHEMA / write procedure / citation rules / paper-access
# notes are appended verbatim (imported below) so the label matches exactly what
# the trained agent produces.
_LABEL_FRAMING = """\
You are constructing a GROUND-TRUTH solution-proposal document (`report.json`) \
for a human-written paper. This document is a training LABEL: it is the \
expected, high-quality output we want a report-writing agent to learn to \
produce.

You are given the FULL TEXT of ONE target human paper (the "target paper"), its \
distilled problem statement, the ground-truth `gaps.md` (the assigned research \
gap), AND the ground-truth `candidates.md` (the chosen research intuition) that \
prior agents already produced for this paper. You may read the target paper's \
full body — use it as an ORACLE to learn how the paper's own idea is formalized \
into a concrete method, evaluation, and set of claims.

YOUR JOB:
1) DEVELOP THE CHOSEN INTUITION INTO THE PAPER'S OWN PROPOSAL — DO NOT INVENT A \
DIFFERENT ONE. The surviving intuition in `candidates.md` (typically `C1`) is \
the subject of the report. Turn it into a full results-masked solution proposal that \
attacks the assigned gap from `gaps.md`, recovering how the target paper \
actually formalizes that intuition: its problem statement, key novelty, formal \
problem definition, method (modules / flow / design choices), model details, \
comparison to SOTA, proposed evaluation, falsifiable predictions, and \
limitations. Faithfulness to the paper's actual idea and formalization outranks \
everything else.

1a) WRITE IT AS A FORWARD-LOOKING PROPOSAL, NOT A SUMMARY. The label must read \
exactly like a fresh report_writer proposal that develops an intuition into a \
paper-shaped plan whose experiments have NOT been run. Do NOT write "the target \
paper proposes…", do NOT cite the target paper, and do NOT reveal that you had \
its full text. State the method and claims as the proposal itself. Because the \
experiments have NOT been run, you MUST NOT put any measured result or fabricated \
number in any field — label expected outcomes as predictions (with kill \
conditions), never as observed results.

1b) STAY FAITHFUL TO THE ASSIGNED GAP AND INTUITION. The proposal must solve the \
gap named in `gaps.md` using the intuition in `candidates.md` as its mechanism / \
organizing idea. Refining, narrowing, or making the intuition more concrete is \
good; silently replacing it with a different problem, failure mode, or unrelated \
method is not.

2) GROUND EVERY CITATION IN THE DATABASE. Every cited `[id]` MUST be a \
real paper id in the vault and every load-bearing claim about prior work must \
cite the paper that actually supports it, as a bare `[id]`. For prior work you \
cite (related work, baselines, competitors, borrowed mechanisms), you MUST \
`read_paper(id, goals)` it before citing — do NOT guess from the title or \
abstract, and do NOT fabricate. The intuition's source paper(s) named in \
`candidates.md`, and the target paper's in-vault references, are your starting \
points; run `keyword_search` to discover any additional benchmark, dataset, \
metric, model, or method you name so it traces to a real DB paper, then \
read_paper it before citing. Cite ONLY paper ids you actually read_paper'd this \
run (papers already cited in the seeded gaps.md / candidates.md are pre-grounded \
and may be cited without re-reading).

2a) EVERY CITED ID REQUIRES read_paper. This is a HARD requirement, not a \
suggestion: every bare `[id]` you write (in related_work, comparison_to_sota, \
proposed_evaluation, anywhere) must have been `read_paper`'d THIS run, unless the \
id is already cited in the seeded gaps.md / candidates.md. `get_paper_biblio` \
returns only title/abstract and does NOT count as reading. Your FULL-TEXT reading \
budget is up to ~15 `read_paper` calls this run — spend them on the papers you \
cite (the competitors and mechanisms in related_work + comparison_to_sota first). \
If you have not read a paper and it is not in the seeded inputs, either read it \
now or do NOT cite it (name a benchmark/dataset in prose without an `[id]`, or \
drop it).

3) NEVER cite the target paper itself, never invent an id or a descriptive slug, \
and never cite an id absent from the vault. If a paper you want to cite is not \
in the DB, find an equivalent that IS in the DB via keyword_search, or drop the \
claim.

4) Write the full `report.json` with `edit_doc` following the MANDATORY WRITE \
PROCEDURE below: assemble ONE valid JSON object citing prior work as bare `[id]`, \
then self-check that it parses. The final file must be exactly one valid JSON \
object — no markdown fences, no prose before or after.

The SCHEMA, write procedure, citation rules, and quality bar below are exactly \
those the trained agent is scored against. Follow them precisely.

============================================================
PRODUCTION report_writer SCHEMA & RULES (the label must satisfy these):
============================================================
"""


def build_label_generator_prompt() -> str:
    """Assemble the label-generator system prompt.

    Prepends the label-mode framing to the production ``report_writer`` role
    prompt so the results-masked schema / write procedure / citation rules /
    paper-access notes are byte-identical to what the trained agent sees.
    """
    return _LABEL_FRAMING + T.ROLES["report_writer"].prompt


def build_target_paper_message(
    title: str,
    problem: str,
    full_text: str,
    gaps_md: str,
    candidates_md: str,
    references: list[dict] | None = None,
) -> str:
    """The first user turn: inject the oracle (target paper full text), the
    distilled problem statement, the ground-truth ``gaps.md`` (assigned gap),
    the ground-truth ``candidates.md`` (chosen intuition), and the target
    paper's in-vault reference list (id + title) so the generator can ground
    every citation in the paper's own cited prior work before broadening via
    keyword_search.

    ``references`` is a list of ``{"id": int, "title": str}`` for the works the
    target paper cites that resolve to a paper in the vault.
    """
    refs_block = "(none resolved in the vault)"
    if references:
        lines = [f"- [{r['id']}] {r.get('title', '').strip()}" for r in references]
        refs_block = "\n".join(lines)
    return (
        f"# Target paper (oracle full text)\n"
        f"Title: {title}\n\n"
        f"## Distilled problem statement\n{problem}\n\n"
        f"## Ground-truth research gap (gaps.md — assigned to this proposal)\n"
        f"The proposal must close a specific gap from this document.\n\n"
        f"{gaps_md}\n\n"
        f"## Ground-truth research intuition (candidates.md — the subject of the report)\n"
        f"Develop the surviving candidate (its intuition + source inspiration) "
        f"into the full proposal. Do NOT invent a different idea.\n\n"
        f"{candidates_md}\n\n"
        f"## Target paper's in-vault references\n"
        f"These are the works the target paper cites that ALSO exist in our "
        f"vault. Review them FIRST when grounding related work, baselines, and "
        f"the borrowed mechanism: read_paper the relevant ones before citing them "
        f"as [id], then broaden via keyword_search as needed.\n"
        f"{refs_block}\n\n"
        f"## Full text\n{full_text}\n\n"
        f"---\n"
        f"Now develop the chosen intuition into the complete report.json "
        f"solution proposal. FIRST read gaps.md / candidates.md above; identify "
        f"every paper you will cite and read_paper each one that is not already "
        f"cited in the seeded inputs; THEN assemble and write ONE valid JSON "
        f"object to report.json per the schema, write procedure, and citation "
        f"rules above (every cited [id] a real DB id you read_paper'd this run; "
        f"never cite the target paper; no measured results or fabricated numbers). "
        f"Write it as a forward-looking proposal, never as a summary of the "
        f"target paper. When report.json is complete (it parses AND every cited id "
        f"resolves), reply with a one-line status — nothing else."
    )
