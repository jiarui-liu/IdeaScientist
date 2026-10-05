"""System prompt for the gap-finder GROUND-TRUTH LABEL generator.

This is NOT the production ``gap_finder`` sub-agent (that lives in
``harness/roles.py:ROLES['gap_finder']`` and never sees the target paper's
body). This label generator is *privileged*: it is given the FULL TEXT of a
human-written target paper (the oracle) and produces a complete ``gaps.md`` that

  * follows the gaps the paper itself argues for (introduction / related work /
    motivation) — one ``## Axis`` per distinct methodological challenge axis, and
    the paper decides how many (1, 2, or 3); never pad,
  * grounds every gap ONLY in prior work that exists in the vault and
    was actually ``read_paper``'d this run (so the label passes the same
    deterministic gates the reward uses — see ``rubric.py``), and
  * is written in the EXACT parseable schema the production ``gap_finder`` emits,
    so the labels are in-distribution for the agent we train.

The gap SCHEMA + HARD RULES below are copied from
``ROLES['gap_finder'].prompt`` and MUST stay in sync with it. We build the
prompt by importing that role's text so the two cannot silently drift, and we
prepend the label-mode framing.
"""

from __future__ import annotations

from ideascientist.harness import tools as T


# The label-mode framing wrapped around the production gap_finder rules. The
# production prompt's SCOPE / SCHEMA / HARD RULES are appended verbatim (imported
# below) so the label matches exactly what the trained agent must produce.
_LABEL_FRAMING = """\
You are constructing a GROUND-TRUTH research-gap document (`gaps.md`) for a \
human-written paper. This document is a training LABEL: it is the expected, \
high-quality output we want a gap-finding agent to learn to produce.

You are given the FULL TEXT of ONE target human paper (the "target paper"). \
You may read the target paper's full body — \
use it as an ORACLE to learn (a) which methodological challenge axes the paper \
actually addresses, and (b) the real research gaps the paper argues its prior \
work leaves open (look in the introduction, related work, and motivation).

YOUR JOB:
1) DERIVE THE AXES FROM THE PAPER, DO NOT INVENT THEM. Read the target paper \
and identify the distinct methodological challenge axes it positions itself \
against. A paper may argue for ONE research gap or for two or three — emit \
exactly as many `## Axis` sections as the paper genuinely motivates, no more. \
Do NOT pad with axes the paper does not argue for.

1a) BE FAITHFUL TO THE PAPER — EVERY AXIS AND GAP MUST BE TRACEABLE TO IT. \
Each axis and each gap you emit must correspond to a challenge the target paper \
ITSELF explicitly argues (in its abstract, introduction, related work, or \
motivation), not one you find interesting or that the retrieved prior work \
suggests. Before writing an axis, be able to point to where in the paper it is \
motivated. Do NOT introduce an axis or gap the paper does not actually raise, \
even if it is a real research gap; do NOT reshape the paper's argument to fit \
prior work you happened to retrieve. If the paper argues fewer distinct gaps than \
you found near-miss work for, emit fewer gaps — faithfulness to the paper's own \
framing outranks retrieval coverage.

1b) THE PAPER'S OWN CORE CONTRIBUTION MUST APPEAR AS A GAP. The target paper \
proposes a method that closes some central methodological gap; that gap is the \
single most important label to get right. Identify the core gap this paper's \
method claims to close, and ensure AT LEAST ONE `## Axis` + gap captures exactly \
that gap — grounded in the near-miss prior work the paper improves on. Write it \
as the GAP the prior work leaves open (the hole the paper's method fills), NOT as \
the paper's solution, and never cite the target paper itself. If a paper argues \
several gaps, this core-contribution gap must be one of the emitted gaps.

1c) EVERY GAP MUST BE GENUINELY DISTINCT. Each emitted gap (whether within one \
axis or across axes) must name a DIFFERENT methodological deficiency, closed by a \
DIFFERENT methodological move. Do NOT split one gap into two by rephrasing it, \
listing a symptom and its cause as separate gaps, or narrowing scope (e.g. "gap \
X" and "gap X for rare tokens"). Before emitting a second gap, state to yourself \
how its `Methodological deficiency` differs from the first; if they collapse to \
the same underlying deficiency, emit only ONE gap. Fewer, non-overlapping gaps \
are strictly better than more, overlapping ones — the rubric rewards depth and \
distinctness, not count.

2) FIRST REVIEW THE TARGET PAPER'S OWN REFERENCES. You are given the list of \
works the target paper cites that ALSO exist in the vault ("Target paper's \
in-vault references" below). Before any broad search, go through this list \
and, for each axis, pick the references that are relevant to that axis and \
`read_paper(id, goals)` them. These are the prior works the paper itself \
positioned against, so they are the highest-priority near-miss candidates.

3) THEN GROUND EACH AXIS MORE BROADLY VIA DATABASE SEARCH. The target paper's \
reference list is not exhaustive for our corpus. For each axis, ALSO run \
`keyword_search` with 4-8 queries phrased in the EXACT terminology of that axis \
to surface additional same-problem prior work in the vault that the paper may \
not have cited. Re-phrase and re-run if the pool is thin. `read_paper` the most \
relevant hits. Cite ONLY paper ids you retrieved (from the reference list or \
from search) AND read_paper'd this run.

4) NEVER cite the target paper itself as near-miss prior work, and never cite \
an id you did not read_paper this run. If the paper argues a gap but you cannot \
find any grounding near-miss work for it in the vault after an honest search, \
DROP that gap rather than fabricate — a label with fewer, fully-grounded gaps is \
better than one with ungrounded gaps.

5) Write the full `gaps.md` with `edit_doc`: one `## Axis: <name>` section per \
axis, each with `### What has been done` (grouped, mechanism-level, citing db \
ids) and `### Open gaps on this axis` (1-2 gaps in the exact bullet schema \
below). Grow the file with edit_doc insert (insert_line=999999); do not \
overwrite earlier axes.

Budget: read at most ~12 papers total across all axes (prioritise the most \
load-bearing near-miss work).

The SCHEMA, quality bar, and HARD RULES below are exactly those the trained \
agent is scored against. Follow them precisely for every gap you write.

============================================================
PRODUCTION gap_finder SCHEMA & RULES (the label must satisfy these):
============================================================
"""


def build_label_generator_prompt() -> str:
    """Assemble the label-generator system prompt.

    Prepends the label-mode framing to the production ``gap_finder`` role prompt
    so the gap schema / HARD RULES / paper-access notes are byte-identical to
    what the trained agent sees.
    """
    return _LABEL_FRAMING + T.ROLES["gap_finder"].prompt


def build_target_paper_message(
    title: str,
    problem: str,
    full_text: str,
    references: list[dict] | None = None,
) -> str:
    """The first user turn: inject the oracle (target paper full text) plus the
    target paper's in-vault reference list (id + title), so the generator can
    review the paper's own cited prior work before broadening via keyword_search.

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
        f"## Target paper's in-vault references\n"
        f"These are the works the target paper cites that ALSO exist in our "
        f"vault. Review them FIRST: for each axis, read_paper the ones "
        f"relevant to that axis before broadening via keyword_search.\n"
        f"{refs_block}\n\n"
        f"## Full text\n{full_text}\n\n"
        f"---\n"
        f"Now derive the paper's methodological challenge axes. For each axis, "
        f"FIRST read_paper the relevant works from the in-vault reference list "
        f"above, THEN run keyword_search to find additional relevant prior work "
        f"in the vault, read the best hits, and write the complete gaps.md "
        f"label per the schema and rules above (cite ONLY ids you read_paper'd "
        f"this run). When gaps.md is complete, reply with a one-line status "
        f"naming each axis and its number of gaps — nothing else."
    )
