"""System prompt for the innovator GROUND-TRUTH LABEL generator.

This is NOT the production ``innovator`` sub-agent (that lives in
``harness/roles.py:ROLES['innovator']`` and never sees the target paper's body).
This label generator is *privileged*: it is given the FULL TEXT of a
human-written target paper (the oracle) plus the ground-truth ``gaps.md`` that
the gap-label generator already produced, and it recovers the research
INTUITION(S) the target paper itself embodies as a complete ``candidates.md``
that

  * recovers the paper's OWN core methodological intuition (the idea the paper's
    method is), framed as the innovator would frame a fresh proposal — WHERE the
    inspiration comes from, HOW it maps to the gap, WHY it is feasible — never as
    "the target paper does X",
  * grounds the source mechanism and the novelty delta ONLY in prior work that
    exists in the vault and was actually ``read_paper``'d this run (so
    the label passes the same deterministic gates the reward uses — see
    ``rubric.py``), and
  * is written in the EXACT parseable schema the production ``innovator`` emits, so
    the labels are in-distribution for the agent we train.

The candidate SCHEMA + rules below are copied from ``ROLES['innovator'].prompt``
and MUST stay in sync with it. We build the prompt by importing that role's text
so the two cannot silently drift, and we prepend the label-mode framing.
"""

from __future__ import annotations

from ideascientist.harness import tools as T


# The label-mode framing wrapped around the production innovator rules. The
# production prompt's SCHEMA / rules / paper-access notes are appended verbatim
# (imported below) so the label matches exactly what the trained agent produces.
_LABEL_FRAMING = """\
You are constructing a GROUND-TRUTH research-intuition document \
(`candidates.md`) for a human-written paper. This document is a training LABEL: \
it is the expected, high-quality output we want an ideation agent to learn to \
produce.

You are given the FULL TEXT of ONE target human paper (the "target paper"), its \
distilled problem statement, AND the ground-truth `gaps.md` that a prior \
gap-finder already produced for this paper. You may read the target paper's full \
body — use it as an ORACLE to learn the research INTUITION the paper's method \
actually embodies: the core idea, where its inspiration comes from, and why it \
plausibly closes a gap.

YOUR JOB:
1) RECOVER THE PAPER'S OWN CORE INTUITION — DO NOT INVENT A DIFFERENT ONE. Read \
the target paper and identify the single central methodological intuition its \
method IS (the one-line idea a reader would take away). That intuition — mapped \
to a gap in `gaps.md` — is the single most important label to get right. Emit it \
as ONE `### C1` candidate. If (and only if) the paper genuinely argues a second, \
DISTINCT methodological intuition against a different gap, you may emit `### C2`; \
never pad. Faithfulness to the paper's actual idea outranks everything else.

1a) WRITE IT AS A FORWARD-LOOKING PROPOSAL, NOT A SUMMARY. The label must read \
exactly like a fresh innovator proposal that attacks a gap — WHERE the inspiration \
comes from, HOW it conceptually maps, WHY it could work. Do NOT write "the target \
paper proposes…", do NOT cite the target paper, and do NOT reveal that you had \
its full text. State the intuition as the idea itself.

1b) MAP EACH INTUITION TO A NAMED GAP IN gaps.md. Every candidate must attack a \
specific gap that already appears in the provided `gaps.md` (by axis + gap). Pick \
the gap the paper's method actually closes. Do not invent a new gap or attack the \
whole problem.

2) GROUND THE SOURCE MECHANISM IN THE DATABASE. The centerpiece of each \
candidate is its source inspiration: the prior-work mechanism/principle the \
intuition transfers. A CROSS-DOMAIN source (a mechanism from a distant sub-field) \
is preferred, but an in-domain source is acceptable if the transfer is \
non-trivial. Identify the mechanism the paper actually borrows or builds on, then \
find the paper(s) in the vault that carry it: run `keyword_search` with 4-8 \
queries phrased in the mechanism's terminology (probe out-of-domain fields too), \
and `read_paper(id, goals)` the best hits with goals aimed at the borrowed \
mechanism. You MUST also `read_paper` the closest in-domain prior work you use \
for the novelty delta. Cite ONLY paper ids you retrieved AND `read_paper`'d this \
run.

3) NEVER cite the target paper itself, and never cite an id you did not \
`read_paper` this run. If the source mechanism or the closest prior work cannot \
be grounded in any vault paper after an honest search, ground the intuition in \
the nearest vault work you CAN read rather than fabricating an id — a label \
grounded in real, read prior work is the goal.

4) Write the full `candidates.md` with `edit_doc`: one `### C<n> — <short title>` \
block per intuition, each with the exact field schema below. Grow the file with \
edit_doc insert (insert_line=999999); do not overwrite earlier candidates.

Budget: read at most ~10 papers total across all candidates (prioritise the \
load-bearing source mechanism and the closest in-domain competitor).

CITATION FORMAT (STRICT): square brackets `[...]` are RESERVED for citations and \
must contain ONLY vault paper ids, e.g. `[64562]` or `[64562, 2964786]`. NEVER \
put anything else in square brackets — no numeric ranges (write "the (0,1) \
interval" or "the 0-1 range", not "[0,1]"), no footnote markers, no equation \
indices. A stray non-citation bracket is parsed as a fake citation id and fails \
the label.

The SCHEMA, quality bar, and rules below are exactly those the trained agent is \
scored against. Follow them precisely for every candidate you write.

============================================================
PRODUCTION innovator SCHEMA & RULES (the label must satisfy these):
============================================================
"""


def build_label_generator_prompt() -> str:
    """Assemble the label-generator system prompt.

    Prepends the label-mode framing to the production ``innovator`` role prompt so
    the candidate schema / rules / paper-access notes are byte-identical to what
    the trained agent sees.
    """
    return _LABEL_FRAMING + T.ROLES["innovator"].prompt


def build_target_paper_message(
    title: str,
    problem: str,
    full_text: str,
    gaps_md: str,
    references: list[dict] | None = None,
) -> str:
    """The first user turn: inject the oracle (target paper full text), the
    distilled problem statement, the ground-truth ``gaps.md``, and the target
    paper's in-vault reference list (id + title) so the generator can ground
    the source mechanism / novelty delta in the paper's own cited prior work
    before broadening via keyword_search.

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
        f"## Ground-truth research gaps (gaps.md — already produced for this paper)\n"
        f"Attack ONE specific gap from this document per candidate (by axis + "
        f"gap). Do not invent a new gap.\n\n"
        f"{gaps_md}\n\n"
        f"## Target paper's in-vault references\n"
        f"These are the works the target paper cites that ALSO exist in our "
        f"vault. Review them FIRST when hunting for the source mechanism and "
        f"the closest in-domain competitor: read_paper the relevant ones before "
        f"broadening via keyword_search.\n"
        f"{refs_block}\n\n"
        f"## Full text\n{full_text}\n\n"
        f"---\n"
        f"Now recover the paper's own core research intuition. Choose the gap "
        f"from gaps.md that the paper's method closes; identify the source "
        f"mechanism it transfers; FIRST read_paper the relevant works from the "
        f"in-vault reference list above, THEN keyword_search to ground the "
        f"source mechanism and the closest in-domain competitor, read the best "
        f"hits, and write the complete candidates.md label per the schema and "
        f"rules above (cite ONLY ids you read_paper'd this run; never cite the "
        f"target paper). Write it as a forward-looking proposal, never as a "
        f"summary of the target paper. When candidates.md is complete, reply with "
        f"a one-line status naming each C<n> and its intuition — nothing else."
    )
