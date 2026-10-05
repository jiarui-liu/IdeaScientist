"""Parse a ``candidates.md`` document into structured candidates / fields.

The innovator emits ``candidates.md`` in a fixed schema (see
``harness/roles.py:ROLES["innovator"]`` and the reference labels under
the innovator reference artifacts):

    ### C1 — <short title>

    - **Intuition**: <one-to-two sentences>
    - **Gap it attacks**: <named gap + concrete failure>
    - **If improving prior candidate**: <optional>
    - **Source inspiration**: <source paper ids + mechanism>  [id] / [id, id]
    - **How it maps to this problem**: <conceptual translation>
    - **Why it could work / feasibility**: <feasibility argument>
    - **Novelty vs. closest in-domain work**: <named competitor + delta>
    - **Main risk**: <one line>
    (repeat the block per candidate under its own ### C<n> header)

Note the emitted label style is ``- **Label**: value`` (colon OUTSIDE the bold),
which differs from gaps.md's ``- **Gap:** value`` (colon INSIDE the bold); the
field regex here tolerates BOTH forms so the parser is robust to either.

This module is a PURE function of the text — no DB, no LLM — so the reward path
and the label acceptance-checker agree byte-for-byte on "what is a valid
candidate". It is shared by ``innovator_reward`` (scoring a rollout) and by the
dataset builder (reading reference labels). Mirrors ``gap_finder/parser``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# The 7 required bullet fields every candidate block must carry, in emission
# order. Kept in sync with
# ``rubrics.RESEARCH_INTUITION_RUBRIC["required_candidate_fields"]``.
CANDIDATE_FIELDS: tuple[str, ...] = (
    "Intuition",
    "Gap it attacks",
    "Source inspiration",
    "How it maps to this problem",
    "Why it could work / feasibility",
    "Novelty vs. closest in-domain work",
    "Main risk",
)

# One optional field the innovator may emit; recognized (so it does not pollute the
# preceding field's value) but not required.
OPTIONAL_CANDIDATE_FIELDS: tuple[str, ...] = (
    "If improving prior candidate",
)

_ALL_LABELS: tuple[str, ...] = CANDIDATE_FIELDS + OPTIONAL_CANDIDATE_FIELDS

# Field-label bullet, tolerant of all emitted variants seen in the labels:
#   ``- **Field**: value``   (bullet, colon outside bold)
#   ``- **Field:** value``   (bullet, colon inside bold)
#   ``**Field** value``      (no bullet, no colon — value on same line)
#   ``**Field**``            (no bullet, value on following lines)
# The bullet marker is OPTIONAL and, when present, must be followed by whitespace
# (so the leading ``*`` of ``**Field**`` is not mistaken for a bullet). A trailing
# ``(...)`` parenthetical in the label is stripped by ``_norm_label``. The field
# label is matched case-insensitively against _ALL_LABELS.
_FIELD_RE = re.compile(
    r"^\s*(?:[-*]\s+)?\*\*\s*(?P<label>[^:*]+?)\s*(?::\s*)?\*\*\s*:?\s*(?P<value>.*)$"
)
# ``### C<n> — <title>`` (em-dash or hyphen); the C-number starts a new candidate.
_CAND_RE = re.compile(
    r"^\s*###\s+(?P<cid>C\s*\d+)\b\s*(?:[—\-–]\s*(?P<title>.*))?$",
    re.IGNORECASE,
)
_CITED_ID_RE = re.compile(r"\[(?P<ids>[\d,;\s]+)\]")


def _extract_ids(text: str) -> list[int]:
    ids: set[int] = set()
    for m in _CITED_ID_RE.finditer(text):
        for tok in re.split(r"[,;\s]+", m.group("ids")):
            tok = tok.strip()
            if tok.isdigit():
                ids.add(int(tok))
    return sorted(ids)


def _norm_label(label: str) -> str | None:
    """Map a raw bullet label to its canonical field name, else None.

    Strips a trailing ``(...)`` parenthetical from the label before matching.
    """
    low = re.sub(r"\s*\(.*\)\s*$", "", label.strip()).strip().lower()
    for f in _ALL_LABELS:
        if low == f.lower():
            return f
    return None


@dataclass
class Candidate:
    """One research-intuition candidate: its title, fields, and cited paper ids."""

    cid: str = ""
    title: str = ""
    fields: dict[str, str] = field(default_factory=dict)
    cited_ids: list[int] = field(default_factory=list)
    raw: str = ""

    def has_all_fields(self) -> bool:
        return all((self.fields.get(f) or "").strip() for f in CANDIDATE_FIELDS)

    def missing_fields(self) -> list[str]:
        return [f for f in CANDIDATE_FIELDS if not (self.fields.get(f) or "").strip()]


def _flush_candidate(
    cid: str, title: str, cur: dict[str, str], raw_lines: list[str]
) -> Candidate | None:
    if not cid and not cur:
        return None
    raw = "\n".join(raw_lines).strip()
    return Candidate(
        cid=cid,
        title=title,
        fields=dict(cur),
        cited_ids=_extract_ids(raw),
        raw=raw,
    )


def parse_candidates_md(text: str) -> list[Candidate]:
    """Parse a full ``candidates.md`` string into a list of Candidate.

    A new candidate starts at each ``### C<n>`` header. Fields between two headers
    belong to the current candidate. Non-field lines (blank / continuation) are
    appended to the last field's value so multi-line values survive.
    """
    if not text or not text.strip():
        return []
    candidates: list[Candidate] = []
    cur_cid: str = ""
    cur_title: str = ""
    cur: dict[str, str] = {}
    raw_lines: list[str] = []
    last_field: str | None = None

    for line in text.splitlines():
        cm = _CAND_RE.match(line)
        if cm:
            c = _flush_candidate(cur_cid, cur_title, cur, raw_lines)
            if c is not None:
                candidates.append(c)
            cur_cid = re.sub(r"\s+", "", cm.group("cid")).upper()
            cur_title = (cm.group("title") or "").strip()
            cur = {}
            raw_lines = [line]
            last_field = None
            continue
        if not cur_cid:
            continue
        raw_lines.append(line)
        m = _FIELD_RE.match(line)
        label = _norm_label(m.group("label")) if m else None
        if label is not None:
            cur[label] = m.group("value").strip()
            last_field = label
        elif last_field and line.strip():
            cur[last_field] = (cur[last_field] + " " + line.strip()).strip()

    c = _flush_candidate(cur_cid, cur_title, cur, raw_lines)
    if c is not None:
        candidates.append(c)
    return candidates




