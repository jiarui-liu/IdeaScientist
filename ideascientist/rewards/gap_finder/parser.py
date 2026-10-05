"""Parse a ``gaps.md`` document into structured axes / gaps / fields.

The gap_finder emits ``gaps.md`` in a fixed schema (see
``harness/roles.py:ROLES["gap_finder"]`` and the reference labels under
the gap-finder reference artifacts):

    ## Axis: <name>

    ### What has been done
    - ...prose bullets...

    ### Open gaps on this axis
    - **Gap:** <one sentence>
    - **Near-miss methods and coverage:** [id] — ...; [id] — ...
    - **Methodological deficiency:** ...
    - **Why it matters:** ...
    - **Evidence route:** ...
    (repeat the 5-field block per gap)

This module is a PURE function of the text — no DB, no LLM — so the reward path
and the label acceptance-checker agree byte-for-byte on "what is a valid gap".
It is shared by ``gap_finder_reward`` (scoring a rollout) and by the dataset
builder (reading reference labels).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# The 5 bullet fields every gap block must carry, in emission order. Kept in
# sync with ``rubrics.RESEARCH_GAP_RUBRIC["required_gap_fields"]``.
GAP_FIELDS: tuple[str, ...] = (
    "Gap",
    "Near-miss methods and coverage",
    "Methodological deficiency",
    "Why it matters",
    "Evidence route",
)

# ``- **Field:** value`` — tolerant of leading whitespace and optional bold on
# the value. The field label is matched case-insensitively against GAP_FIELDS.
_FIELD_RE = re.compile(
    r"^\s*[-*]\s*\*\*\s*(?P<label>[^:*]+?)\s*:\s*\*\*\s*(?P<value>.*)$"
)
_AXIS_RE = re.compile(r"^\s*##\s+Axis\s*:\s*(?P<name>.+?)\s*$", re.IGNORECASE)
_SUBHEAD_RE = re.compile(r"^\s*###\s+(?P<name>.+?)\s*$")
_CITED_ID_RE = re.compile(r"\[(\d+)\]")


def _norm_label(label: str) -> str | None:
    low = label.strip().lower()
    for f in GAP_FIELDS:
        if low == f.lower():
            return f
    return None


@dataclass
class Gap:
    """One open gap: its 5 fields + the paper ids it cites."""

    fields: dict[str, str] = field(default_factory=dict)
    cited_ids: list[int] = field(default_factory=list)
    raw: str = ""

    def has_all_fields(self) -> bool:
        return all((self.fields.get(f) or "").strip() for f in GAP_FIELDS)

    def missing_fields(self) -> list[str]:
        return [f for f in GAP_FIELDS if not (self.fields.get(f) or "").strip()]


@dataclass
class Axis:
    """One ``## Axis:`` block: its name, what-has-been-done prose, and gaps."""

    name: str = ""
    what_has_been_done: str = ""
    gaps: list[Gap] = field(default_factory=list)


def _flush_gap(cur: dict[str, str], raw_lines: list[str]) -> Gap | None:
    if not cur:
        return None
    raw = "\n".join(raw_lines).strip()
    ids = sorted({int(m) for m in _CITED_ID_RE.findall(raw)})
    return Gap(fields=dict(cur), cited_ids=ids, raw=raw)


def _parse_gaps_block(lines: list[str]) -> list[Gap]:
    """Parse the ``### Open gaps on this axis`` body into a list of Gap.

    A new gap starts at each ``**Gap:**`` bullet. Fields between two ``**Gap:**``
    markers belong to the current gap. Non-field lines (blank / continuation)
    are appended to the last field's value so multi-line values survive.
    """
    gaps: list[Gap] = []
    cur: dict[str, str] = {}
    raw_lines: list[str] = []
    last_field: str | None = None

    for line in lines:
        m = _FIELD_RE.match(line)
        label = _norm_label(m.group("label")) if m else None
        if label == "Gap":
            g = _flush_gap(cur, raw_lines)
            if g is not None:
                gaps.append(g)
            cur = {"Gap": m.group("value").strip()}
            raw_lines = [line]
            last_field = "Gap"
        elif label is not None and cur:
            cur[label] = m.group("value").strip()
            raw_lines.append(line)
            last_field = label
        elif cur:
            # Continuation of the current field's value (wrapped line).
            raw_lines.append(line)
            if last_field and line.strip():
                cur[last_field] = (cur[last_field] + " " + line.strip()).strip()

    g = _flush_gap(cur, raw_lines)
    if g is not None:
        gaps.append(g)
    return gaps


def parse_gaps_md(text: str) -> list[Axis]:
    """Parse a full ``gaps.md`` string into a list of Axis."""
    if not text or not text.strip():
        return []
    axes: list[Axis] = []
    cur_axis: Axis | None = None
    section: str | None = None  # "done" | "gaps" | None
    done_lines: list[str] = []
    gaps_lines: list[str] = []

    def _close_axis() -> None:
        nonlocal cur_axis, done_lines, gaps_lines
        if cur_axis is None:
            return
        cur_axis.what_has_been_done = "\n".join(done_lines).strip()
        cur_axis.gaps = _parse_gaps_block(gaps_lines)
        axes.append(cur_axis)
        done_lines = []
        gaps_lines = []

    for line in text.splitlines():
        am = _AXIS_RE.match(line)
        if am:
            _close_axis()
            cur_axis = Axis(name=am.group("name").strip())
            section = None
            continue
        sm = _SUBHEAD_RE.match(line)
        if sm and cur_axis is not None:
            head = sm.group("name").strip().lower()
            if head.startswith("what has been done"):
                section = "done"
            elif head.startswith("open gaps"):
                section = "gaps"
            else:
                section = None
            continue
        if cur_axis is None:
            continue
        if section == "done":
            done_lines.append(line)
        elif section == "gaps":
            gaps_lines.append(line)

    _close_axis()
    return axes


def all_gaps(axes: list[Axis]) -> list[Gap]:
    return [g for ax in axes for g in ax.gaps]


