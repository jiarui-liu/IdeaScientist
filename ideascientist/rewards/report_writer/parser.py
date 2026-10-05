"""Parse a ``report.json`` document into a structured proposal + citation ids.

The report_writer emits ONE results-masked JSON object to ``report.json`` (see
``harness/roles.py:ROLES["report_writer"]`` and the reference labels under
the reference artifacts). Unlike the markdown-emitting sub-agents
there is no bespoke text grammar — the artifact is JSON — so this module is a thin
wrapper that:

  1. parses the JSON (``json.loads``), and
  2. extracts the citation id sets used by the reward,

REUSING the production label validators from
``ideascientist.vault.references.report_writer``
(``_load_report`` / ``_anchor_ids`` / ``_inline_anchor_ids`` /
``_mechanism_cited_ids``) so the reward path and the label acceptance-checker
agree byte-for-byte on "what is a valid report and what does it cite". No
reimplementation — import, don't copy.

This is the report_writer analogue of ``gap_finder/parser`` /
``innovator/parser``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

# Reuse the reference-generator's validators so a trained report is checked by
# exactly the code that validated its reference artifact.
from ideascientist.vault.references.report_writer import (
    _anchor_ids as _label_anchor_ids,
    _inline_anchor_ids as _label_inline_anchor_ids,
    _mechanism_cited_ids as _label_mechanism_cited_ids,
)

# The 14 results-masked schema fields every report must carry, in schema order,
# mirroring ``rubric.SOLUTION_PROPOSAL_RUBRIC["required_schema_fields"]``. There is
# deliberately no ``references`` field: citations are bare inline ``[id]`` markers,
# which is what makes them checkable against the read digests.
REPORT_FIELDS: tuple[str, ...] = (
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
)


@dataclass
class Report:
    """One parsed report.json proposal + its citation id sets.

    ``obj`` is None when the text does not parse as a JSON object; ``parse_error``
    then holds the diagnosis. ``anchor_ids`` are the ``related_work[].citations``
    ids, ``inline_ids`` the bare ``[id]`` ids cited inline in any string value,
    ``mechanism_ids`` the ids cited at the mechanism level (related_work /
    comparison_to_sota). ``cited_ids`` is the union used for the citation-F1
    component.
    """

    obj: dict | None = None
    parse_error: str | None = None
    anchor_ids: list[int] = field(default_factory=list)
    inline_ids: list[int] = field(default_factory=list)
    mechanism_ids: list[int] = field(default_factory=list)

    @property
    def parses(self) -> bool:
        return self.obj is not None

    @property
    def cited_ids(self) -> list[int]:
        """Union of anchor + inline ids — the report's full citation set."""
        return sorted(set(self.anchor_ids) | set(self.inline_ids))

    def has_all_fields(self) -> bool:
        if not self.obj:
            return False
        return all(_field_nonempty(self.obj.get(f)) for f in REPORT_FIELDS)

    def missing_fields(self) -> list[str]:
        if not self.obj:
            return list(REPORT_FIELDS)
        return [f for f in REPORT_FIELDS if not _field_nonempty(self.obj.get(f))]


def _field_nonempty(val: object) -> bool:
    if val is None:
        return False
    if hasattr(val, "__len__"):
        return len(val) > 0  # type: ignore[arg-type]
    return True


def parse_report_text(text: str) -> Report:
    """Parse a report.json STRING into a Report.

    Uses the production ``_load_report`` on a temporary parse: since that helper
    takes a Path, we parse the string directly here with the SAME semantics
    (``json.loads`` + isinstance dict guard) and delegate id extraction to the
    production validators.
    """
    if not text or not text.strip():
        return Report(obj=None, parse_error="report.json is empty")
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as exc:
        return Report(obj=None, parse_error=f"invalid JSON: {exc}")
    if not isinstance(obj, dict):
        return Report(obj=None, parse_error="report.json is not a JSON object")
    return Report(
        obj=obj,
        parse_error=None,
        anchor_ids=sorted(_label_anchor_ids(obj)),
        inline_ids=sorted(_label_inline_anchor_ids(obj)),
        mechanism_ids=sorted(_label_mechanism_cited_ids(obj)),
    )


