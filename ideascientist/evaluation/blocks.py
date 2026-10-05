"""Render papers into the reference blocks the judges are shown.

Per-paper character budgets keep the assembled block inside the judge's context
window even when a dozen comparison papers accompany a long target paper.
Each paper is rendered from its results-masked record where one exists, and
falls back to full text and then to bare fields.
"""

from __future__ import annotations

import json

from ideascientist.evaluation.loaders import SourcePaper
from ideascientist.evaluation.similar_papers import SimilarPaper

SIMILAR_RECORD_CAP = 3000
SIMILAR_FULLTEXT_CAP = 3000
TARGET_RECORD_CAP = 12000
TARGET_FULLTEXT_CAP = 30000
MAX_REFERENCE_PAPERS = 12


RECORD_FIELDS = (
    "title", "one_sentence_thesis", "core_problem", "key_novelty",
    "problem_definition", "related_work", "method", "model_details",
    "comparison_to_sota", "proposed_evaluation", "falsifiable_predictions",
    "limitations",
)


def record_block(obj: dict, cap: int) -> str:
    """Serialize a results-masked record into compact labeled text for a judge."""
    parts = []
    for key in RECORD_FIELDS:
        value = obj.get(key)
        if value in (None, "", [], {}):
            continue
        parts.append(f"**{key}:** " + json.dumps(value, ensure_ascii=False))
    return "\n".join(parts)[:cap]


def comparison_set_block(sims: list[SimilarPaper]) -> str:
    blocks = []
    for i, s in enumerate(sims[:MAX_REFERENCE_PAPERS], 1):
        header = (
            f"## Reference {i}: {s.title} (paper_id={s.paper_id}; "
            f"challenge_sim={s.challenge_score}, pd_sim={s.problem_definition_score})"
        )
        if s.record:
            body = record_block(s.record, SIMILAR_RECORD_CAP)
        elif (s.full_text or "").strip():
            body = "**full_text (results-masked summary unavailable):**\n" + s.full_text[:SIMILAR_FULLTEXT_CAP]
        else:
            body = (
                f"**Problem definition:** {(s.problem_definition or 'N/A')[:1500]}\n"
                f"**Challenge:** {(s.challenge or 'N/A')[:1500]}"
            )
        blocks.append(f"{header}\n{body}")
    return "\n\n".join(blocks) if blocks else "(no similar prior papers found)"


def reference_block(source: SourcePaper, sims: list[SimilarPaper]) -> str:
    if source.record:
        src_body = record_block(source.record, TARGET_RECORD_CAP)
    elif (source.full_text or "").strip():
        src_body = ("**full_text (results-masked summary unavailable):**\n"
                    + source.full_text[:TARGET_FULLTEXT_CAP])
    else:
        src_body = (source.solution or "N/A")[:TARGET_FULLTEXT_CAP]
    blocks = [
        f"## Human source paper: {source.title or 'N/A'} (paper_id={source.paper_id})\n"
        f"{src_body}"
    ]
    for i, s in enumerate(sims[:MAX_REFERENCE_PAPERS], 1):
        if s.record:
            body = record_block(s.record, SIMILAR_RECORD_CAP)
        elif (s.full_text or "").strip():
            body = "**full_text (results-masked summary unavailable):**\n" + s.full_text[:SIMILAR_FULLTEXT_CAP]
        else:
            continue
        blocks.append(
            f"## Prior paper {i}: {s.title} (paper_id={s.paper_id})\n{body}"
        )
    return "\n\n".join(blocks)


