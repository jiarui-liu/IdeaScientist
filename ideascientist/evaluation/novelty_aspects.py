"""Aspect-level novelty judges: In-domain transfer and Mechanism non-obviousness.

Both are reported in the paper and both use the prompts in
:mod:`ideascientist.evaluation.prompts`, which are reproduced from the
appendix.

*In-domain novelty* is a binary judgment rather than a 1-10 comparison against
a literature set: it asks whether the proposal imports a mechanism from a
different problem setting and adapts it through a shared underlying structure.
That is precisely the behaviour the innovator's cross-domain retrieval regime
is meant to produce, so measuring it directly tests the paper's thesis instead
of assuming a gestalt novelty score captures it.

*Mechanism non-obviousness* runs as two calls. The first never sees the
proposal and derives candidate mechanisms from the nearest pre-cutoff work on
its own; the second grades the proposal against that frozen list. A single call
can always reverse-engineer a derivation once it has seen the answer, so the
split is what makes the measure meaningful. An invalid candidate set is
unscored and retried rather than counted as obvious.

Both metrics average over the fixed test-set denominator, so an unscored
proposal contributes zero.
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional

from ideascientist.evaluation.prompts import (
    OBVIOUSNESS_LEVELS,
    OBVIOUSNESS_SCORE,
    in_domain_transfer_system,
    in_domain_transfer_user,
    non_obviousness_derive_user,
    non_obviousness_grade_system,
    non_obviousness_grade_user,
    NON_OBVIOUSNESS_DERIVE_SYSTEM,
)

EVIDENCE_QUOTE_MAX_WORDS = 35

_JSON_RE = re.compile(r"\{.*\}", re.S)


def parse_json_object(text: str) -> Optional[dict[str, Any]]:
    """Extract one JSON object from a judge reply, tolerating fences and prose."""
    if not text:
        return None
    body = text.rsplit("</think>", 1)[-1]
    body = re.sub(r"^\s*```(?:json)?\s*", "", body, flags=re.IGNORECASE)
    body = re.sub(r"\s*```\s*$", "", body).strip()
    for candidate in (body, (_JSON_RE.search(body).group(0) if _JSON_RE.search(body) else "")):
        if not candidate:
            continue
        try:
            obj = json.loads(candidate)
        except (ValueError, TypeError):
            continue
        if isinstance(obj, dict):
            return obj
    return None


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip().lower()


def quote_is_grounded(quote: str, source_text: str, *, min_words: int = 4) -> bool:
    """True when ``quote`` really occurs in ``source_text``.

    Judges paraphrase even when told to quote verbatim, so exact substring
    matching is too strict and a length check too lax. A quote passes if it is
    a normalized substring, or if at least 70% of its content words appear in
    the source. A three-word fragment is not evidence.
    """
    q, s = _norm(quote), _norm(source_text)
    if not q or not s:
        return False
    words = [w for w in re.findall(r"[a-z0-9]+", q) if len(w) > 2]
    if len(words) < min_words:
        return False
    if q in s:
        return True
    return sum(1 for w in set(words) if w in s) / max(1, len(set(words))) >= 0.7


def proposal_mechanism_block(report: dict[str, Any], *, cap: int = 5000) -> str:
    """The part of a proposal that carries its claimed contribution.

    Excludes ``related_work`` and ``proposed_evaluation``: a judge shown the
    proposal's own literature framing tends to adopt it, which is the bias
    these metrics exist to avoid.
    """
    parts = []
    for f in ("title", "one_sentence_thesis", "core_problem", "problem_definition",
              "key_novelty", "method", "comparison_to_sota"):
        v = report.get(f)
        if isinstance(v, (list, dict)):
            v = json.dumps(v, ensure_ascii=False)
        v = (v or "").strip() if isinstance(v, str) else ""
        if v:
            parts.append(f"## {f}\n{v}")
    return "\n\n".join(parts)[:cap]


def prior_paper_block(p: dict[str, Any], *, cap: int = 2600) -> str:
    """Compact judge-facing rendering of one prior paper."""
    parts = [f"## title\n{(p.get('title') or '').strip()}"]
    for f in ("problem_definition", "challenge", "solution"):
        v = (p.get(f) or "").strip()
        if v:
            parts.append(f"## {f}\n{v}")
    return "\n\n".join(parts)[:cap]


# --------------------------------------------------------------------------- #
# In-domain novelty: structural cross-setting transfer
# --------------------------------------------------------------------------- #

IN_DOMAIN_FIELDS = (
    "source_setting", "target_setting", "transferred_mechanism",
    "shared_structure", "adaptation",
)


def build_in_domain_messages(
    mechanism_block: str, *, few_shot: str = ""
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": in_domain_transfer_system(few_shot)},
        {"role": "user", "content": in_domain_transfer_user(mechanism_block)},
    ]


def parse_in_domain(text: str, mechanism_block: str) -> Optional[dict[str, Any]]:
    """Parse an In-domain verdict, enforcing the evidence requirement.

    A score of 1 requires all four structural fields and a grounded quote
    within the word limit. The judge frequently asserts transfer it cannot
    quote, so an ungrounded 1 is demoted to 0 rather than dropped: the metric
    is a rate over a fixed denominator.
    """
    obj = parse_json_object(text)
    if obj is None:
        return None

    raw_score = obj.get("score")
    try:
        score = int(raw_score)
    except (TypeError, ValueError):
        return None
    if score not in (0, 1):
        return None

    quote = (obj.get("evidence_quote") or "") if isinstance(obj.get("evidence_quote"), str) else ""
    fields = {f: (obj.get(f) or None) for f in IN_DOMAIN_FIELDS}
    quote_ok = (
        bool(quote)
        and len(quote.split()) <= EVIDENCE_QUOTE_MAX_WORDS
        and quote_is_grounded(quote, mechanism_block)
    )
    all_fields = all(isinstance(fields[f], str) and fields[f].strip() for f in IN_DOMAIN_FIELDS)

    final = 1 if (score == 1 and quote_ok and all_fields) else 0
    return {
        **fields,
        "evidence_quote": quote,
        "evidence_ok": quote_ok,
        "reasoning": obj.get("reasoning", ""),
        "judge_score": score,
        "score": final,
        "demoted": score == 1 and final == 0,
    }


def aggregate_in_domain(verdicts: list[Optional[dict[str, Any]]], denominator: int) -> dict[str, Any]:
    """Rate over the fixed test-set denominator; an unscored proposal counts zero."""
    scored = [v for v in verdicts if v is not None]
    positives = sum(1 for v in scored if v["score"] == 1)
    return {
        "in_domain_novelty": positives / denominator if denominator else 0.0,
        "n_scored": len(scored),
        "n_unscored": denominator - len(scored),
        "n_positive": positives,
        "n_demoted": sum(1 for v in scored if v.get("demoted")),
        "denominator": denominator,
    }


# --------------------------------------------------------------------------- #
# Mechanism non-obviousness: proposal-blind derivation, then frozen comparison
# --------------------------------------------------------------------------- #

def build_derive_messages(
    problem_definition: str, challenge: str, priors_block: str
) -> list[dict[str, str]]:
    """Call 1. The proposal is deliberately absent from these messages."""
    return [
        {"role": "system", "content": NON_OBVIOUSNESS_DERIVE_SYSTEM},
        {
            "role": "user",
            "content": non_obviousness_derive_user(problem_definition, challenge, priors_block),
        },
    ]


def parse_candidates(text: str) -> Optional[list[dict[str, Any]]]:
    """Parse call 1's candidate list. An empty list is a valid result."""
    obj = parse_json_object(text)
    if obj is None or not isinstance(obj.get("candidates"), list):
        return None
    return [c for c in obj["candidates"] if isinstance(c, dict)]


def build_grade_messages(
    frozen_candidates: list[dict[str, Any]],
    priors_block: str,
    mechanism_block: str,
    *,
    few_shot: str = "",
) -> list[dict[str, str]]:
    """Call 2. The candidate list is passed through verbatim and never edited."""
    return [
        {"role": "system", "content": non_obviousness_grade_system(few_shot)},
        {
            "role": "user",
            "content": non_obviousness_grade_user(
                json.dumps({"candidates": frozen_candidates}, ensure_ascii=False),
                priors_block,
                mechanism_block,
            ),
        },
    ]


def parse_non_obviousness(text: str) -> Optional[dict[str, Any]]:
    """Parse call 2's verdict.

    Returns None when the judge declares the candidate set invalid or emits an
    unrecognized level, so the caller retries rather than reading a failed
    derivation as evidence of non-obviousness.
    """
    obj = parse_json_object(text)
    if obj is None:
        return None
    if obj.get("candidate_set_valid") is False:
        return None

    level = obj.get("obviousness")
    if not isinstance(level, str) or level.strip().lower() not in OBVIOUSNESS_LEVELS:
        return None
    level = level.strip().lower()

    return {
        "central_mechanism": obj.get("central_mechanism", ""),
        "closest_candidate_id": obj.get("closest_candidate_id", "NONE"),
        "shared_steps": obj.get("shared_steps") or [],
        "missing_consequential_steps": obj.get("missing_consequential_steps") or [],
        "obviousness": level,
        "hardest_to_guess_step": obj.get("hardest_to_guess_step", ""),
        "confidence": obj.get("confidence"),
        "score": OBVIOUSNESS_SCORE[level],
    }


def aggregate_non_obviousness(
    verdicts: list[Optional[dict[str, Any]]], denominator: int
) -> dict[str, Any]:
    """Mean over the fixed denominator; an unscored proposal contributes zero."""
    scored = [v for v in verdicts if v is not None]
    total = sum(v["score"] for v in scored)
    by_level = {lvl: sum(1 for v in scored if v["obviousness"] == lvl) for lvl in OBVIOUSNESS_LEVELS}
    return {
        "mechanism_non_obviousness": total / denominator if denominator else 0.0,
        "n_scored": len(scored),
        "n_unscored": denominator - len(scored),
        "by_level": by_level,
        "denominator": denominator,
    }
