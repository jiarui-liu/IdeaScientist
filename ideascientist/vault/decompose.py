"""Idea decomposition: paper full text -> results-masked record.

One LLM call per paper turns the full text into the schema in
:mod:`ideascientist.vault.schema`. The extraction prompt forbids measured
outcomes and asks instead for the evaluation the authors *propose* and the
falsifiable predictions their method implies, so the stored record describes the
direction a human team pursued without revealing whether it worked.

The JSON repair ladder exists because open models reliably emit raw LaTeX inside
string values. Strict parsing is always tried first.
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional

from ideascientist.utils.llm import LLMSettings, chat_with_usage
from ideascientist.vault.schema import RESULTS_MASKED_SCHEMA

EXTRACTION_SYSTEM_PROMPT = """\
You are a research paper analysis assistant. Your task is to read a published \
research paper and extract its content into a STRUCTURED JSON object following \
the schema below.

IMPORTANT — WHAT TO EXTRACT AND WHAT NOT TO:
- Extract the paper's PROBLEM, METHOD, MODEL ARCHITECTURE, and DESIGN CHOICES.
- Do NOT extract actual experimental results, metric numbers, benchmark scores, \
or any measured outcomes. Instead, describe what evaluation the authors PROPOSE \
(benchmarks, baselines, metrics, ablations) and what falsifiable predictions \
their method implies — as if the experiments had NOT yet been run.
- Extract ONLY information present in the paper. Do NOT fabricate or hallucinate.
- If a field cannot be filled from the paper, OMIT the entire key.
- For references, use the citation keys as they appear in the paper (e.g. author names + year).
- Output ONLY a single valid JSON object — no markdown fences, no prose before or after.
- Be precise: use exact model names, dataset names from the paper.

SCHEMA (omit any field whose value would be "", [], null, or "Not applicable"):
""" + RESULTS_MASKED_SCHEMA

EXTRACTION_USER_PROMPT = """\
Extract structured fields from the following published research paper into \
the JSON schema described in your instructions.

Paper title: {title}

--- FULL TEXT ---
{fulltext}
--- END FULL TEXT ---

Output ONLY the JSON object. No markdown fences, no commentary."""

_TOP_LEVEL_KEYS = (
    "topic_relevance", "one_sentence_thesis", "core_problem", "key_novelty",
    "problem_definition", "key_terms", "related_work", "method",
)

_VALID_JSON_ESCAPES = set('"\\/bfnrt')
_HEX = set("0123456789abcdefABCDEF")
_SINGLE_QUOTED_KEY = re.compile(r"([{,]\s*)'([A-Za-z_][A-Za-z0-9_]*)'(\s*:)")


def _strip_thinking(text: str) -> str:
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()


def _escape_stray_backslashes(text: str) -> str:
    """Double backslashes that do not begin a valid JSON escape.

    Raw LaTeX in a string value is invalid JSON. ``\\u`` is only an escape when
    followed by four hex digits, so ``\\underline`` must be doubled too.
    """
    out: list[str] = []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch != "\\":
            out.append(ch)
            i += 1
            continue
        nxt = text[i + 1] if i + 1 < n else ""
        if nxt == "u":
            hexq = text[i + 2:i + 6]
            if len(hexq) == 4 and all(c in _HEX for c in hexq):
                out.append("\\u")
                i += 2
            else:
                out.append("\\\\")
                i += 1
        elif nxt in _VALID_JSON_ESCAPES:
            out.extend((ch, nxt))
            i += 2
        else:
            out.append("\\\\")
            i += 1
    return "".join(out)


def _escape_inner_quotes(text: str) -> str:
    """Escape bare double quotes inside a string value.

    A quote legitimately closes a string only when the next non-space character
    is one of ``,}]:`` or EOF; anything else is an inner quote.
    """
    out: list[str] = []
    i, n = 0, len(text)
    instr = False
    while i < n:
        ch = text[i]
        if not instr:
            out.append(ch)
            if ch == '"':
                instr = True
            i += 1
            continue
        if ch == "\\":
            out.append(ch)
            if i + 1 < n:
                out.append(text[i + 1])
                i += 2
            else:
                i += 1
            continue
        if ch == '"':
            j = i + 1
            while j < n and text[j] in " \t\r\n":
                j += 1
            if (text[j] if j < n else "") in ",}]:" or j >= n:
                out.append(ch)
                instr = False
            else:
                out.append('\\"')
            i += 1
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def extract_json(text: str) -> dict:
    """Parse a JSON object from model output, repairing common defects."""
    text = _strip_thinking(text)
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```\s*$", "", text).strip()
    if not text:
        raise ValueError("empty model output (no JSON content)")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    if start == -1:
        return json.loads(_escape_stray_backslashes(text))

    body = text[start:]
    for repair in (
        lambda s: s,
        _escape_stray_backslashes,
        lambda s: _escape_inner_quotes(_escape_stray_backslashes(s)),
        lambda s: _SINGLE_QUOTED_KEY.sub(
            r'\1"\2"\3', _escape_inner_quotes(_escape_stray_backslashes(s))
        ),
    ):
        try:
            obj, _ = json.JSONDecoder().raw_decode(repair(body))
            return obj
        except json.JSONDecodeError:
            continue
    raise ValueError("could not parse a JSON object from model output")


def extract_json_with_title(
    text: str, known_title: str, anchor_keys: tuple = _TOP_LEVEL_KEYS
) -> dict:
    """Parse model output, splicing in a known title if that field is corrupted.

    ``title`` is copied verbatim from the paper and so is the field most likely
    to carry LaTeX display quotes or math markup that defeats generic repair.
    We already hold the authoritative title, so on total failure we drop
    everything between ``"title"`` and the next known top-level key.
    """
    try:
        return extract_json(text)
    except (json.JSONDecodeError, ValueError):
        pass
    body = _strip_thinking(text)
    body = re.sub(r"^```(?:json)?\s*", "", body)
    body = re.sub(r"\s*```\s*$", "", body).strip()
    start = body.find("{")
    if start == -1:
        raise ValueError("no JSON object found")
    body = body[start:]
    tm = re.search(r'"title"\s*:', body)
    if not tm:
        raise ValueError("no title field to rescue")
    for k in anchor_keys:
        km = re.search(r'"' + k + r'"\s*:', body[tm.end():])
        if km:
            rebuilt = (
                '{\n  "title": '
                + json.dumps(known_title, ensure_ascii=False)
                + ",\n  "
                + body[tm.end() + km.start():]
            )
            return extract_json(rebuilt)
    raise ValueError("no anchor key after title to rescue")


def decompose_paper(
    title: str,
    fulltext: str,
    *,
    settings: Optional[LLMSettings] = None,
    model: Optional[str] = None,
    max_tokens: int = 16384,
    max_chars: int = 180_000,
) -> dict[str, Any]:
    """Decompose one paper into its structured record."""
    if len(fulltext) > max_chars:
        fulltext = fulltext[:max_chars] + "\n\n[... truncated due to length ...]"

    user = EXTRACTION_USER_PROMPT.format(title=title or "Unknown", fulltext=fulltext)

    content, usage = chat_with_usage(
        [{"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
         {"role": "user", "content": user}],
        settings=settings,
        model=model,
        temperature=0.6,
        max_tokens=max_tokens,
        extra_body={"top_p": 0.95, "top_k": 20},
    )
    record = extract_json_with_title(content, title or "")
    record["_usage"] = usage
    return record
