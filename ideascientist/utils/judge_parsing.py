"""Robust extraction of rubric scores from LLM judge responses."""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Optional


def _sanitize_json_escapes(text: str) -> str:
    out: list[str] = []
    in_string = False
    i = 0
    while i < len(text):
        char = text[i]
        if not in_string:
            out.append(char)
            if char == '"':
                in_string = True
            i += 1
            continue
        if char == '"':
            out.append(char)
            in_string = False
            i += 1
            continue
        if char != "\\":
            out.append(char)
            i += 1
            continue

        following = text[i + 1] if i + 1 < len(text) else ""
        if following in '"\\/bfnrt':
            out.extend((char, following))
            i += 2
        elif following == "u" and re.fullmatch(
            r"[0-9a-fA-F]{4}", text[i + 2 : i + 6]
        ):
            out.append(text[i : i + 6])
            i += 6
        else:
            out.append("\\\\")
            i += 1
    return "".join(out)


def _candidate_bodies(text: str) -> list[str]:
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1]
    text = re.sub(r"^\s*```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```\s*$", "", text)
    bodies = [text.strip()]
    repaired = _sanitize_json_escapes(bodies[0])
    if repaired != bodies[0]:
        bodies.append(repaired)
    return bodies


def _decode_first_object(body: str) -> Optional[dict[str, Any]]:
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", body):
        try:
            candidate, _ = decoder.raw_decode(body, match.start())
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict):
            return candidate
    return None


def _salvage_scores(
    body: str,
    item_ids: list[str],
    coerce_score: Callable[[Any], Optional[int]],
) -> dict[str, dict[str, Any]]:
    found: dict[str, dict[str, Any]] = {}
    for rubric_id in item_ids:
        start = re.search(rf'"{re.escape(rubric_id)}"\s*:\s*\{{', body)
        if not start:
            continue
        tail = body[start.end() :]
        boundaries = [
            match.start()
            for other_id in item_ids
            if other_id != rubric_id
            if (match := re.search(rf'"{re.escape(other_id)}"\s*:', tail))
        ]
        if boundaries:
            tail = tail[: min(boundaries)]
        score_match = re.search(
            r'"score"\s*:\s*("(?:\\.|[^"\\])*"|-?\d+(?:\.\d+)?|true|false)',
            tail,
            flags=re.IGNORECASE,
        )
        if not score_match:
            continue
        raw_score = score_match.group(1)
        try:
            value = json.loads(raw_score)
        except json.JSONDecodeError:
            value = raw_score.strip('"')
        score = coerce_score(value)
        if score is not None:
            found[rubric_id] = {
                "score": score,
                "explanation": "(score salvaged from malformed judge JSON)",
            }
    return found


def parse_judge_scores(
    text: str,
    item_ids: list[str],
    coerce_score: Callable[[Any], Optional[int]],
) -> tuple[dict[str, dict[str, Any]], set[str]]:
    """Extract all usable rubric scores, repairing common model JSON defects.

    Strict JSON is preferred. On failure, invalid LaTeX escapes are repaired and
    parsing is retried. As a final fallback, each requested rubric's score is
    recovered independently so a malformed explanation does not discard valid
    supervision.
    """
    defaults: dict[str, dict[str, Any]] = {
        rubric_id: {"score": 0, "explanation": "(missing from judge reply)"}
        for rubric_id in item_ids
    }
    if not text:
        return defaults, set()

    bodies = _candidate_bodies(text)
    obj = next(
        (candidate for body in bodies if (candidate := _decode_first_object(body))),
        None,
    )
    parsed: set[str] = set()
    if obj is not None:
        for rubric_id in item_ids:
            if rubric_id not in obj:
                continue
            value = obj[rubric_id]
            if isinstance(value, dict):
                score = coerce_score(value.get("score"))
                explanation = str(value.get("explanation", ""))
            else:
                score = coerce_score(value)
                explanation = ""
            if score is not None:
                defaults[rubric_id] = {
                    "score": score,
                    "explanation": explanation,
                }
                parsed.add(rubric_id)

    if parsed != set(item_ids):
        for body in bodies:
            for rubric_id, record in _salvage_scores(
                body, item_ids, coerce_score
            ).items():
                if rubric_id not in parsed:
                    defaults[rubric_id] = record
                    parsed.add(rubric_id)
    return defaults, parsed
