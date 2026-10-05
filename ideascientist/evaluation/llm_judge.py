"""LLM-as-judge transport shared by every evaluation metric.

All judge calls go through :func:`judge_json`, which sends a system and user
prompt and parses a single JSON object out of the reply. The reported primary
judge is Qwen3.6-27B with reasoning disabled; override with
``EVAL_JUDGE_MODEL`` or the ``model=`` argument to reproduce the cross-judge
agreement analysis.

Endpoints and keys resolve through :mod:`ideascientist.utils.llm`.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Optional

from ideascientist.utils.llm import DEFAULT_MODEL, LLMSettings, chat

DEFAULT_JUDGE_MODEL = os.environ.get("EVAL_JUDGE_MODEL") or DEFAULT_MODEL

# Reasoning off, as reported. A judgement needs no derivation, and a reasoning
# block consumes the reply budget before the JSON is emitted.
JUDGE_EXTRA_BODY: dict[str, Any] = {"chat_template_kwargs": {"enable_thinking": False}}


def _extract_json(raw: str) -> Optional[dict[str, Any]]:
    """Pull the first balanced ``{...}`` JSON object out of a model reply.

    Tolerant of ```` ```json ```` fences and trailing prose, which judge models
    occasionally add despite instructions.
    """
    if not raw:
        return None
    s = raw.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[-1]
        if s.rstrip().endswith("```"):
            s = s.rstrip()[:-3]
        s = s.strip()
    # Greedy first-object match, then progressively shrink on decode failure.
    start = s.find("{")
    if start == -1:
        return None
    # Try the largest candidate first (handles nested objects), shrinking the
    # closing brace inward until it parses.
    end = s.rfind("}")
    while end > start:
        chunk = s[start : end + 1]
        try:
            obj = json.loads(chunk)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
        end = s.rfind("}", start, end)
    return None


def judge_json(
    system: str,
    user: str,
    *,
    model: Optional[str] = None,
    settings: Optional[LLMSettings] = None,
    max_tokens: int = 2000,
    max_attempts: int = 3,
) -> dict[str, Any]:
    """Call the judge model and return the parsed JSON object.

    Retries up to ``max_attempts`` times, appending an increasingly firm
    "return ONLY JSON" nudge when parsing fails. Raises ``RuntimeError`` if no
    JSON could be parsed after all attempts.

    When ``settings`` is provided the call uses that endpoint instead of the
    configured default, which is how judge calls are routed to a local vLLM
    pool during GRPO training.
    """
    model = model or (settings.model if settings else None) or DEFAULT_JUDGE_MODEL
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    chat_kw: dict[str, Any] = {"model": model, "max_tokens": max_tokens,
                               "extra_body": JUDGE_EXTRA_BODY}
    if settings is not None:
        chat_kw["settings"] = settings
    last_raw = ""
    for attempt in range(max_attempts):
        raw = chat(messages, **chat_kw) or ""
        last_raw = raw
        obj = _extract_json(raw)
        if obj is not None:
            return obj
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": raw},
            {
                "role": "user",
                "content": "Your previous reply was not valid JSON. Reply with "
                "ONLY a single JSON object, no prose, no code fences.",
            },
        ]
    raise RuntimeError(
        f"judge model returned no parseable JSON after {max_attempts} attempts; "
        f"last reply: {last_raw[:500]!r}"
    )


def clamp_score(val: Any, lo: float = 1.0, hi: float = 10.0) -> Optional[float]:
    """Coerce a model-emitted score into ``[lo, hi]``; ``None`` if unparseable."""
    if val is None:
        return None
    try:
        f = float(val)
    except (TypeError, ValueError):
        # Sometimes models write "8/10" or "8 out of 10".
        m = re.search(r"-?\d+(?:\.\d+)?", str(val))
        if not m:
            return None
        f = float(m.group(0))
    return max(lo, min(hi, f))
