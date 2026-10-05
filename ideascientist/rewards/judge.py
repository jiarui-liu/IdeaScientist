"""Shared judge plumbing for the three role rewards.

The judge is the same open backbone the system runs on, with reasoning off and
the model's recommended sampling parameters rather than greedy decoding. That
matters for the retry policy below: each attempt samples a fresh reply, so a
retry after a malformed one has a real chance of parsing where a greedy repeat
would reproduce the same broken text.

If no attempt parses, the rollout is dropped from the GRPO batch rather than
scored with a fabricated value. :data:`JUDGE_FAILED_REWARD_SENTINEL` sits
outside the legitimate reward range so the driver can identify those rows
unambiguously and set their loss multiplier to zero.
"""

from __future__ import annotations

import os
import re
from typing import Any, Optional

from ideascientist.utils.judge_parsing import parse_judge_scores
from ideascientist.utils.llm import LLMSettings, chat_with_usage

JUDGE_TEMPERATURE: float = 0.7
JUDGE_SAMPLING_EXTRA: dict[str, Any] = {
    "top_p": 0.80,
    "top_k": 20,
    "presence_penalty": 1.5,
    # Reasoning off, as reported. Scoring a rubric item is a judgement, not a
    # derivation, and a reasoning block would consume the reply budget below
    # before the JSON is emitted — every call would fail to parse.
    "chat_template_kwargs": {"enable_thinking": False},
}
JUDGE_MAX_PARSE_RETRIES: int = 3

# Bounds one judge call end to end. urllib's socket timeout is per-recv and does
# not bound an endpoint that trickles bytes, which under reward-phase concurrency
# would block a reward thread for minutes.
JUDGE_CALL_DEADLINE_S: float = float(os.environ.get("IDEASCIENTIST_JUDGE_DEADLINE", "90"))

JUDGE_FAILED_REWARD_SENTINEL: float = -999.0


class JudgeParseError(RuntimeError):
    """A batched judge reply could not be parsed after every retry.

    Carries the raw text of every attempt, including the malformed ones: a
    dropped rollout is the case most worth inspecting offline.
    """

    def __init__(
        self,
        message: str,
        *,
        group_ids: Optional[list[str]] = None,
        attempts: Optional[list[dict]] = None,
        last_reason: str = "",
    ) -> None:
        super().__init__(message)
        self.group_ids: list[str] = group_ids or []
        self.attempts: list[dict] = attempts or []
        self.last_reason: str = last_reason
        self.group_name: str = ""
        self.groups_meta_so_far: dict[str, dict] = {}


def coerce_score(val: Any) -> Optional[int]:
    """Coerce a judge-returned score to an int, or None if unparseable."""
    if isinstance(val, bool):
        return int(val)
    if isinstance(val, (int, float)):
        return int(val)
    if isinstance(val, str):
        m = re.search(r"-?\d+", val)
        if m:
            return int(m.group(0))
    return None


def parse_scores(text: str, item_ids: list[str]) -> tuple[dict[str, dict], set[str]]:
    """Parse a batched reply, reporting which ids were actually recovered."""
    return parse_judge_scores(text, item_ids, coerce_score)


def render_rubric_item(item: dict, kind: str) -> str:
    parts = [f"- id: {item['id']}", f"- kind: {kind}", f"- criterion: {item['criterion']}"]
    if item.get("explanation"):
        parts.append(f"- explanation: {item['explanation']}")
    if item.get("examples"):
        parts.append("- examples: " + " | ".join(item["examples"]))
    return "\n".join(parts)


def judge_groups(rubric: dict, match_ids: "set[str] | frozenset[str]") -> list[tuple[str, str, list[dict]]]:
    """Partition a rubric's scored items into the three batched judge calls.

    The split is what lets one call score intrinsic quality and another score
    agreement with the reference without either seeing the other's instruction.
    A positive item that is not a ``matches_reference_*`` sub-item falls into
    ``base_positive``, so adding a rubric item cannot silently drop it.
    """
    base_pos, match = [], []
    for item in rubric["positive_rubrics"]:
        (match if item["id"] in match_ids else base_pos).append(item)
    return [
        ("base_positive", "POSITIVE", base_pos),
        ("matches_reference", "POSITIVE", match),
        ("negative", "NEGATIVE", list(rubric["negative_rubrics"])),
    ]


def render_rubric_items(items: list[dict], kind: str) -> str:
    return "\n\n".join(render_rubric_item(it, kind) for it in items)


def judge_items(
    prompt: str,
    item_ids: list[str],
    settings: LLMSettings,
    *,
    max_parse_retries: int = JUDGE_MAX_PARSE_RETRIES,
    max_tokens: int = 1024,
) -> tuple[dict[str, dict], dict]:
    """Score a group of rubric items in one call, retrying on parse failure.

    An attempt succeeds only when every id in the group parses to a coercible
    score; scores recovered by earlier attempts are kept across retries.

    Returns ``(scored, meta)`` where ``meta`` carries the full per-attempt log.
    Raises :class:`JudgeParseError` if no combination of attempts covers the
    group.
    """
    if not item_ids:
        return {}, {"attempts": [], "n_retries": 0, "raw_reply": ""}

    want = set(item_ids)
    recovered: dict[str, dict] = {}
    attempts: list[dict] = []
    last_reason = "no judge attempt made"

    for attempt_i in range(max_parse_retries + 1):
        try:
            content, _usage = chat_with_usage(
                [{"role": "user", "content": prompt}],
                settings=settings,
                max_tokens=max_tokens,
                temperature=JUDGE_TEMPERATURE,
                total_deadline=JUDGE_CALL_DEADLINE_S,
                extra_body=JUDGE_SAMPLING_EXTRA,
            )
        except Exception as e:  # noqa: BLE001 - network failure is retryable
            last_reason = f"judge call error: {type(e).__name__}: {e}"
            attempts.append({"attempt": attempt_i, "raw_reply": None, "outcome": last_reason})
            continue

        scored, parsed = parse_scores(content, item_ids)
        for rubric_id in parsed:
            recovered.setdefault(rubric_id, scored[rubric_id])
        if set(recovered) == want:
            attempts.append({"attempt": attempt_i, "raw_reply": content, "outcome": "parsed"})
            return recovered, {
                "attempts": attempts,
                "n_retries": attempt_i,
                "raw_reply": content,
            }
        last_reason = f"unparsed ids after judge reply: {sorted(want - set(recovered))}"
        attempts.append({"attempt": attempt_i, "raw_reply": content, "outcome": last_reason})

    raise JudgeParseError(
        f"group [{', '.join(item_ids)}] failed to parse after "
        f"{max_parse_retries + 1} attempts ({last_reason})",
        group_ids=item_ids,
        attempts=attempts,
        last_reason=last_reason,
    )
