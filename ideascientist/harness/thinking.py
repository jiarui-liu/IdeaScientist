"""Shared history-think stripping primitives.

Used by both the production client and the GRPO training environment so the policy
is defined exactly once. Pure functions, no I/O, no heavy imports — safe to import
from either side.

Background: Qwen "thinking" models emit ``<think>...</think>`` reasoning before
each turn's answer. Best practice (and their chat template) is to drop *prior*
turns' think from history while keeping the *current* turn's. Whether we actually
strip is gated by a ``strip_history_think`` flag on each caller; these helpers are
the shared mechanism, not the policy.
"""
from __future__ import annotations

import re
from typing import Any

# A fully-bracketed <think>...</think> span (DOTALL so it spans newlines).
_THINK_SPAN_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
# An unclosed <think> running to end-of-text (truncated / cut-off reasoning).
_THINK_OPEN_TO_END_RE = re.compile(r"<think>.*\Z", re.DOTALL)


def after_think(text: str) -> str:
    """Return the text after a closing ``</think>`` if the model emitted one.

    Used for tool-call parsing (the tool call lives after the reasoning):
    - if ``</think>`` is present, return everything after the last one;
    - if only an unclosed ``<think>`` is present, return "" (it's all reasoning);
    - otherwise return ``text`` unchanged.

    The single copy: ``harness/client.py`` and the training environment both
    import it, so inference and rollouts strip think blocks identically.
    """
    if "</think>" in text:
        return text.rsplit("</think>", 1)[-1]
    if "<think>" in text:
        return ""
    return text


def strip_leading_think(text: str) -> str:
    """Strip only a LEADING ``<think>...</think>`` block, keeping the rest verbatim.

    Unlike :func:`after_think` (which returns the tail after the LAST ``</think>``),
    this removes only the reasoning that opens the message. Use this for parsing
    tool calls out of assistant TEXT: a tool call's JSON payload can itself contain
    a literal ``</think>`` (e.g. a report about reasoning models), and the
    last-split would truncate the payload and silently drop the call.

    - If the text (after leading whitespace) starts with ``<think>``, drop through
      its first ``</think>`` (or, if unclosed, treat it all as reasoning → "").
    - Otherwise return ``text`` unchanged.
    """
    s = text.lstrip()
    if not s.startswith("<think>"):
        return text
    close = s.find("</think>")
    if close == -1:
        return ""
    return s[close + len("</think>"):]


def strip_think_spans(text: str) -> str:
    """Remove ``<think>...</think>`` reasoning from ``text``, keeping everything
    else (both the text before a span and the answer after it).

    Handles closed spans and a trailing unclosed ``<think>`` (to EOT). Unlike
    :func:`after_think` (which keeps only the tail after the last ``</think>``),
    this preserves non-think content on both sides — the right behavior when
    condensing a prior assistant turn's history down to just its answer.
    """
    if "<think>" not in text:
        return text
    text = _THINK_SPAN_RE.sub("", text)
    text = _THINK_OPEN_TO_END_RE.sub("", text)
    return text


def strip_prior_turn_thinking(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return a copy of ``messages`` with ``<think>...</think>`` removed from the
    ``content`` of every assistant message EXCEPT the last assistant message.

    Mirrors the Qwen chat template's "no thinking content in history" behavior:
    the current (last) assistant turn keeps its reasoning; earlier ones are
    condensed to their answers. Non-assistant messages pass through untouched.

    The ``_harmony_reasoning`` key (gpt-oss ``/responses`` reasoning state, which
    that backend requires echoed back) is preserved — only string ``content`` is
    rewritten. The returned list is a new list; only rewritten messages are
    shallow-copied, so callers may safely reuse the result.

    Leading newlines left behind by the removed span are dropped, so a turn shaped
    ``"<think>...</think>\\n\\nanswer"`` (what both the model and
    :func:`inline_reasoning_content` produce) condenses to exactly ``"answer"``.
    """
    last_assistant = -1
    for i, m in enumerate(messages):
        if m.get("role") == "assistant":
            last_assistant = i

    out: list[dict[str, Any]] = []
    for i, m in enumerate(messages):
        if m.get("role") == "assistant" and i != last_assistant:
            content = m.get("content")
            if isinstance(content, str) and "<think>" in content:
                m = {**m, "content": strip_think_spans(content).lstrip("\n")}
        out.append(m)
    return out


def inline_reasoning_content(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Fold each assistant message's ``reasoning_content`` back into its ``content``
    as a leading ``<think>...</think>`` block.

    Why this is needed: vLLM served with ``--reasoning-parser qwen3`` splits the
    model's ``<think>`` block out of ``content`` into a separate
    ``reasoning_content`` response field. On the way BACK IN, however, vLLM's
    request parsing keeps only the standard OpenAI message keys — a
    ``reasoning_content`` key on an input message is silently dropped before the
    chat template runs, so a history turn carrying the field renders exactly as
    one carrying nothing. The only way to put a prior turn's reasoning back into
    the prompt over
    ``/chat/completions`` is inline, inside ``content``, where the Qwen chat
    template's ``'</think>' in content`` branch picks it up again.

    This restores train/deploy parity: the GRPO rollout is one append-only raw-text
    sequence in which every turn's reasoning stays visible to later turns (the
    Qwen template agrees — inside a ``<tool_response>`` tool loop it renders
    history ``<think>`` blocks rather than dropping them).

    No-ops for messages that have no string ``reasoning_content``, whose reasoning
    is blank, or whose ``content`` already carries a ``<think>`` block (so this is
    idempotent and never double-wraps). Returns a new list; only rewritten
    messages are shallow-copied. The ``reasoning_content`` key is left in place —
    it is inert for servers that ignore it.

    NOTE: the ``<think>`` tag pair is the Qwen convention. Callers must not apply
    this to non-Qwen reasoning families (e.g. gemma4's ``<|channel>thought``);
    ``AgenticLLMClient`` gates it on the model name for exactly that reason.
    """
    out: list[dict[str, Any]] = []
    for m in messages:
        if m.get("role") != "assistant":
            out.append(m)
            continue
        reasoning = m.get("reasoning_content")
        content = m.get("content")
        if not isinstance(content, str):
            content = "" if content is None else str(content)
        if (
            isinstance(reasoning, str)
            and reasoning.strip()
            and "<think>" not in content
        ):
            m = {**m, "content": f"<think>\n{reasoning.strip()}\n</think>\n\n{content}"}
        out.append(m)
    return out
