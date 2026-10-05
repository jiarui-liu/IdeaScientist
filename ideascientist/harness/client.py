"""OpenAI-compatible tool-use client with full request/response logging.

Every call is appended to ``llm_calls.json`` in the run directory; the run log
is the only record of what each agent saw and did.

Two robustness paths exist because served open models misbehave in ways the API
contract does not cover. ``text_tools`` carries tool definitions in the prompt
and parses calls back out of the response, for endpoints whose native function
calling is unavailable. The JSON-repair ladder recovers tool calls whose
arguments are malformed, which is common enough at this scale that dropping
them would bias the rollout distribution.
"""

from __future__ import annotations

import ast
import asyncio
import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from ideascientist.harness.runlog import StreamingJsonArrayWriter
from ideascientist.harness.thinking import (
    inline_reasoning_content,
    strip_leading_think as _strip_leading_think,
    strip_prior_turn_thinking,
)
from ideascientist.utils.llm import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    build_opener,
    uses_completion_tokens,
)


@dataclass
class CallRecord:
    call_id: int
    step_name: str
    system_prompt: str
    user_prompt: str
    response: str
    model: str
    temperature: float
    latency_s: float
    input_tokens: int
    output_tokens: int
    timestamp: str
    error: str = ""


class LoggedLLMClient:
    """LLM client that logs every call to JSONL + stdout."""

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        log_dir: Path | str = ".",
        proxy: str = "",
        shared_log: "SharedLogHandle | None" = None,
    ):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.log_dir = Path(log_dir)
        self.proxy = proxy
        self._call_counter = 0
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        # Size of the prompt (history) sent on the MOST RECENT call, i.e. how
        # full the context window currently is. Distinct from the cumulative
        # _total_input_tokens. 0 until the first call returns usage.
        self.last_prompt_tokens = 0
        self._log_path = self.log_dir / "llm_calls.json"
        self._log_writer = None
        # When set, log records are appended to a shared per-role/paper file
        # (with conversation_id + session_seq injected) instead of this client's
        # own llm_calls.json. The shared writer is owned by a SharedLogRegistry,
        # so this client must NOT close it — the registry finalizes it.
        self._shared_log = shared_log
        self._step_name = ""

    def set_step(self, name: str) -> None:
        self._step_name = name

    def _account_usage(self, usage: dict) -> None:
        self.last_prompt_tokens = usage.get("prompt_tokens", 0)
        self._total_input_tokens += usage.get("prompt_tokens", 0)
        self._total_output_tokens += usage.get("completion_tokens", 0)

    def _ensure_log(self):
        if self._shared_log is not None:
            return  # writes go to the registry-owned shared file
        if self._log_writer is None:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self._log_writer = StreamingJsonArrayWriter(self._log_path)

    def _make_opener(self):
        # Preserves POST across 307 redirects; with an empty proxy urllib keeps
        # honouring HTTP(S)_PROXY.
        return build_opener(self.proxy)

    def _http_call(self, system: str, user: str, max_retries: int = 6) -> tuple[str, dict]:
        return self._http_call_openai(system, user, max_retries)

    def _http_call_openai(self, system: str, user: str, max_retries: int = 6) -> tuple[str, dict]:
        url = f"{self.base_url}/chat/completions"
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        # gpt-5 / o-series require max_completion_tokens and reject a non-default
        # temperature; other models take max_tokens + temperature.
        if uses_completion_tokens(self.model):
            body["max_completion_tokens"] = self.max_tokens
        else:
            body["max_tokens"] = self.max_tokens
            body["temperature"] = self.temperature
        payload = json.dumps(body).encode("utf-8")

        opener = self._make_opener()

        for attempt in range(max_retries):
            req = urllib.request.Request(
                url,
                data=payload,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                },
                method="POST",
            )
            try:
                resp = opener.open(req, timeout=120)
                body = json.loads(resp.read().decode("utf-8"))
                text = body["choices"][0]["message"]["content"]
                usage = body.get("usage", {})
                return text, usage
            except urllib.error.HTTPError as e:
                if e.code == 429 and attempt < max_retries - 1:
                    wait = 2 ** (attempt + 1)
                    print(f"  rate-limited, retrying in {wait}s...", flush=True)
                    time.sleep(wait)
                    continue
                raise

        return "", {}

    def _log_and_print(self, record: CallRecord) -> None:
        self._ensure_log()
        entry = {
            "call_id": record.call_id,
            "step": record.step_name,
            "model": record.model,
            "temperature": record.temperature,
            "system_prompt": record.system_prompt,
            "user_prompt": record.user_prompt,
            "response": record.response,
            "input_tokens": record.input_tokens,
            "output_tokens": record.output_tokens,
            "latency_s": round(record.latency_s, 2),
            "timestamp": record.timestamp,
            "error": record.error,
        }
        if self._shared_log is not None:
            self._shared_log.write(entry)
        else:
            self._log_writer.write(entry)

        # Real-time stdout
        sep = "─" * 72
        print(f"\n{sep}", flush=True)
        print(f"📤 LLM Call #{record.call_id}  │  step: {record.step_name}  │  {record.model}", flush=True)
        print(f"   tokens: {record.input_tokens} in / {record.output_tokens} out  │  {record.latency_s:.1f}s", flush=True)
        if record.error:
            print(f"   ❌ ERROR: {record.error}", flush=True)
        else:
            preview = record.response[:300].replace("\n", " ")
            if len(record.response) > 300:
                preview += "..."
            print(f"   → {preview}", flush=True)
        print(sep, flush=True)

    def call(self, system: str, user: str) -> str:
        """Sync LLM call with logging. Use this as llm_call in stage modules."""
        self._call_counter += 1
        call_id = self._call_counter
        ts = datetime.now(timezone.utc).isoformat()

        t0 = time.monotonic()
        try:
            text, usage = self._http_call(system, user)
            latency = time.monotonic() - t0
            in_tok = usage.get("prompt_tokens", 0)
            out_tok = usage.get("completion_tokens", 0)
            self._account_usage(usage)

            record = CallRecord(
                call_id=call_id,
                step_name=self._step_name,
                system_prompt=system,
                user_prompt=user,
                response=text,
                model=self.model,
                temperature=self.temperature,
                latency_s=latency,
                input_tokens=in_tok,
                output_tokens=out_tok,
                timestamp=ts,
            )
            self._log_and_print(record)
            return text

        except Exception as exc:
            latency = time.monotonic() - t0
            record = CallRecord(
                call_id=call_id,
                step_name=self._step_name,
                system_prompt=system,
                user_prompt=user,
                response="",
                model=self.model,
                temperature=self.temperature,
                latency_s=latency,
                input_tokens=0,
                output_tokens=0,
                timestamp=ts,
                error=str(exc),
            )
            self._log_and_print(record)
            raise

    async def acall(self, system: str, user: str) -> str:
        """Async LLM call — runs the sync HTTP call in a thread."""
        return await asyncio.to_thread(self.call, system, user)

    def summary(self) -> dict[str, Any]:
        return {
            "total_calls": self._call_counter,
            "total_input_tokens": self._total_input_tokens,
            "total_output_tokens": self._total_output_tokens,
            "total_tokens": self._total_input_tokens + self._total_output_tokens,
            "total_retries_on_parser_error": getattr(
                self, "_total_retries_on_parser_error", 0),
            "log_file": str(self._log_path),
        }

    def close(self) -> None:
        # Shared writers are owned + finalized by the SharedLogRegistry; only
        # close a writer this client opened itself.
        if self._shared_log is None and self._log_writer is not None:
            self._log_writer.close()
            self._log_writer = None


# Raise both when fanning many runs at one served pool: queued requests plus a
# long report_writer generation exceed the 120s default, and the retry then
# lands on a pool that is still busy.
_HTTP_TIMEOUT = int(os.environ.get("IDEASCIENTIST_HTTP_TIMEOUT", "120") or "120")
_MAX_RETRIES_DEFAULT = int(os.environ.get("IDEASCIENTIST_MAX_RETRIES", "6") or "6")


_CONTEXT_OVERFLOW_RE = re.compile(
    r"maximum context length is (\d+) tokens.*?"
    r"requested (\d+) output tokens.*?"
    r"prompt contains at least (\d+) input tokens",
    re.DOTALL | re.IGNORECASE,
)


def _parse_context_overflow(body: str) -> Optional[tuple[int, int, int]]:
    """Parse vLLM's max-context error body. Returns (max_context, requested_out, input_tokens) or None."""
    m = _CONTEXT_OVERFLOW_RE.search(body)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2)), int(m.group(3))




# Model families whose served reasoning is NOT delimited by ``<think>...</think>``,
# so folding a prior turn's ``reasoning_content`` back in with those tags would be
# off-distribution. Matched as substrings against the served model name/path.
# gemma4 uses ``<|channel>thought``; gpt-oss carries reasoning as Responses-API
# entries (handled separately by ``_harmony_reasoning``).
_NON_THINK_MODEL_FAMILIES = ("gemma", "gpt-oss", "gptoss")


# Text tool-call protocol, for endpoints whose native function calling is
# unavailable. Tools are carried in the prompt and parsed back out of the
# response text into OpenAI-shape calls, so the loop behaves identically. The
# training environment parses the same way.


def _parse_bare_calls(text: str, known_names: tuple[str, ...]) -> list[dict[str, Any]]:
    """Fallback parser for bare Python-call syntax ``name(arg=...)``.

    Deliberately conservative: only KNOWN tool names, only calls that cleanly
    ``ast.parse`` into a Call with literal-evaluable keyword arguments. Anything
    ambiguous is skipped, so prose mentioning a tool name cannot spuriously
    trigger a call.
    """
    results: list[dict[str, Any]] = []
    if not known_names:
        return results
    pattern = r"\b(" + "|".join(re.escape(n) for n in known_names) + r")\s*\("
    for m in re.finditer(pattern, text):
        name = m.group(1)
        open_paren = m.end() - 1
        depth = 0
        in_str: Optional[str] = None
        esc = False
        end: Optional[int] = None
        for i in range(open_paren, len(text)):
            ch = text[i]
            if in_str is not None:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == in_str:
                    in_str = None
                continue
            if ch in "\"'":
                in_str = ch
            elif ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    end = i
                    break
        if end is None:
            continue
        call_src = text[m.start():end + 1].strip()
        try:
            node = ast.parse(call_src, mode="eval")
        except (SyntaxError, ValueError):
            continue
        if not isinstance(node.body, ast.Call):
            continue
        args: dict[str, Any] = {}
        ok = True
        for kw in node.body.keywords:
            if kw.arg is None:
                ok = False
                break
            try:
                args[kw.arg] = ast.literal_eval(kw.value)
            except (ValueError, SyntaxError):
                ok = False
                break
        if not ok:
            continue
        results.append({
            "id": f"call_{len(results)}",
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(args, default=str)},
        })
    return results


def _sanitize_json_escapes(fragment: str) -> str:
    """Escape stray backslashes that are illegal JSON escape sequences.

    Models writing LaTeX inside a JSON string arg (e.g. an edit_doc ``content``
    with ``\\(Br(B)=0\\)`` or ``\\alpha``) emit backslashes that JSON forbids —
    JSON only allows ``\\" \\\\ \\/ \\b \\f \\n \\r \\t \\uXXXX``. ``raw_decode``
    then dies with "Invalid \\escape" and the whole tool call is dropped. We
    walk the text tracking string state and double any backslash that does not
    begin a valid JSON escape, leaving already-valid escapes untouched.
    """
    out: list[str] = []
    in_str = False
    i = 0
    n = len(fragment)
    while i < n:
        ch = fragment[i]
        if not in_str:
            out.append(ch)
            if ch == '"':
                in_str = True
            i += 1
            continue
        if ch == '"':
            out.append(ch)
            in_str = False
            i += 1
            continue
        if ch == "\\":
            nxt = fragment[i + 1] if i + 1 < n else ""
            if nxt in '"\\/bfnrt':
                out.append(ch)
                out.append(nxt)
                i += 2
                continue
            if nxt == "u" and re.match(r"[0-9a-fA-F]{4}", fragment[i + 2:i + 6]):
                out.append(fragment[i:i + 6])
                i += 6
                continue
            # Illegal escape (e.g. LaTeX \( \alpha) — double the backslash.
            out.append("\\\\")
            i += 1
            continue
        out.append(ch)
        i += 1
    return out and "".join(out) or fragment


def _repair_tool_call_json(fragment: str) -> dict[str, Any] | None:
    """Best-effort recovery of a ``<tool_call>`` body that isn't valid JSON.

    Qwen intermittently drops the final closing brace/bracket of a tool call
    (e.g. emits ``{"name":..,"arguments":{..]}`` with one ``}`` instead of two),
    which makes ``raw_decode`` fail and silently drops the call — aborting the
    whole agent turn. We take the fragment up to the closing ``</tool_call>``
    (or the next marker / end), strip trailing junk, and append the missing
    ``}``/``]`` to balance open braces/brackets (respecting string/escape state).
    Returns the decoded dict or None if unrecoverable.
    """
    end = fragment.find("</tool_call>")
    if end == -1:
        nxt = fragment.find("<tool_call>")
        end = nxt if nxt != -1 else len(fragment)
    cand = fragment[:end].strip()
    if not cand or cand[0] != "{":
        return None
    # First try: the only defect may be illegal LaTeX-style backslash escapes
    # (\( \alpha ...) in a string arg — sanitize and decode before the brace walk.
    sanitized = _sanitize_json_escapes(cand)
    if sanitized != cand:
        try:
            obj = json.loads(sanitized)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
        cand = sanitized
    # Walk the candidate tracking string state to know how many closers are open.
    stack: list[str] = []
    in_str = False
    esc = False
    for ch in cand:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if stack:
                stack.pop()
    if in_str:
        cand += '"'
    closers = "".join("}" if c == "{" else "]" for c in reversed(stack))
    for attempt in (cand, cand + closers):
        try:
            obj = json.loads(attempt)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def _parse_text_tool_calls(
    text: str, known_names: tuple[str, ...]
) -> list[dict[str, Any]]:
    """Extract tool calls from assistant TEXT into OpenAI-shape tool_calls.

    Strips only a LEADING ``<think>...</think>`` (a literal ``</think>`` can appear
    inside a tool call's JSON payload, so a last-``</think>`` split would drop the
    call); decodes ONE balanced JSON object after each ``<tool_call>`` marker
    (closing tag optional — matches training); falls back to bare-call
    ``name(arg=...)`` for ``known_names`` if no ``<tool_call>`` block parsed.
    """
    if not text:
        return []
    body = _strip_leading_think(text)
    results: list[dict[str, Any]] = []
    decoder = json.JSONDecoder()
    for m in re.finditer(r"<tool_call>\s*", body):
        try:
            obj, _ = decoder.raw_decode(body, m.end())
        except json.JSONDecodeError:
            # Qwen sometimes emits a tool_call with an unbalanced/truncated JSON
            # body (missing final brace). Try to repair rather than drop the call
            # (dropping aborts the whole turn with an empty final message).
            obj = _repair_tool_call_json(body[m.end():])
            if obj is None:
                continue
        except ValueError:
            # A DEGENERATE generation can emit a giant number (e.g. a 5000+ digit
            # integer) as a tool-call arg; json's int conversion then trips
            # Python's int_max_str_digits cap (default 4300) and raises a plain
            # ValueError (NOT JSONDecodeError). Fail SOFT — skip this malformed
            # call rather than letting the exception kill the whole rollout.
            continue
        if not isinstance(obj, dict):
            continue
        results.append({
            "id": f"call_{len(results)}",
            "type": "function",
            "function": {
                "name": obj.get("name", ""),
                "arguments": json.dumps(obj.get("arguments", {})),
            },
        })
    if not results:
        results = _parse_bare_calls(body, known_names)
    return results


# Public alias: the GRPO training env imports this to parse assistant TEXT into
# OpenAI-shape tool_calls, so training and inference use ONE parser (no copies).
parse_text_tool_calls = _parse_text_tool_calls


def _render_tools_as_text(tools: list[dict[str, Any]]) -> str:
    """Render OpenAI tool specs into a system block instructing the model to
    emit ``<tool_call>{...}</tool_call>`` in its text output.

    Mirrors the training env's initial-prompt block that fixed cold-start
    tool emission (JSON syntax + a worked keyword_search example).
    """
    lines: list[str] = [
        "You are an agent that acts ONLY by calling tools. You have NO reliable",
        "internal knowledge of the paper corpus, prior results, or files — every",
        "fact you use MUST come from a tool result. NEVER answer from memory,",
        "NEVER fabricate paper titles, authors, findings, or citations.",
        "",
        "To call a tool, your response must contain a line of the EXACT form:",
        '<tool_call>{"name": "<tool_name>", "arguments": {<json args>}}</tool_call>',
        "",
        "Rules:",
        "- When you need information or need to act, emit tool call(s) — do NOT",
        "  write prose describing what you would do or listing what you 'know'.",
        "- Emit ONE tool call per line. You may write brief reasoning first, but",
        "  the turn is only useful if it ends with real <tool_call> line(s).",
        "- Use the tool JSON exactly; do NOT wrap it in markdown fences.",
        "- Only stop calling tools and write a final answer when the task is",
        "  genuinely complete and all needed artifacts have been written.",
        "",
        "Example — the ONLY acceptable way to search:",
        '<tool_call>{"name": "keyword_search", "arguments": {"queries": ["diffusion model editing"], "k": 10}}</tool_call>',
        "",
        "Available tools:",
    ]
    for t in tools or []:
        fn = t.get("function", t) if isinstance(t, dict) else {}
        name = fn.get("name", "")
        desc = fn.get("description", "")
        params = fn.get("parameters", {"type": "object", "properties": {}})
        lines.append(f"\n- {name}: {desc}")
        lines.append(f"  parameters (JSON schema): {json.dumps(params)}")
    return "\n".join(lines)


def _known_tool_names(tools: list[dict[str, Any]] | None) -> tuple[str, ...]:
    names: list[str] = []
    for t in tools or []:
        fn = t.get("function", t) if isinstance(t, dict) else {}
        n = fn.get("name")
        if n:
            names.append(n)
    return tuple(names)


class AgenticLLMClient(LoggedLLMClient):
    """LLM client that supports tool-use via OpenAI or Anthropic API."""

    def __init__(
        self,
        *args: Any,
        text_tools: bool = False,
        strip_history_think: bool | None = None,
        history_reasoning: bool | None = None,
        extra_body: dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        # When True, tools are carried in the prompt (text protocol) instead of
        # the API ``tools`` param, and parsed back out of the response text.
        # Used for endpoints whose native function-calling is broken/unavailable
        # (e.g. the gpt-5 compat endpoint on this cluster).
        self.text_tools = text_tools
        # Extra top-level fields merged into every /chat/completions payload, e.g.
        # {"chat_template_kwargs": {"enable_thinking": False}} to run a served Qwen
        # model with thinking OFF. Used for non-trained tool subagents (paper_reader)
        # to skip generating reasoning we would only strip — a large rollout speedup.
        # The trained policy is driven by the training loop, not this client, so this
        # never touches policy thinking.
        self.extra_body = extra_body or None
        # When True, drop prior-turn <think>...</think> from history before each
        # call (Qwen "no thinking content in history" best practice), keeping the
        # current turn's think. Mirrors the training-side strip_history_think flag
        # so a policy trained with stripping is served the same way. Default off
        if strip_history_think is None:
            strip_history_think = os.environ.get(
                "IDEASCIENTIST_STRIP_HISTORY_THINK", ""
            ).strip().lower() in ("1", "true", "yes")
        self.strip_history_think = strip_history_think
        # Fold each history turn's reasoning back into its content as an inline
        # <think> block, so the served policy sees the context shape it saw in
        # training, where the rollout is one append-only sequence. Without this the
        # reasoning is lost twice: the server's parser splits it out, and vLLM
        # drops the key on the way back in. "auto" uses a blocklist, since a
        # trained adapter is served under a path carrying no family name.
        if history_reasoning is None:
            _hr = os.environ.get("IDEASCIENTIST_HISTORY_REASONING", "auto").strip().lower()
            if _hr in ("1", "true", "yes", "on"):
                history_reasoning = True
            elif _hr in ("0", "false", "no", "off"):
                history_reasoning = False
            else:  # "auto" (or anything unrecognized): on unless a non-<think> family
                _m = (self.model or "").lower()
                history_reasoning = not any(f in _m for f in _NON_THINK_MODEL_FAMILIES)
        self.history_reasoning = history_reasoning

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        max_retries: int = _MAX_RETRIES_DEFAULT,
    ) -> dict[str, Any]:
        """Send a multi-turn conversation with optional tools.

        Returns the full API response dict in **OpenAI format** regardless
        of the underlying provider.  If the response contains
        ``tool_calls``, the caller should execute them and send results
        back via another ``chat()`` call with the updated messages.
        """
        # History-reasoning restore, then the optional strip — in that order, so
        # ``strip_history_think`` still sees (and condenses) prior-turn think that
        # only existed in the ``reasoning_content`` field. Both are no-ops when
        # their flag is off; neither mutates the caller's list.
        if self.history_reasoning:
            messages = inline_reasoning_content(messages)
        # Strip prior-turn think on the OpenAI/text-tools path only (see __init__).
        # No-op when the flag is off or no <think> is present; never mutates the
        # caller's list (strip_prior_turn_thinking returns a new list).
        if self.strip_history_think:
            messages = strip_prior_turn_thinking(messages)
        return self._chat_openai(messages, tools, max_retries)

    # ------------------------------------------------------------------
    # OpenAI implementation
    # ------------------------------------------------------------------

    def _chat_openai(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        max_retries: int = _MAX_RETRIES_DEFAULT,
    ) -> dict[str, Any]:
        self._call_counter += 1
        call_id = self._call_counter

        # Text-tools mode: carry tools in the prompt and strip the API `tools`
        # param. Build a normalized shallow copy of `messages` (never mutate the
        # caller's list) that the compat endpoint accepts, and remember the tool
        # names so we can parse calls back out of the response text.
        text_tools_on = bool(self.text_tools and tools)
        # Text-tools streams by default; some endpoints drop content otherwise.
        stream_on = text_tools_on and os.environ.get(
            "IDEASCIENTIST_TEXT_TOOLS_NOSTREAM", "").strip().lower() not in ("1", "true", "yes", "on")
        known_names = _known_tool_names(tools) if text_tools_on else ()
        send_messages = (
            self._normalize_text_tools_messages(messages, tools)
            if text_tools_on else messages
        )
        send_tools = None if text_tools_on else tools

        # Effective max-output cap for this call. May be lowered dynamically
        # if the server returns a "context overflow" 400. vLLM's reported
        # input_tokens is a LOWER BOUND ("at least X"), not the true count —
        # if we just shave by safety each time, we converge linearly and burn
        # retries before fitting. So on the FIRST overflow we use the parsed
        # recommendation; on subsequent overflows we HALVE the cap so we
        # converge exponentially toward the true value (true X is somewhere
        # in [reported, max_ctx)). Context-overflow retries use their own
        # counter so they don't consume the main 429/5xx retry budget.
        cap = self.max_tokens
        SAFETY = 256          # leave room for chat-template overhead
        MIN_OUT = 256         # don't bother retrying below this — model won't say anything useful
        ctx_overflow_attempts = 0
        MAX_CTX_OVERFLOW_RETRIES = 8  # halving 8× from any reasonable start hits MIN_OUT

        def _build_payload(out_cap: int) -> bytes:
            p: dict[str, Any] = {"model": self.model, "messages": send_messages}
            if uses_completion_tokens(self.model):
                p["max_completion_tokens"] = out_cap
            else:
                p["max_tokens"] = out_cap
                p["temperature"] = self.temperature
            if send_tools:
                p["tools"] = send_tools
            # Merge caller-supplied extra fields (e.g. chat_template_kwargs to disable
            # thinking on a served Qwen). Set before stream flags so it can't clobber
            # them; keys here are top-level request fields the server understands.
            if self.extra_body:
                p.update(self.extra_body)
            # gpt-oss's chat template rejects ``enable_thinking`` with a 400; extra_body
            # is keyed by role rather than model, so drop it for that family.
            if ("gpt-oss" in (self.model or "").lower() or "gptoss" in (self.model or "").lower()) \
                    and isinstance(p.get("chat_template_kwargs"), dict):
                p["chat_template_kwargs"].pop("enable_thinking", None)
                if not p["chat_template_kwargs"]:
                    p.pop("chat_template_kwargs", None)
            # Text-tools mode MUST stream: some endpoints'
            # NON-streaming aggregator drops this reasoning model's output —
            # it returns completion_tokens>0 but message.content="" — while the
            # streamed deltas carry the real text (the emitted <tool_call>
            # block). So we request a stream and reassemble it below.
            if stream_on:
                p["stream"] = True
                p["stream_options"] = {"include_usage": True}
            return json.dumps(p).encode("utf-8")

        body_bytes = _build_payload(cap)
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        opener = self._make_opener()
        t0 = time.monotonic()
        last_exc: Optional[Exception] = None

        attempt = 0
        # Separate counter for abstain retries (200 OK but no content + no
        # tool_calls — happens with thinking models like Qwen3 when reasoning
        # concludes without committing to either a tool or a text reply).
        # Resampling at temperature > 0 usually breaks out of it.
        abstain_attempts = 0
        MAX_ABSTAIN_RETRIES = 5
        while attempt < max_retries:
            try:
                req = urllib.request.Request(url, data=body_bytes, headers=headers)
                with opener.open(req, timeout=_HTTP_TIMEOUT) as resp:
                    if stream_on:
                        result = self._read_stream_as_result(resp)
                    else:
                        result = json.loads(resp.read().decode("utf-8"))

                latency = time.monotonic() - t0
                msg = result["choices"][0]["message"]
                usage = result.get("usage", {})

                # Text-tools mode: the server can't emit native tool_calls, so
                # parse them out of the response text into OpenAI shape. Done
                # before the abstain check so a valid text call isn't treated
                # as an empty sample.
                if text_tools_on and not msg.get("tool_calls"):
                    parsed = _parse_text_tool_calls(msg.get("content") or "", known_names)
                    if parsed:
                        msg["tool_calls"] = parsed

                # Detect abstain: no tool calls and no content. Retry without
                # consuming the main retry budget (since the server is healthy
                # — just the sample was empty).
                if not msg.get("tool_calls") and not (msg.get("content") or "").strip():
                    if abstain_attempts < MAX_ABSTAIN_RETRIES:
                        abstain_attempts += 1
                        print(f"  abstain (empty content + no tool_calls): "
                              f"resample {abstain_attempts}/{MAX_ABSTAIN_RETRIES}", flush=True)
                        continue
                    # Out of abstain retries — let the empty response through.
                    # Agent loop will exit; record the count so we can audit.

                self._account_usage(usage)

                if abstain_attempts:
                    result.setdefault("_meta", {})["abstain_retries"] = abstain_attempts
                self._log_chat_call(call_id, messages, msg, usage, latency,
                                    extra={"abstain_retries": abstain_attempts}
                                    if abstain_attempts else None)
                return result

            except urllib.error.HTTPError as e:
                if e.code == 400:
                    body = ""
                    try:
                        body = e.read().decode("utf-8", errors="replace")
                    except Exception:
                        pass
                    parsed = _parse_context_overflow(body)
                    if parsed is not None:
                        if ctx_overflow_attempts == 0:
                            max_ctx, _req_out, input_tok = parsed
                            new_cap = max_ctx - input_tok - SAFETY
                        else:
                            new_cap = cap // 2
                        if new_cap >= MIN_OUT and new_cap < cap and \
                                ctx_overflow_attempts < MAX_CTX_OVERFLOW_RETRIES:
                            print(f"  context-overflow #{ctx_overflow_attempts+1}: "
                                  f"capping max_tokens {cap} → {new_cap}", flush=True)
                            cap = new_cap
                            body_bytes = _build_payload(cap)
                            ctx_overflow_attempts += 1
                            last_exc = e
                            # Context-overflow retries don't consume the main budget.
                            continue
                    # Not a context overflow: the body is the ONLY description of what
                    # the server rejected, and re-raising HTTPError drops it (urllib
                    # only carries the status line). Surface it before giving up —
                    # otherwise a whole run dies as a bare "HTTP Error 400: Bad Request".
                    print(f"  HTTP 400 from {url} (model={self.model}): {body[:2000]}",
                          flush=True)
                    raise
                if e.code == 429 and attempt < max_retries - 1:
                    wait = 2 ** (attempt + 1)
                    print(f"  rate-limited, retrying in {wait}s...", flush=True)
                    time.sleep(wait)
                    last_exc = e
                    attempt += 1
                    continue
                raise
            except (urllib.error.URLError, TimeoutError) as e:
                if attempt < max_retries - 1:
                    time.sleep(min(2 ** attempt, 15))
                    last_exc = e
                    attempt += 1
                    continue
                raise

        raise Exception(f"chat() failed after {max_retries} attempts: {last_exc}")

    @staticmethod
    def _read_stream_as_result(resp: Any) -> dict[str, Any]:
        """Aggregate an SSE ``/chat/completions`` stream into a non-streaming
        result dict (``{choices:[{message:{...}}], usage:{...}}``).

        Required for text-tools mode: some endpoints'
        non-streaming aggregator drops this reasoning model's output
        (message.content=""), but the streamed deltas carry the real text.
        We concatenate ``delta.content``, ``delta.reasoning_content`` and any
        native ``delta.tool_calls``.

        Reasoning is kept in a separate ``reasoning_content`` field on the
        assembled message, mirroring what a NON-streaming vLLM response with
        ``--reasoning-parser`` returns — so both paths hand the caller the same
        shape and ``inline_reasoning_content`` can fold it back into the prompt on
        the next turn. Dropping it here (the previous behavior) silently discarded
        ~2/3 of every generated turn: measured over 8443 logged gap_finder calls,
        66-68% of completion tokens were reasoning and NONE of it reached the
        next turn's context, while GRPO training kept all of it.
        """
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        reasoning_key = ""
        tool_calls: dict[int, dict[str, Any]] = {}
        finish_reason = "stop"
        usage: dict[str, Any] = {}
        model = ""
        resp_id = ""
        for raw_line in resp:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line or not line.startswith("data:"):
                continue
            payload = line[len("data:"):].strip()
            if payload == "[DONE]":
                break
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                continue
            model = obj.get("model") or model
            resp_id = obj.get("id") or resp_id
            if obj.get("usage"):
                usage = obj["usage"]
            for ch in obj.get("choices", []):
                delta = ch.get("delta") or {}
                if delta.get("content"):
                    content_parts.append(delta["content"])
                # vLLM emits ``reasoning_content``; some builds/parsers use the
                # shorter ``reasoning``. Take whichever key this stream started
                # with and ignore the other, so a server that echoes both cannot
                # double-count the same reasoning.
                for _key in ("reasoning_content", "reasoning"):
                    if reasoning_key and _key != reasoning_key:
                        continue
                    chunk = delta.get(_key)
                    if isinstance(chunk, str) and chunk:
                        reasoning_key = _key
                        reasoning_parts.append(chunk)
                        break
                for tc in delta.get("tool_calls") or []:
                    idx = tc.get("index", 0)
                    slot = tool_calls.setdefault(idx, {
                        "id": tc.get("id", f"call_{idx}"),
                        "type": "function",
                        "function": {"name": "", "arguments": ""},
                    })
                    fn = tc.get("function") or {}
                    if fn.get("name"):
                        slot["function"]["name"] = fn["name"]
                    if fn.get("arguments"):
                        slot["function"]["arguments"] += fn["arguments"]
                if ch.get("finish_reason"):
                    finish_reason = ch["finish_reason"]
        message: dict[str, Any] = {
            "role": "assistant",
            "content": "".join(content_parts),
        }
        if reasoning_parts:
            message["reasoning_content"] = "".join(reasoning_parts)
        if tool_calls:
            message["tool_calls"] = [tool_calls[i] for i in sorted(tool_calls)]
        return {
            "id": resp_id,
            "object": "chat.completion",
            "model": model,
            "choices": [{"index": 0, "message": message,
                         "finish_reason": finish_reason}],
            "usage": usage,
        }

    @staticmethod
    def _normalize_text_tools_messages(
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
    ) -> list[dict[str, Any]]:
        """Rewrite messages for the text-tools protocol without mutating input.

        - Merge the rendered tool block into the first system message (or prepend
          one). The endpoint has no ``tools`` param, so tool specs must live here.
        - Any assistant message that carried synthetic ``tool_calls`` is sent as
          plain assistant content (the server can't parse the tool_calls key);
          if it had no text, re-serialize the calls as ``<tool_call>`` lines so
          the model still sees what it committed to.
        - Consecutive ``role:"tool"`` results from ONE assistant turn are merged
          into a SINGLE ``role:"user"`` message wrapping every result in one
          ``<tool_response>`` block, each entry prefixed ``[<tool_name>] <result>``
          and joined by a blank line. This is byte-identical to the GRPO training
          env's observation (``environments._wrap_observation`` +
          ``<tool_response>``), so a trained policy sees the SAME observation
          format in production as it did in training. (The compat endpoint also
          rejects orphan tool-role messages without native tool_calls, so they
          must become user turns regardless.)
        """
        tool_block = _render_tools_as_text(tools or [])
        out: list[dict[str, Any]] = []
        merged_system = False
        pending_tool: list[str] = []

        def _flush_tool_results() -> None:
            if not pending_tool:
                return
            body = "\n\n".join(pending_tool)
            out.append({
                "role": "user",
                "content": f"<tool_response>\n{body}\n</tool_response>",
            })
            pending_tool.clear()

        for m in messages:
            role = m.get("role")
            if role == "tool":
                name = m.get("name", "")
                content = m.get("content")
                if not isinstance(content, str):
                    content = json.dumps(content) if content is not None else ""
                pending_tool.append(f"[{name}] {content}")
                continue
            # Any non-tool message ends the current run of tool results.
            _flush_tool_results()
            if role == "system" and not merged_system:
                content = m.get("content") or ""
                out.append({"role": "system",
                            "content": f"{content}\n\n{tool_block}" if content else tool_block})
                merged_system = True
            elif role == "assistant" and m.get("tool_calls"):
                text = m.get("content") or ""
                # Test the post-</think> remainder, not the raw string: once
                # reasoning has been folded into content, a turn that committed
                # only a tool call still has non-empty text, and testing that
                # would skip re-serialization and lose the call from history.
                if not _strip_leading_think(text).strip():
                    parts = []
                    for tc in m["tool_calls"]:
                        fn = tc.get("function") or {}
                        args = fn.get("arguments", "{}")
                        try:
                            args_obj = json.loads(args) if isinstance(args, str) else args
                        except (json.JSONDecodeError, TypeError):
                            args_obj = {}
                        parts.append(
                            '<tool_call>' +
                            json.dumps({"name": fn.get("name", ""), "arguments": args_obj}) +
                            '</tool_call>'
                        )
                    calls_text = "\n".join(parts)
                    # Keep any think block that preceded the (absent) answer text:
                    # append the calls after it instead of replacing it.
                    text = f"{text.rstrip()}\n\n{calls_text}" if text.strip() else calls_text
                out.append({"role": "assistant", "content": text})
            else:
                out.append(m)
        _flush_tool_results()
        if not merged_system:
            out.insert(0, {"role": "system", "content": tool_block})
        return out

    # ------------------------------------------------------------------
    # Anthropic implementation
    # ------------------------------------------------------------------

    def _log_chat_call(
        self,
        call_id: int,
        messages: list[dict],
        response_msg: dict,
        usage: dict,
        latency: float,
        extra: dict | None = None,
    ) -> None:
        from datetime import datetime, timezone

        self._ensure_log()

        tool_calls = response_msg.get("tool_calls", [])
        content = response_msg.get("content", "")

        entry = {
            "call_id": call_id,
            "step": self._step_name,
            "model": self.model,
            "temperature": self.temperature,
            "n_messages": len(messages),
            # FULL input context fed to the model on this turn (system + user +
            # prior assistant turns + tool-result messages). Per project
            # convention: store everything, never truncate at write time.
            # Yes this is redundant across turns; that is intentional —
            # any single line of the JSONL must be self-contained.
            "input_messages": messages,
            "response_content": content or "",
            # Reasoning split off by the server's ``--reasoning-parser``, kept
            # out of ``response_content`` but logged so a run stays replayable.
            "response_reasoning": response_msg.get("reasoning_content") or "",
            "tool_calls": [
                {"id": tc.get("id", ""),
                 "name": tc["function"]["name"],
                 "args": tc["function"]["arguments"]}
                for tc in tool_calls
            ] if tool_calls else [],
            "finish_reason": "tool_calls" if tool_calls else "stop",
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
            "latency_s": round(latency, 2),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if extra:
            entry.update(extra)
            # Also track aggregate retry pressure on the client object so
            # client.summary() can roll it up into the run-level summary.json.
            if "retries_on_parser_error" in extra:
                self._total_retries_on_parser_error = (
                    getattr(self, "_total_retries_on_parser_error", 0)
                    + extra["retries_on_parser_error"]
                )

        if self._shared_log is not None:
            self._shared_log.write(entry)
        else:
            self._log_writer.write(entry)

        sep = "-" * 72
        if tool_calls:
            tool_names = ", ".join(tc["function"]["name"] for tc in tool_calls)
            print(f"{sep}", flush=True)
            print(
                f"Agent turn #{call_id} | tools: {tool_names} | "
                f"{usage.get('prompt_tokens', 0)} in / "
                f"{usage.get('completion_tokens', 0)} out | "
                f"{latency:.1f}s",
                flush=True,
            )
            print(f"{sep}", flush=True)
        elif content:
            preview = content[:200].replace("\n", " ")
            print(f"{sep}", flush=True)
            print(
                f"Agent turn #{call_id} | response | "
                f"{usage.get('prompt_tokens', 0)} in / "
                f"{usage.get('completion_tokens', 0)} out | "
                f"{latency:.1f}s",
                flush=True,
            )
            print(f"  {preview}{'...' if len(content) > 200 else ''}", flush=True)
            print(f"{sep}", flush=True)
