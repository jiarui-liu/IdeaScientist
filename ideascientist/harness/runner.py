"""The agent loop.

The orchestrator maintains ``plan.md`` and delegates to sub-agents that each run
in their own client with their own context. Sub-agent spawns issued in one
orchestrator turn run concurrently; so do the ``read_paper`` calls inside a
sub-agent turn, which is what makes a run finish in reasonable wall-clock given
that reading dominates it.

Turn caps, token budgets, and the phase-completion contract all come from
:mod:`ideascientist.harness.tools`, shared with the GRPO training environment so
a trained role meets identical conditions at deployment.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import os
import threading
from pathlib import Path
from typing import Optional

from ideascientist.harness import tools as T
from ideascientist.harness.client import AgenticLLMClient
from ideascientist.harness.orchestrator import orchestrator_prompt
from ideascientist.harness.runlog import RunLogger, SharedLogRegistry
from ideascientist.utils.llm import DEFAULT_BASE_URL, context_window_for


def _api_key() -> str:
    return (
        os.environ.get("IDEASCIENTIST_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or "EMPTY"
    )


# --------------------------------------------------------------- constants
# Per-role subagent turn caps. The trainable roles take the shared phase-turn
# budget from tools.py, so train and deploy see the same horizon.
_SUB_TURN_CAPS = {r: T.PHASE_TURN_BUDGET for r in T.ROLE_REQUIRED_ARTIFACT}
_DEFAULT_SUB_TURNS = 10

# Input budget for one paper read, as reported in the paper. The reader is a
# single-turn tool, so this bounds the whole context it ever sees.
READER_INPUT_TOKEN_CAP = 30_000

# The phase-completion contract (which roles must write which artifact, min chars, nudge
# count, nudge text) is the single source of truth in tools.py — imported below as needed
# via T.ROLE_REQUIRED_ARTIFACT / T.ARTIFACT_MIN_CHARS / T.MAX_PHASE_NUDGES /
# T.artifact_nudge, shared verbatim with the GRPO training env.

_ROLE_ENV_MAP_CACHE: dict[str, dict[str, int]] = {}


def _role_env_int_map(var_name: str) -> dict[str, int]:
    if var_name not in _ROLE_ENV_MAP_CACHE:
        raw = os.environ.get(var_name, "").strip()
        parsed: dict[str, int] = {}
        if raw:
            try:
                parsed = {str(k): int(v) for k, v in json.loads(raw).items()}
            except (ValueError, TypeError, AttributeError):
                parsed = {}
        _ROLE_ENV_MAP_CACHE[var_name] = parsed
    return _ROLE_ENV_MAP_CACHE[var_name]


# Per-role endpoint and model routing, as JSON {role: str} maps. This is what
# serves the three trained roles from their own checkpoints while the
# orchestrator, paper reader, and reviewer stay on the base model. A role absent
# from the map keeps the process-global routing, so leaving both unset is the
# single-endpoint case.
#   IDEASCIENTIST_ROLE_BASE_URL  e.g. {"gap_finder": "http://host:8000/v1", ...}
#   IDEASCIENTIST_ROLE_MODEL     e.g. {"gap_finder": "<merged adapter path>", ...}
# Only _run_subagent consults these; the reader's single-turn path stays on base.
_ROLE_ENV_STR_CACHE: dict[str, dict[str, str]] = {}


def _role_env_str_map(var_name: str) -> dict[str, str]:
    if var_name not in _ROLE_ENV_STR_CACHE:
        raw = os.environ.get(var_name, "").strip()
        parsed: dict[str, str] = {}
        if raw:
            try:
                parsed = {str(k): str(v) for k, v in json.loads(raw).items()}
            except (ValueError, TypeError, AttributeError):
                parsed = {}
        _ROLE_ENV_STR_CACHE[var_name] = parsed
    return _ROLE_ENV_STR_CACHE[var_name]


def _route_for_model(
    model: str,
    base_url_override: str | None = None,
    text_tools_override: bool | None = None,
) -> tuple[str, bool]:
    """Resolve ``(base_url, text_tools)`` for one call.

    The per-call overrides exist so concurrent sub-agents can target different
    servers in one process: the env-var route is single-valued and would
    serialize the parallel ``read_paper`` calls that dominate a run.
    """
    base = (
        base_url_override
        or os.environ.get("AGENT_LOOP_BASE_URL")
        or os.environ.get("IDEASCIENTIST_BASE_URL")
        or DEFAULT_BASE_URL
    )
    if text_tools_override is not None:
        return base, text_tools_override
    flag = os.environ.get("IDEASCIENTIST_TEXT_TOOLS", "")
    return base, flag.strip().lower() in ("1", "true", "yes", "on")


def _spawn_spec(roles: list[str]) -> dict:
    return {"type": "function", "function": {
        "name": "spawn_subagent",
        "description": "Delegate a heavy task to a fresh subagent with its OWN context; it returns "
                       "only a short status (full paper text stays in the subagent). roles: "
                       + ", ".join(roles),
        "parameters": {"type": "object", "properties": {
            "role": {"type": "string", "enum": roles}, "task": {"type": "string"}},
            "required": ["role", "task"]}}}


def _log_heartbeat(log, prompt_tokens, window):
    """Emit the context-budget heartbeat as a run-log event ONLY — never into the
    model-visible context.

    Train/deploy parity: the GRPO training env (training/environment.py) never puts a
    ``[budget]`` line into the model's system prompt, so production must not either.
    We keep the signal for observability by logging it. Caller-relative:
    ``prompt_tokens`` is THIS agent's own current context size."""
    hb = T.budget_heartbeat(prompt_tokens, window)
    if log is not None:
        log.event("budget_heartbeat", prompt_tokens=prompt_tokens,
                  window=window, heartbeat=hb)


def _needs_artifact_nudge(ctx, required_artifact) -> bool:
    """True iff ``required_artifact`` is set but not yet written on disk with at least
    ``T.ARTIFACT_MIN_CHARS`` non-whitespace chars. Reads from the same docs dir that
    ``edit_doc`` writes to (``T._docs_root(ctx)``; see tools.py:_exec_edit_doc)."""
    if not required_artifact:
        return False
    try:
        p = T._docs_root(ctx) / required_artifact
        if not p.exists():
            return True
        return len(p.read_text(encoding="utf-8").strip()) < T.ARTIFACT_MIN_CHARS
    except Exception:  # noqa: BLE001 — never let the gate crash the loop
        return False


TOOL_ARG_LOG_CHARS = 400


def _parse_tool_args(tc):
    """Parse a tool call's JSON arguments.

    Returns ``(args, None)`` or ``(None, (message, raw))``. The message goes back
    to the model as the tool result; the raw text goes to the run log, without
    which a parse failure leaves no evidence of what the model actually emitted.
    """
    raw = tc["function"].get("arguments") or "{}"
    try:
        parsed = json.loads(raw)
    except Exception:
        parsed = None
    else:
        # A model that double-encodes its arguments yields a str here, which
        # every caller would then treat as a mapping.
        if isinstance(parsed, dict):
            return parsed, None
    return None, (
        "ERROR: tool arguments were not valid JSON — the content was likely too long and "
        "got truncated. Send SMALLER content; build long documents one section per call "
        "using edit_doc (command='insert' at the end).",
        raw[:TOOL_ARG_LOG_CHARS],
    )


def _tool_loop(client, messages, tools, ctx, log, step_prefix, max_turns,
               fulltext_cap=None, read_paper_cap=None, solo_first_turn=False,
               required_artifact=None):
    """Generic tool-use loop (no spawn_subagent). Returns (final_text, n_calls).

    ``fulltext_cap`` refuses full-text reads past the limit instead of executing
    them, which in practice holds a reader to its one assigned paper.

    ``solo_first_turn`` protects the reader's early exit: turn 0 must be the lone
    view of the existing digest, so "all goals already answered" can fire before
    any full text is pulled. From turn 1 the reader may batch freely.

    ``required_artifact`` mirrors the training env's phase-completion gate — a
    no-tool turn does not end the loop until the artifact exists on disk, up to
    ``T.MAX_PHASE_NUDGES`` nudges. None finishes on the first no-tool turn.
    """
    n = 0
    final = ""
    ft_used = 0
    rp_used = 0
    phase_nudges = 0
    window = context_window_for(client.model)
    for t in range(max_turns):
        client.set_step(f"{step_prefix}:{t}")
        # Context-budget heartbeat: LOG only, never into the model-visible context
        # (train/deploy parity — the GRPO env has no [budget] line). Caller-relative.
        _log_heartbeat(log, client.last_prompt_tokens, window)
        # Neither loop compacts its history: the GRPO environment does not, so
        # condensing here would make served transcripts diverge from trained ones.
        msg = client.chat(messages, tools=tools)["choices"][0]["message"]
        messages.append(msg)
        tcs = msg.get("tool_calls") or []
        if not tcs:
            # Phase-completion gate (train/deploy parity): the model wants to finish.
            # If a required artifact is not yet written, nudge (up to T.MAX_PHASE_NUDGES)
            # instead of terminating — matches agent_environment._process_episode_turn.
            if _needs_artifact_nudge(ctx, required_artifact) and phase_nudges < T.MAX_PHASE_NUDGES:
                phase_nudges += 1
                messages.append({"role": "user", "content": T.artifact_nudge(required_artifact)})
                log.event("phase_nudge", step=step_prefix, artifact=required_artifact,
                          nudge=phase_nudges)
                continue
            final = msg.get("content") or ""
            break
        # Union this turn's read_paper goals per paper, so the first read already
        # targets all of them and the rest hit the reader's early exit with the
        # same digest. The training env unions before its prefetch for the same
        # reason; the digest is append-only, so it still accumulates across turns.
        goal_union = T.union_read_paper_goals(tcs)
        for i, tc in enumerate(tcs):
            n += 1
            fn = tc["function"]["name"]
            fargs, perr = _parse_tool_args(tc)
            if solo_first_turn and t == 0 and i > 0:
                # paper_reader batched several calls into turn 0; only the first ran.
                # Refuse the rest (each still needs a tool result for the API) so the
                # cheap step-2 early-exit (view digest -> maybe stop) can fire before any
                # full text is pulled. Batching is allowed from turn 1 on.
                res = ("(refused: on your FIRST turn call ONLY edit_doc(command='view') on the "
                       "digest and wait for its result — the early-exit must be able to fire before "
                       "any full text is pulled. Re-issue this call next turn; from then on you MAY "
                       "batch independent calls in one turn.)")
                log.event("parallel_call_refused", step=step_prefix, tool=fn, position=i)
            elif perr:
                res = perr[0]
                log.event("tool_error", tool=fn, error="bad_json_args", raw_args=perr[1])
            elif (fn in T.FULLTEXT_TOOLS and fulltext_cap is not None
                  and ft_used >= fulltext_cap):
                res = (f"(refused: this role may read at most {fulltext_cap} full text(s) per "
                       f"task and has already used {ft_used}. Answer the goals from the one paper "
                       f"you already read; if it genuinely cannot, say so in the digest.)")
                log.event("fulltext_cap_refused", step=step_prefix, used=ft_used, cap=fulltext_cap)
            elif (fn == T.READ_PAPER_TOOL and read_paper_cap is not None
                  and rp_used >= read_paper_cap):
                res = (f"(refused: this role may read at most {read_paper_cap} paper(s) per task "
                       f"and has already used {rp_used}. Work from the digests you already pulled "
                       f"(edit_doc command='view' the papers/<id>.md you created) + get_paper_biblio + "
                       f"keyword_search; prioritise the most load-bearing reads.)")
                log.event("read_paper_cap_refused", step=step_prefix, used=rp_used, cap=read_paper_cap)
            else:
                try:
                    if fn == T.READ_PAPER_TOOL and fargs is not None:
                        pid = T.coerce_paper_id(fargs)
                        if pid is not None and pid in goal_union:
                            fargs = {**fargs, "goals": goal_union[pid]}
                    res = T.run_tool(fn, fargs, ctx, tool_call_id=tc["id"])
                    if fn in T.FULLTEXT_TOOLS:
                        ft_used += 1
                    if fn == T.READ_PAPER_TOOL:
                        rp_used += 1
                except Exception as exc:  # noqa: BLE001
                    res = f"ERROR in {fn}: {exc}"
                    log.event("tool_error", tool=fn, error=str(exc), tool_call_id=tc["id"])
            # ``name`` is REQUIRED for text-tools parity: _normalize_text_tools_messages
            # renders each result as ``[<name>] <result>`` (harness/client.py). Without it the
            # observation becomes ``[] <result>`` — off-distribution vs the GRPO env,
            # which always tags results ``[<fn_name>]`` (training/environments.py).
            messages.append({"role": "tool", "tool_call_id": tc["id"], "name": fn, "content": res})
    return final, n


_READER_TOK = None


def _reader_tokenizer():
    global _READER_TOK
    if _READER_TOK is None:
        from transformers import AutoTokenizer
        path = os.environ.get(
            "IDEASCIENTIST_READER_TOKENIZER",
            os.environ.get("IDEASCIENTIST_MODEL", "Qwen3.6-27B"),
        )
        _READER_TOK = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    return _READER_TOK


def _count_reader_tokens(text: str) -> int:
    if not text:
        return 0
    try:
        return len(_reader_tokenizer()(text, add_special_tokens=False)["input_ids"])
    except Exception:
        return len(text) // 3   # conservative (min ~3 chars/token)


def _cap_fulltext_head_tail(text: str, cap_tokens: int) -> str:
    """Truncate `text` to <= cap_tokens by keeping the HEAD (~80%) + TAIL (~20%) and
    dropping the MIDDLE. For gap-finding the tail (Conclusion/Limitations/Future Work)
    is as valuable as the head (Intro/Method), so we never drop it. Falls back to a
    conservative char cap if the tokenizer can't load (never crash a rollout)."""
    if not text:
        return text
    try:
        tok = _reader_tokenizer()
        ids = tok(text, add_special_tokens=False)["input_ids"]
        if len(ids) <= cap_tokens:
            return text
        head = int(cap_tokens * 0.8)
        tail = cap_tokens - head
        head_txt = tok.decode(ids[:head])
        tail_txt = tok.decode(ids[-tail:])
        return (head_txt +
                "\n\n[... MIDDLE OF PAPER TRUNCATED TO FIT READER CONTEXT — "
                "intro/method (above) and conclusion/limitations (below) preserved ...]\n\n" +
                tail_txt)
    except Exception:
        # conservative char fallback: ~3 chars/token floor guarantees <= cap tokens
        budget = cap_tokens * 3
        if len(text) <= budget:
            return text
        h = int(budget * 0.8)
        return text[:h] + "\n\n[... middle truncated ...]\n\n" + text[-(budget - h):]


def build_reader_task(paper_id, goals: list, prior_digest: str = "") -> str:
    """The reader's task, with the paper's full text and biblio prefetched to the
    front and the current digest injected.

    Front-loading the full text makes the ``[system + full text]`` prefix
    identical across the generations of one prompt, so the served pool's prefix
    cache reuses that prefill instead of recomputing it per generation. Having
    the text in context also removes the fetch turns entirely.

    Shared by the deployment ``read_paper`` handler and the training
    environment's reader, so both see an identical task.
    """
    from ideascientist.harness import corpus as C
    from ideascientist.harness.tools import FULLTEXT_CHAR_LIMIT

    pid = int(paper_id)
    goal_lines = "\n".join(f"- {g}" for g in goals) or \
        "- summarize the paper's method, key results, and stated limitations"

    # Pre-fetch body + biblio (cheap local DB reads; the LLM cost is the prefill,
    # which prefix-caching now shares across generations).
    try:
        raw_fulltext = C.get_stored_full_text(paper_id=pid, key="") or ""
        # Head-and-tail truncation: the intro, method, conclusion and limitations
        # are where a gap claim is grounded, and plain end-truncation would cut the
        # limitations first. 0 disables and falls back to the character cap.
        if _tok_cap > 0:
            # The full text gets its own budget, shrunk below it only when the
            # whole input (text + accumulated digest + goals) would exceed the
            # model's input ceiling. Budgeting this way means a small digest
            # never starves the text, and a large one trims it just enough to
            # fit rather than dropping the paper.
            _ft_budget = _tok_cap
            _limit = int(os.environ.get("IDEASCIENTIST_READER_INPUT_LIMIT", "0") or "0")
            if _limit > 0:
                _reserve = _count_reader_tokens(prior_digest or "") + _count_reader_tokens(goal_lines) + 800
                _ft_budget = min(_ft_budget, _limit - _reserve)
            _ft_budget = max(4000, _ft_budget)
            fulltext = _cap_fulltext_head_tail(raw_fulltext, _ft_budget)
        else:
            fulltext = raw_fulltext[:FULLTEXT_CHAR_LIMIT]
    except Exception:
        fulltext = ""
    try:
        bib = C.get_paper_biblio(pid) or {}
    except Exception:
        bib = {}
    title = (bib.get("title") or "").strip() if isinstance(bib, dict) else ""
    abstract = (bib.get("abstract") or "").strip() if isinstance(bib, dict) else ""
    arxiv_id = (bib.get("arxiv_id") or "").strip() if isinstance(bib, dict) else ""

    prior = (prior_digest or "").strip()
    if prior:
        digest_block = (
            f"CURRENT DIGEST — papers/{pid}.md (append NEW goals; do NOT re-answer "
            f"goals already covered here):\n{prior}"
        )
    else:
        digest_block = (
            f"CURRENT DIGEST — papers/{pid}.md does NOT exist yet (FIRST read). "
            f"Create it."
        )

    # Full text FIRST (the shared, prefix-cacheable block), then goals/digest.
    if fulltext:
        head = (
            f"PAPER {pid} — TITLE: {title}\n\n"
            f"FULL TEXT (read this to answer the goals; it is the ONLY source — do "
            f"NOT call get_full_text or get_paper_biblio):\n{fulltext}\n"
        )
        fetch_note = (
            "The full text and title are ABOVE — do NOT call get_full_text or "
            "get_paper_biblio."
        )
    else:
        # Not stored locally: fall back to the agentic path (reader fetches).
        head = (
            f"PAPER {pid} — TITLE: {title}\nABSTRACT: {abstract}\n"
            f"(Full text is NOT pre-loaded"
            + (f"; if needed call fetch_full_text('{arxiv_id}')." if arxiv_id else ".")
            + ")\n"
        )
        fetch_note = (
            "Full text is not pre-loaded above"
            + (f"; call fetch_full_text('{arxiv_id}') to read the body."
               if arxiv_id else " and no arxiv_id is available — answer from the abstract.")
        )

    return (
        f"{head}\n"
        f"====================\n"
        f"Read paper id {pid}. Reading goals:\n{goal_lines}\n\n{digest_block}\n\n"
        f"{fetch_note} If the digest already answers EVERY goal, reply "
        f"'done — all goals already in digest' with NO tool calls; otherwise answer "
        f"the not-yet-answered goals from the text above and append them to "
        f"papers/{pid}.md via edit_doc (create the file first if it does not exist)."
    )


def read_paper_single_turn(paper_id, goals: list, prior_digest: str = "", *,
                           model, base_url=None, api_key=None, run_dir=None) -> str:
    """Read one paper in a single completion, as reported in the paper.

    The model gets the full text and the goals and answers them directly; the
    harness writes the digest. Returns the content to append, or "" when every
    goal is already covered. An agentic reader has nothing to decide here — it
    would spend one turn on edit_doc and another signalling done, doubling the
    calls and adding tool-parsing failures for no benefit.
    """
    from ideascientist.harness.client import AgenticLLMClient
    from ideascientist.harness.thinking import after_think

    pid = int(paper_id)
    task = build_reader_task(pid, goals, prior_digest)  # fulltext + digest + goals block
    # The "output NONE" escape hatch is ONLY valid when a prior digest exists (the
    # append/re-read case: the goals may already be covered). On a FIRST read the digest
    # is empty, so NONE is logically impossible — and offering it anyway gives the
    # no-think reader a deterministic way out of a hard goal, which it takes. Gate
    # it on prior_digest so it appears only when it can truthfully apply.
    if (prior_digest or "").strip():
        system = (
            "You read ONE paper's full text (provided in the message) and answer the given "
            "research goals for a gap-finding digest. Output ONLY the markdown digest content "
            "that answers the not-yet-covered goals — findings, method, and stated limitations "
            "relevant to each goal, using [paper_id] citations exactly as they appear in the "
            "text. Do NOT use any tools. Do NOT restate goals you were told are already covered. "
            "If the CURRENT DIGEST already answers EVERY goal, output exactly: NONE"
        )
    else:
        system = (
            "You read ONE paper's full text (provided in the message) and answer the given "
            "research goals for a gap-finding digest. Output ONLY the markdown digest content "
            "— findings, method, and stated limitations relevant to each goal, using "
            "[paper_id] citations exactly as they appear in the text. Do NOT use any tools. "
            "If the paper does not directly address a goal, still summarize what the paper's "
            "method and findings ARE so the citation is grounded. Always produce digest content."
        )
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": task}]
    eb = {"chat_template_kwargs": {"enable_thinking": False}}
    if os.environ.get("IDEASCIENTIST_READER_DECODE", "").strip().lower() in ("1", "true", "yes"):
        eb.update({"temperature": 0.7, "top_p": 0.80, "top_k": 20, "min_p": 0.0,
                   "presence_penalty": 1.5, "repetition_penalty": 1.0})
    rmt = int(os.environ.get("IDEASCIENTIST_READER_MAX_TOKENS", "0") or "0") or 12288
    burl, _ = _route_for_model(model, base_url_override=base_url, text_tools_override=False)
    # Key the reader's log directory by paper_id so the directory count stays
    # bounded by the corpus rather than growing per call, and concurrent readers
    # of different papers never collide.
    _reader_log_root = str(run_dir) if run_dir else (os.environ.get("RUN_DIR") or os.environ.get("TMPDIR") or ".")
    _reader_log_dir = os.path.join(_reader_log_root, "logs", "_sublogs",
                                   "reader_single_turn", str(paper_id))
    client = AgenticLLMClient(
        api_key=api_key or _api_key(), model=model, base_url=burl,
        text_tools=False, max_tokens=rmt, extra_body=eb,
        log_dir=_reader_log_dir,
    )
    resp = client.chat(messages, tools=[])
    # chat() returns the raw OpenAI-style response dict -> choices[0].message.content
    content = ""
    if isinstance(resp, dict):
        choices = resp.get("choices") or []
        if choices:
            msg = choices[0].get("message") or {}
            content = msg.get("content") or ""
        elif isinstance(resp.get("message"), dict):        # defensive alt shape
            content = resp["message"].get("content") or ""
    else:
        content = str(resp or "")
    content = after_think(content).strip()
    if not content or content.upper().startswith("NONE"):
        return ""
    return content


def _run_subagent(role, task, ctx, model, log, log_key, conversation_id,
                  max_sub_turns: int | None = None,
                  *, base_url: str | None = None,
                  text_tools: bool | None = None, api_key: str | None = None,
                  solo_first_turn: bool | None = None):
    """Run one subagent to completion in its own client, so many can run
    concurrently without shared-state races.

    Records append to a shared per-role or per-paper log via ``ctx.log_registry``,
    tagged with ``conversation_id`` so one role's sessions stay in one file.

    ``base_url`` / ``text_tools`` / ``api_key`` are per-call routing overrides
    that bypass the process-global lookups, which is what lets N subagents target
    N different servers at once.
    """
    if role not in T.ROLES:
        return {"role": role, "digest": f"(unknown role: {role})", "tokens": 0, "calls": 0}
    # Subagents own the files they produce, so they run with the FULL write allowlist
    # regardless of the caller's scope. The orchestrator's ctx carries
    # write_scope={"plan.md"}; clear it on a shallow copy so this subagent (and any
    # papers/<id>.md it writes via read_paper) is not restricted to plan.md.
    if ctx.write_scope is not None:
        ctx = dataclasses.replace(ctx, write_scope=None)
    if max_sub_turns is None:
        max_sub_turns = _SUB_TURN_CAPS.get(role, _DEFAULT_SUB_TURNS)
    # Orchestrator spawns pass one uniform cap, so IDEASCIENTIST_ROLE_TURN_CAP is
    # how a role gets back the per-role horizon it was trained under.
    _turn_override = _role_env_int_map("IDEASCIENTIST_ROLE_TURN_CAP").get(role)
    if _turn_override is not None and _turn_override > 0:
        max_sub_turns = _turn_override
    # Per-role endpoint and model routing: unless the caller pinned base_url/model,
    # IDEASCIENTIST_ROLE_BASE_URL / IDEASCIENTIST_ROLE_MODEL send this role to its
    # own pool and checkpoint. A role absent from the map falls through to the
    # process-global routing; an explicit per-call base_url/model still wins.
    if base_url is None:
        _url_override = _role_env_str_map("IDEASCIENTIST_ROLE_BASE_URL").get(role)
        if _url_override:
            base_url = _url_override
    _model_override = _role_env_str_map("IDEASCIENTIST_ROLE_MODEL").get(role)
    if _model_override:
        model = _model_override
    base_url, text_tools = _route_for_model(
        model, base_url_override=base_url, text_tools_override=text_tools
    )
    shared_log = None
    client_kwargs = {}
    if ctx.log_registry is not None:
        log_path = ctx.run_dir / "logs" / "_sublogs" / f"{log_key}.json"
        shared_log = ctx.log_registry.acquire(log_path, conversation_id)
    else:
        client_kwargs["log_dir"] = ctx.run_dir / "logs" / "_sublogs" / conversation_id.replace("#", "_")
    # The paper reader is an untrained extraction tool, so reasoning tokens it
    # emits are generated only to be stripped. Applied here rather than at a
    # call site so training rollouts and deployment stay in parity; the trained
    # policies are not driven through this client and keep their reasoning.
    extra_body = None
    reader_max_tokens = None
    if role == "paper_reader":
        _no_think = os.environ.get("IDEASCIENTIST_READER_NO_THINK", "").strip().lower() in ("1", "true", "yes")
        eb = {}
        if _no_think:
            eb["chat_template_kwargs"] = {"enable_thinking": False}
        # The model's recommended non-thinking decode parameters for the reader.
        if os.environ.get("IDEASCIENTIST_READER_DECODE", "").strip().lower() in ("1", "true", "yes"):
            eb.update({"temperature": 0.7, "top_p": 0.80, "top_k": 20, "min_p": 0.0,
                       "presence_penalty": 1.5, "repetition_penalty": 1.0})
        extra_body = eb or None
        # Reader output cap: IDEASCIENTIST_READER_MAX_TOKENS (default keeps the generic
        # 32000). Reader digests are ~1k tokens; a lower cap (e.g. 12288) bounds the rare
        # runaway-loop reader without touching normal digests.
        _rmt = int(os.environ.get("IDEASCIENTIST_READER_MAX_TOKENS", "0") or "0")
        if _rmt > 0:
            reader_max_tokens = _rmt
    elif role in T.ROLE_REQUIRED_ARTIFACT:
        # Force thinking on for the trained roles to match the training env, which
        # templates enable_thinking and re-primes <think> every turn. top_p/top_k
        # mirror the GRPO rollout sampling so the served policy decodes under the
        # distribution it was trained on.
        extra_body = {"chat_template_kwargs": {"enable_thinking": True},
                      "top_p": 0.95, "top_k": 20}
    elif role == "judge":
        # In-harness adversarial reviewer: score candidates with thinking OFF (served Qwen
        # chat template). Client-side per-request control — takes effect only on the
        # openai/Qwen route (anthropic / openai_responses ignore extra_body), so non-Qwen
        # judge runs are unaffected. Mirrors the paper_reader / eval-judge thinking-off policy.
        extra_body = {"chat_template_kwargs": {"enable_thinking": False}}
    # Output cap: the reader takes IDEASCIENTIST_READER_MAX_TOKENS, a trained role
    # takes its IDEASCIENTIST_ROLE_MAX_TOKENS entry, otherwise 32000.
    _eff_max_tokens = reader_max_tokens or _role_env_int_map("IDEASCIENTIST_ROLE_MAX_TOKENS").get(role) or 32000
    client = AgenticLLMClient(
        api_key=api_key or _api_key(), model=model, base_url=base_url,
        text_tools=text_tools,
        max_tokens=_eff_max_tokens,  # long sections fit in one call; client auto-caps to context
        shared_log=shared_log, extra_body=extra_body, **client_kwargs,
    )
    user_content = task
    if role in T.PROBLEM_CONTEXT_ROLES:
        user_content = T.build_subagent_user_message(task, ctx.query_text)
    messages = [{"role": "system", "content": T.role_prompt(role)},
                {"role": "user", "content": user_content}]
    tools = T.openai_specs(T.role_tools(role))  # scoped + external-gated + reader-only
    # L1 roles that can pull papers get the read_paper soft cap; the paper_reader
    # (run via read_paper) instead gets the full-text cap. A role never has both.
    read_paper_cap = T.READ_PAPER_CAP if T.READ_PAPER_TOOL in T.role_tools(role) else None
    _solo = (role == "paper_reader") if solo_first_turn is None else solo_first_turn
    final, n = _tool_loop(client, messages, tools, ctx, log, f"sub:{role}:{conversation_id}",
                          max_sub_turns, fulltext_cap=T.fulltext_cap(role),
                          read_paper_cap=read_paper_cap,
                          solo_first_turn=_solo,
                          required_artifact=T.ROLE_REQUIRED_ARTIFACT.get(role))
    s = client.summary()
    client.close()
    log.event("subagent_done", role=role, sub_tool_calls=n, tokens=s["total_tokens"],
              retries_on_parser_error=s.get("total_retries_on_parser_error", 0))
    return {"role": role, "digest": final or "(done)", "tokens": s["total_tokens"],
            "calls": s["total_calls"],
            "retries_on_parser_error": s.get("total_retries_on_parser_error", 0)}


async def run_ideation(
    problem: str,
    run_dir: str | Path,
    *,
    model: str | None = None,
    query_paper_id: Optional[int] = None,
    max_turns: int = 100,
    max_sub_turns: int = 40,
) -> dict:
    run_dir = Path(run_dir).resolve()
    docs_dir = run_dir / "outputs"   # agent-written working docs
    logs_dir = run_dir / "logs"      # debug logs (harness_log/llm_calls/summary/_sublogs)
    (docs_dir / "papers").mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    log = RunLogger(logs_dir)
    log_registry = SharedLogRegistry()
    log.event("run_start", model=model, query_paper_id=query_paper_id)
    (docs_dir / "problem.md").write_text(problem, encoding="utf-8")

    orch_sys = orchestrator_prompt(
        write_hint="edit_doc(command='view'|'create'|'str_replace'|'insert', path, ...)",
        delegate_hint="spawn_subagent(role, task)")
    roles = T.available_roles()

    base_url, text_tools = _route_for_model(model)
    client = AgenticLLMClient(api_key=_api_key(), model=model,
                              base_url=base_url, text_tools=text_tools,
                              max_tokens=8192, log_dir=logs_dir,
                              # The orchestrator reasons with thinking on.
                              extra_body={"chat_template_kwargs": {"enable_thinking": True}})
    ctx = T.ToolContext(run_dir=run_dir, docs_dir=docs_dir,
                        query_paper_id=query_paper_id, query_text=problem, log=log,
                        log_registry=log_registry)

    # read_paper runs a reader over the raw full text in its own context and
    # returns the cumulative digest. Every call invokes the reader rather than
    # short-circuiting on cache, so it can reuse already-answered goals and append
    # only the new ones.
    reader_acc = {"tokens": 0, "calls": 0, "n": 0, "parser_retries": 0}

    def _read_paper(paper_id: int, goals: list) -> str:
        digest_path = docs_dir / "papers" / f"{paper_id}.md"
        with _reader_lock:
            reader_acc["n"] += 1
            idx = reader_acc["n"]
        # The digest persists on disk across calls and is injected into the task,
        # so the reader never spends a turn viewing it first.
        prior = ""
        try:
            if digest_path.exists():
                prior = digest_path.read_text(encoding="utf-8").strip()
        except OSError:
            prior = ""
        task = build_reader_task(paper_id, goals, prior)
        # Single-turn by default, matching the reported setup and the training
        # environment. Set the variable to 0 for the agentic reader instead.
        if os.environ.get("IDEASCIENTIST_READER_SINGLE_TURN", "1").strip().lower() not in ("0", "false", "no"):
            content = read_paper_single_turn(paper_id, goals, prior, model=model,
                                             base_url=None, api_key=None, run_dir=run_dir)
            new_digest = (prior + "\n\n" + content).strip() if (prior and content) else (content or prior).strip()
            try:
                digest_path.parent.mkdir(parents=True, exist_ok=True)
                digest_path.write_text(new_digest, encoding="utf-8")
            except OSError:
                pass
            return new_digest or "(reader produced no digest)"
        r = _run_subagent("paper_reader", task, ctx, model, log,
                          log_key=f"paper_{paper_id}",
                          conversation_id=f"{paper_id}#{idx}",
                          max_sub_turns=None, solo_first_turn=False)
        with _reader_lock:
            reader_acc["tokens"] += r.get("tokens", 0)
            reader_acc["calls"] += r.get("calls", 0)
            reader_acc["parser_retries"] += r.get("retries_on_parser_error", 0)
        if digest_path.exists():
            content = digest_path.read_text(encoding="utf-8")
            if content.strip():
                return content
        return r.get("digest", "(reader produced no digest)")

    ctx.reader_fn = _read_paper

    # The orchestrator may write only plan.md. Each other artifact is owned by the
    # subagent that produces it and is read-only here, so changing one means
    # respawning its owner with feedback. Subagents run with write_scope cleared.
    ctx.write_scope = frozenset({"plan.md"})

    # The orchestrator gets no retrieval and no paper-content tool. It edits
    # plan.md, reads the subagents' files, and spawns subagents — so full text and
    # raw search results never enter its context, which is what keeps the
    # decomposition inspectable.
    orch_tools = T.openai_specs([
        "edit_doc",
    ]) + [_spawn_spec(roles)]
    messages = [{"role": "system", "content": orch_sys},
                {"role": "user", "content": T.ORCHESTRATOR_TASK.format(problem=problem)}]

    n_tool_calls = n_subagents = sub_tokens = sub_calls = 0
    sub_parser_retries = 0   # aggregated retries_on_parser_error across all subagents
    orch_nudges = 0          # premature-stop nudges (see no-tool-call guard below)
    MAX_ORCH_NUDGES = 8
    window = context_window_for(model)
    for turn in range(max_turns):
        # Context-budget heartbeat: LOG only, never into the model-visible context
        # (train/deploy parity — no [budget] line). Shows the orchestrator's OWN usage.
        _log_heartbeat(log, client.last_prompt_tokens, window)
        client.set_step(f"orch:{turn}")
        msg = client.chat(messages, tools=orch_tools)["choices"][0]["message"]
        messages.append(msg)
        tcs = msg.get("tool_calls") or []
        if not tcs:
            # Premature-stop guard: some models (notably gpt-oss via /responses) emit
            # analysis-only with NO tool call before the report exists, which would end
            # the run with no report.json. If the final artifact isn't produced yet and
            # nudges remain, inject a continue-nudge and keep going instead of ending.
            if not (docs_dir / "report.json").exists() and orch_nudges < MAX_ORCH_NUDGES:
                orch_nudges += 1
                log.event("orch_nudge", turn=turn, n=orch_nudges, reason="no_tool_calls_before_report")
                messages.append({"role": "user", "content": (
                    "You have NOT yet produced the final report (report.json), so the task is "
                    "not complete. Do not stop or reply with analysis only. Continue the workflow "
                    "by issuing a tool call now — e.g. spawn_subagent(role='report_writer', task=...) "
                    "once candidate ideas exist, or spawn the next needed subagent / edit_doc. "
                    "Respond with a tool call, not prose.")})
                continue
            log.event("loop_end", turn=turn, reason="no_tool_calls")
            break

        results: dict[str, str] = {}
        spawns = []  # (tool_call_id, coroutine)
        for tc in tcs:
            n_tool_calls += 1
            fn = tc["function"]["name"]
            fargs, perr = _parse_tool_args(tc)
            if perr:
                results[tc["id"]] = perr[0]
                log.event("tool_error", tool=fn, error="bad_json_args", raw_args=perr[1])
                continue
            if fn == "spawn_subagent":
                _srole = fargs.get("role", "")
                # A PIPELINE ABLATION removes whole stages from the action space. The
                # spawn_subagent enum is already narrowed via T.available_roles(), so
                # this only catches a model that ignores the schema — refuse rather
                # than silently run a role the arm is supposed to have removed.
                if _srole not in roles:
                    results[tc["id"]] = (
                        f"ERROR: role {_srole!r} is not available in this run. "
                        f"Available roles: {', '.join(roles)}."
                    )
                    log.event("tool_error", tool="spawn_subagent",
                              error=f"role_not_available:{_srole}")
                    continue
                n_subagents += 1
                log.event("spawn_subagent", role=_srole)
                spawns.append((tc["id"], asyncio.to_thread(
                    _run_subagent, _srole, fargs.get("task", ""),
                    ctx, model, log, _srole, f"{_srole}#{n_subagents}",
                    max_sub_turns)))
            else:
                try:
                    results[tc["id"]] = T.run_tool(fn, fargs, ctx, tool_call_id=tc["id"])
                except Exception as exc:  # noqa: BLE001
                    results[tc["id"]] = f"ERROR in {fn}: {exc}"
                    log.event("tool_error", tool=fn, error=str(exc), tool_call_id=tc["id"])

        if spawns:  # run all subagents of this turn concurrently
            done = await asyncio.gather(*[c for _, c in spawns], return_exceptions=True)
            for (tid, _), r in zip(spawns, done):
                if isinstance(r, Exception):
                    results[tid] = f"ERROR in subagent: {r}"
                    log.event("tool_error", tool="spawn_subagent", error=str(r))
                else:
                    results[tid] = r["digest"]
                    sub_tokens += r["tokens"]; sub_calls += r["calls"]
                    sub_parser_retries += r.get("retries_on_parser_error", 0)

        for tc in tcs:  # append tool results in original order (matched by id)
            # ``name`` REQUIRED for text-tools parity — see _tool_loop for rationale.
            messages.append({"role": "tool", "tool_call_id": tc["id"],
                             "name": tc["function"]["name"],
                             "content": results.get(tc["id"], "(no result)")})
    else:
        log.event("loop_end", turn=max_turns, reason="max_turns")

    produced = sorted(p.name for p in docs_dir.glob("*.md")) + \
        sorted(p.name for p in docs_dir.glob("*.json")) + \
        [f"papers/{p.name}" for p in (docs_dir / "papers").glob("*.md")]
    orch = client.summary()
    orch_parser_retries = orch.get("total_retries_on_parser_error", 0)
    # Fold nested paper_reader (read_paper) usage into the subagent totals.
    sub_tokens += reader_acc["tokens"]
    sub_calls += reader_acc["calls"]
    sub_parser_retries += reader_acc["parser_retries"]
    summary = {
        "run_dir": str(run_dir), "model": model,
        "orchestrator_tool_calls": n_tool_calls, "subagents_spawned": n_subagents,
        "papers_read": reader_acc["n"],
        "orchestrator_tokens": orch["total_tokens"], "orchestrator_calls": orch["total_calls"],
        "subagent_tokens": sub_tokens, "subagent_calls": sub_calls,
        "total_tokens": orch["total_tokens"] + sub_tokens,
        # One count per re-issued call whose prior attempt did not parse, summed
        # across the orchestrator and every subagent.
        "orchestrator_retries_on_parser_error": orch_parser_retries,
        "subagent_retries_on_parser_error": sub_parser_retries,
        "total_retries_on_parser_error": orch_parser_retries + sub_parser_retries,
        "produced_files": produced,
    }
    log.event("run_done", orchestrator_tool_calls=n_tool_calls, subagents_spawned=n_subagents, produced=produced)
    log_registry.close_all()
    log.close()
    client.close()
    return summary
