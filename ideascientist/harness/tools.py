"""Tool specifications and execution for the harness.

Two policies live here because both the production loop and the GRPO training
environment import them, and the system only works if train and deploy agree:

*Reader-only full text.* No first-tier role is granted a full-text tool. The
only route to a paper's body is ``read_paper``, which runs a paper reader in a
separate context and returns a digest, so full text never enters a planning or
synthesis context.

*The phase-completion contract.* A producer that tries to finish without
writing its artifact gets the same nudge text and the same turn budget in
training as in deployment.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from ideascientist.harness.corpus import (
    get_paper_biblio,
    get_stored_full_text,
    keyword_search,
)
from ideascientist.harness.roles import ROLES

logger = logging.getLogger(__name__)

FULLTEXT_CHAR_LIMIT = 120_000

# Only the paper reader may pull a paper's body. Enforced at the tool-availability
# level in role_tools(): a role with cap 0 is never granted a full-text tool, so
# full text cannot reach a first-tier context even if a prompt asked for it.
FULLTEXT_TOOLS = {"get_full_text"}
FULLTEXT_CAPS = {"paper_reader": 1}
DEFAULT_FULLTEXT_CAP = 0

# Soft cap on reads per sub-agent task, enforced by the runner's tool loop.
READ_PAPER_TOOL = "read_paper"
READ_PAPER_CAP = 20


# Initial task handed to the orchestrator ({problem} is filled in by the runner).
ORCHESTRATOR_TASK = "Research problem:\n\n{problem}\n\nRun the full procedure now."

# Subagent roles that receive the problem context (challenge + problem_definition)
# prepended to their task. Kept identical across production and GRPO training so a
# trained subagent sees the exact same user turn at deploy time.
PROBLEM_CONTEXT_ROLES = frozenset({"gap_finder", "innovator", "report_writer"})


# The phase-completion contract. Both the production tool loop and the GRPO
# environment import these, so a trained policy meets an identical turn budget and
# nudge text at deploy time. Never re-declare them in a consumer.
ROLE_REQUIRED_ARTIFACT = {"gap_finder": "gaps.md", "innovator": "candidates.md",
                          "report_writer": "report.json"}
ARTIFACT_MIN_CHARS = 20   # a written artifact must have >= this many non-whitespace chars
MAX_PHASE_NUDGES = 3      # max "you have not written X yet" nudges before finishing anyway
PHASE_TURN_BUDGET = 20    # per-turn cap for these roles (training MAX_PHASE_TURNS)
# Per-artifact tail appended to the generic nudge body (agent-specific noun/instruction).
ARTIFACT_NUDGE_TAILS = {
    "gaps.md": (
        "(or command='insert' to grow it) with your actual gap analysis. Do NOT "
        "reply with a plain status message until gaps.md exists.)"
    ),
    "candidates.md": (
        "(or command='insert' to grow it) with your actual research-intuition "
        "candidate. Do NOT reply with a plain status message until candidates.md "
        "exists.)"
    ),
    "report.json": (
        "with your actual results-masked solution proposal as a single JSON object. Do "
        "NOT reply with a plain status message until report.json exists.)"
    ),
}


def artifact_nudge(artifact: str) -> str:
    """Phase-completion nudge shown when a role tries to finish without writing its
    required artifact. Shared verbatim by the production loop and the GRPO training env
    so the trained policy sees the SAME nudge text at train and deploy time."""
    tail = ARTIFACT_NUDGE_TAILS.get(artifact, "")
    return (
        f"(You have not written {artifact} yet — the task is not complete until you "
        f'do. Emit a tool call now: <tool_call>{{"name": "edit_doc", "arguments": '
        f'{{"command": "create", "path": "{artifact}", "content": "..."}}}}</tool_call> '
        f"{tail}"
    )



def build_subagent_user_message(task: str, problem: str | None) -> str:
    """The subagent's user turn: the problem context, then the orchestrator's task.

    Production passes the ``--problem`` string and GRPO training passes the
    label's challenge and problem definition, through this one function, so the
    two framings cannot drift.
    """
    problem = (problem or "").strip()
    task = (task or "").strip()
    return f"{problem}\n\n{task}" if problem else task


# ---- Context-budget heartbeat (built-in, NOT a tool) ----
# The runner injects this one line into EACH agent's context every turn (caller-
# relative: prompt_tokens is THIS agent's own current context size). It is not a
# tool the agent elects to call — it is always present, so the agent always knows
# how full its context is and how much room is left.
def budget_heartbeat(prompt_tokens: int, window: int) -> str:
    if not prompt_tokens:
        return f"[budget] your context: starting (~0 used of {window:,} token window)."
    if not window:
        return f"[budget] your context: {prompt_tokens:,} tokens used (window unknown)."
    left = max(0, window - prompt_tokens)
    pct = int(round(100 * prompt_tokens / window))
    return (
        f"[budget] your context: {prompt_tokens:,} / {window:,} tokens used "
        f"({left:,} left, {pct}% full). Persist progress to disk and wrap up before it fills."
    )


# --------------------------------------------------------------- execution ctx


@dataclass
class ToolContext:
    run_dir: Path
    query_paper_id: Optional[int] = None
    query_text: Optional[str] = None
    log: Any = None
    # Agent-written artifacts live in the run's outputs/ subfolder; debug logs
    # live under logs/. Falls back to run_dir when unset.
    docs_dir: Optional[Path] = None
    # Backs the read_paper tool: runs a paper reader over one paper's full text
    # in its own context and returns the digest. The tool errors when unwired.
    reader_fn: Optional[Callable[[int, list], str]] = None
    # Hands out shared per-role and per-paper log writers, so every sub-agent of
    # a role appends to one file rather than opening its own.
    log_registry: Any = None
    # Which files this caller may WRITE. None means the full allowlist. The
    # runner sets {"plan.md"} for the orchestrator: it owns the research plan and
    # every other artifact belongs to the sub-agent that produces it. Reads are
    # never scoped — the orchestrator has to be able to see all of them.
    write_scope: Optional[frozenset] = None


def _log(ctx: ToolContext, kind: str, **kw: Any) -> None:
    if ctx.log is not None:
        ctx.log.event(kind, **kw)


def _safe_path(run_dir: Path, rel: str) -> Path:
    # Resolve both sides: run_dir may cross a symlink, and comparing a resolved
    # candidate against an unresolved root makes a legitimate path look like an escape.
    root = run_dir.resolve()
    p = (root / rel).resolve()
    if root not in p.parents and p != root:
        raise ValueError(f"path escapes run dir: {rel}")
    return p


def _docs_root(ctx: ToolContext) -> Path:
    return ctx.docs_dir or ctx.run_dir


# The fixed set of files an agent may write. Anything else is rejected, so a
# run directory always has the same shape and the reward can rely on it.
_ALLOWED_DOCS = frozenset(
    {
        "plan.md",
        "problem.md",
        "gaps.md",
        "candidates.md",
        "scores.md",
        "report.json",
        "report.md",
    }
)
_PAPERS_RE = re.compile(r"^papers/\d+\.md$")  # per-paper reader digests


def _check_allowed(rel: str, scope: Optional[frozenset] = None) -> None:
    norm = rel.strip().lstrip("./")
    if norm not in _ALLOWED_DOCS and not _PAPERS_RE.match(norm):
        raise ValueError(
            f"file not permitted: {rel!r}. Writable files are: "
            f"{sorted(_ALLOWED_DOCS)} or papers/<id>.md"
        )
    # Per-caller write scope (e.g. the orchestrator may write only plan.md). None
    # means no extra restriction beyond the global allowlist above.
    if scope is not None and norm not in scope:
        raise ValueError(
            f"write not permitted for this caller: {rel!r}. You may WRITE only "
            f"{sorted(scope)} — other files are read-only for you. To change a file "
            "you do not own, spawn the subagent responsible for it with feedback."
        )


# --------------------------------------------------------------- executors


def _exec_keyword_search(args: dict, ctx: ToolContext) -> str:
    hits = keyword_search(list(args.get("queries", [])), k=int(args.get("k", 30)))
    _log(ctx, "tool:keyword_search", n=len(hits))
    return json.dumps(hits, ensure_ascii=False)


def _exec_get_paper_biblio(args: dict, ctx: ToolContext) -> str:
    f = get_paper_biblio(int(args["paper_id"])) or {}
    _log(ctx, "tool:get_paper_biblio", paper_id=args.get("paper_id"), found=bool(f))
    return json.dumps(f, ensure_ascii=False)


def _exec_read_paper(args: dict, ctx: ToolContext) -> str:
    """Pull ONE paper's digest on demand. Delegates to ctx.reader_fn, which runs a
    paper_reader over the raw full text in its own context and returns the digest
    (full text never enters the caller's context)."""
    if ctx.reader_fn is None:
        return (
            "(read_paper is not wired in this harness; work from get_paper_biblio + "
            "keyword_search, or ask the orchestrator)"
        )
    paper_id = int(args["paper_id"])
    goals = args.get("goals") or []
    if isinstance(goals, str):
        goals = [goals]
    _log(ctx, "tool:read_paper", paper_id=paper_id, n_goals=len(goals))
    return ctx.reader_fn(paper_id, list(goals))


def coerce_paper_id(args: dict):
    """Best-effort int paper_id from tool args, tolerating ``paper_id`` or
    ``paper_ids`` (scalar or non-empty list). Returns int, or None if absent /
    unparseable. Shared by the harness executor and the GRPO training env so both
    group read_paper calls under the SAME id."""
    # The model occasionally emits `arguments: "null"` / a scalar / a bare list,
    # which json.loads() turns into a non-dict (None/int/list) WITHOUT raising, so
    # guard here (the shared helper) rather than crash the whole rollout in .get().
    if not isinstance(args, dict):
        return None
    pid = args.get("paper_id")
    if pid is None:
        plural = args.get("paper_ids")
        if isinstance(plural, list) and plural:
            pid = plural[0]
        elif plural is not None and not isinstance(plural, list):
            pid = plural
    if pid is None:
        return None
    try:
        return int(pid)
    except (TypeError, ValueError):
        return None


def coerce_tool_args(args: Any) -> dict:
    """Normalize parsed tool-call ``arguments`` into a dict the executors can index.

    ``json.loads`` turns some malformed arguments into a non-dict without
    raising — ``"null"``, ``"5"``, ``"[1,2]"``. Every executor then does
    ``args.get(...)``, so the failure would surface as an AttributeError deep in
    a rollout and, in training, take the whole batch down. Coercing to an empty
    dict at this one shared choke point lets the executor answer with its own
    missing-field message, which the model can recover from. A real dict passes
    through untouched.
    """
    if isinstance(args, dict):
        return args
    return {}


def union_read_paper_goals(tool_calls: "list[dict] | None") -> "dict[int, list[str]]":
    """Map each paper id to the union of goals across one turn's read_paper calls.

    Executing several calls for the same paper independently makes the first
    digest miss the later calls' goals, and training and deployment then diverge:
    the training env prefetches and dedups by id, while the serial loop answers
    each in turn. Unioning up front means every call in the turn returns the same
    complete digest. Per turn only — cross-turn accumulation stays with the
    reader's append-only digest.
    """
    out: dict[int, list[str]] = {}
    for tc in tool_calls or []:
        fn = tc.get("function") or {}
        if fn.get("name") != READ_PAPER_TOOL:
            continue
        raw = fn.get("arguments")
        if raw is None:
            raw = "{}"
        try:
            args = json.loads(raw) if isinstance(raw, str) else dict(raw)
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
        pid = coerce_paper_id(args)
        if pid is None:
            continue
        goals = args.get("goals") or []
        if isinstance(goals, str):
            goals = [goals]
        elif not isinstance(goals, (list, tuple)):
            # model occasionally emits a scalar (e.g. an int) for goals; wrap
            # defensively so iteration below never raises 'int is not iterable'.
            goals = [goals]
        bucket = out.setdefault(pid, [])
        for g in goals:
            g = g if isinstance(g, str) else str(g)
            if g not in bucket:
                bucket.append(g)
    return out


def _exec_get_full_text(args: dict, ctx: ToolContext) -> str:
    paper_id = args.get("paper_id")
    if paper_id is None:
        return "(get_full_text needs a paper_id)"
    txt = get_stored_full_text(int(paper_id))
    _log(ctx, "tool:get_full_text", paper_id=paper_id, chars=len(txt))
    return txt[:FULLTEXT_CHAR_LIMIT] if txt else "(not stored locally)"


def _validate_doc(path: str, text: str) -> "str | None":
    """Return a refusal message if the write would leave an allowed file broken,
    else None. Currently: report.json must parse as JSON (ACI validate-on-write)."""
    norm = path.strip().lstrip("./")
    if norm == "report.json":
        try:
            json.loads(text)
        except json.JSONDecodeError as e:
            return (
                f"refused: report.json would not be valid JSON ({e}). "
                "Fix the object and retry — nothing was written."
            )
    return None


def _edit_doc_core(args: dict, get, put, log=None, write_scope=None) -> str:
    """The storage-agnostic engine behind edit_doc.

    Backends supply ``get(path)``, ``put(path, text)``, an optional ``log``, and
    an optional ``write_scope`` restricting which filenames mutating commands may
    target — reads are never restricted.

    Commands: ``view``, ``create``, ``str_replace``, ``insert``. str_replace
    fails loudly on zero matches, or on more than one without ``replace_all``,
    rather than silently editing the wrong place. Mutating commands also pass the
    filename allowlist and leave report.json parseable.
    """

    def _emit(**kw):
        if log:
            log(**kw)

    cmd = args.get("command") or ""
    if not isinstance(cmd, str):
        cmd = str(cmd)
    cmd = cmd.strip()
    path = args.get("path", "")
    # The model may emit a non-string / missing path (or a non-string command); every
    # branch below calls path.strip() / get(path), so coerce here rather than crash the
    # rollout. Empty/blank path -> a normal refusal the model can recover from.
    if not isinstance(path, str):
        path = "" if path is None else str(path)
    if not path.strip():
        return "refused: edit_doc needs a non-empty string 'path'."

    # view is the only read; it does NOT require the allowlist (papers/<id>.md ok).
    if cmd == "view":
        text = get(path)
        if text is None:
            return f"(no such file: {path}) — use command='create' first"
        lines = text.splitlines()
        start, end = 1, len(lines)
        vr = args.get("view_range")
        if isinstance(vr, (list, tuple)) and len(vr) == 2:
            # Tolerate a malformed view_range (non-int / null elements) — fall back to
            # the whole-file default rather than raise int() TypeError/ValueError.
            try:
                start = max(1, int(vr[0]))
                end = len(lines) if int(vr[1]) == -1 else min(len(lines), int(vr[1]))
            except (TypeError, ValueError):
                start, end = 1, len(lines)
        numbered = "\n".join(f"{i}\t{lines[i-1]}" for i in range(start, end + 1))
        _emit(command="view", path=path, lines=len(lines), start=start, end=end)
        return numbered if numbered else "(empty file)"

    # all mutating commands require the allowlist (+ optional per-caller write scope).
    _check_allowed(path, write_scope)

    if cmd == "create":
        content = args.get("content", "")
        if not isinstance(content, str):
            content = "" if content is None else str(content)
        bad = _validate_doc(path, content)
        if bad:
            return bad
        put(path, content)
        _emit(command="create", path=path, chars=len(content))
        return f"wrote {path} ({len(content)} chars)"

    if cmd == "str_replace":
        text = get(path)
        if text is None:
            return f"(no such file: {path}) — use command='create' first"
        old = args.get("old_str", "")
        new = args.get("new_str", "")
        if not isinstance(old, str):
            old = "" if old is None else str(old)
        if not isinstance(new, str):
            new = "" if new is None else str(new)
        replace_all = bool(args.get("replace_all", False))
        if not old:
            return "refused: str_replace needs a non-empty old_str"
        n = text.count(old)
        if n == 0:
            return (
                "refused: old_str not found in "
                f"{path} — copy the exact existing text (whitespace included)."
            )
        if n > 1 and not replace_all:
            return (
                f"refused: old_str matches {n} times in {path} — add surrounding "
                "context so it is unique, or pass replace_all=true to change all."
            )
        count = n if replace_all else 1
        updated = text.replace(old, new, count)
        bad = _validate_doc(path, updated)
        if bad:
            return bad
        put(path, updated)
        _emit(
            command="str_replace",
            path=path,
            removed=len(old),
            added=len(new),
            occurrences=count,
        )
        return f"replaced {count} occurrence{'s' if count != 1 else ''} in {path}"

    if cmd == "insert":
        content = args.get("content", "")
        if not isinstance(content, str):
            content = "" if content is None else str(content)
        text = get(path) or ""
        lines = text.splitlines(keepends=True)
        try:
            k = int(args.get("insert_line"))
        except (TypeError, ValueError):
            return "refused: insert needs an integer insert_line (0 = top of file)"
        k = max(0, min(k, len(lines)))
        chunk = content if content.endswith("\n") or content == "" else content + "\n"
        updated = "".join(lines[:k]) + chunk + "".join(lines[k:])
        bad = _validate_doc(path, updated)
        if bad:
            return bad
        put(path, updated)
        _emit(command="insert", path=path, insert_line=k, chars=len(content))
        return f"inserted {len(content)} chars after line {k} of {path}"

    return (
        f"refused: unknown command {cmd!r}. Use one of: "
        "view, create, str_replace, insert."
    )


def _exec_edit_doc(args: dict, ctx: ToolContext) -> str:
    """Disk-backed edit_doc: wraps _edit_doc_core with a filesystem backend that
    resolves every path against the run's docs dir via _safe_path (escape guard)."""
    root = _docs_root(ctx)

    def get(path: str):
        p = _safe_path(root, path)
        return p.read_text(encoding="utf-8") if p.exists() else None

    def put(path: str, text: str) -> None:
        p = _safe_path(root, path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")

    return _edit_doc_core(
        args,
        get,
        put,
        log=lambda **kw: _log(ctx, "tool:edit_doc", **kw),
        write_scope=ctx.write_scope,
    )


# DB read/write skills (local capability, dedicated harness DB)






@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict
    run: Callable[[dict, ToolContext], str]



TOOLS: dict[str, ToolSpec] = {
    "keyword_search": ToolSpec(
        "keyword_search",
        "Keyword FTS5/BM25 search over stored paper text for a list of queries. Returns ranked "
        "papers (id, title, bm25). Only papers before the cutoff date are returned. Each query is "
        "matched as an OR of its words, so use a SHORT set of focused keywords per query (a handful "
        "of terms) — not a sentence or paragraph. At most 48 words per query and 12 queries per "
        "call are used; anything beyond that is ignored, so keep queries concise and targeted.",
        {
            "type": "object",
            "properties": {
                "queries": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "List of short keyword queries (at most 12 used). Keep EACH query to a "
                        "handful of focused terms — at most 48 words; words are OR-matched and any "
                        "beyond 48 are ignored. Do NOT pass full sentences or paragraphs."
                    ),
                },
                "k": {"type": "integer"},
            },
            "required": ["queries"],
        },
        _exec_keyword_search,
    ),
    "get_paper_biblio": ToolSpec(
        "get_paper_biblio",
        "Get a paper's ROOT bibliographic fields by id (ground-truth `papers` columns only). "
        "Returns: id, title, abstract, date, year. Does NOT return any generated summary "
        "(no problem_definition / challenge / techniques). Use for titles, dates, and citation "
        "anchors. To understand a paper's actual content/method, call read_paper.",
        {
            "type": "object",
            "properties": {"paper_id": {"type": "integer"}},
            "required": ["paper_id"],
        },
        _exec_get_paper_biblio,
    ),
    "read_paper": ToolSpec(
        "read_paper",
        "Read ONE paper's full body in a FRESH context and get back a focused digest (the full "
        "text never enters your context). Pass the paper id and 1-5 specific reading goals "
        "(concrete questions). Returns the cumulative goal-by-goal digest (papers/<id>.md). "
        "Re-calling with NEW goals appends their answers; goals already answered are reused, not "
        "re-read. Soft-capped per task.",
        {
            "type": "object",
            "properties": {
                "paper_id": {"type": "integer"},
                "goals": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["paper_id", "goals"],
        },
        _exec_read_paper,
    ),
    "get_full_text": ToolSpec(
        "get_full_text",
        "Read the STORED full text of a paper (LOCAL, no network) by vault paper id. "
        "Returns '(not stored locally...)' if we don't have the body cached.",
        {
            "type": "object",
            "properties": {"paper_id": {"type": "integer"}},
            "required": ["paper_id"],
        },
        _exec_get_full_text,
    ),
    "edit_doc": ToolSpec(
        "edit_doc",
        "Unified file I/O over the run's allowed docs (plan.md, gaps.md, candidates.md, "
        "scores.md, report.json/report.md, problem.md, papers/<id>.md). "
        "ONE tool, four commands via `command`:\n"
        "  • view — read a file; optional view_range [start,end] (1-indexed, end=-1 means EOF). "
        "Returns line-numbered content.\n"
        "  • create — write/overwrite the WHOLE file with `content` (use for a new file or a full rewrite).\n"
        "  • str_replace — replace `old_str` with `new_str` IN PLACE. `old_str` must appear "
        "EXACTLY ONCE (copy the existing text verbatim, whitespace included); 0 matches is "
        "rejected, and >1 matches is rejected unless you pass replace_all=true (then every "
        "occurrence is replaced).\n"
        "  • insert — insert `content` AFTER line `insert_line` (0 = top of file).\n"
        "report.json must remain valid JSON after any write, or the write is refused.",
        {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "enum": ["view", "create", "str_replace", "insert"],
                },
                "path": {"type": "string"},
                "content": {"type": "string"},
                "old_str": {"type": "string"},
                "new_str": {"type": "string"},
                "replace_all": {"type": "boolean"},
                "insert_line": {"type": "integer"},
                "view_range": {"type": "array", "items": {"type": "integer"}},
            },
            "required": ["command", "path"],
        },
        _exec_edit_doc,
    ),
}


def fulltext_cap(role: str) -> int:
    """How many full-text reads a role may perform per task."""
    return FULLTEXT_CAPS.get(role, DEFAULT_FULLTEXT_CAP)



def openai_specs(names: list[str]) -> list[dict]:
    """Function-calling specs for the named tools."""
    return [
        {
            "type": "function",
            "function": {
                "name": TOOLS[n].name,
                "description": TOOLS[n].description,
                "parameters": TOOLS[n].parameters,
            },
        }
        for n in names
    ]


def run_tool(name: str, args: dict, ctx: ToolContext, tool_call_id: str = "") -> str:
    """Run a tool and emit a ``tool_result`` event with the FULL return string.

    ``tool_call_id`` is the model's per-call id (when known) so the visualizer can
    pair each tool_call with its result. SDK callers may pass an empty string.
    Per project convention: store the entire result; never truncate at write time.
    """
    if name not in TOOLS:
        result = f"(unknown tool: {name})"
        _log(
            ctx,
            "tool_result",
            tool=name,
            tool_call_id=tool_call_id,
            result=result,
            chars=len(result),
            error="unknown_tool",
        )
        return result
    # A malformed tool call must not take down the loop, so bad arguments become
    # {} and an executor exception becomes an error observation. Both guards are
    # scoped to one call; the judge, reward and transport paths do not go through
    # run_tool, so this cannot mask a real infrastructure failure.
    safe_args = coerce_tool_args(args)
    try:
        result = TOOLS[name].run(safe_args, ctx)
    except Exception as exc:  # noqa: BLE001 — per-tool-call robustness boundary
        result = f"(tool {name} error: {type(exc).__name__}: {exc})"
        import sys as _sys
        print(
            f"[run_tool] non-fatal tool error in {name!r}: "
            f"{type(exc).__name__}: {exc} | args={safe_args!r}",
            file=_sys.stderr, flush=True,
        )
        _log(
            ctx,
            "tool_result",
            tool=name,
            tool_call_id=tool_call_id,
            result=result,
            chars=len(result),
            error=f"{type(exc).__name__}: {exc}",
        )
        return result
    _log(
        ctx,
        "tool_result",
        tool=name,
        tool_call_id=tool_call_id,
        result=result,
        chars=len(result or ""),
    )
    return result


# --------------------------------------------------------------- subagent roles

def role_prompt(role: str) -> str:
    """``role``'s system prompt.

    Training and inference both read prompts through here so they cannot drift.
    """
    return ROLES[role].prompt


def available_roles() -> list[str]:
    """Roles the orchestrator may spawn.

    The paper reader is excluded: it is not a peer sub-agent but the second tier
    behind the ``read_paper`` tool.
    """
    return [r for r in ROLES if r != "paper_reader"]


def role_tools(role: str) -> list[str]:
    """A role's tool names, with full-text tools withheld unless its cap allows."""
    names = ROLES[role].tools
    if fulltext_cap(role) == 0:
        names = [t for t in names if t not in FULLTEXT_TOOLS]
    return list(names)
