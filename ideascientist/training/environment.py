"""Shared single-phase multi-turn NeMo-RL environment for the per-role GRPO training.

``BaseAgentEnvironment`` is the single source of truth for the multi-turn rollout
loop that ``gap_finder`` / ``innovator`` / ``report_writer`` all use. Each agent's
overlay module subclasses it and sets a handful of class attributes / hooks; the
rest (tool parsing, ChatML re-priming, concurrent paper_reader prefetch, phase
gate + nudges, per-rollout persistence, terminal reward wiring) is identical and
lives here so a fix while debugging one agent applies to all three.

Kept verbatim from the reference end-to-end env (they are the hard-won fixes):
  - ``_parse_tool_calls`` (3 formats; JSON repair reuses the production
    ``harness.clients.tool_use``) and the bare-call fallback.
  - ChatML re-priming (``_wrap_observation`` / ``_OBS_PRE`` / ``_OBS_SUF``).
  - Tool dispatch delegating to production ``harness.corpus``.
  - Phase-completion gate + nudges (the model must WRITE the required artifact
    before it can signal completion) and productive-tool-call accounting.
  - Per-rollout persistence under ``output_dir/rollouts/<split>/step_<NNN>/``.

Per-agent overrides (see the thin subclasses in ``environments.py``):
  ROLE, REQUIRED_ARTIFACT, INPUT_SEEDS, REWARD_MODULE, SURVEY_PROMPT,
  AGG_KEYS, _build_details(), _persist_extra().
"""
from __future__ import annotations

import importlib
import importlib.util
import json
import os
import shutil
import tempfile
import threading
import time
import traceback
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional

import ray
import torch
from nemo_rl.data.interfaces import LLMMessageLogType
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.environments.interfaces import EnvironmentInterface, EnvironmentReturn
from nemo_rl.environments.metrics import calculate_pass_rate_per_prompt

# Shared primitives — the canonical copies live in production; import (don't
# re-implement) so the strip policy / id-coercion stay in lockstep with the harness.
from ideascientist.harness.thinking import after_think as _after_think
from ideascientist.harness.tools import coerce_paper_id as _coerce_paper_id

# Turn bounds + artifact minimum + nudge text: SINGLE SOURCE OF TRUTH in production
# (harness/tools.py), imported here so train and deploy share identical values/text.
# "Import production, never copy" — do NOT redeclare these literals in the overlay.
from ideascientist.harness.tools import (
    ARTIFACT_MIN_CHARS as _ARTIFACT_MIN_CHARS,
    MAX_PHASE_NUDGES,
    PHASE_TURN_BUDGET as MAX_PHASE_TURNS,
    artifact_nudge as _artifact_nudge,
)

# Concurrent paper_reader subagents per pool server. A reader has at most one
# chat call in flight, so N readers on a server make N concurrent requests. The
# cap lets the pool feed the in-flight training rollouts while leaving headroom
# for the judge calls that share it, and scales with the number of live servers.
_READER_CONCURRENCY_PER_SERVER = 32

_KNOWN_TOOLS = (
    "keyword_search", "get_paper_biblio", "read_paper", "edit_doc",
)

# Loose alias — per-sample metadata is a free-form dict threaded from the data
# processor's ``extra_env_info``; concrete fields vary per agent.
AgentMetadata = Dict[str, Any]


def _read_paper_cap() -> int:
    from ideascientist.harness import tools as T
    return int(getattr(T, "READ_PAPER_CAP", 20))


# How often _maybe_refresh_judge_urls re-reads the pool discovery dir.
_ADDR_REFRESH_SEC = 60.0


def _probe_endpoint_alive(base_url: str, timeout: float = 8.0) -> bool:
    """True iff base_url (an OpenAI-style .../v1) answers GET /models with 200.

    Used to drop stale .addr endpoints for crashed pool servers from the judge
    round-robin. A crashed server can leave its .addr file behind pointing at a
    dead port; polling /models is the same readiness signal the serve script and
    the smoke test use.
    """
    import urllib.request

    url = f"{base_url.rstrip('/')}/models"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status == 200
    except Exception:  # noqa: BLE001 — any failure means "not usable"
        return False

# Multi-turn ChatML re-priming (see reference env for the full rationale): make
# each observation a self-contained ChatML user turn + assistant re-prime so the
# vendored raw-text rollout loop re-opens the assistant turn (thinking ON) rather
# than emitting an immediate empty <|im_end|>.
_OBS_PRE = "\n<|im_start|>user\n"
_OBS_SUF = "<|im_end|>\n<|im_start|>assistant\n<think>\n"


def _wrap_observation(content: str) -> str:
    return f"{_OBS_PRE}{content}{_OBS_SUF}"


# --- Observation-string interning --------------------------------------------
# Ray's pickler memoizes by object identity, so two slots holding the same str
# serialize once. Episodes that read the same paper get a byte-identical digest,
# but wrapping each in a fresh f-string per episode destroys that identity and a
# batch balloons into a plasma object large enough for the raylet to call
# unpullable — at which point the reward threads hang in ray.get. Interning
# restores identity without touching content; bounded so it cannot grow forever.
def _intern_str(s: str) -> str:
    """Return a canonical (shared-identity) copy of ``s`` so repeated identical
    per-episode observation fragments serialize once under pickle. The returned
    object is ``== s`` (byte-identical content); only its ``id()`` may differ from
    the argument (it will match across calls for equal inputs)."""
    with _OBS_INTERN_LOCK:
        cached = _OBS_INTERN_CACHE.get(s)
        if cached is not None:
            _OBS_INTERN_CACHE.move_to_end(cached)
            return cached
        _OBS_INTERN_CACHE[s] = s
        if len(_OBS_INTERN_CACHE) > _OBS_INTERN_MAX:
            _OBS_INTERN_CACHE.popitem(last=False)
        return s


# --- Terminal-return payload slimming ----------------------------------------
# On a terminal turn the async driver reads only the reward, the terminated flag
# and the observation; it never feeds the metadata back. But the accumulated read
# digests ride along in the pickled return, pushing it over Ray's inline threshold
# and through the object store — whose plasma notification deadlocks the
# thread-saturated driver worker, so ray.get never returns.
#
# The reward is already computed actor-side and the full breakdown already written
# to disk before the return is built, so nothing downstream needs those blobs.
# Drop the digest bodies, keep the keys so the shape is unchanged. Non-terminal
# returns are left whole: their files do round-trip to the next turn.
def _slim_terminal_meta(meta: AgentMetadata) -> AgentMetadata:
    """Shallow-copy ``meta`` with heavy ``papers/<id>.md`` digest bodies stripped
    from a copied ``files`` dict, for the DRIVER-BOUND terminal return only.

    The async driver never re-consumes terminal metadata; the reward is already
    computed and persisted from the full ``meta`` before this runs. Digest KEYS are
    preserved (empty string values) so any downstream shape assumption still holds,
    while the ~11.6 KB-each bodies (the sole plasma-tipping payload) are dropped. The
    small model artifact and any non-``papers/`` files are kept byte-identical."""
    slim = dict(meta)
    files = meta.get("files")
    if isinstance(files, dict):
        slim["files"] = {
            k: ("" if k.startswith("papers/") else v)
            for k, v in files.items()
        }
    return slim


def _parse_tool_calls(assistant_msg: dict) -> list[dict]:
    """Extract tool calls: native OpenAI tool_calls first, else parse the
    assistant TEXT via the PRODUCTION parser so training and inference decode
    identically (Qwen <tool_call> blocks, JSON repair, bare-call fallback)."""
    tcs = assistant_msg.get("tool_calls")
    if tcs:
        return tcs
    content = assistant_msg.get("content", "")
    if not content:
        return []
    from ideascientist.harness.client import parse_text_tool_calls
    return parse_text_tool_calls(content, _KNOWN_TOOLS)


def annotate_turns(conversation: "list[dict] | None", role: str = "") -> list[dict]:
    """Attribute each turn a kind (single-role, so no [ROLE:] handoffs)."""
    out: list[dict] = []
    if not conversation:
        return out
    for i, msg in enumerate(conversation):
        if not isinstance(msg, dict):
            continue
        chatml_role = msg.get("role", "")
        content = msg.get("content")
        if not isinstance(content, str):
            content = "" if content is None else str(content)
        tool_calls: list[dict] = []
        if chatml_role == "assistant":
            for tc in _parse_tool_calls(msg):
                fn = tc.get("function", {})
                raw_args = fn.get("arguments", "{}")
                try:
                    parsed_args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                except (json.JSONDecodeError, TypeError):
                    parsed_args = {"_raw": raw_args}
                tool_calls.append({"name": fn.get("name", ""), "arguments": parsed_args})
        if chatml_role == "assistant":
            if tool_calls:
                kind = "assistant_tool_call"
            elif _after_think(content).strip():
                kind = "assistant_status"
            else:
                kind = "assistant_reasoning_only"
        elif i == 0:
            kind = "role_prompt"
        else:
            low = content
            if "Rollout complete. Reward:" in low:
                kind = "rollout_complete"
            elif "have not written" in low:
                kind = "nudge"
            elif "Phase turn limit" in low:
                kind = "phase_limit"
            else:
                kind = "tool_response"
        out.append({
            "turn_index": i,
            "role": chatml_role,
            "pipeline_role": role,
            "turn_kind": kind,
            "tool_calls": tool_calls,
            "chars": len(content),
            "content": content,
        })
    return out


def _execute_tool(name: str, args: dict, meta: AgentMetadata,
                  reader_cfg: Optional[dict] = None) -> str:
    """Run one tool for the training rollout, non-fatally.

    Mirrors the production ``harness.tools.run_tool`` backstop so train and deploy
    handle a malformed model tool call IDENTICALLY: coerce a non-dict ``arguments``
    (json.loads of "null"/"5"/"[...]") to {} via the shared helper, then catch any
    executor exception and return a short error observation instead of raising —
    which the vendored ``nemo_rl`` rollout driver would otherwise re-raise as
    ``Error in sample <i> rollout``, killing the whole 1024-episode batch. Scoped to
    ONE tool call: the reward/judge + DB-sentinel + Ray transport paths do NOT run
    here (rewards are computed separately in ``_compute_terminal_reward``), so this
    catch cannot mask real infra failures — only tool-arg/parsing/execution errors."""
    from ideascientist.harness.tools import coerce_tool_args
    safe_args = coerce_tool_args(args)
    try:
        return _dispatch_tool(name, safe_args, meta, reader_cfg)
    except Exception as e:  # noqa: BLE001 — per-tool-call robustness boundary
        import sys as _sys
        print(
            f"[gap_finder] non-fatal tool error in {name!r}: "
            f"{type(e).__name__}: {e} | args={safe_args!r}",
            file=_sys.stderr, flush=True,
        )
        return f"(tool {name} error: {type(e).__name__}: {e})"


def _coerce_path(args: dict) -> str:
    for key in ("path", "filename", "file", "file_path", "name"):
        v = args.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def _dispatch_tool(name: str, args: dict, meta: AgentMetadata,
                   reader_cfg: Optional[dict] = None) -> str:
    files = meta["files"]

    if name == "edit_doc":
        from ideascientist.harness.tools import _edit_doc_core
        a = dict(args)
        a["path"] = _coerce_path(a)
        return _edit_doc_core(a, files.get, files.__setitem__)

    if name == "keyword_search":
        from ideascientist.harness.corpus import keyword_search
        queries = list(args.get("queries", []))
        k = int(args.get("k", 30) or 30)
        hits = keyword_search(queries, k=k)
        return json.dumps(hits, ensure_ascii=False) if hits else "(no results)"

    if name == "get_paper_biblio":
        from ideascientist.harness.corpus import get_paper_biblio
        paper_id = _coerce_paper_id(args)
        if paper_id is None:
            return "(get_paper_biblio needs a paper_id)"
        f = get_paper_biblio(int(paper_id))
        return json.dumps(f or {}, ensure_ascii=False)

    if name == "read_paper":
        paper_id = _coerce_paper_id(args)
        goals = args.get("goals", [])
        if paper_id is None:
            return "(read_paper needs a paper_id)"
        used = int(meta.get("read_paper_used", 0) or 0)
        cap = _read_paper_cap()
        if used >= cap:
            return (
                f"(refused: this role may read at most {cap} paper(s) per task "
                f"and has already used {used}. Work from the digests you already "
                f"pulled + get_paper_biblio + keyword_search; prioritise the most "
                f"load-bearing reads.)"
            )
        meta["read_paper_used"] = used + 1
        digest_path = f"papers/{int(paper_id)}.md"
        if digest_path in files:
            return files[digest_path]
        return _read_paper_via_subagent(paper_id, goals, files, reader_cfg or {})

    if name in ("external_search", "fetch_full_text", "traverse_citations",
                "embedding_score", "db_store_paper", "db_get_paper",
                "get_full_text", "embedding_search"):
        return f"({name} not available in training environment)"

    return f"(unknown tool: {name})"


def _read_paper_via_subagent(paper_id, goals: list, files: dict,
                             reader_cfg: dict) -> str:
    """Run the PRODUCTION paper_reader second-tier sub-agent for one paper and return its
    digest, matching inference exactly.

    Unlike a raw DB dump, this invokes ``runner._run_subagent('paper_reader', ...)``
    which reads the full body in the reader's OWN context (capped at the real
    ``FULLTEXT_CHAR_LIMIT`` inside ``_exec_get_full_text``) and writes a summarized
    ``papers/<id>.md`` digest. Full text never enters the calling agent's context. The
    reader's LLM calls are routed to the served judge/reader pool via
    ``AGENT_LOOP_BASE_URL`` + ``HARNESS_TEXT_TOOLS=1`` (native tool-calling is off on
    that pool). The digest is read back into ``files['papers/<id>.md']`` — the exact
    key the reward's ``_cited_ids_were_read`` gate checks.
    """
    from ideascientist.harness import tools as T
    from ideascientist.harness.runlog import RunLogger
    from ideascientist.harness.runner import _run_subagent, build_reader_task

    if paper_id is None:
        return "(read_paper needs a paper_id)"
    pid = int(paper_id)
    digest_path = f"papers/{pid}.md"

    base_url = reader_cfg.get("base_url") or "http://localhost:8001/v1"
    model = reader_cfg.get("model") or "Qwen/Qwen3.6-27B"
    api_key = reader_cfg.get("api_key") or "EMPTY"
    output_dir = reader_cfg.get("output_dir") or tempfile.gettempdir()

    # Per-episode isolated run dir: outputs/ (docs, with papers/) + logs/.
    run_dir = Path(output_dir) / "reader_tmp" / f"{pid}_{uuid.uuid4().hex[:8]}"
    docs_dir = run_dir / "outputs"
    logs_dir = run_dir / "logs"
    (docs_dir / "papers").mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    log = RunLogger(logs_dir)

    # Goal-aware caching, for train/deploy parity. In production the digest
    # persists across read_paper calls, so a second read with new goals lets the
    # reader keep only the unanswered ones and append, while identical goals hit
    # its early exit. The training run_dir is ephemeral, so without seeding it the
    # reader would always see an empty digest and re-read the whole body.
    prior = files.get(digest_path, "")
    if isinstance(prior, str) and prior.strip():
        (docs_dir / "papers" / f"{pid}.md").write_text(prior, encoding="utf-8")

    ctx = T.ToolContext(
        run_dir=run_dir,
        docs_dir=docs_dir,
        query_text="",
        log=log,
    )
    # Inject the current digest (seeded above from the caller's cache) INTO the task
    # so the reader skips the turn-0 edit_doc(view) round-trip. Shared production
    # builder => identical task to inference (train/deploy parity).
    task = build_reader_task(pid, goals, prior)

    # IDEASCIENTIST_READER_SINGLE_TURN: skip the agentic subagent (edit_doc + done turns)
    # entirely — one completion answers the goals, the HARNESS appends to papers/<id>.md.
    # Halves pool calls, removes tool parsing + runaway loops. Env-gated for A/B.
    if os.environ.get("IDEASCIENTIST_READER_SINGLE_TURN", "").strip().lower() in ("1", "true", "yes"):
        from ideascientist.harness.runner import read_paper_single_turn
        content = read_paper_single_turn(pid, goals, prior, model=model,
                                         base_url=base_url, api_key=api_key)
        if content:                                    # NONE/empty = all goals covered
            digest = (prior + "\n\n" + content).strip() if prior else content.strip()
        else:
            digest = prior.strip()
        try:
            shutil.rmtree(run_dir, ignore_errors=True)
        except OSError:
            pass
        if not digest:
            return f"(reader produced no digest for paper {pid})"
        files[digest_path] = digest
        return digest

    # Per-call routing rather than the process-global env var: that is what lets
    # many readers in one step run against different pool servers without a race.
    # text_tools because the pool is served without auto tool choice.
    _run_subagent(
        "paper_reader", task, ctx, model=model, log=log,
        log_key=f"paper_{pid}", conversation_id=f"{pid}#1",
        allow_external=False, max_sub_turns=None,
        base_url=base_url, text_tools=True, api_key=api_key,
        solo_first_turn=False,
    )

    disk_path = docs_dir / "papers" / f"{pid}.md"
    digest = ""
    if disk_path.exists():
        digest = disk_path.read_text(encoding="utf-8").strip()
    try:
        shutil.rmtree(run_dir, ignore_errors=True)
    except OSError:
        pass
    if not digest:
        return f"(reader produced no digest for paper {pid})"
    files[digest_path] = digest
    return digest


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _metrics(batch: BatchedDataDict) -> dict:
    rewards = batch["rewards"] if batch["rewards"].ndim == 1 else batch["rewards"][:, 0]
    rewards = rewards * batch["is_end"]
    return {
        "total_reward": rewards.mean().item(),
        "pass@samples_per_prompt": calculate_pass_rate_per_prompt(
            batch["text"], rewards
        ),
        "fraction_of_samples_properly_ended": batch["is_end"].float().mean().item(),
        "num_problems_in_batch": batch["is_end"].shape[0],
        "generation_lengths": batch["generation_lengths"].float().mean().item(),
        "prompt_lengths": batch["prompt_lengths"].float().mean().item(),
    }


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
class BaseAgentEnvironment(EnvironmentInterface):
    """Single-phase multi-turn env: the agent writes ``REQUIRED_ARTIFACT`` for ONE
    assigned unit of work, then a rubric-based reward grades it. Concrete agents
    subclass this, set the class attributes below, override ``_build_details`` /
    ``_persist_extra``, and apply ``@ray.remote`` to the subclass."""

    # ---- per-agent configuration (overridden by subclasses) ----
    ROLE: str = ""
    REQUIRED_ARTIFACT: str = ""
    # {artifact_name: metadata_reference_key} seeded into the rollout files at step
    # start so the model can edit_doc(view) its inputs (empty = no seeding).
    INPUT_SEEDS: Dict[str, str] = {}
    REWARD_MODULE: str = ""
    SURVEY_PROMPT: str = ""
    # (The phase-completion nudge text — body + per-artifact tail — is centralized in
    # production tools.py:artifact_nudge / ARTIFACT_NUDGE_TAILS, keyed by REQUIRED_ARTIFACT.)
    # Numeric reward-detail keys aggregated in global_post_process_and_metrics.
    AGG_KEYS: tuple = ("total", "rubric_reward", "process_reward")

    def __init__(self, cfg: Optional[dict] = None):
        self.cfg = cfg or {}
        self.db_path = self.cfg.get("db_path", "")
        self.judge_base_url = self.cfg.get("judge_base_url", "http://localhost:8001/v1")
        self.judge_model = self.cfg.get("judge_model", "Qwen/Qwen3.6-27B")
        self.judge_api_key = self.cfg.get("judge_api_key", "EMPTY")
        self.judge_server_dir = self.cfg.get("judge_server_dir", "")
        self._judge_urls_raw: list[str] = self._read_judge_addrs()
        self._judge_urls: list[str] = self._discover_judge_urls()
        self._judge_urls_ts = time.monotonic()
        self._judge_rr = 0
        self.judge_extra_body = self.cfg.get(
            "judge_extra_body", {"chat_template_kwargs": {"enable_thinking": False}}
        )
        self.max_phase_turns = self.cfg.get("max_phase_turns", MAX_PHASE_TURNS)
        self.reward_weights_dict = self.cfg.get("reward_weights", {})

        self._reward_details: list[dict] = []

        self.output_dir = self.cfg.get("output_dir", "")
        self._rollout_dir: Optional[Path] = None
        self._episode_ct = 0
        self._persist_lock = threading.Lock()
        self._cur_split = "train"
        self._cur_step = 0
        self._instep_ct = 0
        if self.output_dir:
            try:
                self._rollout_dir = Path(self.output_dir) / "rollouts"
                self._rollout_dir.mkdir(parents=True, exist_ok=True)
            except OSError:
                self._rollout_dir = None

    def set_rollout_context(self, split: str, step: int) -> None:
        with self._persist_lock:
            self._cur_split = str(split)
            self._cur_step = int(step)
            self._instep_ct = 0

    def _read_judge_addrs(self) -> list[str]:
        urls: list[str] = []
        d = self.judge_server_dir
        if d and Path(d).is_dir():
            for f in sorted(Path(d).glob("*.addr")):
                try:
                    hp = f.read_text().strip()
                except OSError:
                    continue
                if hp:
                    urls.append(f"http://{hp}/v1")
        return urls

    def _discover_judge_urls(self) -> list[str]:
        urls = self._read_judge_addrs()
        if not urls:
            urls = [self.judge_base_url]
            return urls
        # A server writes its .addr file before the checkpoint loads and may die
        # afterwards, and round-robin has no failover, so a corpse would poison
        # 1/N of all judge and reader calls. Keep only endpoints that answer; if
        # none do, fall back to the raw list rather than strand the env.
        alive = [u for u in urls if _probe_endpoint_alive(u)]
        return alive if alive else urls

    def _maybe_refresh_judge_urls(self) -> None:
        """Pick up a judge pool that moved nodes.

        The pool is a separate job. If it is preempted and restarted it comes
        back on a DIFFERENT host and rewrites its .addr files, but discovery ran
        once in __init__, so every judge and reader call would keep hitting the
        dead host for the remainder of the run — silently, since read_paper
        errors are non-fatal, so training keeps burning nodes while every read
        fails with 'Connection refused'.

        Re-read the discovery dir at most once per _ADDR_REFRESH_SEC (a cheap glob,
        no probing) and swap endpoints in only when the address set actually
        changed. A benign race between worker threads just costs a second refresh.
        """
        now = time.monotonic()
        if now - self._judge_urls_ts < _ADDR_REFRESH_SEC:
            return
        self._judge_urls_ts = now
        snap = self._read_judge_addrs()
        if not snap or snap == self._judge_urls_raw:
            return
        alive = [u for u in snap if _probe_endpoint_alive(u)]
        self._judge_urls_raw = snap
        self._judge_urls = alive or snap
        self._judge_rr = 0
        print(f"[{getattr(self, 'ROLE', '?')}] judge pool moved: rediscovered "
              f"{len(self._judge_urls)} endpoint(s), e.g. {self._judge_urls[:1]}",
              flush=True)

    def _next_judge_url(self) -> str:
        self._maybe_refresh_judge_urls()
        url = self._judge_urls[self._judge_rr % len(self._judge_urls)]
        self._judge_rr += 1
        return url

    def _reader_cfg(self) -> dict:
        """Config for the production paper_reader subagent behind read_paper: route
        its LLM calls to the SAME served pool the judge uses (round-robin), reuse the
        served checkpoint + key, and give it a writable per-episode scratch dir."""
        return {
            "base_url": self._next_judge_url(),
            "model": self.judge_model,
            "api_key": self.judge_api_key,
            "output_dir": self.output_dir or "",
        }

    def _get_role_prompt(self) -> str:
        try:
            from ideascientist.harness import tools as _T
            if self.ROLE in _T.ROLES:
                return _T.role_prompt(self.ROLE)
        except ImportError:
            pass
        return self.SURVEY_PROMPT

    def _seed_input_files(self, meta: AgentMetadata) -> None:
        """Seed the agent's INPUT artifacts into the rollout files (once).

        Production hands the agent its input docs (e.g. gaps.md / candidates.md) it
        reads via edit_doc(view); some rewards' gates key on them. Seed each from
        its reference-label metadata key so the model can view its inputs (exactly
        as production). Idempotent: only writes if absent (the model never
        overwrites its input). No-op when INPUT_SEEDS is empty (gap_finder)."""
        files = meta["files"]
        for artifact, ref_key in self.INPUT_SEEDS.items():
            if artifact in files:
                continue
            ref = meta.get(ref_key, "") or ""
            if isinstance(ref, str) and ref.strip():
                files[artifact] = ref

    def step(
        self,
        message_log_batch: list[LLMMessageLogType],
        metadata_batch: list[AgentMetadata],
    ) -> EnvironmentReturn:
        # PASS 1 (serial): copy each rollout's meta, parse its tool calls, and
        # collect every uncached read_paper across the batch. read_paper is the
        # only slow tool — it drives a reader subagent — and running them one at a
        # time under a global lock left the pool idle and starved the GPUs. So
        # prefetch them concurrently in PASS 2 and let PASS 3 cache-hit.
        from ideascientist.harness.tools import union_read_paper_goals
        prepared: list[dict] = []
        reader_jobs: list[dict] = []
        for conversation, meta in zip(message_log_batch, metadata_batch):
            if meta.get("done", False):
                prepared.append({"conversation": conversation, "meta": meta, "skip": True})
                continue
            meta = deepcopy(meta)
            meta.setdefault("files", {})
            meta.setdefault("phase_turn", 0)
            meta.setdefault("phase_nudges", 0)
            self._seed_input_files(meta)
            last_msg = conversation[-1] if conversation else {}
            tool_calls = (
                _parse_tool_calls(last_msg)
                if last_msg.get("role") == "assistant" else []
            )
            prepared.append({
                "conversation": conversation, "meta": meta,
                "skip": False, "tool_calls": tool_calls,
            })
            # Prefetch jobs, each pinned to a distinct pool server. Papers already
            # in meta["files"] are deliberately not skipped: a re-read with new
            # goals must re-spawn the reader so it can append the unanswered ones,
            # as production does. seen_pids still dedups one pid within one turn,
            # since two jobs writing one digest path would race.
            goal_union = union_read_paper_goals(tool_calls)
            seen_pids: set[int] = set()
            read_paper_cap = _read_paper_cap()
            scheduled_reads = 0
            for tc in tool_calls:
                if tc.get("function", {}).get("name", "") != "read_paper":
                    continue
                try:
                    fn_args = json.loads(tc.get("function", {}).get("arguments", "{}"))
                except json.JSONDecodeError:
                    continue
                pid = _coerce_paper_id(fn_args)
                if pid is None:
                    continue
                pid = int(pid)
                # cap look-ahead counts re-reads too, matching PASS 3 which increments
                # read_paper_used on EVERY read_paper call (cache hits included).
                if int(meta.get("read_paper_used", 0) or 0) + scheduled_reads >= read_paper_cap:
                    continue
                if pid in seen_pids:
                    continue
                seen_pids.add(pid)
                scheduled_reads += 1
                reader_jobs.append({
                    "files": meta["files"],
                    "pid": pid,
                    "goals": goal_union.get(pid, fn_args.get("goals", [])),
                    "reader_cfg": self._reader_cfg(),
                })

        # PASS 2 (concurrent): fill each rollout's papers/<id>.md cache in parallel.
        self._prefetch_readers(reader_jobs)

        # PASS 3 (serial): per-rollout assembly. read_paper cache-hits in
        # _dispatch_tool, so this loop does not block on reader latency.
        observations = []
        rewards_list = []
        terminated_list = []
        answers_list = []
        new_metadata = []
        new_stop_strings = []

        for item in prepared:
            conversation = item["conversation"]
            meta = item["meta"]
            if item["skip"]:
                observations.append({"role": "environment", "content": "(already done)"})
                rewards_list.append(0.0)
                terminated_list.append(True)
                answers_list.append(None)
                new_metadata.append(meta)
                new_stop_strings.append(None)
                continue

            tool_calls = item["tool_calls"]
            meta["total_turns"] = meta.get("total_turns", 0) + 1

            # Per-episode backstop. Each branch below appends to all six parallel
            # lists as one group at its end, so an unanticipated exception can only
            # fire before this episode has appended anything. Checkpoint the length
            # and emit a non-terminal error observation rather than let it become
            # nemo_rl's re-raise, which kills the whole batch.
            _n_before = len(observations)
            try:
                self._process_episode_turn(
                    item, conversation, meta, tool_calls,
                    observations, rewards_list, terminated_list,
                    answers_list, new_metadata, new_stop_strings,
                )
            except Exception as exc:  # noqa: BLE001 — per-episode robustness boundary
                import sys as _sys
                import traceback as _tb
                print(
                    f"[gap_finder] non-fatal per-episode error (kept alive): "
                    f"{type(exc).__name__}: {exc}\n{_tb.format_exc()}",
                    file=_sys.stderr, flush=True,
                )
                # Roll back any partial appends this episode may have made so the six
                # lists stay length-aligned, then append exactly one safe fallback.
                del observations[_n_before:]
                del rewards_list[_n_before:]
                del terminated_list[_n_before:]
                del answers_list[_n_before:]
                del new_metadata[_n_before:]
                del new_stop_strings[_n_before:]
                observations.append({
                    "role": "environment",
                    "content": _wrap_observation(
                        "(internal error processing your tool call; it was skipped. "
                        "Continue: write your required artifact, then stop.)"
                    ),
                })
                rewards_list.append(0.0)
                terminated_list.append(False)
                answers_list.append(None)
                new_metadata.append(meta)
                new_stop_strings.append(None)

        return EnvironmentReturn(
            observations=observations,
            metadata=new_metadata,
            next_stop_strings=new_stop_strings,
            rewards=torch.tensor(rewards_list, dtype=torch.float32),
            terminateds=torch.tensor(terminated_list, dtype=torch.bool),
            answers=answers_list,
        )

    def _process_episode_turn(
        self, item, conversation, meta, tool_calls,
        observations, rewards_list, terminated_list,
        answers_list, new_metadata, new_stop_strings,
    ) -> None:
        """Process ONE episode's turn and append exactly one aligned entry to each of
        the six parallel output lists. Extracted from ``step`` so the per-episode
        backstop there can wrap it: every branch below appends its whole group at the
        end, so if this raises the caller rolls back and substitutes a safe fallback.
        Behavior is byte-identical to the previous inline code for valid calls."""
        if tool_calls:
            from ideascientist.harness.tools import union_read_paper_goals
            results = []
            reader_cfg = self._reader_cfg()
            # Same per-turn goal union as PASS 1 / production _tool_loop: every
            # read_paper for a pid this turn gets that pid's full goal set, so the
            # digest returned here matches the prefetched one and the deploy loop.
            pass3_goal_union = union_read_paper_goals(tool_calls)
            for tc in tool_calls:
                fn_name = tc.get("function", {}).get("name", "")
                try:
                    fn_args = json.loads(tc.get("function", {}).get("arguments", "{}"))
                except json.JSONDecodeError:
                    fn_args = {}
                if fn_name == "read_paper":
                    pid = _coerce_paper_id(fn_args)
                    if pid is not None and int(pid) in pass3_goal_union:
                        fn_args = {**fn_args, "goals": pass3_goal_union[int(pid)]}
                result = _execute_tool(fn_name, fn_args, meta, reader_cfg)
                results.append(f"[{fn_name}] {result}")
                if fn_name and not result.startswith("(") \
                        and result != "OK — wrote 0 chars to ":
                    meta["productive_tool_calls"] = (
                        meta.get("productive_tool_calls", 0) + 1
                    )

            obs_content = "\n\n".join(results)
            meta["phase_turn"] = meta.get("phase_turn", 0) + 1
            if meta["phase_turn"] >= self.max_phase_turns:
                obs_content += (
                    f"\n\n(Phase turn limit ({self.max_phase_turns}) reached. "
                    f"Please provide your final status now, no more tool calls.)"
                )
            obs_content = _wrap_observation(
                f"<tool_response>\n{obs_content}\n</tool_response>"
            )
            # Intern the assembled observation so episodes that produced a
            # byte-identical one this turn share a single object — see _intern_str.
            obs_content = _intern_str(obs_content)
            observations.append({"role": "environment", "content": obs_content})
            rewards_list.append(0.0)
            terminated_list.append(False)
            answers_list.append(None)
            new_metadata.append(meta)
            new_stop_strings.append(None)
            return

        # No tool call → the model wants to finish. Gate on the artifact.
        artifact = meta["files"].get(self.REQUIRED_ARTIFACT, "")
        has_artifact = (
            isinstance(artifact, str)
            and len(artifact.strip()) >= _ARTIFACT_MIN_CHARS
        )
        nudges = meta.get("phase_nudges", 0)
        if not has_artifact and nudges < MAX_PHASE_NUDGES:
            meta["phase_nudges"] = nudges + 1
            observations.append({
                "role": "environment",
                "content": _wrap_observation(_artifact_nudge(self.REQUIRED_ARTIFACT)),
            })
            rewards_list.append(0.0)
            terminated_list.append(False)
            answers_list.append(None)
            new_metadata.append(meta)
            new_stop_strings.append(None)
            return

        # Terminal: compute the rubric reward from the required artifact.
        reward, details = self._compute_terminal_reward(meta, conversation)
        self._reward_details.append(details)
        meta["done"] = True
        observations.append({
            "role": "environment",
            "content": f"Rollout complete. Reward: {reward:.4f}",
        })
        rewards_list.append(reward)
        terminated_list.append(True)
        # On a terminal turn the driver reads only rewards, terminateds and
        # observations, so ship slim metadata and scalar-only answers; the full
        # breakdown is already on disk. See _slim_terminal_meta.
        answers_list.append(json.dumps(
            {k: details.get(k) for k in ("total", "rubric_reward", "process_reward")},
            default=str,
        ))
        new_metadata.append(_slim_terminal_meta(meta))
        new_stop_strings.append(None)

    def _prefetch_readers(self, jobs: list[dict]) -> None:
        """Run every uncached read_paper in the batch CONCURRENTLY, each pinned to a
        distinct pool server. Each job writes its digest into its own rollout's
        ``files`` dict (a different dict per rollout; unique keys within one), and
        ``_read_paper_via_subagent`` builds a fresh client + uuid run_dir per call, so
        there is no shared mutable state across threads. A job that fails (returns no
        digest / raises) simply leaves its cache empty; PASS 3 then retries that one
        read serially — same result as before, just not prefetched."""
        if not jobs:
            return
        # Cross-generation dedup. The generations of one prompt call read_paper on
        # the same paper with the same goals, and the reader is a fixed tool rather
        # than the trained policy, so for a given (pid, goals, prior digest) the
        # digest is identical. Run the reader once per unique key and fan it out.
        # The prior digest is in the key, so a re-read with a divergent accumulated
        # digest is never overwritten by another rollout's.
        groups: dict[tuple, list[dict]] = {}
        for j in jobs:
            dp = f"papers/{int(j['pid'])}.md"
            prior = j["files"].get(dp, "")
            prior = prior if isinstance(prior, str) else ""
            key = (int(j["pid"]), frozenset(j.get("goals") or []), prior)
            groups.setdefault(key, []).append(j)
        reps = [m[0] for m in groups.values()]
        print(
            f"[gap_finder] reader dedup: {len(jobs)} read_paper jobs -> "
            f"{len(reps)} unique (pid,goals,prior) reads "
            f"({(1 - len(reps)/max(1,len(jobs)))*100:.0f}% deduped)",
            flush=True,
        )
        max_workers = min(
            len(reps),
            max(1, len(self._judge_urls) * _READER_CONCURRENCY_PER_SERVER),
        )
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            fut2key = {
                ex.submit(
                    _read_paper_via_subagent,
                    r["pid"], r["goals"], r["files"], r["reader_cfg"],
                ): key
                for key, members in groups.items()
                for r in (members[0],)
            }
            for f in as_completed(fut2key):
                try:
                    f.result()
                except Exception:  # noqa: BLE001 — a failed prefetch just isn't cached
                    continue
                members = groups[fut2key[f]]
                rep = members[0]
                if len(members) == 1:
                    continue
                dp = f"papers/{int(rep['pid'])}.md"
                digest = rep["files"].get(dp)
                if digest:
                    for j in members[1:]:
                        j["files"][dp] = digest

    # ---- per-agent reward-detail hooks (override in subclass) ----
    def _build_details(self, result) -> dict:
        raise NotImplementedError

    def _persist_extra(self, meta: AgentMetadata, details: dict) -> dict:
        return {}

    def _import_reward_module(self):
        """Import this env's reward module (``self.REWARD_MODULE``).

        Default (per-agent runs): a plain ``importlib.import_module`` — only one
        agent dir is on ``sys.path`` so the flat name resolves unambiguously.

        Mixed multi-agent run (``IDEASCIENTIST_MIXED=1``): all three agent dirs
        share ``sys.path``, and each ships flat-named sibling modules
        (``rubrics_with_reference`` / ``rubrics`` / ``candidates_parser``). A bare
        flat ``import rubrics_with_reference`` resolves to whichever agent is
        FIRST on the path (gap_finder), so innovator/report_writer reward imports
        grab the wrong siblings → ``ImportError``/wrong symbols. Fix: load the
        reward module as a submodule of a per-agent synthetic package rooted at
        the reward file's agent dir, so the reward module's RELATIVE imports
        (tried first in every shim, e.g. ``from .rubrics_with_reference import``)
        succeed and resolve to the correct sibling within that package — no flat
        import ever runs, no cross-agent collision. Cached in ``sys.modules`` so
        each env process loads its cluster once.
        """
        name = self.REWARD_MODULE
        if os.environ.get("IDEASCIENTIST_MIXED", "").strip() != "1":
            return importlib.import_module(name)

        import sys
        import types
        # importlib and importlib.util are imported at module scope. Importing
        # either here would rebind `importlib` as a local for the whole function,
        # so the earlier `importlib.import_module` call would raise
        # UnboundLocalError on every rollout.

        # Locate the reward module's directory without introspecting the class:
        # inside a Ray actor the env module has no __file__, and torch.package
        # makes inspect.getfile raise for actor-loaded classes. Each REWARD_MODULE
        # name is unique to one agent directory.
        agent_dir = None
        for entry in sys.path:
            if entry and os.path.isfile(os.path.join(entry, name + ".py")):
                agent_dir = os.path.abspath(entry)
                break
        if agent_dir is None:
            # Reward file not found on sys.path — fall back to the plain import
            # (per-agent-run behavior) rather than crash.
            return importlib.import_module(name)
        pkg_name = "_mixed_" + os.path.basename(agent_dir)  # e.g. _mixed_innovator
        full = pkg_name + "." + name
        if full in sys.modules:
            return sys.modules[full]
        if pkg_name not in sys.modules:
            pkg = types.ModuleType(pkg_name)
            pkg.__path__ = [agent_dir]  # relative imports resolve within agent_dir
            pkg.__package__ = pkg_name
            sys.modules[pkg_name] = pkg
        reward_path = os.path.join(agent_dir, name + ".py")
        spec = importlib.util.spec_from_file_location(full, reward_path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[full] = mod
        spec.loader.exec_module(mod)  # relative sibling imports now resolve correctly
        return mod

    def _compute_terminal_reward(
        self, meta: AgentMetadata, conversation: "LLMMessageLogType | None" = None
    ) -> tuple[float, dict]:
        from ideascientist.utils.llm import LLMSettings
        reward_mod = self._import_reward_module()
        JUDGE_FAILED_REWARD_SENTINEL = reward_mod.JUDGE_FAILED_REWARD_SENTINEL
        RewardWeights = reward_mod.RewardWeights
        compute_reward = reward_mod.compute_reward

        judge_settings = LLMSettings(
            api_key=self.judge_api_key,
            base_url=self._next_judge_url(),
            model=self.judge_model,
            extra_body=self.judge_extra_body,
        )
        rw = self.reward_weights_dict
        weights = RewardWeights(
            rubric=rw.get("rubric", 0.85),
            process=rw.get("process", 0.15),
            base_quality=rw.get("base_quality", 1.0),
            reference_quality=rw.get("reference_quality", 1.0),
            citation_f1=rw.get("citation_f1", 1.0),
        )
        try:
            result = compute_reward(
                files=meta["files"],
                metadata=meta,
                judge_settings=judge_settings,
                weights=weights,
            )
            details = self._build_details(result)
            self._persist_rollout(meta, details, conversation)
            # The judge could not be parsed after retries: emit the sentinel reward
            # so the rollout driver DROPS this row from the GRPO batch
            # (loss_multiplier=0) rather than training on a fabricated score.
            if result.judge_failed:
                return JUDGE_FAILED_REWARD_SENTINEL, details
            return result.total, details
        except Exception:
            err = {"error": traceback.format_exc()}
            self._persist_rollout(meta, err, conversation)
            return 0.0, err

    def _persist_rollout(
        self,
        meta: AgentMetadata,
        details: dict,
        conversation: "LLMMessageLogType | None" = None,
    ) -> None:
        if self._rollout_dir is None:
            return
        try:
            with self._persist_lock:
                self._episode_ct += 1
                split = self._cur_split
                step = self._cur_step
                k = self._instep_ct
                self._instep_ct += 1
            # Prefer the per-rollout target_weight_version stamped at generation
            # time: it is the optimizer step this rollout trains toward, and unlike
            # the shared _cur_step it is race-free under async, where _cur_step
            # holds the latest target rather than this rollout's.
            injected_step = meta.get("target_weight_version")
            if injected_step is not None:
                try:
                    step = int(injected_step)
                except (TypeError, ValueError):
                    pass  # keep the _cur_step fallback
            pid = meta.get("source_paper_id", "NA")
            files = meta.get("files", {}) or {}

            step_dir = self._rollout_dir / split / f"step_{step:03d}"
            step_dir.mkdir(parents=True, exist_ok=True)
            stats_path = step_dir / "rewards.jsonl"

            row = {
                "split": split,
                "step": step,
                "target_weight_version": injected_step,
                "instep_idx": k,
                "episode": self._episode_ct,
                "source_paper_id": pid,
                "source_arxiv": meta.get("source_arxiv", ""),
                "total_turns": meta.get("total_turns", 0),
                "productive_tool_calls": meta.get("productive_tool_calls", 0),
                "total": details.get("total"),
                "rubric_reward": details.get("rubric_reward"),
                "process_reward": details.get("process_reward"),
                "num_files": len(files),
                f"has_{self.REQUIRED_ARTIFACT.replace('.', '_')}": (
                    self.REQUIRED_ARTIFACT in files
                ),
                "error": details.get("error"),
            }
            row.update(self._persist_extra(meta, details))
            with self._persist_lock:
                with open(stats_path, "a") as fh:
                    fh.write(json.dumps(row, default=str) + "\n")

            ep_dir = step_dir / f"ep_{k:02d}_paper{pid}"
            ep_dir.mkdir(parents=True, exist_ok=True)
            for fname, content in files.items():
                if fname.startswith("papers/"):
                    continue
                try:
                    fp = ep_dir / fname
                    fp.parent.mkdir(parents=True, exist_ok=True)
                    fp.write_text(content if isinstance(content, str) else str(content))
                except OSError:
                    continue

            # Full per-unit reward breakdown (gates, judgments, scores) + raw judge.
            try:
                (ep_dir / "reward_details.json").write_text(
                    json.dumps(
                        {k2: v for k2, v in details.items() if k2 != "judge_responses"},
                        ensure_ascii=False, indent=2, default=str,
                    )
                )
            except OSError:
                pass
            judge_responses = details.get("judge_responses")
            if judge_responses:
                try:
                    (ep_dir / "judge_responses.json").write_text(
                        json.dumps(judge_responses, ensure_ascii=False,
                                   indent=2, default=str)
                    )
                except OSError:
                    pass

            if conversation:
                try:
                    msgs_path = ep_dir / "messages.jsonl"
                    with open(msgs_path, "w") as fh:
                        for r in annotate_turns(conversation, self.ROLE):
                            fh.write(json.dumps(r, ensure_ascii=False,
                                                default=str) + "\n")
                except OSError:
                    pass
        except Exception:
            return

    def global_post_process_and_metrics(
        self, batch: BatchedDataDict
    ) -> tuple[BatchedDataDict, dict]:
        metrics = _metrics(batch)
        if self._reward_details:
            n = len(self._reward_details)
            agg: dict[str, float] = {}
            for d in self._reward_details:
                for k in self.AGG_KEYS:
                    v = d.get(k, 0.0)
                    if isinstance(v, (int, float)):
                        agg[k] = agg.get(k, 0.0) + v
            for k in agg:
                metrics[k] = agg[k] / n
            self._reward_details = []
        return batch, metrics
