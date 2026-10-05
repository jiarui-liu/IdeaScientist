"""Verbose run logger for the harness — stores FULL content, never slices.

Per project convention (see the README / memory): prompts and responses are
written out in full; only the console preview is shortened. Never truncate
with ``[:N]`` at write time.

Records are streamed to a human-readable indented JSON array
(``harness_log.json``) via :class:`StreamingJsonArrayWriter`: each record is
written and flushed as it arrives (crash-safe like JSONL), and the array's
closing ``]`` is added on :meth:`RunLogger.close`, so a log from a crashed run
is missing only that bracket.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class StreamingJsonArrayWriter:
    """Append records to a growing, human-readable indented JSON array.

    Each record is written as an ``indent=2`` block the moment it arrives and
    flushed immediately, so a crash leaves every record so far on disk (same
    crash-safety as JSONL). The trade-off vs. JSONL: the array's closing ``]``
    is only written by :meth:`close`, so a killed process leaves the file
    without it. Readers must tolerate a missing trailing ``]`` — see
    :func:`load_json_array`.

    Not thread-safe on its own; callers that log concurrently must hold a lock
    (``RunLogger`` and the LLM clients already do).
    """

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(self.path, "w", encoding="utf-8")
        self._count = 0
        self._closed = False

    def write(self, rec: dict[str, Any]) -> None:
        block = json.dumps(rec, ensure_ascii=False, indent=2)
        block = "\n".join("  " + ln for ln in block.splitlines())  # nest under array
        if self._count == 0:
            self._f.write("[\n" + block)
        else:
            self._f.write(",\n" + block)
        self._count += 1
        self._f.flush()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._f.write("\n]\n" if self._count else "[]\n")
            self._f.flush()
            self._f.close()
        except Exception:
            pass




class SharedLogHandle:
    """Write handle for one subagent session into a shared per-role/paper log.

    Many concurrent subagents of the same role (or many readers of the same
    paper) append to ONE file. Each handle carries this session's
    ``conversation_id`` (globally unique, e.g. ``innovator#4`` or
    ``1744462#8``) and its ``session_seq`` (1-based order the conversation
    first appears in this file). :meth:`write` injects both keys and appends
    under the shared writer's lock so concurrent sessions don't interleave a
    half-written record.
    """

    def __init__(self, writer: StreamingJsonArrayWriter, lock: threading.Lock,
                 conversation_id: str, session_seq: int):
        self._writer = writer
        self._lock = lock
        self.conversation_id = conversation_id
        self.session_seq = session_seq

    def write(self, rec: dict[str, Any]) -> None:
        rec = {"conversation_id": self.conversation_id,
               "session_seq": self.session_seq, **rec}
        with self._lock:
            self._writer.write(rec)


class SharedLogRegistry:
    """Process-level registry sharing one writer + lock per log file path.

    Sub-agents call :meth:`acquire` with a target path (``_sublogs/innovator.json``,
    ``_sublogs/paper_1744462.json``) and a conversation id; every session
    targeting the same path appends to one :class:`StreamingJsonArrayWriter`
    behind one lock, so a role's whole history lands in a single readable file
    rather than one file per spawn. :meth:`close_all` finalizes them at
    teardown.
    """

    def __init__(self):
        self._writers: dict[Path, StreamingJsonArrayWriter] = {}
        self._locks: dict[Path, threading.Lock] = {}
        self._seqs: dict[Path, dict[str, int]] = {}
        self._guard = threading.Lock()

    def acquire(self, path: Path | str, conversation_id: str) -> SharedLogHandle:
        path = Path(path)
        with self._guard:
            writer = self._writers.get(path)
            if writer is None:
                writer = StreamingJsonArrayWriter(path)
                self._writers[path] = writer
                self._locks[path] = threading.Lock()
                self._seqs[path] = {}
            seqs = self._seqs[path]
            if conversation_id not in seqs:
                seqs[conversation_id] = len(seqs) + 1
            session_seq = seqs[conversation_id]
            lock = self._locks[path]
        return SharedLogHandle(writer, lock, conversation_id, session_seq)

    def close_all(self) -> None:
        with self._guard:
            writers = list(self._writers.values())
        for w in writers:
            w.close()


class RunLogger:
    def __init__(self, run_dir: str | Path):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.run_dir / "harness_log.json"
        self._writer = StreamingJsonArrayWriter(self.path)
        self._seq = 0
        self._lock = threading.Lock()  # subagents log concurrently

    def _ts(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _write(self, rec: dict[str, Any]) -> None:
        with self._lock:
            rec["seq"] = self._seq
            rec["ts"] = self._ts()
            self._seq += 1
            self._writer.write(rec)

    def event(self, kind: str, **data: Any) -> None:
        self._write({"type": "event", "kind": kind, **data})
        preview = " ".join(f"{k}={str(v)[:140]}" for k, v in data.items())
        print(f"  ▸ [{kind}] {preview}", flush=True)

    def llm_call(self, role: str, system: str, user: str, response: str, **meta: Any) -> None:
        # FULL content — no slicing at write time.
        self._write({
            "type": "llm_call", "role": role,
            "system": system, "user": user, "response": response,
            "system_chars": len(system), "user_chars": len(user),
            "response_chars": len(response), **meta,
        })
        print(f"  ▸ [llm:{role}] in {len(system)}+{len(user)} chars → out {len(response)} chars", flush=True)

    def sdk_message(self, message: Any) -> None:
        """Serialize a Claude Agent SDK streamed message in full."""
        self._write({"type": "sdk_message", **_serialize_sdk_message(message)})

    def close(self) -> None:
        self._writer.close()


def _serialize_sdk_message(message: Any) -> dict[str, Any]:
    out: dict[str, Any] = {"cls": type(message).__name__}
    content = getattr(message, "content", None)
    if content is None:
        # System/Result messages — dump whatever attrs are JSON-able. This is
        # where per-run token usage lives: `usage` (aggregate input/output/cache
        # tokens) and `model_usage` (per-model breakdown, incl. costUSD), plus
        # timing/turn counts and any error status. Capture them all so run stats
        # can be aggregated from harness_log.json without re-running.
        for attr in ("subtype", "result", "total_cost_usd", "duration_ms",
                     "duration_api_ms", "num_turns", "is_error", "data",
                     "usage", "model_usage", "stop_reason", "session_id",
                     "errors", "api_error_status"):
            v = getattr(message, attr, None)
            if v is not None:
                out[attr] = _jsonable(v)
        return out
    blocks = []
    for b in content if isinstance(content, list) else [content]:
        bt = type(b).__name__
        if hasattr(b, "text"):
            blocks.append({"block": bt, "text": b.text})
        elif hasattr(b, "name") and hasattr(b, "input"):  # ToolUseBlock
            blocks.append({"block": bt, "name": b.name, "input": _jsonable(b.input)})
        elif hasattr(b, "content"):  # ToolResultBlock
            blocks.append({"block": bt, "content": _jsonable(getattr(b, "content")),
                           "tool_use_id": getattr(b, "tool_use_id", None)})
        else:
            blocks.append({"block": bt, "repr": repr(b)})
    out["blocks"] = blocks
    return out


def _jsonable(v: Any) -> Any:
    try:
        json.dumps(v)
        return v
    except Exception:
        return repr(v)
