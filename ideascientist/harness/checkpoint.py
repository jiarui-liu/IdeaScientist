"""Durable call journal for the ideation runner.

Each agent replays its own numbered calls to reconstruct its Python state.
Completed calls return their recorded results; only unfinished calls execute.
Document changes and their results commit together before files are materialized.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import sys
import threading
from contextlib import ExitStack
from pathlib import Path


class CheckpointError(RuntimeError):
    """Recovery cannot safely continue. Never turn this into a tool observation."""


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def runtime_config():
    """Fingerprint executable code, behavior settings, and local corpus identity.

    Credentials are deliberately excluded so they can be rotated on resume.
    Corpus metadata is cheap to check even for a multi-gigabyte vault; it is not
    a content hash. Keep the corpus and served model weights unchanged.
    """
    from ideascientist.harness import corpus

    names = (
        "AGENT_LOOP_BASE_URL", "CONTEXT_WINDOW", "IDEA_VAULT_DB",
        "IDEASCIENTIST_BASE_URL", "IDEASCIENTIST_TEXT_TOOLS",
        "IDEASCIENTIST_MODEL",
        "IDEASCIENTIST_ROLE_BASE_URL", "IDEASCIENTIST_ROLE_MODEL",
        "IDEASCIENTIST_ROLE_TURN_CAP", "IDEASCIENTIST_ROLE_MAX_TOKENS",
        "IDEASCIENTIST_READER_INPUT_LIMIT", "IDEASCIENTIST_READER_TOKENIZER",
        "IDEASCIENTIST_READER_SINGLE_TURN", "IDEASCIENTIST_READER_NO_THINK",
        "IDEASCIENTIST_READER_DECODE", "IDEASCIENTIST_READER_MAX_TOKENS",
        "IDEASCIENTIST_STRIP_HISTORY_THINK", "IDEASCIENTIST_HISTORY_REASONING",
        "IDEASCIENTIST_EXTRA_BODY", "IDEASCIENTIST_CUTOFF_DATE",
        "IDEASCIENTIST_KEYWORD_INDEX", "IDEASCIENTIST_EMBEDDING_DIR",
        "IDEASCIENTIST_KW_MAX_TERMS", "IDEASCIENTIST_KW_MAX_QUERIES",
        "IDEASCIENTIST_KW_SEARCH_TIMEOUT", "IDEASCIENTIST_HTTP_TIMEOUT",
        "IDEASCIENTIST_MAX_RETRIES", "IDEASCIENTIST_SOCK_TIMEOUT",
        "IDEASCIENTIST_TOTAL_DEADLINE",
    )
    package = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(package.rglob("*")):
        if path.suffix in {".py", ".txt", ".yaml", ".json", ".jinja"}:
            digest.update(path.relative_to(package).as_posix().encode())
            digest.update(path.read_bytes())
    data = {}
    for path in (corpus._vault_path(), corpus._keyword_index_path()):
        path = path.resolve()
        stat = path.stat() if path.exists() else None
        data[str(path)] = [stat.st_size, stat.st_mtime_ns] if stat else None
    return {"code": digest.hexdigest(), "python": list(sys.version_info[:2]),
            "environment": {n: os.environ[n] for n in names if n in os.environ},
            "corpus": data, "cutoff": corpus.CUTOFF_DATE}


class Checkpoint:
    def __init__(self, run_dir, *, resume=False):
        self.run_dir = Path(run_dir).resolve()
        self.root = self.run_dir / "checkpoint"
        self.path = self.root / "state.sqlite3"
        self.resume = resume
        self._guard = threading.RLock()
        self._lock_file = None
        self.db = None
        self.resources = ExitStack()
        self.stopping = threading.Event()

    def __enter__(self):
        if self.resume and not self.path.is_file():
            raise CheckpointError("This run has no checkpoint; document-only runs cannot be resumed.")
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock_file = (self.root / "run.lock").open("a+b")
        self._lock_file.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self._lock_file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self._lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self._lock_file.close()
            raise CheckpointError("Another process is using this run directory.") from exc
        try:
            if not self.resume and self.path.exists():
                raise CheckpointError("Run already has a checkpoint. Use --resume or a new directory.")
            self.db = sqlite3.connect(self.path, check_same_thread=False)
            self.db.execute("PRAGMA synchronous=FULL")
            self.db.executescript("""
                CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS calls (
                    key TEXT PRIMARY KEY, request TEXT NOT NULL, result TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS documents (
                    path TEXT PRIMARY KEY, content TEXT NOT NULL, dirty INTEGER NOT NULL);
            """)
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *exc):
        try:
            self.resources.close()
        finally:
            if self.db is not None:
                self.db.close()
            if self._lock_file is not None:
                self._lock_file.close()

    def get(self, key):
        with self._guard:
            row = self.db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
            return json.loads(row[0]) if row else None

    def set(self, key, value):
        try:
            with self._guard, self.db:
                self.db.execute("INSERT OR REPLACE INTO meta VALUES (?, ?)", (key, _json(value)))
        except (sqlite3.Error, OSError) as exc:
            raise CheckpointError("Could not persist checkpoint metadata.") from exc

    def prepare(self, config):
        previous = self.get("config")
        if self.resume:
            if previous is None or previous != config:
                raise CheckpointError("Run configuration, code, or corpus changed; use the original settings to resume.")
        else:
            self.set("config", config)
        # Validate before touching artifacts or logs. Dirty files are precisely
        # the writes committed to SQLite but not yet acknowledged on disk.
        with self._guard:
            rows = self.db.execute("SELECT path, content, dirty FROM documents").fetchall()
            for rel, content, dirty in rows:
                path = self.document_path(rel)
                if not dirty and (not path.is_file() or path.read_bytes() != content.encode("utf-8")):
                    raise CheckpointError(f"Artifact changed outside the run: {rel}")
            self._materialize()

    def document_path(self, rel):
        root = (self.run_dir / "outputs").resolve()
        path = (root / rel).resolve()
        if root not in path.parents:
            raise CheckpointError(f"Document path escapes outputs: {rel}")
        return path

    def _materialize(self):
        for rel, content in self.db.execute(
                "SELECT path, content FROM documents WHERE dirty=1").fetchall():
            path = self.document_path(rel)
            path.parent.mkdir(parents=True, exist_ok=True)
            temp = path.with_name(path.name + ".checkpoint-tmp")
            with temp.open("w", encoding="utf-8", newline="") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp, path)
            with self.db:
                self.db.execute("UPDATE documents SET dirty=0 WHERE path=?", (rel,))

    def scope(self, name):
        return Scope(self, name)

    def reader_usage(self):
        rows = self.db.execute("SELECT value FROM meta WHERE key LIKE 'reader:%'").fetchall()
        stats = [json.loads(row[0]) for row in rows]
        return {k: sum(s.get(k, 0) for s in stats)
                for k in ("tokens", "calls", "n", "parser_retries")}


class Scope:
    def __init__(self, checkpoint, name):
        self.checkpoint = checkpoint
        self.name = name

    def child(self, name):
        return Scope(self.checkpoint, self.name + "/" + name)

    def call(self, key, request, fn, *, documents=False):
        """Execute once after a successful commit, with optional staged file writes.

        Nested calls and network requests run without the database mutex. Local
        document operations hold it through read/modify/commit/materialization.
        """
        cp = self.checkpoint
        key = self.name + "/" + key
        request = hashlib.sha256(_json(request).encode()).hexdigest()

        def execute():
            with cp._guard:
                row = cp.db.execute("SELECT request, result FROM calls WHERE key=?", (key,)).fetchone()
            if row:
                if row[0] != request:
                    raise CheckpointError(f"Replay inputs changed at {key}")
                return json.loads(row[1])
            if cp.stopping.is_set():
                raise CheckpointError("Run interrupted; resume to continue pending calls.")
            writes = {}
            result = fn(writes) if documents else fn()
            encoded = _json(result)
            with cp._guard, cp.db:
                cp.db.execute("INSERT INTO calls VALUES (?, ?, ?)", (key, request, encoded))
                for rel, content in writes.items():
                    cp.document_path(rel)
                    cp.db.execute("INSERT OR REPLACE INTO documents VALUES (?, ?, 1)", (rel, content))
            if documents:
                cp._materialize()
            return json.loads(encoded)

        try:
            if documents:
                with cp._guard:
                    return execute()
            return execute()
        except (sqlite3.Error, OSError) as exc:
            raise CheckpointError(f"Checkpoint operation failed at {key}") from exc

    def write(self, key, rel, content):
        def stage(writes):
            writes[rel] = content
        return self.call(key, [rel, content], stage, documents=True)

    def chat(self, key, client, messages, tools):
        # Keep all response fields (including opaque provider continuation data),
        # and restore accounting before the next real call.
        fields = ("_call_counter", "_total_input_tokens", "_total_output_tokens",
                  "_total_retries_on_parser_error", "last_prompt_tokens")

        def invoke():
            response = client.chat(messages, tools=tools)
            return {"response": response, "state": {k: getattr(client, k, 0) for k in fields}}

        record = self.call(key, [messages, tools], invoke)
        for name, value in record["state"].items():
            setattr(client, name, value)
        return record["response"]
