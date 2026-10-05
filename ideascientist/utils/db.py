"""SQLite helpers for the Idea Vault database.

The vault is written by one process at a time and tables are only ever created,
never dropped or replaced: two concurrent writers on the same file silently lose
rows.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_VAULT_PATH = Path(
    os.environ.get("IDEA_VAULT_DB", "data/idea_vault.db")
).expanduser()


PAPER_FULL_TEXT_SCHEMA = """
CREATE TABLE IF NOT EXISTS paper_full_text (
    paper_id       INTEGER PRIMARY KEY,
    arxiv_id       TEXT,
    source_url     TEXT,
    content_source TEXT,
    full_text      TEXT,
    raw_tex        TEXT,
    bbl_content    TEXT,
    char_len       INTEGER,
    fetch_status   TEXT,
    fetch_error    TEXT,
    fetched_at     TEXT
);
CREATE INDEX IF NOT EXISTS idx_pft_status ON paper_full_text(fetch_status);
"""


def ensure_paper_full_text_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(PAPER_FULL_TEXT_SCHEMA)
    conn.commit()


def open_readonly(path: Path | str | None = None) -> sqlite3.Connection:
    """Open the vault for reading with zero locking.

    ``immutable=1`` rather than ``mode=ro``: it skips the WAL locking protocol
    entirely, which matters both on network filesystems that do not support
    POSIX advisory locks and under the many concurrent readers a GRPO rollout
    batch creates. The vault is static at read time, so this is safe.
    """
    p = Path(path) if path else DEFAULT_VAULT_PATH
    con = sqlite3.connect(f"file:{p}?mode=ro&immutable=1", uri=True)
    con.execute("PRAGMA busy_timeout=60000")
    return con


def assert_no_other_writer(script_basename: str) -> None:
    """Exit with status 2 if another python process is running the same script."""
    try:
        out = subprocess.run(
            ["ps", "-eo", "pid,cmd"], capture_output=True, text=True, check=True
        ).stdout
    except Exception:
        logger.warning("could not run ps to check for other writers; proceeding")
        return
    mine = {os.getpid(), os.getppid()}
    for line in out.splitlines():
        line = line.strip()
        if not line or line.startswith("PID"):
            continue
        pid_str, _, cmd = line.partition(" ")
        try:
            pid = int(pid_str)
        except ValueError:
            continue
        if pid in mine or not cmd.split():
            continue
        if not os.path.basename(cmd.split()[0]).startswith("python"):
            continue
        if script_basename in cmd:
            logger.error("refusing to start: another %s is alive (pid %d)", script_basename, pid)
            sys.exit(2)
