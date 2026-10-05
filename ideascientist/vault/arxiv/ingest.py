"""Write helpers for the Idea Vault SQLite database.

Four tables participate:

    papers                  bibliographic metadata, one row per paper
    paper_full_text         e-print LaTeX and bbl plus cleaned full text
    metadata_results_masked the results-masked record (see vault.schema)
    citations               edges from a citing paper to a cited paper
                            (written by vault.citations.expand)

Inserts are idempotent on ``arxiv_id``. Tables are only created, never dropped,
and a single writer is required: ``INSERT OR IGNORE`` does not set
``lastrowid``, so every insert gates on ``rowcount`` and falls back to a lookup.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from typing import Any, Mapping

from ideascientist.utils.db import (
    assert_no_other_writer,
    ensure_paper_full_text_schema as ensure_full_text_schema,
)
from ideascientist.vault.arxiv.crawl import ArxivPaper
from ideascientist.vault.citations.canonical import norm_arxiv

logger = logging.getLogger(__name__)

RESULTS_MASKED_TABLE = "metadata_results_masked"

PAPERS_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS papers (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    title    TEXT UNIQUE,
    authors  TEXT,
    date     TEXT,
    venue    TEXT,
    source   TEXT,
    url      TEXT,
    abstract TEXT,
    arxiv_id TEXT,
    doi      TEXT
);
CREATE INDEX IF NOT EXISTS idx_papers_arxiv ON papers(arxiv_id);
CREATE INDEX IF NOT EXISTS idx_papers_date  ON papers(date);

CREATE TABLE IF NOT EXISTS {RESULTS_MASKED_TABLE} (
    paper_id    INTEGER PRIMARY KEY,
    fields_json TEXT,
    model       TEXT,
    created_at  TEXT
);
"""

__all__ = [
    "RESULTS_MASKED_TABLE",
    "assert_no_other_writer",
    "ensure_full_text_schema",
    "ensure_vault_schema",
    "find_paper_id_by_arxiv",
    "has_full_text",
    "has_results_masked",
    "insert_full_text",
    "insert_paper",
    "insert_results_masked",
    "norm_arxiv",
    "verify_counts",
]


def ensure_vault_schema(conn: sqlite3.Connection) -> None:
    """Create every vault table if absent. Safe to call repeatedly."""
    from ideascientist.vault.citations.expand import ensure_schema as _ensure_citations

    conn.executescript(PAPERS_SCHEMA)
    _ensure_citations(conn)
    conn.commit()


def find_paper_id_by_arxiv(conn: sqlite3.Connection, arxiv_id: str) -> int | None:
    ax = norm_arxiv(arxiv_id)
    if not ax:
        return None
    row = conn.execute(
        "SELECT id FROM papers WHERE arxiv_id = ? LIMIT 1", (ax,)
    ).fetchone()
    return row[0] if row else None


def has_full_text(conn: sqlite3.Connection, paper_id: int) -> bool:
    row = conn.execute(
        "SELECT fetch_status FROM paper_full_text WHERE paper_id = ? LIMIT 1",
        (paper_id,),
    ).fetchone()
    return bool(row and row[0] == "ok")


def has_results_masked(conn: sqlite3.Connection, paper_id: int) -> bool:
    row = conn.execute(
        f"SELECT 1 FROM {RESULTS_MASKED_TABLE} WHERE paper_id = ? LIMIT 1",
        (paper_id,),
    ).fetchone()
    return row is not None


def insert_paper(
    conn: sqlite3.Connection, p: ArxivPaper, *, source: str = "arxiv_crawl"
) -> tuple[int | None, str]:
    """Insert a paper row; return ``(paper_id, status)``.

    ``status`` is ``new``, ``existing_by_arxiv``, ``existing_by_title`` (the
    same work reached us through a different identifier, so the new full text
    attaches to the existing row), or ``fail``.
    """
    ax = norm_arxiv(p.arxiv_id)
    title = (p.title or "").strip()
    if not title:
        return None, "fail"

    if ax:
        existing = find_paper_id_by_arxiv(conn, ax)
        if existing is not None:
            return existing, "existing_by_arxiv"

    cur = conn.execute(
        "INSERT OR IGNORE INTO papers "
        "(title, authors, date, venue, source, url, abstract, arxiv_id, doi) "
        "VALUES (?,?,?,?,?,?,?,?,?)",
        (
            title, p.authors or "", p.published_date, p.primary_category or "arXiv",
            source, p.abs_url, p.abstract or "", ax, "",
        ),
    )
    conn.commit()
    if cur.rowcount > 0:
        return cur.lastrowid, "new"
    row = conn.execute("SELECT id FROM papers WHERE title = ? LIMIT 1", (title,)).fetchone()
    return (row[0], "existing_by_title") if row else (None, "fail")


def insert_full_text(
    conn: sqlite3.Connection,
    paper_id: int,
    arxiv_id: str,
    *,
    content_source: str,
    full_text: str | None,
    raw_tex: str | None,
    bbl_content: str | None,
    fetch_status: str,
    fetch_error: str | None = None,
    source_url: str | None = None,
) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO paper_full_text "
        "(paper_id, arxiv_id, source_url, content_source, full_text, raw_tex, "
        " bbl_content, char_len, fetch_status, fetch_error, fetched_at) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (
            paper_id, norm_arxiv(arxiv_id), source_url, content_source,
            full_text, raw_tex, bbl_content, len(full_text) if full_text else 0,
            fetch_status, fetch_error,
            datetime.now(timezone.utc).isoformat(timespec="seconds"),
        ),
    )
    conn.commit()


def insert_results_masked(
    conn: sqlite3.Connection,
    paper_id: int,
    record: Mapping[str, Any],
    *,
    model: str = "",
) -> None:
    """Store one paper's results-masked record (see :mod:`vault.decompose`)."""
    conn.execute(
        f"INSERT OR REPLACE INTO {RESULTS_MASKED_TABLE} "
        "(paper_id, fields_json, model, created_at) VALUES (?,?,?,?)",
        (
            paper_id,
            json.dumps(record, ensure_ascii=False),
            model,
            datetime.now(timezone.utc).isoformat(timespec="seconds"),
        ),
    )
    conn.commit()


def verify_counts(db_path: str) -> dict[str, int]:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        return {
            "papers_total": conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0],
            "full_text_ok": conn.execute(
                "SELECT COUNT(*) FROM paper_full_text WHERE fetch_status = 'ok'"
            ).fetchone()[0],
            "results_masked_total": conn.execute(
                f"SELECT COUNT(*) FROM {RESULTS_MASKED_TABLE}"
            ).fetchone()[0],
            "citations_total": conn.execute("SELECT COUNT(*) FROM citations").fetchone()[0],
        }
    finally:
        conn.close()
