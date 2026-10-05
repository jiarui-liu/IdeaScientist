"""The retrieval and paper-access surface the harness exposes to its agents.

Everything an agent can learn about the literature comes through this module,
and every path through it is filtered by :data:`CUTOFF_DATE`. That filter is a
correctness constraint, not a tunable: the evaluation measures how well a
system anticipates directions that human researchers pursued after the cutoff,
so a single post-cutoff paper reaching an agent invalidates the run.

Two retrieval interfaces, over the same corpus:

``keyword_search`` is the one the agents actually use. The two retrieval
regimes in the paper are not two tools — they are the same BM25 interface used
with differently phrased queries, which is why the regime is a property of the
role prompt rather than of this module.

``embedding_search`` thresholds the field spaces from
:mod:`ideascientist.vault.fields`. Thresholding on ``challenge`` alone finds
work that fought the same difficulty in another setting; adding
``problem_definition`` narrows to same-problem prior art.
"""

from __future__ import annotations

import logging
import os
import re
import sqlite3
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

from ideascientist.utils.db import DEFAULT_VAULT_PATH, open_readonly
from ideascientist.vault.fields import field_texts

logger = logging.getLogger(__name__)

# Retrieved papers must never be at or after this date.
CUTOFF_DATE = os.environ.get("IDEASCIENTIST_CUTOFF_DATE", "2026-01-01")

RESULTS_MASKED_TABLE = "metadata_results_masked"
DEFAULT_SEARCH_SPACES = ("challenge", "problem_definition")


def _vault_path() -> Path:
    return Path(os.environ.get("IDEA_VAULT_DB", str(DEFAULT_VAULT_PATH)))


def _keyword_index_path() -> Path:
    return Path(os.environ.get("IDEASCIENTIST_KEYWORD_INDEX", "data/keyword_index.db"))


def _embedding_dir() -> Path:
    return Path(os.environ.get("IDEASCIENTIST_EMBEDDING_DIR", "data/paper_embeddings"))


# --------------------------------------------------------------- keyword search

_KW_VALIDATED = False


def _ensure_keyword_index() -> None:
    """Validate the prebuilt FTS5 index; never build it in the request path.

    Building here would stall every caller on a cache miss, including each GRPO
    rollout worker. Build it with ``ideascientist.vault.index`` instead.
    """
    global _KW_VALIDATED
    if _KW_VALIDATED:
        return
    path = _keyword_index_path()
    if path.exists():
        try:
            con = sqlite3.connect(f"file:{path}?mode=ro&immutable=1", uri=True)
            con.execute("PRAGMA busy_timeout=60000")
            tables = {
                r[0]
                for r in con.execute(
                    "SELECT name FROM sqlite_master WHERE type IN ('table','view')"
                )
            }
            ok = (
                "papers_fts" in tables
                and "papers_meta" in tables
                and con.execute("SELECT rowid FROM papers_fts LIMIT 1").fetchone()
                and con.execute("SELECT paper_id FROM papers_meta LIMIT 1").fetchone()
            )
            con.close()
            if ok:
                _KW_VALIDATED = True
                return
        except sqlite3.Error as exc:
            logger.warning("keyword index validation failed for %s: %s", path, exc)
    raise RuntimeError(
        f"keyword index not available at {path}. Build it with "
        f"`python -m ideascientist.vault.index keyword`."
    )


def _fts_query(q: str) -> str:
    """Turn a natural-language query into a bounded FTS5 OR-of-terms expression.

    The term count is capped because ``MATCH ... ORDER BY bm25`` must score the
    union of every term's postings before LIMIT, so a verbose model query makes
    the match set approach the whole corpus.
    """
    words = [w for w in re.findall(r"[A-Za-z0-9]+", q) if len(w) >= 3]
    try:
        cap = int(os.environ.get("IDEASCIENTIST_KW_MAX_TERMS", "48") or 48)
    except ValueError:
        cap = 48
    return " OR ".join(f'"{w}"' for w in (words[:cap] if cap > 0 else words))


def keyword_search(
    queries: list[str], k: int = 40, cutoff_date: str = CUTOFF_DATE
) -> list[dict[str, Any]]:
    """BM25 over the corpus for a list of model-written queries.

    Returns papers ranked by their best BM25 across queries. A single scan has
    no built-in timeout, so a wall-clock guard interrupts the connection and
    returns the best results gathered so far rather than wedging the caller.
    """
    _ensure_keyword_index()
    con = sqlite3.connect(f"file:{_keyword_index_path()}?mode=ro&immutable=1", uri=True)
    con.execute("PRAGMA busy_timeout=60000")

    try:
        timeout_s = float(os.environ.get("IDEASCIENTIST_KW_SEARCH_TIMEOUT", "300") or 300)
    except ValueError:
        timeout_s = 300.0
    timed_out = {"v": False}

    def _interrupt() -> None:
        timed_out["v"] = True
        try:
            con.interrupt()
        except Exception:  # noqa: BLE001 - best effort
            pass

    timer = threading.Timer(timeout_s, _interrupt) if timeout_s > 0 else None
    if timer is not None:
        timer.daemon = True
        timer.start()

    try:
        max_q = int(os.environ.get("IDEASCIENTIST_KW_MAX_QUERIES", "12") or 12)
    except ValueError:
        max_q = 12
    q_list = queries[:max_q] if max_q > 0 else list(queries)

    # Compare on the 4-char year prefix: a year-only date ("2026") sorts before
    # the full cutoff ("2026-01-01"), so a raw string compare would leak it.
    cutoff_year = str(cutoff_date)[:4]
    best: dict[int, dict[str, Any]] = {}
    try:
        for q in q_list:
            if timed_out["v"]:
                break
            expr = _fts_query(q or "")
            if not expr:
                continue
            try:
                cur = con.execute(
                    "SELECT f.rowid AS paper_id, m.title, bm25(papers_fts) AS score "
                    "FROM papers_fts f "
                    "JOIN papers_meta m ON m.paper_id = f.rowid "
                    "WHERE papers_fts MATCH ? "
                    "AND m.collect_date != '' AND substr(m.collect_date, 1, 4) < ? "
                    "ORDER BY score LIMIT ?",
                    (expr, cutoff_year, k),
                )
                for pid, title, score in cur.fetchall():
                    pid = int(pid)
                    if pid not in best or score < best[pid]["bm25"]:
                        best[pid] = {"id": pid, "title": title, "bm25": round(float(score), 3)}
            except sqlite3.Error as exc:
                if timed_out["v"]:
                    logger.warning(
                        "keyword_search timed out after %.0fs; returning %d partial results",
                        timeout_s, len(best),
                    )
                    break
                logger.warning("keyword_search failed for query=%r: %s", q, exc)
                continue
    finally:
        if timer is not None:
            timer.cancel()
        con.close()
    return sorted(best.values(), key=lambda x: x["bm25"])[:k]


# ------------------------------------------------------------ embedding search

@lru_cache(maxsize=1)
def _field_matrices():
    import json

    import numpy as np

    emb_dir = _embedding_dir()
    meta = json.loads((emb_dir / "metadata.json").read_text())
    pid = np.array([int(m["paper_id"]) for m in meta], dtype=np.int64)

    def _norm(name: str):
        m = np.asarray(np.load(emb_dir / name), dtype=np.float32)
        n = np.linalg.norm(m, axis=1, keepdims=True)
        return (m / np.where(n == 0, 1.0, n)).astype(np.float32), (n[:, 0] > 1e-3)

    challenge, challenge_valid = _norm("challenge.npy")
    problem_definition, pd_valid = _norm("problem_definition.npy")
    return {
        "pid": pid,
        "titles": [m.get("title", "") for m in meta],
        "dates": [m.get("collect_date", "") or "" for m in meta],
        "row_of_pid": {int(p): i for i, p in enumerate(pid)},
        "challenge": challenge,
        "challenge_valid": challenge_valid,
        "problem_definition": problem_definition,
        "problem_definition_valid": pd_valid,
    }


def _query_vec(space: str, query_paper_id: Optional[int], query_text: Optional[str]):
    import numpy as np

    g = _field_matrices()
    if query_paper_id is not None and int(query_paper_id) in g["row_of_pid"]:
        return g[space][g["row_of_pid"][int(query_paper_id)]]
    if not query_text:
        raise ValueError("provide query_paper_id (present in the field spaces) or query_text")
    from ideascientist.harness.embedding_spaces import encode_query

    v = np.asarray(encode_query(space, query_text), dtype=np.float32)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def embedding_search(
    *,
    query_paper_id: Optional[int] = None,
    query_text: Optional[str] = None,
    challenge_threshold: float = 0.75,
    problem_definition_threshold: Optional[float] = None,
    k: int = 50,
    cutoff_date: str = CUTOFF_DATE,
    exclude_self: bool = True,
) -> list[dict[str, Any]]:
    """Threshold retrieval over the challenge and problem-definition spaces.

    Thresholding ``challenge`` alone selects same-challenge work in any setting,
    which is where a transferable mechanism comes from. Adding the
    problem-definition threshold narrows to same-problem prior art, which is
    what a gap claim has to be defended against.
    """
    import numpy as np

    g = _field_matrices()
    ch_scores = g["challenge"] @ _query_vec("challenge", query_paper_id, query_text)
    mask = (ch_scores >= challenge_threshold) & g["challenge_valid"]

    pd_scores = None
    if problem_definition_threshold is not None:
        pd_scores = g["problem_definition"] @ _query_vec(
            "problem_definition", query_paper_id, query_text
        )
        mask = mask & (pd_scores >= problem_definition_threshold) & g["problem_definition_valid"]

    rows = np.nonzero(mask)[0]
    rows = rows[np.argsort(-ch_scores[rows])]
    out: list[dict[str, Any]] = []
    for r in rows:
        pid = int(g["pid"][r])
        if exclude_self and query_paper_id is not None and pid == int(query_paper_id):
            continue
        cdate = g["dates"][r]
        if cutoff_date and (not cdate or str(cdate)[:4] >= str(cutoff_date)[:4]):
            continue
        rec = {"id": pid, "title": g["titles"][r], "challenge_score": round(float(ch_scores[r]), 4)}
        if pd_scores is not None:
            rec["problem_definition_score"] = round(float(pd_scores[r]), 4)
        out.append(rec)
        if len(out) >= k:
            break
    return out


def embedding_score(
    *,
    challenge_text: str = "",
    problem_definition_text: str = "",
    query_paper_id: Optional[int] = None,
    query_text: Optional[str] = None,
) -> dict[str, float]:
    """Cosine of a candidate paper's field text against the target problem."""
    import numpy as np

    from ideascientist.harness.embedding_spaces import encode_query

    out: dict[str, float] = {}
    for space, text in (
        ("challenge", challenge_text),
        ("problem_definition", problem_definition_text),
    ):
        if not text:
            continue
        v = np.asarray(encode_query(space, text), dtype=np.float32)
        v = v / (np.linalg.norm(v) + 1e-9)
        out[f"{space}_score"] = round(
            float(v @ _query_vec(space, query_paper_id, query_text)), 4
        )
    return out


# --------------------------------------------------------------- paper access

def get_paper_biblio(paper_id: int) -> Optional[dict[str, Any]]:
    """Title, abstract, and date only.

    Deliberately excludes the results-masked record: this is the untrusted
    surface a first-tier agent browses. Anything requiring the paper's content
    must go through ``read_paper``, which runs a paper reader over the body.
    """
    con = open_readonly(_vault_path())
    try:
        row = con.execute(
            "SELECT title, abstract, date FROM papers WHERE id = ? LIMIT 1",
            (int(paper_id),),
        ).fetchone()
    finally:
        con.close()
    if not row:
        return None
    title, abstract, date = row
    year = str(date)[:4] if date and str(date)[:4].isdigit() else ""
    return {
        "id": int(paper_id),
        "title": title or "",
        "abstract": abstract or "",
        "date": date or "",
        "year": year,
    }


def get_paper_record(paper_id: int) -> Optional[dict[str, Any]]:
    """The stored results-masked record for one paper, or None."""
    import json

    con = open_readonly(_vault_path())
    try:
        row = con.execute(
            f"SELECT fields_json FROM {RESULTS_MASKED_TABLE} WHERE paper_id = ? LIMIT 1",
            (int(paper_id),),
        ).fetchone()
    except sqlite3.Error:
        return None
    finally:
        con.close()
    if not row or not row[0]:
        return None
    try:
        obj = json.loads(row[0])
    except (ValueError, TypeError):
        return None
    return obj if isinstance(obj, dict) else None


def get_paper_fields(paper_id: int) -> Optional[dict[str, Any]]:
    """Bibliographic columns plus the four field texts for one paper."""
    import json

    con = open_readonly(_vault_path())
    try:
        row = con.execute(
            "SELECT p.title, p.abstract, p.authors, p.date, p.venue, p.url, "
            "p.arxiv_id, p.doi, b.fields_json "
            f"FROM papers p LEFT JOIN {RESULTS_MASKED_TABLE} b ON b.paper_id = p.id "
            "WHERE p.id = ? LIMIT 1",
            (int(paper_id),),
        ).fetchone()
    finally:
        con.close()
    if not row:
        return None
    title, abstract, authors, date, venue, url, arxiv_id, doi, fields_json = row
    try:
        record = json.loads(fields_json) if fields_json else {}
    except (ValueError, TypeError):
        record = {}
    texts = field_texts(record if isinstance(record, dict) else {})
    year = str(date)[:4] if date and str(date)[:4].isdigit() else ""
    return {
        "id": int(paper_id),
        "title": title or "",
        "authors": authors or "",
        "date": date or "",
        "year": year,
        "venue": venue or "",
        "url": url or "",
        "arxiv_id": arxiv_id or "",
        "doi": doi or "",
        "abstract": abstract or "",
        **texts,
    }


def get_stored_full_text(paper_id: int) -> str:
    """The paper's stored body, or '' when we do not have it."""
    con = open_readonly(_vault_path())
    try:
        row = con.execute(
            "SELECT full_text FROM paper_full_text WHERE paper_id = ? "
            "AND full_text IS NOT NULL AND full_text != '' LIMIT 1",
            (int(paper_id),),
        ).fetchone()
    except sqlite3.Error as exc:
        logger.warning("full-text lookup failed for paper_id=%s: %s", paper_id, exc)
        row = None
    finally:
        con.close()
    return row[0] if row and row[0] else ""
