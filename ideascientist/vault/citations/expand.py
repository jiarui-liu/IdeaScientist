"""Citation-graph expansion of the Idea Vault.

Iterate the citing papers (those with bbl_content + recent date), parse
their bibs, dedup the references against everything already in our DB,
and pull the new ones in — arxiv when resolvable, metadata-only when not.

Reuses :mod:`eval.citations.parse`, :mod:`eval.citations.canonical`, and
:func:`eval.citations.extract.fetch_latex_source_with_bbl`. Reuses our
own :mod:`ideascientist.vault.text._clean_latex`
to derive ``full_text`` from ``raw_tex``.
"""

from __future__ import annotations

import logging
import re
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

from ideascientist.vault.citations.canonical import norm_arxiv, norm_doi
from ideascientist.vault.citations.extract import fetch_latex_source_with_bbl
from ideascientist.vault.citations.parse import parse_bbl_entries, parse_bib_file

from ideascientist.vault.text import _clean_latex
from ideascientist.utils.db import ensure_paper_full_text_schema

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# arxiv-id extraction (the bib parser leaves arxiv_id='' when the id is
# only embedded in the venue/note string — we recover it here)
# ---------------------------------------------------------------------------

# Modern arxiv IDs: ``YYMM.NNNNN`` (4-5 digit numeric suffix). Bare digit
# pattern catches every variant we've seen — ``eprint = {2305.13048}``,
# ``journal = {arXiv preprint arXiv:2506.10943}``, ``url = {https://arxiv.org/abs/2305.13048}``,
# ``note = {Available at arXiv:2305.13048v2}``, ``doi = {10.48550/arxiv.2305.13048}``.
# Use word boundaries so it doesn't grab digits inside dates/page ranges.
_MODERN_ARXIV_RE = re.compile(r"\b(\d{4}\.\d{4,5})(?:v\d+)?\b")

# Old-style arxiv IDs (pre-April 2007): ``cs/0301012`` / ``hep-th/9711200`` / etc.
# Anchor on a known archive prefix to keep recall narrow and precision high.
_OLD_ARXIV_PREFIXES = (
    "cs", "math", "stat", "physics", "astro-ph", "cond-mat", "gr-qc",
    "hep-ex", "hep-lat", "hep-ph", "hep-th", "nucl-ex", "nucl-th",
    "nlin", "quant-ph", "q-bio", "q-fin", "eess",
)
_OLD_ARXIV_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(p) for p in _OLD_ARXIV_PREFIXES) + r")/\d{7}\b",
    re.IGNORECASE,
)

# URL ending in ``.pdf`` — anywhere in the raw bib entry. Allow query
# strings (``…paper.pdf?download=true``) and fragments
# (``…paper.pdf#page=3``). Stop on whitespace/braces/quotes/angle
# brackets so we don't swallow trailing punctuation.
_PDF_URL_RE = re.compile(
    r"https?://[^\s{}\"<>]+?\.pdf(?:[?#][^\s{}\"<>]*)?",
    re.IGNORECASE,
)




def extract_arxiv_id_from_bib_entry(entry) -> str:
    """Pull a canonical arxiv id (``YYMM.NNNNN`` or old-style) from a
    parsed bib entry.

    Strategy — checked in order:

    1. The parser's own ``arxiv_id`` field (filled when the bib has an
       explicit ``eprint = {...}``).
    2. Regex-scan ``entry.raw_entry_text`` — the verbatim bib block — for
       any ``YYMM.NNNNN`` or old-style ``cs/0301012``. This is the
       robust fallback for arxiv ids hidden in ``url`` / ``note`` /
       ``howpublished`` / ``journal`` / etc. fields that the structured
       parser dropped on the floor.
    3. As a last resort, scan the individual ``doi`` / ``venue`` /
       ``title`` fields (kept for parity with older callers that may
       have empty ``raw_entry_text``).

    Returns ``""`` if nothing matches.
    """
    if entry.arxiv_id:
        n = norm_arxiv(entry.arxiv_id)
        if n:
            return n

    raw = getattr(entry, "raw_entry_text", "") or ""
    if raw:
        m = _MODERN_ARXIV_RE.search(raw)
        if m:
            return m.group(1)
        m = _OLD_ARXIV_RE.search(raw)
        if m:
            return m.group(0).lower()

    for fld in (entry.doi, entry.venue, entry.title):
        if not fld:
            continue
        m = _MODERN_ARXIV_RE.search(fld)
        if m:
            return m.group(1)
        m = _OLD_ARXIV_RE.search(fld)
        if m:
            return m.group(0).lower()
    return ""


# ---------------------------------------------------------------------------
# Title normalization for fuzzy dedup
# ---------------------------------------------------------------------------

_NONALNUM = re.compile(r"[^a-z0-9]+")


def norm_title(t: str | None) -> str:
    if not t:
        return ""
    return _NONALNUM.sub("", t.strip().lower())


# ---------------------------------------------------------------------------
# Stats reporting
# ---------------------------------------------------------------------------

# A handful of papers have a corrupt / pathological ``bbl_content`` — tens of MB
# parsing to tens of thousands of bogus "references". Processing those injects
# huge amounts of junk ``metadata_only`` papers and stalls the run for many
# minutes each. Real bibliographies are comfortably under these bounds, so we
# skip anything beyond them.
MAX_BBL_BYTES = 5_000_000        # ~5 MB; legit bbls are << this
MAX_REFS_PER_PAPER = 5_000       # legit surveys top out in the low thousands


@dataclass
class ExpandStats:
    citing_papers_total: int = 0
    citing_papers_skipped_oversized: int = 0
    citing_papers_with_bbl: int = 0
    refs_total: int = 0
    refs_unique: int = 0
    refs_already_in_db: int = 0
    refs_new_arxiv_attempted: int = 0
    refs_new_arxiv_inserted_with_fulltext: int = 0
    refs_new_arxiv_no_source: int = 0
    refs_new_arxiv_fetch_error: int = 0
    refs_new_metadata_only_inserted: int = 0
    refs_unresolvable_skipped: int = 0
    citation_rows_inserted: int = 0

    def __str__(self) -> str:
        return "\n".join(f"  {k}: {v}" for k, v in vars(self).items())


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

class CitationExpander:
    """Expand the DB by ingesting every citation of every recent paper.

    Resumable / idempotent: the existence checks against ``papers``
    (by arxiv_id, by doi, by normalised title) and against the
    ``citations`` table both prevent duplicate work on re-runs.
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
        *,
        start_date: str = "2026-04-01",
        rate_limit_sec: float = 3.0,
        max_fetch: int | None = None,
        cited_source_tag: str = "cited_paper",
    ) -> None:
        self.conn = conn
        self.start_date = start_date
        self.rate_limit_sec = rate_limit_sec
        self.max_fetch = max_fetch
        self.cited_source_tag = cited_source_tag
        self.stats = ExpandStats()

        # In-memory dedup indexes (built once at start).
        self._arxiv_to_id: dict[str, int] = {}
        self._doi_to_id: dict[str, int] = {}
        self._title_to_id: dict[str, int] = {}

    # ----------------------- index existing papers ---------------------------
    def build_existing_index(self) -> None:
        logger.info("building in-memory dedup index over papers table ...")
        cur = self.conn.execute(
            "SELECT id, arxiv_id, doi, title FROM papers"
        )
        n = 0
        for pid, ax, doi, title in cur:
            n += 1
            ax_n = norm_arxiv(ax) if ax else ""
            if ax_n:
                self._arxiv_to_id.setdefault(ax_n, pid)
            doi_n = norm_doi(doi) if doi else ""
            if doi_n:
                self._doi_to_id.setdefault(doi_n, pid)
            t_n = norm_title(title)
            if t_n:
                self._title_to_id.setdefault(t_n, pid)
        logger.info(
            "indexed %d papers: %d by arxiv, %d by doi, %d by title",
            n, len(self._arxiv_to_id), len(self._doi_to_id),
            len(self._title_to_id),
        )

    def lookup_existing(
        self, arxiv_id: str, doi: str, title: str,
    ) -> int | None:
        if arxiv_id:
            ax_n = norm_arxiv(arxiv_id)
            if ax_n and ax_n in self._arxiv_to_id:
                return self._arxiv_to_id[ax_n]
        if doi:
            doi_n = norm_doi(doi)
            if doi_n and doi_n in self._doi_to_id:
                return self._doi_to_id[doi_n]
        t_n = norm_title(title)
        if t_n and t_n in self._title_to_id:
            return self._title_to_id[t_n]
        return None

    # ----------------------- citing-paper iteration --------------------------
    def iter_citing_papers(self) -> Iterable[tuple[int, str, str]]:
        """Yield ``(paper_id, title, bbl_content)`` for each citing paper.

        The candidate ids are materialised up-front and each paper's bbl is then
        fetched with a short, self-closing query. We deliberately do NOT stream a
        single long-lived SELECT cursor across the insert loop: an open read
        cursor pins the WAL and prevents checkpointing, so on large batches the
        WAL grows without bound and throughput degrades badly. (We filter on
        ``bbl_content IS NOT NULL`` only — a cheap header check — and let the
        per-paper ``parse_refs``/``if not refs`` path drop empty bbls, instead of
        ``LENGTH(...) > 0`` which would force reading every blob up-front.)
        """
        ids = [
            r[0]
            for r in self.conn.execute(
                """
                SELECT p.id
                FROM papers p
                JOIN paper_full_text ft ON ft.paper_id = p.id
                WHERE p.date >= ?
                  AND ft.bbl_content IS NOT NULL
                ORDER BY p.id
                """,
                (self.start_date,),
            ).fetchall()
        ]
        for pid in ids:
            row = self.conn.execute(
                "SELECT p.title, ft.bbl_content "
                "FROM papers p JOIN paper_full_text ft ON ft.paper_id = p.id "
                "WHERE p.id = ?",
                (pid,),
            ).fetchone()
            if row is None or not row[1]:
                continue
            yield pid, row[0], row[1]

    # ----------------------- bib parsing -------------------------------------
    @staticmethod
    def parse_refs(bbl: str) -> list:
        """Try the BibTeX parser first (``parse_bib_file``); if it yields no
        usable refs (e.g. resolved ``.bbl`` files), fall back to
        ``parse_bbl_entries`` which handles ``\\bibitem{...}`` form."""
        try:
            refs = parse_bib_file(bbl)
        except Exception:
            refs = []
        if not refs:
            try:
                refs = parse_bbl_entries(bbl)
            except Exception:
                refs = []
        return refs

    # ----------------------- arxiv fetch + insert ----------------------------
    def _insert_papers_row(
        self, *, title: str, authors: str, year: str, venue: str,
        arxiv_id: str, doi: str, url: str,
    ) -> int | None:
        """Insert a new row in ``papers`` (idempotent on title); return id.

        Mirrors the dedup semantics of ``ingest.insert_paper`` but does
        NOT generate an arxiv URL when missing — non-arxiv citations
        keep their original ``url`` (often empty)."""
        title = (title or "").strip()
        if not title:
            return None
        ax = norm_arxiv(arxiv_id) if arxiv_id else ""
        date_str = (year or "")[:4] if (year and year[:4].isdigit()) else ""
        cur = self.conn.execute(
            "INSERT OR IGNORE INTO papers "
            "(title, authors, date, venue, source, url, abstract, arxiv_id, doi) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (title, authors or "", date_str, venue or "",
             self.cited_source_tag, url or "", "", ax, norm_doi(doi)),
        )
        self.conn.commit()
        if cur.rowcount > 0:
            paper_id = cur.lastrowid
        else:
            row = self.conn.execute(
                "SELECT id FROM papers WHERE title = ? LIMIT 1", (title,),
            ).fetchone()
            if not row:
                return None
            paper_id = row[0]
        # Update indexes so subsequent refs in the same run see this row.
        if ax:
            self._arxiv_to_id.setdefault(ax, paper_id)
        dn = norm_doi(doi)
        if dn:
            self._doi_to_id.setdefault(dn, paper_id)
        self._title_to_id.setdefault(norm_title(title), paper_id)
        return paper_id

    def _insert_full_text(
        self, paper_id: int, arxiv_id: str,
        raw_tex: str | None, bbl: str | None, full_text: str | None,
        status: str, error: str | None = None,
    ) -> None:
        char_len = len(full_text) if full_text else 0
        self.conn.execute(
            "INSERT OR REPLACE INTO paper_full_text "
            "(paper_id, arxiv_id, source_url, content_source, "
            " full_text, raw_tex, bbl_content, char_len, fetch_status, "
            " fetch_error, fetched_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (paper_id, norm_arxiv(arxiv_id),
             f"https://arxiv.org/e-print/{arxiv_id}" if arxiv_id else None,
             "latex" if status == "ok" else "none",
             full_text, raw_tex, bbl, char_len, status, error,
             datetime.utcnow().isoformat(timespec="seconds")),
        )
        self.conn.commit()

    def _fetch_arxiv_and_store(self, paper_id: int, arxiv_id: str) -> str:
        try:
            raw_tex, bbl = fetch_latex_source_with_bbl(arxiv_id)
        except Exception as e:
            self._insert_full_text(
                paper_id, arxiv_id, None, None, None, "error", str(e),
            )
            return "error"
        if not raw_tex:
            self._insert_full_text(
                paper_id, arxiv_id, None, None, bbl, "no_source",
            )
            return "no_source"
        try:
            cleaned = _clean_latex(raw_tex)
        except Exception as e:
            self._insert_full_text(
                paper_id, arxiv_id, raw_tex, bbl, None, "error",
                f"clean_latex: {e}",
            )
            return "error"
        self._insert_full_text(
            paper_id, arxiv_id, raw_tex, bbl, cleaned or None,
            "ok" if (cleaned and len(cleaned) > 500) else "no_source",
        )
        return "ok" if (cleaned and len(cleaned) > 500) else "no_source"

    # ----------------------- citation-edge insert ----------------------------
    def _insert_citation_edge(
        self,
        source_paper_id: int,
        cited_paper_id: int | None,
        cited_title: str,
        verified_status: str,
    ) -> None:
        """Append a ``citations`` row, deduped on source/cited/title.

        ``cited_paper_id`` is NULL for a reference we could not resolve to a
        paper in the vault.
        """
        row = self.conn.execute(
            "SELECT 1 FROM citations WHERE source_paper_id = ? "
            "AND ((cited_paper_id IS NOT NULL AND cited_paper_id = ?) "
            "     OR (cited_paper_id IS NULL AND cited_title = ?))"
            " LIMIT 1",
            (source_paper_id, cited_paper_id, cited_title),
        ).fetchone()
        if row:
            return
        self.conn.execute(
            "INSERT INTO citations "
            "(source_paper_id, cited_paper_id, direction, cited_title, "
            " verified_status, fetched_at) "
            "VALUES (?,?,?,?,?,?)",
            (source_paper_id, cited_paper_id, "out", cited_title,
             verified_status,
             datetime.utcnow().isoformat(timespec="seconds")),
        )
        self.conn.commit()
        self.stats.citation_rows_inserted += 1

    # ----------------------- main loop ---------------------------------------
    def run(self) -> ExpandStats:
        logger.info("=== citation-expand starting (start_date=%s) ===",
                    self.start_date)
        self.build_existing_index()

        # First pass: count citing papers and accumulate per-paper refs
        # in a generator-friendly way (we walk once, parsing per paper).
        fetch_done = 0
        last_fetch_ts = 0.0

        for citing_id, citing_title, bbl in self.iter_citing_papers():
            self.stats.citing_papers_total += 1
            if self.stats.citing_papers_total % 100 == 0:
                logger.info(
                    "[%d papers walked] refs=%d new_arxiv=%d new_meta=%d "
                    "fetched=%d edges=%d",
                    self.stats.citing_papers_total, self.stats.refs_total,
                    self.stats.refs_new_arxiv_inserted_with_fulltext,
                    self.stats.refs_new_metadata_only_inserted,
                    fetch_done, self.stats.citation_rows_inserted,
                )
            # Skip corrupt/oversized bibliographies before the expensive parse
            # + per-ref inserts (see MAX_BBL_BYTES / MAX_REFS_PER_PAPER).
            if bbl is not None and len(bbl) > MAX_BBL_BYTES:
                logger.warning(
                    "skipping paper id=%s: oversized bbl_content (%d bytes)",
                    citing_id, len(bbl),
                )
                self.stats.citing_papers_skipped_oversized += 1
                continue
            refs = self.parse_refs(bbl)
            if not refs:
                continue
            if len(refs) > MAX_REFS_PER_PAPER:
                logger.warning(
                    "skipping paper id=%s: %d parsed refs exceeds cap %d "
                    "(corrupt bibliography)",
                    citing_id, len(refs), MAX_REFS_PER_PAPER,
                )
                self.stats.citing_papers_skipped_oversized += 1
                continue
            self.stats.citing_papers_with_bbl += 1

            for ref in refs:
                self.stats.refs_total += 1

                ax = extract_arxiv_id_from_bib_entry(ref)
                doi = (ref.doi or "").strip()
                title = (ref.title or "").strip()

                # 1) Lookup existing paper
                existing = self.lookup_existing(ax, doi, title)
                if existing:
                    self.stats.refs_already_in_db += 1
                    self._insert_citation_edge(
                        citing_id, existing, title, "resolved_existing",
                    )
                    continue

                # 2) Decide arxiv-resolvable vs metadata-only
                if ax:
                    self.stats.refs_new_arxiv_attempted += 1
                    new_paper_id = self._insert_papers_row(
                        title=title, authors=ref.authors or "",
                        year=ref.year or "", venue=ref.venue or "",
                        arxiv_id=ax, doi=doi,
                        url=f"https://arxiv.org/abs/{ax}",
                    )
                    if new_paper_id is None:
                        self.stats.refs_unresolvable_skipped += 1
                        continue

                    # Rate-limit before each arxiv.org hit
                    if self.max_fetch is not None and fetch_done >= self.max_fetch:
                        logger.info(
                            "max_fetch=%d reached; inserting %s row "
                            "without downloading e-print",
                            self.max_fetch, ax,
                        )
                    else:
                        elapsed = time.time() - last_fetch_ts
                        if elapsed < self.rate_limit_sec:
                            time.sleep(self.rate_limit_sec - elapsed)
                        last_fetch_ts = time.time()

                        status = self._fetch_arxiv_and_store(new_paper_id, ax)
                        fetch_done += 1
                        if status == "ok":
                            self.stats.refs_new_arxiv_inserted_with_fulltext += 1
                        elif status == "no_source":
                            self.stats.refs_new_arxiv_no_source += 1
                        else:
                            self.stats.refs_new_arxiv_fetch_error += 1

                    self._insert_citation_edge(
                        citing_id, new_paper_id, title, "resolved_arxiv_new",
                    )
                    continue

                # 3) Non-arxiv: metadata-only insert
                if not title:
                    self.stats.refs_unresolvable_skipped += 1
                    continue
                new_paper_id = self._insert_papers_row(
                    title=title, authors=ref.authors or "",
                    year=ref.year or "", venue=ref.venue or "",
                    arxiv_id="", doi=doi, url="",
                )
                if new_paper_id is None:
                    self.stats.refs_unresolvable_skipped += 1
                    continue
                self.stats.refs_new_metadata_only_inserted += 1
                self._insert_citation_edge(
                    citing_id, new_paper_id, title, "metadata_only",
                )

        # Unique refs is approximated as inserted_arxiv + inserted_meta
        # + already_in_db (one row per unique cited paper *visit*).
        self.stats.refs_unique = (
            self.stats.refs_new_arxiv_inserted_with_fulltext
            + self.stats.refs_new_arxiv_no_source
            + self.stats.refs_new_arxiv_fetch_error
            + self.stats.refs_new_metadata_only_inserted
            + self.stats.refs_already_in_db
        )
        return self.stats


def ensure_schema(conn: sqlite3.Connection) -> None:
    """Create ``paper_full_text`` and ``citations`` if absent."""
    ensure_paper_full_text_schema(conn)
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS citations (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            source_paper_id  INTEGER NOT NULL,
            cited_paper_id   INTEGER,
            direction        TEXT NOT NULL,
            cited_title      TEXT NOT NULL,
            verified_status  TEXT,
            confidence       REAL,
            s2_paper_id      TEXT,
            fetched_at       TEXT NOT NULL,
            publication_date TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_cit_src
            ON citations(source_paper_id);
        CREATE INDEX IF NOT EXISTS idx_cit_dst
            ON citations(cited_paper_id);
        """
    )
    conn.commit()
