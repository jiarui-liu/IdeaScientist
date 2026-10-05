#!/usr/bin/env python3
"""Build the Svalbard Idea Vault, one stage at a time.

The stages are separate commands because each is long-running, resumable, and
independently useful — and because most users will substitute their own corpus
for the first two rather than re-crawl arXiv.

    init        create the schema
    crawl       arXiv metadata over a date range     -> papers
    fulltext    e-print LaTeX for crawled papers     -> paper_full_text
    citations   one-hop reference expansion          -> citations, papers
    decompose   full text -> results-masked record   -> metadata_results_masked
    index       BM25 index and field embeddings
    splits      target filtering and the partition

Every stage is idempotent: re-running skips work already done. One writer at a
time — two concurrent writers on the same file silently lose rows.
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
from datetime import timedelta
from pathlib import Path

from ideascientist.utils.db import DEFAULT_VAULT_PATH, assert_no_other_writer
from ideascientist.vault.arxiv.ingest import (
    RESULTS_MASKED_TABLE,
    ensure_vault_schema,
    insert_full_text,
    insert_paper,
    insert_results_masked,
    verify_counts,
)

logger = logging.getLogger("build_vault")

# A month at a time: the arXiv API caps a single windowed query, and a smaller
# window also makes the crawl resumable at a useful granularity.
CRAWL_WINDOW_DAYS = 30


def _connect(db: Path) -> sqlite3.Connection:
    db.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(db))
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA busy_timeout=60000")
    return con


def stage_init(con: sqlite3.Connection, args) -> None:
    ensure_vault_schema(con)
    logger.info("schema ready")


def stage_crawl(con: sqlite3.Connection, args) -> None:
    from ideascientist.vault.arxiv.crawl import (
        DEFAULT_CATEGORIES,
        parse_date,
        search_arxiv_window,
    )

    if not (args.start and args.end):
        raise SystemExit("crawl needs --start and --end (YYYY-MM-DD)")
    ensure_vault_schema(con)

    start, end = parse_date(args.start), parse_date(args.end)
    categories = args.categories or list(DEFAULT_CATEGORIES)
    logger.info("crawling %s..%s over %d categories", start, end, len(categories))

    total = new = 0
    window_start = start
    while window_start <= end:
        window_end = min(window_start + timedelta(days=CRAWL_WINDOW_DAYS - 1), end)
        papers = search_arxiv_window(window_start, window_end, categories=categories)
        for p in papers:
            pid, status = insert_paper(con, p)
            total += 1
            new += status == "new"
            if args.limit and total >= args.limit:
                break
        logger.info("%s..%s: %d seen, %d new overall", window_start, window_end,
                    len(papers), new)
        if args.limit and total >= args.limit:
            break
        window_start = window_end + timedelta(days=1)
    logger.info("crawl done: %d papers seen, %d new", total, new)


def stage_fulltext(con: sqlite3.Connection, args) -> None:
    from ideascientist.vault.citations.extract import fetch_latex_source_with_bbl
    from ideascientist.vault.text import _clean_latex

    rows = con.execute(
        "SELECT p.id, p.arxiv_id FROM papers p "
        "LEFT JOIN paper_full_text f ON f.paper_id = p.id "
        "WHERE p.arxiv_id != '' AND f.paper_id IS NULL "
        + (f"LIMIT {int(args.limit)}" if args.limit else "")
    ).fetchall()
    logger.info("%d papers need full text", len(rows))

    ok = 0
    for paper_id, arxiv_id in rows:
        try:
            tex, bbl = fetch_latex_source_with_bbl(arxiv_id)
        except Exception as exc:  # noqa: BLE001 - one bad paper must not stop the run
            insert_full_text(con, paper_id, arxiv_id, content_source="latex",
                             full_text=None, raw_tex=None, bbl_content=None,
                             fetch_status="error", fetch_error=str(exc)[:500])
            continue
        if not tex:
            insert_full_text(con, paper_id, arxiv_id, content_source="latex",
                             full_text=None, raw_tex=None, bbl_content=None,
                             fetch_status="no_source")
            continue
        insert_full_text(con, paper_id, arxiv_id, content_source="latex",
                         full_text=_clean_latex(tex), raw_tex=tex,
                         bbl_content=bbl, fetch_status="ok")
        ok += 1
    logger.info("full text: %d/%d retrieved", ok, len(rows))


def stage_citations(con: sqlite3.Connection, args) -> None:
    from ideascientist.vault.citations.expand import CitationExpander, ensure_schema

    ensure_schema(con)
    expander = CitationExpander(
        con,
        start_date=args.start or "2026-01-01",
        max_fetch=args.limit or None,
    )
    expander.build_existing_index()
    stats = expander.run()
    logger.info("citation expansion: %s", stats)


def stage_decompose(con: sqlite3.Connection, args) -> None:
    from ideascientist.vault.decompose import decompose_paper

    rows = con.execute(
        "SELECT p.id, p.title, f.full_text FROM papers p "
        "JOIN paper_full_text f ON f.paper_id = p.id "
        f"LEFT JOIN {RESULTS_MASKED_TABLE} b ON b.paper_id = p.id "
        "WHERE f.fetch_status = 'ok' AND f.full_text != '' AND b.paper_id IS NULL "
        + (f"LIMIT {int(args.limit)}" if args.limit else "")
    ).fetchall()
    logger.info("%d papers need a results-masked record", len(rows))

    ok = 0
    for paper_id, title, full_text in rows:
        try:
            record = decompose_paper(title or "", full_text, model=args.model)
        except Exception as exc:  # noqa: BLE001 - skip and continue
            logger.warning("decompose failed for %s: %s", paper_id, exc)
            continue
        insert_results_masked(con, paper_id, record, model=args.model or "")
        ok += 1
        if ok % 50 == 0:
            logger.info("decomposed %d/%d", ok, len(rows))
    logger.info("decompose: %d/%d records written", ok, len(rows))


def stage_index(con: sqlite3.Connection, args) -> None:
    from ideascientist.vault.index import build_field_embeddings, build_keyword_index

    con.close()
    build_keyword_index(args.db, Path(args.keyword_out))
    if not args.skip_embeddings:
        build_field_embeddings(args.db, Path(args.embedding_dir))


def stage_splits(con: sqlite3.Connection, args) -> None:
    from ideascientist.vault.splits import candidate_ids, partition, write_splits

    con.close()
    ids = candidate_ids(args.db)
    if not args.skip_methodology:
        from ideascientist.vault.splits import is_methodology_paper

        meta = _paper_meta(args.db, ids)
        ids = [i for i in ids if is_methodology_paper(*meta.get(i, ("", "")))[0]]
        logger.info("after the methodology filter: %d", len(ids))
    splits = partition(ids)
    write_splits(splits, Path(args.splits_out))
    for name, members in splits.items():
        print(f"{name:>5}: {len(members)}")


def _paper_meta(db: Path, ids: list[int]) -> dict[int, tuple[str, str]]:
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        return {
            int(pid): (t or "", a or "")
            for pid, t, a in con.execute(
                "SELECT id, title, abstract FROM papers WHERE id IN "
                "(" + ",".join("?" * len(ids)) + ")", ids
            )
        }
    finally:
        con.close()


STAGES = {
    "init": stage_init,
    "crawl": stage_crawl,
    "fulltext": stage_fulltext,
    "citations": stage_citations,
    "decompose": stage_decompose,
    "index": stage_index,
    "splits": stage_splits,
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=list(STAGES))
    ap.add_argument("--db", type=Path, default=DEFAULT_VAULT_PATH)
    ap.add_argument("--start", help="crawl: start date; citations: expand papers from this date")
    ap.add_argument("--end", help="crawl: end date (YYYY-MM-DD)")
    ap.add_argument("--categories", nargs="*",
                    help="crawl: arXiv categories (default: the cs.* defaults; "
                         "pass the full list to reproduce the all-subject vault)")
    ap.add_argument("--limit", type=int, default=0, help="cap items processed this run")
    ap.add_argument("--model", default=None, help="decompose: model override")
    ap.add_argument("--keyword-out", default="data/keyword_index.db")
    ap.add_argument("--embedding-dir", default="data/paper_embeddings")
    ap.add_argument("--skip-embeddings", action="store_true",
                    help="index: build only the BM25 index (embeddings need a GPU)")
    ap.add_argument("--splits-out", default="data/splits")
    ap.add_argument("--skip-methodology", action="store_true",
                    help="splits: skip the LLM contribution-type filter")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    assert_no_other_writer("build_vault.py")

    con = _connect(args.db)
    try:
        STAGES[args.stage](con, args)
    finally:
        try:
            con.close()
        except sqlite3.ProgrammingError:
            pass

    if args.stage not in ("index", "splits"):
        logger.info("vault counts: %s", verify_counts(str(args.db)))


if __name__ == "__main__":
    main()
