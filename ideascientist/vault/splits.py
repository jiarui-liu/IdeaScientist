"""Target-paper selection and the shared train / val / test partition.

Applies the filter chain from the paper to the post-cutoff papers in the vault
and partitions the survivors. All three roles train on the same partition, so
a paper can never be a training target for one role and a test target for
another.

The chain, in order:

1. published on or after the cutoff, with LaTeX source successfully fetched;
2. more than ``ACCESSIBILITY_THRESHOLD`` of cited works have usable full text
   in the vault, so the agent can actually read the work it must ground in;
3. a results-masked record exists with a non-trivial challenge and problem
   definition;
4. an LLM classifies the paper as methodological, excluding surveys, position
   papers, dataset and benchmark releases, and empirical engineering reports.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sqlite3
from pathlib import Path
from typing import Iterable, Optional

from ideascientist.harness.corpus import CUTOFF_DATE
from ideascientist.utils.db import DEFAULT_VAULT_PATH
from ideascientist.utils.llm import LLMSettings, chat

logger = logging.getLogger(__name__)

ACCESSIBILITY_THRESHOLD = 0.70
MIN_FIELD_CHARS = 20
FULLTEXT_SOURCES = ("latex", "pdf", "html")
RESULTS_MASKED_TABLE = "metadata_results_masked"

N_VAL = 200
N_TEST = 277
SPLIT_SEED = 42

METHODOLOGY_SYSTEM = (
    "You classify research papers by their PRIMARY contribution type. "
    "A paper is a 'methodology paper' iff its main contribution is a NEW "
    "technical method, algorithm, model, architecture, training procedure, "
    "or framework that solves a defined problem. Be STRICT.\n\n"
    "FALSE for: pure surveys / literature reviews, position or opinion "
    "papers, pure benchmark or dataset releases (without a novel method), "
    "applications of existing methods to a new domain (without methodological "
    "novelty), pure empirical analyses or measurement studies, system / "
    "engineering reports without a novel technique, tutorials.\n\n"
    "If the paper introduces a clearly named method/model/architecture AND "
    "evaluates it against baselines, it is methodology=true. If the main "
    "story is 'we built a benchmark' or 'we measured X' or 'we surveyed Y' "
    "with no novel method, it is methodology=false.\n\n"
    "Respond with ONLY a JSON object: "
    '{"is_methodology": true|false, "reason": "<= 25 words"}'
)


def candidate_ids(
    db: Path = DEFAULT_VAULT_PATH,
    *,
    cutoff: str = CUTOFF_DATE,
    threshold: float = ACCESSIBILITY_THRESHOLD,
) -> list[int]:
    """Paper ids passing the structural filters (steps 1-3).

    The citation-accessibility join is materialized into temp tables; as a
    single query over millions of citation edges it does not finish.
    """
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    con.execute("PRAGMA busy_timeout=60000")
    src_ph = ",".join("?" for _ in FULLTEXT_SOURCES)
    try:
        con.execute("CREATE TEMP TABLE s1(id INTEGER PRIMARY KEY)")
        con.execute(
            "INSERT INTO s1 SELECT p.id FROM papers p "
            "JOIN paper_full_text f ON f.paper_id = p.id "
            "WHERE f.content_source = 'latex' AND f.fetch_status = 'ok' "
            "AND f.char_len > 0 "
            "AND p.date GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]*' "
            "AND p.date >= ?",
            (cutoff,),
        )
        logger.info("step 1 (post-cutoff with LaTeX): %d", _count(con, "s1"))

        con.execute(
            f"""CREATE TEMP TABLE cit AS
                SELECT c.source_paper_id AS sid,
                       COUNT(*) AS total,
                       SUM(CASE WHEN f2.paper_id IS NOT NULL THEN 1 ELSE 0 END) AS acc
                FROM citations c
                JOIN s1 ON s1.id = c.source_paper_id
                LEFT JOIN paper_full_text f2
                       ON f2.paper_id = c.cited_paper_id
                      AND f2.content_source IN ({src_ph})
                      AND f2.fetch_status = 'ok'
                      AND f2.char_len > 0
                GROUP BY c.source_paper_id""",
            FULLTEXT_SOURCES,
        )
        con.execute("CREATE INDEX temp_cit_sid ON cit(sid)")
        con.execute("CREATE TEMP TABLE s2(id INTEGER PRIMARY KEY)")
        con.execute(
            "INSERT INTO s2 SELECT sid FROM cit "
            "WHERE total > 0 AND CAST(acc AS REAL) / total > ?",
            (threshold,),
        )
        logger.info("step 2 (citation accessibility): %d", _count(con, "s2"))

        rows = con.execute(
            f"SELECT s2.id, b.fields_json FROM s2 "
            f"JOIN {RESULTS_MASKED_TABLE} b ON b.paper_id = s2.id"
        ).fetchall()
    finally:
        con.close()

    from ideascientist.vault.fields import field_texts

    kept = []
    for pid, fields_json in rows:
        try:
            record = json.loads(fields_json) if fields_json else {}
        except (ValueError, TypeError):
            continue
        if not isinstance(record, dict):
            continue
        texts = field_texts(record)
        if (
            len(texts["challenge"]) >= MIN_FIELD_CHARS
            and len(texts["problem_definition"]) >= MIN_FIELD_CHARS
        ):
            kept.append(int(pid))
    logger.info("step 3 (results-masked record present): %d", len(kept))
    return sorted(kept)


def _count(con: sqlite3.Connection, table: str) -> int:
    return con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


def is_methodology_paper(
    title: str,
    abstract: str,
    *,
    settings: Optional[LLMSettings] = None,
    model: Optional[str] = None,
) -> tuple[bool, str]:
    """Step 4: classify one paper's primary contribution type."""
    reply = chat(
        [
            {"role": "system", "content": METHODOLOGY_SYSTEM},
            {"role": "user", "content": f"Title: {title}\n\nAbstract: {abstract}"},
        ],
        settings=settings,
        model=model,
        temperature=0.0,
        max_tokens=256,
    )
    try:
        obj = json.loads(reply[reply.find("{"): reply.rfind("}") + 1])
        return bool(obj.get("is_methodology")), str(obj.get("reason", ""))
    except (ValueError, json.JSONDecodeError):
        return False, "unparseable classifier reply"


def partition(
    ids: Iterable[int],
    *,
    n_val: int = N_VAL,
    n_test: int = N_TEST,
    seed: int = SPLIT_SEED,
) -> dict[str, list[int]]:
    """Randomly partition target ids into train / val / test, fully disjoint."""
    pool = sorted(set(int(i) for i in ids))
    random.Random(seed).shuffle(pool)
    test = sorted(pool[:n_test])
    val = sorted(pool[n_test:n_test + n_val])
    train = sorted(pool[n_test + n_val:])
    return {"train": train, "val": val, "test": test}


def write_splits(splits: dict[str, list[int]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ("train", "val", "test"):
        (out_dir / f"{name}.txt").write_text(
            "\n".join(str(i) for i in splits.get(name, [])) + "\n"
        )
    (out_dir / "split_meta.json").write_text(
        json.dumps(
            {
                "cutoff": CUTOFF_DATE,
                "accessibility_threshold": ACCESSIBILITY_THRESHOLD,
                "seed": SPLIT_SEED,
                **{f"n_{k}": len(v) for k, v in splits.items()},
            },
            indent=2,
        )
        + "\n"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the shared paper split.")
    ap.add_argument("--db", type=Path, default=DEFAULT_VAULT_PATH)
    ap.add_argument("--out", type=Path, default=Path("data/splits"))
    ap.add_argument("--skip-methodology", action="store_true",
                    help="skip the LLM classifier (steps 1-3 only)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ids = candidate_ids(args.db)

    if not args.skip_methodology:
        con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
        try:
            meta = {
                int(pid): (t or "", a or "")
                for pid, t, a in con.execute(
                    "SELECT id, title, abstract FROM papers WHERE id IN "
                    "(" + ",".join("?" * len(ids)) + ")",
                    ids,
                )
            }
        finally:
            con.close()
        ids = [i for i in ids if is_methodology_paper(*meta.get(i, ("", "")))[0]]
        logger.info("step 4 (methodological): %d", len(ids))

    splits = partition(ids)
    write_splits(splits, args.out)
    for k, v in splits.items():
        print(f"{k:>5}: {len(v)}")


if __name__ == "__main__":
    main()
