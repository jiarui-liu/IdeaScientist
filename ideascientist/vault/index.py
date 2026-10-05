"""Build the vault's two retrieval indexes.

``build_keyword_index`` writes a contentless FTS5 index over title plus body,
where body is the paper's full text when we have it and the concatenated field
texts otherwise, with a ``papers_meta`` sidecar carrying the title and date that
the cutoff filter needs. Reads are sharded across threads because the per-row
latency, not throughput, is the bottleneck on a corpus this size, and the index
is built on local scratch and copied once at the end since FTS writes are
random.

``build_field_embeddings`` writes one row-aligned ``.npy`` per field space, so
a query vector can be scored against any of the four spaces in
:mod:`ideascientist.vault.fields` independently. An empty field keeps a zero
row so every space stays aligned to ``metadata.json``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import queue
import shutil
import sqlite3
import tempfile
import threading
import time
from pathlib import Path

from ideascientist.harness.corpus import CUTOFF_DATE
from ideascientist.utils.db import DEFAULT_VAULT_PATH
from ideascientist.vault.fields import FIELD_NAMES, field_texts

logger = logging.getLogger(__name__)

RESULTS_MASKED_TABLE = "metadata_results_masked"
DEFAULT_KEYWORD_INDEX = Path("data/keyword_index.db")
DEFAULT_EMBEDDING_DIR = Path("data/paper_embeddings")
DEFAULT_PRE_CUTOFF_DIR = Path("data/paper_embeddings_pre2026")
# The Cutoff comparison set is ranked on these two spaces only.
PRE_CUTOFF_FIELDS = ["challenge", "problem_definition"]

_SENTINEL = None


def _record_body(abstract: str | None, fields_json: str | None) -> str:
    try:
        record = json.loads(fields_json) if fields_json else {}
    except (ValueError, TypeError):
        record = {}
    if not isinstance(record, dict):
        record = {}
    texts = field_texts(record)
    return " ".join(filter(None, [abstract or "", *(texts[f] for f in FIELD_NAMES)]))


def _id_bounds(db: Path) -> tuple[int, int]:
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        lo, hi = con.execute("SELECT MIN(id), MAX(id) FROM papers").fetchone()
    finally:
        con.close()
    return int(lo), int(hi)


def _reader(db: Path, lo: int, hi: int, q: queue.Queue, batch: int) -> None:
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    con.execute("PRAGMA busy_timeout=60000")
    con.execute("PRAGMA cache_size=-262144")
    cur = con.execute(
        "SELECT p.id, p.date, p.title, p.abstract, b.fields_json, f.full_text "
        "FROM papers p "
        f"LEFT JOIN {RESULTS_MASKED_TABLE} b ON b.paper_id = p.id "
        "LEFT JOIN paper_full_text f ON f.paper_id = p.id "
        "WHERE p.id >= ? AND p.id <= ? AND ("
        "      (f.full_text IS NOT NULL AND f.full_text != '') "
        f"   OR p.id IN (SELECT paper_id FROM {RESULTS_MASKED_TABLE}))",
        (lo, hi),
    )
    buf: list[tuple] = []
    for pid, date, title, abstract, fields_json, full_text in cur:
        pid = int(pid or 0)
        if not pid or not title:
            continue
        body = full_text if full_text else _record_body(abstract, fields_json)
        buf.append((pid, title, date or "", body, 1 if full_text else 0))
        if len(buf) >= batch:
            q.put(buf)
            buf = []
    if buf:
        q.put(buf)
    con.close()
    q.put(_SENTINEL)


def _writer(build_path: Path, q: queue.Queue, n_readers: int) -> tuple[int, int]:
    out = sqlite3.connect(str(build_path))
    out.execute("PRAGMA journal_mode=OFF")
    out.execute("PRAGMA synchronous=OFF")
    out.execute("PRAGMA cache_size=-1048576")
    out.execute("CREATE VIRTUAL TABLE papers_fts USING fts5(title, body, content='')")
    out.execute(
        "CREATE TABLE papers_meta "
        "(paper_id INTEGER PRIMARY KEY, title TEXT, collect_date TEXT)"
    )
    n = n_ft = done = 0
    while done < n_readers:
        item = q.get()
        if item is _SENTINEL:
            done += 1
            continue
        out.executemany(
            "INSERT INTO papers_fts(rowid, title, body) VALUES (?,?,?)",
            [(pid, title, body) for (pid, title, _d, body, _h) in item],
        )
        out.executemany(
            "INSERT INTO papers_meta(paper_id, title, collect_date) VALUES (?,?,?)",
            [(pid, title, d) for (pid, title, d, _b, _h) in item],
        )
        n_ft += sum(h for (*_, h) in item)
        n += len(item)
    out.execute("INSERT INTO papers_fts(papers_fts) VALUES('optimize')")
    out.commit()
    out.close()
    return n, n_ft


def _pick_build_dir(final_out: Path, need_gb: int = 120) -> Path:
    candidates = []
    if os.environ.get("IDEASCIENTIST_SCRATCH"):
        candidates.append(Path(os.environ["IDEASCIENTIST_SCRATCH"]))
    candidates += [Path(tempfile.gettempdir()), Path("/scratch"), Path("/local")]
    for d in candidates:
        try:
            if not d.is_dir() or not os.access(d, os.W_OK):
                continue
            st = os.statvfs(d)
            if st.f_bavail * st.f_frsize >= need_gb * (1024 ** 3):
                return Path(tempfile.mkdtemp(prefix="kwidx_", dir=str(d)))
        except OSError:
            continue
    logger.warning("no local scratch with %dGB free; building in place", need_gb)
    return final_out.parent


def build_keyword_index(
    db: Path = DEFAULT_VAULT_PATH,
    out_path: Path = DEFAULT_KEYWORD_INDEX,
    *,
    n_readers: int = 12,
    batch: int = 500,
) -> tuple[int, int]:
    t0 = time.time()
    lo, hi = _id_bounds(db)
    build_dir = _pick_build_dir(out_path)
    build_path = build_dir / "keyword_index.db.building"
    build_path.unlink(missing_ok=True)

    q: queue.Queue = queue.Queue(maxsize=n_readers * 4)
    step = (hi - lo + n_readers) // n_readers
    readers = []
    for i in range(n_readers):
        r_lo = lo + i * step
        if r_lo > hi:
            break
        t = threading.Thread(
            target=_reader,
            args=(db, r_lo, min(r_lo + step - 1, hi), q, batch),
            daemon=True,
        )
        t.start()
        readers.append(t)

    n, n_ft = _writer(build_path, q, len(readers))
    for t in readers:
        t.join()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_final = out_path.with_suffix(".db.copying")
    shutil.copyfile(build_path, tmp_final)
    os.replace(tmp_final, out_path)
    build_path.unlink(missing_ok=True)
    if build_dir != out_path.parent:
        try:
            build_dir.rmdir()
        except OSError:
            pass
    logger.info(
        "indexed %d papers (%d with full text) -> %s in %.0fs",
        n, n_ft, out_path, time.time() - t0,
    )
    return n, n_ft


def load_records(
    db: Path = DEFAULT_VAULT_PATH,
    *,
    max_date: str | None = None,
    shard: int = 0,
    num_shards: int = 1,
) -> list[dict]:
    """Row-aligned records for the embedding spaces, ordered by paper id.

    ``max_date`` filters on publication date, which is what the pre-cutoff
    novelty index needs; the target-pool index is unfiltered.
    """
    where = "WHERE p.date < ? " if max_date else ""
    shard_clause = "AND b.paper_id % ? = ? " if num_shards > 1 else ""
    if shard_clause and not where:
        where, shard_clause = "WHERE 1=1 ", shard_clause
    params: list = ([max_date] if max_date else []) + (
        [num_shards, shard] if num_shards > 1 else []
    )

    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        rows = con.execute(
            "SELECT b.paper_id, p.title, p.date, b.fields_json "
            f"FROM {RESULTS_MASKED_TABLE} b JOIN papers p ON p.id = b.paper_id "
            f"{where}{shard_clause}"
            "ORDER BY b.paper_id",
            params,
        ).fetchall()
    finally:
        con.close()

    records = []
    for pid, title, date, fields_json in rows:
        try:
            parsed = json.loads(fields_json)
        except (ValueError, TypeError):
            parsed = {}
        records.append(
            {
                "paper_id": int(pid),
                "title": title or "",
                "collect_date": (str(date)[:10] if date else ""),
                "texts": field_texts(parsed if isinstance(parsed, dict) else {}),
            }
        )
    return records


def _write_metadata(out_dir: Path, records: list[dict], fields: list[str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metadata.json").write_text(
        json.dumps(
            [
                {
                    "paper_id": r["paper_id"],
                    "title": r["title"],
                    "collect_date": r["collect_date"],
                    **{f"{f}_len": len(r["texts"][f]) for f in fields},
                }
                for r in records
            ]
        ),
        encoding="utf-8",
    )


def build_field_embeddings(
    db: Path = DEFAULT_VAULT_PATH,
    out_dir: Path = DEFAULT_EMBEDDING_DIR,
    *,
    fields: "list[str] | None" = None,
    max_date: str | None = None,
    shard: int = 0,
    num_shards: int = 1,
    gpu_memory_utilization: float = 0.85,
    limit: int = 0,
) -> None:
    """Encode each field into its own row-aligned space.

    With ``num_shards > 1`` this writes one shard per invocation — the pre-cutoff
    cohort is millions of papers, so it is run as a job array and consolidated
    afterwards by :func:`merge_field_embedding_shards`.
    """
    import numpy as np

    from ideascientist.serving.qwen3_embedder import EMB_DIM, build_llm, embed_documents

    fields = list(fields or FIELD_NAMES)
    unknown = [f for f in fields if f not in FIELD_NAMES]
    if unknown:
        raise ValueError(f"unknown field(s) {unknown}; valid: {list(FIELD_NAMES)}")

    out_dir.mkdir(parents=True, exist_ok=True)
    sharded = num_shards > 1
    if sharded and (out_dir / f"shard_{shard}.done").exists():
        logger.info("shard %d already done", shard)
        return

    records = load_records(db, max_date=max_date, shard=shard, num_shards=num_shards)
    if limit:
        records = records[:limit]
    n = len(records)
    logger.info("%d records (max_date=%s, shard %d/%d)", n, max_date, shard, num_shards)

    if sharded:
        for f in fields:
            (out_dir / f).mkdir(exist_ok=True)
        if n == 0:
            (out_dir / f"shard_{shard}.done").write_text("empty\n")
            return
    else:
        _write_metadata(out_dir, records, fields)

    llm = build_llm(gpu_memory_utilization=gpu_memory_utilization)
    for field in fields:
        texts = [r["texts"][field] for r in records]
        nonempty = [i for i, t in enumerate(texts) if t]
        embs = np.zeros((n, EMB_DIM), dtype=np.float32)
        if nonempty:
            vecs = embed_documents(llm, [texts[i] for i in nonempty])
            for j, i in enumerate(nonempty):
                embs[i] = vecs[j]
        dest = (out_dir / field / f"shard_{shard}.npy") if sharded else (out_dir / f"{field}.npy")
        _save_atomic(dest, embs)
        logger.info("%s: %d / %d non-empty", field, len(nonempty), n)

    if sharded:
        _save_atomic(out_dir / f"shard_{shard}.ids.npy",
                     np.asarray([r["paper_id"] for r in records], dtype=np.int64))
        (out_dir / f"shard_{shard}.done").write_text(f"{n}\n")


def _save_atomic(dest: Path, arr) -> None:
    import numpy as np

    tmp = dest.with_suffix(".tmp.npy")
    np.save(tmp, arr)
    tmp.replace(dest)


def merge_field_embedding_shards(
    out_dir: Path,
    db: Path = DEFAULT_VAULT_PATH,
    *,
    fields: "list[str] | None" = None,
    max_date: str | None = None,
    num_shards: int = 8,
    allow_missing: bool = False,
) -> None:
    """Consolidate job-array shards into the per-field stores readers expect.

    Shards partition by ``paper_id % num_shards``, so concatenating them gives
    an interleaved order; rows are re-sorted by paper id to match the ordering
    :func:`load_records` produces, which is what keeps every field space and
    ``metadata.json`` on one index.
    """
    import numpy as np

    fields = list(fields or FIELD_NAMES)
    missing = [s for s in range(num_shards) if not (out_dir / f"shard_{s}.done").exists()]
    if missing and not allow_missing:
        raise FileNotFoundError(f"shards not done: {missing}")
    if missing:
        logger.warning("merging without shards %s", missing)

    present = [s for s in range(num_shards) if (out_dir / f"shard_{s}.ids.npy").exists()]
    ids = np.concatenate([np.load(out_dir / f"shard_{s}.ids.npy") for s in present])
    order = np.argsort(ids, kind="stable")
    ids_sorted = ids[order]

    for field in fields:
        arr = np.concatenate(
            [np.load(out_dir / field / f"shard_{s}.npy") for s in present]
        )[order].astype(np.float32)
        _save_atomic(out_dir / f"{field}.npy", arr)
        logger.info("%s: %s, %d non-empty", field, arr.shape,
                    int((np.linalg.norm(arr, axis=1) >= 0.01).sum()))

    records = [r for r in load_records(db, max_date=max_date)
               if r["paper_id"] in set(int(x) for x in ids_sorted.tolist())]
    if [r["paper_id"] for r in records] != [int(x) for x in ids_sorted.tolist()]:
        raise RuntimeError("merged shard ids do not match the record ordering")
    _write_metadata(out_dir, records, fields)
    _save_atomic(out_dir / "pre_cutoff_ids.npy", ids_sorted)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the vault retrieval indexes.")
    ap.add_argument("what", choices=["keyword", "embeddings", "merge", "all"])
    ap.add_argument("--db", type=Path, default=DEFAULT_VAULT_PATH)
    ap.add_argument("--keyword-out", type=Path, default=DEFAULT_KEYWORD_INDEX)
    ap.add_argument("--embedding-dir", type=Path, default=DEFAULT_EMBEDDING_DIR)
    ap.add_argument("--readers", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--cohort", choices=["target", "pre-cutoff"], default="target",
                    help="pre-cutoff builds the comparison index for Cutoff novelty")
    ap.add_argument("--cutoff-date", default=CUTOFF_DATE)
    ap.add_argument("--fields", default="",
                    help="comma-separated subset of the field spaces to encode")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--allow-missing", action="store_true",
                    help="merge: proceed with shards still unfinished")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    pre_cutoff = args.cohort == "pre-cutoff"
    max_date = args.cutoff_date if pre_cutoff else None
    fields = [f.strip() for f in args.fields.split(",") if f.strip()] or (
        PRE_CUTOFF_FIELDS if pre_cutoff else list(FIELD_NAMES)
    )
    out_dir = args.embedding_dir
    if pre_cutoff and out_dir == DEFAULT_EMBEDDING_DIR:
        out_dir = DEFAULT_PRE_CUTOFF_DIR

    if args.what in ("keyword", "all"):
        build_keyword_index(args.db, args.keyword_out, n_readers=args.readers)
    if args.what in ("embeddings", "all"):
        build_field_embeddings(args.db, out_dir, fields=fields, max_date=max_date,
                               shard=args.shard, num_shards=args.num_shards,
                               limit=args.limit)
    if args.what == "merge":
        merge_field_embedding_shards(out_dir, args.db, fields=fields, max_date=max_date,
                                     num_shards=args.num_shards,
                                     allow_missing=args.allow_missing)


if __name__ == "__main__":
    main()
