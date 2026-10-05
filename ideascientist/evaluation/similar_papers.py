"""Retrieve the prior papers most similar to the source paper.

"Similar challenge + similar problem definition" is defined (per the eval spec)
as the union of three embedding rankings against the source paper's stored
Qwen3-Embedding field vectors:

  * top-5 by **challenge** cosine,
  * top-5 by **problem-definition** cosine,
  * top-10 by **challenge + problem-definition** cosine.

Vectors come from the precomputed ``data/paper_embeddings/{challenge,
problem_definition}.npy`` matrices — the same space the harness searches. A
source paper whose stored vector for a space is missing or zero (not every
held-out paper yielded a problem definition) falls back to re-encoding its text
through the query server, so the ranking stays meaningful rather than empty.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Optional

import numpy as np

from ideascientist.utils.db import DEFAULT_VAULT_PATH
from ideascientist.harness.corpus import (
    _field_matrices,
    get_paper_fields,
    get_paper_record,
)


@dataclass
class SimilarPaper:
    paper_id: int
    title: str
    challenge_score: float
    problem_definition_score: float
    sum_score: float
    selected_by: list[str] = field(default_factory=list)  # {"challenge","pd","sum"}
    problem_definition: str = ""
    challenge: str = ""
    solution: str = ""
    arxiv_id: str = ""
    doi: str = ""
    record: Optional[dict] = None  # the paper's results-masked record, if present
    full_text: str = ""  # capped full text, used when no record exists


def _query_vec(space: str, source_paper_id: int, fallback_text: str) -> Optional[np.ndarray]:
    """Unit query vector for ``space``: stored source vector if valid, else
    Qwen3-Embedding-encode ``fallback_text``. Returns ``None`` if neither is available."""
    g = _field_matrices()
    row = g["row_of_pid"].get(int(source_paper_id))
    if row is not None and g[f"{space}_valid"][row]:
        return g[space][row]
    if fallback_text and fallback_text.strip():
        from ideascientist.harness.embedding_spaces import encode_query
        v = np.asarray(encode_query(space, fallback_text), dtype=np.float32)
        n = float(np.linalg.norm(v))
        return v / n if n > 0 else None
    return None


def _load_full_text(paper_id: int, cap: int) -> str:
    import sqlite3
    from pathlib import Path
    autodb = DEFAULT_VAULT_PATH
    if not autodb.exists():
        return ""
    con = sqlite3.connect(f"file:{autodb}?mode=ro", uri=True)
    try:
        row = con.execute(
            "SELECT full_text FROM paper_full_text WHERE paper_id=? LIMIT 1",
            (int(paper_id),),
        ).fetchone()
    except sqlite3.Error:
        return ""
    finally:
        con.close()
    return (row[0] or "")[:cap] if row else ""


def find_precutoff_papers(
    source_paper_id: int,
    *,
    top_challenge: int = 5,
    top_pd: int = 5,
    top_sum: int = 10,
    solution_char_cap: int = 4000,
) -> list[SimilarPaper]:
    """Prior work for the pre-cutoff novelty judgement.

    Ranks the source paper against ``data/paper_embeddings_pre2026/`` --
    2,620,467 papers with a publication date before 2026-01-01 -- using the same
    three rankings as :func:`find_similar_papers`.

    Do not implement this by date-filtering :func:`find_similar_papers`. That
    function ranks against the target-pool index, whose dates are ingest dates
    rather than publication dates, so the filter matches nothing and silently
    returns an empty comparison set — which would turn the Cutoff score into a
    reference-free one without any error.

    Retrieval streams the whole pre-cutoff index, so scoring many targets is
    much cheaper batched into one pass than called per paper.
    """
    import numpy as np

    from ideascientist.harness.corpus import _field_matrices

    emb = Path(os.environ.get("IDEASCIENTIST_PRECUTOFF_EMBEDDINGS",
                              "data/paper_embeddings_pre2026"))
    pid_cache = emb / "pre_cutoff_ids.npy"
    if not emb.exists():
        raise FileNotFoundError(f"pre-cutoff embedding index missing: {emb}")

    g = _field_matrices()
    row = g["row_of_pid"].get(int(source_paper_id))
    if row is None or not (g["challenge_valid"][row] and g["problem_definition_valid"][row]):
        return []
    qc = np.asarray(g["challenge"][row], dtype=np.float32)
    qp = np.asarray(g["problem_definition"][row], dtype=np.float32)

    if pid_cache.exists():
        pids = np.load(pid_cache)
    else:
        import json as _json
        meta = _json.loads((emb / "metadata.json").read_text())
        pids = np.array([m["paper_id"] for m in meta], dtype=np.int64)

    C = np.load(emb / "challenge.npy", mmap_mode="r")
    P = np.load(emb / "problem_definition.npy", mmap_mode="r")
    cs = np.empty(C.shape[0], dtype=np.float32)
    ps = np.empty(C.shape[0], dtype=np.float32)
    step = 200_000
    for s in range(0, C.shape[0], step):
        e = min(C.shape[0], s + step)
        c = np.asarray(C[s:e]); p = np.asarray(P[s:e])
        nc = np.linalg.norm(c, axis=1); npd = np.linalg.norm(p, axis=1)
        cs[s:e] = (c @ qc) / np.where(nc == 0, 1.0, nc)
        ps[s:e] = (p @ qp) / np.where(npd == 0, 1.0, npd)

    sel: dict[int, set[str]] = {}
    for scores, k, tag in ((cs, top_challenge, "challenge"), (ps, top_pd, "pd"),
                           (cs + ps, top_sum, "sum")):
        for i in np.argsort(-scores)[:k]:
            pid = int(pids[i])
            if pid != int(source_paper_id):
                sel.setdefault(pid, set()).add(tag)

    out: list[SimilarPaper] = []
    for pid, methods in sel.items():
        i = int(np.searchsorted(pids, pid))
        fields = get_paper_fields(pid) or {}
        out.append(SimilarPaper(
            paper_id=pid, title=fields.get("title", ""),
            challenge_score=round(float(cs[i]), 4),
            problem_definition_score=round(float(ps[i]), 4),
            sum_score=round(float(cs[i] + ps[i]), 4),
            selected_by=sorted(methods),
            problem_definition=fields.get("problem_definition", ""),
            challenge=fields.get("challenge", ""),
            solution=(fields.get("solution") or "")[:solution_char_cap],
            arxiv_id=fields.get("arxiv_id", ""), doi=fields.get("doi", ""),
            record=get_paper_record(pid), full_text=""))
    out.sort(key=lambda s: -s.sum_score)
    return out


def find_similar_papers(
    source_paper_id: int,
    *,
    challenge_fallback_text: str = "",
    problem_definition_fallback_text: str = "",
    top_challenge: int = 5,
    top_pd: int = 5,
    top_sum: int = 10,
    solution_char_cap: int = 4000,
    full_text_char_cap: int = 8000,
    cutoff_date: Optional[str] = None,
) -> list[SimilarPaper]:
    """Return the union of the three similarity rankings, enriched with each
    paper's structured ``problem_definition`` / ``challenge`` / ``solution``.

    When ``cutoff_date`` is given (e.g. ``"2026-01-01"``), only papers with a
    KNOWN collect_date strictly before it are eligible — i.e. the prior work the
    agent could actually have retrieved. The top-k sizes are the same either
    way, so the two comparison sets stay the same size and the All and Cutoff
    novelty scores remain comparable."""
    g = _field_matrices()
    pids = g["pid"]
    self_id = int(source_paper_id)

    qch = _query_vec("challenge", self_id, challenge_fallback_text)
    qpd = _query_vec("problem_definition", self_id, problem_definition_fallback_text)

    n = len(pids)
    cs = g["challenge"] @ qch if qch is not None else np.zeros(n, dtype=np.float32)
    ps = g["problem_definition"] @ qpd if qpd is not None else np.zeros(n, dtype=np.float32)
    ch_valid = g["challenge_valid"]
    pd_valid = g["problem_definition_valid"]

    pool = pids != self_id
    if cutoff_date:
        dates = np.array([d or "" for d in g["dates"]], dtype=object)
        date_ok = np.array([(d != "" and d < cutoff_date) for d in dates], dtype=bool)
        pool = pool & date_ok

    def _rank(scores: np.ndarray, valid_mask: np.ndarray, k: int) -> list[int]:
        mask = valid_mask & pool
        idx = np.nonzero(mask)[0]
        if idx.size == 0:
            return []
        idx = idx[np.argsort(-scores[idx])][:k]
        return [int(i) for i in idx]

    sel: dict[int, set[str]] = {}  # paper_id -> {methods}

    if qch is not None:
        for i in _rank(cs, ch_valid, top_challenge):
            sel.setdefault(int(pids[i]), set()).add("challenge")
    if qpd is not None:
        for i in _rank(ps, pd_valid, top_pd):
            sel.setdefault(int(pids[i]), set()).add("pd")
    if qch is not None and qpd is not None:
        for i in _rank(cs + ps, ch_valid & pd_valid, top_sum):
            sel.setdefault(int(pids[i]), set()).add("sum")

    row_of = g["row_of_pid"]
    out: list[SimilarPaper] = []
    for pid, methods in sel.items():
        r = row_of.get(pid)
        c = float(cs[r]) if r is not None else 0.0
        p = float(ps[r]) if r is not None else 0.0
        fields = get_paper_fields(pid) or {}
        sol = (fields.get("solution") or "")[:solution_char_cap]
        title = fields.get("title", "") or (g["titles"][r] if r is not None else "")
        record = get_paper_record(pid)
        ft = "" if record is not None else _load_full_text(pid, full_text_char_cap)
        out.append(SimilarPaper(
            paper_id=pid,
            title=title,
            challenge_score=round(c, 4),
            problem_definition_score=round(p, 4),
            sum_score=round(c + p, 4),
            selected_by=sorted(methods),
            problem_definition=fields.get("problem_definition", ""),
            challenge=fields.get("challenge", ""),
            solution=sol,
            arxiv_id=fields.get("arxiv_id", ""),
            doi=fields.get("doi", ""),
            record=record,
            full_text=ft,
        ))
    # Stable, informative order: by combined score desc.
    out.sort(key=lambda s: -s.sum_score)
    return out
