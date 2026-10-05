"""Citation precision and recall over the proposal's literature grounding.

We compare the set of papers **cited in the proposal** against two
ground-truth sets:

  1. the **target paper's related-work citations** (from
     the vault, ``citations.is_related_work=1``);
  2. that set **unioned with the most-similar prior papers**
     (same-challenge / same-problem-definition, from
     :mod:`similar_papers`).

Papers are matched across the three sets by *canonical keys* (arxiv / doi /
normalized title — see ``eval.citations.canonical``). A report paper counts as
a true positive if ANY of its keys appears in the ground-truth key universe,
and vice-versa for recall. This tolerates the common arxiv-vs-title-only
asymmetry between report anchors (which resolve to arxiv) and the source
bibliography (which is frequently title-only).
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

from ideascientist.vault.citations.canonical import (
    arxiv_from_url,
    norm_arxiv,
    norm_doi,
    norm_title,
)

from ideascientist.utils.db import DEFAULT_VAULT_PATH
from ideascientist.harness.corpus import CUTOFF_DATE, get_paper_fields
from .similar_papers import SimilarPaper

_VAULT = DEFAULT_VAULT_PATH


# --------------------------------------------------------------------------- #
# Paper identity
# --------------------------------------------------------------------------- #


@dataclass
class KeyedPaper:
    """A paper represented by the set of canonical keys it can match on."""

    label: str  # human-readable identifier (anchor / title)
    title: str = ""
    arxiv_id: str = ""
    doi: str = ""
    keys: frozenset[str] = field(default_factory=frozenset)
    origin: str = ""  # how it was resolved (debugging/transparency)

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "title": self.title,
            "arxiv_id": self.arxiv_id,
            "doi": self.doi,
            "keys": sorted(self.keys),
            "origin": self.origin,
        }


def _loose_title_key(title: str) -> str:
    """A 'loose' title key = normalized title with any subtitle (after the first
    colon) stripped. Catches the arXiv-vs-published mismatch where one side adds
    a ': subtitle'. Gated on length to avoid generic short stems colliding."""
    main = (title or "").split(":", 1)[0]
    nt = norm_title(main)
    return f"ltitle:{nt}" if len(nt) >= 12 else ""


def paper_keys(arxiv_id: str = "", doi: str = "", title: str = "", url: str = "") -> frozenset[str]:
    """Build the set of canonical keys for a paper (arxiv / doi / title / loose-title).

    Emits independent keys so two records match if they agree on ANY identifier,
    plus a loose-title key tolerant of subtitle differences.
    """
    keys: set[str] = set()
    ax = norm_arxiv(arxiv_id) or arxiv_from_url(url)
    if ax:
        keys.add(f"arxiv:{ax}")
    nd = norm_doi(doi)
    if nd and not nd.startswith("10.48550/arxiv"):
        keys.add(f"doi:{nd}")
    nt = norm_title(title)
    if nt:
        keys.add(f"title:{nt}")
    lt = _loose_title_key(title)
    if lt:
        keys.add(lt)
    keys.discard("")
    return frozenset(keys)


# --------------------------------------------------------------------------- #
# Cross-identifier bridging through the vault
# --------------------------------------------------------------------------- #
# Source related-work citations are parsed from .bbl/.bib and are almost always
# TITLE-ONLY (no arxiv/doi), while report citations resolve to arxiv. To match
# them we bridge through the vault: any paper there contributes a "group"
# of co-referring keys (its arxiv + doi + title + loose-title), so a title-only
# citation inherits the arxiv of the same DB paper, and vice-versa.


@lru_cache(maxsize=1)
def _db_key_groups() -> dict[str, frozenset[str]]:
    groups: dict[str, set[str]] = {}
    if not _VAULT.exists():
        return {}
    con = sqlite3.connect(f"file:{_VAULT}?mode=ro", uri=True)
    try:
        rows = con.execute(
            "SELECT title, url, arxiv_id, doi FROM papers WHERE title IS NOT NULL"
        ).fetchall()
    except sqlite3.Error:
        return {}
    finally:
        con.close()
    for title, url, arxiv_id, doi in rows:
        ks = paper_keys(arxiv_id or "", doi or "", title or "", url or "")
        if len(ks) < 2:
            continue  # nothing to bridge from a single-key paper
        for k in ks:
            g = groups.get(k)
            if g is None:
                groups[k] = set(ks)
            else:
                g.update(ks)
    return {k: frozenset(v) for k, v in groups.items()}


@lru_cache(maxsize=4)
def _accessible_pre_cutoff_keys(cutoff_year: int) -> frozenset[str]:
    """Keys of papers we have SUMMARIZED (metadata row) AND that predate the
    cutoff year — i.e. the prior work the agent could actually have retrieved
    and cited. Year is taken from papers.date (publication date)."""
    if not _VAULT.exists():
        return frozenset()
    con = sqlite3.connect(f"file:{_VAULT}?mode=ro", uri=True)
    try:
        rows = con.execute(
            "SELECT p.title, p.url, p.arxiv_id, p.doi, p.date "
            "FROM papers p INNER JOIN metadata_results_masked m ON m.paper_id = p.id"
        ).fetchall()
    except sqlite3.Error:
        return frozenset()
    finally:
        con.close()
    out: set[str] = set()
    for title, url, arxiv_id, doi, date in rows:
        y = str(date or "")[:4]
        if not (y.isdigit() and int(y) < cutoff_year):
            continue
        out |= set(paper_keys(arxiv_id or "", doi or "", title or "", url or ""))
    return frozenset(out)


def _bridge(keys: frozenset[str]) -> frozenset[str]:
    groups = _db_key_groups()
    expanded = set(keys)
    for k in keys:
        g = groups.get(k)
        if g:
            expanded |= g
    return frozenset(expanded)


def _resolve_numeric_anchor(paper_id: int) -> Optional[KeyedPaper]:
    f = get_paper_fields(paper_id)
    if not f:
        return None
    return KeyedPaper(
        label=f"ref-{paper_id}",
        title=f.get("title", ""),
        arxiv_id=f.get("arxiv_id", ""),
        doi=f.get("doi", ""),
        keys=paper_keys(f.get("arxiv_id", ""), f.get("doi", ""), f.get("title", ""), f.get("url", "")),
        origin=f"paper_id={paper_id}",
    )


def keyed_report_cited(
    report_obj: "dict | None",
) -> tuple[list[KeyedPaper], dict[str, Any]]:
    """Resolve the papers cited in a report.json object to keyed papers.

    Reuses the report_writer's own extraction — ``related_work[].citations``
    anchors plus inline bare ``[id]`` — so the evaluation and the reward score
    the same citation set. Ids that do not resolve to a paper are dropped.
    """
    from ideascientist.vault.references.report_writer import (
        _anchor_ids, _inline_anchor_ids,
    )
    ids = sorted(_anchor_ids(report_obj) | _inline_anchor_ids(report_obj)) if report_obj else []
    resolved = [kp for i in ids if (kp := _resolve_numeric_anchor(int(i)))]
    diag = {"num_report_cited": len(resolved), "cited_ids": ids,
            "source": "report_json_cited_ids"}
    return resolved, diag


# --------------------------------------------------------------------------- #
# Ground-truth sets
# --------------------------------------------------------------------------- #


def source_citations(paper_id: int) -> list[KeyedPaper]:
    """All citations of the source paper from the vault."""
    if not _VAULT.exists():
        return []
    con = sqlite3.connect(f"file:{_VAULT}?mode=ro", uri=True)
    try:
        rows = con.execute(
            "SELECT p.title, p.arxiv_id, p.doi FROM citations c "
            "JOIN papers p ON p.id = c.cited_paper_id "
            "WHERE c.source_paper_id=?",
            (int(paper_id),),
        ).fetchall()
    except sqlite3.Error:
        return []
    finally:
        con.close()
    out: list[KeyedPaper] = []
    for title, arxiv_id, doi in rows:
        keys = paper_keys(arxiv_id or "", doi or "", title or "")
        if not keys:
            continue
        out.append(KeyedPaper(
            label=title or (arxiv_id or "?"),
            title=title or "",
            arxiv_id=arxiv_id or "",
            doi=doi or "",
            keys=keys,
            origin="source_paper.citations",
        ))
    return out


def similar_as_keyed(sims: list[SimilarPaper]) -> list[KeyedPaper]:
    out: list[KeyedPaper] = []
    for s in sims:
        keys = paper_keys(s.arxiv_id, s.doi, s.title)
        if not keys:
            continue
        out.append(KeyedPaper(
            label=s.title or f"paper_id={s.paper_id}",
            title=s.title,
            arxiv_id=s.arxiv_id,
            doi=s.doi,
            keys=keys,
            origin=f"similar(paper_id={s.paper_id})",
        ))
    return out


def _bridge_papers(papers: list[KeyedPaper]) -> list[KeyedPaper]:
    return [KeyedPaper(p.label, p.title, p.arxiv_id, p.doi, _bridge(p.keys), p.origin)
            for p in papers]


def _filter_citable(gt_papers: list[KeyedPaper], cutoff_year: int) -> list[KeyedPaper]:
    """Keep only ground-truth papers the agent could realistically have cited:
    summarized in our DB (metadata) AND published before the cutoff year."""
    acc = _accessible_pre_cutoff_keys(cutoff_year)
    return [p for p in gt_papers if p.keys & acc]


def _dedupe(papers: list[KeyedPaper]) -> list[KeyedPaper]:
    merged: list[KeyedPaper] = []
    for p in papers:
        hit = None
        for m in merged:
            if p.keys & m.keys:
                hit = m
                break
        if hit is None:
            merged.append(KeyedPaper(p.label, p.title, p.arxiv_id, p.doi, p.keys, p.origin))
        else:
            hit.keys = frozenset(hit.keys | p.keys)
            if not hit.title and p.title:
                hit.title = p.title
    return merged


# --------------------------------------------------------------------------- #
# Precision / recall
# --------------------------------------------------------------------------- #


def precision_recall(
    report_papers: list[KeyedPaper], gt_papers: list[KeyedPaper]
) -> dict[str, Any]:
    """Set-level P/R/F1 with key-set-overlap matching."""
    report = _dedupe(report_papers)
    gt = _dedupe(gt_papers)
    gt_keys: set[str] = set().union(*[set(p.keys) for p in gt]) if gt else set()
    report_keys: set[str] = set().union(*[set(p.keys) for p in report]) if report else set()

    tp_report = [p for p in report if p.keys & gt_keys]
    covered_gt = [p for p in gt if p.keys & report_keys]

    n_report, n_gt = len(report), len(gt)
    precision = len(tp_report) / n_report if n_report else 0.0
    recall = len(covered_gt) / n_gt if n_gt else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "num_report_cited": n_report,
        "num_ground_truth": n_gt,
        "num_true_positives": len(tp_report),
        "num_ground_truth_covered": len(covered_gt),
        "matched_titles": [p.title or p.label for p in tp_report],
        "missed_ground_truth_titles": [p.title or p.label for p in gt if not (p.keys & report_keys)],
    }


def evaluate_citations(
    report_md: str,
    papers_dir: Optional[Path],
    source_paper_id: int,
    sims_all: list[SimilarPaper],
    sims_pre_cutoff: Optional[list[SimilarPaper]] = None,
    cutoff_date: str = CUTOFF_DATE,
    report_cited_override: Optional[list[KeyedPaper]] = None,
) -> dict[str, Any]:
    """Full task-3 result.

    Reports each precision/recall in TWO variants:
      * ``all`` — every ground-truth paper (absolute coverage);
      * ``citable_pre_cutoff`` — only ground-truth papers the agent could have
        retrieved (summarized in our DB AND pre-cutoff) — the fair denominator.

    Ground-truth comes in two flavors: the source paper's full citation list,
    and that set unioned with the most-similar prior papers. All keys are
    bridged through the vault so title-only ↔ arxiv-only records unify.
    """
    cutoff_year = int(cutoff_date[:4]) if cutoff_date else 9999
    sims_pre_cutoff = sims_pre_cutoff if sims_pre_cutoff is not None else []

    if report_cited_override is not None:
        report_papers = report_cited_override
        diag = {"num_anchors": 0, "anchors": [], "num_papers_dir_files": 0,
                "unresolved_anchors": [], "source": "source_paper_own_citations",
                "num_report_cited": len(report_papers)}
    else:
        report_papers, diag = [], {"num_report_cited": 0, "source": "no_override"}
    report_papers = _bridge_papers(report_papers)

    gt = _bridge_papers(source_citations(source_paper_id))
    gt_citable = _filter_citable(gt, cutoff_year)
    sim_all = _bridge_papers(similar_as_keyed(sims_all))
    sim_pre = _bridge_papers(similar_as_keyed(sims_pre_cutoff))

    union_all = _dedupe(gt + sim_all)
    union_citable = _dedupe(gt_citable + sim_pre)

    return {
        "report_resolution": diag,
        "cutoff_date": cutoff_date,
        "precision_recall_vs_source_citations": {
            "all": precision_recall(report_papers, gt),
            "citable_pre_cutoff": precision_recall(report_papers, gt_citable),
        },
        "precision_recall_vs_union": {
            "all": precision_recall(report_papers, union_all),
            "citable_pre_cutoff": precision_recall(report_papers, union_citable),
        },
        "sets": {
            "report_cited": [p.to_dict() for p in _dedupe(report_papers)],
            "source_citations_all": [p.to_dict() for p in _dedupe(gt)],
            "source_citations_citable_pre_cutoff": [p.to_dict() for p in _dedupe(gt_citable)],
            "similar_all": [p.to_dict() for p in _dedupe(sim_all)],
            "similar_pre_cutoff": [p.to_dict() for p in _dedupe(sim_pre)],
            "union_all": [p.to_dict() for p in union_all],
            "union_citable_pre_cutoff": [p.to_dict() for p in union_citable],
        },
    }
