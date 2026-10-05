"""Load a harness run-output folder and resolve its ground-truth source paper.

A run folder is named ``<timestamp>_pid<P>``, where ``P`` is the ``papers.id``
of the held-out paper the agent's problem was distilled from.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ideascientist.utils.db import DEFAULT_VAULT_PATH
from ideascientist.harness.corpus import get_paper_fields, get_paper_record

_VAULT = DEFAULT_VAULT_PATH

_RUNDIR_RE = re.compile(r"_pid(\d+)")


@dataclass
class SourcePaper:
    """The human ground-truth paper a run was generated from."""

    paper_id: int  # the vault papers.id (embedding-space + report anchor namespace)
    title: str = ""
    arxiv_id: str = ""
    doi: str = ""
    problem_definition: str = ""
    challenge: str = ""
    solution: str = ""  # structured solution if available, else derived from full text
    full_text: str = ""
    bbl_content: str = ""
    solution_is_full_text: bool = False  # True when structured solution was empty
    record: Optional[dict] = None  # the paper's results-masked record, if present


@dataclass
class RunContext:
    run_dir: Path
    report_md: str
    problem_md: str
    problem_definition: str  # parsed from problem.md (the agent's actual input)
    challenge: str  # parsed from problem.md
    source: SourcePaper
    papers_dir: Optional[Path] = None  # run_dir/papers, set by load_run


def _load_report_markdown(docs: Path) -> str:
    """The proposal as markdown.

    The report writer emits ``report.json``; it is rendered here so every judge
    sees one prose form of the proposal regardless of how it was produced. A
    plain ``report.md`` is accepted as a fallback for externally generated runs.
    """
    import json

    path = docs / "report.json"
    if path.exists():
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ValueError(f"{path} is not valid JSON: {exc}") from exc
        if isinstance(obj, dict):
            from .render import render_report

            return render_report(obj)

    text = _read(docs / "report.md")
    if not text.strip():
        raise FileNotFoundError(f"no report.json or report.md in {docs}")
    return text


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return ""


def _parse_problem_md(text: str) -> tuple[str, str]:
    """Split ``problem.md`` into (problem_definition, challenge) text blocks.

    Format produced by the harness::

        Problem definition:
        <...>

        Challenge:
        <...>
    """
    pd, ch = "", ""
    # Problem definition: everything between the two headers.
    m = re.search(r"Problem definition:\s*(.*?)(?:\n\s*Challenge:|\Z)", text,
                  re.DOTALL | re.IGNORECASE)
    if m:
        pd = m.group(1).strip()
    m = re.search(r"Challenge:\s*(.*)\Z", text, re.DOTALL | re.IGNORECASE)
    if m:
        ch = m.group(1).strip()
    return pd, ch


def _load_source_fulltext(paper_id: int) -> tuple[str, str]:
    full_text, bbl = "", ""
    if _VAULT.exists():
        con = sqlite3.connect(f"file:{_VAULT}?mode=ro", uri=True)
        try:
            row = con.execute(
                "SELECT full_text, bbl_content FROM paper_full_text "
                "WHERE paper_id=? LIMIT 1",
                (int(paper_id),),
            ).fetchone()
            if row:
                full_text, bbl = row[0] or "", row[1] or ""
        except sqlite3.Error:
            pass
        finally:
            con.close()
    return full_text, bbl


def load_source_paper(paper_id: int) -> SourcePaper:
    """Resolve the source paper's structured fields + full text.

    The structured ``solution`` / ``problem_definition`` may be absent for
    held-out dev papers (their summaries were intentionally not extracted); in
    that case we fall back to the raw ``full_text`` so the rubric / novelty
    judges still have the real method + experiments to compare against.
    """
    fields = get_paper_fields(paper_id) or {}
    full_text, bbl = _load_source_fulltext(paper_id)

    solution = (fields.get("solution") or "").strip()
    solution_is_full_text = False
    if not solution and full_text:
        solution = full_text
        solution_is_full_text = True

    return SourcePaper(
        paper_id=int(paper_id),
        title=fields.get("title", ""),
        arxiv_id=fields.get("arxiv_id", ""),
        doi=fields.get("doi", ""),
        problem_definition=(fields.get("problem_definition") or "").strip(),
        challenge=(fields.get("challenge") or "").strip(),
        solution=solution,
        full_text=full_text,
        bbl_content=bbl,
        solution_is_full_text=solution_is_full_text,
        record=get_paper_record(paper_id),
    )


def _resolve_source_id(
    *, paper_id: Optional[int] = None, arxiv_id: Optional[str] = None
) -> int:
    """Resolve ``papers.id`` from an explicit paper id or arxiv id.

    Used when a run folder is NOT named ``<ts>_pid<P>`` (e.g. ad-hoc
    harness_test runs) and the source paper is given out-of-band.
    """
    if paper_id is not None:
        return int(paper_id)
    if not _VAULT.exists():
        raise FileNotFoundError(f"Idea Vault not found at {_VAULT}")
    con = sqlite3.connect(f"file:{_VAULT}?mode=ro", uri=True)
    try:
        if arxiv_id:
            ax = arxiv_id.strip()
            row = con.execute(
                "SELECT id FROM papers "
                "WHERE arxiv_id LIKE ? OR url LIKE ? LIMIT 1",
                (f"%{ax}%", f"%{ax}%"),
            ).fetchone()
        else:
            raise ValueError("provide paper_id or arxiv_id")
    finally:
        con.close()
    if not row:
        raise ValueError(f"source paper not found (arxiv_id={arxiv_id})")
    return int(row[0])


def load_run(
    run_dir: str | Path,
    *,
    source_paper_id: Optional[int] = None,
    source_arxiv: Optional[str] = None,
) -> RunContext:
    """Load a run folder into a :class:`RunContext`.

    The source paper is taken from the folder name (``<ts>_pid<P>``) when
    present; otherwise pass ``source_paper_id`` or ``source_arxiv`` to
    identify the human ground-truth paper explicitly.
    """
    run_dir = Path(run_dir).resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"run dir not found: {run_dir}")

    m = _RUNDIR_RE.search(run_dir.name)
    if m and source_paper_id is None and not source_arxiv:
        paper_id = int(m.group(1))
    elif source_paper_id is not None or source_arxiv:
        paper_id = _resolve_source_id(
            paper_id=source_paper_id, arxiv_id=source_arxiv
        )
    else:
        raise ValueError(
            f"cannot parse pid from run dir name {run_dir.name!r}; "
            "pass source_paper_id= or source_arxiv= to identify the source paper"
        )

    docs = run_dir / "outputs" if (run_dir / "outputs").is_dir() else run_dir
    report_md = _load_report_markdown(docs)
    problem_md = _read(docs / "problem.md")
    pd, ch = _parse_problem_md(problem_md)

    source = load_source_paper(paper_id)

    ctx = RunContext(
        run_dir=run_dir,
        report_md=report_md,
        problem_md=problem_md,
        problem_definition=pd,
        challenge=ch,
        source=source,
    )
    ctx.papers_dir = docs / "papers"
    return ctx
