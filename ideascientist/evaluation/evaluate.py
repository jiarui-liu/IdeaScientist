"""Evaluate one harness run and persist the result.

Loads the run, retrieves the two comparison sets novelty is scored against,
runs each metric family, and writes ``<run_dir>/evaluation/evaluation.json``
plus a human-readable summary.

Two metrics are reported at two scopes, to separate absolute quality from what
was actually achievable:

* **novelty** against the whole corpus (All) and against pre-cutoff work only
  (Cutoff) — the agent could only ever retrieve the latter;
* **citation precision and recall** against every ground-truth citation, and
  against only those that are in the vault and pre-cutoff, which is the subset
  the agent could have cited at all.

In-domain novelty and Mechanism non-obviousness are not computed here. They are
batched per target paper across systems by
``scripts/compute_novelty_aspects.py``, because their proposal-blind derivation
call is shared.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Optional

from ideascientist.harness.corpus import CUTOFF_DATE

from . import novelty as novelty_metric
from . import quality as quality_metric
from . import reference_grounded as reference_metric
from .citations import evaluate_citations, keyed_report_cited
from .llm_judge import DEFAULT_JUDGE_MODEL
from .loaders import RunContext, load_run
from .similar_papers import SimilarPaper, find_precutoff_papers, find_similar_papers

logger = logging.getLogger(__name__)


def _similar_to_dict(s: SimilarPaper) -> dict[str, Any]:
    return {
        "paper_id": s.paper_id,
        "title": s.title,
        "challenge_score": s.challenge_score,
        "problem_definition_score": s.problem_definition_score,
        "sum_score": s.sum_score,
        "selected_by": s.selected_by,
        "arxiv_id": s.arxiv_id,
        "has_solution": bool((s.solution or "").strip()),
    }


def _report_object(run_dir: Path) -> Optional[dict]:
    docs = run_dir / "outputs" if (run_dir / "outputs").is_dir() else run_dir
    try:
        obj = json.loads((docs / "report.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return obj if isinstance(obj, dict) else None


def _report_cited_ids(run_dir: Path) -> list:
    docs = run_dir / "outputs" if (run_dir / "outputs").is_dir() else run_dir
    path = docs / "report.json"
    obj = None
    if path.exists():
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            obj = None
    cited, _ = keyed_report_cited(obj)
    return cited


def evaluate_run(
    run_dir: str | Path,
    *,
    model: Optional[str] = None,
    write: bool = True,
    cutoff_date: str = CUTOFF_DATE,
    source_paper_id: Optional[int] = None,
    source_arxiv: Optional[str] = None,
    skip_citations: bool = False,
    reference_report: Optional[Path] = None,
) -> dict[str, Any]:
    """Evaluate one run folder and optionally persist the result.

    For run folders not named ``<timestamp>_pid<id>``, pass ``source_paper_id``
    or ``source_arxiv`` to identify the held-out target paper.

    ``skip_citations`` exists for cross-judge passes: citation overlap does not
    depend on the judge, and it is the database-bound step, so recomputing it
    once per judge wastes the whole budget for identical numbers.
    """
    model = model or DEFAULT_JUDGE_MODEL
    ctx = load_run(run_dir, source_paper_id=source_paper_id, source_arxiv=source_arxiv)
    target = ctx.source

    logger.info("evaluating %s against target paper %s (%s)",
                ctx.run_dir, target.paper_id, target.title)

    comparison_all = find_similar_papers(
        target.paper_id,
        challenge_fallback_text=target.challenge or ctx.challenge,
    )
    comparison_cutoff = find_precutoff_papers(target.paper_id)
    logger.info("comparison sets: %d (all), %d (pre-cutoff)",
                len(comparison_all), len(comparison_cutoff))

    quality = quality_metric.score_proposal_quality(ctx.report_md, model=model)
    relevance = quality_metric.score_relevance(
        ctx.problem_definition, ctx.challenge, ctx.report_md, model=model
    )
    novelty_all = novelty_metric.score_novelty(ctx.report_md, comparison_all, model=model)
    novelty_cutoff = novelty_metric.score_novelty(ctx.report_md, comparison_cutoff, model=model)

    report_obj = _report_object(ctx.run_dir)
    reference_grounded = reference_metric.score_reference_grounded(
        report_obj, target, comparison_cutoff, model=model
    ) if report_obj else {}

    if skip_citations:
        citations = {"skipped": True}
    else:
        citations = evaluate_citations(
            ctx.report_md, ctx.papers_dir, target.paper_id,
            comparison_all, comparison_cutoff, cutoff_date=cutoff_date,
            report_cited_override=_report_cited_ids(ctx.run_dir),
        )

    results: dict[str, Any] = {
        "run_dir": str(ctx.run_dir),
        "evaluated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "judge_model": model,
        "cutoff_date": cutoff_date,
        "target_paper": {
            "paper_id": target.paper_id,
            "title": target.title,
            "arxiv_id": target.arxiv_id,
            "doi": target.doi,
            "structured_record_available": not target.solution_is_full_text,
        },
        "comparison_papers": {
            "all": [_similar_to_dict(s) for s in comparison_all],
            "cutoff": [_similar_to_dict(s) for s in comparison_cutoff],
        },
        "quality": {**quality, "relevance": relevance},
        "novelty": {"all": novelty_all, "cutoff": novelty_cutoff},
        "reference_grounded": reference_grounded,
        "citations": citations,
        "summary": _summary(quality, relevance, novelty_all, novelty_cutoff, citations),
    }

    if reference_report is not None:
        results["report_writer_reward"] = score_report_writer_reward(
            ctx, Path(reference_report).read_text(encoding="utf-8"), model=model
        )

    if write:
        out_dir = ctx.run_dir / "evaluation"
        out_dir.mkdir(exist_ok=True)
        (out_dir / "evaluation.json").write_text(
            json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (out_dir / "evaluation.md").write_text(
            render_markdown(results, ctx), encoding="utf-8"
        )
        logger.info("wrote %s", out_dir / "evaluation.json")

    return results


def score_report_writer_reward(
    ctx: RunContext,
    reference_report_json: str,
    *,
    model: Optional[str] = None,
) -> dict[str, Any]:
    """Score a run's report with the report_writer training reward.

    Run against a reference report, this is what produces the per-field
    ``matches_reference_*`` satisfactions that
    :func:`ideascientist.evaluation.aggregate.report_metrics` turns into the
    ``reference_*`` family. It is the training objective applied to an
    evaluation run, so a system is comparable against the checkpoint that was
    trained on it.
    """
    from ideascientist.rewards.report_writer.reward import compute_reward
    from ideascientist.harness.corpus import get_paper_biblio
    from ideascientist.utils.llm import LLMSettings

    outputs = ctx.run_dir / "outputs"
    files: dict[str, str] = {}
    for name in ("report.json", "gaps.md", "candidates.md"):
        p = outputs / name
        if p.exists():
            files[name] = p.read_text(encoding="utf-8")
    papers_dir = outputs / "papers"
    if papers_dir.is_dir():
        # The key keeps the ".md" suffix: the reward's cited-ids-were-read gate
        # looks up "papers/<id>.md" exactly.
        for md in papers_dir.glob("*.md"):
            files[f"papers/{md.name}"] = md.read_text(encoding="utf-8")
    if "report.json" not in files:
        return {"scored": False, "error": "no report.json"}

    res = compute_reward(
        files,
        {
            "problem": ctx.problem_definition,
            "challenge": ctx.challenge,
            "reference_report_json": reference_report_json,
            "reference_candidates_md": files.get("candidates.md", ""),
        },
        LLMSettings.from_env(model=model or DEFAULT_JUDGE_MODEL),
        biblio_fn=get_paper_biblio,
    )
    return {
        "scored": True,
        "total": res.total,
        "rubric_reward": res.rubric_reward,
        "process_reward": res.process_reward,
        "citation_f1": res.citation_f1,
        "gated": res.gated,
        "gates": res.gates,
        "judge_failed": res.judge_failed,
        "reward_breakdown": res.reward_breakdown,
        "error": res.error or None,
    }


def _summary(quality, relevance, novelty_all, novelty_cutoff, citations) -> dict[str, Any]:
    s: dict[str, Any] = {
        dim: quality[dim]["score"] for dim in quality_metric.DIMENSIONS
    }
    s["relevance"] = relevance["score"]
    s["quality_average"] = quality["average"]
    s["novelty_all"] = novelty_all["score"]
    s["novelty_cutoff"] = novelty_cutoff["score"]

    for label, key in (("source", "precision_recall_vs_source_citations"),
                       ("union", "precision_recall_vs_union")):
        block = citations.get(key)
        if not block:
            continue
        for scope in ("all", "citable_pre_cutoff"):
            v = block[scope]
            s[f"citation_precision_{label}_{scope}"] = v["precision"]
            s[f"citation_recall_{label}_{scope}"] = v["recall"]
            s[f"citation_f1_{label}_{scope}"] = v["f1"]
    return s


def _row(label: str, entry: dict[str, Any]) -> str:
    score = entry.get("score")
    shown = f"{score}" if score is not None else ("n/a" if entry.get("applicable") is False else "—")
    return f"| {label} | {shown} | {entry.get('reasoning', '')} |"


def _pr_line(v: dict[str, Any]) -> str:
    return (f"precision **{v['precision']}**, recall **{v['recall']}**, f1 **{v['f1']}** "
            f"(cited {v['num_report_cited']}, ground truth {v['num_ground_truth']}, "
            f"matched {v['num_true_positives']})")


def render_markdown(results: dict[str, Any], ctx: RunContext) -> str:
    quality = results["quality"]
    novelty = results["novelty"]
    citations = results["citations"]
    target = results["target_paper"]

    L: list[str] = [
        f"# Evaluation — {ctx.run_dir.name}",
        "",
        f"- **Judge:** `{results['judge_model']}`",
        f"- **Evaluated at:** {results['evaluated_at']}",
        f"- **Cutoff:** {results['cutoff_date']}",
        f"- **Target paper:** `{target['paper_id']}` — *{target['title']}* "
        f"(arXiv: {target['arxiv_id'] or 'n/a'})",
        f"- **Comparison papers:** {len(results['comparison_papers']['all'])} (all), "
        f"{len(results['comparison_papers']['cutoff'])} (pre-cutoff)",
        "",
        "## Proposal quality",
        "",
        "| Dimension | Score (1–10) | Reasoning |",
        "|---|---|---|",
    ]
    for dim in quality_metric.DIMENSIONS:
        L.append(_row(dim, quality[dim]))
    L.append(_row("relevance", quality["relevance"]))
    L.append(f"| **Average** | **{quality['average']}** | |")
    L += [
        "",
        "## Novelty",
        "",
        f"- **All** (whole corpus): **{novelty['all']['score']}** — {novelty['all']['reasoning']}",
        f"- **Cutoff** (pre-cutoff work only): **{novelty['cutoff']['score']}** "
        f"— {novelty['cutoff']['reasoning']}",
        "",
    ]

    if not citations.get("skipped"):
        L += [
            "## Citation coverage",
            "",
            "_Two scopes per ground-truth set: **all** is every cited work; "
            f"**citable** is only work in the vault published before {results['cutoff_date']}, "
            "which is what the agent could have cited._",
            "",
        ]
        for label, key in (
            ("vs. the target paper's citations", "precision_recall_vs_source_citations"),
            ("vs. those citations plus the comparison set", "precision_recall_vs_union"),
        ):
            L += [f"### {label}",
                  f"- **all:** {_pr_line(citations[key]['all'])}",
                  f"- **citable:** {_pr_line(citations[key]['citable_pre_cutoff'])}",
                  ""]

    L.append("> Full scores, per-metric reasoning, and the citation sets are in "
             "`evaluation.json`.")
    L.append("")
    return "\n".join(L)
