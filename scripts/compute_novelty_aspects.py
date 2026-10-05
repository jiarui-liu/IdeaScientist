#!/usr/bin/env python3
"""Score In-domain novelty and Mechanism non-obviousness over a system's runs.

These two metrics are batched across a whole system rather than computed per
run, because the non-obviousness derivation call depends only on the target
paper — every system evaluated on the same paper shares it, and re-deriving per
system would both waste calls and let the derivations disagree.

Both metrics compare against work published strictly before the *target paper*,
not before a single global date. A per-paper cutoff is the well-defined version
of the question: it asks what was derivable when that paper was written.

Results are checkpointed to JSONL as they land, so a run is resumable.
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
from pathlib import Path
from typing import Any, Optional

from ideascientist.evaluation import novelty_aspects as NA
from ideascientist.evaluation.llm_judge import JUDGE_EXTRA_BODY
from ideascientist.utils.db import DEFAULT_VAULT_PATH
from ideascientist.utils.llm import LLMSettings, chat

logger = logging.getLogger("novelty_aspects")

# Prior work shown to each judge. The derivation call sees the mechanism-side
# neighbours; a wider pool makes the derivation stronger but the comparison
# noisier, so it stays small and fixed.
N_PRIOR_MECHANISM = 3


def load_proposals(base: Path) -> dict[int, dict[str, Any]]:
    """``{target paper id: report}`` for every scorable run under ``base``."""
    out: dict[int, dict[str, Any]] = {}
    for report_path in sorted(Path(base).glob("runs/*/outputs/report.json")):
        try:
            report = json.loads(report_path.read_text())
        except (OSError, ValueError):
            continue
        if not isinstance(report, dict) or not NA.proposal_mechanism_block(report).strip():
            continue
        evaluation = report_path.parents[1] / "evaluation" / "evaluation.json"
        paper_id = None
        if evaluation.is_file():
            try:
                meta = json.loads(evaluation.read_text()).get("target_paper") or {}
                paper_id = meta.get("paper_id")
            except (OSError, ValueError):
                paper_id = None
        if paper_id is not None:
            out[int(paper_id)] = report
    return out


def target_problem(db: Path, paper_id: int) -> tuple[str, str]:
    """The target paper's problem definition and challenge."""
    from ideascientist.harness.corpus import get_paper_fields

    fields = get_paper_fields(paper_id) or {}
    return fields.get("problem_definition", ""), fields.get("challenge", "")


def prior_work(db: Path, paper_id: int, k: int) -> list[dict[str, Any]]:
    """The ``k`` nearest papers published strictly before the target paper.

    Ranked in the mechanism spaces (intuition and solution), since the question
    is whether the proposal's mechanism was already available, not whether the
    problem was already studied.
    """
    import numpy as np

    from ideascientist.harness.corpus import _field_matrices, get_paper_fields

    g = _field_matrices()
    row_of = g["row_of_pid"]
    if paper_id not in row_of:
        return []
    target_row = row_of[paper_id]

    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        row = con.execute("SELECT date FROM papers WHERE id = ?", (paper_id,)).fetchone()
    finally:
        con.close()
    target_date = str(row[0])[:10] if row and row[0] else ""
    if not target_date:
        return []

    scores = np.zeros(len(g["pid"]), dtype=np.float32)
    for space in ("intuition", "solution"):
        if space in g:
            scores += g[space] @ g[space][target_row]

    order = np.argsort(-scores)
    picked: list[dict[str, Any]] = []
    for idx in order:
        pid = int(g["pid"][idx])
        if pid == paper_id:
            continue
        if not g["dates"][idx] or str(g["dates"][idx])[:10] >= target_date:
            continue
        fields = get_paper_fields(pid)
        if fields:
            picked.append(fields)
        if len(picked) >= k:
            break
    return picked


def _ask(messages: list[dict[str, str]], settings: LLMSettings, max_tokens: int) -> str:
    return chat(messages, settings=settings, temperature=0.2, max_tokens=max_tokens,
                extra_body=JUDGE_EXTRA_BODY)


def score_paper(
    report: dict[str, Any],
    problem_definition: str,
    challenge: str,
    priors: list[dict[str, Any]],
    settings: LLMSettings,
    *,
    cached_candidates: Optional[list[dict[str, Any]]] = None,
) -> dict[str, Any]:
    """Score one proposal on both aspect metrics.

    ``cached_candidates`` is the proposal-blind derivation for this target
    paper, reused across systems. It is only ever produced by a call that has
    not seen any proposal.
    """
    mechanism = NA.proposal_mechanism_block(report)
    priors_block = "\n\n".join(NA.prior_paper_block(p) for p in priors)

    in_domain = NA.parse_in_domain(
        _ask(NA.build_in_domain_messages(mechanism), settings, 900), mechanism
    )

    candidates = cached_candidates
    if candidates is None:
        candidates = NA.parse_candidates(
            _ask(
                NA.build_derive_messages(problem_definition, challenge, priors_block),
                settings, 2000,
            )
        )

    non_obvious = None
    if candidates:
        non_obvious = NA.parse_non_obviousness(
            _ask(
                NA.build_grade_messages(candidates, priors_block, mechanism),
                settings, 1200,
            )
        )
    return {"in_domain": in_domain, "non_obviousness": non_obvious, "candidates": candidates}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path, help="a system's run directory")
    ap.add_argument("--db", type=Path, default=DEFAULT_VAULT_PATH)
    ap.add_argument("--out", type=Path, default=Path("results/novelty_aspects.jsonl"))
    ap.add_argument("--denominator", type=int, default=277,
                    help="test split size; an unscored proposal counts as zero")
    ap.add_argument("--model", default=None, help="judge model override")
    ap.add_argument("--candidates", type=Path, default=None,
                    help="JSONL of proposal-blind derivations to reuse across systems")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    settings = LLMSettings.from_env(model=args.model)

    shared: dict[int, list[dict[str, Any]]] = {}
    if args.candidates and args.candidates.is_file():
        for line in args.candidates.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                shared[int(rec["paper_id"])] = rec["candidates"]
        logger.info("reusing %d cached derivations", len(shared))

    done: set[int] = set()
    if args.out.is_file():
        for line in args.out.read_text().splitlines():
            if line.strip():
                done.add(int(json.loads(line)["paper_id"]))
        logger.info("resuming; %d already scored", len(done))

    proposals = load_proposals(args.run_dir)
    logger.info("%d scorable proposals in %s", len(proposals), args.run_dir)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    in_domain_verdicts, non_obvious_verdicts = [], []
    with args.out.open("a") as fh:
        for paper_id, report in sorted(proposals.items()):
            if paper_id in done:
                continue
            pd_text, challenge = target_problem(args.db, paper_id)
            priors = prior_work(args.db, paper_id, N_PRIOR_MECHANISM)
            try:
                scored = score_paper(report, pd_text, challenge, priors, settings,
                                     cached_candidates=shared.get(paper_id))
            except Exception as exc:  # noqa: BLE001 - one paper must not stop the run
                logger.warning("paper %s failed: %s", paper_id, exc)
                continue
            fh.write(json.dumps({"paper_id": paper_id, **scored}) + "\n")
            fh.flush()
            in_domain_verdicts.append(scored["in_domain"])
            non_obvious_verdicts.append(scored["non_obviousness"])

    # Both aggregates report n_scored / n_unscored / denominator, so they are
    # kept in separate blocks: a proposal can be scored on one metric and not
    # the other, and flattening would silently report one metric's counts as
    # though they were the other's.
    print(json.dumps({
        "in_domain": NA.aggregate_in_domain(in_domain_verdicts, args.denominator),
        "non_obviousness": NA.aggregate_non_obviousness(non_obvious_verdicts, args.denominator),
    }, indent=2))


if __name__ == "__main__":
    main()
