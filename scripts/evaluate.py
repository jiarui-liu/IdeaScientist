#!/usr/bin/env python3
"""Evaluate one or more harness run directories.

Produces, per run, the six proposal-quality dimensions, novelty at both scopes
(All and Cutoff), and citation precision and recall — written into
``<run_dir>/evaluation/``.

In-domain novelty and Mechanism non-obviousness are batched separately by
``compute_novelty_aspects.py``; system-level scores are then assembled by
``ideascientist.evaluation.aggregate``.
"""

from __future__ import annotations

import argparse
import logging
import sys
import traceback
from pathlib import Path

from ideascientist.evaluation import evaluate_run

logger = logging.getLogger("evaluate")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", nargs="+", help="harness run output folder(s)")
    ap.add_argument("--model", default=None, help="judge model override")
    ap.add_argument("--no-write", action="store_true",
                    help="print scores without writing evaluation/ into the run folder")
    ap.add_argument("--skip-citations", action="store_true",
                    help="skip citation overlap; use for extra judge passes, where "
                         "it would recompute identical numbers at the highest cost")
    ap.add_argument("--target-arxiv", default=None,
                    help="target paper arXiv id, for runs not named <timestamp>_pid<id>")
    ap.add_argument("--reference-report", type=Path, default=None,
                    help="reference report.json; adds the report_writer reward block, "
                         "which is what the reference_* aggregate family comes from")
    ap.add_argument("--target-paper-id", type=int, default=None,
                    help="target paper vault id, for runs not named <timestamp>_pid<id>")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    exit_code = 0
    for run_dir in args.run_dir:
        try:
            result = evaluate_run(
                run_dir,
                model=args.model,
                write=not args.no_write,
                skip_citations=args.skip_citations,
                source_arxiv=args.target_arxiv,
                source_paper_id=args.target_paper_id,
                reference_report=args.reference_report,
            )
            s = result["summary"]
            print(f"\n=== {run_dir} ===")
            print(f"  quality avg {s['quality_average']} | "
                  f"novelty all/cutoff {s['novelty_all']}/{s['novelty_cutoff']} | "
                  f"relevance {s['relevance']}")
        except Exception as exc:  # noqa: BLE001 - keep going across run dirs
            traceback.print_exc()
            print(f"failed on {run_dir}: {exc}", file=sys.stderr)
            exit_code = 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
