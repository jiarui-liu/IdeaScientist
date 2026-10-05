#!/usr/bin/env python3
"""Run the ideation harness on one research problem.

The ``--problem`` string is the input the orchestrator receives. For a run that
reproduces the evaluation setup it carries both fields the producer roles are
given, in the form the harness prepends to every producer task::

    Challenge: ...

    Problem definition: ...

Run directory layout::

    runs/<timestamp>_<slug>/
        outputs/
            problem.md      the input
            plan.md         orchestrator's research plan
            papers/<id>.md  per-paper reader digests
            gaps.md         gap finder
            candidates.md   innovator
            scores.md       reviewer
            report.json     report writer
        logs/
            harness_log.json, llm_calls.json, _sublogs/, summary.json
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import re
from datetime import datetime
from pathlib import Path

from ideascientist.harness import run_ideation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_ideation")

_STOP_WORDS = frozenset(
    "a an and are as at be been being but by can could do does did for from "
    "had has have he her his how i if in into is it its may me my no nor not "
    "of on or our out over own she should so some such that the their them "
    "then there these they this to too up us was we were what when where "
    "which while who whom why will with would you your about above after "
    "again all also any because before below between both during each few "
    "further get got here let like make many more most must only other same "
    "still through under until used using improve improved improving best "
    "better effective approach method technique based new novel propose "
    "proposed".split()
)


def _slug(text: str) -> str:
    words = re.sub(r"[^a-z0-9]+", " ", text.lower()).split()
    keywords = [w for w in words if w not in _STOP_WORDS and len(w) > 1][:4]
    return "_".join(keywords + [hashlib.md5(text.encode()).hexdigest()[:4]])


async def _run(args: argparse.Namespace) -> None:
    problem = Path(args.problem_file).read_text(encoding="utf-8") if args.problem_file else args.problem
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.run_dir) / f"{ts}_{args.problem_id or _slug(problem)}"
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info("run directory: %s", run_dir)


    kwargs = {"max_turns": args.max_turns, "max_sub_turns": args.max_sub_turns}
    if args.model:
        kwargs["model"] = args.model
    if args.query_paper_id:
        kwargs["query_paper_id"] = args.query_paper_id

    summary = await run_ideation(problem, run_dir, **kwargs)

    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    (logs_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    logger.info("run complete -> %s", run_dir)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--problem", help="the research problem definition and challenge")
    src.add_argument("--problem-file", help="read the problem from a file instead")
    p.add_argument("--problem-id", default="", help="override the auto-derived run slug")
    p.add_argument("--query-paper-id", type=int, default=0,
                   help="vault id of the target paper, used to exclude it from retrieval")
    p.add_argument("--model", default="", help="model override (default: IDEASCIENTIST_MODEL)")
    p.add_argument("--run-dir", default="runs", help="base directory for run outputs")
    p.add_argument("--max-turns", type=int, default=100,
                   help="orchestrator turn cap (paper setting: 100)")
    p.add_argument("--max-sub-turns", type=int, default=40,
                   help="per-sub-agent tool-use turn cap; per-role caps in tools.py bind first")
    asyncio.run(_run(p.parse_args()))


if __name__ == "__main__":
    main()
