#!/usr/bin/env python3
"""Launch GRPO training for one sub-agent role.

Requires a NeMo-RL checkout on the path; see the README. The shipped configs
reproduce the setup reported in the paper, so overrides here are for adapting
to different hardware rather than changing the method.

Build the role's dataset first:

    python -m ideascientist.training.datasets.gap_finder
    python scripts/train_role.py gap_finder
"""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path

ROLES = ("gap_finder", "innovator", "report_writer")
CONFIG_DIR = Path(__file__).resolve().parents[1] / "ideascientist" / "training" / "configs"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("role", choices=ROLES)
    ap.add_argument("--config", type=Path, default=None,
                    help="override the shipped config for this role")
    args, overrides = ap.parse_known_args()

    config = args.config or CONFIG_DIR / f"{args.role}.yaml"
    if not config.is_file():
        raise SystemExit(f"config not found: {config}")

    # run_grpo takes the role positionally and forwards the rest to its own parser.
    sys.argv = ["ideascientist.training.run_grpo", args.role,
                "--config", str(config), *overrides]
    runpy.run_module("ideascientist.training.run_grpo", run_name="__main__")


if __name__ == "__main__":
    main()
