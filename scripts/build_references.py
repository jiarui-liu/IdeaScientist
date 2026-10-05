#!/usr/bin/env python3
"""Generate the per-stage reference artifacts used as training targets.

One privileged pass per target paper per stage. Unlike the production roles,
the generator is given the target paper's full text and works backwards from
it — but it must still ground every citation in papers it actually read through
the same environment, so its output passes the same deterministic gates the
reward applies. That is what keeps the references in-distribution for the
policy being trained.

Run the stages in order: each consumes the previous stage's artifact.

    python scripts/build_references.py gap_finder
    python scripts/build_references.py innovator
    python scripts/build_references.py report_writer
"""

from __future__ import annotations

import argparse
import runpy
import sys

ROLES = ("gap_finder", "innovator", "report_writer")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=ROLES)
    args, rest = ap.parse_known_args()

    module = f"ideascientist.vault.references.{args.stage}"
    sys.argv = [module, *rest]
    runpy.run_module(module, run_name="__main__")


if __name__ == "__main__":
    main()
