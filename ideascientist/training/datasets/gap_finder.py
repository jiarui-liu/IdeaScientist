#!/usr/bin/env python
"""Build gap_finder GRPO training data from the ground-truth ``gaps.md`` labels.

One JSONL row per (source paper, axis): the orchestrator determines the axes at
generation time, so we replay the axes the label already produced — each axis
becomes a training sample whose reward grades a FRESH gap analysis for THAT axis,
using the validated label's ``gaps.md`` as reference grounding for the judge.

Pipeline:
  1. Index ``logs/gap_labels_devtest_20260714/manifest.jsonl`` once
     (paper_id -> {status, run_dir}) — avoids globbing the whole tree.
  2. For each split id (train/val) with manifest status ``ok``, load its label
     ``outputs/gaps.md``, parse the ``## Axis:`` blocks.
  3. Emit one row per axis: the gap_finder initial prompt (role prompt + this
     axis' task + the exact <tool_call> format block) and metadata carrying
     ``{source_paper_id, axis, problem_definition, reference_gaps_md, ...}``.

Usage:
    python -m ideascientist.training.datasets.gap_finder --n-train 10 --n-val 5 \
        --out data/grpo/gap_finder_grpo
"""
from __future__ import annotations

import argparse
import json
import random
import os
from pathlib import Path


from ideascientist.harness import tools as T

# Reuse the SAME parser the reward uses so training and reward agree on axes.
from ideascientist.rewards.gap_finder.parser import parse_gaps_md

DEFAULT_LABELS = Path(os.environ.get("IDEASCIENTIST_REFERENCES", "data/references")) / "gap_finder"
DEFAULT_SPLITS = Path(os.environ.get("IDEASCIENTIST_SPLITS", "data/splits"))


def _index_manifest(labels_dir: Path) -> dict[int, dict]:
    index: dict[int, dict] = {}
    manifest = labels_dir / "manifest.jsonl"
    with open(manifest) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            pid = row.get("paper_id")
            if pid is None:
                continue
            index[int(pid)] = {
                "status": row.get("status", ""),
                "run_dir": row.get("run_dir", ""),
            }
    return index


def _resolve_run_dir(run_dir: str, labels_dir: Path) -> Path | None:
    if not run_dir:
        return None
    p = Path(run_dir)
    if p.is_dir():
        return p
    # Fall back to matching by basename under labels_dir (handles path remaps).
    cand = labels_dir / p.name
    return cand if cand.is_dir() else None


def _read_problem_challenge(run_dir: Path) -> tuple[str, str]:
    """Extract (challenge, problem_definition) from inputs/problem_challenge.md.

    The label pipeline (generate_gap_labels.get_problem_challenge) handed the
    generator BOTH results-masked fields — ``Challenge:`` (problem_statement +
    why_it_matters + concrete_example) and ``Problem definition:`` (formal_setting
    + inputs + outputs) — written to this file as
    ``Challenge: ...\n\nProblem definition: ...``. Give the trainee the SAME two
    fields so training input matches the label distribution. Abstract-fallback
    labels have neither marker → both come back "".
    """
    pc = run_dir / "inputs" / "problem_challenge.md"
    if not pc.is_file():
        return "", ""
    text = pc.read_text(errors="ignore")
    pd_marker = "Problem definition:"
    ch_marker = "Challenge:"
    problem_definition = ""
    challenge = ""
    if pd_marker in text:
        problem_definition = text.split(pd_marker, 1)[1].strip()
    if ch_marker in text:
        # Challenge runs from its marker up to the problem-definition marker.
        after_ch = text.split(ch_marker, 1)[1]
        challenge = after_ch.split(pd_marker, 1)[0].strip()
    return challenge, problem_definition


def _problem_context(challenge: str, problem_definition: str) -> str:
    """Combine the results-masked fields into the SAME string production hands the
    subagent as ``ctx.query_text`` (``Challenge: ...\\n\\nProblem definition: ...``).

    Mirrors ``generate_gap_labels.get_problem_challenge`` so training and the
    label pipeline agree on the problem framing.
    """
    parts: list[str] = []
    if challenge:
        parts.append(f"Challenge: {challenge}")
    if problem_definition:
        parts.append(f"Problem definition: {problem_definition}")
    return "\n\n".join(parts)


def _build_initial_prompt(axis: str, challenge: str,
                          problem_definition: str) -> str:
    """gap_finder initial USER turn for ONE axis, byte-identical to what the
    production ``_run_subagent`` builds for the subagent.

    In production the subagent's user turn is
    ``build_subagent_user_message(task, ctx.query_text)`` = problem context
    (challenge + problem_definition) followed by the orchestrator's task (the
    axis). The role prompt + tool-call format block live in the SYSTEM turn
    (added by the data processor), NOT here — so we do NOT restate them.
    """
    problem = _problem_context(challenge, problem_definition)
    return T.build_subagent_user_message(axis, problem)


def _read_split_ids(path: Path) -> list[int]:
    ids: list[int] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line.isdigit():
                ids.append(int(line))
    return ids


def build_split(ids: list[int], index: dict[int, dict], labels_dir: Path,
                limit: int) -> list[dict]:
    """Build rows for a split, one per (paper, axis), until `limit` papers used."""
    rows: list[dict] = []
    n_papers = 0
    for pid in ids:
        if n_papers >= limit:
            break
        entry = index.get(pid)
        if not entry or entry.get("status") != "ok":
            continue
        run_dir = _resolve_run_dir(entry.get("run_dir", ""), labels_dir)
        if run_dir is None:
            continue
        gaps_path = run_dir / "outputs" / "gaps.md"
        if not gaps_path.is_file():
            continue
        ref_md = gaps_path.read_text(errors="ignore")
        axes = parse_gaps_md(ref_md)
        if not axes:
            continue
        challenge, problem_definition = _read_problem_challenge(run_dir)
        used = False
        for ax in axes:
            if not ax.name.strip():
                continue
            prompt = _build_initial_prompt(
                ax.name, challenge, problem_definition,
            )
            metadata = {
                "source_paper_id": pid,
                "axis": ax.name,
                "challenge": challenge,
                "problem_definition": problem_definition,
                # Reference grounding: the WHOLE validated gaps.md (judge sees the
                # coverage bar; reward does NOT require verbatim match).
                "reference_gaps_md": ref_md,
                "reference_cited_ids": sorted(
                    {i for g in ax.gaps for i in g.cited_ids}
                ),
            }
            rows.append({"input": prompt, "output": "", "metadata": metadata})
            used = True
        if used:
            n_papers += 1
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Build gap_finder GRPO data from ground-truth gaps.md labels"
    )
    parser.add_argument("--labels", type=str, default=str(DEFAULT_LABELS))
    parser.add_argument("--splits", type=str, default=str(DEFAULT_SPLITS))
    parser.add_argument(
        "--out", type=str,
        default=str(Path("data") / "grpo" / "gap_finder"),
    )
    parser.add_argument("--n-train", type=int, default=10,
                        help="Number of TRAIN papers to include (axes expand rows).")
    parser.add_argument("--n-val", type=int, default=5,
                        help="Number of VAL papers to include.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    labels_dir = Path(args.labels)
    splits_dir = Path(args.splits)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Indexing manifest under {labels_dir} ...")
    index = _index_manifest(labels_dir)
    n_ok = sum(1 for v in index.values() if v.get("status") == "ok")
    print(f"  indexed {len(index)} papers ({n_ok} ok)")

    train_ids = _read_split_ids(splits_dir / "train.txt")
    val_ids = _read_split_ids(splits_dir / "val.txt")
    # Shuffle deterministically so the "first N ok" pick is stable but not just
    # the lowest ids.
    random.seed(args.seed)
    random.shuffle(train_ids)
    random.shuffle(val_ids)

    train_rows = build_split(train_ids, index, labels_dir, args.n_train)
    val_rows = build_split(val_ids, index, labels_dir, args.n_val)

    for name, rows in [("train", train_rows), ("validation", val_rows)]:
        path = out_dir / f"{name}.jsonl"
        with open(path, "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        n_papers = len({r["metadata"]["source_paper_id"] for r in rows})
        print(f"Wrote {len(rows)} rows ({n_papers} papers) to {path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
