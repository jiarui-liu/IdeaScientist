#!/usr/bin/env python
"""Build innovator GRPO training data from the ground-truth ``candidates.md`` labels.

One JSONL row per (source paper, candidate): each validated label candidate
(``### C<n>`` block) attacks a NAMED gap, so we replay that gap as the trainee's
task — the reward grades a FRESH single-candidate ideation for THAT gap, using the
validated label candidate as the judge's reference anchor and the whole label
``gaps.md`` as the input doc the trainee views via edit_doc.

Pipeline:
  1. Index ``logs/innovator_labels_devtest_20260714/manifest.jsonl`` once
     (paper_id -> {status, run_dir}) — avoids globbing the whole tree.
  2. For each split id (train/val) with manifest status ``ok``, load its label
     ``outputs/candidates.md`` (parse ``### C<n>`` blocks) and ``outputs/gaps.md``.
  3. Emit one row per candidate: the innovator initial prompt (problem context +
     the gap that candidate attacks as the task) and metadata carrying
     ``{source_paper_id, gap, problem_definition, reference_gaps_md,
     reference_candidates_md, reference_cited_ids}``.

Usage:
    python -m ideascientist.training.datasets.innovator --n-train 10 --n-val 5 \
        --out data/grpo/innovator_grpo
"""
from __future__ import annotations

import argparse
import json
import random
import re
import os
from pathlib import Path


from ideascientist.harness import tools as T

# Reuse the SAME parser the reward uses so training and reward agree on candidates.
from ideascientist.rewards.innovator.parser import parse_candidates_md

DEFAULT_LABELS = Path(os.environ.get("IDEASCIENTIST_REFERENCES", "data/references")) / "innovator"
DEFAULT_SPLITS = Path(os.environ.get("IDEASCIENTIST_SPLITS", "data/splits"))


def _strip_lead_paren(s: str) -> str:
    """Strip a leading ``(...)`` instructional parenthetical the label template
    embeds on the field label line (e.g. ``(name the SPECIFIC gap ...)``), which
    the candidates parser folds onto the front of the field value."""
    return re.sub(r"^\s*\([^)]*\)\s*", "", s or "").strip()


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
    cand = labels_dir / p.name
    return cand if cand.is_dir() else None


def _read_problem_challenge(run_dir: Path) -> tuple[str, str]:
    """Extract (challenge, problem_definition) from inputs/problem_challenge.md.

    Same results-masked two-field format the label pipeline wrote
    (``Challenge: ...\n\nProblem definition: ...``). Abstract-fallback labels have
    neither marker -> both come back "".
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
        after_ch = text.split(ch_marker, 1)[1]
        challenge = after_ch.split(pd_marker, 1)[0].strip()
    return challenge, problem_definition


def _problem_context(challenge: str, problem_definition: str) -> str:
    """Combine the results-masked fields into the SAME string production hands the
    subagent as ``ctx.query_text`` (``Challenge: ...\\n\\nProblem definition: ...``)."""
    parts: list[str] = []
    if challenge:
        parts.append(f"Challenge: {challenge}")
    if problem_definition:
        parts.append(f"Problem definition: {problem_definition}")
    return "\n\n".join(parts)


def _build_initial_prompt(gap: str, challenge: str,
                          problem_definition: str) -> str:
    """innovator initial USER turn for ONE gap, byte-identical to what production
    ``_run_subagent`` builds: ``build_subagent_user_message(task, ctx.query_text)``
    = problem context (challenge + problem_definition) followed by the task (here
    the gap the candidate attacks). The role prompt + tool-call format block live
    in the SYSTEM turn (added by the data processor), NOT here."""
    problem = _problem_context(challenge, problem_definition)
    return T.build_subagent_user_message(gap, problem)


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
    """Build rows for a split, one per (paper, candidate), until `limit` papers used."""
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
        cands_path = run_dir / "outputs" / "candidates.md"
        gaps_path = run_dir / "outputs" / "gaps.md"
        if not cands_path.is_file() or not gaps_path.is_file():
            continue
        cands_md = cands_path.read_text(errors="ignore")
        ref_gaps_md = gaps_path.read_text(errors="ignore")
        candidates = parse_candidates_md(cands_md)
        if not candidates:
            continue
        challenge, problem_definition = _read_problem_challenge(run_dir)
        used = False
        for c in candidates:
            gap = _strip_lead_paren(c.fields.get("Gap it attacks") or "")
            if not gap:
                continue
            prompt = _build_initial_prompt(gap, challenge, problem_definition)
            metadata = {
                "source_paper_id": pid,
                "gap": gap,
                "challenge": challenge,
                "problem_definition": problem_definition,
                # Seeds gaps.md into the rollout's files (input the trainee views
                # via edit_doc) AND anchors the gap_exists gate.
                "reference_gaps_md": ref_gaps_md,
                # The validated label candidate for THIS gap: the judge's scoring
                # anchor (reward does NOT require verbatim match).
                "reference_candidates_md": c.raw,
                "reference_cited_ids": sorted(set(c.cited_ids)),
            }
            rows.append({"input": prompt, "output": "", "metadata": metadata})
            used = True
        if used:
            n_papers += 1
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Build innovator GRPO data from ground-truth candidates.md labels"
    )
    parser.add_argument("--labels", type=str, default=str(DEFAULT_LABELS))
    parser.add_argument("--splits", type=str, default=str(DEFAULT_SPLITS))
    parser.add_argument(
        "--out", type=str,
        default=str(Path("data") / "grpo" / "innovator"),
    )
    parser.add_argument("--n-train", type=int, default=10,
                        help="Number of TRAIN papers to include (candidates expand rows).")
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
