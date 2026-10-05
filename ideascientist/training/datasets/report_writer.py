#!/usr/bin/env python
"""Build report_writer GRPO training data from the ground-truth ``report.json`` labels.

One JSONL row per source paper: the report_writer develops the SURVIVING intuition
(the chosen candidate, typically ``C1`` in ``candidates.md``) attacking the assigned
gap into a complete results-masked solution proposal (``report.json``). We replay that
task — the reward grades a FRESH report against the validated label report as the
judge's reference anchor, and the label ``gaps.md`` + ``candidates.md`` are seeded
into the rollout's files as the input docs the trainee views via edit_doc.

Pipeline:
  1. Index ``logs/report_labels_devtest_20260716/manifest.jsonl`` once
     (paper_id -> {status, run_dir}) — avoids globbing the whole tree.
  2. For each split id (train/val) with manifest status ``ok``, load its label
     ``outputs/report.json``, ``outputs/gaps.md``, ``outputs/candidates.md``.
  3. Emit one row per paper: the report_writer initial prompt (problem context +
     the surviving intuition as the task) and metadata carrying ``{source_paper_id,
     problem, gap, intuition, reference_report_json, reference_gaps_md,
     reference_candidates_md, reference_cited_ids}``.

Usage:
    python -m ideascientist.training.datasets.report_writer --n-train 10 --n-val 5 \
        --out data/grpo/report_writer_grpo
"""
from __future__ import annotations

import argparse
import json
import random
import re
import os
from pathlib import Path


from ideascientist.harness import tools as T

# Reuse the SAME parsers the reward uses so training and reward agree.
# candidates_parser lives ONLY in the innovator role dir; this is a standalone
# offline builder (not the RL env), so the cross-role import-collision risk that
# affects the live envs does not apply here.
from ideascientist.rewards.innovator.parser import parse_candidates_md
from ideascientist.rewards.report_writer.parser import parse_report_text

DEFAULT_LABELS = Path(os.environ.get("IDEASCIENTIST_REFERENCES", "data/references")) / "report_writer"
DEFAULT_SPLITS = Path(os.environ.get("IDEASCIENTIST_SPLITS", "data/splits"))


def _strip_lead_paren(s: str) -> str:
    """Strip a leading ``(...)`` instructional parenthetical the label template
    embeds on the field label line, folded onto the field value by the parser."""
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
    """``Challenge: ...\\n\\nProblem definition: ...`` — the SAME query_text string
    production hands the subagent."""
    parts: list[str] = []
    if challenge:
        parts.append(f"Challenge: {challenge}")
    if problem_definition:
        parts.append(f"Problem definition: {problem_definition}")
    return "\n\n".join(parts)


def _build_initial_prompt(intuition: str, challenge: str,
                          problem_definition: str) -> str:
    """report_writer initial USER turn, byte-identical to what production
    ``_run_subagent`` builds: ``build_subagent_user_message(task, ctx.query_text)``
    = problem context (challenge + problem_definition) followed by the task (here
    developing the surviving intuition into a full results-masked proposal). The role
    prompt + tool-call block live in the SYSTEM turn (added by the data processor)."""
    problem = _problem_context(challenge, problem_definition)
    return T.build_subagent_user_message(intuition, problem)


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
    """Build rows for a split, one per paper, until `limit` papers used."""
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
        report_path = run_dir / "outputs" / "report.json"
        gaps_path = run_dir / "outputs" / "gaps.md"
        cands_path = run_dir / "outputs" / "candidates.md"
        if not (report_path.is_file() and gaps_path.is_file() and cands_path.is_file()):
            continue
        report_json = report_path.read_text(errors="ignore")
        ref_gaps_md = gaps_path.read_text(errors="ignore")
        ref_cands_md = cands_path.read_text(errors="ignore")

        report = parse_report_text(report_json)
        if not report.parses:
            continue

        # The surviving intuition = the chosen candidate (typically C1). Take the
        # first parsed candidate as the one the report develops.
        candidates = parse_candidates_md(ref_cands_md)
        if not candidates:
            continue
        surviving = candidates[0]
        intuition = _strip_lead_paren(surviving.fields.get("Intuition") or "")
        gap = _strip_lead_paren(surviving.fields.get("Gap it attacks") or "")
        if not intuition:
            continue

        challenge, problem_definition = _read_problem_challenge(run_dir)
        problem = _problem_context(challenge, problem_definition)
        prompt = _build_initial_prompt(intuition, challenge, problem_definition)
        metadata = {
            "source_paper_id": pid,
            "problem": problem,
            "gap": gap,
            "intuition": intuition,
            # Judge's scoring anchor: the validated label report (reward does NOT
            # require verbatim match — it grades the method/eval/setup content).
            "reference_report_json": report_json,
            # Seeded into the rollout's files as the input docs the trainee views.
            "reference_gaps_md": ref_gaps_md,
            "reference_candidates_md": ref_cands_md,
            # Anchor + inline citation ids of the reference report (citation-F1).
            "reference_cited_ids": sorted(set(report.cited_ids)),
        }
        rows.append({"input": prompt, "output": "", "metadata": metadata})
        n_papers += 1
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Build report_writer GRPO data from ground-truth report.json labels"
    )
    parser.add_argument("--labels", type=str, default=str(DEFAULT_LABELS))
    parser.add_argument("--splits", type=str, default=str(DEFAULT_SPLITS))
    parser.add_argument(
        "--out", type=str,
        default=str(Path("data") / "grpo" / "report_writer"),
    )
    parser.add_argument("--n-train", type=int, default=10,
                        help="Number of TRAIN papers to include (one report each).")
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
