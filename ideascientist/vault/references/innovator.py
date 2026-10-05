"""Generate GROUND-TRUTH ``candidates.md`` research-intuition labels for the
innovator agent.

Given a human-written paper already ingested in the vault AND the
ground-truth ``gaps.md`` a prior gap-label run produced for it, we call an LLM
with the paper's FULL TEXT as an oracle plus the production retrieval tools
(``keyword_search`` / ``get_paper_biblio`` / ``read_paper`` / ``edit_doc``). The
model recovers the research INTUITION the paper's method embodies, grounds its
source mechanism + novelty delta in prior work that EXISTS in our database (via
keyword_search + read_paper), and writes a complete ``candidates.md`` in the
exact schema the production ``innovator`` emits. That ``candidates.md`` is the label.

ISOLATION FROM GAP LABELS (by design):
  * The existing gap-label run dir is treated as READ-ONLY. We never write into
    it. We COPY its ``gaps.md`` into a NEW intuition run dir so the innovator can
    ``read_doc('gaps.md')`` without touching the gap output.
  * The intuition run writes its OWN ``candidates.md`` and its OWN
    ``papers/<id>.md`` digests under a separate output tree, because the papers
    the innovator reads (source mechanisms, cross-domain inspiration) differ from
    the papers the gap-finder read (same-problem prior work). Both are kept
    separately so each task's read-set is auditable on its own.

All heavy lifting reuses production code (tools / paper_reader wiring / model
routing), exactly like ``gap_finder.generate_gap_labels``.

Usage:
    python -m ideascientist.vault.references.innovator \\
        --paper-id 94188 \\
        --gaps-root data/references/gap_finder \\
        --model Qwen3.6-27B
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import re
import shutil
import sys
import os
from pathlib import Path

from ideascientist.harness import tools as T
from ideascientist.harness import corpus as C
from ideascientist.harness.client import AgenticLLMClient
from ideascientist.harness.runlog import RunLogger, SharedLogRegistry
from ideascientist.harness import runner as MG
from ideascientist.utils.llm import context_window_for

from ideascientist.vault.references.gap_finder import (
    get_indb_references,
    get_problem_challenge,
    _slug,
    _summarize_tool_result,
)

from ideascientist.vault.references.innovator_prompts import build_label_generator_prompt, build_target_paper_message

# Tools the label generator gets — the SAME set the production innovator has.
_LABEL_TOOLS = ["keyword_search", "get_paper_biblio", "read_paper", "edit_doc"]

_DEFAULT_MAX_TURNS = 60


def find_gaps_md(gaps_root: Path, paper_id: int) -> Path | None:
    """Locate the existing (read-only) gaps.md for a paper under a gap-label root.

    Gap-label dirs are named ``<paper_id>_<slug>/outputs/gaps.md``. Returns the
    first matching path with a non-empty gaps.md, or None.
    """
    if not gaps_root.exists():
        return None
    for entry in sorted(gaps_root.glob(f"{paper_id}_*")):
        gaps = entry / "outputs" / "gaps.md"
        if gaps.exists() and gaps.read_text(encoding="utf-8").strip():
            return gaps
    return None


def generate_for_paper(
    paper_id: int,
    *,
    model: str,
    gaps_root: Path,
    out_root: Path,
    max_turns: int = _DEFAULT_MAX_TURNS,
) -> dict:
    """Run the intuition-label generator for one paper. Returns a summary dict."""
    biblio = C.get_paper_biblio(paper_id)
    if not biblio:
        raise SystemExit(f"paper id {paper_id} not found in the vault")
    title = biblio.get("title", "") or f"paper_{paper_id}"
    full_text = C.get_stored_full_text(paper_id=paper_id)
    if not full_text or not full_text.strip():
        raise SystemExit(
            f"paper id {paper_id} has no stored full text — cannot use it as oracle"
        )

    gaps_src = find_gaps_md(gaps_root, paper_id)
    if gaps_src is None:
        raise SystemExit(
            f"no existing gaps.md for paper {paper_id} under {gaps_root} — run "
            f"gap-label generation first"
        )
    gaps_md = gaps_src.read_text(encoding="utf-8")

    references = get_indb_references(paper_id)
    problem_pc = get_problem_challenge(paper_id)
    problem = problem_pc or (biblio.get("abstract") or "").strip()
    problem_source = "metadata_results_masked" if problem_pc else "abstract"

    run_dir = (out_root / f"{paper_id}_{_slug(title)}").resolve()
    docs_dir = run_dir / "outputs"       # candidates.md + papers/<id>.md live here
    logs_dir = run_dir / "logs"
    inputs_dir = run_dir / "inputs"
    papers_dir = docs_dir / "papers"
    # Fresh start: clear stale artifacts from a prior INTUITION run of the same
    # paper so candidates.md + papers/ reflect only this run (the "read this run"
    # acceptance gate depends on papers/ being run-scoped). This only ever touches
    # the intuition out_root — never the gap-label root.
    if run_dir.exists():
        shutil.rmtree(run_dir)
    papers_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    inputs_dir.mkdir(parents=True, exist_ok=True)

    # Copy the ground-truth gaps.md into the intuition outputs/ so the innovator can
    # read_doc('gaps.md'). The source (gap-label run dir) is left untouched.
    (docs_dir / "gaps.md").write_text(gaps_md, encoding="utf-8")

    # Provenance: record exactly what oracle text + gaps + config we fed in.
    (inputs_dir / "target_fulltext.md").write_text(
        f"# {title}\n\n{full_text}", encoding="utf-8")
    (inputs_dir / "gaps.md").write_text(gaps_md, encoding="utf-8")
    (inputs_dir / "references.json").write_text(
        json.dumps(references, indent=2, ensure_ascii=False), encoding="utf-8")
    (inputs_dir / "problem_challenge.md").write_text(
        f"# {title}\n\n{problem}", encoding="utf-8")
    (inputs_dir / "config.json").write_text(json.dumps({
        "paper_id": paper_id,
        "title": title,
        "model": model,
        "cutoff_date": C.CUTOFF_DATE,
        "max_turns": max_turns,
        "generated_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "fulltext_chars": len(full_text),
        "n_indb_references": len(references),
        "problem_source": problem_source,
        "gaps_source": str(gaps_src),
    }, indent=2), encoding="utf-8")

    log = RunLogger(logs_dir)
    log_registry = SharedLogRegistry()
    log.event("innovator_label_start", paper_id=paper_id, model=model, title=title,
              n_indb_references=len(references), gaps_source=str(gaps_src))

    ctx = T.ToolContext(
        run_dir=run_dir, docs_dir=docs_dir,
        query_paper_id=paper_id, query_text=problem,
        log=log, log_registry=log_registry,
    )

    # Wire read_paper via the PRODUCTION paper_reader sub-agent (identical grounding
    # to the live pipeline, so cited ids pass the source_ids_were_read gate).
    reader_acc = {"n": 0, "tokens": 0, "calls": 0}

    def _read_paper(pid: int, goals: list) -> str:
        digest_path = docs_dir / "papers" / f"{pid}.md"
        reader_acc["n"] += 1
        goal_lines = "\n".join(f"- {g}" for g in goals) or \
            "- summarize the paper's method, key results, and stated limitations"
        task = (f"Read paper id {pid}. Reading goals:\n{goal_lines}\n"
                f"Update papers/{pid}.md (append new goals; reuse any already answered).")
        r = MG._run_subagent(
            "paper_reader", task, ctx, model, log,
            log_key=f"paper_{pid}", conversation_id=f"{pid}#{reader_acc['n']}",
            allow_external=False, max_sub_turns=None,
        )
        reader_acc["tokens"] += r.get("tokens", 0)
        reader_acc["calls"] += r.get("calls", 0)
        if digest_path.exists():
            content = digest_path.read_text(encoding="utf-8")
            if content.strip():
                return content
        digest = r.get("digest") or ""
        if digest.strip():
            digest_path.write_text(digest, encoding="utf-8")
            return digest
        return "(reader produced no digest)"

    ctx.reader_fn = _read_paper

    provider, base_url, text_tools = MG._route_for_model(model)
    client = AgenticLLMClient(
        api_key=MG._resolve_llama_key(), model=model, base_url=base_url,
        provider=provider, text_tools=text_tools,
        max_tokens=32000, log_dir=logs_dir,
    )

    system = build_label_generator_prompt()
    user0 = build_target_paper_message(title, problem, full_text, gaps_md, references)
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": user0}]
    tools = T.openai_specs(_LABEL_TOOLS)
    window = context_window_for(model)  # noqa: F841 (kept for parity/debugging)

    trajectory: list[dict] = []
    step = 0
    read_paper_used = 0
    READ_PAPER_CAP = 20

    for turn in range(max_turns):
        client.set_step(f"innovator_label:{turn}")
        msg = client.chat(messages, tools=tools)["choices"][0]["message"]
        messages.append(msg)
        tcs = msg.get("tool_calls") or []
        if not tcs:
            final = msg.get("content") or ""
            log.event("innovator_label_final", text=final[:500])
            break
        for tc in tcs:
            step += 1
            fn = tc["function"]["name"]
            fargs, perr = MG._parse_tool_args(tc)
            if perr:
                res = perr
            elif fn == "read_paper" and read_paper_used >= READ_PAPER_CAP:
                res = (f"(refused: read_paper cap {READ_PAPER_CAP} reached. Work from the "
                       f"papers/<id>.md digests you already pulled + get_paper_biblio + "
                       f"keyword_search, then finish candidates.md.)")
            else:
                try:
                    res = T.run_tool(fn, fargs or {}, ctx, tool_call_id=tc["id"])
                    if fn == "read_paper":
                        read_paper_used += 1
                except Exception as exc:  # noqa: BLE001
                    res = f"ERROR in {fn}: {exc}"
                    log.event("tool_error", tool=fn, error=str(exc))
            trajectory.append({
                "step": step, "turn": turn, "tool": fn,
                "record": _summarize_tool_result(fn, fargs or {}, res),
            })
            messages.append({"role": "tool", "tool_call_id": tc["id"], "content": res})
    else:
        log.event("innovator_label_max_turns", max_turns=max_turns)

    (logs_dir / "trajectory.json").write_text(
        json.dumps(trajectory, indent=2, ensure_ascii=False), encoding="utf-8")
    s = client.summary()
    client.close()

    candidates_path = docs_dir / "candidates.md"
    cited = _cited_ids(candidates_path)
    n_candidates = _count_candidates(candidates_path)
    summary = {
        "paper_id": paper_id,
        "title": title,
        "candidates_md": str(candidates_path),
        "candidates_exists": candidates_path.exists(),
        "n_candidates": n_candidates,
        "cited_ids": sorted(cited),
        "papers_read": reader_acc["n"],
        "tool_calls": len(trajectory),
        "tokens": s.get("total_tokens", 0) + reader_acc["tokens"],
        "run_dir": str(run_dir),
        "gaps_source": str(gaps_src),
    }
    (logs_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    log.event("innovator_label_done", **{k: summary[k] for k in
              ("cited_ids", "n_candidates", "papers_read", "tool_calls", "tokens")})
    return summary


def _cited_ids(candidates_path: Path) -> set[int]:
    """Extract [<id>] citations from candidates.md.

    Handles single-id brackets ``[123]`` AND multi-id brackets such as
    ``[123, 456]`` / ``[123; 456]`` — every integer inside a bracket that
    contains only ids/separators is treated as a citation, so source-mechanism
    ids grouped in one bracket still enter the "was it read this run" gate.
    """
    if not candidates_path.exists():
        return set()
    text = candidates_path.read_text(encoding="utf-8")
    ids: set[int] = set()
    for inner in re.findall(r"\[([\d,;\s]+)\]", text):
        for m in re.findall(r"\d+", inner):
            ids.add(int(m))
    return ids


def _count_candidates(candidates_path: Path) -> int:
    """Count '### C<n>' candidate blocks in candidates.md.

    Regex parity with the GRPO training parser (``rewards/innovator/parser.py``
    ``_CAND_RE``): case-insensitive, tolerate leading whitespace and ``C 1``
    spacing so the count matches what the trained parser segments on.
    """
    if not candidates_path.exists():
        return 0
    text = candidates_path.read_text(encoding="utf-8")
    return len(re.findall(r"(?im)^\s*###\s+C\s*\d+\b", text))


def _acceptance_checks(summary: dict, target_id: int) -> list[str]:
    problems: list[str] = []
    if not summary["candidates_exists"]:
        problems.append("candidates.md was not written")
        return problems
    if summary["n_candidates"] < 1:
        problems.append("candidates.md has no '### C<n>' candidate block")
    cited = set(summary["cited_ids"])
    if not cited:
        problems.append("candidates.md cites no paper ids")
    if target_id in cited:
        problems.append(f"target paper {target_id} is cited (forbidden)")
    docs = Path(summary["run_dir"]) / "outputs"
    for cid in cited:
        if cid == target_id:
            continue
        if not C.get_paper_biblio(cid):
            problems.append(f"cited id {cid} does not resolve in corpus")
        if not (docs / "papers" / f"{cid}.md").exists():
            problems.append(f"cited id {cid} has no read_paper digest (not read this run)")
    if summary["papers_read"] < 1:
        problems.append(f"only {summary['papers_read']} papers read (<1)")
    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--paper-id", type=int, required=True,
                    help="papers.id in the vault (e.g. 94188 for mixsd)")
    ap.add_argument("--gaps-root", required=True,
                    help="root dir of existing gap labels (read-only), e.g. "
                         "data/references/gap_finder")
    ap.add_argument("--model", default=os.environ.get("IDEASCIENTIST_MODEL", "Qwen3.6-27B"))
    ap.add_argument("--out-dir", default=None,
                    help="output root (default data/references/innovator/)")
    ap.add_argument("--max-turns", type=int, default=_DEFAULT_MAX_TURNS)
    args = ap.parse_args(argv)

    repo = Path(__file__).resolve().parents[4]
    gaps_root = Path(args.gaps_root)
    if not gaps_root.is_absolute():
        gaps_root = repo / gaps_root

    if args.out_dir:
        out_root = Path(args.out_dir)
        if not out_root.is_absolute():
            out_root = repo / out_root
    else:
        date = _dt.date.today().strftime("%Y%m%d")
        out_root = repo / "logs" / "tmp" / f"innovator_labels_{date}"
    out_root.mkdir(parents=True, exist_ok=True)

    summary = generate_for_paper(
        args.paper_id, model=args.model, gaps_root=gaps_root,
        out_root=out_root, max_turns=args.max_turns)

    problems = _acceptance_checks(summary, args.paper_id)
    print(json.dumps(summary, indent=2))
    if problems:
        print("\nACCEPTANCE PROBLEMS:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        return 1
    print("\nAcceptance checks PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
