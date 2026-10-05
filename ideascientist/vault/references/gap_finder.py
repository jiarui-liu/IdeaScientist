"""Generate GROUND-TRUTH ``gaps.md`` labels for the gap_finder agent.

Given a human-written paper that is already ingested in the vault, we
call an LLM (default GPT-5.4) with the paper's FULL TEXT as an oracle plus the
production retrieval tools (``keyword_search`` / ``get_paper_biblio`` /
``read_paper`` / ``edit_doc``). The model derives the methodological challenge
axes the paper argues for, grounds each in prior work that EXISTS in our database
(via keyword_search + read_paper), and writes a complete ``gaps.md`` in the exact
schema the production ``gap_finder`` emits. That ``gaps.md`` is the label.

All heavy lifting reuses production code:
  * tools + ToolContext + run_tool + role_tools + openai_specs  <- harness.tools
  * keyword_search / get_paper_biblio / get_stored_full_text    <- harness.corpus
  * paper_reader wiring for read_paper (_run_subagent)          <- harness.runner
  * model routing / key resolution                              <- harness.runner

We ADD only: (1) inject the target paper's full text, (2) drive the top-level
tool loop, (3) record a compact ``trajectory.json`` (keyword queries + hits,
read_paper goals, biblio calls, edit_doc ops) so labels are auditable.

Usage:
    python -m ideascientist.vault.references.gap_finder \\
        --paper-id 94188 --model openai-gpt-5-4-responses

Requires a reachable chat endpoint; see the README for the environment variables.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import re
import sys
from pathlib import Path
from typing import Any

from ideascientist.harness import tools as T
from ideascientist.harness import corpus as C
from ideascientist.harness.client import AgenticLLMClient
from ideascientist.harness.runlog import RunLogger, SharedLogRegistry
from ideascientist.harness import runner as MG
from ideascientist.utils.llm import context_window_for

from ideascientist.vault.references.gap_finder_prompts import (
    build_label_generator_prompt,
    build_target_paper_message,
)

# Tools the label generator gets — the SAME set the production gap_finder has,
# so the label is produced with the exact retrieval/writing surface the trained
# agent uses.
_LABEL_TOOLS = ["keyword_search", "get_paper_biblio", "read_paper", "edit_doc"]

# Bound the top-level loop. The generator writes gaps.md incrementally, reads a
# bounded number of papers, and stops. Plenty of headroom over a normal run.
_DEFAULT_MAX_TURNS = 60

# Cap the oracle fulltext injected into the first user turn. Some papers have a
# 150K-700K-token body that alone blows the 131072 serving window (HTTP 400 on
# turn 1, or mid-loop once reader digests accumulate). We keep the HEAD — where
# the paper argues its axes (title/abstract/intro/related-work/method) — and drop
# the tail (experiments/appendix/references), leaving room for the system prompt,
# reference list, and the accumulating agentic loop. ~50K tokens ~= 200K chars:
# comfortably covers a paper's argued-axes head while leaving ~80K tokens of the
# 131072 window for the system prompt + references + reader digests mid-loop
# (turn-1 AND mid-loop overflows both observed below this before the cap).
_MAX_FULLTEXT_CHARS = 200_000


def _truncate_fulltext(full_text: str) -> tuple[str, bool]:
    """Head-truncate an oversized oracle body to fit the context window.

    Returns ``(text, was_truncated)``. Keeps the leading ``_MAX_FULLTEXT_CHARS``
    (intro/related-work/method carry the gap axes) and appends a visible marker so
    the model knows the tail (typically experiments/appendix) was dropped.
    """
    if len(full_text) <= _MAX_FULLTEXT_CHARS:
        return full_text, False
    head = full_text[:_MAX_FULLTEXT_CHARS]
    marker = (
        "\n\n[... TRUNCATED: the paper's body exceeds the context window; the "
        "tail (typically experiments/appendix/references) was dropped. Derive the "
        "methodological axes and gaps from the retained head — abstract, "
        "introduction, related work, and method — which is where the paper argues "
        "them.]"
    )
    return head + marker, True


def _slug(text: str, n: int = 24) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", (text or "").lower()).strip("_")
    return s[:n] or "paper"


def get_indb_references(paper_id: int) -> list[dict]:
    """Target paper's cited works that resolve to a paper in the vault.

    Reads the ``citations`` table (source_paper_id = target) and keeps rows whose
    ``cited_paper_id`` is non-null (i.e. the citation was matched to a paper in
    our corpus). Returns ``[{"id": int, "title": str}]`` with the corpus title
    (falling back to the citation's recorded title).
    """
    import sqlite3

    con = C._autodb_ro()
    try:
        rows = con.execute(
            "SELECT DISTINCT c.cited_paper_id, "
            "COALESCE(p.title, c.cited_title) AS title "
            "FROM citations c LEFT JOIN papers p ON p.id = c.cited_paper_id "
            "WHERE c.source_paper_id = ? AND c.cited_paper_id IS NOT NULL "
            "ORDER BY c.cited_paper_id",
            (int(paper_id),),
        ).fetchall()
    except sqlite3.Error:
        return []
    finally:
        con.close()
    return [{"id": int(cid), "title": (title or "").strip()} for cid, title in rows]


def get_problem_challenge(paper_id: int) -> str:
    """Problem definition + challenge for the target paper.

    Uses the PRODUCTION field extractor ``corpus.get_paper_fields`` (which
    derives labeled field texts from ``metadata_results_masked.fields_json`` via
    ``_variantb_field_texts``), and surfaces the two fields that describe the
    problem the agent is handed: ``challenge`` (problem_statement +
    why_it_matters + concrete_example) and ``problem_definition`` (formal_setting
    + inputs + outputs). Returns "" if the paper has no results-masked fields.

    NOTE: this is byte-for-byte the production ``challenge`` / ``problem_definition``
    text. It may include the paper's own method (``problem_definition.formal_setting``
    sometimes names the target method) — leakage handling is applied by the caller,
    not here, so this stays faithful to production.
    """
    fields = C.get_paper_fields(paper_id)
    if not fields:
        return ""
    challenge = (fields.get("challenge") or "").strip()
    problem_definition = (fields.get("problem_definition") or "").strip()
    parts: list[str] = []
    if challenge:
        parts.append(f"Challenge: {challenge}")
    if problem_definition:
        parts.append(f"Problem definition: {problem_definition}")
    return "\n\n".join(parts)


def _summarize_tool_result(tool: str, args: dict, result: str) -> Any:
    if tool == "keyword_search":
        try:
            hits = json.loads(result)
        except (ValueError, TypeError):
            hits = []
        return {
            "queries": args.get("queries", []),
            "k": args.get("k"),
            "n_hits": len(hits) if isinstance(hits, list) else 0,
            "hits": hits[:40] if isinstance(hits, list) else [],
        }
    if tool == "read_paper":
        return {
            "paper_id": args.get("paper_id"),
            "goals": args.get("goals", []),
            "digest_chars": len(result or ""),
        }
    if tool == "get_paper_biblio":
        try:
            b = json.loads(result)
        except (ValueError, TypeError):
            b = {}
        return {"paper_id": args.get("paper_id"),
                "title": b.get("title", "") if isinstance(b, dict) else ""}
    if tool == "edit_doc":
        return {"command": args.get("command"), "path": args.get("path"),
                "result": (result or "")[:160]}
    return {"result": (result or "")[:200]}


def generate_for_paper(
    paper_id: int,
    *,
    model: str,
    out_root: Path,
    max_turns: int = _DEFAULT_MAX_TURNS,
) -> dict:
    """Run the label generator for one paper. Returns a summary dict."""
    biblio = C.get_paper_biblio(paper_id)
    if not biblio:
        raise SystemExit(f"paper id {paper_id} not found in the vault")
    title = biblio.get("title", "") or f"paper_{paper_id}"
    full_text = C.get_stored_full_text(paper_id=paper_id)
    if not full_text or not full_text.strip():
        raise SystemExit(
            f"paper id {paper_id} has no stored full text — cannot use it as oracle"
        )
    fulltext_chars_raw = len(full_text)
    full_text, fulltext_truncated = _truncate_fulltext(full_text)
    references = get_indb_references(paper_id)
    problem_pc = get_problem_challenge(paper_id)
    problem = problem_pc or (biblio.get("abstract") or "").strip()
    problem_source = "metadata_results_masked" if problem_pc else "abstract"

    run_dir = (out_root / f"{paper_id}_{_slug(title)}").resolve()
    docs_dir = run_dir / "outputs"      # gaps.md + papers/<id>.md live here
    logs_dir = run_dir / "logs"
    inputs_dir = run_dir / "inputs"
    papers_dir = docs_dir / "papers"
    # Fresh start: clear stale artifacts from a prior run of the same paper so
    # gaps.md and the papers/ digests reflect only this run (the "read this run"
    # acceptance gate depends on papers/ being run-scoped).
    if run_dir.exists():
        import shutil
        shutil.rmtree(run_dir)
    papers_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    inputs_dir.mkdir(parents=True, exist_ok=True)

    # Provenance: record exactly what oracle text and config we fed in.
    (inputs_dir / "target_fulltext.md").write_text(
        f"# {title}\n\n{full_text}", encoding="utf-8")
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
        "fulltext_chars_raw": fulltext_chars_raw,
        "fulltext_truncated": fulltext_truncated,
        "n_indb_references": len(references),
        "problem_source": problem_source,
    }, indent=2), encoding="utf-8")

    log = RunLogger(logs_dir)
    log_registry = SharedLogRegistry()
    log.event("label_gen_start", paper_id=paper_id, model=model, title=title,
              n_indb_references=len(references))

    # ToolContext: docs go to outputs/, query is the target paper. write_scope is
    # left None (full allowlist) so the generator may write gaps.md + papers/<id>.md.
    ctx = T.ToolContext(
        run_dir=run_dir, docs_dir=docs_dir,
        query_paper_id=paper_id, query_text=problem,
        log=log, log_registry=log_registry,
    )

    # Wire read_paper via the PRODUCTION paper_reader sub-agent (identical grounding
    # to the live pipeline, so cited ids pass the cited_ids_were_read gate).
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
        # Reader returned a digest without persisting papers/<id>.md — write it
        # ourselves so every read paper has an auditable, gate-passing digest.
        digest = r.get("digest") or ""
        if digest.strip():
            digest_path.write_text(digest, encoding="utf-8")
            return digest
        return "(reader produced no digest)"

    ctx.reader_fn = _read_paper

    # Top-level client (the generator itself). Mirrors the first-tier client.
    provider, base_url, text_tools = MG._route_for_model(model)
    client = AgenticLLMClient(
        api_key=MG._resolve_llama_key(), model=model, base_url=base_url,
        provider=provider, text_tools=text_tools,
        max_tokens=32000, log_dir=logs_dir,
    )

    system = build_label_generator_prompt()
    user0 = build_target_paper_message(title, problem, full_text, references)
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": user0}]
    tools = T.openai_specs(_LABEL_TOOLS)
    window = context_window_for(model)

    trajectory: list[dict] = []
    step = 0
    read_paper_used = 0
    READ_PAPER_CAP = 20  # bound nested-reader cost

    for turn in range(max_turns):
        client.set_step(f"label:{turn}")
        msg = client.chat(messages, tools=tools)["choices"][0]["message"]
        messages.append(msg)
        tcs = msg.get("tool_calls") or []
        if not tcs:
            final = msg.get("content") or ""
            log.event("label_gen_final", text=final[:500])
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
                       f"keyword_search, then finish gaps.md.)")
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
        log.event("label_gen_max_turns", max_turns=max_turns)

    # Persist trajectory + close out.
    (logs_dir / "trajectory.json").write_text(
        json.dumps(trajectory, indent=2, ensure_ascii=False), encoding="utf-8")
    s = client.summary()
    client.close()

    gaps_path = docs_dir / "gaps.md"
    cited = _cited_ids(gaps_path)
    summary = {
        "paper_id": paper_id,
        "title": title,
        "gaps_md": str(gaps_path),
        "gaps_exists": gaps_path.exists(),
        "cited_ids": sorted(cited),
        "papers_read": reader_acc["n"],
        "tool_calls": len(trajectory),
        "tokens": s.get("total_tokens", 0) + reader_acc["tokens"],
        "run_dir": str(run_dir),
    }
    (logs_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    log.event("label_gen_done", **{k: summary[k] for k in
              ("cited_ids", "papers_read", "tool_calls", "tokens")})
    return summary


def _cited_ids(gaps_path: Path) -> set[int]:
    """Extract [<id>] citations from gaps.md (integer ids only).

    Regex parity with the GRPO training parser (``rewards/gap_finder/parser.py``
    ``_CITED_ID_RE = re.compile(r"\\[(\\d+)\\]")``): a bare ``[n]`` counts as a
    citation with NO LaTeX-skip lookbehind. This deliberately also matches
    sub/superscripts like ``^[2]`` / ``_{[2]}`` — the trained parser does, so the
    label generator must too (train/serve parity), even though a couple of math
    tokens may be miscounted as citations.
    """
    if not gaps_path.exists():
        return set()
    text = gaps_path.read_text(encoding="utf-8")
    return {int(m) for m in re.findall(r"\[(\d+)\]", text)}


def _acceptance_checks(summary: dict, target_id: int) -> list[str]:
    problems: list[str] = []
    if not summary["gaps_exists"]:
        problems.append("gaps.md was not written")
        return problems
    cited = set(summary["cited_ids"])
    if not cited:
        problems.append("gaps.md cites no paper ids")
    if target_id in cited:
        problems.append(f"target paper {target_id} is cited as near-miss (forbidden)")
    docs = Path(summary["run_dir"]) / "outputs"
    for cid in cited:
        if cid == target_id:
            continue
        if not C.get_paper_biblio(cid):
            problems.append(f"cited id {cid} does not resolve in corpus")
        if not (docs / "papers" / f"{cid}.md").exists():
            problems.append(f"cited id {cid} has no read_paper digest (not read this run)")
    if summary["papers_read"] < 2:
        problems.append(f"only {summary['papers_read']} papers read (<2)")
    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--paper-id", type=int, required=True,
                    help="papers.id in the vault (e.g. 94188 for mixsd)")
    ap.add_argument("--model", default="openai-gpt-5-4-responses")
    ap.add_argument("--out-dir", default=None,
                    help="output root (default data/references/gap_finder/)")
    ap.add_argument("--max-turns", type=int, default=_DEFAULT_MAX_TURNS)
    args = ap.parse_args(argv)

    if args.out_dir:
        out_root = Path(args.out_dir)
    else:
        repo = Path(__file__).resolve().parents[4]
        date = _dt.date.today().strftime("%Y%m%d")
        out_root = repo / "logs" / "tmp" / f"gap_labels_{date}"
    out_root.mkdir(parents=True, exist_ok=True)

    summary = generate_for_paper(
        args.paper_id, model=args.model, out_root=out_root, max_turns=args.max_turns)

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
