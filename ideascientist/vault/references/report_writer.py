"""Generate GROUND-TRUTH ``report.json`` solution-proposal labels for the
report_writer agent.

Given a human-written paper already ingested in the vault AND the
ground-truth ``gaps.md`` (assigned gap) + ``candidates.md`` (chosen intuition)
that prior label runs produced for it, we call an LLM with the paper's FULL TEXT
as an oracle plus the production retrieval tools (``keyword_search`` /
``get_paper_biblio`` / ``read_paper`` / ``edit_doc``). The model develops the
surviving intuition into a complete results-masked solution proposal, grounds every
cited ``[id]`` and load-bearing claim in prior work that EXISTS in our
database and was read_paper'd this run (via keyword_search + read_paper), and
writes a single valid ``report.json`` in the exact schema the production
``report_writer`` emits. That ``report.json`` is the label.

ISOLATION FROM UPSTREAM LABELS (by design):
  * The upstream innovator-label run dir (which holds both ``gaps.md`` and
    ``candidates.md``) is treated as READ-ONLY. We never write into it. We COPY
    its ``gaps.md`` + ``candidates.md`` into a NEW report run dir so the
    report_writer can ``read_doc`` them without touching the upstream output.
  * The report run writes its OWN ``report.json`` and its OWN ``papers/<id>.md``
    digests under a separate output tree, because the papers the report_writer
    reads (competitors, borrowed mechanisms, benchmarks) may differ from the
    papers upstream agents read. Each task's read-set is auditable on its own.

All heavy lifting reuses production code (tools / paper_reader wiring / model
routing), exactly like ``gap_finder.generate_gap_labels`` and
``innovator.generate_innovator_labels``.

Usage:
    python -m ideascientist.vault.references.report_writer \\
        --paper-id 94188 \\
        --candidates-root data/references/innovator \\
        --model Qwen3.6-27B
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import re
import shutil
import sys
from functools import lru_cache
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
    _truncate_fulltext,
)

from ideascientist.vault.references.report_writer_prompts import build_label_generator_prompt, build_target_paper_message

# Tools the label generator gets — the SAME set the production report_writer has.
_LABEL_TOOLS = ["keyword_search", "get_paper_biblio", "read_paper", "edit_doc"]

_DEFAULT_MAX_TURNS = 60

# Required top-level fields the report.json label must fill (mirrors the
# ``required_schema_fields`` gate in the training rubric so labels are in-spec).
_REQUIRED_FIELDS = [
    "title", "topic_relevance", "one_sentence_thesis", "core_problem",
    "key_novelty", "problem_definition", "key_terms", "related_work", "method",
    "model_details", "comparison_to_sota", "proposed_evaluation",
    "falsifiable_predictions", "limitations",
]


def find_candidates_md(candidates_root: Path, paper_id: int) -> Path | None:
    """Locate the existing (read-only) candidates.md for a paper under a label root.

    Innovator-label dirs are named ``<paper_id>_<slug>/outputs/candidates.md``.
    Returns the first matching path with a non-empty candidates.md, or None.
    """
    if not candidates_root.exists():
        return None
    for entry in sorted(candidates_root.glob(f"{paper_id}_*")):
        cand = entry / "outputs" / "candidates.md"
        if cand.exists() and cand.read_text(encoding="utf-8").strip():
            return cand
    return None


def find_gaps_md(root: Path, paper_id: int) -> Path | None:
    """Locate a non-empty gaps.md for a paper under a label root.

    Both innovator-label and gap-label dirs expose ``<paper_id>_<slug>/outputs/gaps.md``.
    """
    if not root.exists():
        return None
    for entry in sorted(root.glob(f"{paper_id}_*")):
        gaps = entry / "outputs" / "gaps.md"
        if gaps.exists() and gaps.read_text(encoding="utf-8").strip():
            return gaps
    return None


def generate_for_paper(
    paper_id: int,
    *,
    model: str,
    candidates_root: Path,
    gaps_root: Path | None = None,
    out_root: Path,
    max_turns: int = _DEFAULT_MAX_TURNS,
) -> dict:
    """Run the report-label generator for one paper. Returns a summary dict."""
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

    cand_src = find_candidates_md(candidates_root, paper_id)
    if cand_src is None:
        raise SystemExit(
            f"no existing candidates.md for paper {paper_id} under {candidates_root} "
            f"— run innovator-label generation first"
        )
    candidates_md = cand_src.read_text(encoding="utf-8")

    # gaps.md lives beside candidates.md in the innovator-label tree; fall back to a
    # separate gaps-root if provided.
    gaps_src = find_gaps_md(candidates_root, paper_id)
    if gaps_src is None and gaps_root is not None:
        gaps_src = find_gaps_md(gaps_root, paper_id)
    if gaps_src is None:
        raise SystemExit(
            f"no existing gaps.md for paper {paper_id} under {candidates_root} "
            f"(or --gaps-root) — the proposal needs the assigned gap"
        )
    gaps_md = gaps_src.read_text(encoding="utf-8")

    references = get_indb_references(paper_id)
    problem_pc = get_problem_challenge(paper_id)
    problem = problem_pc or (biblio.get("abstract") or "").strip()
    problem_source = "metadata_results_masked" if problem_pc else "abstract"

    run_dir = (out_root / f"{paper_id}_{_slug(title)}").resolve()
    docs_dir = run_dir / "outputs"       # report.json + papers/<id>.md live here
    logs_dir = run_dir / "logs"
    inputs_dir = run_dir / "inputs"
    papers_dir = docs_dir / "papers"
    # Fresh start: clear stale artifacts from a prior REPORT run of the same paper
    # so report.json + papers/ reflect only this run. This only ever touches the
    # report out_root — never the upstream (gap/innovator) label roots.
    if run_dir.exists():
        shutil.rmtree(run_dir)
    papers_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    inputs_dir.mkdir(parents=True, exist_ok=True)

    # Copy the ground-truth gaps.md + candidates.md into the report outputs/ so
    # the report_writer can read_doc them. The sources are left untouched.
    (docs_dir / "gaps.md").write_text(gaps_md, encoding="utf-8")
    (docs_dir / "candidates.md").write_text(candidates_md, encoding="utf-8")

    # Provenance: record exactly what oracle text + gaps + candidates + config we fed in.
    (inputs_dir / "target_fulltext.md").write_text(
        f"# {title}\n\n{full_text}", encoding="utf-8")
    (inputs_dir / "gaps.md").write_text(gaps_md, encoding="utf-8")
    (inputs_dir / "candidates.md").write_text(candidates_md, encoding="utf-8")
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
        "gaps_source": str(gaps_src),
        "candidates_source": str(cand_src),
    }, indent=2), encoding="utf-8")

    log = RunLogger(logs_dir)
    log_registry = SharedLogRegistry()
    log.event("report_label_start", paper_id=paper_id, model=model, title=title,
              n_indb_references=len(references), gaps_source=str(gaps_src),
              candidates_source=str(cand_src))

    ctx = T.ToolContext(
        run_dir=run_dir, docs_dir=docs_dir,
        query_paper_id=paper_id, query_text=problem,
        log=log, log_registry=log_registry,
    )

    # Wire read_paper via the PRODUCTION paper_reader sub-agent (identical grounding
    # to the live pipeline).
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
    user0 = build_target_paper_message(
        title, problem, full_text, gaps_md, candidates_md, references)
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": user0}]
    tools = T.openai_specs(_LABEL_TOOLS)
    window = context_window_for(model)  # noqa: F841 (kept for parity/debugging)

    trajectory: list[dict] = []
    step = 0
    read_paper_used = 0
    READ_PAPER_CAP = 20
    MAX_REMEDIATION = 2
    upstream_ids = _upstream_verified_ids(str(run_dir))

    remediation = 0
    while True:
        for turn in range(max_turns):
            client.set_step(f"report_label:{turn}")
            msg = client.chat(messages, tools=tools)["choices"][0]["message"]
            messages.append(msg)
            tcs = msg.get("tool_calls") or []
            if not tcs:
                final = msg.get("content") or ""
                log.event("report_label_final", text=final[:500])
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
                           f"keyword_search, then finish report.json.)")
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
            log.event("report_label_max_turns", max_turns=max_turns)

        # Remediation step 0: the report must actually EXIST and PARSE before any
        # content-level check. Some models (observed with GPT-5.4) call edit_doc
        # view/str_replace on report.json BEFORE ever creating it, get a
        # "(no such file)", then falsely reply "complete and validated" without
        # writing anything. Without this branch the loop would see an empty
        # mechanism set and break, wasting the run. Force a create and continue.
        report_now, parse_err_now = _load_report(docs_dir / "report.json")
        if report_now is None and remediation < MAX_REMEDIATION:
            remediation += 1
            log.event("report_label_remediation_missing", round=remediation,
                      parse_error=parse_err_now)
            messages.append({"role": "user", "content": (
                f"BLOCKING ISSUE — report.json is still not a valid file "
                f"({parse_err_now}). You have NOT produced the label yet. You must "
                f"write it now: call edit_doc(command='create', path='report.json', "
                f"content=<the COMPLETE JSON object>). 'create' makes the file from "
                f"scratch — do NOT use view/str_replace/insert on report.json before "
                f"it exists, they will fail with '(no such file)'. The content must be "
                f"a SINGLE valid JSON object with every required field filled and every "
                f"cited paper written as a bare [id]. Do not reply "
                f"that it is done until edit_doc has actually written it.")})
            continue

        # Remediation: if the finished report cites ANY paper that was neither
        # read_paper'd this run nor grounded upstream in gaps.md / candidates.md,
        # feed those ids back and require the agent to either read_paper them or
        # remove the citation. This mirrors the production ``cited_ids_were_read``
        # gate exactly (every cited [id] must be read-or-seeded), turning the
        # otherwise-common cite-without-reading failure into a self-correcting pass.
        read_now = _read_ids(str(run_dir))
        cited_now = _anchor_ids(report_now) | _inline_anchor_ids(report_now)
        cited_unread_now = sorted(cited_now - read_now - upstream_ids - {paper_id})
        if not cited_unread_now or remediation >= MAX_REMEDIATION:
            break
        remediation += 1
        budget_left = READ_PAPER_CAP - read_paper_used
        log.event("report_label_remediation", round=remediation,
                  cited_unread=cited_unread_now, read_budget_left=budget_left)
        if budget_left > 0:
            action = (
                f"For EACH id above, do ONE of: (a) call read_paper(id, goals) now to "
                f"ground it (read_paper budget left: {budget_left}), or (b) remove that "
                f"[id] from EVERY string in the report (drop the citation entirely).")
        else:
            action = (
                f"Your read_paper budget is EXHAUSTED (cap {READ_PAPER_CAP} reached), so "
                f"you CANNOT read more papers. For EACH id above you MUST remove that "
                f"[id] from EVERY string in the report (drop the citation entirely). "
                f"Do not attempt read_paper; it will be refused.")
        messages.append({"role": "user", "content": (
            f"BLOCKING ISSUE — your report.json cites papers you neither read_paper'd "
            f"this run nor took from gaps.md / candidates.md. Every cited [id] MUST be "
            f"grounded by a read this run (or come from the seeded inputs); these are "
            f"not: {cited_unread_now}. {action} "
            f"Do NOT invent replacements. Then rewrite report.json with edit_doc so it "
            f"parses as one valid JSON object. "
            f"When done, reply with a one-line status.")})

    (logs_dir / "trajectory.json").write_text(
        json.dumps(trajectory, indent=2, ensure_ascii=False), encoding="utf-8")
    s = client.summary()
    client.close()

    report_path = docs_dir / "report.json"
    report, parse_err = _load_report(report_path)
    anchors = _anchor_ids(report)
    inline_anchors = _inline_anchor_ids(report)
    mech_cited = _mechanism_cited_ids(report)
    read_ids = _read_ids(str(run_dir))
    upstream_ids = _upstream_verified_ids(str(run_dir))
    # A cited id is grounded if the paper was read_paper'd this run OR the id was
    # already grounded upstream in gaps.md / candidates.md (mirrors the production
    # ``cited_ids_were_read`` gate). ``mechanism_unread_ids`` kept for diagnostics.
    cited_ids = anchors | inline_anchors
    cited_unread = sorted(cited_ids - read_ids - upstream_ids - {paper_id})
    mech_unread = sorted(mech_cited - read_ids - upstream_ids - {paper_id})
    summary = {
        "paper_id": paper_id,
        "title": title,
        "report_json": str(report_path),
        "report_exists": report_path.exists(),
        "report_parses": report is not None,
        "parse_error": parse_err,
        "anchor_ids": sorted(anchors),
        "inline_anchor_ids": sorted(inline_anchors),
        "cited_ids": sorted(cited_ids),
        "cited_unread_ids": cited_unread,
        "mechanism_cited_ids": sorted(mech_cited),
        "mechanism_unread_ids": mech_unread,
        "papers_read": reader_acc["n"],
        "tool_calls": len(trajectory),
        "tokens": s.get("total_tokens", 0) + reader_acc["tokens"],
        "run_dir": str(run_dir),
        "gaps_source": str(gaps_src),
        "candidates_source": str(cand_src),
    }
    (logs_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    log.event("report_label_done", **{k: summary[k] for k in
              ("anchor_ids", "report_parses", "papers_read", "cited_unread_ids",
               "tool_calls", "tokens")})
    return summary


def _load_report(report_path: Path) -> tuple[dict | None, str | None]:
    """Parse report.json. Returns ``(obj, None)`` or ``(None, error_str)``.

    Tolerates an accidental ```` ```json ```` fence wrapper only for diagnosis —
    the strict ``json_parses`` gate is reported by ``_acceptance_checks``.
    """
    if not report_path.exists():
        return None, "report.json was not written"
    text = report_path.read_text(encoding="utf-8")
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as exc:
        return None, f"invalid JSON: {exc}"
    if not isinstance(obj, dict):
        return None, "report.json is not a JSON object"
    return obj, None


def _walk_strings(obj: object):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _walk_strings(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _walk_strings(v)


# Bare inline citation: [123] inside a string value (innovator/gap_finder style).
# SINGLE id per bracket only (mirrors gap_finder ``_CITED_ID_RE``) so the comma
# form ``[0, 1]`` (a math interval) is never read as a citation. The negative
# lookbehind rejects array indexing / attribute access like ``shape[0]`` or
# ``arr[1]`` (a citation is never glued to a preceding word char / ``)`` / ``]``).
_BARE_CITE_RE = re.compile(r"(?<![\w)\]])\[(\d+)\]")


@lru_cache(maxsize=200_000)
def _id_resolves(pid: int) -> bool:
    """True if ``pid`` is a real paper in the vault. Fail-OPEN on a DB error
    (so parsing still works without the DB) — the reward-time gate re-checks with
    the injected biblio_fn anyway.
    """
    try:
        return bool(C.get_paper_biblio(int(pid)))
    except Exception:
        return True


def _bare_ids_in(text: str) -> set[int]:
    return {int(m) for m in _BARE_CITE_RE.findall(str(text)) if _id_resolves(int(m))}


def _anchor_ids(report: dict | None) -> set[int]:
    if not report:
        return set()
    ids: set[int] = set()
    for rw in report.get("related_work") or []:
        if not isinstance(rw, dict):
            continue
        for c in rw.get("citations") or []:
            c = str(c).strip()
            if c.isdigit() and _id_resolves(int(c)):
                ids.add(int(c))
    return ids


def _inline_anchor_ids(report: dict | None) -> set[int]:
    """Integer ids cited inline as bare ``[id]`` anywhere in the JSON's string
    values (innovator/gap_finder citation style). String leaves only — never the
    raw ``json.dumps`` blob, so numeric JSON arrays are not mis-read as citations.
    """
    if not report:
        return set()
    ids: set[int] = set()
    for text in _walk_strings(report):
        ids |= _bare_ids_in(text)
    return ids


def _mechanism_cited_ids(report: dict | None) -> set[int]:
    """Ids cited at the MECHANISM level: inside related_work (what_it_does /
    open_gap / citations) or comparison_to_sota (related_methods / key_differences).

    These are the claims that require a body-level read (see rule 2a in the label
    prompt): a mechanism sentence like "X reweights samples by pretrained loss"
    cannot be written from a title/abstract alone. Benchmark/dataset/model ids
    cited only in proposed_evaluation / model_details are intentionally excluded —
    those may be cited from biblio.
    """
    if not report:
        return set()
    ids: set[int] = set()

    def _inline(text: str) -> None:
        ids.update(_bare_ids_in(text))

    for rw in report.get("related_work") or []:
        if not isinstance(rw, dict):
            continue
        _inline(rw.get("what_it_does", ""))
        _inline(rw.get("open_gap", ""))
        for c in rw.get("citations") or []:
            c = str(c).strip()
            if c.isdigit() and _id_resolves(int(c)):
                ids.add(int(c))
    cmp = report.get("comparison_to_sota") or {}
    if isinstance(cmp, dict):
        for key in ("related_methods", "key_differences"):
            for entry in cmp.get(key) or []:
                _inline(entry)
    return ids


def _read_ids(run_dir: str) -> set[int]:
    papers = Path(run_dir) / "outputs" / "papers"
    if not papers.exists():
        return set()
    return {int(p.stem) for p in papers.glob("*.md") if p.stem.isdigit()}


def _upstream_verified_ids(run_dir: str) -> set[int]:
    """Ids the upstream gap_finder / innovator labels already grounded.

    Any id appearing in this run's ``gaps.md`` or ``candidates.md`` was written
    there by the gap/intuition label generators, which read those papers' full
    text before asserting the mechanism claim. The report_writer legitimately
    reuses those upstream-verified mechanism descriptions (e.g. a related-work
    axis's ``[43441], [43455]`` citation list, or a candidate's source
    inspiration) WITHOUT re-reading, so such ids are exempt from the
    ``mechanism_claims_are_read`` gate. Ids the report_writer introduces on its
    own (absent from gaps.md / candidates.md) still require a read_paper.
    """
    ids: set[int] = set()
    out = Path(run_dir) / "outputs"
    for name in ("gaps.md", "candidates.md"):
        p = out / name
        if not p.exists():
            continue
        text = p.read_text(encoding="utf-8", errors="ignore")
        for m in re.findall(r"\[(\d+)\]", text):
            ids.add(int(m))
        for m in re.findall(r"#ref-(\d+)", text):
            ids.add(int(m))
    return ids


def _acceptance_checks(summary: dict, target_id: int) -> list[str]:
    """Lightweight rubric-mirroring checks (mirror the report_writer det. gates).

    Returns a list of problems (empty = ok).
    """
    problems: list[str] = []
    if not summary["report_exists"]:
        problems.append("report.json was not written")
        return problems
    if not summary["report_parses"]:
        problems.append(f"report.json does not parse: {summary.get('parse_error')}")
        return problems

    report, _ = _load_report(Path(summary["report_json"]))
    # schema_valid: every required field present and non-empty.
    for field in _REQUIRED_FIELDS:
        val = (report or {}).get(field)
        if val is None or (hasattr(val, "__len__") and len(val) == 0):
            problems.append(f"required field '{field}' is missing or empty")

    cited = set(summary["anchor_ids"]) | set(summary["inline_anchor_ids"])
    if not cited:
        problems.append("report.json cites no paper ids")
    if target_id in cited:
        problems.append(f"target paper {target_id} is cited (forbidden)")
    # cited_ids_resolve: every cited [id] resolves to a real paper in the corpus.
    for cid in cited:
        if cid == target_id:
            continue
        if not C.get_paper_biblio(cid):
            problems.append(f"cited id {cid} does not resolve in corpus")
    # cited_ids_were_read (mirrors the production gate): EVERY cited [id] must be
    # grounded — read_paper'd this run OR already present in the seeded gaps.md /
    # candidates.md inputs. A citation to a paper the label never read is exactly
    # the cite-without-reading failure we forbid.
    cited_unread = summary.get("cited_unread_ids")
    if cited_unread is None:  # older summary without the field — recompute
        report2, _ = _load_report(Path(summary["report_json"]))
        cited_unread = sorted(
            (_anchor_ids(report2) | _inline_anchor_ids(report2))
            - _read_ids(summary["run_dir"])
            - _upstream_verified_ids(summary["run_dir"])
            - {target_id})
    if cited_unread:
        problems.append(
            f"cited ids not read_paper'd this run and absent from "
            f"gaps/candidates: {cited_unread}")
    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--paper-id", type=int, required=True,
                    help="papers.id in the vault (e.g. 94188 for mixsd)")
    ap.add_argument("--candidates-root", required=True,
                    help="root dir of existing innovator labels (read-only), which "
                         "holds candidates.md AND gaps.md, e.g. "
                         "data/references/innovator")
    ap.add_argument("--gaps-root", default=None,
                    help="optional separate root for gaps.md if not beside "
                         "candidates.md")
    ap.add_argument("--model", default=os.environ.get("IDEASCIENTIST_MODEL", "Qwen3.6-27B"))
    ap.add_argument("--out-dir", default=None,
                    help="output root (default data/references/report_writer/)")
    ap.add_argument("--max-turns", type=int, default=_DEFAULT_MAX_TURNS)
    args = ap.parse_args(argv)

    repo = Path(__file__).resolve().parents[4]

    def _resolve(p: str) -> Path:
        path = Path(p)
        return path if path.is_absolute() else repo / path

    candidates_root = _resolve(args.candidates_root)
    gaps_root = _resolve(args.gaps_root) if args.gaps_root else None

    if args.out_dir:
        out_root = _resolve(args.out_dir)
    else:
        date = _dt.date.today().strftime("%Y%m%d")
        out_root = repo / "logs" / "tmp" / f"report_labels_{date}"
    out_root.mkdir(parents=True, exist_ok=True)

    summary = generate_for_paper(
        args.paper_id, model=args.model, candidates_root=candidates_root,
        gaps_root=gaps_root, out_root=out_root, max_turns=args.max_turns)

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
