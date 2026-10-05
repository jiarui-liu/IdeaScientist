"""Render a report_writer ``report.json`` object into human-readable markdown.

The judges and the citation metric read prose, not JSON, so the structured
report has to become a paper-shaped document first.

A field is rendered only if present — the writer omits empty ones. Inline
citations arrive as ``[short-label year](#ref-<anchor>)``, so a ``## References``
section emits matching ``<a id="ref-<anchor>">`` entries for the links to
resolve against and :mod:`ideascientist.evaluation.citations` to extract.
"""
from __future__ import annotations

from typing import Any


def _is_empty(v: Any) -> bool:
    return v is None or v == "" or v == [] or v == {}


def _asdict(v: Any) -> dict:
    return v if isinstance(v, dict) else {}


def _inline(v: Any) -> str:
    if isinstance(v, dict):
        return "; ".join(f"**{k}**: {v[k]}" for k in v if not _is_empty(v[k]))
    return str(v)


def _bullets(items: list, indent: str = "") -> list[str]:
    out = []
    for it in items:
        if _is_empty(it):
            continue
        out.append(f"{indent}- {_inline(it)}")
    return out


def _h(level: int, text: str) -> str:
    return f"{'#' * level} {text}"


def _render_meta(rep: dict, out: list[str]) -> None:
    title = rep.get("title")
    if title:
        out += [_h(1, title), ""]
    thesis = rep.get("one_sentence_thesis")
    if thesis:
        out += [f"*{thesis}*", ""]
    tr = _asdict(rep.get("topic_relevance"))
    bits = []
    if tr.get("sub_topic"):
        bits.append("**Sub-topics:** " + ", ".join(tr["sub_topic"]))
    if tr.get("primary_focus"):
        bits.append("**Primary focus:** " + ", ".join(tr["primary_focus"]))
    if bits:
        out += [" &nbsp;·&nbsp; ".join(bits), ""]


def _render_problem(rep: dict, out: list[str]) -> None:
    cp = _asdict(rep.get("core_problem"))
    if _is_empty(cp):
        return
    out += [_h(2, "Problem"), ""]
    if cp.get("problem_statement"):
        out += [cp["problem_statement"], ""]
    if cp.get("why_it_matters"):
        out += ["**Why it matters**", ""] + _bullets(cp["why_it_matters"]) + [""]
    if cp.get("concrete_example"):
        out += ["**Concrete example**", "", cp["concrete_example"], ""]


def _render_novelty(rep: dict, out: list[str]) -> None:
    kn = _asdict(rep.get("key_novelty"))
    if _is_empty(kn):
        return
    out += [_h(2, "Key Novelty"), ""]
    if kn.get("main_idea"):
        out += [f"**{kn['main_idea']}**", ""]
    if kn.get("explanation"):
        out += _bullets(kn["explanation"]) + [""]


def _render_problem_def(rep: dict, out: list[str]) -> None:
    pd = _asdict(rep.get("problem_definition"))
    if _is_empty(pd):
        return
    out += [_h(2, "Problem Definition"), ""]
    for label, key in [("Formal setting", "formal_setting"),
                       ("Inputs", "inputs"), ("Outputs", "outputs")]:
        if pd.get(key):
            out += [f"- **{label}:** {pd[key]}"]
    out += [""]


def _render_related_work(rep: dict, out: list[str]) -> None:
    rw = rep.get("related_work") or []
    if _is_empty(rw):
        return
    out += [_h(2, "Related Work"), ""]
    for grp in rw:
        if _is_empty(grp):
            continue
        if not isinstance(grp, dict):
            out += [f"- {_inline(grp)}"]
            continue
        sub = grp.get("sub_area") or "—"
        out += [_h(3, sub), ""]
        if grp.get("what_it_does"):
            out += [grp["what_it_does"], ""]
        if grp.get("open_gap"):
            out += [f"**Open gap:** {grp['open_gap']}", ""]


def _render_method(rep: dict, out: list[str]) -> None:
    m = _asdict(rep.get("method"))
    if _is_empty(m):
        return
    out += [_h(2, "Method / Approach"), ""]
    if m.get("high_level_flow"):
        out += ["**High-level flow**", ""] + _bullets(m["high_level_flow"]) + [""]
    mods = m.get("modules") or []
    if mods:
        out += ["**Modules**", ""]
        out += ["| Module | Role | Model | Inputs | Outputs | Notes |",
                "|---|---|---|---|---|---|"]
        for md in mods:
            if not isinstance(md, dict):
                out += ["| " + str(md).replace("|", "\\|").replace("\n", " ")
                        + " |  |  |  |  |  |"]
                continue
            row = [md.get("module", ""), md.get("role", ""), md.get("model", ""),
                   ", ".join(md.get("inputs", []) or []),
                   ", ".join(md.get("outputs", []) or []),
                   md.get("notes", "")]
            out += ["| " + " | ".join(str(c).replace("|", "\\|").replace("\n", " ")
                                      for c in row) + " |"]
        out += [""]
    if m.get("novel_architectural_elements"):
        out += ["**Novel architectural elements**", ""] + \
            _bullets(m["novel_architectural_elements"]) + [""]
    if m.get("pseudocode"):
        out += ["**Pseudocode**", "", "```", m["pseudocode"].strip(), "```", ""]
    if m.get("design_choices"):
        out += ["**Key design choices**", ""] + _bullets(m["design_choices"]) + [""]
    if m.get("theoretical_grounding"):
        out += ["**Theoretical grounding**", "", m["theoretical_grounding"], ""]


def _render_model_details(rep: dict, out: list[str]) -> None:
    md = _asdict(rep.get("model_details"))
    if _is_empty(md):
        return
    out += [_h(2, "Model Details"), ""]
    if md.get("base_model"):
        out += [f"- **Base model:** {md['base_model']}"]
    tr = _asdict(md.get("training"))
    if not _is_empty(tr):
        if tr.get("method"):
            out += [f"- **Training method:** {tr['method']}"]
        if tr.get("adaptation"):
            out += [f"- **Adaptation:** {tr['adaptation']}"]
        if tr.get("objective_functions"):
            out += ["- **Objective functions:**"] + \
                _bullets(tr["objective_functions"], indent="  ")
        if tr.get("key_hyperparameters"):
            hp = "; ".join(f"{k}={v}" for k, v in tr["key_hyperparameters"].items())
            out += [f"- **Key hyperparameters:** {hp}"]
    out += [""]


def _render_comparison(rep: dict, out: list[str]) -> None:
    cs = _asdict(rep.get("comparison_to_sota"))
    if _is_empty(cs):
        return
    out += [_h(2, "Comparison to SOTA"), ""]
    if cs.get("related_methods"):
        out += ["**Related methods**", ""] + _bullets(cs["related_methods"]) + [""]
    if cs.get("key_differences"):
        out += ["**Key differences**", ""] + _bullets(cs["key_differences"]) + [""]


def _render_proposed_eval(rep: dict, out: list[str]) -> None:
    pe = _asdict(rep.get("proposed_evaluation"))
    if _is_empty(pe):
        return
    out += [_h(2, "Proposed Evaluation"), "",
            "> *Planned experiments — not yet run; no measured results.*", ""]
    if pe.get("evaluation_setting"):
        out += [pe["evaluation_setting"], ""]
    bm = pe.get("benchmarks") or []
    if bm:
        out += ["**Benchmarks**", ""]
        for b in bm:
            if not isinstance(b, dict):
                out += [f"- {_inline(b)}"]
                continue
            extra = []
            if b.get("task_type"):
                extra.append(b["task_type"])
            if b.get("newly_constructed"):
                extra.append("newly constructed")
            tail = f" ({'; '.join(extra)})" if extra else ""
            out += [f"- {b.get('name', '')}{tail}"]
        out += [""]
    for label, key in [("Baselines", "baselines"),
                       ("Metrics", "main_metrics"), ("Ablations", "ablations")]:
        if pe.get(key):
            out += [f"**{label}**", ""] + _bullets(pe[key]) + [""]


def _render_falsifiable(rep: dict, out: list[str]) -> None:
    fp = rep.get("falsifiable_predictions") or []
    if _is_empty(fp):
        return
    out += [_h(2, "Falsifiable Predictions"), ""]
    out += ["| Pillar | Prediction | Kill condition |", "|---|---|---|"]
    for p in fp:
        if not isinstance(p, dict):
            row = [_inline(p), "", ""]
        else:
            row = [p.get("pillar", ""), p.get("prediction", ""), p.get("kill_condition", "")]
        out += ["| " + " | ".join(str(c).replace("|", "\\|").replace("\n", " ")
                                  for c in row) + " |"]
    out += [""]


def _render_limitations(rep: dict, out: list[str]) -> None:
    lim = rep.get("limitations") or []
    if _is_empty(lim):
        return
    out += [_h(2, "Limitations & Scope"), ""] + _bullets(lim) + [""]


def _render_key_terms(rep: dict, out: list[str]) -> None:
    kt = _asdict(rep.get("key_terms"))
    if _is_empty(kt):
        return
    out += [_h(2, "Key Terms"), ""]
    for term in sorted(kt):
        if not _is_empty(kt[term]):
            out += [f"- **{term}** — {kt[term]}"]
    out += [""]


def _render_references(rep: dict, out: list[str]) -> None:
    refs = rep.get("references") or []
    if _is_empty(refs):
        return
    out += [_h(2, "References"), ""]
    refs = [r for r in refs if isinstance(r, dict)]
    for r in sorted(refs, key=lambda x: str(x.get("short_label", "")).lower()):
        anchor = r.get("anchor", "")
        label = r.get("short_label", "?")
        year = r.get("year")
        title = r.get("title", "")
        date = r.get("date")
        head = f"**[{label}, {year}]**" if year is not None else f"**[{label}]**"
        tail = f" {date}." if date else ""
        out += [f'<a id="ref-{anchor}"></a>{head} {title}.{tail}', ""]


def render_report(rep: dict) -> str:
    """Render a report_writer report dict into a markdown string."""
    out: list[str] = []
    _render_meta(rep, out)
    _render_problem(rep, out)
    _render_novelty(rep, out)
    _render_problem_def(rep, out)
    _render_related_work(rep, out)
    _render_method(rep, out)
    _render_model_details(rep, out)
    _render_comparison(rep, out)
    _render_proposed_eval(rep, out)
    _render_falsifiable(rep, out)
    _render_limitations(rep, out)
    _render_key_terms(rep, out)
    _render_references(rep, out)
    text = "\n".join(out).rstrip() + "\n"
    return text
