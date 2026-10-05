"""Reward for one ``report_writer`` rollout: one results-masked proposal.

Same three layers as the gap_finder reward, with the unit being the whole
``report.json``. The deterministic gates run in code first — does it parse, is
every required field present, does each cited id resolve, and was it grounded.
Grounding here is looser than for the other two roles: an id already named in
the seeded gaps.md or candidates.md counts, since those are the writer's
problem statement rather than something it had to go and find. Fabricated
numbers are not gated — see ``predictions_as_results`` in rubric.py.

The judge then scores base positives, per-field reference matches, and
negatives in three batched calls, and ``score_document_combined`` gates
globally before taking the weighted mean. report_writer has no gate positive,
so only the deterministic gates can veto.
"""
from __future__ import annotations

import json
import re
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from ideascientist.rewards.judge import (
    judge_items,
    JUDGE_MAX_PARSE_RETRIES,
    JudgeParseError,
    render_rubric_items,
)
from ideascientist.utils.llm import LLMSettings

from .parser import Report, parse_report_text
from .reference_rubric import (
    JUDGE_PROMPT_TEMPLATE,
    MATCH_IDS,
    SOLUTION_PROPOSAL_RUBRIC,
    SOLUTION_PROPOSAL_RUBRIC_COMBINED,
    citation_set_f1,
    score_document_combined,
)
from ideascientist.rewards.judge import judge_groups
from ideascientist.rewards.common import RewardWeights, available_papers_block, cited_ids_resolve, component_breakdown, gate_check_ids, gates_pass, ids_in_seeded_docs


# --------------------------------------------------------------------------- #
# Reward weights + result container
# --------------------------------------------------------------------------- #
# Judge decoding + retry policy (identical to gap_finder / innovator).
# Terminal sentinel reward the env emits when the judge cannot be parsed after
# retries. Out of the legitimate reward range so the driver drops the rollout.
# Unlike citations, fabricated numbers are not gated deterministically. A regex
# cannot separate a quantitative prediction from a claimed measurement, and
# ``falsifiable_predictions`` is supposed to contain the former, so a pattern
# gate zeroes well-formed proposals. The graded ``predictions_as_results``
# negative in rubric.py makes that distinction instead.




@dataclass
class RewardResult:
    total: float = 0.0
    rubric_reward: float = 0.0
    process_reward: float = 0.0
    citation_f1: float = -1.0
    gates: dict[str, bool] = field(default_factory=dict)
    gated: bool = False
    missing_fields: list[str] = field(default_factory=list)
    judge_responses: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    judge_failed: bool = False
    # Per-field + per-component reward breakdown for persistence + TensorBoard.
    reward_breakdown: dict = field(default_factory=dict)








def _cited_ids_were_read(
    ids: list[int],
    files: dict[str, str],
    seed_docs: tuple[str, ...] = ("gaps.md", "candidates.md"),
) -> bool:
    """Every cited id is GROUNDED: read via read_paper (a ``papers/<id>.md`` digest
    exists in files) OR already present in a seeded input doc (gaps.md / candidates.md),
    whose content the agent was handed. Empty citations still fail.

    Mirrors innovator's ``source_ids_were_read`` (relaxed for seeded inputs); the extra
    seed doc is candidates.md, which report_writer also receives as an input artifact.
    """
    if not ids:
        return False
    exempt = ids_in_seeded_docs(files, seed_docs)
    for pid in ids:
        if pid in exempt:
            continue
        digest = files.get(f"papers/{pid}.md", "")
        if not (isinstance(digest, str) and digest.strip()):
            return False
    return True


def _compute_gates(
    report: Report,
    biblio_fn: Callable[[int], Optional[dict]],
    files: dict[str, str],
) -> dict[str, bool]:
    gates = {
        "json_parses": report.parses,
        "schema_valid": report.has_all_fields(),
        "cited_ids_resolve": cited_ids_resolve(report.cited_ids, biblio_fn),
        "cited_ids_were_read": _cited_ids_were_read(report.cited_ids, files),
    }
    # Offline-baseline scoring only: the read-grounding gate assumes OUR harness
    # layout, where a cited paper first gets a read_paper digest (papers/<id>.md)
    # or appears in seeded gaps.md / candidates.md. External baselines cite real
    # DB papers but leave no such per-paper artifacts, so this one gate trips on
    # ~100% of their reports regardless of quality. When RW_DROP_CITED_READ_GATE
    # is set we drop THIS gate only (the other three still apply). Default OFF, so
    # training and production reward computation are completely unchanged.
    import os
    if os.environ.get("RW_DROP_CITED_READ_GATE", "").strip().lower() in ("1", "true", "yes"):
        gates.pop("cited_ids_were_read", None)
    return gates







def _judge_group(
    problem_text: str,
    gap_text: str,
    intuition_text: str,
    reference_proposal: str,
    available_papers: str,
    proposal_text: str,
    items: list[dict],
    kind: str,
    settings: LLMSettings,
    *,
    template: str = JUDGE_PROMPT_TEMPLATE,
    max_parse_retries: int = JUDGE_MAX_PARSE_RETRIES,
) -> tuple[dict[str, dict], dict]:
    item_ids = [it["id"] for it in items]
    prompt = (
        template
        .replace("<<problem>>", problem_text or "(no problem text provided)")
        .replace("<<gap>>", gap_text or "(no gap text provided)")
        .replace("<<intuition>>", intuition_text or "(no intuition text provided)")
        .replace("<<reference_proposal>>", reference_proposal or "(no reference proposal provided)")
        .replace("<<available_papers>>", available_papers)
        .replace("<<proposal>>", proposal_text)
        .replace("<<rubric_items>>", render_rubric_items(items, kind))
    )
    return judge_items(prompt, item_ids, settings,
                       max_parse_retries=max_parse_retries, max_tokens=max(1024, 256 * len(item_ids)))


def _judge_document(
    proposal_text: str,
    problem_text: str,
    gap_text: str,
    intuition_text: str,
    reference_proposal: str,
    available_papers: str,
    settings: LLMSettings,
    rubric: dict,
    *,
    template: str = JUDGE_PROMPT_TEMPLATE,
) -> tuple[dict[str, int], dict[str, dict]]:
    judgments: dict[str, int] = {}
    raw: dict[str, dict] = {}
    groups_meta: dict[str, dict] = {}
    for name, kind, items in judge_groups(rubric, MATCH_IDS):
        if not items:
            continue
        try:
            scored, meta = _judge_group(
                problem_text, gap_text, intuition_text, reference_proposal,
                available_papers, proposal_text, items, kind, settings,
                template=template,
            )
        except JudgeParseError as e:
            e.group_name = name
            e.groups_meta_so_far = groups_meta
            raise
        groups_meta[name] = meta
        for rid, rec in scored.items():
            judgments[rid] = rec["score"]
            raw[rid] = rec
    raw["_groups"] = groups_meta
    return judgments, raw


# --------------------------------------------------------------------------- #
# Process / format shaping
# --------------------------------------------------------------------------- #
def _process_reward(
    report_text: str,
    report: Report,
    productive_tool_calls: int = 0,
) -> float:
    """Graded partial credit in [0, 1] for cold-start GRPO variance.

      0.00–0.20  productive tool calls: min(n, 4) * 0.05
      0.20  report.json is non-trivial (>= 200 chars)
      0.25  report.json parses as a JSON object
      0.35  every required results-masked field is present (schema-complete)
    """
    score = 0.05 * min(int(productive_tool_calls or 0), 4)
    if isinstance(report_text, str) and len(report_text.strip()) >= 200:
        score += 0.20
    if report.parses:
        score += 0.25
    if report.has_all_fields():
        score += 0.35
    return min(score, 1.0)


# --------------------------------------------------------------------------- #
# Reference helpers
# --------------------------------------------------------------------------- #
def _reference_cited_ids(metadata: dict[str, Any], ref_json: str) -> list[int]:
    """The reference (label) cited-paper anchor id set for this proposal.

    Prefers the dataset builder's pre-computed ``reference_cited_ids``. When
    absent, fall back to parsing the reference ``report.json`` and using its
    anchor ids. Returns [] if neither source yields ids.
    """
    raw = metadata.get("reference_cited_ids")
    if raw:
        try:
            return sorted({int(i) for i in raw})
        except (TypeError, ValueError):
            pass
    if not ref_json:
        return []
    ref = parse_report_text(ref_json)
    return sorted(set(ref.cited_ids))


def compute_reward(
    files: dict[str, str],
    metadata: dict[str, Any],
    judge_settings: LLMSettings,
    *,
    weights: Optional[RewardWeights] = None,
    biblio_fn: Optional[Callable[[int], Optional[dict]]] = None,
    rubric: Optional[dict] = None,
) -> RewardResult:
    """Compute the report_writer rollout reward (COMBINED, reference-anchored).

    Args:
        files: in-memory filesystem {path: content}; must contain ``report.json``
            (the artifact) and ``papers/<id>.md`` read digests.
        metadata: per-sample metadata. Uses ``problem`` + ``gap`` + ``intuition``
            (upstream framing, for judge grounding), ``reference_report_json``
            (validated label, supplied to the judge as a scoring anchor),
            ``reference_cited_ids``, and ``productive_tool_calls``.
        judge_settings: LLMSettings pointing at a served Qwen3.6-27B judge.
        weights: reward component weights (default RewardWeights()).
        biblio_fn: get_paper_biblio(id) -> dict|None. Defaults to the production
            capability (lazy-imported); injectable for offline tests.
        rubric: rubric dict to score against. Defaults to
            ``SOLUTION_PROPOSAL_RUBRIC_COMBINED``.
    """
    w = weights or RewardWeights()
    result = RewardResult()

    rubric = rubric or SOLUTION_PROPOSAL_RUBRIC_COMBINED
    judge_template = JUDGE_PROMPT_TEMPLATE
    _score_document = score_document_combined

    if biblio_fn is None:
        from ideascientist.harness.corpus import get_paper_biblio as biblio_fn  # type: ignore

    report_text = files.get("report.json", "") or ""
    report = parse_report_text(report_text)

    # Process/format shaping always applies (even when no gate can pass yet).
    result.process_reward = _process_reward(
        report_text, report, metadata.get("productive_tool_calls", 0)
    )
    result.missing_fields = report.missing_fields()

    # Grounding framing for the judge.
    problem_text = metadata.get("problem", "") or ""
    gap_text = metadata.get("gap", "") or ""
    intuition_text = metadata.get("intuition", "") or ""
    ref_json = metadata.get("reference_report_json", "") or ""
    reference_proposal = ref_json if ref_json.strip() else "(no reference proposal)"

    try:
        # 1) Deterministic gates. ANY gate failure vetoes the WHOLE proposal
        #    (rubric reward 0, no judge calls). report_writer has no gate positive,
        #    so this is the entire global gate.
        det = _compute_gates(report, biblio_fn, files)
        result.gates = det
        passed = gates_pass(det)
        result.gated = not passed
        if not passed:
            result.rubric_reward = 0.0
            result.total = (
                w.rubric * result.rubric_reward + w.process * result.process_reward
            )
            return result

        # 2) Judge the proposal once, doc-to-doc, over every rubric item.
        avail_all = available_papers_block(report.cited_ids, files)
        proposal_text = json.dumps(report.obj, ensure_ascii=False, indent=2)
        judgments, raw = _judge_document(
            proposal_text, problem_text, gap_text, intuition_text,
            reference_proposal, avail_all, judge_settings, rubric,
            template=judge_template,
        )
        result.judge_responses["document"] = raw

        # 2b) Citation third: set-F1 of the generated vs reference anchor ids.
        gen_ids = sorted(set(report.cited_ids))
        ref_ids = _reference_cited_ids(metadata, ref_json)
        result.citation_f1 = citation_set_f1(gen_ids, ref_ids)

        # 3) Document-level aggregation. All gates pass here → all-True ``det``.
        det_all = {cid: True for cid in gate_check_ids(rubric)}
        result.rubric_reward = _score_document(
            1, judgments, det=det_all,
            citation_f1=result.citation_f1,
            component_weights=w.component_weights(), rubric=rubric,
        )
        # Per-field + per-component breakdown (reconciles with rubric_reward).
        result.reward_breakdown = component_breakdown(
            det_all, judgments,
            base_rubric=SOLUTION_PROPOSAL_RUBRIC, combined_rubric=rubric,
            scope="proposal",
            citation_f1=result.citation_f1, component_weights=w.component_weights(),
        )
    except JudgeParseError as e:
        result.judge_failed = True
        result.judge_responses["document"] = {
            "_groups": e.groups_meta_so_far,
            "_failed_group": {
                "group_name": e.group_name,
                "group_ids": e.group_ids,
                "last_reason": e.last_reason,
                "attempts": e.attempts,
            },
        }
        result.error += f"judge_parse_failed: {e}\n"
        return result
    except Exception:
        result.error += f"reward: {traceback.format_exc()}\n"

    result.total = (
        w.rubric * result.rubric_reward + w.process * result.process_reward
    )
    return result
