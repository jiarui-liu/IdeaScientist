"""Reward for one ``gap_finder`` rollout, against one assigned axis.

Three layers turn ``gaps.md`` into a scalar in [0, 1]:

  1. Deterministic gates, computed in code against the rollout's own files
     before any judge call. The hackable parts — does the rollout stay on the
     assigned axis, do its cited ids resolve, did it actually read them — are
     never left to the judge. One gate failing on one gap vetoes the document.
  2. The judge, scoring the assigned axis' block against the label's block for
     the *same* axis rather than the whole label; the alignment gate is what
     makes that scoping sound.
  3. Aggregation over the document, vetoed outright past MAX_GAPS_PER_AXIS.

A small process term on top gives early rollouts that cannot yet clear a gate
some variance for GRPO to work with.

Pure apart from the judge call and the DB-backed gate helper, both injected, so
the aggregation is testable offline.
"""
from __future__ import annotations

import re
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from ideascientist.rewards.judge import (
    judge_items,
    JUDGE_FAILED_REWARD_SENTINEL,
    JUDGE_MAX_PARSE_RETRIES,
    JudgeParseError,
    render_rubric_items,
)
from ideascientist.utils.llm import LLMSettings, chat_with_usage

from .parser import Axis, Gap, parse_gaps_md
from .reference_rubric import (
    JUDGE_PROMPT_TEMPLATE,
    MATCH_IDS,
    RESEARCH_GAP_RUBRIC,
    RESEARCH_GAP_RUBRIC_COMBINED,
    citation_set_f1,
    score_document_combined,
)
from ideascientist.rewards.judge import judge_groups
from ideascientist.rewards.common import RewardWeights, available_papers_block, cited_ids_resolve, component_breakdown, gate_check_ids, gates_pass


# --------------------------------------------------------------------------- #
# Reward weights + result container
# --------------------------------------------------------------------------- #
# Hard cap on the gaps the agent may emit for one axis. Exceeding it vetoes the
# whole document (rubric reward 0) rather than decaying the score: listing many
# shallow gaps should not out-earn analysing two properly.
MAX_GAPS_PER_AXIS: int = 2



@dataclass
class RewardResult:
    total: float = 0.0
    rubric_reward: float = 0.0
    process_reward: float = 0.0
    # Set-F1 between the generated and reference cited-paper id sets (the citation
    # third of the combined reward). -1.0 means "not computed" (gated/no-gaps path).
    citation_f1: float = -1.0
    # Per-gap breakdown for persistence/inspection.
    n_gaps: int = 0
    n_gaps_passed_gates: int = 0
    per_gap: list[dict] = field(default_factory=list)
    # Raw judge outputs (per gap, per rubric item): {gap_idx: {rubric_id: {...}}}.
    judge_responses: dict[str, Any] = field(default_factory=dict)
    axis_name: str = ""
    error: str = ""
    # True when a batched judge group could not be parsed even after retries. The
    # env turns this into ``JUDGE_FAILED_REWARD_SENTINEL`` so the rollout is DROPPED
    # from the GRPO batch (loss_multiplier=0), never scored with a fabricated value.
    judge_failed: bool = False
    # Per-field + per-component reward breakdown (base_quality, reference_quality,
    # citation_f1, negative_penalty, and every rubric field's judged score) for
    # persistence + TensorBoard. Empty on gated / no-judge paths.
    reward_breakdown: dict = field(default_factory=dict)




def _cited_ids_were_read(ids: list[int], files: dict[str, str]) -> bool:
    """Every cited id has a read_paper digest (``papers/<id>.md``) in files."""
    if not ids:
        return False
    for pid in ids:
        digest = files.get(f"papers/{pid}.md", "")
        if not (isinstance(digest, str) and digest.strip()):
            return False
    return True


def _compute_gates(
    gap: Gap,
    files: dict[str, str],
    biblio_fn: Callable[[int], Optional[dict]],
) -> dict[str, bool]:
    return {
        "schema_valid": gap.has_all_fields(),
        "cited_ids_resolve": cited_ids_resolve(gap.cited_ids, biblio_fn),
        "cited_ids_were_read": _cited_ids_were_read(gap.cited_ids, files),
    }







def _judge_group(
    axis_text: str,
    available_papers: str,
    gap_text: str,
    items: list[dict],
    kind: str,
    settings: LLMSettings,
    *,
    template: str = JUDGE_PROMPT_TEMPLATE,
    reference_gap: str = "",
    max_parse_retries: int = JUDGE_MAX_PARSE_RETRIES,
) -> tuple[dict[str, dict], dict]:
    """Score one group of rubric items in a single call, retrying on parse failure.

    The judge samples rather than decoding greedily, so a retry after a
    malformed reply has a real chance of parsing where a repeat would not. An
    attempt counts only if every id in the group parses; if none do,
    ``JudgeParseError`` carries all attempts so the driver can drop the rollout
    rather than invent scores.
    """
    item_ids = [it["id"] for it in items]
    prompt = (
        template
        .replace("<<axis>>", axis_text)
        .replace("<<available_papers>>", available_papers)
        .replace("<<gap>>", gap_text)
        .replace("<<rubric_items>>", render_rubric_items(items, kind))
        .replace("<<reference_gap>>", reference_gap or "(no reference gap provided)")
    )
    return judge_items(prompt, item_ids, settings,
                       max_parse_retries=max_parse_retries, max_tokens=1024)


def _judge_document(
    gaps: list[Gap],
    axis_text: str,
    available_papers: str,
    settings: LLMSettings,
    rubric: dict,
    *,
    template: str = JUDGE_PROMPT_TEMPLATE,
    reference_gap: str = "",
) -> tuple[dict[str, int], dict[str, dict]]:
    """Score the target axis over every rubric item in three batched calls.

    The items are partitioned into base positives, reference matches, and
    negatives, so each call sees one homogeneous instruction instead of a mixed
    rubric. Returns ``(judgments, raw)``; ``raw["_groups"]`` keeps each group's
    retry count and raw text for offline audit, and a group that never parses
    propagates with the groups that already did.
    """
    judgments: dict[str, int] = {}
    raw: dict[str, dict] = {}
    groups_meta: dict[str, dict] = {}
    doc_text = "\n\n".join(g.raw or g.fields.get("Gap", "") for g in gaps)
    for name, kind, items in judge_groups(rubric, MATCH_IDS):
        if not items:
            continue
        try:
            scored, meta = _judge_group(
                axis_text, available_papers, doc_text, items, kind, settings,
                template=template, reference_gap=reference_gap,
            )
        except JudgeParseError as e:
            # Attach the group name + the groups that DID parse before this one so
            # the caller can persist the complete judge trace for the dropped row.
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
    gaps_md: str,
    axes: list[Axis],
    productive_tool_calls: int = 0,
) -> float:
    """Graded partial credit in [0, 1], for cold-start variance.

    gap_finder saturates the tool-call credit at two calls rather than four:
    a single axis needs a shallower search than a whole proposal, so a higher
    cap would reward depth the task does not have.
    """
    score = 0.10 * min(int(productive_tool_calls or 0), 2)
    if isinstance(gaps_md, str) and len(gaps_md.strip()) >= 200:
        score += 0.20
    all_gaps = [g for ax in axes for g in ax.gaps]
    if all_gaps:
        score += 0.25
    if any(g.has_all_fields() for g in all_gaps):
        score += 0.35
    return min(score, 1.0)


# --------------------------------------------------------------------------- #
# Top-level reward
# --------------------------------------------------------------------------- #
def _reference_cited_ids(
    metadata: dict[str, Any], ref_md: str, axis_name: str
) -> list[int]:
    """The reference (label) cited-paper id set for the assigned axis.

    Prefers the dataset builder's pre-computed per-axis ``reference_cited_ids``
    (see ``build_gap_finder_dataset.build_split``). When that key is absent (e.g.
    an older dataset or an ad-hoc call), fall back to parsing the reference
    ``gaps.md`` and selecting the assigned axis by EXACT name match, so both sides
    are compared on the SAME axis rather than the whole label. Returns [] if
    neither source yields ids (F1 then compares against an empty reference set).
    """
    raw = metadata.get("reference_cited_ids")
    if raw:
        try:
            return sorted({int(i) for i in raw})
        except (TypeError, ValueError):
            pass
    if not ref_md:
        return []
    target = _select_reference_axis(parse_gaps_md(ref_md), axis_name)
    if target is None:
        return []
    return sorted({i for g in target.gaps for i in g.cited_ids})


def _select_target_axis(axes: list[Axis], axis_name: str) -> Optional[Axis]:
    """Pick the axis the rollout was assigned. Match by name (case-insensitive
    substring), else fall back to the single/first axis emitted.

    Reached only AFTER the axis-alignment gate (``_axis_alignment_ok``) has
    confirmed exactly one axis whose name exactly matches ``axis_name``, so in
    practice this returns that single axis; the substring/``axes[0]`` fallbacks
    are dead paths kept for direct/ad-hoc callers.
    """
    if not axes:
        return None
    if axis_name:
        low = axis_name.strip().lower()
        for ax in axes:
            if ax.name.strip().lower() == low:
                return ax
        for ax in axes:
            if low in ax.name.strip().lower() or ax.name.strip().lower() in low:
                return ax
    return axes[0]


def _axis_alignment_ok(axes: list[Axis], axis_name: str) -> bool:
    """Gate: the rollout's ``gaps.md`` must carry EXACTLY ONE ``## Axis:`` block
    whose name EQUALS the assigned axis (case-insensitive, whitespace-trimmed).

    This is the deterministic axis-alignment gate. A rollout that omits the axis,
    renames/rephrases it, or emits more than one axis block is off-task ("跑题")
    and is vetoed upstream in ``compute_reward`` (whole-doc rubric reward 0). It
    lets the reference comparison downstream trust that the generated axis matches
    the assigned one, so the reference is scoped to that same axis.
    """
    if len(axes) != 1:
        return False
    return axes[0].name.strip().lower() == axis_name.strip().lower()


def _select_reference_axis(axes: list[Axis], axis_name: str) -> Optional[Axis]:
    """Pick the assigned axis from the REFERENCE label by EXACT name match
    (case-insensitive, whitespace-trimmed); return None on no match.

    Unlike ``_select_target_axis`` there is NO substring / ``axes[0]`` fallback:
    the dataset builder replays the label's exact axis names
    (``build_gap_finder_dataset.build_split``), so the assigned axis is present
    verbatim in the reference; anything else means we must NOT hand the judge a
    different axis' reference gaps.
    """
    if not axes or not axis_name:
        return None
    low = axis_name.strip().lower()
    for ax in axes:
        if ax.name.strip().lower() == low:
            return ax
    return None


def _render_reference_axis(ax: Axis) -> str:
    """Reconstruct one reference ``## Axis:`` block as the judge's
    ``<<reference_gap>>``, in the SAME schema the generated gap is shown in:
    the axis header, the ``### What has been done`` prose, and each gap's raw
    text under ``### Open gaps on this axis``.
    """
    parts = [f"## Axis: {ax.name}"]
    if ax.what_has_been_done.strip():
        parts.append("### What has been done\n" + ax.what_has_been_done.strip())
    if ax.gaps:
        gap_block = "\n\n".join(g.raw for g in ax.gaps if g.raw.strip())
        parts.append("### Open gaps on this axis\n" + gap_block)
    return "\n\n".join(parts)



def compute_reward(
    files: dict[str, str],
    metadata: dict[str, Any],
    judge_settings: LLMSettings,
    *,
    weights: Optional[RewardWeights] = None,
    biblio_fn: Optional[Callable[[int], Optional[dict]]] = None,
    rubric: Optional[dict] = None,
) -> RewardResult:
    """Score one rollout: gates first, then ``(base + reference + citation) / 3``.

    ``metadata`` supplies the assigned ``axis``, the ``challenge`` and
    ``problem_definition`` the trainee saw, ``reference_gaps_md``, and
    ``productive_tool_calls``. ``biblio_fn`` defaults to the production lookup
    and is injectable so the gates can run without a database.
    """
    w = weights or RewardWeights()
    result = RewardResult()

    rubric = rubric or RESEARCH_GAP_RUBRIC_COMBINED
    judge_template = JUDGE_PROMPT_TEMPLATE
    _score_document = score_document_combined

    if biblio_fn is None:
        from ideascientist.harness.corpus import get_paper_biblio as biblio_fn  # type: ignore

    gaps_md = files.get("gaps.md", "") or ""
    axes = parse_gaps_md(gaps_md)

    # Process/format shaping always applies (even when no gate can pass yet).
    result.process_reward = _process_reward(
        gaps_md, axes, metadata.get("productive_tool_calls", 0)
    )

    axis_name = metadata.get("axis", "") or ""
    # Axis-alignment gate: the rollout must emit EXACTLY ONE ``## Axis:`` block
    # whose name equals the assigned axis (case-insensitive). An off-task ("跑题")
    # or multi-axis rollout is vetoed here (whole-doc rubric reward 0, no judge
    # calls); process reward still carries so GRPO keeps variance. Running it
    # before _select_target_axis means the substring/axes[0] fallbacks there are
    # never reached for a real rollout — the reference can trust the axis matches.
    if axis_name and not _axis_alignment_ok(axes, axis_name):
        result.axis_name = axis_name
        result.rubric_reward = 0.0
        result.total = w.process * result.process_reward
        result.error += "axis_alignment_gate_failed\n"
        return result
    # Grounding: give the judge the SAME problem framing the trainee got — the
    # results-masked challenge + problem_definition (matches the label pipeline). Fall
    # back to the axis name alone when a field is absent.
    axis_text = axis_name
    grounding_parts = []
    if metadata.get("challenge"):
        grounding_parts.append(f"Challenge: {metadata['challenge']}")
    if metadata.get("problem_definition"):
        grounding_parts.append(f"Problem definition: {metadata['problem_definition']}")
    if grounding_parts:
        axis_text = axis_name + "\n\n" + "\n\n".join(grounding_parts)
    ref_md = metadata.get("reference_gaps_md", "") or ""
    # The reference is a first-class scoring anchor supplied via the judge's
    # dedicated ``<<reference_gap>>`` block, kept OUT of the axis text. Scope it to
    # the assigned axis' block ONLY (not the whole label): the axis-alignment gate
    # above guarantees the generated axis matches, so the judge's per-field
    # ``matches_reference_*`` comparison is against the RIGHT reference axis and
    # cannot farm credit from a sibling axis' gaps.
    ref_axis = _select_reference_axis(parse_gaps_md(ref_md), axis_name)
    reference_gap = (
        _render_reference_axis(ref_axis) if ref_axis
        else "(no reference gap for this axis)"
    )

    target = _select_target_axis(axes, axis_name)
    if target is None or not target.gaps:
        # Nothing parseable to score → rubric reward 0, process reward carries it.
        result.axis_name = axis_name
        result.total = w.process * result.process_reward
        return result

    result.axis_name = target.name
    result.n_gaps = len(target.gaps)

    try:
        # 1a) Gap-count gate: more than the target number of gaps for the axis
        #     vetoes the whole document, before any judge call.
        if result.n_gaps > MAX_GAPS_PER_AXIS:
            result.rubric_reward = 0.0
            result.total = (
                w.rubric * result.rubric_reward + w.process * result.process_reward
            )
            return result

        # 1b) Deterministic gates across ALL gaps. ANY gate failure on ANY gap
        #    vetoes the WHOLE document (rubric reward 0, no judge calls) — global,
        #    in-code gating.
        all_gates_pass = True
        for gi, gap in enumerate(target.gaps):
            det = _compute_gates(gap, files, biblio_fn)
            gap_detail = {
                "gap_index": gi,
                "gates": det,
                "cited_ids": gap.cited_ids,
                "missing_fields": gap.missing_fields(),
            }
            passed = gates_pass(det)
            gap_detail["gated"] = not passed
            result.per_gap.append(gap_detail)
            if passed:
                result.n_gaps_passed_gates += 1
            else:
                all_gates_pass = False

        if not all_gates_pass:
            # A bad gap zeroes the document; process reward still carries.
            result.rubric_reward = 0.0
            result.total = (
                w.rubric * result.rubric_reward + w.process * result.process_reward
            )
            return result

        # 2) Judge the WHOLE gaps.md once, doc-to-doc, over every rubric item.
        avail_all = available_papers_block(
            sorted({i for g in target.gaps for i in g.cited_ids}), files
        )
        judgments, raw = _judge_document(
            target.gaps, axis_text, avail_all, judge_settings, rubric,
            template=judge_template, reference_gap=reference_gap,
        )
        result.judge_responses["document"] = raw

        # 2b) Citation third: set-F1 of the generated vs reference cited-paper ids.
        #     Deterministic (no judge). The reference set is the per-axis
        #     ``reference_cited_ids`` supplied by the dataset builder; when absent we
        #     fall back to parsing the reference label and selecting the same axis.
        gen_ids = sorted({i for g in target.gaps for i in g.cited_ids})
        ref_ids = _reference_cited_ids(metadata, ref_md, axis_name)
        result.citation_f1 = citation_set_f1(gen_ids, ref_ids)

        # 3) Document-level aggregation. All gates pass and n_gaps is within the
        #    hard cap here, so pass an all-True ``det``; the code-counted depth
        #    decay is now a no-op (the gap-count gate handles breadth) and only the
        #    axis-scope laundry multiplier can still adjust the score.
        det_all = {cid: True for cid in gate_check_ids(rubric)}
        result.rubric_reward = _score_document(
            result.n_gaps, judgments, det=det_all,
            citation_f1=result.citation_f1,
            component_weights=w.component_weights(), rubric=rubric,
        )
        # Per-field + per-component breakdown for persistence + TensorBoard. Same
        # inputs as _score_document, so its aggregates reconcile with rubric_reward.
        result.reward_breakdown = component_breakdown(
            det_all, judgments,
            base_rubric=RESEARCH_GAP_RUBRIC, combined_rubric=rubric, scope="gap",
            citation_f1=result.citation_f1, component_weights=w.component_weights(),
        )
    except JudgeParseError as e:
        # The judge could not be parsed after retries: DROP this rollout from the
        # GRPO batch instead of scoring it. The env reads ``judge_failed`` and emits
        # the sentinel reward; the rollout driver zeroes its loss_multiplier. Do NOT
        # let process_reward leak into ``total`` here — a dropped row's total is
        # replaced by the sentinel downstream, so leave it at the sentinel-signaling
        # default and record why. Persist the FULL judge trace (every attempt's raw
        # text for the failed group + the groups that already parsed) so the dropped
        # rollout is debuggable offline.
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
