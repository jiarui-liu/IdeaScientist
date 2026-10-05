"""Reward for one ``innovator`` rollout: one intuition attacking one gap.

Same three layers as the gap_finder reward, with the unit being a candidate
rather than an axis. The deterministic gates run in code first — schema,
whether each cited id resolves, whether it was actually read, and whether the
candidate names a gap — because those are the parts a rollout could otherwise
fabricate past a judge. Emitting more than one candidate vetoes too: the job is
one intuition per call.

The judge then scores base positives, per-field reference matches, and
negatives in three batched calls, and ``score_document_combined`` gates
globally before taking the weighted mean.
"""
from __future__ import annotations

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

from .parser import Candidate, parse_candidates_md
from .reference_rubric import (
    JUDGE_PROMPT_TEMPLATE,
    MATCH_IDS,
    RESEARCH_INTUITION_RUBRIC,
    RESEARCH_INTUITION_RUBRIC_COMBINED,
    citation_set_f1,
    score_document_combined,
)
from ideascientist.rewards.judge import judge_groups
from ideascientist.rewards.common import RewardWeights, available_papers_block, cited_ids_resolve, component_breakdown, gate_check_ids, gates_pass, ids_in_seeded_docs


# --------------------------------------------------------------------------- #
# Reward weights + result container
# --------------------------------------------------------------------------- #
# Hard cap on the number of candidates one innovator call may emit. The innovator's
# job is ONE clearly-stated intuition per call ("not a menu of N options"), so
# emitting more than one vetoes the WHOLE document (rubric reward 0).
MAX_CANDIDATES_PER_CALL: int = 1



@dataclass
class RewardResult:
    total: float = 0.0
    rubric_reward: float = 0.0
    process_reward: float = 0.0
    # Set-F1 between the generated and reference cited-paper id sets. -1.0 means
    # "not computed" (gated/no-candidate path).
    citation_f1: float = -1.0
    n_candidates: int = 0
    n_passed_gates: int = 0
    per_candidate: list[dict] = field(default_factory=list)
    judge_responses: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    # True when a batched judge group could not be parsed even after retries.
    judge_failed: bool = False
    # Per-field + per-component reward breakdown for persistence + TensorBoard.
    reward_breakdown: dict = field(default_factory=dict)








def _cited_ids_were_read(
    ids: list[int], files: dict[str, str], seed_docs: tuple[str, ...] = ("gaps.md",)
) -> bool:
    """Every cited id is GROUNDED: either read via read_paper (a ``papers/<id>.md``
    digest exists in files) OR already present in a seeded input doc (gaps.md),
    whose content the agent was handed. Empty citations still fail.
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


def _gap_exists(cand: Candidate, files: dict[str, str]) -> bool:
    """The candidate names a gap to attack (non-empty field) and a gaps.md exists.

    Deterministic proxy for the ``gap_exists`` gate: the candidate must carry a
    non-empty "Gap it attacks" field AND the run must have been supplied a
    non-empty gaps.md to anchor to. Whether the named gap genuinely matches a
    gaps.md gap (and its stakes) is judged downstream by ``grounded_gap_and_stakes``.
    """
    gaps_md = files.get("gaps.md", "")
    if not (isinstance(gaps_md, str) and gaps_md.strip()):
        return False
    return bool((cand.fields.get("Gap it attacks") or "").strip())


def _compute_gates(
    cand: Candidate,
    files: dict[str, str],
    biblio_fn: Callable[[int], Optional[dict]],
) -> dict[str, bool]:
    return {
        "schema_valid": cand.has_all_fields(),
        "source_ids_resolve": cited_ids_resolve(cand.cited_ids, biblio_fn),
        "source_ids_were_read": _cited_ids_were_read(cand.cited_ids, files),
        "gap_exists": _gap_exists(cand, files),
    }







def _judge_group(
    gap_text: str,
    available_papers: str,
    candidate_text: str,
    items: list[dict],
    kind: str,
    settings: LLMSettings,
    *,
    template: str = JUDGE_PROMPT_TEMPLATE,
    reference_candidate: str = "",
    max_parse_retries: int = JUDGE_MAX_PARSE_RETRIES,
) -> tuple[dict[str, dict], dict]:
    """Call the judge for a GROUP of rubric items, retrying on parse failure.

    All items in the group are rendered into ``<<rubric_items>>`` and scored in a
    single reply keyed by id. The judge SAMPLES a fresh reply each attempt, so a
    retry after a malformed reply has a real chance of parsing.

    Returns ``(scored, meta)`` on success. Raises ``JudgeParseError`` (carrying
    every attempt's raw text) if no attempt fully parses.
    """
    item_ids = [it["id"] for it in items]
    prompt = (
        template
        .replace("<<gap>>", gap_text)
        .replace("<<reference_candidate>>", reference_candidate or "(no reference candidate provided)")
        .replace("<<available_papers>>", available_papers)
        .replace("<<candidate>>", candidate_text)
        .replace("<<rubric_items>>", render_rubric_items(items, kind))
    )
    return judge_items(prompt, item_ids, settings,
                       max_parse_retries=max_parse_retries, max_tokens=1024)


def _judge_document(
    candidate_text: str,
    gap_text: str,
    available_papers: str,
    settings: LLMSettings,
    rubric: dict,
    *,
    template: str = JUDGE_PROMPT_TEMPLATE,
    reference_candidate: str = "",
) -> tuple[dict[str, int], dict[str, dict]]:
    """Judge the candidate over every rubric item in THREE batched calls.

    The generated candidate is scored against the reference candidate doc-to-doc.
    Items are partitioned by ``judge_groups`` into three homogeneous batches —
    base positives, matches_reference_*, negatives — each scored in a single call
    whose reply is keyed by item id.

    Returns ``(judgments, raw)``. ``raw`` carries a per-group audit under
    ``raw["_groups"][group_name]``. On a group that never parses,
    ``JudgeParseError`` propagates with the failed group's raw attempts AND the
    groups that already parsed.
    """
    judgments: dict[str, int] = {}
    raw: dict[str, dict] = {}
    groups_meta: dict[str, dict] = {}
    for name, kind, items in judge_groups(rubric, MATCH_IDS):
        if not items:
            continue
        try:
            scored, meta = _judge_group(
                gap_text, available_papers, candidate_text, items, kind, settings,
                template=template, reference_candidate=reference_candidate,
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
    candidates_md: str,
    candidates: list[Candidate],
    productive_tool_calls: int = 0,
) -> float:
    """Graded partial credit in [0, 1] for cold-start GRPO variance.

      0.00–0.20  productive tool calls: min(n, 4) * 0.05
      0.20  candidates.md is non-trivial (>= 200 chars)
      0.25  at least one candidate parsed
      0.35  at least one candidate has ALL required fields (schema-complete)
    """
    score = 0.05 * min(int(productive_tool_calls or 0), 4)
    if isinstance(candidates_md, str) and len(candidates_md.strip()) >= 200:
        score += 0.20
    if candidates:
        score += 0.25
    if any(c.has_all_fields() for c in candidates):
        score += 0.35
    return min(score, 1.0)


# --------------------------------------------------------------------------- #
# Reference selection helpers
# --------------------------------------------------------------------------- #
def _reference_cited_ids(metadata: dict[str, Any], ref_md: str) -> list[int]:
    """The reference (label) cited-paper id set for this gap.

    Prefers the dataset builder's pre-computed ``reference_cited_ids``. When
    absent, fall back to parsing the reference ``candidates.md`` and unioning all
    candidate cited ids. Returns [] if neither source yields ids.
    """
    raw = metadata.get("reference_cited_ids")
    if raw:
        try:
            return sorted({int(i) for i in raw})
        except (TypeError, ValueError):
            pass
    if not ref_md:
        return []
    return sorted({i for c in parse_candidates_md(ref_md) for i in c.cited_ids})


def _select_target_candidate(candidates: list[Candidate]) -> Optional[Candidate]:
    """Pick the candidate the rollout produced.

    The innovator emits ONE new C<n> per call. Reached only after the count gate
    confirms exactly one candidate, so this returns that candidate; the ``[-1]``
    (newest) fallback is for direct/ad-hoc callers.
    """
    if not candidates:
        return None
    return candidates[-1]


def compute_reward(
    files: dict[str, str],
    metadata: dict[str, Any],
    judge_settings: LLMSettings,
    *,
    weights: Optional[RewardWeights] = None,
    biblio_fn: Optional[Callable[[int], Optional[dict]]] = None,
    rubric: Optional[dict] = None,
) -> RewardResult:
    """Compute the innovator rollout reward (COMBINED, reference-anchored).

    Args:
        files: in-memory filesystem {path: content}; must contain ``candidates.md``
            (the artifact), ``gaps.md`` (the input the intuition attacks), and
            ``papers/<id>.md`` read digests.
        metadata: per-sample metadata. Uses ``gap`` (the assigned gap text, for
            judge grounding), ``reference_candidates_md`` (validated label,
            supplied to the judge as a scoring anchor), ``reference_cited_ids``,
            and ``productive_tool_calls``.
        judge_settings: LLMSettings pointing at a served Qwen3.6-27B judge.
        weights: reward component weights (default RewardWeights()).
        biblio_fn: get_paper_biblio(id) -> dict|None. Defaults to the production
            capability (lazy-imported); injectable for offline tests.
        rubric: rubric dict to score against. Defaults to
            ``RESEARCH_INTUITION_RUBRIC_COMBINED``.
    """
    w = weights or RewardWeights()
    result = RewardResult()

    rubric = rubric or RESEARCH_INTUITION_RUBRIC_COMBINED
    judge_template = JUDGE_PROMPT_TEMPLATE
    _score_document = score_document_combined

    if biblio_fn is None:
        from ideascientist.harness.corpus import get_paper_biblio as biblio_fn  # type: ignore

    candidates_md = files.get("candidates.md", "") or ""
    candidates = parse_candidates_md(candidates_md)

    # Process/format shaping always applies (even when no gate can pass yet).
    result.process_reward = _process_reward(
        candidates_md, candidates, metadata.get("productive_tool_calls", 0)
    )
    result.n_candidates = len(candidates)

    # Grounding: give the judge the assigned gap the trainee got.
    gap_text = metadata.get("gap", "") or "(no gap text provided)"
    ref_md = metadata.get("reference_candidates_md", "") or ""
    reference_candidate = ref_md if ref_md.strip() else "(no reference candidate for this gap)"

    target = _select_target_candidate(candidates)
    if target is None:
        # Nothing parseable to score → rubric reward 0, process reward carries.
        result.total = w.process * result.process_reward
        return result

    try:
        # 1a) Candidate-count gate: emitting more than one candidate vetoes the
        #     WHOLE document (rubric reward 0, no judge calls). The innovator's job
        #     is ONE intuition per call.
        if result.n_candidates > MAX_CANDIDATES_PER_CALL:
            result.rubric_reward = 0.0
            result.total = (
                w.rubric * result.rubric_reward + w.process * result.process_reward
            )
            result.error += "candidate_count_gate_failed\n"
            return result

        # 1b) Deterministic gates. ANY gate failure vetoes the WHOLE document
        #     (rubric reward 0, no judge calls).
        det = _compute_gates(target, files, biblio_fn)
        cand_detail = {
            "gates": det,
            "cited_ids": target.cited_ids,
            "missing_fields": target.missing_fields(),
        }
        passed = gates_pass(det)
        cand_detail["gated"] = not passed
        result.per_candidate.append(cand_detail)
        if passed:
            result.n_passed_gates += 1
        else:
            result.rubric_reward = 0.0
            result.total = (
                w.rubric * result.rubric_reward + w.process * result.process_reward
            )
            return result

        # 2) Judge the candidate once, doc-to-doc, over every rubric item.
        avail_all = available_papers_block(target.cited_ids, files)
        candidate_text = target.raw or "\n".join(
            f"**{k}**: {v}" for k, v in target.fields.items()
        )
        judgments, raw = _judge_document(
            candidate_text, gap_text, avail_all, judge_settings, rubric,
            template=judge_template, reference_candidate=reference_candidate,
        )
        result.judge_responses["document"] = raw

        # 2b) Citation third: set-F1 of the generated vs reference cited-paper ids.
        gen_ids = sorted(set(target.cited_ids))
        ref_ids = _reference_cited_ids(metadata, ref_md)
        result.citation_f1 = citation_set_f1(gen_ids, ref_ids)

        # 3) Document-level aggregation. All gates pass here, so pass an all-True
        #    ``det``; the global gate downstream keys on the deterministic gates
        #    plus the methodological_contribution gate positive.
        det_all = {cid: True for cid in gate_check_ids(rubric)}
        result.rubric_reward = _score_document(
            result.n_candidates, judgments, det=det_all,
            citation_f1=result.citation_f1,
            component_weights=w.component_weights(), rubric=rubric,
        )
        # Per-field + per-component breakdown (reconciles with rubric_reward).
        result.reward_breakdown = component_breakdown(
            det_all, judgments,
            base_rubric=RESEARCH_INTUITION_RUBRIC, combined_rubric=rubric,
            scope="candidate",
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
