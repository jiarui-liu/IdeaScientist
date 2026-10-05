"""The three per-role GRPO environments.

All the rollout machinery — tool parsing, observation wrapping, concurrent
paper reading, the phase gate, per-rollout persistence — lives in
:class:`ideascientist.training.environment.BaseAgentEnvironment`. Each role
only pins what genuinely differs: which artifact it must write, which upstream
artifacts are seeded into its workspace, which reward module grades it, and
which reward fields are worth logging.

``INPUT_SEEDS`` is the substantive difference between them. The gap finder
starts from nothing but its assigned axis. The innovator is seeded with the
reference gap analysis, so the gap it attacks is fixed and its reward's
gap-exists gate has something to key on. The report writer is seeded with both
upstream artifacts. Seeding the *reference* artifact rather than a generated
one is what makes each role trainable in isolation: it always receives the
input the harness would have given it, not a noisy upstream sample.
"""

from __future__ import annotations

import ray

from ideascientist.training.environment import BaseAgentEnvironment

_RAY_OPTIONS = dict(max_restarts=-1, max_task_retries=-1, max_concurrency=1000)

_SHARED_PERSIST_FIELDS = ("base_quality", "reference_quality", "negative_penalty")


def _breakdown(details: dict) -> dict:
    bd = details.get("reward_breakdown") or {}
    return {k: bd.get(k) for k in _SHARED_PERSIST_FIELDS}


@ray.remote(**_RAY_OPTIONS)  # pragma: no cover - requires a Ray cluster
class GapFinderEnvironment(BaseAgentEnvironment):
    """Grades the ``gaps.md`` written for one assigned challenge axis."""

    ROLE = "gap_finder"
    REQUIRED_ARTIFACT = "gaps.md"
    INPUT_SEEDS: dict[str, str] = {}
    REWARD_MODULE = "ideascientist.rewards.gap_finder.reward"
    AGG_KEYS = ("total", "rubric_reward", "process_reward", "n_gaps", "n_gaps_passed_gates")
    SURVEY_PROMPT = (
        "You are a related-work and gap sub-agent. Survey the assigned challenge "
        "axis deeply: read the relevant papers, identify what has been done and "
        "what methodological gaps remain. Write your findings to gaps.md."
    )

    def _build_details(self, result) -> dict:
        return {
            "total": result.total,
            "rubric_reward": result.rubric_reward,
            "process_reward": result.process_reward,
            "citation_f1": result.citation_f1,
            "n_gaps": result.n_gaps,
            "n_gaps_passed_gates": result.n_gaps_passed_gates,
            "per_gap": result.per_gap,
            "judge_responses": result.judge_responses,
            "axis_name": result.axis_name,
            "judge_failed": result.judge_failed,
            "reward_breakdown": result.reward_breakdown,
            "error": result.error,
        }

    def _persist_extra(self, meta, details) -> dict:
        return {
            "axis": meta.get("axis", ""),
            "n_gaps": details.get("n_gaps"),
            "n_gaps_passed_gates": details.get("n_gaps_passed_gates"),
            **_breakdown(details),
        }


@ray.remote(**_RAY_OPTIONS)  # pragma: no cover - requires a Ray cluster
class InnovatorEnvironment(BaseAgentEnvironment):
    """Grades one research intuition written for one assigned gap."""

    ROLE = "innovator"
    REQUIRED_ARTIFACT = "candidates.md"
    INPUT_SEEDS = {"gaps.md": "reference_gaps_md"}
    REWARD_MODULE = "ideascientist.rewards.innovator.reward"
    AGG_KEYS = ("total", "rubric_reward", "process_reward",
                "citation_f1", "n_candidates", "n_passed_gates")
    SURVEY_PROMPT = (
        "You are an innovator sub-agent. Given an assigned gap, form ONE clearly "
        "stated, well-sourced research intuition that attacks it, grounded in "
        "papers you read. Write it to candidates.md."
    )

    def _build_details(self, result) -> dict:
        return {
            "total": result.total,
            "rubric_reward": result.rubric_reward,
            "process_reward": result.process_reward,
            "citation_f1": result.citation_f1,
            "n_candidates": result.n_candidates,
            "n_passed_gates": result.n_passed_gates,
            "per_candidate": result.per_candidate,
            "judge_responses": result.judge_responses,
            "judge_failed": result.judge_failed,
            "reward_breakdown": result.reward_breakdown,
            "error": result.error,
        }

    def _persist_extra(self, meta, details) -> dict:
        return {
            "gap": meta.get("gap", ""),
            "citation_f1": details.get("citation_f1"),
            "n_candidates": details.get("n_candidates"),
            "n_passed_gates": details.get("n_passed_gates"),
            **_breakdown(details),
        }


@ray.remote(**_RAY_OPTIONS)  # pragma: no cover - requires a Ray cluster
class ReportWriterEnvironment(BaseAgentEnvironment):
    """Grades the proposal written for an assigned gap and intuition."""

    ROLE = "report_writer"
    REQUIRED_ARTIFACT = "report.json"
    INPUT_SEEDS = {
        "gaps.md": "reference_gaps_md",
        "candidates.md": "reference_candidates_md",
    }
    REWARD_MODULE = "ideascientist.rewards.report_writer.reward"
    AGG_KEYS = ("total", "rubric_reward", "process_reward", "citation_f1")
    SURVEY_PROMPT = (
        "You are a report-writer sub-agent. Given an assigned gap and a research "
        "intuition, write ONE well-grounded, results-masked solution proposal to "
        "report.json, citing the papers you read."
    )

    def _build_details(self, result) -> dict:
        return {
            "total": result.total,
            "rubric_reward": result.rubric_reward,
            "process_reward": result.process_reward,
            "citation_f1": result.citation_f1,
            "gates": result.gates,
            "gated": result.gated,
            "missing_fields": result.missing_fields,
            "judge_responses": result.judge_responses,
            "judge_failed": result.judge_failed,
            "reward_breakdown": result.reward_breakdown,
            "error": result.error,
        }

    def _persist_extra(self, meta, details) -> dict:
        return {
            "gap": meta.get("gap", ""),
            "citation_f1": details.get("citation_f1"),
            "gated": details.get("gated"),
            "gates": details.get("gates"),
            "missing_fields": details.get("missing_fields"),
            **_breakdown(details),
        }


ENVIRONMENTS = {
    "gap_finder": GapFinderEnvironment,
    "innovator": InnovatorEnvironment,
    "report_writer": ReportWriterEnvironment,
}
