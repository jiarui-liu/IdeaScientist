"""Aggregate per-run evaluation output into the reported system-level scores.

One rule governs every number here: rates and averages divide by a **fixed
denominator** — the size of the test split — not by the number of runs that
happened to produce a scorable proposal. A target paper with no usable output
contributes zero. Dividing by the scored count instead would let a system raise
its average by failing more often, which is the opposite of what the metric is
supposed to measure.

The overall score is the unweighted mean of the quality columns: citation F1,
the six proposal-quality dimensions, and the per-field reference matches.
Validity and gate-pass rates are reported alongside it but deliberately
excluded from it — they measure whether output exists, not whether it is good.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

# Scored on 1-10 by the judges and normalized to [0, 1] for the reported table.
QUALITY_DIMENSIONS = (
    "actionability", "specificity", "clarity", "impact", "soundness", "relevance",
)
JUDGE_SCALE = 10.0

# The results-masked schema fields the reference-grounded score compares.
REFERENCE_FIELDS = (
    "core_problem", "problem_definition", "method", "model_details", "key_novelty",
    "proposed_evaluation", "comparison_to_sota", "related_work",
    "falsifiable_predictions", "limitations",
)

NOVELTY_SCOPES = ("all", "cutoff")


def iter_run_evaluations(base: Path) -> Iterable[dict[str, Any]]:
    """Every parsed ``evaluation.json`` under a system's run directory."""
    for path in sorted(Path(base).glob("runs/*/evaluation/evaluation.json")):
        try:
            yield json.loads(path.read_text())
        except (OSError, ValueError):
            continue


def report_metrics(evaluation: dict[str, Any]) -> dict[str, float]:
    """The scalar metrics one run contributes, flattened and normalized.

    A metric the run did not produce is simply absent; the caller sums over the
    fixed denominator, so absence is what makes it count as zero.
    """
    out: dict[str, float] = {}

    citations = evaluation.get("citations") or {}
    source = (citations.get("precision_recall_vs_source_citations") or {}).get("all") or {}
    if source.get("f1") is not None:
        out["citation_f1"] = float(source["f1"])

    quality = evaluation.get("quality") or {}
    for dim in QUALITY_DIMENSIONS:
        entry = quality.get(dim)
        if isinstance(entry, dict) and entry.get("score") is not None:
            out[dim] = float(entry["score"]) / JUDGE_SCALE

    novelty = evaluation.get("novelty") or {}
    for scope in NOVELTY_SCOPES:
        entry = novelty.get(scope) or {}
        if entry.get("score") is not None:
            out[f"novelty_{scope}"] = float(entry["score"]) / JUDGE_SCALE

    reward = evaluation.get("report_writer_reward")
    if isinstance(reward, dict) and reward.get("scored"):
        out["gate_pass_rate"] = 0.0 if reward.get("gated") else 1.0
        breakdown = reward.get("reward_breakdown") or {}
        for key in ("reference_quality", "base_quality"):
            if breakdown.get(key) is not None:
                out[key] = float(breakdown[key])
        for field, info in (breakdown.get("fields") or {}).items():
            if not (isinstance(info, dict) and info.get("satisfaction") is not None):
                continue
            if field.startswith("matches_reference_"):
                name = field[len("matches_reference_"):]
                if name in REFERENCE_FIELDS:
                    out[f"reference_{name}"] = float(info["satisfaction"])
    return out


def aggregate_system(base: Path, denominator: int) -> dict[str, float]:
    """System-level scores for one run directory, over a fixed denominator."""
    totals: dict[str, float] = {}
    scored = 0
    for evaluation in iter_run_evaluations(base):
        scored += 1
        for key, value in report_metrics(evaluation).items():
            totals[key] = totals.get(key, 0.0) + value

    row = {key: total / denominator for key, total in totals.items()}
    row["valid_generation_rate"] = scored / denominator

    quality = [row.get(d, 0.0) for d in QUALITY_DIMENSIONS]
    row["quality_average"] = sum(quality) / len(quality)
    novelty = [row.get(f"novelty_{s}", 0.0) for s in NOVELTY_SCOPES]
    row["novelty_average"] = sum(novelty) / len(novelty)

    overall_keys = (
        ["citation_f1"]
        + list(QUALITY_DIMENSIONS)
        + [f"reference_{f}" for f in REFERENCE_FIELDS]
    )
    row["overall"] = sum(row.get(k, 0.0) for k in overall_keys) / len(overall_keys)
    return row


def aggregate_systems(
    bases: dict[str, Path], denominator: int
) -> dict[str, dict[str, float]]:
    """``{system name: scores}`` for several run directories on one basis."""
    return {name: aggregate_system(base, denominator) for name, base in bases.items()}
