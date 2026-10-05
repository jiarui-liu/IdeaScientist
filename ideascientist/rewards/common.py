"""Rubric-scoring primitives shared by the three per-role GRPO rewards.

Tier and satisfaction weight maps, rubric-slicing helpers, and the weighted
aggregation scorer. Each role's ``rubric.py`` supplies its own rubric content
and judge prompt and wraps :func:`score_unit` in a domain-named function
(``score_gap`` / ``score_candidate`` / ``score_proposal``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Optional


# Categorical importance -> numeric weight (RaR-style tiering).
TIER_WEIGHT: dict[str, float] = {
    # No ``critical`` tier: gate rubrics use the ``important`` weight and hard-fail
    # the unit when unmet, so their veto — not an inflated weight — is what makes
    # them critical. Keeping only two tiers widens the reward's usable dynamic
    # range among passing units (see ``score_unit``), which GRPO needs for signal.
    "important": 2.0,
    "optional": 1.0,
}

# Per-item satisfaction is judged 0/1/2; this maps it to a [0, 1] fraction of the
# item's weight (HealthBench uses binary; we keep a partial-credit middle tier
# for the softer quality criteria, and require a clean 2 on gate items).
SATISFACTION_FRACTION: dict[int, float] = {0: 0.0, 1: 0.5, 2: 1.0}


def gate_check_ids(rubric: dict) -> list[str]:
    return [c["id"] for c in rubric["verifiable_checks"] if c.get("gate")]


def positive(rubric: dict, scope: str | None = None) -> list[dict]:
    items = rubric["positive_rubrics"]
    return [r for r in items if scope is None or r.get("scope") == scope]


def negative(rubric: dict, scope: str | None = None) -> list[dict]:
    items = rubric["negative_rubrics"]
    return [r for r in items if scope is None or r.get("scope") == scope]


def score_unit(
    det: dict[str, bool],
    judgments: dict[str, int],
    *,
    rubric: dict,
    scope: str,
) -> float:
    """Reward for one scored unit (gap / candidate / proposal) in [0, 1].

    Gated (reward 0) if any deterministic gate fails or any GATE positive rubric
    scores below 2 (Rubicon veto / AdvancedIF all-or-nothing). Otherwise:

        achieved = sum(w_i * satisfaction_fraction(s_i))  over scope positives
                 - sum(w_j)                                over triggered scope negatives
        reward   = clip(achieved / sum(w_i positives), 0, 1)   (HealthBench-style)
    """
    for cid in gate_check_ids(rubric):
        if not det.get(cid, False):
            return 0.0
    for r in positive(rubric, scope=scope):
        if r.get("gate") and judgments.get(r["id"], 0) < 2:
            return 0.0

    pos = positive(rubric, scope=scope)
    total = sum(r["weight"] for r in pos)
    if total == 0:
        return 0.0
    achieved = sum(
        r["weight"] * SATISFACTION_FRACTION.get(judgments.get(r["id"], 0), 0.0)
        for r in pos
    )
    penalty = sum(
        r["weight"]
        for r in negative(rubric, scope=scope)
        if judgments.get(r["id"], 0) >= 1
    )
    return max(0.0, min(1.0, (achieved - penalty) / total))


def component_breakdown(
    det: dict[str, bool],
    judgments: dict[str, int],
    *,
    base_rubric: dict,
    combined_rubric: dict,
    scope: str,
    citation_f1: float | None = None,
    component_weights: dict[str, float] | None = None,
) -> dict:
    """Per-field and per-component reward breakdown for ONE scored document.

    Recomputes the intermediate numbers ``score_document_combined`` discards so
    they can be logged per rollout. Generic across the three roles: pass the
    role's BASE rubric (intrinsic positives, negatives, gates) and its COMBINED
    rubric (adds the ``matches_reference_*`` pillar).

    ``rubric_reward`` is the component-weighted mean of the present components,
    matching ``score_document_combined`` — equal weights give
    ``(base + reference + citation) / 3``.
    """
    base_pos = positive(base_rubric, scope=scope)
    negs = negative(base_rubric, scope=scope)
    match_items = [
        r
        for r in positive(combined_rubric, scope=scope)
        if str(r["id"]).startswith("matches_reference_")
    ]

    base_quality = score_unit(det, judgments, rubric=base_rubric, scope=scope)

    ref_total_w = sum(r["weight"] for r in match_items)
    reference_quality = (
        sum(
            r["weight"] * SATISFACTION_FRACTION.get(judgments.get(r["id"], 0), 0.0)
            for r in match_items
        )
        / ref_total_w
        if ref_total_w
        else 0.0
    )

    negative_penalty = sum(
        r["weight"] for r in negs if judgments.get(r["id"], 0) >= 1
    )
    total_pos_w = sum(r["weight"] for r in base_pos)
    achieved = sum(
        r["weight"] * SATISFACTION_FRACTION.get(judgments.get(r["id"], 0), 0.0)
        for r in base_pos
    )

    gated = False
    for cid in gate_check_ids(base_rubric):
        if not det.get(cid, False):
            gated = True
    for r in base_pos:
        if r.get("gate") and judgments.get(r["id"], 0) < 2:
            gated = True

    fields: dict[str, dict] = {}
    for r in base_pos:
        s = judgments.get(r["id"], 0)
        sat = SATISFACTION_FRACTION.get(s, 0.0)
        fields[r["id"]] = {
            "score": s,
            "weight": r["weight"],
            "kind": "gate" if r.get("gate") else "base_positive",
            "satisfaction": sat,
            "contribution": r["weight"] * sat,
        }
    for r in match_items:
        s = judgments.get(r["id"], 0)
        sat = SATISFACTION_FRACTION.get(s, 0.0)
        fields[r["id"]] = {
            "score": s,
            "weight": r["weight"],
            "kind": "matches_reference",
            "satisfaction": sat,
            "contribution": r["weight"] * sat,
        }
    for r in negs:
        s = judgments.get(r["id"], 0)
        triggered = s >= 1
        fields[r["id"]] = {
            "score": s,
            "weight": r["weight"],
            "kind": "negative",
            "triggered": triggered,
            "contribution": -(r["weight"]) if triggered else 0.0,
        }

    w = component_weights or {"base": 1.0, "reference": 1.0, "citation": 1.0}
    parts: list[tuple[float, float]] = [
        (float(w.get("base", 0.0)), base_quality),
        (float(w.get("reference", 0.0)), reference_quality),
    ]
    cit_val = None
    if citation_f1 is not None:
        cit_val = min(max(float(citation_f1), 0.0), 1.0)
        parts.append((float(w.get("citation", 0.0)), cit_val))
    tw = sum(weight for weight, _ in parts if weight > 0.0)
    rubric_reward = (
        sum(weight * value for weight, value in parts if weight > 0.0) / tw
        if tw > 0.0
        else 0.0
    )

    return {
        "base_quality": base_quality,
        "reference_quality": reference_quality,
        "citation_f1": cit_val,
        "negative_penalty": negative_penalty,
        "achieved": achieved,
        "total_positive_weight": total_pos_w,
        "gated": gated,
        "rubric_reward": rubric_reward,
        "fields": fields,
    }


def citation_set_f1(
    generated_ids: "list[int] | set[int]",
    reference_ids: "list[int] | set[int]",
) -> float:
    """Set-F1 between the generated and reference cited-paper id sets.

    Citations are compared as unordered sets, so a proposal is not penalised for
    citing the same work in a different order or place. Two empty sets agree
    that there was nothing to cite and score 1.0; one empty set scores 0.0.

    Pure — no judge, no database — which is what makes it a verifier rather than
    an opinion, and one third of the rubric reward.
    """
    gen = {int(i) for i in generated_ids}
    ref = {int(i) for i in reference_ids}
    if not gen and not ref:
        return 1.0
    return 2.0 * len(gen & ref) / (len(gen) + len(ref))


# --------------------------------------------------------------------------- #
# Deterministic gate helpers, shared by the three roles
# --------------------------------------------------------------------------- #

_SEED_CITE_ID_RE = re.compile(r"\[([0-9][0-9,\s]*)\]")


def gates_pass(det: dict[str, bool]) -> bool:
    return all(det.values())


def cited_ids_resolve(
    ids: "list[int]",
    biblio_fn: "Callable[[int], Optional[dict]]",
) -> bool:
    """Every cited id resolves to a real paper.

    An empty citation set fails: the artifact is supposed to be grounded in
    prior work, so citing nothing is not a way to pass.
    """
    if not ids:
        return False
    for pid in ids:
        try:
            if not biblio_fn(int(pid)):
                return False
        except Exception:  # noqa: BLE001 — an unresolvable id fails the gate
            return False
    return True


def ids_in_seeded_docs(files: dict[str, str], doc_names: tuple[str, ...]) -> set[int]:
    """Paper ids already cited in the docs the role was seeded with.

    Those docs are the role's problem statement, so a paper named there is
    grounded context already and does not need its own read_paper digest.
    """
    out: set[int] = set()
    for name in doc_names:
        text = files.get(name, "")
        if not isinstance(text, str):
            continue
        for grp in _SEED_CITE_ID_RE.findall(text):
            out.update(int(tok) for tok in (t.strip() for t in grp.split(",")) if tok.isdigit())
    return out


def available_papers_block(cited_ids: list[int], files: dict[str, str]) -> str:
    """The ``<<available_papers>>`` block: id -> the digest's first finding line.

    Only ids the role actually read are listed, so the judge grounds on the same
    evidence the gates verified.
    """
    lines: list[str] = []
    for pid in cited_ids:
        digest = files.get(f"papers/{pid}.md", "")
        finding = ""
        if isinstance(digest, str) and digest.strip():
            for ln in digest.splitlines():
                s = ln.strip()
                if s and not s.startswith("#"):
                    finding = s
                    break
        lines.append(f"[{pid}] {finding}" if finding else f"[{pid}] (read)")
    return "\n".join(lines) if lines else "(no papers read)"


@dataclass
class RewardWeights:
    # The rubric evaluation reward (gates + judge + score_document) is the primary
    # signal.
    rubric: float = 0.85
    # Dense format shaping, weighted low. It exists so a base-model rollout that
    # writes something parseable but cannot yet clear a gate still produces GRPO
    # variance rather than a flat zero.
    process: float = 0.15
    # Relative weights of the three components INSIDE the rubric reward
    # (score_document_combined): the intrinsic base quality, the reference-anchored
    # quality, and the citation set-F1. Only their RATIOS matter — the scorer
    # normalizes by the sum of the components present, so equal weights give the
    # (base + ref + cit) / 3 of the paper, and a role scored without a reference
    # set falls back to (base + ref) / 2. Set one to 0.0 to drop it.
    base_quality: float = 1.0
    reference_quality: float = 1.0
    citation_f1: float = 1.0

    def component_weights(self) -> dict[str, float]:
        """The three intrinsic-rubric component weights as a dict for the scorer."""
        return {
            "base": self.base_quality,
            "reference": self.reference_quality,
            "citation": self.citation_f1,
        }


def reference_quality_tiered(
    judgments: dict[str, int],
    match_items: list[dict],
    match_ids: "list[str] | tuple[str, ...]",
) -> float:
    """Reference half in [0, 1]: tier-weighted mean over the match sub-items.

    The ``matches_reference`` pillar is decomposed into one sub-item per required
    field, each judged 0/1/2 and mapped to 0/0.5/1, then averaged with the
    methodological fields weighted above the peripheral ones. Gating is applied
    once globally by ``score_document_combined``, and negatives subtract from the
    base half, so neither appears here.
    """
    weight_by_id = {r["id"]: r["weight"] for r in match_items}
    total_w = sum(weight_by_id.values())
    if total_w == 0:
        return 0.0
    return sum(
        weight_by_id[i] * SATISFACTION_FRACTION.get(judgments.get(i, 0), 0.0)
        for i in match_ids
    ) / total_w
