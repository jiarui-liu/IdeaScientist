"""The results-masked tuple: one record into four retrieval views.

Each paper in the vault is reduced to four prose views derived from its
results-masked record:

    problem_definition  the formal setting, inputs, and outputs
    challenge           what fails, why it matters, and a concrete failing case
    intuition           the contribution's main idea and plain-language rationale
    solution            the method and model design that realize it

These are the retrieval spaces the harness searches and the units the reference
artifacts are built from. Querying ``problem_definition`` and ``challenge``
together finds same-problem prior art (the gap finder's regime); querying
``challenge`` alone finds work that fought the same difficulty in a different
setting, which is where a transferable ``intuition`` comes from (the
innovator's regime).

This module is the single source of truth for the mapping. The retrieval index
and the harness both derive their text from here, so query-time text and
index-time text can never drift apart.
"""

from __future__ import annotations

from typing import Any

FIELD_NAMES = ("problem_definition", "challenge", "intuition", "solution")


def _join(val: Any) -> str:
    if isinstance(val, list):
        return " ".join(str(x) for x in val if x)
    return str(val) if val else ""


def serialize_labeled(obj: Any) -> str:
    """Flatten nested record values into readable ``Label: value`` prose."""
    if obj is None:
        return ""
    if isinstance(obj, str):
        return obj.strip()
    if isinstance(obj, (int, float, bool)):
        return str(obj)
    if isinstance(obj, list):
        return " ".join(s for s in (serialize_labeled(x) for x in obj) if s)
    if isinstance(obj, dict):
        parts = []
        for k, v in obj.items():
            sv = serialize_labeled(v)
            if not sv:
                continue
            label = k.replace("_", " ").capitalize() if ("_" in k or k.islower()) else k
            parts.append(f"{label}: {sv}")
        return ". ".join(parts)
    return str(obj)


def field_texts(record: dict[str, Any]) -> dict[str, str]:
    """Map one results-masked record to the four field texts."""
    cp = record.get("core_problem") or {}
    kn = record.get("key_novelty") or {}

    problem_definition = (
        serialize_labeled(record["problem_definition"])
        if record.get("problem_definition")
        else ""
    )

    ps = cp.get("problem_statement", "") if isinstance(cp, dict) else ""
    why = _join(cp.get("why_it_matters", "")) if isinstance(cp, dict) else ""
    example = cp.get("concrete_example", "") if isinstance(cp, dict) else ""
    challenge = ". ".join(s for s in (ps, why, example) if s).strip()

    intuition = serialize_labeled(
        {
            "main_idea": kn.get("main_idea") if isinstance(kn, dict) else None,
            "explanation": kn.get("explanation") if isinstance(kn, dict) else None,
        }
    )

    solution_parts = {
        k: record[k] for k in ("method", "model_details") if record.get(k)
    }
    solution = serialize_labeled(solution_parts) if solution_parts else ""

    return {
        "problem_definition": problem_definition,
        "challenge": challenge,
        "intuition": intuition,
        "solution": solution,
    }
