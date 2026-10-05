"""Paper field schemas for the Idea Vault.

``RESULTS_MASKED_SCHEMA`` is the proposal-stage view: it carries the research
problem, method, model design, related work, and limitations, and replaces every
measured outcome with ``proposed_evaluation`` and ``falsifiable_predictions``.
This is what the vault stores per paper, what the report writer emits, and what
the novelty and reference-grounded judges compare against. Masking the results
means a reference artifact records the direction a human team pursued without
revealing whether it worked.

Masking removes the measured outcomes a results-included summary would carry:
reported metric values, baseline comparisons, deltas, and headline results. The
record keeps the planned evaluation and the falsifiable predictions in their
place, so a reader of the record learns what the authors set out to test but
not what they found.
"""


from __future__ import annotations

RESULTS_MASKED_SCHEMA = """\
{
  "title": "<Paper title as it appears>",
  "topic_relevance": {
    "sub_topic": ["specific sub-areas this paper belongs to"],
    "primary_focus": ["aspects addressed within the sub-topic"]
  },
  "one_sentence_thesis": "MUST be under 30 words, no jargon, no metric numbers.",
  "core_problem": {
    "problem_statement": "1-2 sentences: be specific about what fails and why.",
    "why_it_matters": ["motivation ONLY — real-world impact + research gap in prior work; do NOT describe the solution here."],
    "concrete_example": "a specific input/output scenario showing what goes wrong with current approaches."
  },
  "key_novelty": {
    "main_idea": "short name for the contribution",
    "explanation": ["2-3 bullets, <=35 words each, intuition/analogy level, no formal notation"]
  },
  "problem_definition": {
    "formal_setting": "mathematical or formal description of the problem",
    "inputs": "what the system receives",
    "outputs": "what the system produces"
  },
  "key_terms": {"<term>": "<plain-language definition>"},
  "related_work": [
    {
      "sub_area": "...",
      "what_it_does": "mechanism-level summary of prior work in this sub-area",
      "open_gap": "the gap this prior work leaves that the paper addresses",
      "citations": ["author-year keys as they appear in the paper"]
    }
  ],
  "method": {
    "high_level_flow": ["Group name: module1 -> module2"],
    "modules": [
      {
        "module": "name",
        "role": "what it does",
        "model": "exact architecture if any",
        "inputs": ["..."],
        "outputs": ["..."],
        "notes": "implementation details"
      }
    ],
    "novel_architectural_elements": ["STRICTLY structural design choices — how modules are arranged/connected"],
    "pseudocode": "the core algorithm as a code-block string (if present in the paper)",
    "design_choices": ["key design choice + why"],
    "theoretical_grounding": "mathematical / theoretical justification (if present)"
  },
  "model_details": {
    "base_model": "be specific, e.g. 'LLaMA-2-7B' not 'LLaMA-based'",
    "training": {
      "method": "e.g. 'supervised fine-tuning', 'RLHF', 'contrastive learning'",
      "objective_functions": ["Purpose: <plain language>. Formally: <notation>"],
      "adaptation": "e.g. 'LoRA (rank=16)' or 'Full fine-tuning'",
      "key_hyperparameters": {}
    }
  },
  "comparison_to_sota": {
    "related_methods": ["Name: one-line description"],
    "key_differences": ["vs. <Method>: <specific difference>"]
  },
  "proposed_evaluation": {
    "evaluation_setting": "high-level description of how the method WOULD be evaluated (e.g. 'open-domain QA with Wikipedia retrieval')",
    "benchmarks": [
      {
        "name": "benchmark or dataset the authors propose to use",
        "task_type": "e.g. classification, generation, retrieval",
        "newly_constructed": false
      }
    ],
    "baselines": ["methods to compare against"],
    "main_metrics": ["all metrics to report"],
    "ablations": ["ablation studies to run to isolate each contribution"]
  },
  "falsifiable_predictions": [
    {
      "pillar": "which design pillar this tests",
      "prediction": "one empirical claim the method implies",
      "kill_condition": "the observation that would FALSIFY it"
    }
  ],
  "limitations": ["3-5 concise bullets, paper-acknowledged or apparent from the method"],
  "references": [
    {
      "anchor": "citation key as used in the paper",
      "short_label": "1-3 word handle",
      "year": null,
      "title": "full title if available from the text"
    }
  ]
}"""

