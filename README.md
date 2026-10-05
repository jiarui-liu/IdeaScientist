<h1 align="center">IdeaScientist</h1>

<p align="center">
  <b>Orchestrating Agents for Grounded Scientific Ideation</b>
</p>

<p align="center">
  <a href="https://huggingface.co/datasets/Jerry999/SvalbardIdeaVault">
    <img alt="Dataset" src="https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-Svalbard%20Idea%20Vault-yellow"></a>
  <a href="LICENSE">
    <img alt="License" src="https://img.shields.io/badge/License-CC%20BY--NC%204.0-green"></a>
</p>

---

Research ideation can be trained and evaluated on its own, separately from
implementation. IdeaScientist decomposes it into three roles — a **gap finder**
that identifies methodological limitations in closely related work, an
**innovator** that looks for a mechanism in another problem setting whose logic
transfers to that gap, and a **report writer** that develops the surviving
intuition into a complete proposal. Each is trained with reinforcement learning
against a reference artifact reconstructed from a human-authored paper, so a
stage is scored on what it contributed rather than on whether the final idea
happened to land.

Retrieval runs against the **Svalbard Idea Vault**, an offline corpus in which
every paper is reduced to a results-masked record: the problem, method and
design the authors proposed, with every measured outcome removed. An agent can
therefore read what a human team set out to do without learning whether it
worked.

| | |
|---|---|
| 📦 **Dataset** | [`Jerry999/SvalbardIdeaVault`](https://huggingface.co/datasets/Jerry999/SvalbardIdeaVault) — the vault, its results-masked records, and the train / validation / test partition |
| 🧪 **This repository** | the ideation harness, the vault construction code, the RL reward, and the evaluation |

```
ideascientist/
  harness/      the production ideation pipeline
  vault/        construction of the Svalbard Idea Vault
  rewards/      the RL reward: rubrics and validity checks
  training/     GRPO environment, datasets, and per-role configs
  evaluation/   the metrics reported in the paper
  serving/      the embedding encoder and the query-side server
scripts/        the six command-line entry points
```

No data is committed here: the corpus is on the Hub, and the reference
artifacts and trained adapters are reproduced by the code in this repository.

This is the production system and its automatic evaluation. Three things
reported in the paper are deliberately absent: the ablation arms, the human
study, and the LaTeX table builders — experiment scaffolding rather than
reusable code.

---

## Install

```bash
pip install -e .                # core
pip install -e '.[vault]'       # + corpus construction (arXiv, PDF, LaTeX)
pip install -e '.[embed]'       # + the embedding server
pip install -e '.[analysis]'    # + numerical aggregation
```

Everything talks to an OpenAI-compatible endpoint, so a local vLLM server, the
OpenAI API, or a gateway all work:

```bash
export IDEASCIENTIST_BASE_URL=http://localhost:8000/v1   # default
export IDEASCIENTIST_MODEL=Qwen3.6-27B
export IDEASCIENTIST_API_KEY=...                         # unset is fine for local vLLM

export EMBEDDING_BASE_URL=http://localhost:8100/v1       # the query server, below
export EMBEDDING_MODEL=Qwen3-Embedding-0.6B
export QWEN3_EMB_MODEL_PATH=Qwen/Qwen3-Embedding-0.6B    # weights, server side

export IDEA_VAULT_DB=data/idea_vault.db
export IDEASCIENTIST_KEYWORD_INDEX=data/keyword_index.db
```

The reported system runs on Qwen3.6-27B, judged by the same backbone with
reasoning disabled at temperature 0.7 and top-p 0.8.

### Embedding queries

Retrieval queries must be encoded with the Qwen3-Embedding instruction, which
the stored document-side vectors deliberately omit. Serve them with:

```bash
python -m ideascientist.serving.query_server --port 8100
```

A stock pooling endpoint will answer the same requests without the instruction
and silently return weaker neighbours, so point `EMBEDDING_BASE_URL` here.

### Tool calling

Serve the backbone with tool calling enabled, or every request carrying tools
is rejected:

```bash
vllm serve <model> \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes
```

Some checkpoints emit tool calls in a format the server-side parser does not
recognize — the reply arrives as prose and `tool_calls` comes back empty, so
the agent appears to stall without erroring. When that happens, carry the tools
in the prompt instead and let the client parse them back out:

```bash
export IDEASCIENTIST_TEXT_TOOLS=1
```

If the served context is smaller than the model's nominal window, say so, or
the per-role token budgets will exceed it and the server will return 400:

```bash
export CONTEXT_WINDOW=32768
```

---

## 1. The harness

An orchestrator maintains `plan.md` and decides which sub-agent to invoke next.
It cannot retrieve or read: everything it knows about the literature arrives
through the files its sub-agents write. Keeping it blind to paper content is
what makes the decomposition inspectable — each stage's artifact is the whole
interface between stages, which is also what makes the stages separately
trainable.

| Role | Writes | Information access |
|---|---|---|
| orchestrator | `plan.md` | reads every artifact; cannot search or read papers |
| gap finder | `gaps.md` | keyword search, bibliographic metadata, reader digests |
| innovator | `candidates.md` | same |
| report writer | `report.json` | same |
| reviewer | `scores.md` | same |
| paper reader | `papers/<id>.md` | one paper's full text, in an isolated single-turn context |

Only the paper reader ever sees full text, under a 30,000-token input cap.
Every other role reaches a paper's body through `read_paper`, which runs a
reader in a separate context and returns a digest. This is enforced at the
tool-availability level, not by prompting: a role whose full-text cap is zero
is never granted the tool.

Both retrieval regimes in the paper use the same keyword-search interface. The
regime is a property of how a role is instructed to phrase queries, not a
separate tool: the gap finder queries in the exact terminology of its assigned
axis to surface same-problem prior art, while the innovator deliberately leaves
the problem's own sub-field to find a transferable mechanism.

```bash
python scripts/run_ideation.py --problem-file problem.txt --run-dir runs
```

| File | |
|---|---|
| `roles.py` | the five role prompts and their output schemas |
| `orchestrator.py` | the orchestrator prompt |
| `tools.py` | tool surface, turn budgets, the phase-completion contract |
| `corpus.py` | retrieval and the cutoff guard |
| `runner.py` | the agent loop |
| `client.py` | OpenAI-compatible tool-use client with full logging |

---

## 2. The Svalbard Idea Vault

An offline literature database covering all arXiv subject areas, so the
innovator can draw on neighbouring fields. Every paper is reduced to a
**results-masked record**: the research problem, method, model design, related
work, and limitations, with every measured outcome replaced by the evaluation
the authors *proposed* and the predictions their method implies. A record
therefore describes the direction a human team pursued without revealing
whether it worked.

From that record, four prose views are derived — the **results-masked tuple**:

| View | Built from |
|---|---|
| `problem_definition` | the formal setting, inputs, outputs |
| `challenge` | what fails, why it matters, a concrete failing case |
| `intuition` | the contribution's main idea and plain-language rationale |
| `solution` | the method and model design that realize it |

These are the retrieval spaces the harness searches and the units the reference
artifacts are built from. Querying `problem_definition` and `challenge`
together finds same-problem prior art; querying `challenge` alone finds work
that fought the same difficulty elsewhere, which is where a transferable
`intuition` comes from. [`fields.py`](ideascientist/vault/fields.py) is the
single source of truth for this mapping, so index-time and query-time text
cannot drift apart.

### Schema

```sql
papers(id, title, authors, date, venue, source, url, abstract, arxiv_id, doi)
paper_full_text(paper_id, arxiv_id, content_source, full_text, raw_tex,
                bbl_content, char_len, fetch_status, fetched_at)
metadata_results_masked(paper_id, fields_json, model, created_at)
citations(id, source_paper_id, cited_paper_id, direction, cited_title,
          verified_status, confidence, fetched_at, publication_date)
```

The built vault is published as
[`Jerry999/SvalbardIdeaVault`](https://huggingface.co/datasets/Jerry999/SvalbardIdeaVault);
the stages below rebuild it from scratch, and substituting a different corpus
means populating these four tables and rebuilding the indexes. `fields_json`
holds the record described by
[`RESULTS_MASKED_SCHEMA`](ideascientist/vault/schema.py).

### Build

```bash
python scripts/build_vault.py init
python scripts/build_vault.py crawl --start 1986-01-01 --end 2026-06-18
python scripts/build_vault.py fulltext        # arXiv e-print LaTeX
python scripts/build_vault.py citations       # one-hop reference expansion
python scripts/build_vault.py decompose       # -> results-masked records
python scripts/build_vault.py index           # BM25 + the four field spaces
python scripts/build_vault.py splits          # target filtering + partition
```

Every stage is idempotent and resumable; `--limit` caps a single run. `crawl`
defaults to the `cs.*` categories — pass `--categories` with the full list to
reproduce the all-subject vault. `index` needs a GPU for the embedding spaces;
`--skip-embeddings` builds only the BM25 index.

Cutoff novelty is scored against a second, much larger index covering every
paper published before the cutoff, built as a job array and then consolidated:

```bash
for s in $(seq 0 7); do
  python -m ideascientist.vault.index embeddings --cohort pre-cutoff \
      --num-shards 8 --shard "$s"          # one GPU each, run as an array
done
python -m ideascientist.vault.index merge --cohort pre-cutoff --num-shards 8
```

It carries only the `challenge` and `problem_definition` spaces, which are the
two the comparison set is ranked on. Point `IDEASCIENTIST_PRECUTOFF_EMBEDDINGS`
at the result.

`splits` applies the filter chain from the paper — post-cutoff with LaTeX
source, more than 70% of cited works having usable full text, a results-masked
record present, and an LLM check that the paper is methodological rather than a
survey, position paper, dataset release, or empirical report — then partitions
the survivors into train / validation / test. All three roles share one
partition, so a paper is never a training target for one role and a test target
for another.

### Reference artifacts

```bash
python scripts/build_references.py gap_finder     # then innovator, then report_writer
```

One privileged pass per target paper per stage. Unlike the production roles,
the generator sees the target paper's full text and works backwards from it —
but it must still ground every citation in papers it actually read through the
same environment, so its output passes the same deterministic gates the reward
applies. That is what keeps the references in-distribution for the policy being
trained. The generators import the production role prompts rather than forking
them.

---

## 3. The reward

Each role is scored on an artifact reconstructed from a human paper, which
turns a sparse proposal-level outcome into a stage-specific signal without
prescribing a tool-use trajectory.

```
r = 0.85 · g · (r_free + r_ref + r_cit) / 3  +  0.15 · r_proc
```

| Term | |
|---|---|
| `r_free` | reference-free quality: specificity, significance, feasibility, evidence, clarity |
| `r_ref` | agreement with the reference on the substantive research decisions |
| `r_cit` | deterministic set F1 between generated and reference citations |
| `r_proc` | rule-based compliance; penalizes protocol violations only, never rewards search volume |
| `g` | the validity gate, in `{0, 1}` |

A failed gate zeroes the rubric reward while leaving the small shaping term, so
an early rollout that writes something parseable still produces gradient
variance.

| Role | Gates (all must pass) |
|---|---|
| gap finder | valid schema, assigned-axis match, at most two gaps, every cited id resolves and was read, judged methodological focus |
| innovator | valid schema, one candidate building on the given gap, every cited id resolves and was read, judged methodological contribution |
| report writer | all required proposal fields present, every cited id resolves and was read |

The citation gates are computed in code against the rollout's own file state,
before any judge call. Fabricating a near-miss id is the single most available
way to farm this reward, so it is made verifiable rather than judged.

Per role: `rubric.py` (rubric items and deterministic checks),
`reference_rubric.py` (judge prompt, grouping, citation F1, aggregation),
`reward.py` (gates and composition), `parser.py` (artifact to structure).
Shared judge plumbing is in [`judge.py`](ideascientist/rewards/judge.py); the
scoring primitives are in [`common.py`](ideascientist/rewards/common.py).

An unparseable judge reply is retried; if it still fails, the rollout is
dropped from the batch rather than scored with a fabricated value.

### Training

Requires a [NeMo-RL](https://github.com/NVIDIA/NeMo-RL) checkout on the path.

```bash
python -m ideascientist.training.datasets.gap_finder
python scripts/train_role.py gap_finder
```

[`environments.py`](ideascientist/training/environments.py) holds the three
role environments; the difference that matters between them is `INPUT_SEEDS`,
which seeds each role with the *reference* upstream artifact rather than a
generated one — that is what makes a role trainable in isolation.

The shipped configs reproduce the appendix setup: one LoRA adapter per role at
rank 32, α 64, no dropout, on all linear modules except the output projection;
AdamW at a constant 5e-6, weight decay 0.01, gradient clipping 1.0; bfloat16 at
tensor parallel 8; 64 prompts × 8 generations = 512 rollouts per step chunked
into gradient updates of 256; an asymmetric ratio clip of [0.2, 0.28] with no
KL penalty; and asynchronous GRPO at a maximum policy staleness of one step
with the corresponding importance-sampling correction.

| Role | Sequence length | Limit |
|---|---|---|
| gap finder | 26,624 | 2 epochs / 800 steps |
| innovator | 30,720 | 1 epoch / 600 steps |
| report writer | 49,152 | 2 epochs / 600 steps |

---

## 4. Evaluation

```bash
python scripts/evaluate.py runs/<run_dir>
python scripts/evaluate.py runs/<run_dir> --reference-report <label>/report.json
python scripts/compute_novelty_aspects.py runs/<system_dir>
```

`evaluate.py` writes `<run_dir>/evaluation/evaluation.json`; `aggregate.py`
reads those back across a system's `runs/`. Passing `--reference-report` adds
the report_writer training reward scored against a reference proposal, which is
where the per-field `reference_*` scores come from — without it that family is
absent and `overall` is computed as though every field scored zero. Reference
proposals are produced by
[`vault/references/`](ideascientist/vault/references/).

**The judge prompts in [`prompts.py`](ideascientist/evaluation/prompts.py) are
reproduced verbatim from the paper appendix, which is authoritative.** The
reported runs used those prompts, so a variant that differs is superseded
regardless of its date.

| Module | Metric |
|---|---|
| `quality.py` | the six proposal-quality dimensions |
| `novelty.py` | novelty at the All and Cutoff scopes |
| `novelty_aspects.py` | In-domain transfer and Mechanism non-obviousness |
| `citations.py` | citation precision, recall, and F1 |
| `reference_grounded.py` | per-field similarity to the held-out target paper |
| `aggregate.py` | system-level scores over a fixed denominator |

*Novelty.* **All** and **Cutoff** are 1–10 judgments of what the proposal adds
beyond the closest papers, differing only in whether the comparison corpus is
filtered to pre-cutoff work — the same prompt, a different retrieval scope.
**In-domain** is a binary judgment of structural cross-setting transfer, which
is the behaviour the innovator's cross-domain regime is supposed to produce.
**Mechanism non-obviousness** runs as two calls: the first never sees the
proposal and derives candidate mechanisms from the nearest pre-cutoff work on
its own, the second grades the proposal against that frozen list. The split
matters — a single call can always reverse-engineer a derivation once it has
seen the answer.

*Aggregation.* Every reported rate divides by the fixed test-split size, not by
the number of runs that produced output. A target paper with no scorable
proposal contributes zero; dividing by the scored count instead would let a
system raise its average by failing more often.

The evaluation restricts retrieval to papers published before the cutoff and
measures how closely proposals align with directions human researchers pursued
afterwards. `CUTOFF_DATE` in [`corpus.py`](ideascientist/harness/corpus.py) is a
correctness constraint: one post-cutoff paper reaching an agent invalidates a
run.

## Citation

Be here soon!

<!-- ```bibtex
@article{ideascientist,
  title     = {IdeaScientist: Orchestrating Agents for Grounded Scientific Ideation},
  journal={arXiv preprint arXiv:},
  year={2026}
}
``` -->

## License

CC BY-NC 4.0. See [LICENSE](LICENSE).
