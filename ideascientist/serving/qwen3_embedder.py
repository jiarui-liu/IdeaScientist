"""Shared Qwen3-Embedding-0.6B encoder (vLLM pooling backend).

Model path, query instruct prefix, 1024-d pooling, and L2-normalized float32
output. Index-time and query-time encoding both go through here so the two can
never drift out of the same space.

Qwen3-Embedding convention: an *instruction* is prepended on the QUERY side
only; documents are embedded raw. Encode docs with ``embed_documents`` and
queries (agent search text) with ``embed_queries``.
"""

from __future__ import annotations

import os

import numpy as np

# Stored paper vectors and the query encoder must share one 1024-d space, so
# this has to be the same checkpoint that built the index.
MODEL_PATH = os.environ.get("QWEN3_EMB_MODEL_PATH", "Qwen/Qwen3-Embedding-0.6B")
EMB_DIM = 1024
MAX_MODEL_LEN = 32768

# From config_sentence_transformers.json "query" prompt.
QUERY_INSTRUCT = (
    "Instruct: Given a web search query, retrieve relevant passages that "
    "answer the query\nQuery:"
)


def build_llm(gpu_memory_utilization: float = 0.85,
              max_model_len: int = MAX_MODEL_LEN,
              enforce_eager: bool = False):
    """Construct a vLLM pooling LLM for embeddings (vLLM >= 0.21: runner=pooling)."""
    from vllm import LLM
    return LLM(
        model=MODEL_PATH,
        runner="pooling",
        dtype="bfloat16",
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        enforce_eager=enforce_eager,
        trust_remote_code=True,
    )


def _embed(llm, texts: list[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, EMB_DIM), dtype=np.float32)
    outs = llm.embed(texts)
    arr = np.asarray(
        [o.outputs.embedding for o in outs], dtype=np.float32
    )
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (arr / norms).astype(np.float32)


def embed_documents(llm, texts: list[str]) -> np.ndarray:
    """Document side: no instruction prefix."""
    return _embed(llm, texts)


def embed_queries(llm, texts: list[str]) -> np.ndarray:
    """Query side: prepend the Qwen3-Embedding instruction."""
    return _embed(llm, [f"{QUERY_INSTRUCT} {t}" for t in texts])
