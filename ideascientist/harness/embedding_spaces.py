"""Query-side encoding for the vault's paper embedding spaces.

Stored vectors are the document side, written by
:func:`ideascientist.vault.index.build_field_embeddings`. Queries need the
Qwen3-Embedding retrieval instruction prepended, which is why
``EMBEDDING_BASE_URL`` must point at
:mod:`ideascientist.serving.query_server` and not at a stock pooling endpoint:
a stock one returns a document-side vector for a query, which still ranks, just
worse, with nothing in the response to show it.

Retrieval itself lives in :mod:`ideascientist.harness.corpus`, which holds the
cutoff filter every search has to respect.
"""

from __future__ import annotations

import numpy as np


def encode_query(space: str, text: str, api_key: str = "") -> np.ndarray:
    """Encode ``text`` into the vector space shared by every stored field."""
    from ideascientist.utils.llm import EmbeddingSettings, embed_text

    settings = EmbeddingSettings.from_env(api_key=api_key or None)
    return np.asarray(embed_text(text, settings=settings), dtype=np.float32)
