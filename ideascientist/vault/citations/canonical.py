"""Canonical-key helpers for matching papers across identifiers.

A canonical key collapses version / case / whitespace variants of the same
paper to one string, in priority order:

    canonical_key = arxiv:<normalized arxiv id>      (also derived from
                                                      10.48550/arxiv.X DOIs)
                  | doi:<normalized doi>             (non-arXiv DOIs)
                  | title:<lowercased alnum-only title>  (fallback)
"""

from __future__ import annotations

import re


def norm_arxiv(a: str | None) -> str:
    if not a:
        return ""
    a = a.strip().lower()
    m = re.search(r"(\d{4}\.\d{4,5})", a)
    if m:
        return m.group(1)
    return re.sub(r"v\d+$", "", a)


def norm_doi(d: str | None) -> str:
    if not d:
        return ""
    d = d.strip().lower()
    d = re.sub(r"^https?://(dx\.)?doi\.org/", "", d)
    return d




def arxiv_from_url(url: str | None) -> str:
    if not url:
        return ""
    m = re.search(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})", url)
    return norm_arxiv(m.group(1)) if m else ""


def norm_title(t: str | None) -> str:
    if not t:
        return ""
    return re.sub(r"[^a-z0-9]", "", t.lower())


