"""The parsed form of one bibliography entry."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CitedPaper:
    """A paper cited in the reference list."""
    ref_key: str              # e.g. "[23]" or bibtex key
    title: str
    authors: str = ""
    year: str = ""
    venue: str = ""
    is_related_work: bool = False
    arxiv_id: str = ""
    doi: str = ""
    # Raw bib/bbl entry text — keeps the verbatim block so downstream
    # consumers can regex for things the structured parser missed
    # (arxiv ids hidden in url/note/howpublished, old-style ids, etc.).
    raw_entry_text: str = ""




