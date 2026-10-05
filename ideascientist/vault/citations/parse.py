"""Parse a paper's bibliography out of its LaTeX source.

Handles the two forms arXiv e-prints ship: a compiled ``.bbl`` and a ``.bib``
file, across the bibliography styles that differ in how an entry is delimited.
"""

from __future__ import annotations

import logging
import re

from .models import CitedPaper

logger = logging.getLogger(__name__)



def parse_bbl_entries(bbl_content: str) -> list[CitedPaper]:
    """Parse a .bbl file into CitedPaper entries.

    Handles both ``\\bibitem{key} ...`` format and biblatex/biber format.
    """
    entries: list[CitedPaper] = []

    # Pattern for \bibitem{key} or \bibitem[label]{key}.
    # The optional [label] may contain balanced braces (e.g. ACM format's
    # \bibitem[Foo et~al\mbox{.}(2025)]) and may be separated from {key} by a
    # %-comment + newline + whitespace, so allow that gap.
    parts = re.split(
        r"\\bibitem(?:\[(?:[^\]\\]|\\.|\{[^}]*\})*\])?\s*(?:%[^\n]*\n\s*)?\{([^}]+)\}",
        bbl_content,
    )

    # parts[0] is preamble, then alternating key, content
    for i in range(1, len(parts), 2):
        if i + 1 >= len(parts):
            break
        ref_key = parts[i].strip()
        raw_text = parts[i + 1].strip()

        authors, title, year, venue = _parse_newblock_entry(raw_text)

        entries.append(
            CitedPaper(
                ref_key=ref_key,
                title=title,
                authors=authors,
                year=year,
                venue=venue,
                raw_entry_text=raw_text,
            )
        )

    if not entries:
        # Try biblatex/biber format: \entry{key}{type}{...}
        entries = _parse_biblatex_entries(bbl_content)

    return entries


def _iter_bib_entries(bib_content: str):
    r"""Yield (entry_type, ref_key, body, full_block) for each @type{key,...} entry.

    Uses a linear anchor scan + balanced-brace extraction — NO lazy-DOTALL
    regex, so it stays O(n) even on a 48MB bundled anthology.bib. (A real
    paper once shipped the entire ACL Anthology as its .bib; the old
    ``(.*?)\n\s*\}`` regex hung for 20+ minutes on it.)
    """
    anchor = re.compile(r"@(\w+)\s*\{\s*([^,\s}]+)\s*,")
    n = len(bib_content)
    for m in anchor.finditer(bib_content):
        entry_type = m.group(1).lower()
        ref_key = m.group(2).strip()
        brace_start = bib_content.find("{", m.start())
        if brace_start < 0:
            continue
        depth, i, end = 0, brace_start, n
        while i < n:
            c = bib_content[i]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    end = i
                    break
            i += 1
        body = bib_content[m.end():end]
        yield entry_type, ref_key, body, bib_content[m.start():end + 1]




def parse_bib_file(bib_content: str, wanted_keys: set[str] | None = None) -> list[CitedPaper]:
    """Parse a BibTeX .bib file into CitedPaper entries.

    If *wanted_keys* is given, only those entries are parsed — essential for
    huge bundled .bib files. Uses balanced-brace scanning (no catastrophic
    regex backtracking).
    """
    entries: list[CitedPaper] = []

    for entry_type, ref_key, body, _full in _iter_bib_entries(bib_content):
        if entry_type in ("string", "comment", "preamble"):
            continue
        if wanted_keys is not None and ref_key not in wanted_keys:
            continue

        fields = _parse_bibtex_fields(body)
        title = fields.get("title", "")
        # Clean braces from title
        title = re.sub(r"[{}]", "", title).strip()
        if not title:
            continue

        authors = fields.get("author", "")
        authors = re.sub(r"[{}]", "", authors).strip()
        year = fields.get("year", "")
        year = re.sub(r"[{}]", "", year).strip()
        venue = fields.get("journal", "") or fields.get("booktitle", "")
        venue = re.sub(r"[{}]", "", venue).strip()
        doi = fields.get("doi", "")
        doi = re.sub(r"[{}]", "", doi).strip()

        arxiv_id = ""
        eprint = fields.get("eprint", "")
        if eprint:
            eprint = re.sub(r"[{}]", "", eprint).strip()
            if re.match(r"\d{4}\.\d{4,5}", eprint):
                arxiv_id = eprint

        entries.append(CitedPaper(
            ref_key=ref_key,
            title=title,
            authors=authors,
            year=year,
            venue=venue,
            arxiv_id=arxiv_id,
            doi=doi,
            raw_entry_text=_full,
        ))

    return entries


def _parse_bibtex_fields(body: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    # Match field = {value} with nested braces
    pos = 0
    field_pattern = re.compile(r"(\w+)\s*=\s*")
    while pos < len(body):
        m = field_pattern.search(body, pos)
        if not m:
            break
        field_name = m.group(1).lower()
        val_start = m.end()

        if val_start >= len(body):
            break

        char = body[val_start]
        if char == "{":
            # Find matching closing brace
            depth = 1
            i = val_start + 1
            while i < len(body) and depth > 0:
                if body[i] == "{":
                    depth += 1
                elif body[i] == "}":
                    depth -= 1
                i += 1
            fields[field_name] = body[val_start + 1 : i - 1]
            pos = i
        elif char == '"':
            end = body.find('"', val_start + 1)
            if end >= 0:
                fields[field_name] = body[val_start + 1 : end]
                pos = end + 1
            else:
                pos = val_start + 1
        else:
            # Bare value (e.g., a number or macro)
            end = body.find(",", val_start)
            if end < 0:
                end = len(body)
            fields[field_name] = body[val_start:end].strip()
            pos = end + 1

    return fields


















# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _clean_latex(text: str) -> str:
    # Remove \emph{...}, \textbf{...}, etc.
    text = re.sub(
        r"\\(?:emph|textbf|textit|textrm|texttt|text)\{([^}]*)\}", r"\1", text
    )
    # Remove \url{...}, \href{...}{text}
    text = re.sub(r"\\url\{[^}]*\}", "", text)
    text = re.sub(r"\\href\{[^}]*\}\{([^}]*)\}", r"\1", text)
    # Convert \cite{key} to [key] so citation markers stay visible
    text = re.sub(
        r"\\(?:cite[pt]?|citeauthor|citeyear|citealt|citealp|Cite[pt]?)"
        r"(?:\[[^\]]*\])*\{([^}]+)\}",
        r"[\1]",
        text,
    )
    # Remove LaTeX spacing commands (\, \; \: \! \ ) — these become a space
    text = re.sub(r"\\[,;:!\s]", " ", text)
    # Remove remaining backslash commands (but keep content)
    text = re.sub(r"\\[a-zA-Z]+\*?(?:\{[^}]*\})?", "", text)
    # Remove braces
    text = re.sub(r"[{}]", "", text)
    # Clean whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text




def _extract_braced(text: str, start: int) -> str:
    b = text.find("{", start)
    if b < 0:
        return ""
    depth, i, n = 0, b, len(text)
    while i < n:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[b + 1:i]
        i += 1
    return text[b + 1:]


def _parse_acm_entry(raw: str) -> tuple[str, str, str, str]:
    r"""Parse an ACM-Reference-Format .bbl entry built from \bibinfo/\bibfield.

    Fields live in \bibinfo{person}{...}, \showarticletitle{...} or
    \bibinfo{title}{...}, \bibinfo{year}{...}, \bibinfo{booktitle/journal}{...}.
    """
    # Authors: all \bibinfo{person}{...}
    persons = []
    for m in re.finditer(r"\\bibinfo\{person\}", raw):
        persons.append(_clean_latex(_extract_braced(raw, m.end())))
    authors = ", ".join(p for p in persons if p)

    # Title: \showarticletitle{...} or \bibinfo{title}{...}
    title = ""
    m = re.search(r"\\showarticletitle", raw)
    if m:
        title = _clean_latex(_extract_braced(raw, m.end()))
    if not title:
        m = re.search(r"\\bibinfo\{(?:title|booktitle)\}", raw)
        if m:
            title = _clean_latex(_extract_braced(raw, m.end()))

    # Year
    year = ""
    m = re.search(r"\\bibinfo\{year\}", raw)
    if m:
        year = _clean_latex(_extract_braced(raw, m.end()))
        ym = re.search(r"(19|20)\d{2}", year)
        year = ym.group(0) if ym else ""

    # Venue: journal or booktitle
    venue = ""
    m = re.search(r"\\bibinfo\{(?:journal|booktitle)\}", raw)
    if m:
        venue = _clean_latex(_extract_braced(raw, m.end()))

    return authors, title, year, venue


def _parse_newblock_entry(raw: str) -> tuple[str, str, str, str]:
    r"""Parse a ``\bibitem`` entry that uses ``\newblock`` separators.

    Standard .bbl format:
        Authors.
        \newblock Title.
        \newblock Venue, Year.

    Returns ``(authors, title, year, venue)``.
    Falls back to heuristic splitting if no ``\newblock`` is found.
    """
    # ACM-Reference-Format: structured \bibinfo/\showarticletitle commands
    if "\\bibinfo{" in raw or "\\showarticletitle" in raw:
        authors, title, year, venue = _parse_acm_entry(raw)
        if title or authors:
            return authors, title, year, venue

    blocks = re.split(r"\\newblock\s*", raw)

    if len(blocks) >= 3:
        authors = _clean_latex(blocks[0]).strip().rstrip(".")
        title = _clean_latex(blocks[1]).strip().rstrip(".")
        venue_block = _clean_latex(" ".join(blocks[2:])).strip().rstrip(".")
    elif len(blocks) == 2:
        authors = _clean_latex(blocks[0]).strip().rstrip(".")
        title = _clean_latex(blocks[1]).strip().rstrip(".")
        venue_block = ""
    else:
        # No \newblock. These entries are usually structured by lines:
        #     Authors:                  (colon-terminated; initials contain periods)
        #     Title.
        #     Venue (Year)
        # Splitting on ". " breaks on author initials ("P. W. Abrahams"),
        # so use the colon terminator and line structure instead.
        lines = [l.strip() for l in raw.split("\n") if l.strip()]
        first_line_clean = _clean_latex(lines[0]) if lines else ""

        if ":" in first_line_clean:
            # Colon terminates the author list (may be mid-first-line).
            ci = first_line_clean.find(":")
            authors = first_line_clean[:ci].strip()
            after_colon = first_line_clean[ci + 1:].strip()
            tail_lines = [after_colon] if after_colon else []
            tail_lines += [_clean_latex(l).strip() for l in lines[1:]]
            tail_lines = [t for t in tail_lines if t]
            if tail_lines:
                title = tail_lines[0].rstrip(".").strip()
                venue_block = " ".join(tail_lines[1:]).strip()
            else:
                title, venue_block = "", ""
        elif len(lines) >= 2:
            # Line-structured: line 1 = authors, line 2 = title, rest = venue.
            authors = _clean_latex(lines[0]).strip().rstrip(".")
            title = _clean_latex(lines[1]).strip().rstrip(".")
            venue_block = _clean_latex(" ".join(lines[2:])).strip()
        else:
            # Single line: split on ". " but not after a single-capital initial.
            clean = _clean_latex(raw)
            parts = [p.strip() for p in re.split(r"(?<![A-Z])\.\s+", clean) if p.strip()]
            if len(parts) >= 2:
                authors, title = parts[0], parts[1]
                venue_block = ". ".join(parts[2:])
            else:
                return "", clean[:200], "", ""

    year_match = re.search(r"\b((?:19|20)\d{2})\b", venue_block or title or raw)
    year = year_match.group(1) if year_match else ""

    return authors, title, year, venue_block


def _parse_biblatex_entries(bbl_content: str) -> list[CitedPaper]:
    entries: list[CitedPaper] = []
    # biblatex format: \entry{key}{type}{hash}{...}
    pattern = re.compile(r"\\entry\{([^}]+)\}\{([^}]+)\}")

    for m in pattern.finditer(bbl_content):
        key = m.group(1)
        # Find field values within this entry block
        entry_start = m.end()
        next_entry = pattern.search(bbl_content, entry_start)
        entry_end = next_entry.start() if next_entry else len(bbl_content)
        entry_text = bbl_content[entry_start:entry_end]

        title = _extract_biblatex_field(entry_text, "title") or ""
        authors = _extract_biblatex_name(entry_text) or ""
        year = _extract_biblatex_field(entry_text, "year") or ""
        venue = (
            _extract_biblatex_field(entry_text, "journaltitle")
            or _extract_biblatex_field(entry_text, "booktitle")
            or ""
        )
        doi = _extract_biblatex_field(entry_text, "doi") or ""

        if title:
            entries.append(
                CitedPaper(
                    ref_key=key,
                    title=title,
                    authors=authors,
                    year=year,
                    venue=venue,
                    doi=doi,
                    raw_entry_text=entry_text,
                )
            )

    return entries


def _extract_biblatex_field(text: str, field_name: str) -> str | None:
    m = re.search(rf"\\field\{{{field_name}\}}\{{([^}}]*)\}}", text)
    return m.group(1).strip() if m else None


def _extract_biblatex_name(text: str) -> str:
    names: list[str] = []
    for m in re.finditer(r"\\hash.*?\\strng\{family\}\{([^}]+)\}", text):
        names.append(m.group(1))
    return ", ".join(names)
