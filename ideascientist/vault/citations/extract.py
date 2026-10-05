"""Fetch arXiv LaTeX source and extract citation data."""

from __future__ import annotations

import io
import logging
import os
import re
import tarfile
import time
import urllib.error
import urllib.request

logger = logging.getLogger(__name__)

_PROXY = os.environ.get("HTTPS_PROXY", os.environ.get("HTTP_PROXY", ""))




def fetch_latex_source_with_bbl(
    arxiv_id: str, *, max_attempts: int = 5, timeout: int = 120
) -> tuple[str | None, str | None]:
    """Download arXiv source and return (main_tex, bib_content).

    bib_content is ALL .bbl files concatenated, or (if none) ALL .bib files
    concatenated. We concatenate rather than picking the first because a
    paper may ship multiple bib files (e.g. its own ``custom.bib`` plus a
    bundled 48MB ``anthology.bib``) and the cited keys can live in any of
    them. Downstream parsing filters to the cited keys, so extra content is
    harmless (and huge bundled libs are shrunk before storage).

    ``max_attempts`` / ``timeout`` control the retry budget for the e-print
    download. Defaults preserve the original behaviour; callers that fan out
    concurrently (e.g. the bucket-1 fetcher) pass tighter values to avoid a
    few huge/slow e-prints monopolising a worker.
    """
    data = _download_eprint(arxiv_id, max_attempts=max_attempts, timeout=timeout)
    if data is None:
        return None, None

    files = _extract_all_files(data)
    if files is None:
        return None, None

    tex_files = {k: v for k, v in files.items() if k.endswith(".tex")}
    main_tex = _pick_main_tex(tex_files, files) if tex_files else None

    # Prefer .bbl (resolved bibliography); fall back to .bib (bibtex source).
    # Sort by name for deterministic order; smaller/own files tend to sort
    # before bundled ones but order doesn't matter since we parse by key.
    bbl_parts = [c for n, c in sorted(files.items()) if n.endswith(".bbl")]
    if bbl_parts:
        bib_content = "\n\n".join(bbl_parts)
    else:
        bib_parts = [c for n, c in sorted(files.items()) if n.endswith(".bib")]
        bib_content = "\n\n".join(bib_parts) if bib_parts else None

    return main_tex, bib_content


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _download_eprint(arxiv_id: str, max_attempts: int = 5, timeout: int = 120) -> bytes | None:
    """Download the raw e-print archive from arXiv with retries.

    Handles truncated downloads (``IncompleteRead``) and verifies the body
    length against ``Content-Length`` — a partial body produces a corrupt
    tar that silently fails extraction, which was the dominant failure mode.
    """
    import http.client

    url = f"https://arxiv.org/e-print/{arxiv_id}"
    headers = {"User-Agent": "autoresearch/1.0"}

    if _PROXY:
        handler = urllib.request.ProxyHandler({"http": _PROXY, "https": _PROXY})
        opener = urllib.request.build_opener(handler)
    else:
        opener = urllib.request.build_opener()

    for attempt in range(max_attempts):
        try:
            req = urllib.request.Request(url, headers=headers)
            with opener.open(req, timeout=timeout) as resp:
                data = resp.read()
            return data
        except urllib.error.HTTPError as e:
            # Genuine 404 means no source on arXiv — don't retry.
            if e.code == 404:
                logger.warning("Failed to fetch LaTeX for %s: %s", arxiv_id, e)
                return None
            last = e  # 429, 5xx etc. — retryable
        except (urllib.error.URLError, TimeoutError, http.client.IncompleteRead,
                ConnectionError, OSError) as e:
            last = e

        if attempt < max_attempts - 1:
            time.sleep(2 ** (attempt + 1))
            continue
        logger.warning("Failed to fetch LaTeX for %s: %s", arxiv_id, last)
        return None

    return None


def _extract_all_files(data: bytes) -> dict[str, str] | None:
    """Extract all text files from a tar.gz archive into memory.

    If *data* is not a valid tar archive, tries to interpret it as a
    single plain-text .tex file.  Returns ``None`` only when no usable
    content can be recovered.
    """
    files: dict[str, str] = {}

    try:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
            for member in tar.getmembers():
                if not member.isfile():
                    continue
                # Only extract text-ish files we care about
                if not any(
                    member.name.endswith(ext)
                    for ext in (".tex", ".bbl", ".bib", ".sty", ".cls")
                ):
                    continue
                f = tar.extractfile(member)
                if f is not None:
                    files[member.name] = f.read().decode("utf-8", errors="replace")
        if files:
            return files
        return None
    except tarfile.TarError:
        # Not a tar archive -- might be a single .tex file
        try:
            text = data.decode("utf-8", errors="replace")
            if "\\begin{document}" in text or "\\cite" in text:
                return {"main.tex": text}
        except Exception:
            pass
        return None


def _pick_main_tex(tex_files: dict[str, str], all_files: dict[str, str]) -> str | None:
    """Choose the main .tex file and resolve \\input{}/\\include{} directives.

    Prefers the file containing ``\\begin{document}``.  Falls back to
    the largest file.  After picking the main file, recursively inlines
    all ``\\input{}`` and ``\\include{}`` references so the returned
    string contains the full paper content.
    """
    if not tex_files:
        return None

    main_name = None
    for name, content in tex_files.items():
        if "\\begin{document}" in content:
            main_name = name
            break

    if main_name is None:
        main_name = max(tex_files, key=lambda k: len(tex_files[k]))

    return _resolve_inputs(tex_files[main_name], all_files, main_name)


_MAX_RESOLVED_SIZE = 2_000_000  # 2MB limit to prevent runaway resolution


def _resolve_inputs(
    content: str,
    all_files: dict[str, str],
    current_file: str,
    _seen: set[str] | None = None,
) -> str:
    if _seen is None:
        _seen = {current_file}

    if len(content) > _MAX_RESOLVED_SIZE:
        return content

    # Determine the directory of the current file for relative paths
    if "/" in current_file:
        base_dir = current_file.rsplit("/", 1)[0] + "/"
    else:
        base_dir = ""

    def _replace(m: re.Match) -> str:
        ref = m.group(1).strip()
        # Try with and without .tex extension, with and without base_dir
        candidates = [
            ref,
            ref + ".tex",
            base_dir + ref,
            base_dir + ref + ".tex",
        ]
        for cand in candidates:
            if cand in all_files and cand not in _seen:
                _seen.add(cand)
                return _resolve_inputs(all_files[cand], all_files, cand, _seen)
        return m.group(0)  # leave unresolved if file not found

    return re.sub(r"\\(?:input|include)\{([^}]+)\}", _replace, content)
