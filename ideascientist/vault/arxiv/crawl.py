"""Date-bucketed bulk crawler over the public arXiv Atom API.

Paginates a date-range search with stdlib ``urllib`` only, honouring the arXiv
rate limit: 1s between requests, exponential backoff on 429/503 respecting
``Retry-After``, plus a page-level retry so one throttling window does not abort
a multi-page crawl.

``DEFAULT_CATEGORIES`` is only a default. The Svalbard Idea Vault covers every
arXiv subject area, which is what lets the innovator find a mechanism outside
the target problem's own sub-field; pass the full category list to reproduce it.
"""

from __future__ import annotations

import logging
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import date, datetime
from email.utils import parsedate_to_datetime
from typing import Iterable

logger = logging.getLogger(__name__)

ARXIV_API_URL = "https://export.arxiv.org/api/query"
USER_AGENT = "ideascientist/1.0 (research paper collection)"

DEFAULT_CATEGORIES: tuple[str, ...] = ("cs.CL", "cs.LG", "cs.AI", "cs.IR")

REQUEST_DELAY_SEC = 1.0
MAX_RESULTS_PER_PAGE = 100
HTTP_TIMEOUT_SEC = 60
PER_REQUEST_MAX_RETRIES = 5
RATE_LIMIT_BACKOFF_BASE = 30
RATE_LIMIT_BACKOFF_CAP = 300
PAGE_RETRY_PAUSE = 60
PAGE_MAX_ATTEMPTS = 2

ATOM_NS = "{http://www.w3.org/2005/Atom}"
ARXIV_NS = "{http://arxiv.org/schemas/atom}"

_PROXY = os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY") or ""

_last_request_time = 0.0


@dataclass
class ArxivPaper:
    arxiv_id: str
    title: str
    authors: str
    abstract: str
    primary_category: str
    categories: list[str] = field(default_factory=list)
    published_date: str | None = None
    updated_date: str | None = None
    year: str | None = None
    abs_url: str = ""
    pdf_url: str = ""


# ---------------------------------------------------------------------------
# Rate limiting + retry
# ---------------------------------------------------------------------------

def _rate_limit() -> None:
    global _last_request_time
    elapsed = time.time() - _last_request_time
    if elapsed < REQUEST_DELAY_SEC:
        time.sleep(REQUEST_DELAY_SEC - elapsed)
    _last_request_time = time.time()


def _parse_retry_after(headers) -> float | None:
    raw = headers.get("Retry-After") if headers else None
    if not raw:
        return None
    raw = raw.strip()
    try:
        return float(raw)
    except ValueError:
        pass
    try:
        target = parsedate_to_datetime(raw)
        if target is None:
            return None
        now = datetime.now(target.tzinfo) if target.tzinfo else datetime.now()
        delta = (target - now).total_seconds()
        return max(delta, 0.0)
    except (TypeError, ValueError):
        return None


def _backoff_seconds(attempt: int, headers) -> float:
    hint = _parse_retry_after(headers) if headers is not None else None
    if hint is not None:
        return min(max(hint, 1.0), float(RATE_LIMIT_BACKOFF_CAP))
    return float(min(RATE_LIMIT_BACKOFF_BASE * (2**attempt), RATE_LIMIT_BACKOFF_CAP))


def _make_opener():
    if _PROXY:
        handler = urllib.request.ProxyHandler({"http": _PROXY, "https": _PROXY})
        return urllib.request.build_opener(handler)
    return urllib.request.build_opener()


def _make_request(url: str, max_retries: int = PER_REQUEST_MAX_RETRIES) -> str | None:
    _rate_limit()
    opener = _make_opener()
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(max_retries):
        try:
            with opener.open(req, timeout=HTTP_TIMEOUT_SEC) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:
            if e.code in (429, 503):
                if attempt < max_retries - 1:
                    wait = _backoff_seconds(attempt, e.headers)
                    logger.warning(
                        "arXiv API %s, backoff %.0fs (attempt %d/%d)",
                        e.code, wait, attempt + 1, max_retries,
                    )
                    time.sleep(wait)
                    continue
                logger.warning("arXiv API %s on final retry", e.code)
                return None
            logger.warning("HTTP error %s: %s", e.code, e)
            return None
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            logger.warning("request error (attempt %d/%d): %s", attempt + 1, max_retries, e)
            if attempt < max_retries - 1:
                time.sleep(5)
    return None


def _fetch_page(url: str) -> str | None:
    for page_attempt in range(PAGE_MAX_ATTEMPTS):
        content = _make_request(url)
        if content:
            return content
        if page_attempt < PAGE_MAX_ATTEMPTS - 1:
            logger.warning(
                "page fetch failed, sleeping %ds before page-level retry (%d/%d)",
                PAGE_RETRY_PAUSE, page_attempt + 1, PAGE_MAX_ATTEMPTS,
            )
            time.sleep(PAGE_RETRY_PAUSE)
    return None


# ---------------------------------------------------------------------------
# Atom parsing
# ---------------------------------------------------------------------------

def _parse_entry(entry: ET.Element) -> ArxivPaper | None:
    id_elem = entry.find(f"{ATOM_NS}id")
    if id_elem is None or not id_elem.text:
        return None
    m = re.search(r"arxiv\.org/abs/([^v]+?)(?:v\d+)?$", id_elem.text.strip())
    if not m:
        return None
    arxiv_id = m.group(1)

    title_elem = entry.find(f"{ATOM_NS}title")
    title = re.sub(r"\s+", " ",
                   (title_elem.text or "").strip()) if title_elem is not None else ""

    summary_elem = entry.find(f"{ATOM_NS}summary")
    abstract = re.sub(r"\s+", " ",
                      (summary_elem.text or "").strip()) if summary_elem is not None else ""

    authors = []
    for a in entry.findall(f"{ATOM_NS}author"):
        n = a.find(f"{ATOM_NS}name")
        if n is not None and n.text:
            authors.append(n.text.strip())

    pub_elem = entry.find(f"{ATOM_NS}published")
    pub_date = pub_elem.text[:10] if pub_elem is not None and pub_elem.text else None
    upd_elem = entry.find(f"{ATOM_NS}updated")
    upd_date = upd_elem.text[:10] if upd_elem is not None and upd_elem.text else None
    year = pub_date[:4] if pub_date else None

    primary = ""
    pc = entry.find(f"{ARXIV_NS}primary_category")
    if pc is not None:
        primary = pc.get("term", "") or ""
    cats: list[str] = []
    if primary:
        cats.append(primary)
    for c in entry.findall(f"{ATOM_NS}category"):
        t = c.get("term", "") or ""
        if t and t not in cats:
            cats.append(t)

    return ArxivPaper(
        arxiv_id=arxiv_id,
        title=title,
        authors=", ".join(authors),
        abstract=abstract,
        primary_category=primary,
        categories=cats,
        published_date=pub_date,
        updated_date=upd_date,
        year=year,
        abs_url=f"https://arxiv.org/abs/{arxiv_id}",
        pdf_url=f"https://arxiv.org/pdf/{arxiv_id}.pdf",
    )


def _parse_response(xml_text: str) -> list[ArxivPaper]:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        logger.warning("XML parse error: %s", e)
        return []
    out: list[ArxivPaper] = []
    for entry in root.findall(f"{ATOM_NS}entry"):
        p = _parse_entry(entry)
        if p is not None:
            out.append(p)
    return out


# ---------------------------------------------------------------------------
# Query building
# ---------------------------------------------------------------------------

def _category_clause(categories: Iterable[str]) -> str:
    parts = [f"cat:{c}" for c in categories]
    return "(" + " OR ".join(parts) + ")" if parts else ""


def _date_clause(start: date, end: date) -> str:
    s = datetime(start.year, start.month, start.day, 0, 0).strftime("%Y%m%d%H%M")
    e = datetime(end.year, end.month, end.day, 23, 59).strftime("%Y%m%d%H%M")
    return f"submittedDate:[{s} TO {e}]"


def _build_query(categories: Iterable[str], start: date, end: date) -> str:
    cat = _category_clause(categories)
    dt = _date_clause(start, end)
    if cat and dt:
        return f"{cat} AND {dt}"
    return cat or dt


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def search_arxiv_window(
    start: date,
    end: date,
    *,
    categories: Iterable[str] = DEFAULT_CATEGORIES,
    max_results: int = 5000,
) -> list[ArxivPaper]:
    """Fetch every paper submitted in ``[start, end]`` within ``categories``.

    Paginated; sorted by ``submittedDate`` descending so the newest papers
    arrive first (useful for early progress logs).
    """
    query = _build_query(categories, start, end)
    logger.info("arXiv query: %s", query[:200])

    out: list[ArxivPaper] = []
    cursor = 0
    while len(out) < max_results:
        page_size = min(MAX_RESULTS_PER_PAGE, max_results - len(out))
        params = {
            "search_query": query,
            "start": cursor,
            "max_results": page_size,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
        url = f"{ARXIV_API_URL}?{urllib.parse.urlencode(params)}"
        logger.info("  fetching results %d..%d", cursor + 1, cursor + page_size)
        xml_text = _fetch_page(url)
        if not xml_text:
            logger.warning(
                "  page fetch exhausted at start=%d; stopping with %d results",
                cursor, len(out),
            )
            break
        page = _parse_response(xml_text)
        if not page:
            break
        out.extend(page)
        if len(page) < page_size:
            break
        cursor += page_size

    return out[:max_results]




def parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


