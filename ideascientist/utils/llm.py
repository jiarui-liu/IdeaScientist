"""OpenAI-compatible chat and embedding client.

Endpoints, keys, and model names are resolved here from environment variables so
no call site hardcodes them. Any OpenAI-compatible server works: a local vLLM
pool, the OpenAI API, or a gateway.

    IDEASCIENTIST_API_KEY / OPENAI_API_KEY
    IDEASCIENTIST_BASE_URL / OPENAI_BASE_URL   default http://localhost:8000/v1
    IDEASCIENTIST_MODEL                        default Qwen3.6-27B
    EMBEDDING_BASE_URL                         default http://localhost:8100/v1
    EMBEDDING_MODEL / EMBEDDING_DIM            default Qwen3-Embedding-0.6B / 1024
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Optional

DEFAULT_BASE_URL = "http://localhost:8000/v1"
DEFAULT_MODEL = "Qwen3.6-27B"

DEFAULT_EMBEDDING_BASE_URL = "http://localhost:8100/v1"
DEFAULT_EMBEDDING_MODEL = "Qwen3-Embedding-0.6B"
DEFAULT_EMBEDDING_DIM = 1024

_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class _RePostRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Follow 307/308 without urllib's default downgrade of POST to GET."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return urllib.request.Request(
            newurl,
            data=req.data,
            headers=dict(req.header_items()),
            method=req.get_method(),
        )


def build_opener(proxy: str = "") -> urllib.request.OpenerDirector:
    if proxy:
        return urllib.request.build_opener(
            urllib.request.ProxyHandler({"http": proxy, "https": proxy}),
            _RePostRedirectHandler,
        )
    return urllib.request.build_opener(_RePostRedirectHandler)


class LLMTimeoutError(TimeoutError):
    """A request exceeded its hard total wall-clock deadline."""


def _open_with_deadline(
    opener: urllib.request.OpenerDirector,
    req: urllib.request.Request,
    *,
    sock_timeout: float,
    deadline: float,
) -> bytes:
    """``opener.open() + read()`` bounded by a total wall-clock ``deadline``.

    urllib's ``timeout`` is per-socket-operation, so an endpoint that accepts the
    connection and then trickles bytes never trips it. Under the reward-phase
    concurrency of GRPO training that stalls a whole reward thread, so the
    blocking read runs in a daemon thread we abandon at the deadline.
    """
    box: dict[str, Any] = {}

    def _worker() -> None:
        try:
            with opener.open(req, timeout=sock_timeout) as resp:
                box["body"] = resp.read()
        except BaseException as exc:  # noqa: BLE001 - relayed to the caller thread
            box["exc"] = exc

    t = threading.Thread(target=_worker, name="llm-http", daemon=True)
    t.start()
    t.join(deadline)
    if t.is_alive():
        raise LLMTimeoutError(f"request exceeded hard deadline of {deadline:.0f}s")
    if "exc" in box:
        raise box["exc"]
    return box["body"]


def uses_completion_tokens(model: str) -> bool:
    """True for families that require ``max_completion_tokens``."""
    return bool(re.search(r"(?:^|-)(gpt-?5|o[1-9])", (model or "").lower()))


_CONTEXT_WINDOWS: tuple[tuple[str, int], ...] = (
    ("claude", 200_000),
    ("gpt-5", 400_000),
    ("gpt-oss", 128_000),
    ("gptoss", 128_000),
    ("gemma", 128_000),
    ("qwen", 256_000),
    ("llama-4", 1_000_000),
    ("llama", 128_000),
)
DEFAULT_CONTEXT_WINDOW = 128_000


def context_window_for(model: str) -> int:
    """Context window in tokens, for sizing the per-call output budget."""
    override = os.environ.get("CONTEXT_WINDOW", "").strip()
    if override.isdigit():
        return int(override)
    m = (model or "").lower()
    for needle, window in _CONTEXT_WINDOWS:
        if needle in m:
            return window
    return DEFAULT_CONTEXT_WINDOW


def _env_key(*names: str) -> str:
    for n in names:
        v = os.environ.get(n, "").strip()
        if v:
            return v
    return ""


@dataclass(frozen=True)
class LLMSettings:
    api_key: str
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    proxy: str = ""
    extra_body: Optional[dict[str, Any]] = None

    @classmethod
    def from_env(
        cls,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        proxy: Optional[str] = None,
        require_key: bool = False,
    ) -> "LLMSettings":
        key = api_key or _env_key("IDEASCIENTIST_API_KEY", "OPENAI_API_KEY")
        if require_key and not key:
            raise ValueError(
                "No API key. Set IDEASCIENTIST_API_KEY or OPENAI_API_KEY, or pass "
                "api_key=. A local vLLM server usually needs no key."
            )
        extra = os.environ.get("IDEASCIENTIST_EXTRA_BODY")
        return cls(
            api_key=key or "EMPTY",
            base_url=(
                base_url
                or _env_key("IDEASCIENTIST_BASE_URL", "OPENAI_BASE_URL")
                or DEFAULT_BASE_URL
            ).rstrip("/"),
            model=model or os.environ.get("IDEASCIENTIST_MODEL") or DEFAULT_MODEL,
            proxy=proxy if proxy is not None else os.environ.get("HTTPS_PROXY_OVERRIDE", ""),
            extra_body=json.loads(extra) if extra else None,
        )


@dataclass(frozen=True)
class EmbeddingSettings:
    api_key: str
    base_url: str = DEFAULT_EMBEDDING_BASE_URL
    model: str = DEFAULT_EMBEDDING_MODEL
    dim: int = DEFAULT_EMBEDDING_DIM
    proxy: str = ""

    @classmethod
    def from_env(
        cls,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        dim: Optional[int] = None,
        proxy: Optional[str] = None,
    ) -> "EmbeddingSettings":
        env_dim = os.environ.get("EMBEDDING_DIM")
        return cls(
            api_key=api_key or _env_key("EMBEDDING_API_KEY", "IDEASCIENTIST_API_KEY") or "EMPTY",
            base_url=(
                base_url or os.environ.get("EMBEDDING_BASE_URL") or DEFAULT_EMBEDDING_BASE_URL
            ).rstrip("/"),
            model=model or os.environ.get("EMBEDDING_MODEL") or DEFAULT_EMBEDDING_MODEL,
            dim=dim if dim is not None else (int(env_dim) if env_dim else DEFAULT_EMBEDDING_DIM),
            proxy=proxy if proxy is not None else "",
        )


def chat_with_usage(
    messages: list[dict[str, str]],
    *,
    settings: Optional[LLMSettings] = None,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: int = 4096,
    max_retries: int = 3,
    timeout: int = 120,
    total_deadline: Optional[float] = None,
    extra_body: Optional[dict[str, Any]] = None,
) -> tuple[str, dict[str, Any]]:
    """Chat completion returning ``(content, usage)``.

    ``timeout`` bounds one socket operation; ``total_deadline`` bounds the whole
    call including retries and backoff, so this never blocks a caller
    indefinitely on a stalled endpoint.
    """
    s = settings or LLMSettings.from_env()
    model = model or s.model
    url = f"{s.base_url}/chat/completions"

    env_to = os.environ.get("IDEASCIENTIST_SOCK_TIMEOUT", "").strip()
    if env_to:
        timeout = int(env_to)
    if total_deadline is None:
        env_dl = os.environ.get("IDEASCIENTIST_TOTAL_DEADLINE", "").strip()
        total_deadline = float(env_dl) if env_dl else max(timeout * 2, 240)

    payload: dict[str, Any] = {"model": model, "messages": messages}
    if uses_completion_tokens(model):
        payload["max_completion_tokens"] = max_tokens
    else:
        payload["max_tokens"] = max_tokens
        if temperature is not None:
            payload["temperature"] = temperature
    if s.extra_body:
        payload.update(s.extra_body)
    if extra_body:
        payload.update(extra_body)

    body = json.dumps(payload).encode("utf-8")
    headers = {"Authorization": f"Bearer {s.api_key}", "Content-Type": "application/json"}
    # urllib openers are not documented thread-safe; build one per call.
    opener = build_opener(s.proxy)

    start = time.monotonic()
    last_exc: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        remaining = total_deadline - (time.monotonic() - start)
        if remaining <= 0:
            raise LLMTimeoutError(
                f"chat exceeded total deadline of {total_deadline:.0f}s "
                f"after {attempt} attempt(s): {last_exc}"
            )
        try:
            req = urllib.request.Request(url, data=body, headers=headers, method="POST")
            raw = _open_with_deadline(
                opener, req, sock_timeout=min(timeout, remaining), deadline=remaining
            )
            result = json.loads(raw.decode("utf-8"))
            message = result["choices"][0]["message"]
            # A server running a reasoning parser returns ``content: null`` when
            # the reply is truncated inside the reasoning block. Callers parse
            # text, so hand back the reasoning rather than None.
            content = message.get("content")
            if not content:
                content = message.get("reasoning_content") or ""
            return content, result.get("usage", {})
        except LLMTimeoutError:
            raise
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
            last_exc = e
            retryable = not isinstance(e, urllib.error.HTTPError) or e.code in _RETRYABLE_STATUS
            if not retryable or attempt >= max_retries:
                raise
            budget = total_deadline - (time.monotonic() - start)
            if budget <= 0:
                raise
            time.sleep(min(2 ** (attempt + 1), 30, budget))
    raise RuntimeError(f"chat failed after {max_retries + 1} attempts: {last_exc}")


def chat(messages: list[dict[str, str]], **kwargs: Any) -> str:
    return chat_with_usage(messages, **kwargs)[0]


def complete(prompt: str, *, system: Optional[str] = None, **kwargs: Any) -> str:
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    return chat(messages, **kwargs)


def embed_texts(
    texts: list[str],
    *,
    settings: Optional[EmbeddingSettings] = None,
    model: Optional[str] = None,
    dimensions: Optional[int] = None,
    max_input_chars: int = 8000,
    max_retries: int = 3,
    timeout: int = 60,
) -> list[list[float]]:
    """Embed a batch of texts, preserving input order."""
    s = settings or EmbeddingSettings.from_env()
    url = f"{s.base_url}/embeddings"
    headers = {"Authorization": f"Bearer {s.api_key}", "Content-Type": "application/json"}
    body = json.dumps(
        {
            "input": [t[:max_input_chars] for t in texts],
            "model": model or s.model,
            "dimensions": dimensions if dimensions is not None else s.dim,
        }
    ).encode("utf-8")
    opener = build_opener(s.proxy)

    last_exc: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            req = urllib.request.Request(url, data=body, headers=headers, method="POST")
            with opener.open(req, timeout=timeout) as resp:
                result = json.loads(resp.read().decode("utf-8"))
            return [d["embedding"] for d in sorted(result["data"], key=lambda x: x["index"])]
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
            last_exc = e
            retryable = not isinstance(e, urllib.error.HTTPError) or e.code in _RETRYABLE_STATUS
            if not retryable or attempt >= max_retries:
                raise
            time.sleep(min(2 ** (attempt + 1), 30))
    raise RuntimeError(f"embed failed after {max_retries + 1} attempts: {last_exc}")


def embed_text(text: str, **kwargs: Any) -> list[float]:
    return embed_texts([text], **kwargs)[0]
