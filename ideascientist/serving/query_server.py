#!/usr/bin/env python3
"""OpenAI-compatible ``/embeddings`` server for the QUERY side.

Qwen3-Embedding prepends a retrieval instruction to queries only. The stored
field spaces are encoded without it, so a query sent to a stock pooling server
comes back as a document-side vector — which still returns neighbours, just
worse ones, with nothing in the output to indicate it. This server always
applies the instruction, so callers pass raw query text.

Do not send documents here; :func:`ideascientist.vault.index.build_field_embeddings`
owns the document side.

    python -m ideascientist.serving.query_server --port 8100
"""

from __future__ import annotations

import argparse
import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

logger = logging.getLogger("query_server")

_LLM = None
_LLM_LOCK = threading.Lock()
_MODEL_LABEL = "qwen3-embedding-0.6b"


def _embed(texts: list[str]) -> list[list[float]]:
    from ideascientist.serving.qwen3_embedder import embed_queries

    with _LLM_LOCK:
        return [v.tolist() for v in embed_queries(_LLM, texts)]


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        return

    def _send(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.rstrip("/")
        if path in ("/health", "/v1/health"):
            self._send(200, {"status": "ok", "model": _MODEL_LABEL})
        elif path == "/v1/models":
            self._send(200, {"object": "list",
                             "data": [{"id": _MODEL_LABEL, "object": "model"}]})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path.rstrip("/") != "/v1/embeddings":
            self._send(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            inp = json.loads(self.rfile.read(length).decode("utf-8")).get("input", "")
            texts = [inp] if isinstance(inp, str) else list(inp)
            if not texts:
                self._send(400, {"error": "empty input"})
                return
            vectors = _embed(texts)
        except Exception as exc:  # noqa: BLE001 — report to the client, stay up
            logger.exception("embedding request failed")
            self._send(500, {"error": str(exc)})
            return

        n_tok = sum(len(t.split()) for t in texts)
        self._send(200, {
            "object": "list",
            "data": [{"object": "embedding", "index": i, "embedding": v}
                     for i, v in enumerate(vectors)],
            "model": _MODEL_LABEL,
            "usage": {"prompt_tokens": n_tok, "total_tokens": n_tok},
        })


class _Server(ThreadingHTTPServer):
    # A reward phase fans out dozens of concurrent requests; the stdlib backlog
    # of 5 makes the kernel refuse the rest of a burst, which reaches callers as
    # ConnectionRefusedError on random items rather than as slowness.
    request_queue_size = 256
    daemon_threads = True


def main() -> int:
    global _LLM

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--gpu-mem", type=float, default=0.5,
                    help="vLLM gpu_memory_utilization; query traffic is light")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    from ideascientist.serving.qwen3_embedder import build_llm

    _LLM = build_llm(gpu_memory_utilization=args.gpu_mem)
    _embed(["warmup"])  # fail here rather than on the first real request

    server = _Server((args.host, args.port), _Handler)
    logger.info("serving query embeddings on http://%s:%d/v1/embeddings",
                args.host, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
