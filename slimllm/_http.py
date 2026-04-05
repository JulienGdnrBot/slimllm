"""
Zero-dependency HTTP client using stdlib http.client + ssl.

Supports:
  - POST with JSON body → full response dict
  - POST with JSON body → streaming (yields decoded lines one at a time)

Chunked transfer encoding (used by all LLM streaming endpoints) is handled
automatically by the stdlib HTTPResponse.  We iterate line-by-line so each
SSE event is emitted as soon as it arrives — no buffering the full body.
"""
from __future__ import annotations

import http.client
import json
import ssl
import urllib.parse
from typing import Any, Dict, Generator, Optional, Tuple


# Re-use a module-level SSL context so we don't rebuild it on every call.
_SSL_CTX = ssl.create_default_context()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _connect(hostname: str, port: int) -> http.client.HTTPSConnection:
    return http.client.HTTPSConnection(hostname, port, context=_SSL_CTX, timeout=120)


def _build_path(parsed: urllib.parse.ParseResult) -> str:
    path = parsed.path or "/"
    if parsed.query:
        path = f"{path}?{parsed.query}"
    return path


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def post_json(
    url: str,
    headers: Dict[str, str],
    body: Dict[str, Any],
    timeout: int = 120,
) -> Tuple[int, Any]:
    """
    Synchronous POST that reads the full response body.

    Returns (status_code, parsed_json_body).
    Raises json.JSONDecodeError if the body isn't JSON.
    """
    parsed = urllib.parse.urlparse(url)
    port = parsed.port or 443
    payload = json.dumps(body, ensure_ascii=False).encode("utf-8")

    req_headers = {
        "Content-Type": "application/json",
        "Content-Length": str(len(payload)),
        **headers,
    }

    conn = _connect(parsed.hostname, port)
    try:
        conn.request("POST", _build_path(parsed), body=payload, headers=req_headers)
        resp = conn.getresponse()
        status = resp.status
        raw = resp.read()
    finally:
        conn.close()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = raw.decode("utf-8", errors="replace")

    return status, data


def stream_lines(
    url: str,
    headers: Dict[str, str],
    body: Dict[str, Any],
    timeout: int = 120,
) -> Generator[str, None, None]:
    """
    Synchronous POST that yields decoded text lines as they arrive.

    The connection is kept open and lines are emitted one by one — ideal
    for SSE parsing.  Empty lines (SSE heartbeats / event separators) are
    yielded as-is so the SSE parser can act on them.

    Raises SlimLLMError subclasses on HTTP error status codes — the error
    body is read fully before raising.
    """
    from .exceptions import raise_for_status

    parsed = urllib.parse.urlparse(url)
    port = parsed.port or 443
    payload = json.dumps(body, ensure_ascii=False).encode("utf-8")

    req_headers = {
        "Content-Type": "application/json",
        "Content-Length": str(len(payload)),
        **headers,
    }

    conn = _connect(parsed.hostname, port)
    try:
        conn.request("POST", _build_path(parsed), body=payload, headers=req_headers)
        resp = conn.getresponse()

        if resp.status >= 400:
            error_body = resp.read().decode("utf-8", errors="replace")
            raise_for_status(resp.status, error_body, parsed.hostname)

        # Python's http.client.HTTPResponse is a file-like (io.RawIOBase) —
        # readline() reads up to and including the next b"\n".
        while True:
            raw_line = resp.readline()
            if not raw_line:
                break
            yield raw_line.decode("utf-8").rstrip("\r\n")
    finally:
        conn.close()
