"""
Zero-dependency HTTP client using stdlib http.client + ssl.

Supports:
  - POST with JSON body → full response dict
  - POST with JSON body → streaming (yields decoded lines one at a time)

Chunked transfer encoding (used by all LLM streaming endpoints) is handled
automatically by the stdlib HTTPResponse.  We iterate line-by-line so each
SSE event is emitted as soon as it arrives — no buffering the full body.

Retry behaviour
---------------
Both post_json() and stream_lines() accept an optional RetryConfig.
On retryable status codes (429, 5xx) the call is retried up to
RetryConfig.max_retries times with exponential back-off:

    sleep = backoff_base * (2 ** attempt)

The global DEFAULT_RETRY singleton is used when no config is passed.
Pass RetryConfig(max_retries=0) to disable retries entirely.
"""
from __future__ import annotations

import http.client
import json
import ssl
import time
import urllib.parse
from typing import TYPE_CHECKING, Any, Dict, Generator, Optional, Tuple

if TYPE_CHECKING:
    from .types import RetryConfig


# Re-use a module-level SSL context so we don't rebuild it on every call.
_SSL_CTX = ssl.create_default_context()
_SSL_CTX.set_alpn_protocols(["http/1.1"])


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


def _default_retry() -> "RetryConfig":
    """Import lazily to avoid circular imports at module load time."""
    from .types import RetryConfig
    return RetryConfig()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def post_json(
    url: str,
    headers: Dict[str, str],
    body: Dict[str, Any],
    timeout: int = 120,
    retry: Optional["RetryConfig"] = None,
) -> Tuple[int, Any]:
    """
    Synchronous POST that reads the full response body.

    Returns (status_code, parsed_json_body).
    Retries on retryable status codes per *retry* config.
    """
    cfg = retry if retry is not None else _default_retry()
    last_exc: Optional[Exception] = None

    for attempt in range(cfg.max_retries + 1):
        if attempt:
            time.sleep(cfg.backoff_base * (2 ** (attempt - 1)))

        try:
            status, data = _post_json_once(url, headers, body, timeout)
        except Exception as exc:
            last_exc = exc
            continue  # network-level error — retry

        if status in cfg.retryable_status_codes and attempt < cfg.max_retries:
            continue  # retryable HTTP error — retry

        return status, data

    # All attempts exhausted
    if last_exc is not None:
        raise last_exc
    # Should not reach here, but make the type-checker happy
    return _post_json_once(url, headers, body, timeout)


def _post_json_once(
    url: str,
    headers: Dict[str, str],
    body: Dict[str, Any],
    timeout: int,
) -> Tuple[int, Any]:
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
    retry: Optional["RetryConfig"] = None,
) -> Generator[str, None, None]:
    """
    Synchronous POST that yields decoded text lines as they arrive.

    On retryable status codes the initial connection is retried (before any
    bytes are streamed).  Once streaming has started, retries are not
    attempted — partial streams cannot be safely replayed.

    Empty lines (SSE heartbeats / event separators) are yielded as-is so
    the SSE parser can act on them.
    """
    from .exceptions import raise_for_status

    cfg = retry if retry is not None else _default_retry()
    last_exc: Optional[Exception] = None

    for attempt in range(cfg.max_retries + 1):
        if attempt:
            time.sleep(cfg.backoff_base * (2 ** (attempt - 1)))

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
                if resp.status in cfg.retryable_status_codes and attempt < cfg.max_retries:
                    conn.close()
                    continue
                raise_for_status(resp.status, error_body, parsed.hostname)

            # Stream is live — yield lines until EOF, no more retries
            try:
                while True:
                    raw_line = resp.readline()
                    if not raw_line:
                        break
                    yield raw_line.decode("utf-8").rstrip("\r\n")
            finally:
                conn.close()
            return  # success

        except GeneratorExit:
            conn.close()
            return
        except Exception as exc:
            conn.close()
            last_exc = exc
            continue

    if last_exc is not None:
        raise last_exc
