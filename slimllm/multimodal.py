"""
Provider-aware multimodal image content-block helpers.

SlimLLM routes ``claude-*`` and ``anthropic/...`` model ids to the Anthropic
Messages API, which expects a different image content-block shape than the
OpenAI ``image_url`` shape that every other provider in our routing graph
accepts (OpenAI direct, OpenRouter, Mistral, DeepSeek, Google AI Studio).

This module emits the right shape for the resolved slimllm provider given
only the model id string that will be sent to :func:`slimllm.acompletion`.

Quick start
-----------
::

    from slimllm import build_image_part

    # OpenAI / OpenRouter / Mistral / DeepSeek / Gemini route:
    part = build_image_part("https://example.com/a.png", model_id="gpt-4o")
    # → {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}}

    # Anthropic native route (claude-* or anthropic/* model ids):
    part = build_image_part("https://example.com/a.png", model_id="claude-opus-4-7")
    # → {"type": "image", "source": {"type": "url", "url": "https://example.com/a.png"}}

    # Data URI on the Anthropic route → base64 source block:
    part = build_image_part("data:image/png;base64,iVBORw0KGgo...",
                            model_id="claude-3-5-sonnet")
    # → {"type": "image",
    #    "source": {"type": "base64", "media_type": "image/png", "data": "iVBORw0KGgo..."}}

References
----------
- LiteLLM ``convert_to_anthropic_image_obj`` + ``create_anthropic_image_param``
  in ``litellm_core_utils/prompt_templates/factory.py`` — adapted here to
  slimllm's zero-dependency style. The Bedrock branch is dropped (slimllm has
  no Bedrock provider) and the local-file / auto-resize paths are dropped
  (slimllm consumers ship HTTPS URLs or pre-encoded data URIs).
- SlimLLM router rules at ``slimllm/main.py`` — the ``is_anthropic_native_model``
  helper mirrors them exactly so the right shape is emitted for whatever
  ``acompletion(..., model=X)`` will ultimately invoke.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlparse


_DATA_URI_PREFIX = "data:"
_HTTPS = "https://"
_HTTP = "http://"

# Allowed image MIME types for vision providers — the union of what OpenAI,
# Anthropic, Mistral (Pixtral), and Gemini all accept.
_ALLOWED_IMAGE_MIME: frozenset = frozenset({
    "image/png",
    "image/jpeg",
    "image/jpg",
    "image/webp",
    "image/gif",
})

# Map URL path suffixes (lowercase, including dot) to canonical MIME types.
# Mirrors LiteLLM's ``_get_image_mime_type_from_url`` table, narrowed to the
# allowlist above.
_SUFFIX_TO_MIME: Dict[str, str] = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".gif": "image/gif",
}


def is_anthropic_native_model(model_id: str) -> bool:
    """Return True iff slimllm will route this model id to AnthropicProvider.

    Matches the routing rules at ``slimllm/main.py:_route``:

      - ``anthropic/...``  → Anthropic (strip prefix)
      - ``claude-...``     → Anthropic
      - everything else (including ``openrouter/anthropic/claude-...``) → not
        Anthropic native; tunnels through OpenRouter or another OpenAI-compat
        provider.
    """
    if not model_id:
        return False
    return model_id.startswith("anthropic/") or model_id.startswith("claude-")


def extract_mime_from_url(url: str) -> Optional[str]:
    """Best-effort MIME extraction from a URL path's file suffix.

    Strips query strings (S3 presigned URLs carry signatures past ``?``).
    Returns None when no recognised suffix is found — callers should treat
    that as "unknown, default to image/jpeg for the Anthropic base64 path".
    """
    if not url:
        return None
    try:
        path = urlparse(url).path.lower()
    except (ValueError, AttributeError):
        return None
    for suffix, mime in _SUFFIX_TO_MIME.items():
        if path.endswith(suffix):
            return mime
    return None


def extract_mime_and_data_from_data_uri(data_uri: str) -> Optional[Tuple[str, str]]:
    """Parse a ``data:image/png;base64,<DATA>`` URI into ``(mime, data)``.

    Returns None when the input isn't a well-formed data URI we can split.
    Mirrors LiteLLM's ``convert_to_anthropic_image_obj`` parsing, minus the
    network fallback for ``http://`` URLs.
    """
    if not data_uri or not data_uri.startswith(_DATA_URI_PREFIX):
        return None
    try:
        # "data:image/png;base64,XYZ" → mime="image/png", data="XYZ"
        head, data = data_uri.split(";base64,", 1)
        mime = head[len(_DATA_URI_PREFIX):]
        if not mime or not data:
            return None
        return mime, data
    except ValueError:
        return None


def is_safe_image_url(url: Any) -> bool:
    """Defensive scheme check — reject ``javascript:``, ``file:``, etc.

    Used by callers before forwarding an arbitrary URL to a provider.
    Allows: ``https://``, ``http://`` (caller awareness), and ``data:`` when
    the parser above can split it cleanly into ``(mime, base64_data)``.
    """
    if not url or not isinstance(url, str):
        return False
    if url.startswith(_HTTPS) or url.startswith(_HTTP):
        return True
    if url.startswith(_DATA_URI_PREFIX):
        return extract_mime_and_data_from_data_uri(url) is not None
    return False


def build_image_part(
    url: str,
    model_id: Optional[str],
    detail: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Build a provider-correct image content block from a single URL.

    Returns None when ``url`` fails the safe-scheme check — caller should
    skip the image rather than emit a malformed block.

    OpenAI-compat shape (default — covers OpenAI, OpenRouter, Mistral,
    DeepSeek, Google AI Studio shim)::

        {"type": "image_url", "image_url": {"url": "...", "detail": "..."}}

    Anthropic-native shape (when slimllm will route to AnthropicProvider)::

        URL  : {"type": "image", "source": {"type": "url", "url": "..."}}
        data:: {"type": "image", "source": {"type": "base64",
                                            "media_type": "image/png",
                                            "data": "..."}}

    The ``detail`` field is OpenAI-only (``low`` | ``high`` | ``auto``).
    Skipped on the Anthropic path since the Messages API has no equivalent.
    """
    if not is_safe_image_url(url):
        return None

    if is_anthropic_native_model(model_id or ""):
        return _build_anthropic_image_part(url)
    return _build_openai_image_part(url, detail)


def _build_openai_image_part(url: str, detail: Optional[str]) -> Dict[str, Any]:
    """OpenAI-compat shape (also accepted by OpenRouter, Mistral, etc.)."""
    image_url: Dict[str, Any] = {"url": url}
    if detail in ("low", "high", "auto"):
        image_url["detail"] = detail
    return {"type": "image_url", "image_url": image_url}


def _build_anthropic_image_part(url: str) -> Dict[str, Any]:
    """Anthropic Messages API shape.

    HTTPS URLs are passed through as ``source.type=url`` (Anthropic added
    native URL source support in 2024). Data URIs are split into base64
    parts. HTTP URLs are also passed as ``source.type=url`` — we don't
    fetch+base64 like LiteLLM does for Bedrock, because slimllm doesn't
    route Bedrock and consumers ship HTTPS URLs in practice.
    """
    if url.startswith(_DATA_URI_PREFIX):
        parsed = extract_mime_and_data_from_data_uri(url)
        if parsed is None:
            # Fallthrough: emit URL source with the raw data URI. Anthropic
            # will 400 but we've at least signalled intent rather than
            # silently dropping the image.
            return {
                "type": "image",
                "source": {"type": "url", "url": url},
            }
        mime, data = parsed
        if mime not in _ALLOWED_IMAGE_MIME:
            # Best-effort coercion — Anthropic accepts the listed set; if a
            # caller posts image/svg+xml we still ship it and let Anthropic
            # reject with a clearer error than our silent shape mismatch.
            pass
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": mime,
                "data": data,
            },
        }
    # HTTP/HTTPS URL → native URL source
    return {
        "type": "image",
        "source": {"type": "url", "url": url},
    }


__all__ = [
    "build_image_part",
    "extract_mime_and_data_from_data_uri",
    "extract_mime_from_url",
    "is_anthropic_native_model",
    "is_safe_image_url",
]
