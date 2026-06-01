"""
Tests for ``slimllm.multimodal`` — provider-aware image content blocks.

Covers:
  - ``is_anthropic_native_model`` routing mirror (claude-*, anthropic/*,
    openrouter-tunnelled-anthropic, OpenAI, Mistral, Gemini)
  - ``extract_mime_from_url`` suffix-to-MIME table + presigned-URL handling
  - ``extract_mime_and_data_from_data_uri`` data-URI parsing
  - ``is_safe_image_url`` scheme allowlist (https/http/data:* only)
  - ``build_image_part`` — OpenAI-compat vs Anthropic-native shape, ``detail``
    field plumbing, safety rejection
"""
from __future__ import annotations

import base64
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from slimllm.multimodal import (
    build_image_part,
    extract_mime_and_data_from_data_uri,
    extract_mime_from_url,
    is_anthropic_native_model,
    is_safe_image_url,
)


# A 1×1 transparent PNG, base64-encoded — small enough to embed inline.
_TINY_PNG_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
)
_TINY_PNG_DATA_URI = f"data:image/png;base64,{_TINY_PNG_BASE64}"


# ===========================================================================
# is_anthropic_native_model — routing mirror
# ===========================================================================

class TestIsAnthropicNativeModel(unittest.TestCase):
    """Slimllm sends ``anthropic/`` and bare ``claude-`` to AnthropicProvider;
    everything else (including ``openrouter/anthropic/``) tunnels through
    OpenAI-compat providers."""

    def test_bare_claude_prefix_is_anthropic_native(self):
        self.assertTrue(is_anthropic_native_model("claude-3-5-sonnet-20241022"))
        self.assertTrue(is_anthropic_native_model("claude-opus-4-7"))

    def test_anthropic_slash_prefix_is_anthropic_native(self):
        self.assertTrue(is_anthropic_native_model("anthropic/claude-3-haiku"))

    def test_openrouter_anthropic_is_not_native(self):
        """OpenRouter tunnel must stay on the OpenAI-compat shape."""
        self.assertFalse(is_anthropic_native_model("openrouter/anthropic/claude-3-5-sonnet"))

    def test_openai_models_are_not_anthropic(self):
        self.assertFalse(is_anthropic_native_model("gpt-4o"))
        self.assertFalse(is_anthropic_native_model("openai/gpt-4o-mini"))

    def test_mistral_is_not_anthropic(self):
        self.assertFalse(is_anthropic_native_model("mistral-large-latest"))
        self.assertFalse(is_anthropic_native_model("mistral/pixtral-12b"))

    def test_gemini_is_not_anthropic(self):
        self.assertFalse(is_anthropic_native_model("gemini-2.5-pro"))
        self.assertFalse(is_anthropic_native_model("gemini/gemini-2.5-flash"))

    def test_deepseek_is_not_anthropic(self):
        self.assertFalse(is_anthropic_native_model("deepseek-chat"))
        self.assertFalse(is_anthropic_native_model("deepseek/deepseek-r1"))

    def test_empty_and_none_handled(self):
        self.assertFalse(is_anthropic_native_model(""))
        self.assertFalse(is_anthropic_native_model(None or ""))


# ===========================================================================
# extract_mime_from_url — suffix table
# ===========================================================================

class TestExtractMimeFromUrl(unittest.TestCase):
    def test_png_url(self):
        self.assertEqual(extract_mime_from_url("https://x.com/a.png"), "image/png")

    def test_jpeg_jpg_both_mapped(self):
        self.assertEqual(extract_mime_from_url("https://x.com/a.jpg"), "image/jpeg")
        self.assertEqual(extract_mime_from_url("https://x.com/a.jpeg"), "image/jpeg")

    def test_webp_and_gif(self):
        self.assertEqual(extract_mime_from_url("https://x.com/anim.gif"), "image/gif")
        self.assertEqual(extract_mime_from_url("https://x.com/x.webp"), "image/webp")

    def test_presigned_url_query_string_stripped(self):
        """S3 presigned URLs carry ``?X-Amz-Signature=...`` — must not let the
        query string break suffix detection."""
        url = "https://bucket.s3.amazonaws.com/path/screenshot.png?X-Amz-Signature=deadbeef"
        self.assertEqual(extract_mime_from_url(url), "image/png")

    def test_unknown_suffix_returns_none(self):
        self.assertIsNone(extract_mime_from_url("https://x.com/a.bmp"))
        self.assertIsNone(extract_mime_from_url("https://x.com/path-no-ext"))

    def test_empty_url(self):
        self.assertIsNone(extract_mime_from_url(""))

    def test_uppercase_suffix_lowercased(self):
        """Path is lowercased before suffix matching — ``.PNG`` still matches."""
        self.assertEqual(extract_mime_from_url("https://x.com/a.PNG"), "image/png")


# ===========================================================================
# extract_mime_and_data_from_data_uri — data-URI parsing
# ===========================================================================

class TestExtractMimeAndDataFromDataUri(unittest.TestCase):
    def test_valid_png_data_uri(self):
        result = extract_mime_and_data_from_data_uri(_TINY_PNG_DATA_URI)
        self.assertIsNotNone(result)
        mime, data = result  # type: ignore[misc]
        self.assertEqual(mime, "image/png")
        self.assertEqual(data, _TINY_PNG_BASE64)
        # Round-trip — base64 must still decode
        self.assertEqual(base64.b64decode(data)[:8], b"\x89PNG\r\n\x1a\n")

    def test_jpeg_data_uri(self):
        uri = "data:image/jpeg;base64,/9j/4AAQ"
        result = extract_mime_and_data_from_data_uri(uri)
        self.assertEqual(result, ("image/jpeg", "/9j/4AAQ"))

    def test_not_data_uri_returns_none(self):
        self.assertIsNone(extract_mime_and_data_from_data_uri("https://x.com/a.png"))

    def test_malformed_data_uri_missing_base64_separator(self):
        self.assertIsNone(extract_mime_and_data_from_data_uri("data:image/png,abcd"))

    def test_malformed_data_uri_empty_mime(self):
        self.assertIsNone(extract_mime_and_data_from_data_uri("data:;base64,abcd"))

    def test_malformed_data_uri_empty_data(self):
        self.assertIsNone(extract_mime_and_data_from_data_uri("data:image/png;base64,"))

    def test_empty_string(self):
        self.assertIsNone(extract_mime_and_data_from_data_uri(""))


# ===========================================================================
# is_safe_image_url — scheme allowlist
# ===========================================================================

class TestIsSafeImageUrl(unittest.TestCase):
    def test_https_allowed(self):
        self.assertTrue(is_safe_image_url("https://example.com/a.png"))

    def test_http_allowed(self):
        # HTTP is allowed (caller awareness) but C2 keeps it as URL source
        self.assertTrue(is_safe_image_url("http://example.com/a.png"))

    def test_valid_data_uri_allowed(self):
        self.assertTrue(is_safe_image_url(_TINY_PNG_DATA_URI))

    def test_javascript_scheme_rejected(self):
        self.assertFalse(is_safe_image_url("javascript:alert(1)"))

    def test_file_scheme_rejected(self):
        self.assertFalse(is_safe_image_url("file:///etc/passwd"))

    def test_ftp_rejected(self):
        self.assertFalse(is_safe_image_url("ftp://example.com/x.png"))

    def test_empty_rejected(self):
        self.assertFalse(is_safe_image_url(""))

    def test_non_string_rejected(self):
        self.assertFalse(is_safe_image_url(None))
        self.assertFalse(is_safe_image_url(123))

    def test_malformed_data_uri_rejected(self):
        # data: prefix but not parseable — rejected because we couldn't
        # extract a clean (mime, data) pair.
        self.assertFalse(is_safe_image_url("data:image/png,nobase64here"))


# ===========================================================================
# build_image_part — OpenAI-compat shape
# ===========================================================================

class TestBuildImagePartOpenAIShape(unittest.TestCase):
    def test_https_url_emits_image_url_block(self):
        part = build_image_part("https://x.com/a.png", model_id="gpt-4o")
        self.assertEqual(part, {
            "type": "image_url",
            "image_url": {"url": "https://x.com/a.png"},
        })

    def test_openrouter_anthropic_uses_openai_shape(self):
        """Critical: a common Claude route is ``openrouter/anthropic/...``
        which tunnels through OpenRouter and MUST stay on the OpenAI shape."""
        part = build_image_part(
            "https://x.com/a.png",
            model_id="openrouter/anthropic/claude-3-5-sonnet",
        )
        self.assertIsNotNone(part)
        self.assertEqual(part["type"], "image_url")  # type: ignore[index]

    def test_detail_field_plumbed_when_set(self):
        part = build_image_part("https://x.com/a.png", model_id="gpt-4o", detail="low")
        self.assertEqual(part, {
            "type": "image_url",
            "image_url": {"url": "https://x.com/a.png", "detail": "low"},
        })

    def test_detail_high_plumbed(self):
        part = build_image_part("https://x.com/a.png", model_id="gpt-4o", detail="high")
        self.assertIsNotNone(part)
        self.assertEqual(part["image_url"]["detail"], "high")  # type: ignore[index]

    def test_detail_auto_plumbed(self):
        part = build_image_part("https://x.com/a.png", model_id="gpt-4o", detail="auto")
        self.assertIsNotNone(part)
        self.assertEqual(part["image_url"]["detail"], "auto")  # type: ignore[index]

    def test_detail_invalid_value_omitted(self):
        """Defensive — unknown detail values silently dropped (provider
        default applies) rather than 400'd by the provider."""
        part = build_image_part("https://x.com/a.png", model_id="gpt-4o", detail="ultra")
        self.assertIsNotNone(part)
        self.assertNotIn("detail", part["image_url"])  # type: ignore[index]

    def test_no_detail_omits_field(self):
        part = build_image_part("https://x.com/a.png", model_id="gpt-4o")
        self.assertIsNotNone(part)
        self.assertNotIn("detail", part["image_url"])  # type: ignore[index]

    def test_data_uri_passes_through_for_openai(self):
        """OpenAI accepts ``image_url.url`` containing a ``data:`` URI."""
        part = build_image_part(_TINY_PNG_DATA_URI, model_id="gpt-4o")
        self.assertIsNotNone(part)
        self.assertEqual(part["type"], "image_url")  # type: ignore[index]
        self.assertEqual(part["image_url"]["url"], _TINY_PNG_DATA_URI)  # type: ignore[index]

    def test_no_model_id_defaults_to_openai_shape(self):
        """Backward-compat — callers that don't pass model_id get OpenAI shape."""
        part = build_image_part("https://x.com/a.png", model_id=None)
        self.assertIsNotNone(part)
        self.assertEqual(part["type"], "image_url")  # type: ignore[index]

    def test_mistral_pixtral_route_keeps_openai_shape(self):
        part = build_image_part("https://x.com/a.png", model_id="mistral/pixtral-12b")
        self.assertIsNotNone(part)
        self.assertEqual(part["type"], "image_url")  # type: ignore[index]

    def test_gemini_route_keeps_openai_shape(self):
        part = build_image_part("https://x.com/a.png", model_id="gemini-2.5-pro")
        self.assertIsNotNone(part)
        self.assertEqual(part["type"], "image_url")  # type: ignore[index]


# ===========================================================================
# build_image_part — Anthropic native shape
# ===========================================================================

class TestBuildImagePartAnthropicShape(unittest.TestCase):
    def test_https_url_emits_image_source_url(self):
        part = build_image_part("https://x.com/a.png", model_id="claude-3-5-sonnet-20241022")
        self.assertEqual(part, {
            "type": "image",
            "source": {"type": "url", "url": "https://x.com/a.png"},
        })

    def test_anthropic_slash_prefix_also_routes_native(self):
        part = build_image_part("https://x.com/a.png", model_id="anthropic/claude-3-haiku")
        self.assertIsNotNone(part)
        self.assertEqual(part["type"], "image")  # type: ignore[index]
        self.assertEqual(part["source"]["type"], "url")  # type: ignore[index]

    def test_data_uri_emits_base64_source(self):
        """Data URI is split into media_type + data per Anthropic spec."""
        part = build_image_part(_TINY_PNG_DATA_URI, model_id="claude-3-5-sonnet")
        self.assertIsNotNone(part)
        self.assertEqual(part["type"], "image")  # type: ignore[index]
        self.assertEqual(part["source"], {  # type: ignore[index]
            "type": "base64",
            "media_type": "image/png",
            "data": _TINY_PNG_BASE64,
        })

    def test_jpeg_data_uri_anthropic(self):
        uri = "data:image/jpeg;base64,/9j/4AAQABCD"
        part = build_image_part(uri, model_id="claude-opus-4-7")
        self.assertIsNotNone(part)
        self.assertEqual(part["source"]["media_type"], "image/jpeg")  # type: ignore[index]
        self.assertEqual(part["source"]["data"], "/9j/4AAQABCD")  # type: ignore[index]

    def test_detail_silently_dropped_on_anthropic_path(self):
        """Anthropic Messages API has no `detail` concept — must not appear."""
        part = build_image_part("https://x.com/a.png", model_id="claude-3-5-sonnet", detail="low")
        self.assertIsNotNone(part)
        # No `detail` anywhere in the part
        self.assertNotIn("detail", json.dumps(part))

    def test_http_url_emits_url_source_on_anthropic(self):
        """slimllm doesn't fetch+base64 like LiteLLM does for Bedrock — we
        just pass the URL through as source.type=url."""
        part = build_image_part("http://x.com/a.png", model_id="claude-3-5-sonnet")
        self.assertIsNotNone(part)
        self.assertEqual(part["source"]["type"], "url")  # type: ignore[index]
        self.assertEqual(part["source"]["url"], "http://x.com/a.png")  # type: ignore[index]


# ===========================================================================
# build_image_part — safety rejection
# ===========================================================================

class TestBuildImagePartSafetyRejection(unittest.TestCase):
    """Unsafe URLs return None so callers skip the image."""

    def test_javascript_scheme_returns_none_for_openai(self):
        self.assertIsNone(build_image_part("javascript:alert(1)", model_id="gpt-4o"))

    def test_javascript_scheme_returns_none_for_anthropic(self):
        self.assertIsNone(build_image_part("javascript:alert(1)", model_id="claude-3-5-sonnet"))

    def test_file_scheme_returns_none(self):
        self.assertIsNone(build_image_part("file:///etc/passwd", model_id="gpt-4o"))

    def test_ftp_scheme_returns_none(self):
        self.assertIsNone(build_image_part("ftp://x.com/a.png", model_id="gpt-4o"))

    def test_empty_url_returns_none(self):
        self.assertIsNone(build_image_part("", model_id="gpt-4o"))

    def test_malformed_data_uri_returns_none(self):
        self.assertIsNone(build_image_part("data:image/png,nobase64here", model_id="gpt-4o"))


if __name__ == "__main__":
    unittest.main()
