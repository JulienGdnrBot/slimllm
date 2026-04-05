"""
Unit tests — no network calls, stdlib unittest only.

Tests cover:
  - Provider routing
  - Anthropic message/tool format conversion
  - OpenAI response parsing
  - Anthropic response parsing
  - SSE event parsing
  - StreamResponse aggregation
  - Streaming chunk parsing (OpenAI + Anthropic)
"""
from __future__ import annotations

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import slimllm
from slimllm._sse import iter_events
from slimllm.main import _route
from slimllm.providers.anthropic import AnthropicProvider
from slimllm.providers.openai import (
    DeepSeekProvider,
    GoogleAIStudioProvider,
    MistralProvider,
    OpenAIProvider,
    OpenRouterProvider,
)
from slimllm.main import _norm_msg, _norm_tool
from slimllm.types import (
    ContentBlock,
    DeltaMessage,
    FunctionCall,
    FunctionDefinition,
    Message,
    ModelResponse,
    ProviderConfig,
    RetryConfig,
    StreamingChunk,
    StreamingChoice,
    StreamResponse,
    Tool,
    ToolCall,
    Usage,
)


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

class TestRouting(unittest.TestCase):
    def test_openai_default(self):
        provider, model = _route("gpt-4o")
        self.assertIsInstance(provider, OpenAIProvider)
        self.assertEqual(model, "gpt-4o")

    def test_claude_prefix(self):
        provider, model = _route("claude-3-5-sonnet-20241022")
        self.assertIsInstance(provider, AnthropicProvider)
        self.assertEqual(model, "claude-3-5-sonnet-20241022")

    def test_anthropic_prefix(self):
        provider, model = _route("anthropic/claude-opus-4-5")
        self.assertIsInstance(provider, AnthropicProvider)
        self.assertEqual(model, "claude-opus-4-5")

    def test_openrouter_prefix(self):
        provider, model = _route("openrouter/meta-llama/llama-3.3-70b")
        self.assertIsInstance(provider, OpenRouterProvider)
        self.assertEqual(model, "meta-llama/llama-3.3-70b")

    def test_mistral_bare(self):
        provider, model = _route("mistral-large-2512")
        self.assertIsInstance(provider, MistralProvider)
        self.assertEqual(model, "mistral-large-2512")

    def test_mistral_prefix(self):
        provider, model = _route("mistral/mistral-medium")
        self.assertIsInstance(provider, MistralProvider)
        self.assertEqual(model, "mistral-medium")

    def test_codestral_bare(self):
        provider, model = _route("codestral-latest")
        self.assertIsInstance(provider, MistralProvider)
        self.assertEqual(model, "codestral-latest")

    def test_devstral_bare(self):
        provider, model = _route("devstral-2-2512")
        self.assertIsInstance(provider, MistralProvider)
        self.assertEqual(model, "devstral-2-2512")

    def test_deepseek_bare(self):
        provider, model = _route("deepseek-chat")
        self.assertIsInstance(provider, DeepSeekProvider)
        self.assertEqual(model, "deepseek-chat")

    def test_deepseek_prefix(self):
        provider, model = _route("deepseek/deepseek-r1")
        self.assertIsInstance(provider, DeepSeekProvider)
        self.assertEqual(model, "deepseek-r1")

    def test_gemini_bare(self):
        provider, model = _route("gemini-2.5-flash")
        self.assertIsInstance(provider, GoogleAIStudioProvider)
        self.assertEqual(model, "gemini-2.5-flash")

    def test_gemini_prefix(self):
        provider, model = _route("gemini/gemini-2.5-pro")
        self.assertIsInstance(provider, GoogleAIStudioProvider)
        self.assertEqual(model, "gemini-2.5-pro")

    def test_googleaistudio_prefix(self):
        provider, model = _route("googleaistudio/gemini-2.0-flash")
        self.assertIsInstance(provider, GoogleAIStudioProvider)
        self.assertEqual(model, "gemini-2.0-flash")


# ---------------------------------------------------------------------------
# SSE parser
# ---------------------------------------------------------------------------

class TestSSEParser(unittest.TestCase):
    def test_openai_stream(self):
        lines = [
            'data: {"id":"c1","choices":[{"delta":{"content":"Hello"}}]}',
            "",
            "data: [DONE]",
            "",
        ]
        events = list(iter_events(iter(lines)))
        self.assertEqual(events[0], (None, '{"id":"c1","choices":[{"delta":{"content":"Hello"}}]}'))
        self.assertEqual(events[1], (None, "[DONE]"))

    def test_anthropic_stream(self):
        lines = [
            "event: message_start",
            'data: {"type":"message_start","message":{"id":"msg_1"}}',
            "",
            "event: content_block_delta",
            'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"hi"}}',
            "",
        ]
        events = list(iter_events(iter(lines)))
        self.assertEqual(events[0][0], "message_start")
        self.assertEqual(events[1][0], "content_block_delta")

    def test_empty_lines_ignored(self):
        lines = ["", "", "data: hello", ""]
        events = list(iter_events(iter(lines)))
        self.assertEqual(events, [(None, "hello")])


# ---------------------------------------------------------------------------
# Anthropic message/tool conversions
# ---------------------------------------------------------------------------

class TestAnthropicConversions(unittest.TestCase):
    def setUp(self):
        self.p = AnthropicProvider()

    def test_system_extracted(self):
        messages = [
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "Hi"},
        ]
        body = self.p._build_body(
            model="claude-3-haiku-20240307",
            messages=messages,
            stream=False,
            tools=None,
            tool_choice=None,
            temperature=None,
            max_tokens=None,
            top_p=None,
            stop=None,
        )
        self.assertEqual(body["system"], "Be helpful.")
        self.assertTrue(all(m["role"] != "system" for m in body["messages"]))

    def test_tool_result_conversion(self):
        msg = {"role": "tool", "content": '{"temp":72}', "tool_call_id": "toolu_abc"}
        result = self.p._convert_message(msg)
        self.assertEqual(result["role"], "user")
        self.assertEqual(result["content"][0]["type"], "tool_result")
        self.assertEqual(result["content"][0]["tool_use_id"], "toolu_abc")

    def test_assistant_tool_call_conversion(self):
        msg = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "search", "arguments": '{"q":"python"}'},
                }
            ],
        }
        result = self.p._convert_message(msg)
        self.assertEqual(result["role"], "assistant")
        self.assertEqual(result["content"][0]["type"], "tool_use")
        self.assertEqual(result["content"][0]["input"], {"q": "python"})

    def test_tool_definition_conversion(self):
        tool = {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                },
            },
        }
        result = self.p._convert_tool(tool)
        self.assertEqual(result["name"], "get_weather")
        self.assertIn("input_schema", result)
        self.assertEqual(result["input_schema"]["properties"]["city"]["type"], "string")

    def test_tool_choice_auto(self):
        self.assertEqual(self.p._convert_tool_choice("auto"), {"type": "auto"})

    def test_tool_choice_none(self):
        self.assertEqual(self.p._convert_tool_choice("none"), {"type": "none"})

    def test_tool_choice_required(self):
        self.assertEqual(self.p._convert_tool_choice("required"), {"type": "any"})

    def test_tool_choice_function(self):
        tc = {"type": "function", "function": {"name": "search"}}
        self.assertEqual(self.p._convert_tool_choice(tc), {"type": "tool", "name": "search"})

    def test_finish_reason_mapping(self):
        self.assertEqual(AnthropicProvider._map_finish_reason("end_turn"), "stop")
        self.assertEqual(AnthropicProvider._map_finish_reason("max_tokens"), "length")
        self.assertEqual(AnthropicProvider._map_finish_reason("tool_use"), "tool_calls")


# ---------------------------------------------------------------------------
# Response parsing (no network)
# ---------------------------------------------------------------------------

class TestOpenAIResponseParsing(unittest.TestCase):
    def setUp(self):
        self.p = OpenAIProvider()

    def test_simple_text_response(self):
        data = {
            "id": "chatcmpl-xyz",
            "model": "gpt-4o",
            "created": 1234567890,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello world"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        resp = self.p._parse_response(data)
        self.assertIsInstance(resp, ModelResponse)
        self.assertEqual(resp.content, "Hello world")
        self.assertEqual(resp.choices[0].finish_reason, "stop")
        self.assertEqual(resp.usage.total_tokens, 15)

    def test_tool_call_response(self):
        data = {
            "id": "chatcmpl-tc",
            "model": "gpt-4o",
            "created": 0,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"city":"Paris"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
        }
        resp = self.p._parse_response(data)
        self.assertIsNotNone(resp.tool_calls)
        self.assertEqual(len(resp.tool_calls), 1)
        self.assertEqual(resp.tool_calls[0].function.name, "get_weather")
        self.assertEqual(resp.tool_calls[0].id, "call_1")


class TestAnthropicResponseParsing(unittest.TestCase):
    def setUp(self):
        self.p = AnthropicProvider()

    def test_simple_text_response(self):
        data = {
            "id": "msg_001",
            "model": "claude-3-5-sonnet-20241022",
            "stop_reason": "end_turn",
            "content": [{"type": "text", "text": "Hello!"}],
            "usage": {"input_tokens": 8, "output_tokens": 3},
        }
        resp = self.p._parse_response(data)
        self.assertEqual(resp.content, "Hello!")
        self.assertEqual(resp.choices[0].finish_reason, "stop")
        self.assertEqual(resp.usage.prompt_tokens, 8)
        self.assertEqual(resp.usage.completion_tokens, 3)

    def test_tool_use_response(self):
        data = {
            "id": "msg_002",
            "model": "claude-3-5-sonnet-20241022",
            "stop_reason": "tool_use",
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_01",
                    "name": "search",
                    "input": {"q": "python"},
                }
            ],
            "usage": {"input_tokens": 20, "output_tokens": 15},
        }
        resp = self.p._parse_response(data)
        self.assertIsNotNone(resp.tool_calls)
        tc = resp.tool_calls[0]
        self.assertEqual(tc.id, "toolu_01")
        self.assertEqual(tc.function.name, "search")
        self.assertEqual(json.loads(tc.function.arguments), {"q": "python"})
        self.assertEqual(resp.choices[0].finish_reason, "tool_calls")


# ---------------------------------------------------------------------------
# StreamResponse aggregation
# ---------------------------------------------------------------------------

class TestStreamResponse(unittest.TestCase):
    def _make_chunks(self):
        return [
            StreamingChunk(
                id="c1", model="gpt-4o",
                choices=[StreamingChoice(0, DeltaMessage(role="assistant"))],
            ),
            StreamingChunk(
                id="c1", model="gpt-4o",
                choices=[StreamingChoice(0, DeltaMessage(content="Hello "))],
            ),
            StreamingChunk(
                id="c1", model="gpt-4o",
                choices=[StreamingChoice(0, DeltaMessage(content="world"))],
            ),
            StreamingChunk(
                id="c1", model="gpt-4o",
                choices=[StreamingChoice(0, DeltaMessage(), finish_reason="stop")],
            ),
        ]

    def test_text_aggregation(self):
        stream = StreamResponse(iter(self._make_chunks()))
        chunks = list(stream)
        self.assertEqual(len(chunks), 4)
        final = stream.get_final_response()
        self.assertEqual(final.content, "Hello world")
        self.assertEqual(final.choices[0].finish_reason, "stop")
        self.assertEqual(final.model, "gpt-4o")

    def test_get_final_drains_generator(self):
        stream = StreamResponse(iter(self._make_chunks()))
        final = stream.get_final_response()
        self.assertEqual(final.content, "Hello world")

    def test_tool_call_aggregation(self):
        chunks = [
            StreamingChunk(
                id="c1", model="gpt-4o",
                choices=[StreamingChoice(0, DeltaMessage(
                    tool_calls=[ToolCall(id="call_1", function=FunctionCall(name="search", arguments=""))]
                ))],
            ),
            StreamingChunk(
                id="c1", model="gpt-4o",
                choices=[StreamingChoice(0, DeltaMessage(
                    tool_calls=[ToolCall(id="", function=FunctionCall(name="", arguments='{"q":'))]
                ))],
            ),
            StreamingChunk(
                id="c1", model="gpt-4o",
                choices=[StreamingChoice(0, DeltaMessage(
                    tool_calls=[ToolCall(id="", function=FunctionCall(name="", arguments='"python"}'))]
                ))],
            ),
            StreamingChunk(
                id="c1", model="gpt-4o",
                choices=[StreamingChoice(0, DeltaMessage(), finish_reason="tool_calls")],
            ),
        ]
        stream = StreamResponse(iter(chunks))
        final = stream.get_final_response()
        self.assertIsNotNone(final.tool_calls)
        self.assertEqual(final.tool_calls[0].function.name, "search")
        self.assertEqual(final.tool_calls[0].function.arguments, '{"q":"python"}')


# ---------------------------------------------------------------------------
# OpenAI streaming chunk parsing
# ---------------------------------------------------------------------------

class TestOpenAIStreamParsing(unittest.TestCase):
    def setUp(self):
        self.p = OpenAIProvider()

    def test_text_chunk(self):
        raw = {
            "id": "c1", "model": "gpt-4o", "created": 0,
            "choices": [{"index": 0, "delta": {"content": "Hi"}, "finish_reason": None}],
        }
        chunk = self.p._parse_chunk(raw, "gpt-4o")
        self.assertEqual(chunk.choices[0].delta.content, "Hi")

    def test_tool_call_chunk(self):
        raw = {
            "id": "c1", "model": "gpt-4o", "created": 0,
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {"id": "call_x", "function": {"name": "foo", "arguments": ""}}
                        ]
                    },
                    "finish_reason": None,
                }
            ],
        }
        chunk = self.p._parse_chunk(raw, "gpt-4o")
        tcs = chunk.choices[0].delta.tool_calls
        self.assertIsNotNone(tcs)
        self.assertEqual(tcs[0].function.name, "foo")


# ---------------------------------------------------------------------------
# RetryConfig / ProviderConfig dataclasses
# ---------------------------------------------------------------------------

class TestDataclasses(unittest.TestCase):
    def test_retry_config_defaults(self):
        cfg = RetryConfig()
        self.assertEqual(cfg.max_retries, 3)
        self.assertEqual(cfg.backoff_base, 1.0)
        self.assertIn(429, cfg.retryable_status_codes)
        self.assertIn(500, cfg.retryable_status_codes)

    def test_retry_config_custom(self):
        cfg = RetryConfig(max_retries=0)
        self.assertEqual(cfg.max_retries, 0)

    def test_provider_config_defaults(self):
        cfg = ProviderConfig(api_key="sk-test")
        self.assertEqual(cfg.api_key, "sk-test")
        self.assertIsNone(cfg.base_url)
        self.assertIsNone(cfg.extra_headers)
        self.assertIsNone(cfg.retry)

    def test_provider_config_with_retry(self):
        retry = RetryConfig(max_retries=1, backoff_base=0.5)
        cfg = ProviderConfig(api_key="sk-test", retry=retry)
        self.assertEqual(cfg.retry.max_retries, 1)


# ---------------------------------------------------------------------------
# Retry logic (_http.post_json)
# ---------------------------------------------------------------------------

class TestRetryLogic(unittest.TestCase):
    def test_no_retry_on_success(self):
        """post_json should not retry when the first call succeeds."""
        import slimllm._http as http_mod

        call_count = 0

        def fake_once(url, headers, body, timeout):
            nonlocal call_count
            call_count += 1
            return 200, {"ok": True}

        original = http_mod._post_json_once
        http_mod._post_json_once = fake_once
        try:
            status, data = http_mod.post_json("https://example.com", {}, {})
            self.assertEqual(status, 200)
            self.assertEqual(call_count, 1)
        finally:
            http_mod._post_json_once = original

    def test_retries_on_429(self):
        """post_json should retry up to max_retries on retryable status."""
        import slimllm._http as http_mod

        call_count = 0
        sleeps = []

        def fake_once(url, headers, body, timeout):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                return 429, {"error": "rate limit"}
            return 200, {"ok": True}

        def fake_sleep(secs):
            sleeps.append(secs)

        original_once = http_mod._post_json_once
        original_sleep = http_mod.time.sleep
        http_mod._post_json_once = fake_once
        http_mod.time.sleep = fake_sleep
        try:
            cfg = RetryConfig(max_retries=3, backoff_base=0.1)
            status, data = http_mod.post_json("https://example.com", {}, {}, retry=cfg)
            self.assertEqual(status, 200)
            self.assertEqual(call_count, 3)
            self.assertEqual(len(sleeps), 2)  # slept before attempts 2 and 3
        finally:
            http_mod._post_json_once = original_once
            http_mod.time.sleep = original_sleep

    def test_no_retry_when_disabled(self):
        """RetryConfig(max_retries=0) should not retry at all."""
        import slimllm._http as http_mod

        call_count = 0

        def fake_once(url, headers, body, timeout):
            nonlocal call_count
            call_count += 1
            return 500, {"error": "server error"}

        original = http_mod._post_json_once
        http_mod._post_json_once = fake_once
        try:
            cfg = RetryConfig(max_retries=0)
            status, _ = http_mod.post_json("https://example.com", {}, {}, retry=cfg)
            self.assertEqual(status, 500)
            self.assertEqual(call_count, 1)
        finally:
            http_mod._post_json_once = original


# ---------------------------------------------------------------------------
# Message dataclass
# ---------------------------------------------------------------------------

class TestMessage(unittest.TestCase):
    def test_user_factory(self):
        m = Message.user("hello")
        self.assertEqual(m.role, "user")
        self.assertEqual(m.content, "hello")

    def test_system_factory(self):
        m = Message.system("you are helpful")
        self.assertEqual(m.role, "system")

    def test_assistant_factory(self):
        m = Message.assistant("hi there")
        self.assertEqual(m.role, "assistant")
        self.assertEqual(m.content, "hi there")

    def test_tool_result_factory(self):
        m = Message.tool_result("call_123", '{"result": 42}')
        self.assertEqual(m.role, "tool")
        self.assertEqual(m.tool_call_id, "call_123")

    def test_to_dict_simple(self):
        d = Message.user("hello").to_dict()
        self.assertEqual(d, {"role": "user", "content": "hello"})

    def test_to_dict_tool_result(self):
        d = Message.tool_result("call_1", "ok").to_dict()
        self.assertEqual(d["role"], "tool")
        self.assertEqual(d["tool_call_id"], "call_1")

    def test_to_dict_assistant_with_tool_calls(self):
        tc = ToolCall(id="call_1", function=FunctionCall(name="foo", arguments='{"x":1}'))
        m = Message.assistant(tool_calls=[tc])
        d = m.to_dict()
        self.assertEqual(d["role"], "assistant")
        self.assertEqual(d["tool_calls"][0]["id"], "call_1")
        self.assertEqual(d["tool_calls"][0]["function"]["name"], "foo")

    def test_to_dict_content_list(self):
        blocks = [ContentBlock.text_block("hi"), ContentBlock.text_block("there")]
        m = Message.user(blocks)
        d = m.to_dict()
        self.assertEqual(d["content"][0], {"type": "text", "text": "hi"})

    def test_from_dict_simple(self):
        m = Message.from_dict({"role": "user", "content": "hello"})
        self.assertEqual(m.role, "user")
        self.assertEqual(m.content, "hello")

    def test_from_dict_with_tool_calls(self):
        raw = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "c1", "function": {"name": "bar", "arguments": "{}"}}
            ],
        }
        m = Message.from_dict(raw)
        self.assertEqual(m.tool_calls[0].id, "c1")
        self.assertEqual(m.tool_calls[0].function.name, "bar")

    def test_roundtrip(self):
        original = {"role": "user", "content": "test roundtrip"}
        self.assertEqual(Message.from_dict(original).to_dict(), original)


# ---------------------------------------------------------------------------
# ContentBlock dataclass
# ---------------------------------------------------------------------------

class TestContentBlock(unittest.TestCase):
    def test_text_block(self):
        b = ContentBlock.text_block("hello")
        self.assertEqual(b.to_dict(), {"type": "text", "text": "hello"})

    def test_text_block_with_cache(self):
        b = ContentBlock.text_block("sys", cache_control={"type": "ephemeral"})
        d = b.to_dict()
        self.assertEqual(d["cache_control"], {"type": "ephemeral"})

    def test_image_block(self):
        b = ContentBlock.image_block("https://example.com/img.png")
        d = b.to_dict()
        self.assertEqual(d["type"], "image_url")
        self.assertEqual(d["image_url"]["url"], "https://example.com/img.png")

    def test_from_dict(self):
        b = ContentBlock.from_dict({"type": "text", "text": "hi", "cache_control": {"type": "ephemeral"}})
        self.assertEqual(b.text, "hi")
        self.assertEqual(b.cache_control, {"type": "ephemeral"})

    def test_from_dict_unknown_fields_in_extra(self):
        b = ContentBlock.from_dict({"type": "text", "text": "x", "custom_field": "y"})
        self.assertEqual(b.extra, {"custom_field": "y"})


# ---------------------------------------------------------------------------
# Tool dataclass
# ---------------------------------------------------------------------------

class TestTool(unittest.TestCase):
    def test_from_function(self):
        t = Tool.from_function(
            "get_weather",
            description="Get weather",
            parameters={"type": "object", "properties": {"city": {"type": "string"}}},
        )
        self.assertEqual(t.type, "function")
        self.assertEqual(t.function.name, "get_weather")

    def test_to_dict(self):
        t = Tool.from_function("foo", description="bar")
        d = t.to_dict()
        self.assertEqual(d["type"], "function")
        self.assertEqual(d["function"]["name"], "foo")
        self.assertEqual(d["function"]["description"], "bar")

    def test_from_dict_roundtrip(self):
        raw = {
            "type": "function",
            "function": {
                "name": "add",
                "description": "adds two numbers",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        t = Tool.from_dict(raw)
        self.assertEqual(t.to_dict(), raw)


# ---------------------------------------------------------------------------
# Input normalisation
# ---------------------------------------------------------------------------

class TestNormalisation(unittest.TestCase):
    def test_norm_msg_passthrough_dict(self):
        d = {"role": "user", "content": "hi"}
        self.assertIs(_norm_msg(d), d)

    def test_norm_msg_from_dataclass(self):
        m = Message.user("hi")
        result = _norm_msg(m)
        self.assertIsInstance(result, dict)
        self.assertEqual(result["role"], "user")

    def test_norm_tool_passthrough_dict(self):
        d = {"type": "function", "function": {"name": "f"}}
        self.assertIs(_norm_tool(d), d)

    def test_norm_tool_from_dataclass(self):
        t = Tool.from_function("f")
        result = _norm_tool(t)
        self.assertIsInstance(result, dict)
        self.assertEqual(result["function"]["name"], "f")


# ---------------------------------------------------------------------------
# Stream objects — reasoning_content and usage
# ---------------------------------------------------------------------------

class TestStreamObjects(unittest.TestCase):
    def test_delta_message_reasoning_content(self):
        delta = DeltaMessage(content="hello", reasoning_content="let me think")
        self.assertEqual(delta.reasoning_content, "let me think")

    def test_streaming_chunk_usage(self):
        chunk = StreamingChunk(
            model="gpt-4o",
            choices=[],
            usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )
        self.assertEqual(chunk.usage.total_tokens, 15)

    def test_stream_response_aggregates_reasoning(self):
        chunks = [
            StreamingChunk(
                id="c1", model="gpt-4o",
                choices=[StreamingChoice(
                    index=0,
                    delta=DeltaMessage(reasoning_content="thinking... "),
                )],
            ),
            StreamingChunk(
                id="c1", model="gpt-4o",
                choices=[StreamingChoice(
                    index=0,
                    delta=DeltaMessage(content="answer"),
                    finish_reason="stop",
                )],
            ),
        ]
        stream = StreamResponse(iter(chunks))
        final = stream.get_final_response()
        self.assertEqual(final.content, "answer")

    def test_stream_response_final_usage(self):
        chunks = [
            StreamingChunk(id="c1", model="m", choices=[]),
            StreamingChunk(
                id="c1", model="m", choices=[],
                usage=Usage(prompt_tokens=10, completion_tokens=3, total_tokens=13),
            ),
        ]
        stream = StreamResponse(iter(chunks))
        final = stream.get_final_response()
        self.assertIsNotNone(final.usage)
        self.assertEqual(final.usage.total_tokens, 13)


if __name__ == "__main__":
    unittest.main(verbosity=2)
