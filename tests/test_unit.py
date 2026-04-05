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
from slimllm.providers.openai import OpenAIProvider, OpenRouterProvider
from slimllm.types import (
    DeltaMessage,
    FunctionCall,
    ModelResponse,
    StreamingChunk,
    StreamingChoice,
    StreamResponse,
    ToolCall,
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
