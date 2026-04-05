"""
Anthropic Messages API provider.

Wire differences vs OpenAI:
  - Endpoint: POST /v1/messages
  - System prompt is a top-level field, NOT a message
  - Tool definitions use `input_schema` (not `parameters`)
  - Tool results come as user messages with content blocks
  - SSE events have named `event:` fields (content_block_delta, etc.)
  - Tool call streaming uses input_json_delta partial chunks

This module translates the OpenAI-shaped inputs that slimllm exposes into
Anthropic wire format, and translates Anthropic responses back to the
OpenAI-shaped slimllm types.
"""
from __future__ import annotations

import json
import uuid
from typing import Any, Dict, Generator, List, Optional, Union

from .._http import post_json, stream_lines
from .._sse import iter_events
from ..exceptions import ProviderError, raise_for_status
from ..types import (
    Choice,
    ContentBlock,
    DeltaMessage,
    FunctionCall,
    Message,
    ModelResponse,
    StreamingChunk,
    StreamingChoice,
    ToolCall,
    Usage,
)
from ._base import BaseProvider

ANTHROPIC_BASE = "https://api.anthropic.com"
ANTHROPIC_VERSION = "2023-06-01"
DEFAULT_MAX_TOKENS = 4096

# OpenAI-specific params that Anthropic's API does not accept.
# These are silently dropped rather than forwarded.
_ANTHROPIC_DROP_PARAMS = frozenset({
    "user", "stream_options", "logprobs", "logit_bias",
    "n", "presence_penalty", "frequency_penalty", "best_of",
    "suffix", "echo", "completion_tokens_details",
})


class AnthropicProvider(BaseProvider):

    def completion(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        *,
        api_key: str,
        stream: bool = False,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        stop: Optional[Union[str, List[str]]] = None,
        response_format: Optional[Dict[str, Any]] = None,
        extra_headers: Optional[Dict[str, str]] = None,
        api_base: Optional[str] = None,
        **kwargs: Any,
    ) -> Union[ModelResponse, Generator[StreamingChunk, None, None]]:
        base = (api_base or ANTHROPIC_BASE).rstrip("/")
        url = f"{base}/v1/messages"
        headers = self._build_headers(api_key, extra_headers)
        body = self._build_body(
            model=model,
            messages=messages,
            stream=stream,
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stop=stop,
            **kwargs,
        )

        if stream:
            return self._stream(url, headers, body, model)
        return self._complete(url, headers, body)

    # ------------------------------------------------------------------
    # Header / body construction
    # ------------------------------------------------------------------

    def _build_headers(
        self,
        api_key: str,
        extra: Optional[Dict[str, str]],
    ) -> Dict[str, str]:
        h = {
            "x-api-key": api_key,
            "anthropic-version": ANTHROPIC_VERSION,
        }
        if extra:
            h.update(extra)
        return h

    def _build_body(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        stream: bool,
        tools: Optional[List[Dict[str, Any]]],
        tool_choice: Optional[Union[str, Dict]],
        temperature: Optional[float],
        max_tokens: Optional[int],
        top_p: Optional[float],
        stop: Optional[Union[str, List[str]]],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        # Extract system prompt(s) — Anthropic puts them at the top level
        system: Optional[str] = None
        filtered_messages = []
        for msg in messages:
            if msg.get("role") == "system":
                content = msg.get("content", "")
                if isinstance(content, list):
                    # Already a list of content blocks (e.g. with cache_control) — use as-is
                    system = content
                elif system is None:
                    system = content
                else:
                    system += "\n\n" + content
            else:
                filtered_messages.append(self._convert_message(msg))

        body: Dict[str, Any] = {
            "model": model,
            "messages": filtered_messages,
            "max_tokens": max_tokens or DEFAULT_MAX_TOKENS,
            "stream": stream,
        }
        if system:
            body["system"] = system
        if tools:
            body["tools"] = [self._convert_tool(t) for t in tools]
        if tool_choice is not None:
            body["tool_choice"] = self._convert_tool_choice(tool_choice)
        if temperature is not None:
            body["temperature"] = temperature
        if top_p is not None:
            body["top_p"] = top_p
        if stop:
            body["stop_sequences"] = [stop] if isinstance(stop, str) else stop

        # Drop OpenAI-specific params that Anthropic rejects
        body.update({k: v for k, v in kwargs.items() if k not in _ANTHROPIC_DROP_PARAMS})
        return body

    # ------------------------------------------------------------------
    # Message / tool format converters (OpenAI → Anthropic)
    # ------------------------------------------------------------------

    def _convert_message(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        role = msg.get("role")

        # Tool result: OpenAI {"role":"tool","content":"...","tool_call_id":"..."}
        # → Anthropic {"role":"user","content":[{"type":"tool_result",...}]}
        if role == "tool":
            return {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": msg.get("tool_call_id", ""),
                        "content": msg.get("content", ""),
                    }
                ],
            }

        # Assistant message with tool calls:
        # OpenAI: {"role":"assistant","content":null,"tool_calls":[...]}
        # Anthropic: {"role":"assistant","content":[text?, tool_use...]}
        if role == "assistant":
            raw_tcs = msg.get("tool_calls")
            if raw_tcs:
                content_blocks: List[Dict[str, Any]] = []
                text = msg.get("content")
                if text:
                    content_blocks.append({"type": "text", "text": text})
                for tc in raw_tcs:
                    fn = tc.get("function", {})
                    try:
                        tool_input = json.loads(fn.get("arguments", "{}"))
                    except json.JSONDecodeError:
                        tool_input = {}
                    content_blocks.append(
                        {
                            "type": "tool_use",
                            "id": tc.get("id", f"toolu_{uuid.uuid4().hex[:12]}"),
                            "name": fn.get("name", ""),
                            "input": tool_input,
                        }
                    )
                return {"role": "assistant", "content": content_blocks}

        # Plain text message — keep as-is (content may be str or list)
        return {"role": role, "content": msg.get("content", "")}

    def _convert_tool(self, tool: Dict[str, Any]) -> Dict[str, Any]:
        """OpenAI tool → Anthropic tool definition."""
        fn = tool.get("function", {})
        return {
            "name": fn.get("name", ""),
            "description": fn.get("description", ""),
            "input_schema": fn.get("parameters") or {"type": "object", "properties": {}},
        }

    def _convert_tool_choice(
        self,
        tc: Union[str, Dict],
    ) -> Dict[str, Any]:
        if isinstance(tc, dict):
            # OpenAI: {"type": "function", "function": {"name": "..."}}
            if tc.get("type") == "function":
                return {"type": "tool", "name": tc["function"]["name"]}
            return tc  # pass through if already Anthropic-shaped
        # String shortcuts
        if tc == "auto":
            return {"type": "auto"}
        if tc == "none":
            return {"type": "none"}
        if tc == "required":
            return {"type": "any"}
        return {"type": "auto"}

    # ------------------------------------------------------------------
    # Non-streaming
    # ------------------------------------------------------------------

    def _complete(
        self,
        url: str,
        headers: Dict[str, str],
        body: Dict[str, Any],
    ) -> ModelResponse:
        status, data = post_json(url, headers, body)

        if isinstance(data, str):
            raise_for_status(status, data, "anthropic")
        if status >= 400:
            err = data.get("error", {}) if isinstance(data, dict) else {}
            raise_for_status(status, err.get("message", str(data)), "anthropic")

        return self._parse_response(data)

    def _parse_response(self, data: Dict[str, Any]) -> ModelResponse:
        content_blocks = data.get("content", [])
        text_parts: List[str] = []
        tool_calls: List[ToolCall] = []

        for block in content_blocks:
            btype = block.get("type")
            if btype == "text":
                text_parts.append(block.get("text", ""))
            elif btype == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=block.get("id", f"toolu_{uuid.uuid4().hex[:12]}"),
                        function=FunctionCall(
                            name=block.get("name", ""),
                            arguments=json.dumps(block.get("input", {})),
                        ),
                    )
                )

        content = "".join(text_parts) or None
        stop_reason = data.get("stop_reason")
        finish_reason = self._map_finish_reason(stop_reason)

        raw_usage = data.get("usage", {})
        usage = Usage(
            prompt_tokens=raw_usage.get("input_tokens", 0),
            completion_tokens=raw_usage.get("output_tokens", 0),
            total_tokens=raw_usage.get("input_tokens", 0) + raw_usage.get("output_tokens", 0),
        ) if raw_usage else None

        return ModelResponse(
            id=data.get("id", f"msg_{uuid.uuid4().hex[:24]}"),
            model=data.get("model", ""),
            choices=[
                Choice(
                    index=0,
                    message=Message(
                        role="assistant",
                        content=content,
                        tool_calls=tool_calls or None,
                    ),
                    finish_reason=finish_reason,
                )
            ],
            usage=usage,
        )

    @staticmethod
    def _map_finish_reason(stop_reason: Optional[str]) -> Optional[str]:
        mapping = {
            "end_turn": "stop",
            "max_tokens": "length",
            "stop_sequence": "stop",
            "tool_use": "tool_calls",
        }
        return mapping.get(stop_reason or "", stop_reason)

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    def _stream(
        self,
        url: str,
        headers: Dict[str, str],
        body: Dict[str, Any],
        model: str,
    ) -> Generator[StreamingChunk, None, None]:
        """
        Anthropic SSE events we care about:
          message_start          → has usage
          content_block_start    → has block type (text / tool_use)
          content_block_delta    → text_delta or input_json_delta
          content_block_stop     → end of a block
          message_delta          → stop_reason / usage
          message_stop           → stream done
        """
        lines = stream_lines(url, headers, body)
        response_id = f"msg_{uuid.uuid4().hex[:24]}"

        # Per-block state for tool call accumulation
        current_block_type: Optional[str] = None
        current_tool_id: Optional[str] = None
        current_tool_name: Optional[str] = None
        # Whether we've emitted the role "assistant" delta yet
        role_emitted = False

        for event_type, data in iter_events(lines):
            if event_type == "message_stop" or data == "[DONE]":
                return

            try:
                payload = json.loads(data)
            except json.JSONDecodeError as exc:
                raise ProviderError(
                    f"Invalid Anthropic SSE JSON: {data!r}", provider="anthropic"
                ) from exc

            if event_type == "message_start":
                msg = payload.get("message", {})
                response_id = msg.get("id", response_id)
                # Emit role delta once
                if not role_emitted:
                    role_emitted = True
                    yield StreamingChunk(
                        id=response_id,
                        model=msg.get("model", model),
                        choices=[
                            StreamingChoice(
                                index=0,
                                delta=DeltaMessage(role="assistant"),
                            )
                        ],
                    )

            elif event_type == "content_block_start":
                block = payload.get("content_block", {})
                current_block_type = block.get("type")
                if current_block_type == "tool_use":
                    current_tool_id = block.get("id", f"toolu_{uuid.uuid4().hex[:12]}")
                    current_tool_name = block.get("name", "")
                    # Emit the tool call start chunk
                    yield StreamingChunk(
                        id=response_id,
                        model=model,
                        choices=[
                            StreamingChoice(
                                index=0,
                                delta=DeltaMessage(
                                    tool_calls=[
                                        ToolCall(
                                            id=current_tool_id,
                                            function=FunctionCall(
                                                name=current_tool_name,
                                                arguments="",
                                            ),
                                        )
                                    ]
                                ),
                            )
                        ],
                    )

            elif event_type == "content_block_delta":
                delta = payload.get("delta", {})
                dtype = delta.get("type")

                if dtype == "text_delta":
                    text = delta.get("text", "")
                    if text:
                        yield StreamingChunk(
                            id=response_id,
                            model=model,
                            choices=[
                                StreamingChoice(
                                    index=0,
                                    delta=DeltaMessage(content=text),
                                )
                            ],
                        )

                elif dtype == "input_json_delta":
                    partial = delta.get("partial_json", "")
                    if partial and current_tool_id:
                        yield StreamingChunk(
                            id=response_id,
                            model=model,
                            choices=[
                                StreamingChoice(
                                    index=0,
                                    delta=DeltaMessage(
                                        tool_calls=[
                                            ToolCall(
                                                id="",  # no id on delta chunks
                                                function=FunctionCall(
                                                    name="",
                                                    arguments=partial,
                                                ),
                                            )
                                        ]
                                    ),
                                )
                            ],
                        )

            elif event_type == "content_block_stop":
                current_block_type = None
                current_tool_id = None
                current_tool_name = None

            elif event_type == "message_delta":
                delta = payload.get("delta", {})
                stop_reason = delta.get("stop_reason")
                # Usage in message_delta: {"output_tokens": N}
                raw_usage = payload.get("usage", {})
                usage = None
                if raw_usage:
                    usage = Usage(
                        completion_tokens=raw_usage.get("output_tokens", 0),
                    )
                if stop_reason:
                    finish_reason = self._map_finish_reason(stop_reason)
                    yield StreamingChunk(
                        id=response_id,
                        model=model,
                        choices=[
                            StreamingChoice(
                                index=0,
                                delta=DeltaMessage(),
                                finish_reason=finish_reason,
                            )
                        ],
                        usage=usage,
                    )
