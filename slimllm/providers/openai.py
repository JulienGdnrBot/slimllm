"""
OpenAI-compatible provider.

Handles both OpenAI (api.openai.com) and OpenRouter (openrouter.ai) since
they share the same /v1/chat/completions wire format.
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

# Default base URLs — can be overridden via extra_headers / api_base kwarg
OPENAI_BASE = "https://api.openai.com/v1"
OPENROUTER_BASE = "https://openrouter.ai/api/v1"


class OpenAIProvider(BaseProvider):
    """Handles OpenAI and any OpenAI-compatible endpoint (including OpenRouter)."""

    def __init__(self, base_url: str = OPENAI_BASE) -> None:
        self._base_url = base_url.rstrip("/")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

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
        url = f"{api_base.rstrip('/') if api_base else self._base_url}/chat/completions"
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
            response_format=response_format,
            **kwargs,
        )

        if stream:
            return self._stream(url, headers, body, model)
        return self._complete(url, headers, body)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_headers(
        self,
        api_key: str,
        extra: Optional[Dict[str, str]],
    ) -> Dict[str, str]:
        h = {"Authorization": f"Bearer {api_key}"}
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
        response_format: Optional[Dict[str, Any]],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": stream,
        }
        if tools:
            body["tools"] = tools
        if tool_choice is not None:
            body["tool_choice"] = tool_choice
        if temperature is not None:
            body["temperature"] = temperature
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if top_p is not None:
            body["top_p"] = top_p
        if stop is not None:
            body["stop"] = stop
        if response_format is not None:
            body["response_format"] = response_format
        # Pass through any unknown kwargs (e.g. seed, logprobs, etc.)
        body.update(kwargs)
        return body

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
            raise_for_status(status, data, "openai")
        if status >= 400:
            msg = data.get("error", {}).get("message", str(data)) if isinstance(data, dict) else str(data)
            raise_for_status(status, msg, "openai")

        return self._parse_response(data)

    def _parse_response(self, data: Dict[str, Any]) -> ModelResponse:
        choices = []
        for raw in data.get("choices", []):
            msg = raw.get("message", {})
            tool_calls = None
            raw_tcs = msg.get("tool_calls")
            if raw_tcs:
                tool_calls = [
                    ToolCall(
                        id=tc["id"],
                        function=FunctionCall(
                            name=tc["function"]["name"],
                            arguments=tc["function"]["arguments"],
                        ),
                    )
                    for tc in raw_tcs
                ]
            choices.append(
                Choice(
                    index=raw.get("index", 0),
                    message=Message(
                        role=msg.get("role", "assistant"),
                        content=msg.get("content"),
                        tool_calls=tool_calls,
                    ),
                    finish_reason=raw.get("finish_reason"),
                )
            )

        raw_usage = data.get("usage", {})
        usage = Usage(
            prompt_tokens=raw_usage.get("prompt_tokens", 0),
            completion_tokens=raw_usage.get("completion_tokens", 0),
            total_tokens=raw_usage.get("total_tokens", 0),
        ) if raw_usage else None

        return ModelResponse(
            id=data.get("id", f"chatcmpl-{uuid.uuid4().hex[:24]}"),
            model=data.get("model", ""),
            choices=choices,
            usage=usage,
            created=data.get("created", 0),
        )

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
        lines = stream_lines(url, headers, body)

        for _event_type, data in iter_events(lines):
            if data == "[DONE]":
                return

            try:
                chunk = json.loads(data)
            except json.JSONDecodeError as exc:
                raise ProviderError(f"Invalid SSE JSON: {data!r}", provider="openai") from exc

            yield self._parse_chunk(chunk, model)

    def _parse_chunk(
        self,
        chunk: Dict[str, Any],
        fallback_model: str,
    ) -> StreamingChunk:
        choices = []
        for raw in chunk.get("choices", []):
            delta_raw = raw.get("delta", {})
            tool_calls = None
            raw_tcs = delta_raw.get("tool_calls")
            if raw_tcs:
                tool_calls = []
                for tc in raw_tcs:
                    fn = tc.get("function", {})
                    tool_calls.append(
                        ToolCall(
                            id=tc.get("id", ""),
                            function=FunctionCall(
                                name=fn.get("name", ""),
                                arguments=fn.get("arguments", ""),
                            ),
                        )
                    )
            choices.append(
                StreamingChoice(
                    index=raw.get("index", 0),
                    delta=DeltaMessage(
                        role=delta_raw.get("role"),
                        content=delta_raw.get("content"),
                        tool_calls=tool_calls,
                    ),
                    finish_reason=raw.get("finish_reason"),
                )
            )

        return StreamingChunk(
            id=chunk.get("id", ""),
            model=chunk.get("model", fallback_model),
            choices=choices,
            created=chunk.get("created", 0),
        )


class OpenRouterProvider(OpenAIProvider):
    """OpenRouter — identical wire format to OpenAI, different base URL."""

    def __init__(self) -> None:
        super().__init__(base_url=OPENROUTER_BASE)

    def _build_headers(
        self,
        api_key: str,
        extra: Optional[Dict[str, str]],
    ) -> Dict[str, str]:
        h = super()._build_headers(api_key, extra)
        # OpenRouter requires these for rate-limit attribution
        h.setdefault("HTTP-Referer", "https://github.com/slimllm")
        h.setdefault("X-Title", "slimllm")
        return h
