"""
Response and request types — stdlib dataclasses only, no pydantic.

All response shapes mirror the OpenAI API so callers can treat every
provider the same way.  Input types (Message, Tool, ContentBlock) carry
to_dict() / from_dict() so they can be serialised to wire format without
manual dict construction.

Hierarchy
---------
Input (what you pass in)
  ContentBlock         — one block inside a multimodal message content list
  Message              — a single conversation turn (user / assistant / tool)
  FunctionDefinition   — describes a callable function
  Tool                 — wraps a FunctionDefinition

Output (what you get back)
  FunctionCall         — the function the model wants to call
  ToolCall             — wraps a FunctionCall with an id
  Choice               — one candidate in a batch response
  Usage                — token counts
  ModelResponse        — full batch response

Streaming output
  DeltaMessage         — incremental message delta (content / tool_calls / reasoning)
  StreamingChoice      — one choice in a streaming chunk
  StreamingChunk       — one SSE event worth of data
  StreamResponse       — iterable wrapper + get_final_response() aggregator

Config
  RetryConfig          — retry / back-off settings
  ProviderConfig       — provider-level call config
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Literal, Optional, Tuple, Union


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------

@dataclass
class RetryConfig:
    """
    Retry behaviour for HTTP calls.

    Fields
    ------
    max_retries : int
        Total number of retry attempts (0 = no retries, default 3).
    backoff_base : float
        Seconds to sleep before first retry; doubles each attempt (default 1.0).
    retryable_status_codes : tuple[int, ...]
        HTTP status codes that trigger a retry (default: 429, 500, 502, 503, 504).
    """
    max_retries: int = 3
    backoff_base: float = 1.0
    retryable_status_codes: Tuple[int, ...] = (429, 500, 502, 503, 504)


@dataclass
class ProviderConfig:
    """
    Runtime configuration for a provider call.

    Fields
    ------
    api_key : str
        Provider API key (required at call time).
    base_url : str | None
        Override the provider's default base URL (proxies, local models).
    extra_headers : dict | None
        Additional HTTP headers merged into every request.
    retry : RetryConfig | None
        Retry behaviour; uses module default when None.
    """
    api_key: str = ""
    base_url: Optional[str] = None
    extra_headers: Optional[Dict[str, str]] = None
    retry: Optional[RetryConfig] = None


# ---------------------------------------------------------------------------
# Input types — messages and tools
# ---------------------------------------------------------------------------

@dataclass
class ContentBlock:
    """
    A single block inside a multimodal message content list.

    OpenAI / Anthropic both support structured content arrays; this dataclass
    covers the common shapes.  Unknown fields can be shoved into *extra*.

    Common type values
    ------------------
    "text"          plain text block; set *text*
    "image_url"     image reference; set *image_url* = {"url": "https://..."}
    "tool_result"   result of a tool call (Anthropic); set *tool_use_id* + *content*
    "tool_use"      tool call block inside an assistant message (Anthropic)

    cache_control
    -------------
    Anthropic prompt-caching: set cache_control = {"type": "ephemeral"} on a
    block to mark it as a prompt-cache breakpoint.
    """
    type: str
    text: Optional[str] = None
    image_url: Optional[Dict[str, str]] = None
    tool_use_id: Optional[str] = None
    tool_call_id: Optional[str] = None  # OpenAI alias for tool_use_id
    content: Optional[Any] = None  # str or list for tool_result content
    cache_control: Optional[Dict[str, str]] = None
    extra: Optional[Dict[str, Any]] = None  # passthrough for unknown fields

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"type": self.type}
        if self.text is not None:
            d["text"] = self.text
        if self.image_url is not None:
            d["image_url"] = self.image_url
        if self.tool_use_id is not None:
            d["tool_use_id"] = self.tool_use_id
        if self.tool_call_id is not None:
            d["tool_call_id"] = self.tool_call_id
        if self.content is not None:
            d["content"] = self.content
        if self.cache_control is not None:
            d["cache_control"] = self.cache_control
        if self.extra:
            d.update(self.extra)
        return d

    @classmethod
    def text_block(cls, text: str, *, cache_control: Optional[Dict[str, str]] = None) -> "ContentBlock":
        return cls(type="text", text=text, cache_control=cache_control)

    @classmethod
    def image_block(cls, url: str, *, detail: Optional[str] = None) -> "ContentBlock":
        img: Dict[str, str] = {"url": url}
        if detail:
            img["detail"] = detail
        return cls(type="image_url", image_url=img)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ContentBlock":
        known = {"type", "text", "image_url", "tool_use_id", "tool_call_id",
                 "content", "cache_control"}
        extra = {k: v for k, v in d.items() if k not in known} or None
        return cls(
            type=d["type"],
            text=d.get("text"),
            image_url=d.get("image_url"),
            tool_use_id=d.get("tool_use_id"),
            tool_call_id=d.get("tool_call_id"),
            content=d.get("content"),
            cache_control=d.get("cache_control"),
            extra=extra,
        )


# Content can be a plain string or a list of blocks / raw dicts
ContentType = Optional[Union[str, List[Union[ContentBlock, Dict[str, Any]]]]]


@dataclass
class FunctionCall:
    """The function a model wants to call (output)."""
    name: str
    arguments: str  # JSON-encoded string

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "arguments": self.arguments}


@dataclass
class ToolCall:
    """A tool invocation inside an assistant message (output)."""
    id: str
    type: Literal["function"] = "function"
    function: Optional[FunctionCall] = None
    index: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"id": self.id, "type": self.type}
        if self.function is not None:
            d["function"] = self.function.to_dict()
        return d


@dataclass
class Message:
    """
    A single conversation turn.

    Roles
    -----
    "system"     instructions before the conversation
    "user"       human turn
    "assistant"  model turn (may have tool_calls instead of / alongside content)
    "tool"       result of a tool call; must carry tool_call_id

    Content
    -------
    content can be a plain string or a list of ContentBlock / raw dict blocks
    (e.g. for multimodal messages or Anthropic prompt-caching breakpoints).
    """
    role: str
    content: ContentType = None
    tool_calls: Optional[List[ToolCall]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Produce an OpenAI-format message dict suitable for HTTP bodies."""
        d: Dict[str, Any] = {"role": self.role}

        if isinstance(self.content, list):
            d["content"] = [
                b.to_dict() if isinstance(b, ContentBlock) else b
                for b in self.content
            ]
        elif self.content is not None:
            d["content"] = self.content

        if self.tool_calls:
            d["tool_calls"] = [tc.to_dict() for tc in self.tool_calls]
        if self.tool_call_id is not None:
            d["tool_call_id"] = self.tool_call_id
        if self.name is not None:
            d["name"] = self.name
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Message":
        """Reconstruct a Message from an OpenAI-format dict."""
        tool_calls: Optional[List[ToolCall]] = None
        raw_tcs = d.get("tool_calls")
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

        content = d.get("content")
        # If content is a list of dicts, keep as-is (provider handles translation)
        return cls(
            role=d["role"],
            content=content,
            tool_calls=tool_calls,
            tool_call_id=d.get("tool_call_id"),
            name=d.get("name"),
        )

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------

    @classmethod
    def user(cls, content: Union[str, List[Union[ContentBlock, Dict]]]) -> "Message":
        return cls(role="user", content=content)

    @classmethod
    def assistant(
        cls,
        content: Optional[str] = None,
        *,
        tool_calls: Optional[List[ToolCall]] = None,
    ) -> "Message":
        return cls(role="assistant", content=content, tool_calls=tool_calls)

    @classmethod
    def system(cls, content: str) -> "Message":
        return cls(role="system", content=content)

    @classmethod
    def tool_result(cls, tool_call_id: str, content: str) -> "Message":
        """Result of a tool invocation (OpenAI format: role=tool)."""
        return cls(role="tool", content=content, tool_call_id=tool_call_id)


@dataclass
class FunctionDefinition:
    """Describes a callable function (input for tool definitions)."""
    name: str
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None  # JSON Schema object

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"name": self.name}
        if self.description is not None:
            d["description"] = self.description
        if self.parameters is not None:
            d["parameters"] = self.parameters
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FunctionDefinition":
        return cls(
            name=d["name"],
            description=d.get("description"),
            parameters=d.get("parameters"),
        )


@dataclass
class Tool:
    """
    A tool definition passed to the model.

    Usage::

        tool = Tool(function=FunctionDefinition(
            name="get_weather",
            description="Return current weather for a city",
            parameters={
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        ))
        slimllm.completion(model="gpt-4o", messages=[...], tools=[tool])
    """
    function: FunctionDefinition
    type: Literal["function"] = "function"

    def to_dict(self) -> Dict[str, Any]:
        return {"type": self.type, "function": self.function.to_dict()}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Tool":
        return cls(
            type=d.get("type", "function"),
            function=FunctionDefinition.from_dict(d.get("function", {})),
        )

    @classmethod
    def from_function(
        cls,
        name: str,
        description: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> "Tool":
        """Convenience constructor — avoids nesting FunctionDefinition manually."""
        return cls(function=FunctionDefinition(
            name=name, description=description, parameters=parameters,
        ))


# ---------------------------------------------------------------------------
# Non-streaming response
# ---------------------------------------------------------------------------

@dataclass
class Choice:
    index: int
    message: Message
    finish_reason: Optional[str] = None
    logprobs: None = None


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class ModelResponse:
    model: str
    choices: List[Choice]
    id: str = field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:24]}")
    created: int = field(default_factory=lambda: int(time.time()))
    object: str = "chat.completion"
    usage: Optional[Usage] = None

    # Convenience helpers
    @property
    def content(self) -> Optional[str]:
        """Shortcut: first choice text content."""
        if self.choices:
            return self.choices[0].message.content
        return None

    @property
    def tool_calls(self) -> Optional[List[ToolCall]]:
        if self.choices:
            return self.choices[0].message.tool_calls
        return None


# ---------------------------------------------------------------------------
# Streaming response
# ---------------------------------------------------------------------------

@dataclass
class DeltaMessage:
    role: Optional[str] = None
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    # Extended thinking / reasoning models (Anthropic, DeepSeek-R1, QwQ, etc.)
    reasoning_content: Optional[str] = None


@dataclass
class StreamingChoice:
    index: int
    delta: DeltaMessage
    finish_reason: Optional[str] = None
    logprobs: None = None


@dataclass
class StreamingChunk:
    model: str
    choices: List[StreamingChoice]
    id: str = field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:24]}")
    created: int = field(default_factory=lambda: int(time.time()))
    object: str = "chat.completion.chunk"
    # Populated on the final chunk by OpenAI when stream_options=include_usage,
    # and on the message_start / message_delta events for Anthropic.
    usage: Optional[Usage] = None


# ---------------------------------------------------------------------------
# Stream wrapper — makes the provider generator behave like an iterable
# that also exposes a final aggregated response via .get_final_response()
# ---------------------------------------------------------------------------

class StreamResponse:
    """
    Wraps a generator of StreamingChunk objects.

    Usage::

        stream = completion(..., stream=True)
        for chunk in stream:
            print(chunk.choices[0].delta.content or "", end="", flush=True)
        final = stream.get_final_response()
    """

    def __init__(self, generator: Iterator[StreamingChunk]) -> None:
        self._gen = generator
        self._chunks: List[StreamingChunk] = []
        self._exhausted = False

    def __iter__(self) -> Iterator[StreamingChunk]:
        for chunk in self._gen:
            self._chunks.append(chunk)
            yield chunk
        self._exhausted = True

    def get_final_response(self) -> ModelResponse:
        """Aggregate collected chunks into a ModelResponse."""
        if not self._exhausted:
            # Drain remaining chunks
            for _ in self:
                pass

        content_parts: List[str] = []
        reasoning_parts: List[str] = []
        role = "assistant"
        finish_reason: Optional[str] = None
        model = ""
        response_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        final_usage: Optional[Usage] = None

        # tool call accumulation: index → {id, name, arguments_parts}
        tool_acc: Dict[int, Dict[str, Any]] = {}

        for chunk in self._chunks:
            model = chunk.model or model
            response_id = chunk.id or response_id
            if chunk.usage:
                final_usage = chunk.usage
            for choice in chunk.choices:
                delta = choice.delta
                if choice.finish_reason:
                    finish_reason = choice.finish_reason
                if delta.role:
                    role = delta.role
                if delta.content:
                    content_parts.append(delta.content)
                if delta.reasoning_content:
                    reasoning_parts.append(delta.reasoning_content)
                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        idx = tc.index if tc.index is not None else 0
                        if tc.id:
                            tool_acc[idx] = {
                                "id": tc.id,
                                "name": tc.function.name if tc.function else "",
                                "args": tc.function.arguments if tc.function else "",
                            }
                        else:
                            if idx not in tool_acc:
                                tool_acc[idx] = {"id": "", "name": "", "args": ""}
                            if tc.function:
                                if tc.function.name:
                                    tool_acc[idx]["name"] += tc.function.name
                                if tc.function.arguments:
                                    tool_acc[idx]["args"] += tc.function.arguments

        tool_calls = None
        if tool_acc:
            tool_calls = [
                ToolCall(
                    id=v["id"],
                    function=FunctionCall(name=v["name"], arguments=v["args"]),
                )
                for v in tool_acc.values()
            ]

        return ModelResponse(
            id=response_id,
            model=model,
            choices=[
                Choice(
                    index=0,
                    message=Message(
                        role=role,
                        content="".join(content_parts) or None,
                        tool_calls=tool_calls,
                    ),
                    finish_reason=finish_reason,
                )
            ],
            usage=final_usage,
        )
