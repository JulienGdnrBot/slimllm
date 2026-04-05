"""
Response and request types — stdlib dataclasses only, no pydantic.
All response shapes mirror the OpenAI API so callers can treat every
provider the same way.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Literal, Optional


# ---------------------------------------------------------------------------
# Shared primitives
# ---------------------------------------------------------------------------

@dataclass
class FunctionCall:
    name: str
    arguments: str  # JSON-encoded string


@dataclass
class ToolCall:
    id: str
    type: Literal["function"] = "function"
    function: Optional[FunctionCall] = None


@dataclass
class Message:
    role: str
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    tool_call_id: Optional[str] = None  # for role="tool" messages
    name: Optional[str] = None


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


# ---------------------------------------------------------------------------
# Tool / function definitions (inputs)
# ---------------------------------------------------------------------------

@dataclass
class FunctionDefinition:
    name: str
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None


@dataclass
class Tool:
    function: FunctionDefinition
    type: Literal["function"] = "function"


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
        role = "assistant"
        finish_reason: Optional[str] = None
        model = ""
        response_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"

        # tool call accumulation: index → {id, name, arguments_parts}
        tool_acc: Dict[int, Dict[str, Any]] = {}

        for chunk in self._chunks:
            model = chunk.model or model
            response_id = chunk.id or response_id
            for choice in chunk.choices:
                delta = choice.delta
                if choice.finish_reason:
                    finish_reason = choice.finish_reason
                if delta.role:
                    role = delta.role
                if delta.content:
                    content_parts.append(delta.content)
                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        idx = 0  # streaming tools always index 0 for single call
                        if tc.id:
                            tool_acc[idx] = {
                                "id": tc.id,
                                "name": tc.function.name if tc.function else "",
                                "args": tc.function.arguments if tc.function else "",
                            }
                        elif idx in tool_acc and tc.function and tc.function.arguments:
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
        )
