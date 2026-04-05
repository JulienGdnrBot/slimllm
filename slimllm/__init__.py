"""
slimllm — zero-dependency LiteLLM-compatible facade.

Supported providers:
  - OpenAI          (model: "gpt-4o", "gpt-3.5-turbo", …)
  - Anthropic       (model: "claude-3-5-sonnet-20241022", …)
  - OpenRouter      (model: "openrouter/meta-llama/llama-3.3-70b-instruct", …)

Quick start::

    import slimllm

    resp = slimllm.completion(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hello!"}],
        api_key="sk-...",
    )
    print(resp.content)

    # Streaming
    stream = slimllm.completion(..., stream=True)
    for chunk in stream:
        print(chunk.choices[0].delta.content or "", end="")

    # Async
    resp = await slimllm.acompletion(...)
    async for chunk in slimllm.astream(...):
        ...
"""
from .exceptions import (
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    NotFoundError,
    PermissionDeniedError,
    ProviderError,
    RateLimitError,
    SlimLLMError,
    UnsupportedProviderError,
)
from .main import acompletion, astream, completion
from .types import (
    Choice,
    DeltaMessage,
    FunctionCall,
    FunctionDefinition,
    Message,
    ModelResponse,
    StreamingChunk,
    StreamingChoice,
    StreamResponse,
    Tool,
    ToolCall,
    Usage,
)

__version__ = "0.1.0"

__all__ = [
    # Core functions
    "completion",
    "acompletion",
    "astream",
    # Types
    "ModelResponse",
    "Choice",
    "Message",
    "Usage",
    "ToolCall",
    "FunctionCall",
    "Tool",
    "FunctionDefinition",
    "StreamingChunk",
    "StreamingChoice",
    "DeltaMessage",
    "StreamResponse",
    # Exceptions
    "SlimLLMError",
    "AuthenticationError",
    "BadRequestError",
    "PermissionDeniedError",
    "NotFoundError",
    "RateLimitError",
    "InternalServerError",
    "ProviderError",
    "UnsupportedProviderError",
]
