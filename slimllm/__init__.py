"""
slimllm — zero-dependency LiteLLM-compatible facade.

Supported providers:
  - OpenAI          (model: "gpt-4o", "gpt-4.1", …)
  - Anthropic       (model: "claude-3-5-sonnet-20241022", "claude-sonnet-4-6", …)
  - OpenRouter      (model: "openrouter/meta-llama/llama-4-maverick", …)
  - Mistral AI      (model: "mistral-large-2512", "mistral/mistral-medium", …)
  - DeepSeek        (model: "deepseek-chat", "deepseek/deepseek-r1", …)
  - Google AI Studio(model: "gemini-2.5-flash", "gemini/gemini-2.5-pro", …)

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
from .main import acompletion, astream, completion, token_counter
from .types import (
    Choice,
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

from importlib.metadata import version as _pkg_version, PackageNotFoundError as _PNF
try:
    __version__ = _pkg_version("slimllm")
except _PNF:
    __version__ = "0.0.0"  # fallback when running from source without install

__all__ = [
    # Core functions
    "completion",
    "acompletion",
    "astream",
    "token_counter",
    # Types
    "ModelResponse",
    "Choice",
    "Message",
    "ContentBlock",
    "Usage",
    "ToolCall",
    "FunctionCall",
    "Tool",
    "FunctionDefinition",
    "StreamingChunk",
    "StreamingChoice",
    "DeltaMessage",
    "StreamResponse",
    "RetryConfig",
    "ProviderConfig",
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
