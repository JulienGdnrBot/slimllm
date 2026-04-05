"""
Top-level completion() and acompletion() functions.

Provider routing rules (applied in order):
  1. model starts with "openrouter/"      → OpenRouter
  2. model starts with "claude-"          → Anthropic
  3. model starts with "anthropic/"       → Anthropic (strip prefix)
  4. everything else                       → OpenAI

API key resolution order for each provider:
  1. Explicit `api_key` kwarg
  2. Environment variable (OPENAI_API_KEY / ANTHROPIC_API_KEY / OPENROUTER_API_KEY)
"""
from __future__ import annotations

import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Generator, List, Optional, Union

from .exceptions import AuthenticationError, UnsupportedProviderError
from .providers.anthropic import AnthropicProvider
from .providers.openai import OpenAIProvider, OpenRouterProvider
from .types import ModelResponse, StreamingChunk, StreamResponse


def token_counter(model: str, text: str) -> int:
    """
    Estimate the number of tokens in *text* for the given *model*.

    Uses tiktoken when installed (accurate); falls back to a
    character-based approximation (~4 chars per token) otherwise.
    The fallback is intentionally conservative and works well enough
    for context-window monitoring and cost estimation.
    """
    if not text:
        return 0
    try:
        import tiktoken  # type: ignore[import]
        try:
            enc = tiktoken.encoding_for_model(model)
        except KeyError:
            enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except ImportError:
        return max(1, len(text) // 4)

# Module-level provider singletons (stateless, safe to share)
_openai = OpenAIProvider()
_anthropic = AnthropicProvider()
_openrouter = OpenRouterProvider()

# Thread pool for wrapping sync I/O in async calls
_executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="slimllm")


# ---------------------------------------------------------------------------
# Provider routing
# ---------------------------------------------------------------------------

def _route(model: str):
    """Return (provider_instance, resolved_model_name)."""
    if model.startswith("openrouter/"):
        return _openrouter, model[len("openrouter/"):]
    if model.startswith("anthropic/"):
        return _anthropic, model[len("anthropic/"):]
    if model.startswith("claude-"):
        return _anthropic, model
    # Default: OpenAI-compatible
    return _openai, model


def _resolve_api_key(provider, explicit_key: Optional[str]) -> str:
    if explicit_key:
        return explicit_key
    env_map = {
        "AnthropicProvider": "ANTHROPIC_API_KEY",
        "OpenRouterProvider": "OPENROUTER_API_KEY",
        "OpenAIProvider": "OPENAI_API_KEY",
    }
    env_var = env_map.get(type(provider).__name__, "OPENAI_API_KEY")
    key = os.environ.get(env_var, "")
    if not key:
        raise AuthenticationError(
            f"No API key provided. Set {env_var} or pass api_key= explicitly.",
            status_code=401,
            provider=type(provider).__name__,
        )
    return key


# ---------------------------------------------------------------------------
# Synchronous completion
# ---------------------------------------------------------------------------

def completion(
    model: str,
    messages: List[Dict[str, Any]],
    *,
    api_key: Optional[str] = None,
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
) -> Union[ModelResponse, StreamResponse]:
    """
    Synchronous chat completion — mirrors the litellm.completion() signature.

    Parameters
    ----------
    model : str
        Model identifier.  Prefix with ``openrouter/`` or ``anthropic/`` to
        force a specific provider.  ``claude-*`` models are routed to
        Anthropic automatically.
    messages : list[dict]
        OpenAI-format message list.
    api_key : str, optional
        Provider API key.  Falls back to environment variable.
    stream : bool
        When True, returns a :class:`StreamResponse` you can iterate over.
    tools : list[dict], optional
        OpenAI-format tool definitions.
    tool_choice : str | dict, optional
        "auto", "none", "required", or {"type":"function","function":{"name":"…"}}
    temperature, max_tokens, top_p, stop, response_format
        Standard generation parameters.
    extra_headers : dict, optional
        Additional HTTP headers merged into the request.
    api_base : str, optional
        Override the provider base URL (useful for proxies / local models).

    Returns
    -------
    ModelResponse
        When ``stream=False``.
    StreamResponse
        When ``stream=True`` — iterate it to get :class:`StreamingChunk` objects,
        then call ``.get_final_response()`` for a consolidated :class:`ModelResponse`.
    """
    provider, resolved_model = _route(model)
    key = _resolve_api_key(provider, api_key)

    # Accept base_url as an alias for api_base (litellm compatibility)
    if api_base is None and "base_url" in kwargs:
        api_base = kwargs.pop("base_url")

    result = provider.completion(
        model=resolved_model,
        messages=messages,
        api_key=key,
        stream=stream,
        tools=tools,
        tool_choice=tool_choice,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        stop=stop,
        response_format=response_format,
        extra_headers=extra_headers,
        api_base=api_base,
        **kwargs,
    )

    if stream:
        # Wrap the raw generator so callers can also call .get_final_response()
        return StreamResponse(result)  # type: ignore[arg-type]
    return result  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Async completion (non-streaming and streaming)
# ---------------------------------------------------------------------------

async def acompletion(
    model: str,
    messages: List[Dict[str, Any]],
    *,
    api_key: Optional[str] = None,
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
) -> Union[ModelResponse, StreamResponse]:
    """
    Async wrapper around :func:`completion`.

    Uses a thread executor to avoid blocking the event loop during I/O.
    For streaming, the generator is consumed in the background thread and
    the resulting :class:`StreamResponse` is returned when fully built —
    use :func:`astream` instead if you need token-by-token async iteration.
    """
    # Accept base_url as an alias for api_base (litellm compatibility)
    if api_base is None and "base_url" in kwargs:
        api_base = kwargs.pop("base_url")

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        _executor,
        lambda: completion(
            model,
            messages,
            api_key=api_key,
            stream=stream,
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stop=stop,
            response_format=response_format,
            extra_headers=extra_headers,
            api_base=api_base,
            **kwargs,
        ),
    )


async def astream(
    model: str,
    messages: List[Dict[str, Any]],
    *,
    api_key: Optional[str] = None,
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
):
    """
    True async generator — yields :class:`StreamingChunk` objects one by one
    without blocking the event loop between chunks.

    Usage::

        async for chunk in astream("gpt-4o", messages):
            print(chunk.choices[0].delta.content or "", end="")
    """
    provider, resolved_model = _route(model)
    key = _resolve_api_key(provider, api_key)

    # Accept base_url as an alias for api_base (litellm compatibility)
    if api_base is None and "base_url" in kwargs:
        api_base = kwargs.pop("base_url")

    loop = asyncio.get_event_loop()

    # Build a synchronous generator in a thread and pull chunks across
    # via an async queue so we don't block the event loop.
    queue: asyncio.Queue = asyncio.Queue(maxsize=64)
    _SENTINEL = object()

    def _produce() -> None:
        try:
            gen = provider.completion(
                model=resolved_model,
                messages=messages,
                api_key=key,
                stream=True,
                tools=tools,
                tool_choice=tool_choice,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                stop=stop,
                response_format=response_format,
                extra_headers=extra_headers,
                api_base=api_base,
                **kwargs,
            )
            for chunk in gen:  # type: ignore[union-attr]
                asyncio.run_coroutine_threadsafe(queue.put(chunk), loop).result()
        except Exception as exc:
            asyncio.run_coroutine_threadsafe(queue.put(exc), loop).result()
        finally:
            asyncio.run_coroutine_threadsafe(queue.put(_SENTINEL), loop).result()

    _executor.submit(_produce)

    while True:
        item = await queue.get()
        if item is _SENTINEL:
            return
        if isinstance(item, Exception):
            raise item
        yield item
