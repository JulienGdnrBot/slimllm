"""Abstract base class every provider implements."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Generator, Iterator, List, Optional, Union

from ..types import (
    Message,
    ModelResponse,
    StreamingChunk,
    Tool,
)


class BaseProvider(ABC):

    @abstractmethod
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
        **kwargs: Any,
    ) -> Union[ModelResponse, Generator[StreamingChunk, None, None]]:
        ...
