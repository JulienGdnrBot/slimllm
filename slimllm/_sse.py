"""
SSE (Server-Sent Events) line parser.

Both OpenAI and Anthropic stream responses as SSE over HTTP.  The wire
format is simple:

    event: <event_type>     ← optional, Anthropic only
    data: <json_payload>
                            ← blank line = end of one event
    data: [DONE]            ← OpenAI sentinel for end-of-stream

This module extracts (event_type, data_str) tuples from a raw line
iterator (e.g. _http.stream_lines).
"""
from __future__ import annotations

from typing import Generator, Iterator, Optional, Tuple


def iter_events(
    lines: Iterator[str],
) -> Generator[Tuple[Optional[str], str], None, None]:
    """
    Parse SSE lines into (event_type, data) pairs.

    Yields one tuple per complete SSE event (i.e. after the blank-line
    separator).  The event_type is None when no "event:" field appeared.
    """
    event_type: Optional[str] = None
    data_parts: list[str] = []

    for line in lines:
        if line.startswith("event:"):
            event_type = line[len("event:"):].strip()

        elif line.startswith("data:"):
            data_parts.append(line[len("data:"):].strip())

        elif line == "":
            # Blank line → dispatch the accumulated event
            if data_parts:
                data = "\n".join(data_parts)
                yield event_type, data
            # Reset for next event
            event_type = None
            data_parts = []

        # Lines starting with ":" are SSE comments — ignore them.
        # Unknown fields are silently ignored per the SSE spec.

    # Flush any trailing event that wasn't terminated by a blank line
    # (some providers close the connection without a final newline)
    if data_parts:
        data = "\n".join(data_parts)
        yield event_type, data
