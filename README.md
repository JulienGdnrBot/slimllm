# slimllm

Zero-dependency Python facade for OpenAI, Anthropic, and OpenRouter — designed for AWS Lambda.

## Why

- **No external dependencies** — uses only Python stdlib (`http.client`, `ssl`, `json`, `asyncio`)
- **LiteLLM-compatible API** — same `completion()` call signature, swap without rewriting callers
- **Lambda-friendly** — tiny cold start, no bloat
- Supports streaming, tool use, and JSON mode

## Supported providers

| Model prefix | Provider | Env var |
|---|---|---|
| `gpt-*`, `o1-*`, `o3-*` | OpenAI | `OPENAI_API_KEY` |
| `claude-*`, `anthropic/…` | Anthropic | `ANTHROPIC_API_KEY` |
| `openrouter/…` | OpenRouter | `OPENROUTER_API_KEY` |

## Install

```bash
pip install slimllm
```

## Usage

```python
import slimllm

# Non-streaming
resp = slimllm.completion(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(resp.content)

# Streaming
stream = slimllm.completion("claude-3-5-sonnet-20241022", messages, stream=True)
for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="", flush=True)
final = stream.get_final_response()

# Async
resp = await slimllm.acompletion("gpt-4o", messages)

# Async streaming
async for chunk in slimllm.astream("claude-3-5-sonnet-20241022", messages):
    print(chunk.choices[0].delta.content or "", end="")

# Tool use (OpenAI format — auto-converted for Anthropic)
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}]
resp = slimllm.completion("claude-3-5-sonnet-20241022", messages, tools=tools)

# OpenRouter
resp = slimllm.completion("openrouter/meta-llama/llama-3.3-70b-instruct", messages)
```

## API key resolution

Keys are looked up in this order:
1. Explicit `api_key=` kwarg
2. Environment variable (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `OPENROUTER_API_KEY`)

## Response shape

All providers return the same OpenAI-shaped types:

```python
resp.content                          # str | None
resp.tool_calls                       # list[ToolCall] | None
resp.choices[0].finish_reason         # "stop" | "length" | "tool_calls"
resp.usage.prompt_tokens
resp.usage.completion_tokens
```

## License

MIT
