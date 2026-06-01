# CHANGELOG

<!-- version list -->

## v1.4.0 (2026-06-01)

### Bug Fixes

- **anthropic**: Preserve cache_control markers through message conversion
  ([`9e3f800`](https://github.com/JulienGdnrBot/slimllm/commit/9e3f8008f64fa14039f1854d9e6a3bc54f093d1c))

### Features

- **multimodal**: Provider-aware image emitter for OpenAI + Anthropic
  ([`8fe9dae`](https://github.com/JulienGdnrBot/slimllm/commit/8fe9dae11dd7ba5cd1ef6742bef8b1d40bb178aa))


## v1.3.1 (2026-04-12)

### Bug Fixes

- Preserve tool call index in streaming chunks for parallel tool calls
  ([`bf675ff`](https://github.com/JulienGdnrBot/slimllm/commit/bf675ffdb47f14a80d2af5cdc77d3ebae7c9b112))


## v1.3.0 (2026-04-05)

### Features

- **types**: Dataclass input layer for messages, tools, and stream objects
  ([`c43e42d`](https://github.com/JulienGdnrBot/slimllm/commit/c43e42d48748c1138c0913305fc0a753387cb8ea))


## v1.2.0 (2026-04-05)

### Features

- **providers**: Add Mistral, DeepSeek, Google AI Studio + retry with backoff
  ([`c8946bf`](https://github.com/JulienGdnrBot/slimllm/commit/c8946bf87f8b61a5f2981faf670fcef2719aa24c))


## v1.1.0 (2026-04-05)

### Features

- Add token_counter(), base_url alias, Anthropic param filtering
  ([`6381eba`](https://github.com/JulienGdnrBot/slimllm/commit/6381eba06328c4870442b3934f92851ed64236ac))


## v1.0.4 (2026-04-05)

### Bug Fixes

- Handle SSE streams closed without trailing blank line
  ([`363ba0c`](https://github.com/JulienGdnrBot/slimllm/commit/363ba0cb554960f199790f1cc5dc07c59ed410d0))

### Continuous Integration

- Switch PyPI publish to API token, disable attestations
  ([`4d30626`](https://github.com/JulienGdnrBot/slimllm/commit/4d30626083ac2dcc231d930363ac058f05c2b351))


## v1.0.3 (2026-04-05)

### Bug Fixes

- Pin ALPN to http/1.1 to avoid h2 negotiation on streaming endpoints
  ([`833fb20`](https://github.com/JulienGdnrBot/slimllm/commit/833fb20f2d22cc6acb85e8f7d2c02429b770262e))


## v1.0.2 (2026-04-05)

### Bug Fixes

- Read __version__ from package metadata instead of hardcoded string
  ([`1d0e1e4`](https://github.com/JulienGdnrBot/slimllm/commit/1d0e1e4eb7e8a9013f6e4953fb51c519fa2ba70e))


## v1.0.1 (2026-04-05)

### Bug Fixes

- Raise AuthenticationError instead of ValueError for missing API key
  ([`9facf54`](https://github.com/JulienGdnrBot/slimllm/commit/9facf54da3589a83699e34b1eb7d937fcc9a1410))


## v1.0.0 (2026-04-05)

- Initial Release
