# CHANGELOG

<!-- version list -->

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
