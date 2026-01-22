# Integration Notes

FlashToken does not modify model weights and is framework-agnostic. You integrate it at the “text → token IDs” boundary.

## 1) Fixed prefix reuse (system prompt caching)

Typical server flow:

1. Create a cache once at startup: `cache = FixedPrefixTokenCache(enc, system_prompt)`
2. Per request: `tokens = cache.encode_ordinary(user_text)`
3. Feed `tokens` to your model

Good for:

- high-concurrency services with a shared system prompt
- IDE copilots with a long project context and short user deltas
- real-time voice agents with fixed instructions and streaming short utterances

## 2) Append-only chat (KV-cache aligned delta)

If your inference stack maintains a KV-cache and each turn only appends new text to the prompt, use the delta output:

- `delta = cache.append_ordinary(new_text)`
- rollback `delta.rollback_tokens`
- append `delta.tokens_to_append`

This avoids wasting work on re-tokenizing (and re-aligning) the entire growing history every turn.
