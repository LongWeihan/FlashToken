# Integration Notes

FlashToken does not modify model weights and is framework-agnostic. Integrate it at the "text -> token IDs" boundary.

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

If your inference stack maintains a KV-cache and each turn only appends new text to the prompt:

1. Keep a cache: `cache = AppendOnlyPieceTokenCache(enc, initial_text, backtrack_pieces=2)`
2. For each appended chunk:
   - `delta = cache.append_ordinary(new_text)`
   - rollback `delta.rollback_tokens`
   - append `delta.tokens_to_append`

If you do not keep a KV-cache, you can simply use `cache.tokens` as the full token sequence after each append.

