# FlashToken Algorithms

FlashToken speeds up LLM tokenization by reusing work when prompts share long prefixes or when a chat history grows by appending new text.

## 1) FixedPrefixTokenCache (fixed prefix reuse)

Use case: a fixed system prompt/template `P` reused across requests, with a changing suffix `S`.

Idea:

- Use `tiktoken.encode_with_unstable(P)` to split `P` into:
  - `T_stable`: a token prefix that is guaranteed not to change no matter what gets appended
  - `P_tail`: a short "unstable tail" text segment
- For each request, only encode the tail:
  - `tokens = T_stable + encode_ordinary(P_tail + S)`

Properties:

- Exactness: `tokens == encode_ordinary(P + S)`
- Best when `P` is long and reused many times

## 2) AppendOnlyPieceTokenCache (append-only, piece rollback)

Use case: the prompt grows by appending deltas (typical multi-turn chat).

tiktoken-style tokenization can be simplified as:

1) regex pre-tokenization: split text into "pieces"
2) per-piece BPE encoding: encode each piece independently into tokens

Key observation:

- When you append new text, usually only the last few regex pieces can change.

So FlashToken:

- keeps the existing piece boundaries and per-piece tokens
- on append, rolls back the last `backtrack_pieces` pieces and re-tokenizes only the tail
- returns a delta: `(rollback_tokens, tokens_to_append)` so you can update a KV-cache by rolling back/appending the same amount

Parameter guidance:

- `backtrack_pieces` trades safety vs speed.
- Use the correctness suite in `benchmark/run.py` and pick the smallest value that yields `mismatches == 0` for your text distribution.

