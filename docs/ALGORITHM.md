# FlashToken Algorithms

FlashToken focuses on **prefix reuse at the tokenizer stage**. This is often overlooked, but in long-prompt / long-conversation applications it can noticeably impact first-token latency and CPU utilization.

## A) `FixedPrefixTokenCache` (fixed-prefix reuse)

**Use case**: a fixed system prompt / tool instructions / template, with a changing user suffix.

Core idea:

- Split the fixed prefix `P` into:
  - `P_stable_tokens`: a token prefix that is guaranteed to stay unchanged no matter what text is appended later
  - `P_unstable_text`: the remaining “unstable tail” (usually very short)
- For each request suffix `S`:
  - output tokens = `P_stable_tokens + encode(P_unstable_text + S)`

Why it’s fast:

- The long prefix is processed once.
- Each request only encodes “tail + new input”, dramatically reducing repeated work.

## B) `AppendOnlyPieceTokenCache` (append-only, piece rollback)

**Use case**: conversation history is append-only (each turn only appends text to the end).

Tokenizer (tiktoken-style) can be simplified as:

1) regex pre-tokenization: `text -> [piece_0, piece_1, ...]`
2) per-piece BPE: `piece_i -> tokens_i`

Key observation:

- Appending new text usually only affects the segmentation of the last 1~few pieces.
- Therefore we only rollback and re-encode the last `k` pieces (`k = backtrack_pieces`).

Output delta:

- `rollback_tokens`: how many tokens from the end need to be removed
- `tokens_to_append`: new tokens to append

Why this is useful for inference:

- If your generation stack keeps a KV-cache aligned to tokens, you only rollback/append a tiny tail instead of re-tokenizing (and potentially re-aligning) the full prompt each turn.

## Parameter guidance

- `backtrack_pieces=2` is usually a strong default for mixed text distributions.
- If you expect extreme boundary cases (lots of punctuation / special patterns), use the correctness suite in `benchmark/run.py` and pick the smallest `backtrack_pieces` that yields `mismatches == 0`.
