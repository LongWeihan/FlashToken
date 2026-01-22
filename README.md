# FlashToken

FlashToken is a **tokenizer-side prefix caching** library for low-latency LLM systems. It speeds up tokenization without changing model weights: when prompts share long prefixes (system prompts, templates, conversation history), FlashToken avoids re-tokenizing the same text over and over.

## Why it matters (real-world pain points)

- **Real-time AI voice calls**: “dead air” often comes from extra CPU work before the first token is generated. Re-tokenizing long prompts is a common hidden cost.
- **IDE copilots that feel half a beat late**: every completion/chat round may carry a long system prompt and project context; tokenizing from scratch adds latency and burns CPU.
- **Mobile / on-device chat that heats up over time**: longer histories mean more repeated tokenization per turn, increasing CPU time, battery drain, and thermal throttling.

FlashToken targets exactly these “long prefix reuse / append-only history” patterns.

## What you get

FlashToken provides two strategies. Both preserve **exact token IDs** (aligned with `tiktoken.encode_ordinary`) and can be benchmarked without loading any LLM:

1) `FixedPrefixTokenCache`  
   Best for a fixed system prompt / template reused across many requests.

2) `AppendOnlyPieceTokenCache` (piece rollback)  
   Best for append-only conversation growth; returns `(rollback_tokens, tokens_to_append)` to align with KV-cache workflows.

## Install

```bash
pip install -e .
```

## Quickstart

### Fixed prefix reuse

```python
import tiktoken
from flashtoken import FixedPrefixTokenCache

enc = tiktoken.get_encoding("cl100k_base")
cache = FixedPrefixTokenCache(enc, prefix="SYSTEM: ... long ...\n")

tokens = cache.encode_ordinary("User: hello\n")
```

### Append-only delta tokenization

```python
import tiktoken
from flashtoken import AppendOnlyPieceTokenCache

enc = tiktoken.get_encoding("cl100k_base")
cache = AppendOnlyPieceTokenCache(enc, initial_text="SYSTEM: ...\n", backtrack_pieces=2)

delta = cache.append_ordinary("\nUser: hi\nAssistant: hello\n")
# delta.rollback_tokens: how many tokens to rollback
# delta.tokens_to_append: new tokens to append
```

## Benchmark (correctness + performance + plots)

Windows one-click:

- `run_benchmark.cmd` (same as `benchmark\\run_benchmark.cmd`)

It will:

- create a local venv and install dependencies
- run a token-by-token correctness suite (expects `mismatches == 0`)
- run performance benchmarks (multiple workloads + parameter sweeps)
- write `benchmark/out/results.json`, `benchmark/out/summary.md`, and PNG figures

See: `benchmark/README.md`.

## Docs

- Algorithms: `docs/ALGORITHM.md`
- Integration notes: `docs/INTEGRATION.md`
- Benchmark methodology: `benchmark/METHODOLOGY.md`

## Project layout

- `flashtoken/`: library code (pure algorithm + API)
- `benchmark/`: correctness + performance + visualization
- `docs/`: algorithm & integration notes

## Limitations

- `AppendOnlyPieceTokenCache` uses some tiktoken internals (`_pat_str`, `_encode_single_piece`) for speed and exactness; a major tiktoken change may require small updates.
- If your prompts are short or do not reuse prefixes, the speedup will be small.

## Acknowledgements

Inspired by the “prefix reuse / caching” philosophy popularized in KTransformers (applied here to the tokenizer stage).
