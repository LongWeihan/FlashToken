# FlashToken

FlashToken is a **tokenizer-side prefix caching** library for low-latency LLM systems. It speeds up tokenization without changing model weights: when prompts share long prefixes (system prompts, templates, conversation history), FlashToken avoids re-tokenizing the same text over and over.

## Performance at a glance

- Correctness: `mismatches = 0` (token-by-token equality with `tiktoken.encode_ordinary`).
- Speed (median, Windows 10 / Python 3.12 / `cl100k_base`, standard suite, `repeats=3`):
  - Fixed prefix reuse (mixed, 2000 req): `1580.24 ms -> 57.53 ms` (`27.47x`).
  - Append-only chat (mixed, 400 turns): `2203.70 ms -> 58.51 ms` (`37.66x`).
- Details + raw outputs: see [Benchmark results](#benchmark-results-speed--correctness) and [`summary.md`](assets/benchmark/standard_win10_py312/summary.md) / [`results.json`](assets/benchmark/standard_win10_py312/results.json).


## Why it matters

* **AI Agents & ReAct loops**: One user command often triggers multiple "Think-Act-Observe" steps. Re-tokenizing massive tool outputs (JSON blobs, RAG results) at *every single step* creates compounding latency, making the agent feel slow and unresponsive.
* **Real-time AI voice calls**: "dead air" often comes from extra CPU work before the first token is generated. Re-tokenizing long prompts is a common hidden cost.
* **IDE copilots that feel half a beat late**: every completion/chat round may carry a long system prompt and project context; tokenizing from scratch adds latency and burns CPU.
* **Mobile / on-device chat that heats up over time**: longer histories mean more repeated tokenization per turn, increasing CPU time, battery drain, and thermal throttling.

FlashToken targets exactly these "long prefix reuse / append-only history" patterns.

## How it works (principle & architecture)

High level:

```text
text prompt -> FlashToken (tokenizer cache) -> token IDs -> LLM inference
             |-- FixedPrefixTokenCache       (prefix + suffix)
             `-- AppendOnlyPieceTokenCache   (append-only deltas)
```

- **FixedPrefixTokenCache**: uses `tiktoken.encode_with_unstable` to extract a *stable* token prefix once, then per request only encodes the tiny `(unstable_tail + suffix)` tail and concatenates.
- **AppendOnlyPieceTokenCache**: for append-only prompts, it rolls back and re-encodes only the last few regex "pieces" (`backtrack_pieces`), returning a delta that's KV-cache friendly.
- Integration point: plug it in at the `text -> token IDs` boundary (before model inference). No model changes required.
- More details: `docs/ALGORITHM.md` and `docs/INTEGRATION.md`.

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

## Benchmark results (speed + correctness)

All benchmark suites verify **token-by-token equality** with `tiktoken.encode_ordinary` (expected `mismatches == 0`).

### Standard suite highlights (Windows 10, Python 3.12, `cl100k_base`, `repeats=3`)

- Correctness: `mismatches = 0` across `english/chinese/mixed/code/markdown/json/emoji` domains (fixed-prefix and append-only, multiple `backtrack_pieces`).
- Performance (median):

| Scenario | Baseline | FlashToken | Speedup | Notes |
| --- | --- | --- | --- | --- |
| fixed_prefix (mixed, 2000 req) | 1580.24 ms | 57.53 ms | 27.47x | ~31x fewer encoded tokens |
| append_only (mixed, 400 turns) | 2203.70 ms | 58.51 ms | 37.66x | ~229x fewer encoded tokens; `rollback_max=78` |

Figures (generated by `benchmark/run.py`):

![](assets/benchmark/standard_win10_py312/fixed_prefix_speedup.png)
![](assets/benchmark/standard_win10_py312/append_only_speedup.png)
![](assets/benchmark/standard_win10_py312/append_only_work_series.png)
![](assets/benchmark/standard_win10_py312/append_only_rollback_hist.png)

Reproduce on your machine:

```bash
run_benchmark.cmd --suite standard --repeats 3
```

Raw outputs for the numbers above:

- [`assets/benchmark/standard_win10_py312/summary.md`](assets/benchmark/standard_win10_py312/summary.md)
- [`assets/benchmark/standard_win10_py312/results.json`](assets/benchmark/standard_win10_py312/results.json)

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

Inspired by the "prefix reuse / caching" philosophy popularized in KTransformers (applied here to the tokenizer stage).
