# Benchmark Methodology (Academic-Style)

This document describes the experiment design and metrics used by `benchmark/run.py`, enabling reproducible evaluation and paper-quality reporting.

## 1) Research question

FlashToken targets a practical, measurable question:

> In typical LLM applications with long reusable prefixes or append-only chat histories, does tokenization contain significant redundant work?  
> Can caching and incremental tokenization reduce latency and CPU work **while preserving exact token IDs**?

## 2) Baseline & compared methods

### Baseline (common implementation)

- For each request / each chat turn: run `tiktoken.encode_ordinary(full_text)` on the **entire** prompt from scratch.

### FlashToken methods

- `FixedPrefixTokenCache`: cache the stable token prefix of a fixed prompt; per request only tokenize `(unstable_tail + suffix)`.
- `AppendOnlyPieceTokenCache`: rollback and re-encode only the last few regex “pieces”; returns `(rollback_tokens, tokens_to_append)`.

Correctness is defined as **token-by-token equality** with the baseline encoder.

## 3) Metrics

### Correctness

- `verify_mismatches`: number of token sequence mismatches (expected to be **0**)
- `verify_examples`: a few mismatch examples for debugging (not used for timing)

### Performance

Each performance case is repeated `repeats` times; we report the **median**:

- `baseline_ms_median` / `cached_ms_median`
- `speedup = baseline_ms_median / cached_ms_median`

### Work reduction (key explanatory variable)

Tokenizer “work” can be approximated by the number of **newly encoded tokens**:

- baseline: re-encodes the full prompt, so `baseline_encoded_tokens == output_tokens`
- FlashToken:
  - fixed_prefix: `cached_encoded_tokens ≈ output_tokens - stable_prefix_tokens * requests`
  - append_only: `cached_encoded_tokens == Σ len(tokens_to_append)`

We also report:

- `baseline_encoded_tok_per_s` / `cached_encoded_tok_per_s`
- `rollback_tokens_max`, `rollback_tokens_total` (important for append-only behavior)

## 4) Workloads

To cover realistic text distributions, the benchmark generates multiple domains:

- `english` / `chinese` / `mixed`
- `code` / `markdown` / `json`
- `emoji`

And evaluates two core patterns:

1) **fixed_prefix**: `prefix + suffix_i` (models long system prompt reuse)
2) **append_only**: initial text + many appends (models “chat history keeps growing”)

## 5) Visualization

The benchmark generates PNG figures:

- `fixed_prefix_speedup.png`: speedup vs prefix length
- `append_only_speedup.png`: speedup vs number of turns
- `append_only_work_series.png`: why baseline gets “hotter” over time (more tokens per turn)
- `append_only_rollback_hist.png`: rollback token distribution

## 6) Reproducibility

`benchmark/out/results.json` records:

- Python / OS / CPU core count
- `tiktoken` / `regex` versions
- suite name and random seed

Include this JSON when sharing results.
