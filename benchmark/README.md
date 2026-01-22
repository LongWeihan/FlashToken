# FlashToken Benchmark

This directory provides **paper-style, reproducible evaluation**:

- correctness (token-by-token equality)
- performance (latency / throughput / “new work” reduction)
- automated plots and a human-readable summary

## One-click run (Windows)

From `FlashToken_EN/benchmark/`:

- `run_benchmark.cmd`

By default it runs the `standard` suite and writes to `benchmark/out/`:

- `results.json`: full machine-readable results (environment + all cases + metrics)
- `summary.md`: human-readable summary (tables + key highlights + figures)
- `*.png`: plots (speedup curves, work reduction, rollback distribution, etc.)

## Options

```bash
run_benchmark.cmd --suite standard --encoding cl100k_base --repeats 3
```

- `--suite`: `quick` / `standard` / `full`
- `--repeats`: repeats per performance case (median is reported)
- `--no-plot`: data only, skip figure generation

## Key metrics to look at

- `verify_mismatches`: number of token mismatches (expected to be 0)
- `speedup`: `baseline_ms / cached_ms` (higher is better)
- `cached_encoded_tokens`: how many tokens were actually newly encoded (lower is better)
- `rollback_tokens_max`: max rollback per append step (lower is better)
