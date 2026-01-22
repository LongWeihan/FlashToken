from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import random
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import tiktoken

from flashtoken import AppendOnlyPieceTokenCache, FixedPrefixTokenCache

from workloads import TextDomain, make_chat_deltas, make_suffixes, make_text

SuiteName = Literal["quick", "standard", "full"]


def _time_ms(fn) -> float:
    t0 = time.perf_counter()
    fn()
    t1 = time.perf_counter()
    return (t1 - t0) * 1000.0


def _median(xs: List[float]) -> float:
    return float(statistics.median(xs)) if xs else 0.0


def _tok_per_s(tokens: int, ms: float) -> float:
    if ms <= 0:
        return float("inf")
    return tokens / (ms / 1000.0)


def _pkg_version(name: str) -> str:
    try:
        import importlib.metadata

        return importlib.metadata.version(name)
    except Exception:
        return "unknown"


@dataclass(frozen=True)
class EnvInfo:
    python: str
    platform: str
    cpu_count: int
    tiktoken: str
    regex: str
    encoding: str


def collect_env(encoding: str) -> EnvInfo:
    return EnvInfo(
        python=sys.version.replace("\n", " "),
        platform=platform.platform(),
        cpu_count=os.cpu_count() or 0,
        tiktoken=_pkg_version("tiktoken"),
        regex=_pkg_version("regex"),
        encoding=encoding,
    )


def verify_fixed_prefix_case(
    enc: tiktoken.Encoding,
    *,
    domain: TextDomain,
    prefix_chars: int,
    suffix_chars: int,
    n: int,
    seed: int,
) -> Dict[str, Any]:
    prefix = make_text(prefix_chars, seed=seed, domain=domain, header="SYSTEM:")
    suffixes = make_suffixes(n, suffix_chars, seed=seed + 1, domain=domain)
    cache = FixedPrefixTokenCache(enc, prefix)

    mismatches = 0
    examples: List[Dict[str, Any]] = []
    for i, suf in enumerate(suffixes):
        a = enc.encode_ordinary(prefix + suf)
        b = cache.encode_ordinary(suf)
        if a != b:
            mismatches += 1
            if len(examples) < 5:
                examples.append(
                    {
                        "i": i,
                        "prefix_chars": prefix_chars,
                        "suffix_chars": suffix_chars,
                        "len_a": len(a),
                        "len_b": len(b),
                        "suffix_preview": suf[:120],
                    }
                )

    return {
        "case": {
            "domain": domain,
            "prefix_chars": prefix_chars,
            "suffix_chars": suffix_chars,
            "n": n,
            "seed": seed,
        },
        "verify_mismatches": mismatches,
        "verify_examples": examples,
        "stable_prefix_tokens": cache.stable_prefix_token_count,
        "unstable_prefix_chars": cache.unstable_prefix_char_count,
    }


def verify_append_only_case(
    enc: tiktoken.Encoding,
    *,
    domain: TextDomain,
    initial_chars: int,
    turns: int,
    chars_per_turn: int,
    backtrack_pieces: int,
    seed: int,
) -> Dict[str, Any]:
    initial = make_text(initial_chars, seed=seed, domain=domain, header="SYSTEM:")
    deltas = make_chat_deltas(turns, chars_per_turn, seed=seed + 1, domain=domain)
    cache = AppendOnlyPieceTokenCache(enc, initial, backtrack_pieces=backtrack_pieces)

    text = initial
    mismatches = 0
    examples: List[Dict[str, Any]] = []
    rollback_max = 0
    rollback_total = 0

    for i, d in enumerate(deltas):
        text += d
        delta = cache.append_ordinary(d)
        rollback_total += delta.rollback_tokens
        rollback_max = max(rollback_max, delta.rollback_tokens)

        baseline = enc.encode_ordinary(text)
        if baseline != cache.tokens:
            mismatches += 1
            if len(examples) < 3:
                examples.append(
                    {
                        "turn": i,
                        "rollback_tokens": delta.rollback_tokens,
                        "new_tokens": len(delta.tokens_to_append),
                        "baseline_len": len(baseline),
                        "cached_len": len(cache.tokens),
                        "delta_preview": d[:120],
                    }
                )

    return {
        "case": {
            "domain": domain,
            "initial_chars": initial_chars,
            "turns": turns,
            "chars_per_turn": chars_per_turn,
            "backtrack_pieces": backtrack_pieces,
            "seed": seed,
        },
        "verify_mismatches": mismatches,
        "verify_examples": examples,
        "rollback_tokens_total": rollback_total,
        "rollback_tokens_max": rollback_max,
    }


def perf_fixed_prefix_case(
    enc: tiktoken.Encoding,
    *,
    domain: TextDomain,
    prefix_chars: int,
    suffix_chars: int,
    n: int,
    seed: int,
    repeats: int,
    tag: str,
) -> Dict[str, Any]:
    prefix = make_text(prefix_chars, seed=seed, domain=domain, header="SYSTEM:")
    suffixes = make_suffixes(n, suffix_chars, seed=seed + 1, domain=domain)

    output_tokens = 0
    baseline_encoded_tokens = 0
    cached_encoded_tokens = 0

    def baseline_once() -> None:
        nonlocal output_tokens, baseline_encoded_tokens
        output_tokens = 0
        baseline_encoded_tokens = 0
        for suf in suffixes:
            n_tok = len(enc.encode_ordinary(prefix + suf))
            output_tokens += n_tok
            baseline_encoded_tokens += n_tok

    baseline_once()
    baseline_ms: List[float] = []
    for _ in range(repeats):
        baseline_ms.append(_time_ms(baseline_once))

    cache_build_ms = _time_ms(lambda: FixedPrefixTokenCache(enc, prefix))
    cache = FixedPrefixTokenCache(enc, prefix)
    stable_n = cache.stable_prefix_token_count

    def cached_once() -> None:
        nonlocal output_tokens, cached_encoded_tokens
        output_tokens = 0
        cached_encoded_tokens = 0
        for suf in suffixes:
            toks = cache.encode_ordinary(suf)
            output_tokens += len(toks)
            cached_encoded_tokens += max(0, len(toks) - stable_n)

    cached_once()
    cached_ms: List[float] = []
    for _ in range(repeats):
        cached_ms.append(_time_ms(cached_once))

    b_med = _median(baseline_ms)
    c_med = _median(cached_ms)
    return {
        "kind": "fixed_prefix",
        "tag": tag,
        "case": {
            "domain": domain,
            "prefix_chars": prefix_chars,
            "suffix_chars": suffix_chars,
            "n": n,
            "seed": seed,
        },
        "repeats": repeats,
        "baseline_ms_median": b_med,
        "cached_ms_median": c_med,
        "speedup": (b_med / c_med) if c_med > 0 else float("inf"),
        "baseline_output_tok_per_s": _tok_per_s(output_tokens, b_med),
        "cached_output_tok_per_s": _tok_per_s(output_tokens, c_med),
        "baseline_encoded_tok_per_s": _tok_per_s(baseline_encoded_tokens, b_med),
        "cached_encoded_tok_per_s": _tok_per_s(cached_encoded_tokens, c_med),
        "output_tokens": output_tokens,
        "baseline_encoded_tokens": baseline_encoded_tokens,
        "cached_encoded_tokens": cached_encoded_tokens,
        "cache_build_ms": cache_build_ms,
        "stable_prefix_tokens": stable_n,
        "unstable_prefix_chars": cache.unstable_prefix_char_count,
    }


def perf_append_only_case(
    enc: tiktoken.Encoding,
    *,
    domain: TextDomain,
    initial_chars: int,
    turns: int,
    chars_per_turn: int,
    backtrack_pieces: int,
    seed: int,
    repeats: int,
    tag: str,
    capture_series: bool,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    initial = make_text(initial_chars, seed=seed, domain=domain, header="SYSTEM:")
    deltas = make_chat_deltas(turns, chars_per_turn, seed=seed + 1, domain=domain)

    def baseline_once() -> Tuple[int, int]:
        text = initial
        output_total = 0
        encoded_total = 0
        for d in deltas:
            text += d
            n_tok = len(enc.encode_ordinary(text))
            output_total += n_tok
            encoded_total += n_tok
        return output_total, encoded_total

    output_tokens, baseline_encoded_tokens = baseline_once()
    baseline_ms: List[float] = []
    for _ in range(repeats):
        def _run() -> None:
            nonlocal output_tokens, baseline_encoded_tokens
            output_tokens, baseline_encoded_tokens = baseline_once()

        baseline_ms.append(_time_ms(_run))

    cache_build_ms = _time_ms(lambda: AppendOnlyPieceTokenCache(enc, initial, backtrack_pieces=backtrack_pieces))

    cached_encoded_tokens = 0
    rollback_tokens_total = 0
    rollback_tokens_max = 0

    def cached_once() -> Tuple[int, int]:
        cache = AppendOnlyPieceTokenCache(enc, initial, backtrack_pieces=backtrack_pieces)
        output_total = 0
        encoded_total = 0
        rollback_total = 0
        rollback_max = 0
        for d in deltas:
            delta = cache.append_ordinary(d)
            rollback_total += delta.rollback_tokens
            rollback_max = max(rollback_max, delta.rollback_tokens)
            encoded_total += len(delta.tokens_to_append)
            output_total += len(cache.tokens)

        nonlocal rollback_tokens_total, rollback_tokens_max
        rollback_tokens_total = rollback_total
        rollback_tokens_max = rollback_max
        return output_total, encoded_total

    output_tokens, cached_encoded_tokens = cached_once()
    cached_ms: List[float] = []
    for _ in range(repeats):
        def _run() -> None:
            nonlocal output_tokens, cached_encoded_tokens
            output_tokens, cached_encoded_tokens = cached_once()

        cached_ms.append(_time_ms(_run))

    b_med = _median(baseline_ms)
    c_med = _median(cached_ms)

    case_result = {
        "kind": "append_only_piece",
        "tag": tag,
        "case": {
            "domain": domain,
            "initial_chars": initial_chars,
            "turns": turns,
            "chars_per_turn": chars_per_turn,
            "backtrack_pieces": backtrack_pieces,
            "seed": seed,
        },
        "repeats": repeats,
        "baseline_ms_median": b_med,
        "cached_ms_median": c_med,
        "speedup": (b_med / c_med) if c_med > 0 else float("inf"),
        "baseline_output_tok_per_s": _tok_per_s(output_tokens, b_med),
        "cached_output_tok_per_s": _tok_per_s(output_tokens, c_med),
        "baseline_encoded_tok_per_s": _tok_per_s(baseline_encoded_tokens, b_med),
        "cached_encoded_tok_per_s": _tok_per_s(cached_encoded_tokens, c_med),
        "output_tokens": output_tokens,
        "baseline_encoded_tokens": baseline_encoded_tokens,
        "cached_encoded_tokens": cached_encoded_tokens,
        "cache_build_ms": cache_build_ms,
        "rollback_tokens_total": rollback_tokens_total,
        "rollback_tokens_max": rollback_tokens_max,
    }

    series: Dict[str, Any] = {}
    if capture_series:
        cache = AppendOnlyPieceTokenCache(enc, initial, backtrack_pieces=backtrack_pieces)
        text = initial
        turn_index: List[int] = []
        baseline_tokens_len: List[int] = []
        cached_new_tokens_len: List[int] = []
        rollback_tokens: List[int] = []
        for i, d in enumerate(deltas):
            text += d
            turn_index.append(i)
            baseline_tokens_len.append(len(enc.encode_ordinary(text)))
            delta = cache.append_ordinary(d)
            cached_new_tokens_len.append(len(delta.tokens_to_append))
            rollback_tokens.append(delta.rollback_tokens)

        series = {
            "turn_index": turn_index,
            "baseline_tokens_len": baseline_tokens_len,
            "cached_new_tokens_len": cached_new_tokens_len,
            "rollback_tokens": rollback_tokens,
        }

    return case_result, series


def _write_summary_md(results: Dict[str, Any], out_dir: Path) -> None:
    env = results["meta"]["env"]
    perf = results.get("performance", [])
    corr_fixed = results.get("correctness", {}).get("fixed_prefix", [])
    corr_append = results.get("correctness", {}).get("append_only_piece", [])

    def _all_zero(items: List[Dict[str, Any]]) -> bool:
        return all(int(x.get("verify_mismatches", 0)) == 0 for x in items)

    fixed_ok = _all_zero(corr_fixed)
    append_ok = _all_zero(corr_append)

    def md_table(headers: List[str], rows: List[List[str]]) -> List[str]:
        if not rows:
            return []
        lines0: List[str] = []
        lines0.append("| " + " | ".join(headers) + " |")
        lines0.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for r in rows:
            lines0.append("| " + " | ".join(r) + " |")
        return lines0

    lines: List[str] = []
    lines.append("# FlashToken Benchmark Summary")
    lines.append("")
    lines.append("## Environment")
    lines.append(f"- Python: `{env['python']}`")
    lines.append(f"- Platform: `{env['platform']}`")
    lines.append(f"- CPU cores: `{env['cpu_count']}`")
    lines.append(f"- tiktoken: `{env['tiktoken']}`")
    lines.append(f"- regex: `{env['regex']}`")
    lines.append(f"- encoding: `{env['encoding']}`")
    lines.append("")
    lines.append("## Correctness (token-by-token)")
    lines.append(f"- FixedPrefixTokenCache: `mismatches == 0` -> **{fixed_ok}**")
    lines.append(f"- AppendOnlyPieceTokenCache: `mismatches == 0` -> **{append_ok}**")
    lines.append("")

    fixed_rows: List[List[str]] = []
    for r in corr_fixed:
        c = r["case"]
        fixed_rows.append(
            [
                str(c["domain"]),
                str(c["prefix_chars"]),
                str(c["suffix_chars"]),
                str(c["n"]),
                str(r["stable_prefix_tokens"]),
                str(r["unstable_prefix_chars"]),
                str(r["verify_mismatches"]),
            ]
        )
    lines.append("### Correctness: Fixed Prefix")
    lines.extend(
        md_table(
            ["domain", "prefix_chars", "suffix_chars", "n", "stable_prefix_tokens", "unstable_tail_chars", "mismatches"],
            fixed_rows,
        )
    )
    lines.append("")

    append_rows: List[List[str]] = []
    for r in corr_append:
        c = r["case"]
        append_rows.append(
            [
                str(c["domain"]),
                str(c["backtrack_pieces"]),
                str(c["turns"]),
                str(c["chars_per_turn"]),
                str(r["rollback_tokens_max"]),
                str(r["verify_mismatches"]),
            ]
        )
    lines.append("### Correctness: Append-only (piece rollback)")
    lines.extend(md_table(["domain", "backtrack", "turns", "chars/turn", "rollback_max", "mismatches"], append_rows))
    lines.append("")

    main_fixed = next((p for p in perf if p["kind"] == "fixed_prefix" and p["tag"] == "main"), None)
    main_append = next((p for p in perf if p["kind"] == "append_only_piece" and p["tag"] == "main"), None)

    lines.append("## Performance Highlights (median)")
    if main_fixed:
        reduction = (
            float(main_fixed["baseline_encoded_tokens"]) / float(main_fixed["cached_encoded_tokens"])
            if main_fixed["cached_encoded_tokens"]
            else float("inf")
        )
        lines.append(
            f"- fixed_prefix: `{main_fixed['baseline_ms_median']:.2f} ms -> {main_fixed['cached_ms_median']:.2f} ms` "
            f"(`{main_fixed['speedup']:.2f}x`), encoded-token reduction `~{reduction:.2f}x`"
        )
    if main_append:
        reduction = (
            float(main_append["baseline_encoded_tokens"]) / float(main_append["cached_encoded_tokens"])
            if main_append["cached_encoded_tokens"]
            else float("inf")
        )
        lines.append(
            f"- append_only: `{main_append['baseline_ms_median']:.2f} ms -> {main_append['cached_ms_median']:.2f} ms` "
            f"(`{main_append['speedup']:.2f}x`), encoded-token reduction `~{reduction:.2f}x`, "
            f"rollback_max=`{main_append['rollback_tokens_max']}`"
        )
    lines.append("")

    lines.append("### Performance: Main Cases")
    perf_rows: List[List[str]] = []
    for p in [x for x in [main_fixed, main_append] if x]:
        perf_rows.append(
            [
                str(p["kind"]),
                str(p["case"]["domain"]),
                f"{p['baseline_ms_median']:.2f}",
                f"{p['cached_ms_median']:.2f}",
                f"{p['speedup']:.2f}x",
                str(p["baseline_encoded_tokens"]),
                str(p["cached_encoded_tokens"]),
            ]
        )
    lines.extend(
        md_table(
            ["kind", "domain", "baseline_ms", "cached_ms", "speedup", "baseline_encoded", "cached_encoded"],
            perf_rows,
        )
    )
    lines.append("")

    plots = results.get("meta", {}).get("plots", {})
    if plots:
        lines.append("## Figures")
        for k, v in plots.items():
            if v:
                lines.append(f"- {k}: ![]({v})")
        lines.append("")

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_suite(
    *,
    encoding: str,
    suite: SuiteName,
    repeats: int,
    out_dir: Path,
    no_plot: bool,
    seed: int,
) -> Dict[str, Any]:
    enc = tiktoken.get_encoding(encoding)

    # Suite sizes: keep standard runnable on a laptop, but still multi-domain + sweeps.
    if suite == "quick":
        corr_n = 80
        perf_repeats = max(1, min(2, repeats))
        perf_n = 800
        turns = 200
    elif suite == "standard":
        corr_n = 200
        perf_repeats = repeats
        perf_n = 2000
        turns = 400
    else:  # full
        corr_n = 400
        perf_repeats = max(3, repeats)
        perf_n = 4000
        turns = 800

    domains: List[TextDomain] = ["english", "chinese", "mixed", "code", "markdown", "json", "emoji"]

    correctness_fixed: List[Dict[str, Any]] = []
    correctness_append: List[Dict[str, Any]] = []
    performance: List[Dict[str, Any]] = []
    series: Dict[str, Any] = {}

    # ---- Correctness suite (token-by-token) ----
    for domain in domains:
        correctness_fixed.append(
            verify_fixed_prefix_case(
                enc,
                domain=domain,
                prefix_chars=4096,
                suffix_chars=256,
                n=corr_n,
                seed=seed + 10,
            )
        )

    for domain in domains:
        for backtrack in [1, 2, 4]:
            correctness_append.append(
                verify_append_only_case(
                    enc,
                    domain=domain,
                    initial_chars=4096,
                    turns=120 if suite == "quick" else 200,
                    chars_per_turn=96,
                    backtrack_pieces=backtrack,
                    seed=seed + 20,
                )
            )

    # ---- Performance suite ----
    performance.append(
        perf_fixed_prefix_case(
            enc,
            domain="mixed",
            prefix_chars=8000,
            suffix_chars=200,
            n=perf_n,
            seed=seed + 30,
            repeats=perf_repeats,
            tag="main",
        )
    )

    append_main, append_series = perf_append_only_case(
        enc,
        domain="mixed",
        initial_chars=8000,
        turns=turns,
        chars_per_turn=120,
        backtrack_pieces=2,
        seed=seed + 40,
        repeats=perf_repeats,
        tag="main",
        capture_series=True,
    )
    performance.append(append_main)
    series["append_only_main"] = append_series

    # Prefix length sweep (shows the "long system prompt" benefit).
    for prefix_chars in [0, 256, 1024, 4096, 8192, 16384]:
        performance.append(
            perf_fixed_prefix_case(
                enc,
                domain="english",
                prefix_chars=prefix_chars,
                suffix_chars=200,
                n=perf_n,
                seed=seed + 50,
                repeats=max(1, min(2, perf_repeats)),
                tag="prefix_sweep",
            )
        )

    # Turns sweep (shows the "越聊越烫" benefit).
    for t in [50, 100, 200, 400, 800] if suite != "quick" else [50, 100, 200]:
        r, _series0 = perf_append_only_case(
            enc,
            domain="english",
            initial_chars=8000,
            turns=t,
            chars_per_turn=120,
            backtrack_pieces=2,
            seed=seed + 60,
            repeats=max(1, min(2, perf_repeats)),
            tag="turns_sweep",
            capture_series=False,
        )
        performance.append(r)

    # Backtrack sweep (speed vs safety knob). Verify once per value, then time.
    for backtrack in [1, 2, 4, 8]:
        v = verify_append_only_case(
            enc,
            domain="mixed",
            initial_chars=4096,
            turns=120 if suite == "quick" else 200,
            chars_per_turn=96,
            backtrack_pieces=backtrack,
            seed=seed + 70,
        )
        ok = int(v["verify_mismatches"]) == 0
        r, _series0 = perf_append_only_case(
            enc,
            domain="mixed",
            initial_chars=8000,
            turns=200 if suite == "quick" else 400,
            chars_per_turn=120,
            backtrack_pieces=backtrack,
            seed=seed + 80,
            repeats=max(1, min(2, perf_repeats)),
            tag="backtrack_sweep",
            capture_series=False,
        )
        r["verify_mismatches"] = v["verify_mismatches"]
        r["verify_examples"] = v.get("verify_examples", [])
        r["valid"] = ok
        performance.append(r)

    results: Dict[str, Any] = {
        "meta": {
            "suite": suite,
            "repeats": repeats,
            "seed": seed,
            "env": asdict(collect_env(encoding)),
            "plots": {},
        },
        "correctness": {
            "fixed_prefix": correctness_fixed,
            "append_only_piece": correctness_append,
        },
        "performance": performance,
        "series": series,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    if not no_plot:
        from plots import generate_all_plots

        plots = generate_all_plots(results, out_dir)
        results["meta"]["plots"] = plots
        (out_dir / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    _write_summary_md(results, out_dir)
    return results


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="FlashToken benchmark: correctness + performance + plots.")
    parser.add_argument("--encoding", default="cl100k_base")
    parser.add_argument("--suite", choices=["quick", "standard", "full"], default="standard")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default=str(Path(__file__).with_name("out")))
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args(argv)

    if args.repeats < 1:
        raise SystemExit("--repeats must be >= 1")

    # Reduce noise; apply equally to baseline/cached.
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        out_dir = Path(args.out_dir)
        run_suite(
            encoding=args.encoding,
            suite=args.suite,  # type: ignore[arg-type]
            repeats=args.repeats,
            out_dir=out_dir,
            no_plot=bool(args.no_plot),
            seed=args.seed,
        )
        print(f"Wrote: {out_dir / 'results.json'}")
        print(f"Wrote: {out_dir / 'summary.md'}")
        return 0
    finally:
        if gc_was_enabled:
            gc.enable()


if __name__ == "__main__":
    raise SystemExit(main())
