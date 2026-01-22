from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List


def _require_matplotlib():
    import matplotlib.pyplot as plt  # noqa: F401


def _savefig(path: Path) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_fixed_prefix_speedup(perf_cases: List[Dict[str, Any]], out_dir: Path) -> str:
    """
    Line plot: speedup vs prefix_chars (tag=prefix_sweep).
    """
    import matplotlib.pyplot as plt

    cases = [c for c in perf_cases if c["tag"] == "prefix_sweep"]
    if not cases:
        return ""

    # Group by suffix_chars.
    groups: Dict[int, List[Dict[str, Any]]] = {}
    for c in cases:
        groups.setdefault(int(c["case"]["suffix_chars"]), []).append(c)

    plt.figure(figsize=(7.2, 4.3))
    for suffix_chars, rows in sorted(groups.items()):
        rows = sorted(rows, key=lambda r: int(r["case"]["prefix_chars"]))
        xs = [int(r["case"]["prefix_chars"]) for r in rows]
        ys = [float(r["speedup"]) for r in rows]
        plt.plot(xs, ys, marker="o", linewidth=2, label=f"suffix={suffix_chars} chars")

    plt.title("Fixed Prefix: Speedup vs Prefix Length")
    plt.xlabel("prefix length (chars)")
    plt.ylabel("speedup (x)")
    plt.grid(True, alpha=0.25)
    plt.legend()
    path = out_dir / "fixed_prefix_speedup.png"
    _savefig(path)
    return path.name


def plot_append_only_speedup(perf_cases: List[Dict[str, Any]], out_dir: Path) -> str:
    """
    Line plot: speedup vs turns (tag=turns_sweep).
    """
    import matplotlib.pyplot as plt

    cases = [c for c in perf_cases if c["tag"] == "turns_sweep"]
    if not cases:
        return ""

    rows = sorted(cases, key=lambda r: int(r["case"]["turns"]))
    xs = [int(r["case"]["turns"]) for r in rows]
    ys = [float(r["speedup"]) for r in rows]

    plt.figure(figsize=(7.2, 4.3))
    plt.plot(xs, ys, marker="o", linewidth=2)
    plt.title("Append-only: Speedup vs Conversation Turns")
    plt.xlabel("turns")
    plt.ylabel("speedup (x)")
    plt.grid(True, alpha=0.25)
    path = out_dir / "append_only_speedup.png"
    _savefig(path)
    return path.name


def plot_append_only_work_series(series: Dict[str, Any], out_dir: Path) -> str:
    """
    Two-axis plot:
      - baseline output tokens per turn (grows with history)
      - cached encoded tokens per turn (stays small)
    """
    import matplotlib.pyplot as plt

    if not series:
        return ""
    turns = series.get("turn_index")
    if not turns:
        return ""

    base_tokens = series["baseline_tokens_len"]
    cached_new = series["cached_new_tokens_len"]

    plt.figure(figsize=(7.6, 4.6))
    ax = plt.gca()
    ax.plot(turns, base_tokens, color="#1f77b4", linewidth=2, label="baseline: output tokens (per turn)")
    ax.set_xlabel("turn index")
    ax.set_ylabel("tokens (baseline)", color="#1f77b4")
    ax.tick_params(axis="y", labelcolor="#1f77b4")
    ax.grid(True, alpha=0.25)

    ax2 = ax.twinx()
    ax2.plot(turns, cached_new, color="#ff7f0e", linewidth=2, label="FlashToken: encoded new tokens (per turn)")
    ax2.set_ylabel("tokens (FlashToken new work)", color="#ff7f0e")
    ax2.tick_params(axis="y", labelcolor="#ff7f0e")

    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc="upper left")
    plt.title("Append-only Workload: Why Baseline Gets Hotter Over Time")

    path = out_dir / "append_only_work_series.png"
    _savefig(path)
    return path.name


def plot_rollback_hist(series: Dict[str, Any], out_dir: Path) -> str:
    import matplotlib.pyplot as plt

    if not series:
        return ""
    rb = series.get("rollback_tokens")
    if not rb:
        return ""

    plt.figure(figsize=(7.2, 4.3))
    plt.hist(rb, bins=range(0, max(rb) + 2), rwidth=0.9)
    plt.title("Append-only: Rollback Tokens Histogram")
    plt.xlabel("rollback tokens per append")
    plt.ylabel("count")
    plt.grid(True, axis="y", alpha=0.25)
    path = out_dir / "append_only_rollback_hist.png"
    _savefig(path)
    return path.name


def generate_all_plots(results: Dict[str, Any], out_dir: Path) -> Dict[str, str]:
    _require_matplotlib()

    perf_cases = results.get("performance", [])
    series = results.get("series", {}).get("append_only_main", {})

    plots: Dict[str, str] = {}
    plots["fixed_prefix_speedup"] = plot_fixed_prefix_speedup(perf_cases, out_dir) or ""
    plots["append_only_speedup"] = plot_append_only_speedup(perf_cases, out_dir) or ""
    plots["append_only_work_series"] = plot_append_only_work_series(series, out_dir) or ""
    plots["append_only_rollback_hist"] = plot_rollback_hist(series, out_dir) or ""
    return plots

