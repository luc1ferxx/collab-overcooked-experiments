import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path("src/data/gpt-3.5-turbo-0125")
ORDERS = ["boiled_egg", "boiled_mushroom"]
MODES = {
    "baseline": "Baseline",
    "reduced": "Reduced",
}

OFFSETS = {
    ("boiled_egg", "Baseline"):           (10, 6),
    ("boiled_mushroom", "Baseline"):      (-10, 6),
    ("boiled_egg", "Reduced"):      (16, -10),
    ("boiled_mushroom", "Reduced"): (-14, -6),
}

def _annotate_point(ax, x, y, order, mode):
    dx, dy = OFFSETS.get((order, mode), (12, 8))
    ax.annotate(
        f"{order}\n{mode}",
        (x, y),
        textcoords="offset points",
        xytext=(dx, dy),
        ha="left" if dx > 0 else "right",
        va="bottom" if dy > 0 else "top",
        fontsize=8,  # smaller so it doesn’t crowd
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.7", alpha=0.9),
        arrowprops=dict(arrowstyle="-", lw=0.8, color="0.35"),
        clip_on=False,
    )


def save_table_png(df: pd.DataFrame, path: str, title: str | None = None, float_format: str = "{:.2f}") -> None:
    if df.empty:
        return
    display_df = df.copy()
    for col in display_df.select_dtypes(include=[float]).columns:
        display_df[col] = display_df[col].map(lambda x: float_format.format(x) if not math.isnan(x) else "")
    fig, ax = plt.subplots(figsize=(max(8, 2 + 1.5 * len(display_df.columns)), 0.6 * (len(display_df) + 2)))
    ax.axis("off")
    if title:
        ax.set_title(title, pad=20)
    table = ax.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        loc="center",
        cellLoc="center",
    )
    table.scale(1, 1.5)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)

run_records: list[dict] = []
call_records: list[dict] = []

for order in ORDERS:
    for mode_key, mode_name in MODES.items():
        folder = ROOT / order / mode_key
        if not folder.exists():
            continue
        for json_path in sorted(folder.glob(f"experiment_*_{order}.json")):
            with json_path.open(encoding="utf-8") as f:
                data = json.load(f)

            summary = data.get("communication_tokens_summary") or {}
            total_tokens = summary.get("team_total_tokens")
            total_turns = summary.get("team_total_turns")

            per_agent = summary.get("per_agent", [])
            call_tokens: list[int] = []

            for agent_info in per_agent:
                agent_name = agent_info.get("agent", f"agent{agent_info.get('agent_index', 'x')}")
                for call in agent_info.get("by_turn", []):
                    tokens = call.get("tokens")
                    if tokens is None:
                        continue
                    call_tokens.append(tokens)
                    call_records.append(
                        {
                            "order": order,
                            "mode_key": mode_key,
                            "mode": mode_name,
                            "file": json_path.name,
                            "agent": agent_name,
                            "timestep": call.get("t"),
                            "tokens": tokens,
                        }
                    )

            call_tokens_array = np.array(call_tokens, dtype=float) if call_tokens else np.array([], dtype=float)
            positive_tokens = call_tokens_array[call_tokens_array > 0]
            zero_tokens = call_tokens_array[call_tokens_array <= 0]

            avg_tokens_per_call = float(positive_tokens.mean()) if positive_tokens.size > 0 else 0.0
            median_tokens_per_call = float(np.median(positive_tokens)) if positive_tokens.size > 0 else 0.0
            max_tokens_per_call = float(positive_tokens.max()) if positive_tokens.size > 0 else 0.0

            run_records.append(
                {
                    "order": order,
                    "mode_key": mode_key,
                    "mode": mode_name,
                    "file": json_path.name,
                    "tokens": total_tokens,
                    "turns": total_turns,
                    "success": 1 if data.get("total_score", 0) > 0 else 0,
                    "timesteps": len(data.get("total_timestamp", [])),
                    "nonzero_calls": int(positive_tokens.size),
                    "zero_token_calls": int(zero_tokens.size),
                    "avg_tokens_per_call": avg_tokens_per_call,
                    "median_tokens_per_call": median_tokens_per_call,
                    "max_tokens_per_call": max_tokens_per_call,
                }
            )

if not run_records:
    raise SystemExit("No experiment_* files found under src/data/gpt-3.5-turbo-0125.")

runs = pd.DataFrame(run_records).dropna(subset=["tokens"])
if "prompt_tokens" in runs.columns:
    pass
runs.to_csv("results_raw_runs.csv", index=False)

calls = pd.DataFrame(call_records)
if not calls.empty:
    calls.to_csv("call_tokens_raw.csv", index=False)

summary = (
    runs.groupby(["order", "mode"])
    .agg(
        episodes=("file", "count"),
        success_rate=("success", "mean"),
        tokens_mean=("tokens", "mean"),
        tokens_std=("tokens", "std"),
        tokens_min=("tokens", "min"),
        tokens_max=("tokens", "max"),
        turns_mean=("turns", "mean"),
        turns_std=("turns", "std"),
        timesteps_mean=("timesteps", "mean"),
        timesteps_std=("timesteps", "std"),
        nonzero_calls_mean=("nonzero_calls", "mean"),
        avg_tokens_per_call_mean=("avg_tokens_per_call", "mean"),
        median_tokens_per_call_mean=("median_tokens_per_call", "mean"),
    )
    .reset_index()
)

# 95% confidence intervals for tokens / turns
for metric in ["tokens", "turns", "timesteps"]:
    std_col = f"{metric}_std"
    mean_col = f"{metric}_mean"
    if std_col not in summary.columns:
        continue
    n = summary["episodes"].replace(0, np.nan)
    se = summary[std_col] / np.sqrt(n)
    ci = 1.96 * se
    summary[f"{metric}_ci_low"] = summary[mean_col] - ci
    summary[f"{metric}_ci_high"] = summary[mean_col] + ci

summary.to_csv("results_summary.csv", index=False)
save_table_png(summary, "results_summary_table.png", title="Episode-level Metrics")

# Welch-style differences between baseline and reduced (tokens & turns)
diff_rows: list[dict] = []
for order in ORDERS:
    base = runs[(runs["order"] == order) & (runs["mode_key"] == "baseline")]
    red = runs[(runs["order"] == order) & (runs["mode_key"] == "reduced")]
    if base.empty or red.empty:
        continue

    def diff_ci(series1: pd.Series, series2: pd.Series) -> tuple[float, float, float]:
        mean_diff = series1.mean() - series2.mean()
        n1, n2 = len(series1), len(series2)
        var1 = series1.var(ddof=1) if n1 > 1 else 0.0
        var2 = series2.var(ddof=1) if n2 > 1 else 0.0
        se = math.sqrt((var1 / n1 if n1 else 0.0) + (var2 / n2 if n2 else 0.0))
        if se == 0.0:
            return mean_diff, math.nan, math.nan
        margin = 1.96 * se
        return mean_diff, mean_diff - margin, mean_diff + margin

    tok_diff, tok_ci_low, tok_ci_high = diff_ci(base["tokens"], red["tokens"])
    turn_diff, turn_ci_low, turn_ci_high = diff_ci(base["turns"], red["turns"])

    diff_rows.append(
        {
            "order": order,
            "tokens_diff": tok_diff,
            "tokens_diff_ci_low": tok_ci_low,
            "tokens_diff_ci_high": tok_ci_high,
            "turns_diff": turn_diff,
            "turns_diff_ci_low": turn_ci_low,
            "turns_diff_ci_high": turn_ci_high,
        }
    )

if diff_rows:
    diff_df = pd.DataFrame(diff_rows)
    diff_df.to_csv("results_differences.csv", index=False)
    save_table_png(diff_df, "results_differences_table.png", title="Baseline − Reduced (tokens / turns)")
else:
    diff_df = pd.DataFrame()

# Per-call statistics (diagnostics)
if not calls.empty:
    calls["is_positive"] = calls["tokens"] > 0

    def positive_mean(series: pd.Series) -> float:
        positive = series[series > 0]
        return float(positive.mean()) if not positive.empty else 0.0

    def positive_median(series: pd.Series) -> float:
        positive = series[series > 0]
        return float(positive.median()) if not positive.empty else 0.0

    call_summary = (
        calls.groupby(["order", "mode"])
        .agg(
            total_calls=("tokens", "count"),
            positive_calls=("is_positive", "sum"),
            zero_token_calls=("is_positive", lambda x: (~x).sum()),
            mean_tokens_per_call=("tokens", "mean"),
            std_tokens_per_call=("tokens", "std"),
            mean_positive_tokens=("tokens", positive_mean),
            median_positive_tokens=("tokens", positive_median),
        )
        .reset_index()
    )
    call_summary.to_csv("call_tokens_summary.csv", index=False)
    save_table_png(call_summary, "call_tokens_summary_table.png", title="Per-call Token Diagnostics")
else:
    call_summary = pd.DataFrame()

# Plots
pivot_tokens = runs.pivot_table(index="order", columns="mode", values="tokens", aggfunc="mean")
ax = pivot_tokens.plot(kind="bar", figsize=(6, 4))
ax.set_ylabel("Average LLM tokens per episode")
ax.set_title("Communication cost by order / mode")
plt.tight_layout()
plt.savefig("results_tokens_bar.png", dpi=200)
plt.close()

pivot_turns = runs.pivot_table(index="order", columns="mode", values="turns", aggfunc="mean")
ax = pivot_turns.plot(kind="bar", figsize=(6, 4))
ax.set_ylabel("Average communication turns per episode")
ax.set_title("LLM calls by order / mode")
plt.tight_layout()
plt.savefig("results_turns_bar.png", dpi=200)
plt.close()

if not calls.empty and not calls[calls["tokens"] > 0].empty:
    positive_calls = calls[calls["tokens"] > 0]
    box_data = []
    box_labels = []
    for order in ORDERS:
        for mode_key, mode_name in MODES.items():
            subset = positive_calls[(positive_calls["order"] == order) & (positive_calls["mode_key"] == mode_key)]
            if subset.empty:
                continue
            box_data.append(subset["tokens"].values)
            box_labels.append(f"{order}\n{mode_name}")
    if box_data:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.boxplot(box_data, tick_labels=box_labels, showfliers=False)
        ax.set_xticklabels(box_labels, rotation=20, ha="right")
        ax.set_ylabel("Tokens per LLM call")
        ax.set_title("Distribution of tokens per API call")
        plt.tight_layout()
        plt.savefig("call_tokens_boxplot.png", dpi=200)
        plt.close()

# Trade-off scatter plots
success_fig = plt.figure(figsize=(8.2, 5.2))
fig, ax = plt.subplots()
fig.set_size_inches(8, 5)
ax.margins(x=0.10, y=0.15)
ax = success_fig.add_subplot(1, 1, 1)
ax.errorbar(
    summary["tokens_mean"],
    summary["success_rate"],
    xerr=[summary["tokens_mean"] - summary["tokens_ci_low"], summary["tokens_ci_high"] - summary["tokens_mean"]],
    fmt="o",
    capsize=4,
)
for _, row in summary.iterrows():
    offsets = {
        ("boiled_egg", "Baseline"):           (10, 6),
        ("boiled_mushroom", "Baseline"):      (0, -10),
        ("boiled_egg", "Reduced"):      (-12, -10),
        ("boiled_mushroom", "Reduced"): (12, -10),
    }
    dx, dy = offsets.get((row["order"], row["mode"]), (12, 8))
    ax.annotate(
        f"{row['order']}\n{row['mode']}",
        (row["tokens_mean"], row["success_rate"]),
        textcoords="offset points",
        xytext=(dx, dy),
        ha="left" if dx > 0 else "right",
        va="bottom" if dy > 0 else "top",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.7", alpha=0.9),
        arrowprops=dict(arrowstyle="-", lw=0.8, color="0.35"),
        clip_on=False,
    )
ax.set_xlabel("Average LLM tokens per episode")
ax.set_ylabel("Success rate")
ax.set_ylim(0, 1.08)
ax.set_title("Tokens vs success", pad=16)
xmin = float(summary["tokens_ci_low"].min())
xmax = float(summary["tokens_ci_high"].max())
pad = 0.08 * (xmax - xmin if xmax > xmin else 1.0)
ax.set_xlim(xmin - pad, xmax + pad)
ax.margins(x=0.10, y=0.18)
plt.tight_layout()
success_fig.savefig("tradeoff_tokens_success.png", dpi=200)
plt.close(success_fig)

if {"timesteps_mean", "timesteps_ci_low"}.issubset(summary.columns):
    ts_fig = plt.figure(figsize=(7.8, 5.0))
    fig, ax = plt.subplots()
    fig.set_size_inches(7, 5)
    ax.margins(x=0.10, y=0.15)
    ax = ts_fig.add_subplot(1, 1, 1)
    ax.errorbar(
        summary["tokens_mean"],
        summary["timesteps_mean"],
        xerr=[summary["tokens_mean"] - summary["tokens_ci_low"], summary["tokens_ci_high"] - summary["tokens_mean"]],
        yerr=[summary["timesteps_mean"] - summary["timesteps_ci_low"], summary["timesteps_ci_high"] - summary["timesteps_mean"]],
        fmt="o",
        capsize=4,
    )
    for _, row in summary.iterrows():
        if row["mode"] == "Reduced":
            if row["order"] == "boiled_egg":
                dx, dy = (16, -10)
                ha = "left";  va = "top"
            else:  # boiled_mushroom
                dx, dy = (16, 20)
                ha = "left"; va = "top"
        else:
            if row["order"] == "boiled_egg":
                dx, dy = (10, 6);    ha = "left";  va = "bottom"
            else:  # boiled_mushroom baseline
                dx, dy = (10, 20);   ha = "left"; va = "bottom"

        ax.annotate(
            f"{row['order']}\n{row['mode']}",
            (row["tokens_mean"], row["timesteps_mean"]),
            textcoords="offset points",
            xytext=(dx, dy),
            ha=ha, va=va,
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.7", alpha=0.9),
            arrowprops=dict(arrowstyle="-", lw=0.8, color="0.35"),
        clip_on=False,
        zorder=5,
    )

    ax.set_xlabel("Average LLM tokens per episode")
    ax.set_ylabel("Timesteps to completion")
    ax.set_title("Tokens vs timesteps", pad=12)
    xmin = float(summary["tokens_ci_low"].min())
    xmax = float(summary["tokens_ci_high"].max())
    pad = 0.08 * (xmax - xmin if xmax > xmin else 1.0)
    ax.set_xlim(xmin - pad, xmax + pad)
    ax.margins(x=0.10, y=0.12)
    plt.tight_layout()
    ts_fig.savefig("tradeoff_tokens_timesteps.png", dpi=200)
    plt.close(ts_fig)

print("Saved per-run stats to results_raw_runs.csv")
print(runs.head())

print("\nSummary statistics (LaTeX/markdown ready):")
print(summary.to_markdown(index=False, floatfmt=".2f"))

if not diff_df.empty:
    print("\nToken/turn differences (baseline minus reduced):")
    print(diff_df.to_markdown(index=False, floatfmt=".2f"))

if not call_summary.empty:
    print("\nPer-call token diagnostics:")
    print(call_summary.to_markdown(index=False, floatfmt=".2f"))

print("\nWrote results_summary.csv, results_tokens_bar.png, results_turns_bar.png")
print("Additional plots saved: tradeoff_tokens_success.png, tradeoff_tokens_timesteps.png, call_tokens_boxplot.png")