from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MODES = {
    "baseline": "Free-form",
    "reduced": "Reduced",
    "always": "Always Talk",
    "deterministic": "Deterministic Chat",
    "pruning_only": "Pruning Only",
}

OFFSETS = {
    ("boiled_egg", "Baseline"): (10, 6),
    ("boiled_mushroom", "Baseline"): (-10, 6),
    ("boiled_egg", "Reduced"): (16, -10),
    ("boiled_mushroom", "Reduced"): (-14, -6),
}

TIMESTEP_OFFSETS = {
    ("boiled_egg", "Free-form"): (12, 10, "left", "bottom"),
    ("boiled_mushroom", "Free-form"): (70, -33, "right", "top"),
    ("boiled_egg", "Always Talk"): (40, 33, "right", "top"),
    ("boiled_egg", "Deterministic Chat"): (14, -8, "left", "bottom"),
    ("boiled_egg", "Pruning Only"): (-18, 33, "right", "top"),
}

SUCCESS_OFFSETS = {
    ("boiled_egg", "Free-form"): (-42, 10, "left", "bottom"),
    ("boiled_mushroom", "Free-form"): (40, -18, "right", "top"),
    ("boiled_egg", "Always Talk"): (70, 28, "right", "top"),
    ("boiled_egg", "Deterministic Chat"): (14, -8, "left", "bottom"),
    ("boiled_egg", "Pruning Only"): (-18, -16, "right", "top"),
}

MODEL_FOLDERS = {
    "gpt-3.5-turbo-0125": "gpt-3.5-turbo-0125",
    "qwen2.5-3b": "qwen2.5-3b",
}

MODEL_DISPLAY = {
    "gpt-3.5-turbo-0125": "Mistral-7B",
    "qwen2.5-3b": "Qwen2.5-3B",
}

# Absolute label positions (data coords) to avoid overlap, keyed by display model.
MANUAL_SUCCESS_POS: dict[tuple[str, str, str], tuple[float, float, str, str]] = {
    # Mistral-7B
    ("Mistral-7B", "zucchini_green_pea_and_onion_patty", "Reduced"): (5000, 0.5, "left", "center"),
    ("Mistral-7B", "taro_bean_and_bell_pepper_patty", "Reduced"): (5000, 0.97, "left", "center"),
    ("Mistral-7B", "boiled_egg", "Reduced"): (0, 0.74, "left", "center"),
    ("Mistral-7B", "taro_bean_and_bell_pepper_patty", "Free-form"): (13000, 0.8, "left", "center"),
    ("Mistral-7B", "zucchini_green_pea_and_onion_patty", "Free-form"): (17000, 0.65, "left", "center"),
    ("Mistral-7B", "boiled_egg", "Free-form"): (21000, 0.5, "left", "center"),
    ("Mistral-7B", "boiled_egg", "Always Talk"): (34000, 0.75, "left", "center"),
    ("Mistral-7B", "boiled_mushroom", "Free-form"): (30000, 0.84, "left", "center"),
    ("Mistral-7B", "boiled_egg", "Pruning Only"): (22000, 0.82, "left", "center"),
    ("Mistral-7B", "boiled_mushroom", "Reduced"): (0, 1.10, "left", "center"),
    # Qwen2.5-3B
    ("Qwen2.5-3B", "zucchini_green_pea_and_onion_patty", "Reduced"): (4000, 0.5, "left", "center"),
    ("Qwen2.5-3B", "taro_bean_and_bell_pepper_patty", "Reduced"): (0, 0.6, "left", "center"),
    ("Qwen2.5-3B", "boiled_egg", "Reduced"): (7000, 0.74, "left", "center"),
    ("Qwen2.5-3B", "taro_bean_and_bell_pepper_patty", "Free-form"): (12000, 0.4, "left", "center"),
    ("Qwen2.5-3B", "boiled_egg", "Free-form"): (16000, 0.5, "left", "center"),
    ("Qwen2.5-3B", "boiled_egg", "Always Talk"): (18000, 0.2, "left", "center"),
    ("Qwen2.5-3B", "zucchini_green_pea_and_onion_patty", "Free-form"): (25000, 0.8, "left", "center"),
    ("Qwen2.5-3B", "boiled_mushroom", "Free-form"): (17000, 0.84, "left", "center"),
    ("Qwen2.5-3B", "boiled_mushroom", "Reduced"): (1500, 1.10, "left", "center"),
}

MANUAL_TIMESTEPS_POS: dict[tuple[str, str, str], tuple[float, float, str, str]] = {
    # Mistral-7B
    ("Mistral-7B", "zucchini_green_pea_and_onion_patty", "Reduced"): (0, 50.0, "left", "center"),
    ("Mistral-7B", "taro_bean_and_bell_pepper_patty", "Reduced"): (5000, 50.0, "left", "center"),
    ("Mistral-7B", "boiled_egg", "Reduced"): (5000, 15.0, "left", "center"),
    ("Mistral-7B", "boiled_mushroom", "Reduced"): (1000, 20.0, "left", "center"),
    ("Mistral-7B", "taro_bean_and_bell_pepper_patty", "Free-form"): (14000, 50.0, "left", "center"),
    ("Mistral-7B", "zucchini_green_pea_and_onion_patty", "Free-form"): (18000, 50.0, "left", "center"),
    ("Mistral-7B", "boiled_egg", "Free-form"): (23000, 30.0, "left", "center"),
    ("Mistral-7B", "boiled_egg", "Always Talk"): (35000, 30.0, "left", "center"),
    ("Mistral-7B", "boiled_mushroom", "Free-form"): (30000, 30.0, "left", "center"),
    ("Mistral-7B", "boiled_egg", "Pruning Only"): (17000, 22.0, "left", "center"),
    # Qwen2.5-3B
    ("Qwen2.5-3B", "zucchini_green_pea_and_onion_patty", "Reduced"): (4000, 50.0, "left", "center"),
    ("Qwen2.5-3B", "taro_bean_and_bell_pepper_patty", "Reduced"): (0, 45.0, "left", "center"),
    ("Qwen2.5-3B", "boiled_egg", "Reduced"): (0, 20.0, "left", "center"),
    ("Qwen2.5-3B", "boiled_mushroom", "Reduced"): (4000, 20.0, "left", "center"),
    ("Qwen2.5-3B", "taro_bean_and_bell_pepper_patty", "Free-form"): (12000, 50.0, "left", "center"),
    ("Qwen2.5-3B", "zucchini_green_pea_and_onion_patty", "Free-form"): (20000, 50.0, "left", "center"),
    ("Qwen2.5-3B", "boiled_egg", "Free-form"): (14000, 20.0, "left", "center"),
    ("Qwen2.5-3B", "boiled_mushroom", "Free-form"): (22000, 20.0, "left", "center"),
}

# Model-specific manual label offsets to avoid overlap (dx, dy, ha, va).
# Orders used: zucchini_green_pea_and_onion_patty, taro_bean_and_bell_pepper_patty, boiled_egg, boiled_mushroom
MODEL_SUCCESS_OFFSETS: dict[tuple[str, str, str], tuple[int, int, str, str]] = {
    # Mistral-7B
    ("Mistral-7B", "zucchini_green_pea_and_onion_patty", "Reduced"): (-22, 16, "right", "bottom"),
    ("Mistral-7B", "taro_bean_and_bell_pepper_patty", "Reduced"): (18, 18, "left", "bottom"),
    ("Mistral-7B", "boiled_egg", "Reduced"): (18, -52, "left", "top"),
    ("Mistral-7B", "boiled_mushroom", "Reduced"): (-36, 12, "right", "bottom"),
    ("Mistral-7B", "taro_bean_and_bell_pepper_patty", "Free-form"): (-24, 26, "right", "bottom"),
    ("Mistral-7B", "zucchini_green_pea_and_onion_patty", "Free-form"): (34, 32, "left", "bottom"),
    ("Mistral-7B", "boiled_egg", "Free-form"): (18, 18, "left", "bottom"),
    ("Mistral-7B", "boiled_egg", "Always Talk"): (20, 6, "left", "bottom"),
    ("Mistral-7B", "boiled_mushroom", "Free-form"): (18, -54, "left", "top"),
    ("Mistral-7B", "boiled_egg", "Pruning Only"): (-16, -28, "right", "top"),
    # Qwen2.5-3B
    ("Qwen2.5-3B", "zucchini_green_pea_and_onion_patty", "Reduced"): (-22, 14, "right", "bottom"),
    ("Qwen2.5-3B", "taro_bean_and_bell_pepper_patty", "Reduced"): (20, 14, "left", "bottom"),
    ("Qwen2.5-3B", "boiled_egg", "Reduced"): (18, -56, "left", "top"),
    ("Qwen2.5-3B", "boiled_mushroom", "Reduced"): (-34, 10, "right", "bottom"),
    ("Qwen2.5-3B", "taro_bean_and_bell_pepper_patty", "Free-form"): (-26, 26, "right", "bottom"),
    ("Qwen2.5-3B", "boiled_egg", "Free-form"): (6, 30, "left", "bottom"),
    ("Qwen2.5-3B", "zucchini_green_pea_and_onion_patty", "Free-form"): (40, 34, "left", "bottom"),
    ("Qwen2.5-3B", "boiled_egg", "Always Talk"): (22, -2, "left", "top"),
    ("Qwen2.5-3B", "boiled_mushroom", "Free-form"): (32, -62, "left", "top"),
}

MODEL_TIMESTEP_OFFSETS: dict[tuple[str, str, str], tuple[int, int, str, str]] = {
    # Mistral-7B
    ("Mistral-7B", "zucchini_green_pea_and_onion_patty", "Reduced"): (-22, 12, "right", "bottom"),
    ("Mistral-7B", "taro_bean_and_bell_pepper_patty", "Reduced"): (20, 12, "left", "bottom"),
    ("Mistral-7B", "boiled_egg", "Reduced"): (18, -50, "left", "top"),
    ("Mistral-7B", "boiled_mushroom", "Reduced"): (-26, 8, "right", "bottom"),
    ("Mistral-7B", "taro_bean_and_bell_pepper_patty", "Free-form"): (-10, 26, "right", "bottom"),
    ("Mistral-7B", "zucchini_green_pea_and_onion_patty", "Free-form"): (30, 22, "left", "bottom"),
    ("Mistral-7B", "boiled_egg", "Free-form"): (6, -18, "left", "top"),
    ("Mistral-7B", "boiled_egg", "Always Talk"): (10, 10, "left", "bottom"),
    ("Mistral-7B", "boiled_mushroom", "Free-form"): (18, -62, "left", "top"),
    ("Mistral-7B", "boiled_egg", "Pruning Only"): (-12, 20, "right", "bottom"),
    # Qwen2.5-3B
    ("Qwen2.5-3B", "zucchini_green_pea_and_onion_patty", "Reduced"): (-20, 12, "right", "bottom"),
    ("Qwen2.5-3B", "taro_bean_and_bell_pepper_patty", "Reduced"): (18, 12, "left", "bottom"),
    ("Qwen2.5-3B", "boiled_egg", "Reduced"): (20, -52, "left", "top"),
    ("Qwen2.5-3B", "boiled_mushroom", "Reduced"): (-30, 8, "right", "bottom"),
    ("Qwen2.5-3B", "taro_bean_and_bell_pepper_patty", "Free-form"): (-10, 22, "right", "bottom"),
    ("Qwen2.5-3B", "zucchini_green_pea_and_onion_patty", "Free-form"): (32, 20, "left", "bottom"),
    ("Qwen2.5-3B", "boiled_egg", "Free-form"): (-8, -18, "right", "top"),
    ("Qwen2.5-3B", "boiled_egg", "Always Talk"): (12, 12, "left", "bottom"),
    ("Qwen2.5-3B", "boiled_mushroom", "Free-form"): (22, -64, "left", "top"),
    ("Qwen2.5-3B", "boiled_egg", "Pruning Only"): (-12, 18, "right", "bottom"),
}


def _fmt_order(order: str) -> str:
    """Shorten long order names for plotting."""
    return order.replace("_and_", " & ").replace("_", "\n")


def _adjust_labels(texts, ax) -> None:
    """No-op placeholder to keep runtime fast."""
    return


def _repel_texts_in_points(texts: list, min_dist: float = 14.0, max_iter: int = 200) -> None:
    """Simple fallback repulsion in text coordinate space (points)."""
    if not texts:
        return
    min_dist = max(min_dist, 18.0)
    max_iter = max_iter * 2
    for _ in range(max_iter):
        moved = False
        positions = [np.array(t.get_position(), dtype=float) for t in texts]
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                delta = positions[j] - positions[i]
                dist = float(np.hypot(delta[0], delta[1]))
                if dist < 1e-6:
                    delta = np.array([0.5, 0.0])
                    dist = 0.5
                if dist < min_dist:
                    push = (min_dist - dist) / 2.0
                    shift = delta / dist * push
                    positions[i] -= shift
                    positions[j] += shift
                    moved = True
        if not moved:
            break
        for t, pos in zip(texts, positions):
            t.set_position(pos)


def _jitter_offset(key: str, scale: float = 10.0) -> tuple[float, float]:
    """Deterministic small jitter based on a string key."""
    h = abs(hash(key))
    dx = ((h % 1000) / 999.0 - 0.5) * 2 * scale
    dy = (((h // 1000) % 1000) / 999.0 - 0.5) * 2 * scale
    return dx, dy


def _clamp_annotations(ax, texts, pad_px: float = 6.0) -> None:
    """No-op placeholder to keep runtime fast."""
    return


def _spread_annotations(ax, texts, anchors, pad_px: float = 8.0, step_px: float = 4.0, max_iter: int = 200, max_rad_px: float = 120.0) -> None:
    """No-op placeholder to keep runtime fast."""
    return


def _annotate_point(ax, x, y, order, mode):
    dx, dy = OFFSETS.get((order, mode), (12, 8))
    ax.annotate(
        f"{order}\n{mode}",
        (x, y),
        textcoords="offset points",
        xytext=(dx, dy),
        ha="left" if dx > 0 else "right",
        va="bottom" if dy > 0 else "top",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.7", alpha=0.9),
        arrowprops=dict(arrowstyle="-", lw=0.8, color="0.35"),
        clip_on=False,
    )


def save_table_png(df: pd.DataFrame, path: Path, title: str | None = None, float_format: str = "{:.2f}") -> None:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize communication metrics across runs.")
    parser.add_argument(
        "--gpt_model",
        type=str,
        default="gpt-3.5-turbo-0125",
        help="Model whose results to summarize (used if --gpt_models not set)",
    )
    parser.add_argument(
        "--gpt_models",
        type=str,
        default=None,
        help="Comma-separated list of models to summarize. If set, overrides --gpt_model.",
    )
    parser.add_argument(
        "--data_root",
        type=Path,
        default=Path("data"),
        help="Root directory containing model result folders",
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=None,
        help="Optional output root. Defaults to the directory of this script; per-model subfolders will be created.",
    )
    return parser.parse_args()


def get_model_folder_name(model_name: str) -> str:
    """Hard-coded mapping for the two models we use; sanitize as fallback."""
    if model_name in MODEL_FOLDERS:
        return MODEL_FOLDERS[model_name]
    return model_name.replace("/", "_").replace("\\", "_").replace(":", "_")


def get_model_display_name(model_name: str) -> str:
    """Human-friendly display name for plots / tables."""
    return MODEL_DISPLAY.get(model_name, model_name)


def _get_label_offset(model_display: str, order: str, mode: str, model_map: dict, generic_map: dict, default_dxdy: tuple[int, int]) -> tuple[int, int, str, str]:
    if (model_display, order, mode) in model_map:
        dx, dy, ha, va = model_map[(model_display, order, mode)]
        return dx, dy, ha, va
    if (order, mode) in generic_map:
        dx, dy, ha, va = generic_map[(order, mode)]
        return dx, dy, ha, va
    dx, dy = default_dxdy
    return dx, dy, "left" if dx > 0 else "right", "bottom" if dy > 0 else "top"


def _center_bias_offset(x: float, y: float, x_min: float, x_max: float, y_min: float, y_max: float, scale_x: float = 60.0, scale_y: float = 20.0) -> tuple[float, float]:
    """Nudge labels toward the center of the plot to use empty space."""
    if x_max <= x_min:
        return 0.0, 0.0
    dx = ((x_max + x_min) / 2.0 - x) / (x_max - x_min) * scale_x
    dy = 0.0
    if y_max > y_min:
        dy = ((y_max + y_min) / 2.0 - y) / (y_max - y_min) * scale_y
    return dx, dy


def summarize_model(model_name: str, data_root: Path, output_base: Path) -> None:
    """Summarize a single model and emit per-model artifacts. Returns key DataFrames."""
    root = data_root / get_model_folder_name(model_name)
    output_dir = output_base / get_model_folder_name(model_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    display_model = get_model_display_name(model_name)

    run_records: list[dict] = []
    call_records: list[dict] = []

    # auto-discover orders and modes present under this model
    order_dirs = [p for p in root.iterdir() if p.is_dir()]
    orders = [p.name for p in order_dirs]

    for order in orders:
        for mode_key, mode_name in MODES.items():
            folder = root / order / mode_key
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
                                "model": model_name,
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
                        "model": model_name,
                    }
                )

    if not run_records:
        raise SystemExit(f"No experiment_* files found under {root}.")

    runs = pd.DataFrame(run_records).dropna(subset=["tokens"])
    runs.to_csv(output_dir / "results_raw_runs.csv", index=False)

    calls = pd.DataFrame(call_records)
    if not calls.empty:
        calls.to_csv(output_dir / "call_tokens_raw.csv", index=False)

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
    summary["model"] = model_name

    orders = sorted(runs["order"].unique())

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

    summary.to_csv(output_dir / "results_summary.csv", index=False)
    save_table_png(summary, output_dir / "results_summary_table.png", title=f"{display_model} - Episode-level Metrics")

    diff_rows: list[dict] = []
    for order in orders:
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
        diff_df.to_csv(output_dir / "results_differences.csv", index=False)
        save_table_png(
            diff_df,
            output_dir / "results_differences_table.png",
            title=f"{display_model} - Free-form vs Reduced (tokens / turns)",
        )
    else:
        diff_df = pd.DataFrame()

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
        call_summary.to_csv(output_dir / "call_tokens_summary.csv", index=False)
        save_table_png(
            call_summary,
            output_dir / "call_tokens_summary_table.png",
            title=f"{display_model} - Per-call Token Diagnostics",
        )
    else:
        call_summary = pd.DataFrame()

    pivot_tokens = runs.pivot_table(index="order", columns="mode", values="tokens", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    pivot_tokens.plot(kind="bar", ax=ax)
    ax.set_ylabel("Average LLM tokens per episode")
    ax.set_title(f"{display_model} - Communication cost by order / mode")
    ax.set_xticklabels([_fmt_order(o) for o in pivot_tokens.index], rotation=15, ha="right", fontsize=9)
    ax.legend(title="mode", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_dir / "results_tokens_bar.png", dpi=200)
    plt.close(fig)

    pivot_turns = runs.pivot_table(index="order", columns="mode", values="turns", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    pivot_turns.plot(kind="bar", ax=ax)
    ax.set_ylabel("Avg communication turns / episode")
    ax.set_title(f"{display_model} - LLM calls by order / mode")
    ax.set_xticklabels([_fmt_order(o) for o in pivot_turns.index], rotation=15, ha="right", fontsize=9)
    ax.legend(title="mode", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_dir / "results_turns_bar.png", dpi=200)
    plt.close(fig)

    if not calls.empty and not calls[calls["tokens"] > 0].empty:
        positive_calls = calls[calls["tokens"] > 0]
        box_data = []
        box_labels = []
        for order in orders:
            for mode_key, mode_name in MODES.items():
                subset = positive_calls[(positive_calls["order"] == order) & (positive_calls["mode_key"] == mode_key)]
                if subset.empty:
                    continue
                box_data.append(subset["tokens"].values)
                box_labels.append(f"{order}\n{mode_name}")
        if box_data:
            fig, ax = plt.subplots(figsize=(9, 5.5))
            # matplotlib <3.6 uses 'labels' instead of 'tick_labels'
            ax.boxplot(box_data, labels=[_fmt_order(lbl) for lbl in box_labels], showfliers=False)
            ax.set_xticklabels([_fmt_order(lbl) for lbl in box_labels], rotation=15, ha="right", fontsize=9)
            ax.set_ylabel("Tokens per LLM call")
            ax.set_title("Distribution of tokens per API call")
            fig.tight_layout()
            fig.savefig(output_dir / "call_tokens_boxplot.png", dpi=200)
            plt.close(fig)

    success_fig = plt.figure(figsize=(8.5, 5.4))
    overlap_tracker = {}
    fig, ax = plt.subplots()
    fig.set_size_inches(8, 5)
    ax.margins(x=0.10, y=0.15)
    ax = success_fig.add_subplot(1, 1, 1)
    ax.errorbar(
        summary["tokens_mean"],
        summary["success_rate"],
                xerr=[
                    summary["tokens_mean"] - summary["tokens_ci_low"],
                    summary["tokens_ci_high"] - summary["tokens_mean"],
                ],
        fmt="o",
        capsize=4,
    )
    success_texts = []
    xmin = float(summary["tokens_ci_low"].min())
    xmax = float(summary["tokens_ci_high"].max())
    ymin = float(summary["success_rate"].min())
    ymax = float(summary["success_rate"].max())
    for _, row in summary.iterrows():
        manual = MANUAL_SUCCESS_POS.get((display_model, row["order"], row["mode"]))
        if manual:
            x_text, y_text, ha, va = manual
            success_texts.append(
                ax.annotate(
                    f"{_fmt_order(row['order'])}\n{row['mode']}",
                    (row["tokens_mean"], row["success_rate"]),
                    textcoords="data",
                    xytext=(x_text, y_text),
                    ha=ha,
                    va=va,
                    fontsize=7,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.7", alpha=0.9),
                    arrowprops=dict(arrowstyle="-", lw=0.8, color="0.35"),
                    clip_on=False,
                )
            )
        else:
            success_texts.append(
                ax.annotate(
                    f"{_fmt_order(row['order'])}\n{row['mode']}",
                    (row["tokens_mean"], row["success_rate"]),
                    textcoords="offset points",
                    xytext=(12, 8),
                    ha="left",
                    va="bottom",
                    fontsize=7,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.7", alpha=0.9),
                    arrowprops=dict(arrowstyle="-", lw=0.8, color="0.35"),
                    clip_on=False,
                )
            )
    ax.set_xlabel("Average LLM tokens per episode")
    ax.set_ylabel("Success rate")
    top_pad = 0.18
    ax.set_ylim(0, min(1.2, 1.0 + top_pad))
    ax.set_title(f"Tokens vs success ({display_model})", pad=16)
    pad = 0.08 * (xmax - xmin if xmax > xmin else 1.0)
    ax.set_xlim(xmin - pad, xmax + pad)
    ax.margins(x=0.10, y=0.18)
    _adjust_labels(success_texts, ax)
    _clamp_annotations(ax, success_texts, pad_px=10.0)
    _spread_annotations(ax, success_texts, list(zip(summary["tokens_mean"], summary["success_rate"])), pad_px=10.0, step_px=5.0, max_iter=200, max_rad_px=110.0)
    plt.tight_layout()
    success_fig.savefig(output_dir / "tradeoff_tokens_success.png", dpi=200)
    plt.close(success_fig)

    if {"timesteps_mean", "timesteps_ci_low"}.issubset(summary.columns):
        ts_fig = plt.figure(figsize=(8.5, 5.4))
        ts_overlap = {}
        ts_texts = []
        tx_min = float(summary["tokens_ci_low"].min())
        tx_max = float(summary["tokens_ci_high"].max())
        ty_min = float(summary["timesteps_mean"].min())
        ty_max = float(summary["timesteps_mean"].max())
        fig, ax = plt.subplots()
        fig.set_size_inches(8, 5)
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
            manual = MANUAL_TIMESTEPS_POS.get((display_model, row["order"], row["mode"]))
            if manual:
                x_text, y_text, ha, va = manual
                ts_texts.append(
                    ax.annotate(
                        f"{_fmt_order(row['order'])}\n{row['mode']}",
                        (row["tokens_mean"], row["timesteps_mean"]),
                        textcoords="data",
                        xytext=(x_text, y_text),
                        ha=ha,
                        va=va,
                        fontsize=7,
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.7", alpha=0.9),
                        arrowprops=dict(arrowstyle="-", lw=0.8, color="0.35"),
                        clip_on=False,
                        zorder=5,
                    )
                )
            else:
                ts_texts.append(
                    ax.annotate(
                        f"{_fmt_order(row['order'])}\n{row['mode']}",
                        (row["tokens_mean"], row["timesteps_mean"]),
                        textcoords="offset points",
                        xytext=(10, 6),
                        ha="left",
                        va="center",
                        fontsize=7,
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.7", alpha=0.9),
                        arrowprops=dict(arrowstyle="-", lw=0.8, color="0.35"),
                        clip_on=False,
                        zorder=5,
                    )
                )

        ax.set_title(f"Tokens vs timesteps ({display_model})", pad=16)
        _adjust_labels(ts_texts, ax)
        _clamp_annotations(ax, ts_texts, pad_px=10.0)
        _spread_annotations(ax, ts_texts, list(zip(summary["tokens_mean"], summary["timesteps_mean"])), pad_px=10.0, step_px=5.0, max_iter=200, max_rad_px=110.0)
        plt.tight_layout()
        ts_fig.savefig(output_dir / "tradeoff_tokens_timesteps.png", dpi=200)
        plt.close(ts_fig)

    plt.close("all")

    return runs, summary, call_summary

if __name__ == "__main__":
    args = parse_args()
    models: list[str]
    if args.gpt_models:
        models = [m.strip() for m in args.gpt_models.split(",") if m.strip()]
    else:
        models = [args.gpt_model]

    output_root = args.output_root or Path(__file__).resolve().parent
    all_runs: list[pd.DataFrame] = []
    all_summary: list[pd.DataFrame] = []
    for m in models:
        runs_df, summary_df, _ = summarize_model(m, args.data_root, output_root)
        all_runs.append(runs_df)
        all_summary.append(summary_df)

    if len(all_summary) > 1:
        def _plot_multi_model_scatter(summary: pd.DataFrame, x_col: str, y_col: str, x_label: str, y_label: str, filename: str, title: str) -> None:
            out_dir = output_root / "multi_model_comparison"
            out_dir.mkdir(exist_ok=True, parents=True)
            fig, ax = plt.subplots(figsize=(9, 5.5))
            colors = plt.cm.tab10(np.linspace(0, 1, summary["model"].nunique()))
            model_to_color = {model: colors[i] for i, model in enumerate(sorted(summary["model"].unique()))}
            texts = []

            for _, row in summary.iterrows():
                color = model_to_color[row["model"]]
                xerr_low = row[x_col] - row.get(f"{x_col.replace('_mean', '')}_ci_low", row[x_col])
                xerr_high = row.get(f"{x_col.replace('_mean', '')}_ci_high", row[x_col]) - row[x_col]
                yerr_low = row[y_col] - row.get(f"{y_col.replace('_mean', '')}_ci_low", row[y_col])
                yerr_high = row.get(f"{y_col.replace('_mean', '')}_ci_high", row[y_col]) - row[y_col]
                ax.errorbar(
                    row[x_col],
                    row[y_col],
                    xerr=[[xerr_low], [xerr_high]],
                    yerr=[[yerr_low], [yerr_high]] if not pd.isna(yerr_low) and not pd.isna(yerr_high) else None,
                    fmt="o",
                    capsize=3,
                    color=color,
                    alpha=0.9,
                )
                texts.append(
                    ax.annotate(
                        f"{_fmt_order(row['order'])}\n{row['mode']}\n{get_model_display_name(row['model'])}",
                        (row[x_col], row[y_col]),
                        textcoords="offset points",
                        xytext=(10, 8),
                        ha="left",
                        va="bottom",
                        fontsize=7,
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.7", alpha=0.9),
                        arrowprops=dict(arrowstyle="-", lw=0.8, color="0.35"),
                        clip_on=False,
                    )
                )

            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.set_title(title, pad=14)
            handles = [plt.Line2D([0], [0], marker="o", color="w", label=model, markerfacecolor=c, markersize=7) for model, c in model_to_color.items()]
            ax.legend(handles=handles, title="model", fontsize=8)
            _adjust_labels(texts, ax)
            fig.tight_layout()
            fig.savefig(out_dir / filename, dpi=200)
            plt.close(fig)

        combined_runs = pd.concat(all_runs, ignore_index=True)
        combined_summary = pd.concat(all_summary, ignore_index=True)

        out_dir = output_root / "multi_model_comparison"
        out_dir.mkdir(exist_ok=True, parents=True)
        combined_runs.to_csv(out_dir / "combined_runs.csv", index=False)
        combined_summary.to_csv(out_dir / "combined_summary.csv", index=False)
        save_table_png(
            combined_summary,
            out_dir / "combined_summary_table.png",
            title="Per-order metrics across models",
        )

        if {"tokens_mean", "success_rate"}.issubset(combined_summary.columns):
            _plot_multi_model_scatter(
                combined_summary,
                x_col="tokens_mean",
                y_col="success_rate",
                x_label="Average LLM tokens per episode",
                y_label="Success rate",
                filename="multi_tradeoff_tokens_success.png",
                title="Tokens vs success (all models)",
            )

        if {"tokens_mean", "timesteps_mean"}.issubset(combined_summary.columns):
            _plot_multi_model_scatter(
                combined_summary,
                x_col="tokens_mean",
                y_col="timesteps_mean",
                x_label="Average LLM tokens per episode",
                y_label="Avg timesteps per episode",
                filename="multi_tradeoff_tokens_timesteps.png",
                title="Tokens vs timesteps (all models)",
            )
