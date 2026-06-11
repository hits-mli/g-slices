import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import ticker as mticker
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from execute.evaluate_checkpoints_cross_frequency import (
    _apply_dataset_param_overrides,
    _build_paper_plot_panel_data,
    _evaluate_checkpoint_on_dataset,
)
from execute.evaluate_grid_generalisation_irregular import (
    RunSpec,
    _discover_runs,
    _make_regular_eval_dataset_params,
)
from execute.inference_efficiency_gluonts import (
    BenchmarkSpec,
    _build_model,
    _read_yaml,
    _resolve_device,
)
from gslice.dataset import get_dataset_name_from_params


PLOT_FONT_RC = {
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True,
}

plt.rcParams.update(PLOT_FONT_RC)


def _set_plot_theme() -> None:
    plt.rcParams.update(PLOT_FONT_RC)
    plt.rcParams.update(
        {
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def _canonical_k(value: float | int) -> int | float:
    numeric = float(value)
    if math.isclose(numeric, round(numeric), rel_tol=0.0, abs_tol=1e-9):
        return int(round(numeric))
    return numeric


def _format_k(value: float | int) -> str:
    canonical = _canonical_k(value)
    return str(canonical)


def _parse_k_list(values: list[str]) -> list[int | float]:
    parsed: list[int | float] = []
    for raw in values:
        for item in str(raw).split(","):
            stripped = item.strip()
            if not stripped:
                continue
            parsed.append(_canonical_k(float(stripped)))
    if not parsed:
        raise ValueError("At least one k value must be provided.")
    return parsed


def _display_model_name(model_key: str) -> str:
    return "G-SLiCE" if model_key == "slice" else "TSFlow"


def _to_mathtext_words(text: str) -> str:
    escaped = text.replace(" ", r"\ ")
    return rf"$\mathrm{{{escaped}}}$"


def _resolve_dataset_family(runs: list[RunSpec], requested_family: str | None) -> str:
    families = sorted({run.dataset_family for run in runs})
    if requested_family is not None:
        if requested_family not in families:
            raise ValueError(
                f"dataset_family={requested_family!r} not found. Available: {families}"
            )
        return requested_family
    if len(families) != 1:
        raise ValueError(
            "Multiple dataset families were discovered. Pass --dataset_family explicitly. "
            f"Available: {families}"
        )
    return families[0]


def _seed_rank(seed: int | None) -> int:
    return int(seed) if seed is not None else 10**12


def _select_lowest_seed_runs(
    runs: list[RunSpec],
    *,
    ks: list[int | float],
    dataset_family: str,
) -> dict[tuple[str, int | float], RunSpec]:
    requested = {_canonical_k(k) for k in ks}
    selected: dict[tuple[str, int | float], RunSpec] = {}

    for run in runs:
        if run.dataset_family != dataset_family:
            continue
        canonical_k = _canonical_k(run.gamma_k)
        if canonical_k not in requested:
            continue
        key = (run.model_type, canonical_k)
        current = selected.get(key)
        if current is None or _seed_rank(run.train_seed) < _seed_rank(current.train_seed):
            selected[key] = run

    missing = [
        (model_key, k)
        for model_key in ("slice", "tsflow")
        for k in ks
        if (model_key, _canonical_k(k)) not in selected
    ]
    if missing:
        formatted = ", ".join(f"{model}/k={_format_k(k)}" for model, k in missing)
        raise ValueError(f"Missing checkpoints for: {formatted}")

    return selected


def _make_spec(run: RunSpec) -> BenchmarkSpec:
    effective_model_type = "tsflow" if run.model_type == "slice" else run.model_type
    return BenchmarkSpec(
        checkpoint_path=str(run.checkpoint_path),
        config_path=str(run.config_path),
        model_type=effective_model_type,
        label=f"{run.model_type}_k{_format_k(run.gamma_k)}_s{run.train_seed}",
    )


def _panel_label(eval_mode: str, train_k: int | float) -> str:
    if eval_mode == "iid":
        return rf"$k_{{\mathrm{{test}}}}={_format_k(train_k)}$"
    if eval_mode == "regular":
        return r"$k_{\mathrm{test}} \to \infty$"
    raise ValueError(f"Unknown eval_mode: {eval_mode!r}")


def _build_publication_panel_data(payload, *, series_index: int):
    if series_index >= len(payload.forecasts) or series_index >= len(payload.tss):
        return None

    target_series = payload.tss[series_index]
    irregular_time_grids = getattr(target_series, "attrs", {}).get("irregular_time_grids")
    if irregular_time_grids is None:
        return _build_paper_plot_panel_data(payload, series_index=series_index)

    forecast = payload.forecasts[series_index]
    forecast_samples = forecast.samples
    if forecast_samples.ndim == 1:
        forecast_samples = forecast_samples[None, :]
    if forecast_samples.ndim == 3 and forecast_samples.shape[-1] == 1:
        forecast_samples = forecast_samples[..., 0]

    context_time = irregular_time_grids["context_time"]
    future_time = irregular_time_grids["future_time"]
    origin_start = irregular_time_grids.get("origin_start")
    target_values = target_series.iloc[:, 0].to_numpy(dtype=float)
    context_length = int(len(context_time))
    future_length = int(len(future_time))
    if target_values.shape[0] < context_length + future_length:
        return None

    visible_values = target_values[-(context_length + future_length) :]
    forecast_mean = forecast_samples.mean(axis=0)[:future_length]
    forecast_lo = np.quantile(forecast_samples, 0.025, axis=0)[:future_length]
    forecast_hi = np.quantile(forecast_samples, 0.975, axis=0)[:future_length]

    if hasattr(origin_start, "to_timestamp"):
        origin_timestamp = origin_start.to_timestamp()
    else:
        origin_timestamp = pd.Timestamp(origin_start)
    context_x = origin_timestamp + pd.to_timedelta(context_time, unit="h")
    future_x = origin_timestamp + pd.to_timedelta(future_time, unit="h")

    return {
        "visible_index": np.concatenate([context_x.to_numpy(), future_x.to_numpy()], axis=0),
        "visible_values": visible_values,
        "forecast_x": future_x.to_numpy(),
        "forecast_mean": forecast_mean,
        "forecast_lo": forecast_lo,
        "forecast_hi": forecast_hi,
        "forecast_start": future_x[0] if len(future_x) else origin_timestamp,
        "uses_datetime": True,
    }


def _format_global_date_text(panel_data: list[tuple[str, dict[str, Any]]]) -> str | None:
    all_points: list[pd.Timestamp] = []
    for _, panel in panel_data:
        visible = panel["visible_index"]
        if len(visible) == 0:
            continue
        all_points.append(pd.Timestamp(visible[0]))
        all_points.append(pd.Timestamp(visible[-1]))
    if not all_points:
        return None
    start = min(all_points)
    end = max(all_points)
    if start.date() == end.date():
        return start.strftime("%b %-d, %Y")
    if start.year == end.year and start.month == end.month:
        return f"{start.strftime('%b %-d')} to {end.strftime('%-d, %Y')}"
    return f"{start.strftime('%b %-d, %Y')} to {end.strftime('%b %-d, %Y')}"


def _to_plot_x_coords(values: Any) -> np.ndarray:
    arr = np.asarray(values)
    if arr.size == 0:
        return np.asarray([], dtype=float)
    if np.issubdtype(arr.dtype, np.datetime64):
        return mdates.date2num(pd.to_datetime(arr).to_pydatetime())
    if arr.dtype == object and isinstance(arr.flat[0], pd.Timestamp):
        return mdates.date2num(pd.to_datetime(arr).to_pydatetime())
    return arr.astype(float)


def _forecast_boundary_and_interval_bridge(
    panel: dict[str, Any],
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray] | None:
    x_visible = _to_plot_x_coords(panel["visible_index"])
    x_forecast = _to_plot_x_coords(panel["forecast_x"])
    visible_values = np.asarray(panel["visible_values"], dtype=float)
    forecast_lo = np.asarray(panel["forecast_lo"], dtype=float)
    forecast_hi = np.asarray(panel["forecast_hi"], dtype=float)
    if x_visible.size == 0 or x_forecast.size == 0 or forecast_lo.size == 0 or forecast_hi.size == 0:
        return None

    observed_count = visible_values.shape[0] - x_forecast.shape[0]
    if observed_count <= 0:
        return None

    last_observed_idx = observed_count - 1
    last_observed_x = float(x_visible[last_observed_idx])
    last_observed_y = float(visible_values[last_observed_idx])
    first_forecast_x = float(x_forecast[0])
    boundary_x = 0.5 * (last_observed_x + first_forecast_x)
    bridge_x = np.asarray([last_observed_x, first_forecast_x], dtype=float)
    bridge_lo = np.asarray([last_observed_y, float(forecast_lo[0])], dtype=float)
    bridge_hi = np.asarray([last_observed_y, float(forecast_hi[0])], dtype=float)
    return boundary_x, bridge_x, bridge_lo, bridge_hi


def _choose_panel_label_position(panel: dict[str, Any]) -> tuple[float, float, str]:
    x_visible = _to_plot_x_coords(panel["visible_index"])
    x_forecast = _to_plot_x_coords(panel["forecast_x"])
    y_visible = np.asarray(panel["visible_values"], dtype=float)
    y_forecast_mean = np.asarray(panel["forecast_mean"], dtype=float)
    y_forecast_lo = np.asarray(panel["forecast_lo"], dtype=float)
    y_forecast_hi = np.asarray(panel["forecast_hi"], dtype=float)

    x_all = np.concatenate([x_visible, x_forecast, x_forecast, x_forecast], axis=0)
    y_all = np.concatenate([y_visible, y_forecast_mean, y_forecast_lo, y_forecast_hi], axis=0)
    finite_mask = np.isfinite(x_all) & np.isfinite(y_all)
    if not np.any(finite_mask):
        return 0.015, 0.07, "bottom"

    x_all = x_all[finite_mask]
    y_all = y_all[finite_mask]
    x_min, x_max = float(np.min(x_all)), float(np.max(x_all))
    y_min, y_max = float(np.min(y_all)), float(np.max(y_all))
    if np.isclose(x_min, x_max) or np.isclose(y_min, y_max):
        return 0.015, 0.07, "bottom"

    x_norm = (x_all - x_min) / (x_max - x_min)
    y_norm = (y_all - y_min) / (y_max - y_min)
    lower_left_overlap = np.any((x_norm <= 0.34) & (y_norm <= 0.30))
    upper_left_overlap = np.any((x_norm <= 0.34) & (y_norm >= 0.70))
    if lower_left_overlap and not upper_left_overlap:
        return 0.015, 0.93, "top"
    return 0.015, 0.07, "bottom"


def _panel_xy_values(panel: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    x_visible = _to_plot_x_coords(panel["visible_index"])
    x_forecast = _to_plot_x_coords(panel["forecast_x"])
    y_visible = np.asarray(panel["visible_values"], dtype=float)
    y_forecast_mean = np.asarray(panel["forecast_mean"], dtype=float)
    y_forecast_lo = np.asarray(panel["forecast_lo"], dtype=float)
    y_forecast_hi = np.asarray(panel["forecast_hi"], dtype=float)

    x_all = np.concatenate([x_visible, x_forecast, x_forecast, x_forecast], axis=0)
    y_all = np.concatenate([y_visible, y_forecast_mean, y_forecast_lo, y_forecast_hi], axis=0)
    finite_mask = np.isfinite(x_all) & np.isfinite(y_all)
    return x_all[finite_mask], y_all[finite_mask]


def _trajectory_marker_sizes(num_points: int, base_center_size: float) -> tuple[float, float, float]:
    if num_points >= 96:
        scale = 0.52
    elif num_points >= 64:
        scale = 0.64
    elif num_points >= 40:
        scale = 0.78
    else:
        scale = 1.0
    center_size = float(base_center_size) * scale
    halo_size = center_size + max(10.0, center_size * 0.65)
    linewidth_scale = max(0.72, scale)
    return center_size, halo_size, linewidth_scale


def _draw_visible_trajectory(
    ax,
    x_values: Any,
    y_values: Any,
    *,
    color: str,
    linewidth: float,
    marker_center_size: float,
    zorder: float,
) -> None:
    y_arr = np.asarray(y_values, dtype=float)
    center_size, halo_size, linewidth_scale = _trajectory_marker_sizes(
        int(y_arr.shape[0]),
        marker_center_size,
    )
    ax.plot(
        x_values,
        y_values,
        color=color,
        linewidth=float(linewidth) * linewidth_scale,
        solid_capstyle="round",
        zorder=zorder,
    )
    ax.scatter(
        x_values,
        y_values,
        s=halo_size,
        color="#111111",
        edgecolors="none",
        alpha=0.9,
        zorder=zorder + 0.25,
    )
    ax.scatter(
        x_values,
        y_values,
        s=center_size,
        facecolors="white",
        edgecolors=color,
        linewidths=max(0.8, 1.25 * linewidth_scale),
        zorder=zorder + 0.4,
    )


def _expand_y_limits_for_lower_left_label(
    ax,
    panel: dict[str, Any],
    *,
    label_x_fraction: float = 0.38,
    label_top_fraction: float = 0.34,
) -> None:
    if ax.get_yscale() != "linear":
        return
    x_all, y_all = _panel_xy_values(panel)
    if x_all.size == 0 or y_all.size == 0:
        return

    x_min, x_max = float(np.min(x_all)), float(np.max(x_all))
    if np.isclose(x_min, x_max):
        return
    x_norm = (x_all - x_min) / (x_max - x_min)
    left_values = y_all[x_norm <= label_x_fraction]
    if left_values.size == 0:
        return

    y_min, y_max = ax.get_ylim()
    if y_min > y_max:
        y_min, y_max = y_max, y_min
    if np.isclose(y_min, y_max):
        return

    left_min = float(np.min(left_values))
    left_norm = (left_min - y_min) / (y_max - y_min)
    if left_norm >= label_top_fraction:
        return

    desired_min = (left_min - label_top_fraction * y_max) / (1.0 - label_top_fraction)
    pad = 0.04 * (y_max - desired_min)
    ax.set_ylim(desired_min - pad, y_max)


def _add_panel_label(
    ax,
    panel: dict[str, Any],
    panel_label: str,
    *,
    position: str,
    fontsize: float,
) -> None:
    if position == "lower_left":
        _expand_y_limits_for_lower_left_label(ax, panel)
        label_x, label_y, label_va = 0.015, 0.07, "bottom"
    elif position == "auto":
        label_x, label_y, label_va = _choose_panel_label_position(panel)
    else:
        raise ValueError(f"Unknown panel label position: {position!r}")
    ax.text(
        label_x,
        label_y,
        panel_label,
        transform=ax.transAxes,
        ha="left",
        va=label_va,
        fontsize=fontsize,
        fontweight="semibold",
        color="black",
    )


def _build_publication_plot_panels(
    *,
    train_k: int | float,
    iid_payload,
    regular_payload,
) -> list[tuple[str, dict[str, Any]]]:
    del train_k
    ordered_payloads = [
        ("iid", iid_payload),
        ("regular", regular_payload),
    ]
    series_index = ordered_payloads[0][1].selected_indices[0] if ordered_payloads[0][1].selected_indices else 0
    panel_data: list[tuple[str, dict[str, Any]]] = []
    for eval_mode, payload in ordered_payloads:
        panel = _build_publication_panel_data(payload, series_index=series_index)
        if panel is not None:
            panel_data.append((eval_mode, panel))
    return panel_data


def _make_publication_legend_handles() -> list[Any]:
    ground_truth_color = "#3778bf"
    forecast_color = "#cc3366"
    interval_color = "#f3a9bc"
    return [
        Line2D([0], [0], color=ground_truth_color, linewidth=5.0, label=_to_mathtext_words("Ground Truth")),
        Line2D([0], [0], color=forecast_color, linewidth=5.0, label=_to_mathtext_words("Forecast")),
        Patch(facecolor=interval_color, edgecolor="none", alpha=0.25, label=r"$95\%\ \mathrm{CI}$"),
    ]


def _make_overlay_comparison_legend_handles() -> list[Any]:
    return [
        Line2D([0], [0], color="#222222", linewidth=4.0, label=_to_mathtext_words("Ground Truth")),
        Line2D([0], [0], color="#c43b4f", linewidth=4.5, label=_to_mathtext_words("G-SLiCE Mean")),
        Patch(facecolor="#c43b4f", edgecolor="none", alpha=0.18, label=_to_mathtext_words("G-SLiCE CI")),
        Line2D([0], [0], color="#2b6cb0", linewidth=4.5, label=_to_mathtext_words("TSFlow Mean")),
        Patch(facecolor="#2b6cb0", edgecolor="none", alpha=0.14, label=_to_mathtext_words("TSFlow CI")),
    ]


TOP_ROW_Y_LIMITS: tuple[float, float] = None # (-20.0, 40.0)
TOP_ROW_Y_TICKS = None #np.asarray([0.0, 0.5, 1.0], dtype=float)


def _draw_publication_panel(
    *,
    ax,
    eval_mode: str,
    panel: dict[str, Any],
    train_k: int | float,
    show_x_labels: bool,
    show_y_labels: bool,
    show_panel_label: bool = True,
    panel_label_position: str = "auto",
    y_ticks: np.ndarray | None = None,
    y_limits: tuple[float, float] | None = None,
    y_scale: str = "linear",
    symlog_linthresh: float | None = None,
):
    ground_truth_color = "#3778bf"
    forecast_color = "#cc3366"
    interval_color = "#f3a9bc"
    forecast_start_color = "#8a8a8a"

    ax.set_facecolor("white")
    _draw_visible_trajectory(
        ax,
        panel["visible_index"],
        panel["visible_values"],
        color=ground_truth_color,
        linewidth=8.0,
        marker_center_size=32.0,
        zorder=3.0,
    )
    ax.fill_between(
        panel["forecast_x"],
        panel["forecast_lo"],
        panel["forecast_hi"],
        color=interval_color,
        alpha=0.25,
        linewidth=0.0,
        zorder=1,
    )
    boundary_and_bridge = _forecast_boundary_and_interval_bridge(panel)
    if boundary_and_bridge is not None:
        boundary_x, bridge_x, bridge_lo, bridge_hi = boundary_and_bridge
        ax.fill_between(
            bridge_x,
            bridge_lo,
            bridge_hi,
            color=interval_color,
            alpha=0.25,
            linewidth=0.0,
            zorder=1,
        )
    else:
        boundary_x = panel["forecast_start"]
    _draw_visible_trajectory(
        ax,
        panel["forecast_x"],
        panel["forecast_mean"],
        color=forecast_color,
        linewidth=8.0,
        marker_center_size=28.0,
        zorder=4.5,
    )
    ax.axvline(
        boundary_x,
        color=forecast_start_color,
        linestyle="--",
        linewidth=5.0,
        alpha=0.5,
        zorder=2,
    )
    if y_scale == "symlog":
        ax.set_yscale("symlog", linthresh=1.0 if symlog_linthresh is None else symlog_linthresh)
    if y_limits is not None:
        ax.set_ylim(*y_limits)
    if show_panel_label:
        _add_panel_label(
            ax,
            panel,
            _panel_label(eval_mode, train_k),
            position=panel_label_position,
            fontsize=30,
        )
    if y_ticks is None:
        ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5 if y_scale == "symlog" else 4, symmetric=(y_scale == "symlog")))
    else:
        ax.yaxis.set_major_locator(mticker.FixedLocator(y_ticks))
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda value, _: rf"$\mathdefault{{{value:g}}}$")
    )
    ax.tick_params(
        axis="y",
        which="major",
        left=True,
        right=False,
        direction="out",
        labelleft=show_y_labels,
        labelsize=12.5,
        length=9.0,
        width=2.0,
        colors="black",
    )
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(2.5)
    if show_x_labels:
        ax.tick_params(
            axis="x",
            which="major",
            bottom=True,
            top=False,
            direction="out",
            labelbottom=True,
            labelsize=12.5,
            length=9.0,
            width=2.0,
            colors="black",
        )
    else:
        ax.tick_params(axis="x", which="major", bottom=False, top=False, labelbottom=False)


def _symmetric_symlog_limits(panels: list[dict[str, Any]]) -> tuple[float, float, float]:
    values: list[np.ndarray] = []
    for panel in panels:
        values.extend(
            [
                np.asarray(panel["visible_values"], dtype=float),
                np.asarray(panel["forecast_mean"], dtype=float),
                np.asarray(panel["forecast_lo"], dtype=float),
                np.asarray(panel["forecast_hi"], dtype=float),
            ]
        )
    finite_values = [arr[np.isfinite(arr)] for arr in values if arr.size > 0]
    if not finite_values:
        return -1.0, 1.0, 1.0
    finite = np.concatenate(finite_values, axis=0)
    if finite.size == 0:
        return -1.0, 1.0, 1.0
    max_abs = float(np.max(np.abs(finite)))
    if max_abs <= 0.0:
        max_abs = 1.0
    limit = max_abs * 1.08
    linthresh = max(limit * 0.02, 1e-3)
    return -limit, limit, linthresh


def _draw_overlay_comparison_panel(
    *,
    ax,
    eval_mode: str,
    train_k: int | float,
    slice_panel: dict[str, Any],
    tsflow_panel: dict[str, Any],
    show_x_labels: bool,
    show_y_labels: bool,
) -> None:
    ax.set_facecolor("white")
    y_min, y_max, linthresh = _symmetric_symlog_limits([slice_panel, tsflow_panel])

    _draw_visible_trajectory(
        ax,
        slice_panel["visible_index"],
        slice_panel["visible_values"],
        color="#222222",
        linewidth=5.0,
        marker_center_size=22.0,
        zorder=4.0,
    )

    for panel, color, alpha, zorder in [
        (slice_panel, "#c43b4f", 0.18, 2),
        (tsflow_panel, "#2b6cb0", 0.14, 1),
    ]:
        ax.fill_between(
            panel["forecast_x"],
            panel["forecast_lo"],
            panel["forecast_hi"],
            color=color,
            alpha=alpha,
            linewidth=0.0,
            zorder=zorder,
        )
        boundary_and_bridge = _forecast_boundary_and_interval_bridge(panel)
        if boundary_and_bridge is not None:
            _, bridge_x, bridge_lo, bridge_hi = boundary_and_bridge
            ax.fill_between(
                bridge_x,
                bridge_lo,
                bridge_hi,
                color=color,
                alpha=alpha,
                linewidth=0.0,
                zorder=zorder,
            )
        _draw_visible_trajectory(
            ax,
            panel["forecast_x"],
            panel["forecast_mean"],
            color=color,
            linewidth=4.8,
            marker_center_size=18.0,
            zorder=zorder + 4,
        )

    boundary_and_bridge = _forecast_boundary_and_interval_bridge(slice_panel)
    boundary_x = boundary_and_bridge[0] if boundary_and_bridge is not None else slice_panel["forecast_start"]
    ax.axvline(boundary_x, color="#8a8a8a", linestyle="--", linewidth=3.5, alpha=0.5, zorder=3)
    ax.text(
        0.015,
        0.93,
        _panel_label(eval_mode, train_k),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=28,
        fontweight="semibold",
        color="black",
    )

    ax.set_yscale("symlog", linthresh=linthresh)
    ax.set_ylim(y_min, y_max)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5, symmetric=True))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda value, _: rf"$\mathdefault{{{value:g}}}$"))
    ax.tick_params(
        axis="y",
        which="major",
        left=True,
        right=False,
        direction="out",
        labelleft=show_y_labels,
        labelsize=12.5,
        length=9.0,
        width=2.0,
        colors="black",
    )
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(2.5)
    if show_x_labels:
        ax.tick_params(
            axis="x",
            which="major",
            bottom=True,
            top=False,
            direction="out",
            labelbottom=True,
            labelsize=12.5,
            length=9.0,
            width=2.0,
            colors="black",
        )
    else:
        ax.tick_params(axis="x", which="major", bottom=False, top=False, labelbottom=False)


def _render_publication_plot(
    *,
    model_name: str,
    train_k: int | float,
    iid_payload,
    regular_payload,
    same_top_y_ticks: bool,
):
    _set_plot_theme()
    panel_data = _build_publication_plot_panels(
        train_k=train_k,
        iid_payload=iid_payload,
        regular_payload=regular_payload,
    )
    if not panel_data:
        return None

    fig, axes = plt.subplots(
        len(panel_data),
        1,
        figsize=(5.25, max(4.8, 2.55 * len(panel_data))),
        sharex=True,
        facecolor="white",
    )
    if len(panel_data) == 1:
        axes = [axes]

    for idx, (ax, (eval_mode, panel)) in enumerate(zip(axes, panel_data)):
        _draw_publication_panel(
            ax=ax,
            eval_mode=eval_mode,
            panel=panel,
            train_k=train_k,
            show_x_labels=(idx == len(panel_data) - 1),
            show_y_labels=True,
        )

    if all(panel["uses_datetime"] for _, panel in panel_data):
        locator = mdates.AutoDateLocator(minticks=3, maxticks=4)
        axes[-1].xaxis.set_major_locator(locator)
        axes[-1].xaxis.set_major_formatter(
            mticker.FuncFormatter(
                lambda value, _: rf"$\mathdefault{{{mdates.num2date(value).strftime('%H:%M')}}}$"
            )
        )
        # global_date_text = _format_global_date_text(panel_data)
        # if global_date_text:
        #     fig.text(
        #         0.985,
        #         0.045,
        #         global_date_text,
        #         ha="right",
        #         va="center",
        #         fontsize=12.5,
        #         color="black",
        #     )

    fig.legend(
        handles=_make_publication_legend_handles(),
        loc="upper center",
        bbox_to_anchor=(0.56, 0.995),
        ncol=3,
        frameon=False,
        fontsize=20,
        handlelength=1.0,
        handletextpad=0.7,
        columnspacing=1.7,
    )

    fig.subplots_adjust(left=0.12, right=0.99, bottom=0.11, top=0.86, hspace=0.16)
    return fig


def _render_publication_comparison_plot(
    *,
    train_k: int | float,
    slice_iid_payload,
    slice_regular_payload,
    tsflow_iid_payload,
    tsflow_regular_payload,
    same_top_y_ticks: bool,
    use_symlog: bool = False,
):
    _set_plot_theme()

    panels_by_model = {
        "slice": _build_publication_plot_panels(
            train_k=train_k,
            iid_payload=slice_iid_payload,
            regular_payload=slice_regular_payload,
        ),
        "tsflow": _build_publication_plot_panels(
            train_k=train_k,
            iid_payload=tsflow_iid_payload,
            regular_payload=tsflow_regular_payload,
        ),
    }
    if not panels_by_model["slice"] or not panels_by_model["tsflow"]:
        return None

    n_rows = max(len(panels_by_model["slice"]), len(panels_by_model["tsflow"]))
    fig, axes = plt.subplots(
        n_rows,
        2,
        figsize=(10.5, max(4.8, 2.55 * n_rows)),
        sharex=False,
        facecolor="white",
    )
    if n_rows == 1:
        axes = np.asarray([axes])

    model_columns = [
        ("tsflow", "TSFlow"),
        ("slice", "G-SLiCE"),
    ]
    if same_top_y_ticks:
        shared_top_y_limits = TOP_ROW_Y_LIMITS
        shared_top_y_ticks = TOP_ROW_Y_TICKS
    else:
        shared_top_y_limits = None
        shared_top_y_ticks = None
    row_symlog_limits: dict[str, tuple[float, float, float]] = {}
    if use_symlog:
        row_modes = {
            eval_mode
            for model_panels in panels_by_model.values()
            for eval_mode, _ in model_panels
        }
        for eval_mode in row_modes:
            row_panels = [
                panel
                for model_panels in panels_by_model.values()
                for mode, panel in model_panels
                if mode == eval_mode
            ]
            if row_panels:
                row_symlog_limits[eval_mode] = _symmetric_symlog_limits(row_panels)

    for col_idx, (model_key, model_title) in enumerate(model_columns):
        model_panels = panels_by_model[model_key]
        for row_idx in range(n_rows):
            ax = axes[row_idx, col_idx]
            if row_idx >= len(model_panels):
                ax.axis("off")
                continue
            eval_mode, panel = model_panels[row_idx]
            symlog_limits = row_symlog_limits.get(eval_mode)
            _draw_publication_panel(
                ax=ax,
                eval_mode=eval_mode,
                panel=panel,
                train_k=train_k,
                show_x_labels=(row_idx == n_rows - 1),
                show_y_labels=True,
                show_panel_label=True,
                panel_label_position="lower_left",
                y_ticks=(
                    None
                    if symlog_limits is not None
                    else (shared_top_y_ticks if shared_top_y_ticks is not None and row_idx == 0 else None)
                ),
                y_limits=(
                    symlog_limits[:2]
                    if symlog_limits is not None
                    else (shared_top_y_limits if shared_top_y_limits is not None and row_idx == 0 else None)
                ),
                y_scale=("symlog" if symlog_limits is not None else "linear"),
                symlog_linthresh=(symlog_limits[2] if symlog_limits is not None else None),
            )

    for col_idx, (model_key, _) in enumerate(model_columns):
        model_panels = panels_by_model[model_key]
        if all(panel["uses_datetime"] for _, panel in model_panels):
            locator = mdates.AutoDateLocator(minticks=3, maxticks=4)
            axes[n_rows - 1, col_idx].xaxis.set_major_locator(locator)
            axes[n_rows - 1, col_idx].xaxis.set_major_formatter(
                mticker.FuncFormatter(
                    lambda value, _: rf"$\mathdefault{{{mdates.num2date(value).strftime('%H:%M')}}}$"
                )
            )

    fig.legend(
        handles=_make_publication_legend_handles(),
        loc="upper center",
        bbox_to_anchor=(0.55, 0.995),
        ncol=3,
        frameon=False,
        fontsize=25,
        handlelength=2.0,
        handletextpad=0.7,
        columnspacing=1.7,
    )
    fig.text(0.29, 0.00, "TSFlow", ha="center", va="center", fontsize=30, color="black")
    fig.text(0.77, 0.00, "G-SLiCE", ha="center", va="center", fontsize=30, color="black")
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.12, top=0.84, hspace=0.16, wspace=0.18)
    return fig


def _render_overlay_publication_comparison_plot(
    *,
    train_k: int | float,
    slice_iid_payload,
    slice_regular_payload,
    tsflow_iid_payload,
    tsflow_regular_payload,
):
    _set_plot_theme()
    panels_by_model = {
        "slice": dict(
            _build_publication_plot_panels(
                train_k=train_k,
                iid_payload=slice_iid_payload,
                regular_payload=slice_regular_payload,
            )
        ),
        "tsflow": dict(
            _build_publication_plot_panels(
                train_k=train_k,
                iid_payload=tsflow_iid_payload,
                regular_payload=tsflow_regular_payload,
            )
        ),
    }
    row_modes = [mode for mode in ("iid", "regular") if mode in panels_by_model["slice"] and mode in panels_by_model["tsflow"]]
    if not row_modes:
        return None

    fig, axes = plt.subplots(
        len(row_modes),
        1,
        figsize=(6.4, max(4.8, 2.8 * len(row_modes))),
        sharex=True,
        facecolor="white",
    )
    if len(row_modes) == 1:
        axes = [axes]

    for row_idx, eval_mode in enumerate(row_modes):
        _draw_overlay_comparison_panel(
            ax=axes[row_idx],
            eval_mode=eval_mode,
            train_k=train_k,
            slice_panel=panels_by_model["slice"][eval_mode],
            tsflow_panel=panels_by_model["tsflow"][eval_mode],
            show_x_labels=(row_idx == len(row_modes) - 1),
            show_y_labels=True,
        )

    if all(
        panel["uses_datetime"]
        for mode in row_modes
        for panel in (panels_by_model["slice"][mode], panels_by_model["tsflow"][mode])
    ):
        locator = mdates.AutoDateLocator(minticks=3, maxticks=4)
        axes[-1].xaxis.set_major_locator(locator)
        axes[-1].xaxis.set_major_formatter(
            mticker.FuncFormatter(
                lambda value, _: rf"$\mathdefault{{{mdates.num2date(value).strftime('%H:%M')}}}$"
            )
        )

    fig.legend(
        handles=_make_overlay_comparison_legend_handles(),
        loc="upper center",
        bbox_to_anchor=(0.54, 0.995),
        ncol=3,
        frameon=False,
        fontsize=18,
        handlelength=1.4,
        handletextpad=0.55,
        columnspacing=1.1,
    )
    fig.subplots_adjust(left=0.12, right=0.99, bottom=0.11, top=0.82, hspace=0.16)
    return fig


def _evaluate_run(
    *,
    run: RunSpec,
    device,
    num_samples: int,
    eval_seed: int | None,
    max_eval_instances: int | None,
    batch_size: int | None,
) -> tuple[Any, Any, Any, Any]:
    spec = _make_spec(run)
    config = _apply_dataset_param_overrides(
        _read_yaml(spec.config_path),
        fine_to_coarse_eval_adapter_override=None,
    )
    dataset_params = dict(config.get("dataset_params", {}) or {})
    regular_eval_dataset_params = _make_regular_eval_dataset_params(dataset_params)

    model = _build_model(spec, config, device)
    try:
        iid_result, iid_payload = _evaluate_checkpoint_on_dataset(
            spec=spec,
            config=config,
            model=model,
            eval_dataset_name=get_dataset_name_from_params(dataset_params),
            eval_dataset_params=dataset_params,
            eval_freq=None,
            eval_prediction_length=None,
            num_samples=int(num_samples),
            eval_seed=eval_seed,
            plot_dir=None,
            max_eval_instances=max_eval_instances,
            batch_size_override=batch_size,
            regenerate=False,
            adapt_gp_to_eval_dataset=True,
        )
        regular_result, regular_payload = _evaluate_checkpoint_on_dataset(
            spec=spec,
            config=config,
            model=model,
            eval_dataset_name=get_dataset_name_from_params(regular_eval_dataset_params),
            eval_dataset_params=regular_eval_dataset_params,
            eval_freq=None,
            eval_prediction_length=None,
            num_samples=int(num_samples),
            eval_seed=eval_seed,
            plot_dir=None,
            max_eval_instances=max_eval_instances,
            batch_size_override=batch_size,
            regenerate=False,
            adapt_gp_to_eval_dataset=True,
        )
    finally:
        del model

    return iid_result, iid_payload, regular_result, regular_payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Load the lowest-seed irregular checkpoints for TSFlow and SLiCE, evaluate them "
            "on matching-k and regular-grid data, and save stacked publication plots."
        )
    )
    parser.add_argument("--results_root", type=str, required=True)
    parser.add_argument("--ks", nargs="+", required=True, help="List of k values, e.g. --ks 1 10 100")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--dataset_family", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--eval_seed", type=int, default=None)
    parser.add_argument("--max_eval_instances", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--checkpoint_variant", choices=("best", "last"), default="best")
    parser.add_argument(
        "--same_top_y_ticks",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use a shared y-axis scale on the top-row comparison panels.",
    )
    parser.add_argument(
        "--overlay_comparison_models_symlog",
        action="store_true",
        help="Use aligned symmetric symlog y-axes for the TSFlow/G-SLiCE comparison plots.",
    )
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    results_root = Path(args.results_root).resolve()
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir is not None
        else results_root / "lowest_seed_checkpoint_plots"
    )
    ks = _parse_k_list(args.ks)

    runs = _discover_runs(results_root, checkpoint_variant=args.checkpoint_variant)
    if not runs:
        raise ValueError(f"No irregular checkpoints discovered under {results_root}.")
    dataset_family = _resolve_dataset_family(runs, args.dataset_family)
    selected = _select_lowest_seed_runs(runs, ks=ks, dataset_family=dataset_family)

    summary_rows: list[dict[str, Any]] = []
    if args.dry_run:
        for model_key in ("slice", "tsflow"):
            for k in ks:
                run = selected[(model_key, _canonical_k(k))]
                print(
                    json.dumps(
                        {
                            "model": model_key,
                            "train_k": _format_k(k),
                            "train_seed": run.train_seed,
                            "checkpoint_path": str(run.checkpoint_path),
                            "config_path": str(run.config_path),
                        },
                        indent=2,
                    )
                )
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(args.device)
    evaluated_payloads: dict[tuple[str, int | float], dict[str, Any]] = {}

    for model_key in ("slice", "tsflow"):
        model_output_dir = output_dir / model_key
        model_output_dir.mkdir(parents=True, exist_ok=True)
        for k in ks:
            run = selected[(model_key, _canonical_k(k))]
            eval_seed = args.eval_seed if args.eval_seed is not None else run.train_seed
            iid_result, iid_payload, regular_result, regular_payload = _evaluate_run(
                run=run,
                device=device,
                num_samples=int(args.num_samples),
                eval_seed=eval_seed,
                max_eval_instances=args.max_eval_instances,
                batch_size=args.batch_size,
            )
            evaluated_payloads[(model_key, _canonical_k(k))] = {
                "iid_result": iid_result,
                "iid_payload": iid_payload,
                "regular_result": regular_result,
                "regular_payload": regular_payload,
                "run": run,
            }
            fig = _render_publication_plot(
                model_name=_display_model_name(model_key),
                train_k=k,
                iid_payload=iid_payload,
                regular_payload=regular_payload,
                same_top_y_ticks=args.same_top_y_ticks,
            )
            if fig is None:
                raise RuntimeError(f"Could not build plot for {model_key} at k={_format_k(k)}.")

            stem = f"{model_key}_k{_format_k(k)}_seed{run.train_seed}__iid_vs_regular"
            png_path = model_output_dir / f"{stem}.png"
            pdf_path = model_output_dir / f"{stem}.pdf"
            fig.savefig(png_path, dpi=600, bbox_inches="tight", facecolor=fig.get_facecolor())
            fig.savefig(pdf_path, bbox_inches="tight", facecolor=fig.get_facecolor())
            plt.close(fig)

            summary_rows.append(
                {
                    "model": model_key,
                    "model_display": _display_model_name(model_key),
                    "train_k": _canonical_k(k),
                    "train_seed": run.train_seed,
                    "checkpoint_path": str(run.checkpoint_path),
                    "config_path": str(run.config_path),
                    "iid_result": asdict(iid_result),
                    "regular_result": asdict(regular_result),
                    "plot_png": str(png_path),
                    "plot_pdf": str(pdf_path),
                }
            )

    comparison_output_dir = output_dir / "comparison"
    comparison_output_dir.mkdir(parents=True, exist_ok=True)
    for k in ks:
        canonical_k = _canonical_k(k)
        slice_eval = evaluated_payloads[("slice", canonical_k)]
        tsflow_eval = evaluated_payloads[("tsflow", canonical_k)]
        comparison_fig = _render_publication_comparison_plot(
            train_k=k,
            slice_iid_payload=slice_eval["iid_payload"],
            slice_regular_payload=slice_eval["regular_payload"],
            tsflow_iid_payload=tsflow_eval["iid_payload"],
            tsflow_regular_payload=tsflow_eval["regular_payload"],
            same_top_y_ticks=args.same_top_y_ticks,
            use_symlog=bool(args.overlay_comparison_models_symlog),
        )
        if comparison_fig is None:
            raise RuntimeError(f"Could not build comparison plot for k={_format_k(k)}.")
        comparison_stem = (
            f"comparison_k{_format_k(k)}__iid_vs_regular"
            f"{'__symlog' if args.overlay_comparison_models_symlog else ''}"
        )
        comparison_png = comparison_output_dir / f"{comparison_stem}.png"
        comparison_pdf = comparison_output_dir / f"{comparison_stem}.pdf"
        comparison_fig.savefig(comparison_png, dpi=600, bbox_inches="tight", facecolor=comparison_fig.get_facecolor())
        comparison_fig.savefig(comparison_pdf, bbox_inches="tight", facecolor=comparison_fig.get_facecolor())
        plt.close(comparison_fig)

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary_rows, indent=2))
    print(f"Saved plots and summary to {output_dir}")


if __name__ == "__main__":
    main()
