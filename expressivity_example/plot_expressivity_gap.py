import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from matplotlib import ticker as mticker
from matplotlib.lines import Line2D
from matplotlib.colors import to_rgb

from expressivity_example import quantify_expressivity_gap as hc

PLOT_FONT_RC = {
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True,
}

plt.rcParams.update(PLOT_FONT_RC)

INPUT_COLOR = "#265688"
GROUND_TRUTH_COLOR = "#1f1f1f"
SLICE_COLOR = "#015324"
DIAGONAL_COLOR = "#eec20e"
NONSELECTIVE_COLOR = "#ae1111"
SAMPLE_ALPHAS = [1.0, 0.80, 0.64, 0.52]


def _shade_color(base_color: str, sample_idx: int, total_samples: int) -> str:
    rgb = np.asarray(to_rgb(base_color), dtype=float)
    if total_samples <= 1:
        return base_color
    frac = sample_idx / float(total_samples - 1)
    lightness = 0.25 + 0.45 * frac
    shaded = rgb * (1.0 - lightness) + np.ones_like(rgb) * lightness
    return tuple(np.clip(shaded, 0.0, 1.0))


def compute_target(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=int)
    if z.ndim == 1:
        c = np.zeros_like(z, dtype=int)
        c[0] = z[0]
        for k in range(1, z.shape[0]):
            c[k] = z[k] * (1 - c[k - 1])
        return c

    c = np.zeros_like(z, dtype=int)
    c[:, 0] = z[:, 0]
    for k in range(1, z.shape[1]):
        c[:, k] = z[:, k] * (1 - c[:, k - 1])
    return c


def _plot_binary_trace(
    ax,
    y: np.ndarray,
    *,
    series_color: str,
    linestyle: str,
    alpha: float,
    zorder: float,
    x_offset: float,
    y_offset: float,
) -> None:
    n = len(y)
    x = np.arange(1, n + 1)
    x_plot = x + x_offset
    y_plot = y + y_offset
    ax.step(
        x_plot,
        y_plot,
        where="mid",
        color=series_color,
        linewidth=7.0,
        alpha=alpha,
        linestyle=linestyle,
        zorder=zorder,
    )
    ax.plot(
        x,
        y_plot,
        linestyle="None",
        marker="o",
        markersize=15.0,
        markerfacecolor=series_color,
        markeredgecolor="white",
        markeredgewidth=0.6,
        alpha=alpha,
        zorder=zorder + 2,
    )


def plot_binary_cell(
    ax,
    y: np.ndarray,
    series_color: str = INPUT_COLOR,
    linestyle: str = "-",
) -> None:
    y = np.asarray(y)
    if y.ndim == 1:
        y = y[None, :]
    for idx, row in enumerate(y):
        x_offset = -0.15 * idx
        y_offset = -0.10 * idx
        _plot_binary_trace(
            ax,
            row,
            series_color=_shade_color(series_color, idx, y.shape[0]),
            linestyle=linestyle,
            alpha=SAMPLE_ALPHAS[idx % len(SAMPLE_ALPHAS)],
            zorder=2 + idx,
            x_offset=x_offset,
            y_offset=y_offset,
        )

    n = y.shape[1]
    x = np.arange(1, n + 1)
    ax.set_xlim(0.5, n + 0.5)
    ax.set_ylim(-0.0 - 0.2 * (y.shape[0] - 1), 1.12)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_xticks(x.tolist())
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(
            lambda value, _: (
                rf"$\mathdefault{{{int(value)}}}$"
                if float(value).is_integer() and 1 <= value <= n
                else ""
            )
        )
    )
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda value, _: rf"$\mathdefault{{{value:g}}}$")
    )
    ax.grid(True, axis="both", color="#d7dce2", linewidth=0.65, alpha=0.75)
    ax.set_facecolor("white")
    ax.tick_params(
        axis="both",
        which="major",
        direction="out",
        labelsize=12.5,
        length=9.0,
        width=2.0,
        colors="black",
        bottom=True,
        top=False,
        left=True,
        right=False,
    )
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(2.5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot the hard-core expressivity example using the real training loop."
    )
    parser.add_argument("--seq-lengths", type=int, nargs="+", default=[8, 128])
    parser.add_argument("--dense-widths", type=int, nargs="+", default=[4])
    parser.add_argument("--diag-widths", type=int, nargs="+", default=[4])
    parser.add_argument("--p", type=float, default=0.5)
    parser.add_argument("--train-size", type=int, default=1000)
    parser.add_argument("--val-size", type=int, default=200)
    parser.add_argument("--test-size", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--plot-seed", type=int, default=None)
    parser.add_argument("--num-samples", type=int, default=2)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--train-loss", type=str, default="mse")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument(
        "--early-stop-on-perfect",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    return parser.parse_args()


def _train_reference_models(
    args: argparse.Namespace,
) -> tuple[dict[str, torch.nn.Module], int, int, torch.device]:
    train_n = max(args.seq_lengths)
    plot_seed = args.seeds[0] if args.plot_seed is None else args.plot_seed
    if plot_seed not in args.seeds:
        raise ValueError(f"plot_seed={plot_seed} is not part of seeds={args.seeds}")

    device = hc.resolve_device(args.device)
    dense_width = min(args.dense_widths)
    diag_width = min(args.diag_widths)

    selected_names = {
        "DenseSSMWidth2",
        f"dense_w{dense_width}",
        f"diag_w{diag_width}",
    }
    model_specs = hc.build_model_specs(args.dense_widths, args.diag_widths)
    data_seed = 10007 * (plot_seed + 1) + 997 * train_n
    splits = hc.build_random_splits(
        n=train_n,
        p=args.p,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        seed=data_seed,
    )

    trained_models: dict[str, torch.nn.Module] = {}
    for model_idx, (model_name, model_factory) in enumerate(model_specs):
        if model_name not in selected_names:
            continue
        run_seed = 20011 * (plot_seed + 1) + 131 * train_n + model_idx
        print(f"Training model={model_name} n={train_n} seed={plot_seed}")
        _, trained_model = hc.train_one_run(
            model_factory=model_factory,
            splits=splits,
            n=train_n,
            seed=run_seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=device,
            log_every=args.log_every,
            early_stop_on_perfect=args.early_stop_on_perfect,
            loss_name=hc.normalize_loss_name(args.train_loss),
        )
        trained_models[model_name] = trained_model

    return trained_models, train_n, plot_seed, device


def _predict_probs(
    model: torch.nn.Module,
    sequences: np.ndarray,
    device: torch.device,
    batch_size: int | None = None,
) -> np.ndarray:
    model.eval()
    seq_array = np.asarray(sequences)
    if seq_array.ndim == 1:
        seq_array = seq_array[None, :]
    elif seq_array.ndim != 2:
        raise ValueError("sequences must have shape [n] or [batch, n]")

    if batch_size is None or batch_size <= 0:
        batch_size = len(seq_array)

    prob_chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(seq_array), int(batch_size)):
            chunk = seq_array[start : start + int(batch_size)]
            z = torch.tensor(chunk, dtype=torch.float32, device=device)
            _, probs = model(z)
            prob_chunks.append(probs.cpu().numpy())
    return np.concatenate(prob_chunks, axis=0)


def _threshold_probs(values: np.ndarray) -> np.ndarray:
    return (np.asarray(values, dtype=float) >= 0.5).astype(int)


def _build_defined_sequences(expected_n: int) -> np.ndarray:
    # Edit these sequences directly for both the trace plot and the histogram plot.
    sequences = np.array(
        [
            [1, 1, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 0, 1, 0, 1],
        ],
        dtype=int,
    )
    if sequences.ndim != 2:
        raise ValueError("Defined sequences must have shape [num_sequences, n].")
    if sequences.shape[1] != expected_n:
        raise ValueError(
            f"Defined sequences have length {sequences.shape[1]}, but expected {expected_n} "
            "from min(seq_lengths)."
        )
    return sequences


def _build_histogram_sequences(
    n: int,
    *,
    seed: int,
    num_samples: int = 1_000_000,
) -> np.ndarray:
    sequences = hc.generate_random_sequences(num_samples, n, 0.5, seed)
    return sequences.cpu().numpy().astype(int)


def _style_hist_axis(ax, *, x_max: float, y_max: float) -> None:
    ax.set_facecolor("white")
    ax.set_xlim(-10.0, min(90, x_max))
    ax.set_ylim(0.0, min(0.03, y_max))
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=5, integer=True))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=4))
    ax.tick_params(
        axis="both",
        which="major",
        direction="out",
        labelsize=12.5,
        length=9.0,
        width=2.0,
        colors="black",
        bottom=True,
        top=False,
        left=True,
        right=False,
    )
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(2.5)


def _cumsum_rows(values: np.ndarray) -> np.ndarray:
    return np.cumsum(np.asarray(values, dtype=float), axis=1)


def _plot_histogram_cell(
    ax,
    values: np.ndarray,
    *,
    series_color: str,
    bins: np.ndarray,
    alpha: float = 0.22,
    zorder: float = 2.0,
) -> None:
    flat = np.asarray(values, dtype=float).ravel()
    ax.hist(
        flat,
        bins=bins,
        density=True,
        histtype="stepfilled",
        facecolor=series_color,
        edgecolor=series_color,
        linewidth=5.0,
        alpha=alpha,
        zorder=zorder,
    )
    ax.hist(
        flat,
        bins=bins,
        density=True,
        histtype="step",
        color=series_color,
        linewidth=5.0,
        zorder=zorder + 1,
    )


def _histogram_density_upper_bound(
    values_list: list[np.ndarray], bins: np.ndarray
) -> float:
    peak = 0.0
    for values in values_list:
        flat = np.asarray(values, dtype=float).ravel()
        density, _ = np.histogram(flat, bins=bins, density=True)
        if density.size:
            peak = max(peak, float(np.max(density)))
    return max(peak * 1.08, 1e-6)


def _slugify_model_name(model_name: str) -> str:
    cleaned = model_name.replace("$", "")
    cleaned = cleaned.replace("{", "").replace("}", "")
    cleaned = cleaned.replace("\\", "").replace(" ", "_")
    return cleaned.lower()


def _render_trace_figure(
    *,
    model_name: str,
    model_slug: str,
    z: np.ndarray,
    c: np.ndarray,
    slices: np.ndarray,
    model_values: np.ndarray,
    model_color: str,
    output_dir: Path,
) -> None:
    sns.set_theme(context="paper", style="whitegrid", font_scale=1.0, rc=PLOT_FONT_RC)

    column_titles = [
        r"$\mathrm{Sample}\ Z$",
        r"$\mathrm{Target}\ C$",
        r"$\mathrm{G\!-\!SLiCE}$",
        model_name,
    ]
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(11.8, 6.8),
        sharex="col",
        sharey="row",
        constrained_layout=False,
    )
    axes = np.asarray(axes)

    panel_specs = [
        (0, 0, column_titles[0], z, INPUT_COLOR),
        (0, 1, column_titles[1], c, GROUND_TRUTH_COLOR),
        (1, 1, column_titles[2], slices, SLICE_COLOR),
        (1, 0, column_titles[3], model_values, model_color),
    ]

    for row_idx, col_idx, title, values, color in panel_specs:
        plot_binary_cell(
            axes[row_idx, col_idx],
            values,
            series_color=color,
        )
        axes[row_idx, col_idx].set_title(
            title, fontsize=30.0, fontweight="bold", pad=12
        )

    for ax in axes[0, :]:
        ax.tick_params(labelbottom=False)
    for ax in axes[:, 1]:
        ax.tick_params(labelleft=False)

    # fig.supxlabel(r"$\mathrm{Time\ step}\ k$", fontsize=25.0, y=0.00)

    fig.subplots_adjust(
        left=0.08, right=0.995, bottom=0.12, top=0.84, wspace=0.16, hspace=0.40
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"slices_expressivity_trace_{model_slug}"
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {png_path}")
    print(f"Saved {pdf_path}")


def _render_figure(
    *,
    model_name: str,
    model_slug: str,
    z_cumsum: np.ndarray,
    c_cumsum: np.ndarray,
    slices_cumsum: np.ndarray,
    model_cumsum: np.ndarray,
    model_color: str,
    plot_n: int,
    output_dir: Path,
) -> None:
    sns.set_theme(context="paper", style="whitegrid", font_scale=1.0, rc=PLOT_FONT_RC)

    column_titles = [
        r"$\mathrm{Input}\ Z$",
        r"$\mathrm{Ground\ truth}\ C$",
        r"$\mathrm{G\!-\!SLiCE}$",
        model_name,
    ]
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(11.8, 6.8),
        sharex="col",
        sharey="row",
        constrained_layout=False,
    )
    axes = np.asarray(axes)
    bins = np.linspace(0.0, float(plot_n), min(24, max(10, plot_n + 1)))

    panel_specs = [
        (0, 0, column_titles[0], z_cumsum, INPUT_COLOR),
        (0, 1, column_titles[1], c_cumsum, GROUND_TRUTH_COLOR),
        (1, 1, column_titles[2], slices_cumsum, SLICE_COLOR),
        (1, 0, column_titles[3], model_cumsum, model_color),
    ]
    shared_y_max = _histogram_density_upper_bound(
        [z_cumsum, c_cumsum, slices_cumsum, model_cumsum],
        bins,
    )

    for row_idx, col_idx, title, values, color in panel_specs:
        _plot_histogram_cell(
            axes[row_idx, col_idx],
            values,
            series_color=color,
            bins=bins,
        )
        axes[row_idx, col_idx].set_title(
            title, fontsize=30.0, fontweight="bold", pad=12
        )

    for ax in axes[0, :]:
        ax.tick_params(labelbottom=False)
    for ax in axes[:, 1]:
        ax.tick_params(labelleft=False)
    for ax in axes.ravel():
        _style_hist_axis(ax, x_max=float(plot_n), y_max=shared_y_max)

    # fig.supxlabel(r"$\mathrm{Cumulative\ sum}$", fontsize=25.0, y=0.00)

    fig.subplots_adjust(
        left=0.08, right=0.995, bottom=0.12, top=0.84, wspace=0.16, hspace=0.40
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"slices_expressivity_cumsum_hist_{model_slug}"
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {png_path}")
    print(f"Saved {pdf_path}")


def main() -> None:
    args = parse_args()
    trained_models, train_n, plot_seed, device = _train_reference_models(args)

    trace_n = min(args.seq_lengths)
    plot_n = max(args.seq_lengths)
    trace_z = _build_defined_sequences(trace_n)
    trace_c = compute_target(trace_z)
    trace_slices = _predict_probs(trained_models["DenseSSMWidth2"], trace_z, device)
    trace_diagonal = _predict_probs(
        trained_models[f"diag_w{min(args.diag_widths)}"], trace_z, device
    )
    trace_nonselective = _predict_probs(
        trained_models[f"dense_w{min(args.dense_widths)}"], trace_z, device
    )

    hist_seed = 10007 * (plot_seed + 1) + 997 * plot_n + 17
    hist_z = _build_histogram_sequences(plot_n, seed=hist_seed)
    hist_c = compute_target(hist_z)
    hist_batch_size = max(int(args.batch_size), len(hist_z))
    hist_slices = _predict_probs(
        trained_models["DenseSSMWidth2"], hist_z, device, batch_size=hist_batch_size
    )
    hist_diagonal = _predict_probs(
        trained_models[f"diag_w{min(args.diag_widths)}"],
        hist_z,
        device,
        batch_size=hist_batch_size,
    )
    hist_nonselective = _predict_probs(
        trained_models[f"dense_w{min(args.dense_widths)}"],
        hist_z,
        device,
        batch_size=hist_batch_size,
    )

    z_cumsum = _cumsum_rows(hist_z)
    c_cumsum = _cumsum_rows(hist_c)
    slices_cumsum = _cumsum_rows(_threshold_probs(hist_slices))
    diagonal_cumsum = _cumsum_rows(_threshold_probs(hist_diagonal))
    nonselective_cumsum = _cumsum_rows(_threshold_probs(hist_nonselective))

    print(
        "Rendered expressivity figure from real trained models "
        f"(train_n={train_n}, trace_n={trace_n}, seed={plot_seed}, "
        f"dense_width={min(args.dense_widths)}, diag_width={min(args.diag_widths)}, "
        f"plot_n={plot_n}, defined_sequences={len(trace_z)}, histogram_sequences={len(hist_z)})."
    )

    _render_trace_figure(
        model_name=r"$\mathrm{Dense\ SSM}$",
        model_slug="dense_ssm",
        z=trace_z,
        c=trace_c,
        slices=trace_slices,
        model_values=trace_nonselective,
        model_color=NONSELECTIVE_COLOR,
        output_dir=args.output_dir,
    )
    _render_figure(
        model_name=r"$\mathrm{Dense\ SSM}$",
        model_slug="dense_ssm",
        z_cumsum=z_cumsum,
        c_cumsum=c_cumsum,
        slices_cumsum=slices_cumsum,
        model_cumsum=nonselective_cumsum,
        model_color=NONSELECTIVE_COLOR,
        plot_n=plot_n,
        output_dir=args.output_dir,
    )
    _render_trace_figure(
        model_name=r"$\mathrm{Diagonal\ SSM}$",
        model_slug="diagonal_ssm",
        z=trace_z,
        c=trace_c,
        slices=trace_slices,
        model_values=trace_diagonal,
        model_color=DIAGONAL_COLOR,
        output_dir=args.output_dir,
    )
    _render_figure(
        model_name=r"$\mathrm{Diagonal\ SSM}$",
        model_slug="diagonal_ssm",
        z_cumsum=z_cumsum,
        c_cumsum=c_cumsum,
        slices_cumsum=slices_cumsum,
        model_cumsum=diagonal_cumsum,
        model_color=DIAGONAL_COLOR,
        plot_n=plot_n,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
