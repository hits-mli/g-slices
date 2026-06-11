from __future__ import annotations

import argparse
import math
import os
import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
import seaborn as sns

EPS = 1e-12
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


# ----------------------------
# Reproducibility
# ----------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def resolve_device(device_flag: str) -> torch.device:
    if device_flag == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_flag == "cuda" and not torch.cuda.is_available():
        print("Requested cuda but CUDA is not available; falling back to cpu.")
        return torch.device("cpu")
    return torch.device(device_flag)


def enumerate_all_binary_sequences(
    n: int, *, device: torch.device | None = None
) -> torch.Tensor:
    if n < 0:
        raise ValueError("n must be non-negative")
    if n == 0:
        out = torch.empty((1, 0), dtype=torch.float32)
        return out.to(device) if device is not None else out

    num = 1 << n
    ints = torch.arange(num, dtype=torch.long)
    bits = ((ints.unsqueeze(1) >> torch.arange(n - 1, -1, -1)) & 1).to(torch.float32)
    return bits.to(device) if device is not None else bits


# ----------------------------
# Hard-core target
# ----------------------------


def hardcore_target(z: torch.Tensor) -> torch.Tensor:
    """
    C_1 = Z_1
    C_t = Z_t * (1 - C_{t-1})
    """
    if z.ndim != 2:
        raise ValueError("z must have shape [batch, n]")

    z = z.to(torch.float32)
    batch, n = z.shape
    c = torch.zeros((batch, n), dtype=torch.float32, device=z.device)
    if n == 0:
        return c

    c[:, 0] = z[:, 0]
    for t in range(1, n):
        c[:, t] = z[:, t] * (1.0 - c[:, t - 1])
    return c


def is_hardcore_valid(bits: torch.Tensor) -> torch.Tensor:
    if bits.ndim != 2:
        raise ValueError("bits must have shape [batch, n]")
    if bits.shape[1] < 2:
        return torch.ones(bits.shape[0], dtype=torch.bool, device=bits.device)
    return (bits[:, :-1] * bits[:, 1:] == 0).all(dim=1)


# ----------------------------
# Random data
# ----------------------------


def generate_random_sequences(
    num_samples: int, n: int, p: float, seed: int
) -> torch.Tensor:
    if num_samples < 0:
        raise ValueError("num_samples must be non-negative")
    if n <= 0:
        raise ValueError("n must be positive")
    g = torch.Generator().manual_seed(seed)
    return (torch.rand((num_samples, n), generator=g) < p).to(torch.float32)


@dataclass
class DatasetSplits:
    train_z: torch.Tensor
    train_c: torch.Tensor
    val_z: torch.Tensor
    val_c: torch.Tensor
    test_z: torch.Tensor
    test_c: torch.Tensor


def build_random_splits(
    *,
    n: int,
    p: float,
    train_size: int,
    val_size: int,
    test_size: int,
    seed: int,
) -> DatasetSplits:
    train_z = generate_random_sequences(train_size, n, p, seed + 11)
    val_z = generate_random_sequences(val_size, n, p, seed + 23)
    test_z = generate_random_sequences(test_size, n, p, seed + 37)
    return DatasetSplits(
        train_z=train_z,
        train_c=hardcore_target(train_z),
        val_z=val_z,
        val_c=hardcore_target(val_z),
        test_z=test_z,
        test_c=hardcore_target(test_z),
    )


# ----------------------------
# Models
# ----------------------------


class DiagonalSSM(nn.Module):
    """
    Width-d diagonal switched linear SSM:
        h_t = D(z_t) h_{t-1}
        y_t = r^T h_t + b
    where D(0), D(1) are diagonal with positive entries.
    """

    def __init__(self, width: int) -> None:
        super().__init__()
        if width <= 0:
            raise ValueError("width must be positive")
        self.width = width

        self.h0 = nn.Parameter(torch.randn(width) * 0.2)
        self.log_diag0 = nn.Parameter(torch.randn(width) * 0.2)
        self.log_diag1 = nn.Parameter(torch.randn(width) * 0.2)
        self.readout = nn.Parameter(torch.randn(width) * 0.2)
        self.bias = nn.Parameter(torch.zeros(()))

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if z.ndim != 2:
            raise ValueError("z must have shape [batch, n]")

        z = z.to(torch.float32)
        batch, n = z.shape
        h = self.h0.unsqueeze(0).expand(batch, -1)

        logits_steps: List[torch.Tensor] = []
        for t in range(n):
            zt = z[:, t].unsqueeze(1)
            log_diag_t = (1.0 - zt) * self.log_diag0.unsqueeze(
                0
            ) + zt * self.log_diag1.unsqueeze(0)
            diag_t = torch.exp(log_diag_t)
            h = h * diag_t
            logits_steps.append(h @ self.readout + self.bias)

        logits = (
            torch.stack(logits_steps, dim=1)
            if logits_steps
            else z.new_empty((batch, 0))
        )
        probs = torch.sigmoid(logits)
        return logits, probs


class DenseSSM(nn.Module):
    """
    Width-d non-selective linear SSM:
        h_t = A h_{t-1} + B z_t
        y_t = r^T h_t + b
    """

    def __init__(self, width: int) -> None:
        super().__init__()
        if width <= 0:
            raise ValueError("width must be positive")
        self.width = width

        self.h0 = nn.Parameter(torch.randn(width) * 0.2)
        self.A = nn.Parameter(0.05 * torch.randn(width, width))
        self.B = nn.Parameter(torch.randn(width, width) * 0.2)
        self.readout = nn.Parameter(torch.randn(width) * 0.2)
        self.bias = nn.Parameter(torch.zeros(()))
        self.linear_lift = nn.Linear(1, width)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if z.ndim != 2:
            raise ValueError("z must have shape [batch, n]")

        z = z.to(torch.float32)
        batch, n = z.shape
        h = self.h0.unsqueeze(0).expand(batch, -1)
        logits_steps = []
        for t in range(n):
            zt = z[:, t].unsqueeze(1)
            h = h + h @ self.A.T + self.linear_lift(zt) @ self.B.T
            logits_steps.append(h @ self.readout + self.bias)

        logits = (
            torch.stack(logits_steps, dim=1)
            if logits_steps
            else z.new_empty((batch, 0))
        )
        return logits, torch.sigmoid(logits)


class DenseSSMWidth2(nn.Module):
    """
    Dense width-2 switched linear SSM initialized at the exact hard-core solution.

    The exact system is parameterized in additive residual form:
        h_t = h_{t-1} + (M_exact(z_t) + ΔM(z_t)) h_{t-1}
    with h_0, readout, and bias also initialized at the exact solution.
    Training learns only the residual terms Δ.
    """

    def __init__(self, residual_scale: float = 0.1) -> None:
        super().__init__()
        self.residual_scale = float(residual_scale)

        self.register_buffer("exact_h0", torch.tensor([1.0, 0.0], dtype=torch.float32))
        self.register_buffer(
            "exact_m0",
            torch.tensor([[0.0, 0.0], [0.0, -1.0]], dtype=torch.float32),
        )
        self.register_buffer(
            "exact_m1",
            torch.tensor([[0.0, 0.0], [1.0, -2.0]], dtype=torch.float32),
        )
        self.register_buffer(
            "exact_readout", torch.tensor([0.0, 10.0], dtype=torch.float32)
        )
        self.register_buffer("exact_bias", torch.tensor(-5.0, dtype=torch.float32))

        self.residual_h0 = nn.Parameter(torch.zeros(2, dtype=torch.float32))
        self.residual_m0 = nn.Parameter(torch.zeros(2, 2, dtype=torch.float32))
        self.residual_m1 = nn.Parameter(torch.zeros(2, 2, dtype=torch.float32))
        self.residual_readout = nn.Parameter(torch.zeros(2, dtype=torch.float32))
        self.residual_bias = nn.Parameter(torch.zeros((), dtype=torch.float32))

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if z.ndim != 2:
            raise ValueError("z must have shape [batch, n]")

        z = z.to(torch.float32)
        batch, n = z.shape

        h0 = self.exact_h0 + self.residual_scale * self.residual_h0
        readout = self.exact_readout + self.residual_scale * self.residual_readout
        bias = self.exact_bias + self.residual_scale * self.residual_bias

        h = h0.unsqueeze(0).expand(batch, -1)

        logits_steps: List[torch.Tensor] = []
        for t in range(n):
            zt = z[:, t].view(batch, 1, 1)
            exact_mt = (1.0 - zt) * self.exact_m0.unsqueeze(
                0
            ) + zt * self.exact_m1.unsqueeze(0)
            residual_mt = (1.0 - zt) * self.residual_m0.unsqueeze(
                0
            ) + zt * self.residual_m1.unsqueeze(0)
            mt = exact_mt + self.residual_scale * residual_mt
            h = h + torch.bmm(mt, h.unsqueeze(-1)).squeeze(-1)
            logits_steps.append(h @ readout + bias)

        logits = (
            torch.stack(logits_steps, dim=1)
            if logits_steps
            else z.new_empty((batch, 0))
        )
        probs = torch.sigmoid(logits)
        return logits, probs


# ----------------------------
# Metrics / evaluation
# ----------------------------


@dataclass
class Metrics:
    loss: float
    mse: float
    token_acc: float
    exact_acc: float
    validity_rate: float


def normalize_loss_name(loss_name: str) -> str:
    normalized = loss_name.strip().lower()
    if normalized == "bxe":
        return "bce"
    if normalized not in {"bce", "mse"}:
        raise ValueError(
            f"Unsupported loss_name={loss_name!r}; expected one of: bce, bxe, mse"
        )
    return normalized


def compute_supervised_loss(
    *,
    logits: torch.Tensor,
    probs: torch.Tensor,
    target: torch.Tensor,
    loss_name: str,
    reduction: str,
) -> torch.Tensor:
    normalized = normalize_loss_name(loss_name)
    if normalized == "bce":
        return F.binary_cross_entropy_with_logits(logits, target, reduction=reduction)
    return F.mse_loss(probs, target, reduction=reduction)


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    z: torch.Tensor,
    c: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
    loss_name: str,
) -> Metrics:
    model.eval()

    num_samples, n = z.shape
    if num_samples == 0:
        return Metrics(
            loss=math.nan,
            mse=math.nan,
            token_acc=math.nan,
            exact_acc=math.nan,
            validity_rate=math.nan,
        )

    total_loss = 0.0
    total_mse = 0.0
    total_token_correct = 0
    total_exact = 0
    total_valid = 0
    total_tokens = num_samples * n

    bsz = max(1, batch_size)

    for start in range(0, num_samples, bsz):
        end = min(start + bsz, num_samples)
        zb = z[start:end].to(device)
        cb = c[start:end].to(device)

        logits, probs = model(zb)
        pred = (probs >= 0.5).to(cb.dtype)

        total_loss += float(
            compute_supervised_loss(
                logits=logits,
                probs=probs,
                target=cb,
                loss_name=loss_name,
                reduction="sum",
            ).item()
        )
        total_mse += float(F.mse_loss(probs, cb, reduction="sum").item())
        total_token_correct += int((pred == cb).sum().item())
        total_exact += int((pred == cb).all(dim=1).sum().item())
        total_valid += int(is_hardcore_valid(pred).sum().item())

    return Metrics(
        loss=total_loss / float(total_tokens),
        mse=total_mse / float(total_tokens),
        token_acc=total_token_correct / float(total_tokens),
        exact_acc=total_exact / float(num_samples),
        validity_rate=total_valid / float(num_samples),
    )


def minibatch_indices(
    num_items: int, batch_size: int, epoch_seed: int
) -> Sequence[torch.Tensor]:
    g = torch.Generator().manual_seed(epoch_seed)
    perm = torch.randperm(num_items, generator=g)
    return [perm[i : i + batch_size] for i in range(0, num_items, batch_size)]


@torch.no_grad()
def select_plot_sample_index(
    model: nn.Module,
    z: torch.Tensor,
    c: torch.Tensor,
    *,
    device: torch.device,
) -> tuple[int, bool]:
    """
    Pick a test example where the model is not perfectly correct if possible.

    This keeps the overlay plot informative by showing a genuinely hard sample.
    """
    model.eval()
    logits, _ = model(z.to(device))
    pred = (torch.sigmoid(logits) >= 0.5).to(c.dtype)
    mismatches = ~(pred == c.to(device)).all(dim=1)
    mismatch_indices = torch.nonzero(mismatches, as_tuple=False).flatten()
    if mismatch_indices.numel() > 0:
        return int(mismatch_indices[0].item()), True
    return 0, False


# ----------------------------
# Plotting
# ----------------------------


@torch.no_grad()
def plot_cumsum_all_models(
    *,
    selective_dense_model: nn.Module,
    dense_models: Dict[int, nn.Module],
    diag_models: Dict[int, nn.Module],
    n: int,
    z_seq: torch.Tensor,
    c_seq: torch.Tensor,
    device: torch.device,
    out_dir: str = "plots",
    sample_idx: int = 0,
) -> None:
    """
    Plot cumulative sums for:
      - input path
      - target path
      - selective DenseSSMWidth2 prediction
      - all dense-model predictions
      - all diagonal width-d predictions

    Uses vertical offsets so overlapping curves remain visible.
    Input is always the top curve.
    """
    sns.set_theme(style="white", context="paper", font_scale=1.35)

    os.makedirs(out_dir, exist_ok=True)
    selective_dense_model.eval()
    for model in dense_models.values():
        model.eval()
    for model in diag_models.values():
        model.eval()

    if z_seq.ndim == 1:
        z_seq = z_seq.unsqueeze(0)
    if c_seq.ndim == 1:
        c_seq = c_seq.unsqueeze(0)

    z_seq = z_seq[:, :n].to(device)
    c_seq = c_seq[:, :n].to(device)

    selective_dense_logits, _ = selective_dense_model(z_seq)
    selective_dense_pred = (torch.sigmoid(selective_dense_logits) >= 0.5).to(
        torch.float32
    )

    dense_preds: Dict[int, np.ndarray] = {}
    diag_preds: Dict[int, np.ndarray] = {}
    for d, model in sorted(dense_models.items()):
        dense_logits, _ = model(z_seq)
        dense_pred = (torch.sigmoid(dense_logits) >= 0.5).to(torch.float32)
        dense_preds[d] = dense_pred[0].detach().cpu().numpy()
    for d, model in sorted(diag_models.items()):
        diag_logits, _ = model(z_seq)
        diag_pred = (torch.sigmoid(diag_logits) >= 0.5).to(torch.float32)
        diag_preds[d] = diag_pred[0].detach().cpu().numpy()

    x_input = z_seq[0].detach().cpu().numpy()
    x_target = c_seq[0].detach().cpu().numpy()
    x_selective_dense = selective_dense_pred[0].detach().cpu().numpy()

    cs_input = np.cumsum(x_input)
    cs_target = np.cumsum(x_target)
    cs_selective_dense = np.cumsum(x_selective_dense)
    cs_dense = {d: np.cumsum(x_dense) for d, x_dense in dense_preds.items()}
    cs_diags = {d: np.cumsum(x_diag) for d, x_diag in diag_preds.items()}

    t = np.arange(1, len(cs_input) + 1, dtype=float)

    y_max_base = max(
        [cs_input.max(), cs_target.max(), cs_selective_dense.max()]
        + [v.max() for v in cs_dense.values()]
        + [v.max() for v in cs_diags.values()]
    )
    y_min_base = min(
        [cs_input.min(), cs_target.min(), cs_selective_dense.min()]
        + [v.min() for v in cs_dense.values()]
        + [v.min() for v in cs_diags.values()]
    )

    y_tick_min = int(np.floor(y_min_base))
    y_tick_max = int(np.ceil(y_max_base))
    y_ticks = np.arange(y_tick_min, y_tick_max + 1, 1)

    fig, ax = plt.subplots(figsize=(9.2, 6.4), dpi=500)

    band_half_width = 0.38
    for y in y_ticks:
        ax.axhspan(
            y - band_half_width,
            y + band_half_width,
            color="0.85",
            alpha=0.35,
            zorder=0,
        )

    # Ordering from top to bottom:
    # input, target, selective dense baseline, dense models descending by width,
    # then diagonal models descending by width
    ordered_labels = (
        [
            ("input", None),
            ("target", None),
            ("selective_dense", None),
        ]
        + [("dense", d) for d in sorted(dense_models.keys(), reverse=True)]
        + [("diag", d) for d in sorted(diag_models.keys(), reverse=True)]
    )

    y_delta = 0.25
    center_index = (len(ordered_labels) - 2) / 2.0

    offsets = {}
    for i, key in enumerate(ordered_labels):
        offsets[key] = (center_index - i) * y_delta

    line_width = 5.0
    marker_size = 10.0

    sns.lineplot(
        x=t,
        y=cs_input + offsets[("input", None)],
        ax=ax,
        linewidth=line_width,
        marker="o",
        markersize=marker_size,
        label=r"Input $Z_k$",
        zorder=3,
    )
    sns.lineplot(
        x=t,
        y=cs_target + offsets[("target", None)],
        ax=ax,
        linewidth=line_width,
        marker="o",
        markersize=marker_size,
        label=r"Target $C_k$",
        zorder=3,
    )
    sns.lineplot(
        x=t,
        y=cs_selective_dense + offsets[("selective_dense", None)],
        ax=ax,
        linewidth=line_width,
        marker="o",
        markersize=marker_size,
        label=r"LNCDE",
        zorder=3,
    )
    for d in sorted(dense_models.keys(), reverse=True):
        sns.lineplot(
            x=t,
            y=cs_dense[d] + offsets[("dense", d)],
            ax=ax,
            linewidth=line_width,
            marker="o",
            markersize=marker_size,
            label=rf"Dense n.-s. SSM",
            zorder=3,
        )

    for d in sorted(diag_models.keys(), reverse=True):
        sns.lineplot(
            x=t,
            y=cs_diags[d],  # + offsets[("diag", d)],
            ax=ax,
            linewidth=line_width,
            marker="o",
            markersize=marker_size,
            label=rf"Diagonal SSM",
            zorder=3,
        )

    ax.set_xlabel(r"Time step $k$", fontsize=25)
    ax.set_ylabel("Cumulative Sum", fontsize=25, labelpad=18)

    ax.set_xticks(np.arange(1, len(t) + 1, 1))
    ax.set_yticks(y_ticks)

    ax.minorticks_off()
    ax.grid(False)
    ax.set_xlim(0.75, len(t) + 0.25)
    ax.set_ylim(
        y_min_base - (len(ordered_labels) + 1) * y_delta,
        y_max_base + (len(ordered_labels) + 1) * y_delta,
    )

    ax.tick_params(
        axis="x",
        which="major",
        direction="out",
        length=5.0,
        width=2.0,
        bottom=True,
        top=False,
        labelsize=20,
    )
    ax.tick_params(
        axis="y",
        which="major",
        direction="out",
        length=5.0,
        width=2.0,
        left=True,
        right=False,
        labelsize=20,
    )

    legend = ax.legend(
        loc="upper left",
        frameon=True,
        fancybox=True,
        framealpha=0.9,
        fontsize=20,
        borderpad=0.3,
        labelspacing=0.35,
        handlelength=1.6,
    )
    legend.get_frame().set_edgecolor("0.75")
    legend.get_frame().set_linewidth(1.0)

    plt.tight_layout()

    pdf_path = os.path.join(
        out_dir,
        f"compare_dense_and_all_diagonals_n{n}_sample{sample_idx}.pdf",
    )
    png_path = os.path.join(
        out_dir,
        f"compare_dense_and_all_diagonals_n{n}_sample{sample_idx}.png",
    )

    plt.savefig(pdf_path, bbox_inches="tight")
    plt.savefig(png_path, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    print(f"Saved plot: {pdf_path}")
    print(f"Saved plot: {png_path}")


@torch.no_grad()
def plot_enumeration_cumsum_histograms(
    *,
    selective_dense_model: nn.Module,
    diag_model: nn.Module,
    n: int,
    device: torch.device,
    out_dir: str = "plots",
) -> None:
    """
    Enumerate all binary sequences of length n and compare the distributions of
    terminal cumulative sums for:
      - input sequences Z
      - hard-core target C
      - selective DenseSSMWidth2 prediction
      - lowest-width diagonal model prediction
    """
    sns.set_theme(style="whitegrid", context="talk")

    os.makedirs(out_dir, exist_ok=True)

    selective_dense_model.eval()
    diag_model.eval()

    z_all = enumerate_all_binary_sequences(n, device=device)  # [2^n, n]
    c_all = hardcore_target(z_all)  # [2^n, n]

    dense_logits, _ = selective_dense_model(z_all)
    diag_logits, _ = diag_model(z_all)

    dense_pred = (torch.sigmoid(dense_logits) >= 0.5).to(torch.float32)
    diag_pred = (torch.sigmoid(diag_logits) >= 0.5).to(torch.float32)

    # Terminal cumulative sums for each enumerated path.
    z_terminal = torch.cumsum(z_all, dim=1)[:, -1].detach().cpu().numpy()
    c_terminal = torch.cumsum(c_all, dim=1)[:, -1].detach().cpu().numpy()
    dense_terminal = torch.cumsum(dense_pred, dim=1)[:, -1].detach().cpu().numpy()
    diag_terminal = torch.cumsum(diag_pred, dim=1)[:, -1].detach().cpu().numpy()

    values = [z_terminal, c_terminal, dense_terminal, diag_terminal]
    titles = [
        rf"Input paths $Z$",
        r"Target paths $C$",
        r"DenseSSMWidth2",
        rf"Diagonal SSM $d=2$",
    ]

    bins = np.arange(-0.5, n + 1.5, 1.0)

    fig, axes = plt.subplots(
        2, 2, figsize=(13.5, 10.0), dpi=500, sharex=True, sharey=True
    )
    axes = axes.ravel()

    for ax, arr, title in zip(axes, values, titles):
        sns.histplot(
            arr,
            bins=bins,
            stat="probability",
            discrete=True,
            shrink=0.82,
            edgecolor="white",
            linewidth=0.8,
            alpha=0.95,
            ax=ax,
        )

        mean_val = float(np.mean(arr))
        ax.axvline(mean_val, linestyle="--", linewidth=1.3, color="black", alpha=0.85)

        ax.set_title(title, pad=12)
        ax.set_xlabel("Terminal Cumulative Sum")
        ax.set_ylabel("Probability", labelpad=14)
        ax.set_xticks(np.arange(0, n + 1, 1))
        ax.grid(axis="y", alpha=0.25)
        ax.grid(axis="x", visible=False)

    plt.tight_layout()

    pdf_path = os.path.join(out_dir, f"enumeration_terminal_cumsum_histograms_n{n}.pdf")
    png_path = os.path.join(out_dir, f"enumeration_terminal_cumsum_histograms_n{n}.png")

    plt.savefig(pdf_path, bbox_inches="tight")
    plt.savefig(png_path, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    print(f"Saved plot: {pdf_path}")
    print(f"Saved plot: {png_path}")


# ----------------------------
# Training
# ----------------------------


def train_one_run(
    *,
    model_factory: Callable[[], nn.Module],
    splits: DatasetSplits,
    n: int,
    seed: int,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    log_every: int,
    early_stop_on_perfect: bool,
    loss_name: str,
) -> Tuple[Metrics, nn.Module]:
    set_seed(seed)
    normalized_loss_name = normalize_loss_name(loss_name)
    model = model_factory().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_size = splits.train_z.shape[0]
    bsz = min(max(1, batch_size), train_size)

    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_val_exact = -math.inf
    best_val_loss = math.inf

    for epoch in range(1, epochs + 1):
        model.train()
        for idx in minibatch_indices(
            train_size, bsz, epoch_seed=seed * 1000003 + n * 1009 + epoch
        ):
            zb = splits.train_z[idx].to(device)
            cb = splits.train_c[idx].to(device)

            optimizer.zero_grad(set_to_none=True)
            logits, probs = model(zb)
            loss = compute_supervised_loss(
                logits=logits,
                probs=probs,
                target=cb,
                loss_name=normalized_loss_name,
                reduction="mean",
            )
            loss.backward()
            optimizer.step()

        train_metrics = evaluate_model(
            model,
            splits.train_z,
            splits.train_c,
            device=device,
            batch_size=bsz,
            loss_name=normalized_loss_name,
        )
        val_metrics = evaluate_model(
            model,
            splits.val_z,
            splits.val_c,
            device=device,
            batch_size=bsz,
            loss_name=normalized_loss_name,
        )

        better_val = val_metrics.exact_acc > best_val_exact + EPS
        tie_break = (
            abs(val_metrics.exact_acc - best_val_exact) <= EPS
            and val_metrics.loss < best_val_loss - EPS
        )
        if better_val or tie_break:
            best_val_exact = val_metrics.exact_acc
            best_val_loss = val_metrics.loss
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }

        if log_every > 0 and (epoch == 1 or epoch % log_every == 0 or epoch == epochs):
            print(
                f"n={n} seed={seed} epoch={epoch} "
                f"train_exact={train_metrics.exact_acc:.4f} "
                f"val_exact={val_metrics.exact_acc:.4f} "
                f"val_valid={val_metrics.validity_rate:.4f}"
            )

        if (
            early_stop_on_perfect
            and train_metrics.exact_acc >= 1.0 - EPS
            and val_metrics.exact_acc >= 1.0 - EPS
        ):
            break

    model.load_state_dict(best_state)
    model.to(device)

    test_metrics = evaluate_model(
        model,
        splits.test_z,
        splits.test_c,
        device=device,
        batch_size=bsz,
        loss_name=normalized_loss_name,
    )
    return test_metrics, model


# ----------------------------
# Experiment loop
# ----------------------------


def build_model_specs(
    dense_widths: Sequence[int],
    diag_widths: Sequence[int],
) -> List[Tuple[str, Callable[[], nn.Module]]]:
    specs: List[Tuple[str, Callable[[], nn.Module]]] = [
        ("DenseSSMWidth2", DenseSSMWidth2)
    ]
    for d in dense_widths:
        specs.append((f"dense_w{d}", lambda d=d: DenseSSM(width=d)))
    for d in diag_widths:
        specs.append((f"diag_w{d}", lambda d=d: DiagonalSSM(width=d)))
    return specs


def run_experiment(
    *,
    seq_lengths: Sequence[int],
    dense_widths: Sequence[int],
    diag_widths: Sequence[int],
    seeds: Sequence[int],
    p: float,
    train_size: int,
    val_size: int,
    test_size: int,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    log_every: int,
    early_stop_on_perfect: bool,
    max_enumeration_n: int,
    loss_name: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    model_specs = build_model_specs(dense_widths, diag_widths)
    normalized_loss_name = normalize_loss_name(loss_name)

    exact_scores: Dict[str, Dict[int, List[float]]] = {
        name: {n: [] for n in seq_lengths} for name, _ in model_specs
    }
    valid_scores: Dict[str, Dict[int, List[float]]] = {
        name: {n: [] for n in seq_lengths} for name, _ in model_specs
    }
    mse_scores: Dict[str, Dict[int, List[float]]] = {
        name: {n: [] for n in seq_lengths} for name, _ in model_specs
    }

    smallest_n = min(seq_lengths)
    largest_dense_width = max(dense_widths)
    lowest_diag_width = min(diag_widths)

    hist_selective_dense_model: nn.Module | None = None
    hist_diag_model: nn.Module | None = None

    for n in seq_lengths:
        print(f"\n=== Sequence length n={n} ===")
        for seed in seeds:
            data_seed = 10007 * (seed + 1) + 997 * n
            splits = build_random_splits(
                n=n,
                p=p,
                train_size=train_size,
                val_size=val_size,
                test_size=test_size,
                seed=data_seed,
            )

            trained_models: Dict[str, nn.Module] = {}

            for model_idx, (model_name, model_factory) in enumerate(model_specs):
                run_seed = 20011 * (seed + 1) + 131 * n + model_idx
                print(f"\n--- model={model_name} n={n} seed={seed} ---")

                test_metrics, trained_model = train_one_run(
                    model_factory=model_factory,
                    splits=splits,
                    n=n,
                    seed=run_seed,
                    epochs=epochs,
                    batch_size=batch_size,
                    lr=lr,
                    device=device,
                    log_every=log_every,
                    early_stop_on_perfect=early_stop_on_perfect,
                    loss_name=normalized_loss_name,
                )

                trained_models[model_name] = trained_model
                exact_scores[model_name][n].append(test_metrics.exact_acc)
                valid_scores[model_name][n].append(test_metrics.validity_rate)
                mse_scores[model_name][n].append(test_metrics.mse)

                print(
                    f"TEST model={model_name} n={n} seed={seed} "
                    f"exact={test_metrics.exact_acc:.4f} "
                    f"valid={test_metrics.validity_rate:.4f} "
                    f"mse={test_metrics.mse:.6f} "
                    f"token={test_metrics.token_acc:.4f}"
                )

            # Keep one trained pair for the end-of-run enumeration histogram.
            if n == smallest_n and seed == seeds[0]:
                hist_selective_dense_model = trained_models["DenseSSMWidth2"]
                hist_diag_model = trained_models[f"diag_w{lowest_diag_width}"]

            # Plot dense models together with all diagonal models on the same sample.
            if seed == seeds[0]:
                largest_dense_model = trained_models[f"dense_w{largest_dense_width}"]
                plot_sample_idx, found_hard_sample = select_plot_sample_index(
                    largest_dense_model,
                    splits.test_z,
                    splits.test_c,
                    device=device,
                )
                if not found_hard_sample:
                    print(
                        f"No failing test sample found for dense_w{largest_dense_width}; "
                        "using the first test example for the overlay plot."
                    )
                selective_dense_model = trained_models["DenseSSMWidth2"]
                dense_models = {d: trained_models[f"dense_w{d}"] for d in dense_widths}
                diag_models = {d: trained_models[f"diag_w{d}"] for d in diag_widths}

                plot_cumsum_all_models(
                    selective_dense_model=selective_dense_model,
                    dense_models=dense_models,
                    diag_models=diag_models,
                    n=n,
                    z_seq=splits.test_z[plot_sample_idx],
                    c_seq=splits.test_c[plot_sample_idx],
                    device=device,
                    out_dir="plots",
                    sample_idx=plot_sample_idx,
                )

    if (
        hist_selective_dense_model is not None
        and hist_diag_model is not None
        and smallest_n <= max_enumeration_n
    ):
        plot_enumeration_cumsum_histograms(
            selective_dense_model=hist_selective_dense_model,
            diag_model=hist_diag_model,
            n=smallest_n,
            device=device,
            out_dir="plots",
        )
    elif smallest_n > max_enumeration_n:
        print(
            f"Skipping enumeration histogram for n={smallest_n} because "
            f"2^n is too large; set --max-enumeration-n higher to enable it."
        )

    exact_table = pd.DataFrame(
        {
            n: {
                model: float(np.mean(exact_scores[model][n]))
                for model, _ in model_specs
            }
            for n in seq_lengths
        }
    )

    valid_table = pd.DataFrame(
        {
            n: {
                model: float(np.mean(valid_scores[model][n]))
                for model, _ in model_specs
            }
            for n in seq_lengths
        }
    )
    mse_table = pd.DataFrame(
        {
            n: {model: float(np.mean(mse_scores[model][n])) for model, _ in model_specs}
            for n in seq_lengths
        }
    )

    exact_table.index.name = "model"
    valid_table.index.name = "model"
    mse_table.index.name = "model"

    return exact_table, valid_table, mse_table


# ----------------------------
# CLI
# ----------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare dense and diagonal width-d SSMs on random hard-core transduction"
    )

    parser.add_argument("--seq-lengths", type=int, nargs="+", default=[8, 32, 128, 512])
    parser.add_argument("--dense-widths", type=int, nargs="+", default=[8])
    parser.add_argument("--diag-widths", type=int, nargs="+", default=[8])

    parser.add_argument("--p", type=float, default=0.5)
    parser.add_argument("--train-size", type=int, default=1000)
    parser.add_argument("--val-size", type=int, default=200)
    parser.add_argument("--test-size", type=int, default=200)

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])

    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument(
        "--early-stop-on-perfect", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--train-loss",
        type=str,
        default="mse",
        help="Supervised training loss. Supported values: bce, bxe, mse.",
    )
    parser.add_argument(
        "--max-enumeration-n",
        type=int,
        default=20,
        help="Maximum sequence length allowed for exhaustive enumeration plots.",
    )

    parser.add_argument(
        "--save-csv-prefix", type=str, default="figures/expressivity_gap_tables"
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not (0.0 < args.p < 1.0):
        raise ValueError("p must be in (0, 1)")
    if any(n <= 0 for n in args.seq_lengths):
        raise ValueError("all sequence lengths must be positive")
    if any(d <= 0 for d in args.dense_widths):
        raise ValueError("all dense widths must be positive")
    if any(d <= 0 for d in args.diag_widths):
        raise ValueError("all diagonal widths must be positive")
    if args.max_enumeration_n < 0:
        raise ValueError("max_enumeration_n must be non-negative")
    args.train_loss = normalize_loss_name(args.train_loss)

    device = resolve_device(args.device)

    print("=== Hyperparameters ===")
    print(f"device={device}")
    print(f"seq_lengths={args.seq_lengths}")
    print(f"dense_widths={args.dense_widths}")
    print(f"diag_widths={args.diag_widths}")
    print(f"p={args.p}")
    print(f"train/val/test sizes={args.train_size}/{args.val_size}/{args.test_size}")
    print(f"epochs={args.epochs}, lr={args.lr}, batch_size={args.batch_size}")
    print(f"train_loss={args.train_loss}")
    print(f"seeds={args.seeds}")

    exact_table, valid_table, mse_table = run_experiment(
        seq_lengths=args.seq_lengths,
        dense_widths=args.dense_widths,
        diag_widths=args.diag_widths,
        seeds=args.seeds,
        p=args.p,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
        log_every=args.log_every,
        early_stop_on_perfect=args.early_stop_on_perfect,
        max_enumeration_n=args.max_enumeration_n,
        loss_name=args.train_loss,
    )

    pd.set_option("display.precision", 4)

    print("\n=== Table 1: Exact accuracy vs hard-core ground truth ===")
    print(exact_table.to_string(float_format=lambda x: f"{x:.4f}"))

    print("\n=== Table 2: Valid-sequence ratio ===")
    print(valid_table.to_string(float_format=lambda x: f"{x:.4f}"))

    print("\n=== Table 3: Mean squared error vs hard-core ground truth ===")
    print(mse_table.to_string(float_format=lambda x: f"{x:.6f}"))

    exact_csv = f"{args.save_csv_prefix}_exact_accuracy.csv"
    valid_csv = f"{args.save_csv_prefix}_valid_ratio.csv"
    mse_csv = f"{args.save_csv_prefix}_mse.csv"
    exact_table.to_csv(exact_csv)
    valid_table.to_csv(valid_csv)
    mse_table.to_csv(mse_csv)

    print(f"\nSaved: {exact_csv}")
    print(f"Saved: {valid_csv}")
    print(f"Saved: {mse_csv}")


if __name__ == "__main__":
    main()
