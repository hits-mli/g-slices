"""Periodic training batch visualization callback for GluonTS-compatible models.

Every *plot_every* training epochs, this callback visualizes a sample of training batches
to help monitor data quality, preprocessing, and model predictions on training data.

The callback:
1. Captures a configurable number of training batch samples.
2. Generates full sequence predictions (context + future) for the captured batches.
3. Visualizes the context (past_target), target (future_target), and full predictions.
4. Saves multi-panel plots showing individual series and aggregated statistics.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")  # non-interactive backend (safe for SLURM / headless)
import matplotlib.pyplot as plt
import numpy as np
import torch
from pytorch_lightning import Callback

from gslice.utils.variables import get_relative_time_step

log = logging.getLogger(__name__)


def _elapsed_time_axis(length: int, step_hours: float, *, start_hours: float = 0.0) -> np.ndarray:
    if length <= 0:
        return np.array([], dtype=np.float64)
    return start_hours + (np.arange(length, dtype=np.float64) + 1.0) * float(step_hours)


class TrainingBatchPlotCallback(Callback):
    """Periodic training batch visualization callback.

    Parameters
    ----------
    plot_every : int
        Plot training batches every *N* training epochs.
    num_batches : int
        Number of training batches to capture and visualize.
    num_samples : int
        Number of prediction samples to draw from the model.
    max_show : int
        Maximum number of individual series to show in the plot.
    save_dir : str | Path
        Directory where PNG plots are saved.
    """

    def __init__(
        self,
        plot_every: int = 10,
        num_batches: int = 4,
        num_samples: int = 16,
        max_show: int = 8,
        save_dir: str = "./logs/train_plots",
    ) -> None:
        super().__init__()
        self.plot_every = plot_every
        self.num_batches = num_batches
        self.num_samples = num_samples
        self.max_show = max_show
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Storage for batch data
        self.collected_batches = []
        self.current_epoch_to_plot = None

    # ------------------------------------------------------------------ #
    # Lightning hooks
    # ------------------------------------------------------------------ #
    def on_train_epoch_start(self, trainer, pl_module):
        """Mark if we should collect batches this epoch."""
        epoch = trainer.current_epoch
        if (epoch + 1) % self.plot_every == 0:
            self.current_epoch_to_plot = epoch
            self.collected_batches = []
            log.info("[TrainBatchPlot] Will collect batches for visualization at epoch %d", epoch + 1)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Collect batch data and predictions if we're in a plotting epoch."""
        if self.current_epoch_to_plot != trainer.current_epoch:
            return
        if len(self.collected_batches) >= self.num_batches:
            return

        try:
            if hasattr(pl_module, "_batch_to_lcde"):
                batch_data = self._collect_lcde_batch(pl_module, batch)
            else:
                batch_data = self._collect_tsflow_batch(pl_module, batch)
        except Exception:
            log.exception("[TrainBatchPlot] Failed to collect batch %d", batch_idx)
            return

        if batch_data is None:
            return

        self.collected_batches.append(batch_data)
        if len(self.collected_batches) == self.num_batches:
            log.info("[TrainBatchPlot] Collected %d batches", self.num_batches)

    def _collect_lcde_batch(self, pl_module, batch) -> Optional[dict]:
        """Collect denormalized batch data for LCDE-like models."""
        prepared = pl_module._batch_to_lcde(batch)

        with torch.no_grad():
            t = prepared["t"]
            target = prepared["target"]
            control = prepared["control"]
            cond_features = prepared.get("cond_features", None)
            x0 = prepared["x0"]
            obs_mask = prepared["obs_mask"]
            scale = prepared["scale"]
            loc = prepared["loc"]
            gp_mean = prepared.get("gp_mean", None)
            context_h0 = prepared.get("context_h0", None)
            backbone = getattr(pl_module, "backbone", None)

            B, S, T = control.shape[0], control.shape[1], control.shape[2]

            control_flat = control.reshape(-1, T, control.shape[-1])
            cond_repeated = None
            if cond_features is not None:
                cond_repeated = cond_features.reshape(-1, T, cond_features.shape[-1])

            x0_flat = x0.reshape(-1, x0.shape[-1])

            t_repeated = t.unsqueeze(1).expand(B, S, T, 1).reshape(-1, T, 1)
            t0 = t_repeated[:, 0, :]

            context_h0_repeated = None
            if context_h0 is not None:
                context_h0_repeated = context_h0.unsqueeze(1).expand(B, S, -1).reshape(-1, context_h0.shape[-1])

            pred = pl_module.forward(
                control_flat,
                t_repeated,
                cond_features=cond_repeated,
            )
            pred_reshaped = pred.reshape(B, S, T, -1).permute(1, 0, 2, 3)  # (S, B, T, D)
            full_pred_denorm = pred_reshaped * scale.unsqueeze(0).unsqueeze(2) + loc.unsqueeze(0).unsqueeze(2)

            control_recon_denorm = None
            target_recon_denorm = None
            if backbone is not None and hasattr(backbone, "encode_path") and hasattr(backbone, "decode_path"):
                control_latent = backbone.encode_path(control_flat, t_repeated, x0_flat, t0)
                control_recon = backbone.decode_path(control_latent, t_repeated)
                control_recon_denorm = (
                    control_recon.reshape(B, S, T, -1).permute(1, 0, 2, 3) * scale.unsqueeze(0).unsqueeze(2)
                    + loc.unsqueeze(0).unsqueeze(2)
                )

                target_x0 = target[:, 0, :]
                target_latent = backbone.encode_path(target, t, target_x0, t[:, 0, :])
                target_recon = backbone.decode_path(target_latent, t)
                target_recon_denorm = target_recon * scale.unsqueeze(1) + loc.unsqueeze(1)

        future_len = batch["future_target"].shape[1]
        available_context_len = target.shape[1] - future_len
        plot_context_len = int(getattr(pl_module, "context_length", available_context_len))
        plot_context_len = min(plot_context_len, available_context_len)
        context_start = available_context_len - plot_context_len
        context_target_denorm = target[:, context_start:available_context_len, :] * scale.unsqueeze(1) + loc.unsqueeze(1)

        past_target = context_target_denorm.detach().cpu().numpy()
        future_target = batch["future_target"].detach().cpu().numpy()
        predictions = full_pred_denorm.detach().cpu().numpy()  # (S, B, full_len, D)
        control_signal_denorm = control.detach() * scale.unsqueeze(1).unsqueeze(2) + loc.unsqueeze(1).unsqueeze(2)
        control_signal = control_signal_denorm.detach().cpu().numpy()
        control_reconstruction = None
        target_reconstruction = None
        if control_recon_denorm is not None:
            control_reconstruction = control_recon_denorm.detach().cpu().numpy()
        if target_recon_denorm is not None:
            target_reconstruction = target_recon_denorm.detach().cpu().numpy()

        return {
            "past_target": past_target,
            "future_target": future_target,
            "predictions": predictions,
            "control": control_signal,
            "control_reconstruction": control_reconstruction,
            "target_reconstruction": target_reconstruction,
        }

    def _collect_tsflow_batch(self, pl_module, batch) -> Optional[dict]:
        """Collect denormalized batch data for TSFlow-like models."""
        required = {"past_target", "future_target", "past_observed_values", "mean"}
        if not required.issubset(batch.keys()):
            return None

        with torch.no_grad():
            past_target = torch.as_tensor(batch["past_target"], dtype=torch.float32, device=pl_module.device)
            future_target = torch.as_tensor(batch["future_target"], dtype=torch.float32, device=pl_module.device)
            past_observed_values = torch.as_tensor(batch["past_observed_values"], dtype=torch.float32, device=pl_module.device)
            mean = torch.as_tensor(batch["mean"], dtype=torch.float32, device=pl_module.device)

            original_num_samples = getattr(pl_module, "num_samples", None)
            if original_num_samples is not None:
                pl_module.num_samples = self.num_samples
            try:
                pred_future = pl_module.forward(
                    past_target=past_target,
                    past_observed_values=past_observed_values,
                    mean=mean,
                )
            finally:
                if original_num_samples is not None:
                    pl_module.num_samples = original_num_samples

            if pred_future.dim() == 3:
                pred_future = pred_future.unsqueeze(-1)  # (B, S, T) -> (B, S, T, 1)
            pred_future = pred_future.permute(1, 0, 2, 3)  # (S, B, T, D)

            x1, x0, _, loc, scale, _, _ = pl_module._extract_features(
                {
                    "past_target": past_target,
                    "future_target": future_target,
                    "past_observed_values": past_observed_values,
                    "mean": mean,
                }
            )
            context_len = x1.shape[1] - future_target.shape[1]
            context_target_denorm = x1[:, :context_len, :] * scale + loc
            control_full_denorm = x0 * scale + loc  # (B, full_len, D)

            context_for_pred = context_target_denorm.unsqueeze(0).expand(pred_future.shape[0], -1, -1, -1)
            full_pred_denorm = torch.cat([context_for_pred, pred_future], dim=2)  # (S, B, full_len, D)

        available_context_len = context_target_denorm.shape[1]
        plot_context_len = int(getattr(pl_module, "context_length", available_context_len))
        plot_context_len = min(plot_context_len, available_context_len)
        context_start = available_context_len - plot_context_len

        past_target_plot = context_target_denorm[:, context_start:available_context_len, :].detach().cpu().numpy()
        future_target_plot = future_target.detach().cpu().numpy()
        predictions = full_pred_denorm.detach().cpu().numpy()
        control_signal = (
            control_full_denorm.unsqueeze(1)
            .expand(-1, pred_future.shape[0], -1, -1)
            .detach()
            .cpu()
            .numpy()
        )

        return {
            "past_target": past_target_plot,
            "future_target": future_target_plot,
            "predictions": predictions,
            "control": control_signal,
        }

    def on_train_epoch_end(self, trainer, pl_module):
        """Generate and save the plot after epoch ends."""
        epoch = trainer.current_epoch
        if self.current_epoch_to_plot != epoch or not self.collected_batches:
            return

        log.info("[TrainBatchPlot] Generating training batch visualization at epoch %d", epoch + 1)

        try:
            fig = self._plot_training_batches(self.collected_batches, epoch, pl_module)
            save_path = self.save_dir / f"train_batch_epoch_{epoch + 1:04d}.png"
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            log.info("[TrainBatchPlot] Plot saved to %s", save_path)
        except Exception:
            log.exception("[TrainBatchPlot] Failed to generate plot at epoch %d", epoch + 1)
        finally:
            # Clean up
            self.collected_batches = []
            self.current_epoch_to_plot = None

    # ------------------------------------------------------------------ #
    # Plotting
    # ------------------------------------------------------------------ #
    def _plot_training_batches(self, batches, epoch: int, pl_module):
        """Create a multi-panel training batch visualization.

        Layout
        ------
        - **Row 0** (full width): Aggregated mean ± std for context, target, control, and full predictions.
        - **Rows 1‥N** (2 columns): Individual series showing context + target + control + full predictions.
        - **Last row** (full width): Pointwise squared error per series (future horizon only).
        """
        # Collect all sequences
        all_contexts = []
        all_targets = []
        all_preds = []
        all_controls = []
        all_control_recons = []
        all_target_recons = []
        has_reconstruction = False
        
        for batch_data in batches:
            past = batch_data["past_target"]  # shape: (batch_size, context_length, features)
            future = batch_data["future_target"]  # shape: (batch_size, prediction_length, features)
            pred = batch_data["predictions"]  # shape: (num_samples, batch_size, full_length, features)
            ctrl = batch_data["control"]  # shape: (batch_size, num_samples, full_length, features)
            ctrl_recon = batch_data.get("control_reconstruction")
            tgt_recon = batch_data.get("target_reconstruction")
            
            # Handle different dimensions
            if past.ndim == 3:
                past = past[:, :, 0]  # Plot the first feature when extra feature dimensions are present.
            if future.ndim == 3:
                future = future[:, :, 0]
            if pred.ndim == 4:
                pred = pred[:, :, :, 0]  # (samples, batch, time, features) -> (samples, batch, time)
                pred = pred.transpose(1, 0, 2)  # (batch, samples, time)
            if ctrl.ndim == 4:
                ctrl = ctrl[:, :, :, 0]  # (batch, samples, time, features) -> (batch, samples, time)
            if ctrl_recon is not None:
                has_reconstruction = True
                if ctrl_recon.ndim == 4:
                    ctrl_recon = ctrl_recon[:, :, :, 0]
            if tgt_recon is not None:
                has_reconstruction = True
                if tgt_recon.ndim == 3:
                    tgt_recon = tgt_recon[:, :, 0]
                
            all_contexts.append(past)
            all_targets.append(future)
            all_preds.append(pred)
            all_controls.append(ctrl)
            if ctrl_recon is not None:
                all_control_recons.append(ctrl_recon)
            if tgt_recon is not None:
                all_target_recons.append(tgt_recon)

        all_contexts = np.concatenate(all_contexts, axis=0)  # (total_samples, context_length)
        all_targets = np.concatenate(all_targets, axis=0)    # (total_samples, prediction_length)
        all_preds = np.concatenate(all_preds, axis=0)        # (total_samples, num_samples, full_length)
        all_controls = np.concatenate(all_controls, axis=0)  # (total_samples, num_samples, full_length)
        if has_reconstruction:
            all_control_recons = np.concatenate(all_control_recons, axis=0) if all_control_recons else None
            all_target_recons = np.concatenate(all_target_recons, axis=0) if all_target_recons else None
        else:
            all_control_recons = None
            all_target_recons = None

        n_show = min(len(all_contexts), self.max_show)
        context_length = all_contexts.shape[1]
        prediction_length = all_targets.shape[1]
        step_hours = float(getattr(pl_module, "relative_time_step", 0.0) or 0.0)
        if step_hours <= 0.0:
            freq_value = getattr(pl_module, "freq", None)
            freq_str = freq_value if isinstance(freq_value, str) else "1h"
            step_hours = float(get_relative_time_step(freq_str))
        cmap = plt.colormaps["tab20"].colors

        n_indiv_rows = (n_show + 1) // 2
        n_rows = 1 + n_indiv_rows + 1  # aggregated + individual + error
        fig = plt.figure(figsize=(18, 3.5 * n_rows))
        gs = fig.add_gridspec(n_rows, 2, hspace=0.45, wspace=0.30)

        # ---- Row 0: aggregated mean ± std ----
        ax_mean = fig.add_subplot(gs[0, :])

        ctx_subset = all_contexts[:n_show]
        tgt_subset = all_targets[:n_show]
        pred_subset = all_preds[:n_show]
        ctrl_subset = all_controls[:n_show]
        ctrl_recon_subset = all_control_recons[:n_show] if all_control_recons is not None else None
        tgt_recon_subset = all_target_recons[:n_show] if all_target_recons is not None else None

        context_duration_hours = float(context_length) * step_hours
        t_ctx = _elapsed_time_axis(context_length, step_hours)
        t_tgt = _elapsed_time_axis(prediction_length, step_hours, start_hours=context_duration_hours)
        t_full = _elapsed_time_axis(context_length + prediction_length, step_hours)

        ctx_mean, ctx_std = ctx_subset.mean(0), ctx_subset.std(0)
        tgt_mean, tgt_std = tgt_subset.mean(0), tgt_subset.std(0)
        pred_mean = pred_subset.mean(axis=(0, 1))  # Average over samples and batch
        pred_std = pred_subset.std(axis=(0, 1))
        ctrl_mean = ctrl_subset.mean(axis=(0, 1))  # Average over samples and batch
        ctrl_std = ctrl_subset.std(axis=(0, 1))
        if ctrl_recon_subset is not None:
            ctrl_recon_mean = ctrl_recon_subset.mean(axis=(0, 1))
            ctrl_recon_std = ctrl_recon_subset.std(axis=(0, 1))
        if tgt_recon_subset is not None:
            tgt_recon_mean = tgt_recon_subset.mean(axis=0)
            tgt_recon_std = tgt_recon_subset.std(axis=0)

        ax_mean.plot(t_ctx, ctx_mean, lw=2, color="steelblue", label="Context Mean")
        ax_mean.plot(t_tgt, tgt_mean, lw=2, color="black", label="Target Mean")
        ax_mean.plot(t_full, pred_mean, lw=2, color="red", ls="--", label="Full Prediction Mean")
        ax_mean.plot(t_full, ctrl_mean, lw=2, color="green", ls=":", label="Control Mean")
        if ctrl_recon_subset is not None:
            ax_mean.plot(t_full, ctrl_recon_mean, lw=2, color="purple", ls="-.", label="d(e(Control)) Mean")
            ax_mean.fill_between(
                t_full,
                ctrl_recon_mean - ctrl_recon_std,
                ctrl_recon_mean + ctrl_recon_std,
                alpha=0.12,
                color="purple",
                label="d(e(Control)) ±1σ",
            )
        if tgt_recon_subset is not None:
            ax_mean.plot(t_full, tgt_recon_mean, lw=2, color="orange", ls="-.", label="d(e(Target)) Mean")
            ax_mean.fill_between(
                t_full,
                tgt_recon_mean - tgt_recon_std,
                tgt_recon_mean + tgt_recon_std,
                alpha=0.12,
                color="orange",
                label="d(e(Target)) ±1σ",
            )
        ax_mean.fill_between(t_ctx, ctx_mean - ctx_std, ctx_mean + ctx_std, 
                             alpha=0.2, color="steelblue", label="Context ±1σ")
        ax_mean.fill_between(t_tgt, tgt_mean - tgt_std, tgt_mean + tgt_std, 
                             alpha=0.2, color="black", label="Target ±1σ")
        ax_mean.fill_between(t_full, pred_mean - pred_std, pred_mean + pred_std, 
                             alpha=0.2, color="red", label="Prediction ±1σ")
        ax_mean.fill_between(t_full, ctrl_mean - ctrl_std, ctrl_mean + ctrl_std, 
                             alpha=0.15, color="green", label="Control ±1σ")
        ax_mean.axvline(x=context_duration_hours, color="gray", ls="--", alpha=0.5, label="Forecast Start")
        ax_mean.set_title(f"Aggregated Training Data & Predictions — Epoch {epoch + 1}", fontsize=14, fontweight="bold")
        ax_mean.legend(fontsize=9, loc="best", ncol=4)
        ax_mean.grid(True, alpha=0.3)
        ax_mean.set_ylabel("Value")
        ax_mean.set_xlabel("Elapsed Time [h]")

        # ---- Middle rows: individual series (2 cols) ----
        for idx in range(n_show):
            row = 1 + idx // 2
            col = idx % 2
            ax = fig.add_subplot(gs[row, col])

            context_vals = ctx_subset[idx]
            target_vals = tgt_subset[idx]
            pred_samples = pred_subset[idx]  # (num_samples, full_length)
            ctrl_samples = ctrl_subset[idx]  # (num_samples, full_length)

            context_duration_hours = float(len(context_vals)) * step_hours
            t_ctx = _elapsed_time_axis(len(context_vals), step_hours)
            t_tgt = _elapsed_time_axis(len(target_vals), step_hours, start_hours=context_duration_hours)
            t_full = _elapsed_time_axis(len(context_vals) + len(target_vals), step_hours)

            c = cmap[idx % len(cmap)]
            ax.plot(t_ctx, context_vals, color="black", lw=1.5, label="Context", ls="-")
            ax.plot(t_tgt, target_vals, color="black", lw=1.5, ls="--", label="Target")
            
            # Prediction quantiles over full sequence
            pred_med = np.median(pred_samples, axis=0)
            pred_q10 = np.quantile(pred_samples, 0.1, axis=0)
            pred_q90 = np.quantile(pred_samples, 0.9, axis=0)
            pred_mean = pred_samples.mean(axis=0)
            
            # Control signal (mean over samples)
            ctrl_mean = ctrl_samples.mean(axis=0)
            ctrl_recon_mean = None
            if ctrl_recon_subset is not None:
                ctrl_recon_mean = ctrl_recon_subset[idx].mean(axis=0)
            tgt_recon_vals = None
            if tgt_recon_subset is not None:
                tgt_recon_vals = tgt_recon_subset[idx]
            
            ax.plot(t_full, pred_med, color=c, lw=2, label="Prediction Median")
            ax.plot(t_full, pred_mean, color=c, lw=2, ls="-.", alpha=0.7, label="Prediction Mean")
            ax.fill_between(t_full, pred_q10, pred_q90, color=c, alpha=0.25, label="10–90 % CI")
            ax.plot(t_full, ctrl_mean, color="green", lw=2, ls=":", alpha=0.6, label="Control")
            if ctrl_recon_mean is not None:
                ax.plot(t_full, ctrl_recon_mean, color="purple", lw=1.8, ls="-.", alpha=0.8, label="d(e(Control))")
            if tgt_recon_vals is not None:
                ax.plot(t_full, tgt_recon_vals, color="orange", lw=1.8, ls="-.", alpha=0.8, label="d(e(Target))")
            ax.axvline(x=context_duration_hours, color="gray", ls=":", alpha=0.5)
            ax.set_title(f"Series {idx + 1}", fontsize=10)
            ax.grid(True, alpha=0.3)
            if idx == 0:
                ax.legend(fontsize=8)

        # ---- Last row: pointwise squared error (future horizon only) ----
        ax_err = fig.add_subplot(gs[-1, :])
        for idx in range(n_show):
            tv = tgt_subset[idx]
            pm_full = pred_subset[idx].mean(axis=0)  # Mean over samples (full sequence)
            pm_future = pm_full[-prediction_length:]  # Extract future portion only
            err = (pm_future - tv) ** 2
            ax_err.plot(_elapsed_time_axis(prediction_length, step_hours), err, color=cmap[idx % len(cmap)],
                        alpha=0.6, lw=1.5, label=f"Series {idx + 1}")

        ax_err.set_title("Pointwise MSE: Prediction Error Over Horizon", fontsize=14, fontweight="bold")
        ax_err.set_xlabel("Elapsed Time [h]", fontweight="bold")
        ax_err.set_ylabel("Squared Error", fontweight="bold")
        ax_err.legend(fontsize=8, ncol=min(4, n_show))
        ax_err.grid(True, alpha=0.3)

        fig.suptitle(f"Training Batch Samples — Epoch {epoch + 1}", 
                    fontsize=16, fontweight="bold", y=1.01)
        fig.tight_layout()
        return fig
