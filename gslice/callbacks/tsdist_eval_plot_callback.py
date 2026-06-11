import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from pytorch_lightning import Callback

from gslice.utils.signal_utils import gp_regression
from tsdist.timeseries import TimeSeries


class TSDistEvaluationPlotCallback(Callback):
    """
    Plot conditional futures for tsdist experiments.

    For each selected ground-truth series:
      - Condition the true distribution on the observed history and sample multiple futures.
      - Fit a GP on the same history, sample multiple controls, and pass through the model.
    """

    def __init__(
        self,
        series_dataset,
        dist_class,
        distribution_kwargs=None,
        n_series: int = 4,
        n_future_samples: int = 20,
        n_gp_samples: int = 20,
        cutoff_portion: float = 0.5,
        plot_every_n_epochs: int = 10,
        save_dir: str = "./results/val_plots",
        seed: int = 0,
    ):
        super().__init__()
        self.series_dataset = series_dataset
        self.dist_class = dist_class
        self.distribution_kwargs = dict(distribution_kwargs or {})
        self.n_series = n_series
        self.n_future_samples = n_future_samples
        self.n_gp_samples = n_gp_samples
        self.cutoff_portion = cutoff_portion
        self.plot_every_n_epochs = plot_every_n_epochs
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.seed = int(seed)

        dataset_len = len(series_dataset)
        n_pick = min(n_series, dataset_len)
        rng = np.random.default_rng(self.seed)
        self.series_indices = rng.choice(dataset_len, size=n_pick, replace=False).tolist()
        self._warned_missing_hparams = False

    def _series_to_tensors(self, series, device):
        t = torch.as_tensor(series.times, device=device)
        y = torch.as_tensor(series.samples, device=device)
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        if y.dim() == 1:
            y = y.unsqueeze(-1)
        return t, y

    def on_validation_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch + 1) % self.plot_every_n_epochs != 0:
            return

        device = pl_module.device
        pl_module.eval()

        n_rows = len(self.series_indices)
        fig, axes = plt.subplots(
            n_rows,
            2,
            figsize=(14, max(4, 3 * n_rows)),
            sharex="col",
            sharey="row",
            squeeze=False,
        )

        with torch.no_grad():
            for row, series_idx in enumerate(self.series_indices):
                series = self.series_dataset[series_idx]
                t, y = self._series_to_tensors(series, device)
                t_vals = t[:, 0]

                t_cutoff = t_vals[0] + self.cutoff_portion * (t_vals[-1] - t_vals[0])
                obs_mask = t_vals <= t_cutoff
                obs_len = int(obs_mask.sum().item())
                if obs_len < 2:
                    continue

                t_centered = (t_vals - t_cutoff).unsqueeze(-1)
                t_past = t_centered[:obs_len]
                y_past = y[:obs_len]

                gp = gp_regression(t_past.unsqueeze(0), y_past.unsqueeze(0))
                gp_samples = gp.sample(t_centered.unsqueeze(0), num_samples=self.n_gp_samples)

                pred_samples = []
                for s in range(gp_samples.shape[0]):
                    pred = pl_module.inference_step(
                        t_centered.unsqueeze(0),
                        gp_samples[s],
                        mask_percentage=self.cutoff_portion,
                        already_centered=True,
                        is_derived_target=False,
                    )
                    pred_samples.append(pred.squeeze(0).detach().cpu())
                pred_samples = torch.stack(pred_samples, dim=0)  # (S, T, D)
                pred_mean = pred_samples.mean(dim=0)

                true_samples = None
                if self.dist_class is not None:
                    hparams = getattr(series, "metadata", {}).get("hyperparameters", None)
                    if hparams is None:
                        if not self._warned_missing_hparams:
                            print("TSDistEvaluationPlotCallback: missing hyperparameters metadata; skipping true futures.")
                            self._warned_missing_hparams = True
                    else:
                        dist = self.dist_class(**self.distribution_kwargs, **hparams)
                        series_cpu = series.to(device="cpu")
                        series_list = TimeSeries(
                            times=series_cpu.times.tolist(),
                            samples=series_cpu.samples.tolist(),
                            metadata=dict(series_cpu.metadata),
                        )
                        observed = series_list.slice_until(float(t_cutoff))
                        cond = dist.condition_on(observed)
                        rng = np.random.default_rng(self.seed + series_idx)
                        samples_list = []
                        for _ in range(self.n_future_samples):
                            future = cond.sample(random_generator=rng)
                            fut_samples = torch.as_tensor(future.samples)
                            if fut_samples.dim() == 1:
                                fut_samples = fut_samples.unsqueeze(-1)
                            obs_samples = torch.as_tensor(observed.samples)
                            if obs_samples.dim() == 1:
                                obs_samples = obs_samples.unsqueeze(-1)
                            full_samples = torch.cat([obs_samples, fut_samples[1:]], dim=0)
                            samples_list.append(full_samples)
                        true_samples = torch.stack(samples_list, dim=0)  # (S, T, D)

                t_cpu = t_vals.detach().cpu()
                y_cpu = y.detach().cpu()
                pred_samples_cpu = pred_samples.cpu()
                pred_mean_cpu = pred_mean.cpu()

                ax_true = axes[row, 0]
                ax_pred = axes[row, 1]

                ax_true.set_title("True Conditional Futures")
                ax_pred.set_title("Model Predicted Futures (GP -> LCDE)")

                ax_true.plot(
                    t_cpu[:obs_len],
                    y_cpu[:obs_len, 0],
                    color="black",
                    linewidth=2,
                    label="Observed",
                )
                ax_pred.plot(
                    t_cpu[:obs_len],
                    y_cpu[:obs_len, 0],
                    color="black",
                    linewidth=2,
                    label="Observed",
                )

                ax_true.plot(
                    t_cpu,
                    y_cpu[:, 0],
                    color="gray",
                    linewidth=1.5,
                    linestyle="--",
                    label="Ground Truth",
                )
                ax_pred.plot(
                    t_cpu,
                    y_cpu[:, 0],
                    color="gray",
                    linewidth=1.5,
                    linestyle="--",
                    label="Ground Truth",
                )
                ax_pred.plot(
                    t_cpu,
                    pred_mean_cpu[:, 0],
                    color="red",
                    linewidth=2,
                    label="Pred Mean",
                )

                future_start = max(obs_len - 1, 0)
                if true_samples is not None:
                    for s in range(true_samples.shape[0]):
                        ax_true.plot(
                            t_cpu[future_start:],
                            true_samples[s, future_start:, 0].cpu(),
                            color="tab:blue",
                            alpha=0.25,
                        )

                for s in range(pred_samples_cpu.shape[0]):
                    ax_pred.plot(
                        t_cpu,
                        pred_samples_cpu[s, :, 0],
                        color="tab:red",
                        alpha=0.2,
                    )

                ax_true.axvline(t_cutoff.item(), color="black", linestyle=":", linewidth=1.0)
                ax_pred.axvline(t_cutoff.item(), color="black", linestyle=":", linewidth=1.0)

                ax_true.set_xlabel("time")
                ax_pred.set_xlabel("time")
                ax_true.set_ylabel("value")

                if row == 0:
                    ax_true.legend(loc="upper left", fontsize=8)
                    ax_pred.legend(loc="upper left", fontsize=8)

        save_path = self.save_dir / f"tsdist_eval_epoch_{trainer.current_epoch:04d}.png"
        fig.tight_layout()
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)

        pl_module.train()
