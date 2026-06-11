"""Periodic evaluation + forecast-plot callback for GluonTS-compatible LCDE models.

Inspired by ``gslice.callbacks.eval_plot_callback.EvaluationPlotCallback``,
but adapted for the GluonTS forecasting loop (``make_evaluation_predictions``).

Every *eval_every* training epochs the callback:
1. Creates a :class:`PyTorchPredictor` from the current model weights.
2. Runs ``make_evaluation_predictions`` on (a subset of) the test dataset.
3. Computes standard GluonTS metrics (CRPS / ND / NRMSE).
4. Generates a multi-panel forecast plot and saves it as PNG.
"""

import inspect
import logging
from itertools import islice
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")  # non-interactive backend (safe for SLURM / headless)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from pytorch_lightning import Callback

from gluonts.dataset.common import ListDataset
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.loader import InferenceDataLoader
from gluonts.dataset.split import split
from gluonts.dataset.util import period_index
from gluonts.evaluation import Evaluator, make_evaluation_predictions
from gluonts.torch.batchify import batchify
from gluonts.itertools import Cached
from gluonts.model.forecast import SampleForecast
from gluonts.transform import Chain
from tqdm.auto import tqdm

from gslice.irregular import (
    FUTURE_TIME_GRID_FIELD,
    PAST_TIME_GRID_FIELD,
    has_irregular_grid,
    get_irregular_grid_spec,
    maybe_get_irregular_input_names,
)
from gslice.utils.transforms import IrregularInstanceTransform
from gslice.utils.util import create_splitter, filter_metrics, temporary_random_seed
from gslice.utils.variables import get_relative_time_step

log = logging.getLogger(__name__)


def _elapsed_time_axis(length: int, step_hours: float, *, start_hours: float = 0.0) -> np.ndarray:
    if length <= 0:
        return np.array([], dtype=np.float64)
    return start_hours + (np.arange(length, dtype=np.float64) + 1.0) * float(step_hours)


class GluonTSEvalPlotCallback(Callback):
    """Periodic evaluation & forecast-plot callback.

    Parameters
    ----------
    test_dataset
        Raw (un-transformed) GluonTS test dataset.
    transformation
        GluonTS transformation chain (the same one used for training).
    model_params : dict
        Must contain ``context_length``, ``prediction_length`` and ``freq``.
    setting : str
        Must be ``"univariate"``.
    eval_every : int
        Run evaluation every *N* training epochs.
    num_samples : int
        Number of forecast samples to draw.
    max_eval_instances : int | None
        Cap the number of test instances (``None`` = use all).
    batch_size : int
        Batch size for the predictor during evaluation.
    max_show : int
        Maximum number of individual series to show in the plot.
    save_dir : str | Path
        Directory where PNG plots are saved.
    enable_worst_crps_plot : bool
        Whether to save the extra worst-series forecast plot.
    enable_quantile_loss_plots : bool
        Whether to save the quantile-loss diagnostic plots.
    """

    def __init__(
        self,
        test_dataset,
        transformation,
        model_params: dict,
        setting: str = "univariate",
        eval_every: int = 20,
        num_samples: int = 16,
        max_eval_instances: Optional[int] = None,
        batch_size: int = 32,
        max_show: int = 8,
        save_dir: str = "./results/val_plots",
        enable_worst_crps_plot: bool = False,
        enable_quantile_loss_plots: bool = False,
        checkpoint_dir: str | Path | None = None,
        early_stopping_patience_epochs: Optional[int] = None,
        early_stopping_min_delta: float = 0.0,
        reduce_lr_on_plateau: bool = False,
        reduce_lr_patience_epochs: int = 0,
        reduce_lr_factor: float = 0.5,
        reduce_lr_min_lr: float = 1e-6,
        log_prefix: str = "val",
        eval_seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        if setting != "univariate":
            raise ValueError(f"Unsupported setting {setting!r}; only 'univariate' is supported.")
        self.test_dataset = test_dataset
        self.transformation = transformation
        self.model_params = model_params
        self.setting = setting
        self.eval_every = eval_every
        self.num_samples = num_samples
        self.max_eval_instances = max_eval_instances
        self.batch_size = batch_size
        self.max_show = max_show
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.enable_worst_crps_plot = bool(enable_worst_crps_plot)
        self.enable_quantile_loss_plots = bool(enable_quantile_loss_plots)
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir is not None else None
        self.early_stopping_patience_epochs = early_stopping_patience_epochs
        self.early_stopping_min_delta = early_stopping_min_delta
        self._best_crps: Optional[float] = None
        self._best_epoch: Optional[int] = None
        self._epochs_without_improvement = 0
        self.reduce_lr_on_plateau = bool(reduce_lr_on_plateau)
        self.reduce_lr_patience_epochs = int(reduce_lr_patience_epochs)
        self.reduce_lr_factor = float(reduce_lr_factor)
        self.reduce_lr_min_lr = float(reduce_lr_min_lr)
        self._lr_plateau_epochs_without_improvement = 0
        self.log_prefix = str(log_prefix)
        self.eval_seed = None if eval_seed is None else int(eval_seed)

    # ------------------------------------------------------------------ #
    # Lightning hook
    # ------------------------------------------------------------------ #
    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        if (epoch + 1) % self.eval_every != 0:
            return

        log.info("[EvalPlot] Running validation at epoch %d", epoch + 1)

        # Save training state and set model to eval mode
        was_training = pl_module.training
        pl_module.eval()

        try:
            with temporary_random_seed(self.eval_seed):
                self._run_evaluation(trainer, pl_module, epoch)
        finally:
            # Restore training state
            if was_training:
                pl_module.train()

    def _run_evaluation(self, trainer, pl_module, epoch):
        # ---- prepare test data (possibly limited) --------------------
        test_data = self.test_dataset
        if self.max_eval_instances is not None:
            limited = list(islice(test_data, self.max_eval_instances))
            test_data = ListDataset(limited, freq=self.model_params["freq"])

        transformed_testdata = self.transformation.apply(test_data, is_train=False)

        # Compute past_length based on model type
        source_context_length = int(self.model_params.get("source_context_length", self.model_params["context_length"]))
        source_prediction_length = int(self.model_params.get("source_prediction_length", self.model_params["prediction_length"]))
        if hasattr(pl_module, "lags_seq") and hasattr(pl_module, "prior_context_length"):
            # TSFlow models: account for lags and prior context
            max_lag = max(getattr(pl_module, "lags_seq", [0]) or [0])
            past_length = max(
                source_context_length + max_lag,
                pl_module.prior_context_length,
            )
        else:
            # LCDE or other models: include optional long-context window.
            long_context_length = int(self.model_params.get("long_context_length", 0))
            past_length = source_context_length + long_context_length

        uses_irregular_grid = bool(self.model_params.get("uses_irregular_grid", False)) or has_irregular_grid(
            self.model_params.get("dataset_params", {}) or {}
        )
        test_splitter = create_splitter(
            past_length=past_length,
            future_length=source_prediction_length if uses_irregular_grid else self.model_params["prediction_length"],
            mode="test",
            include_time_grid=uses_irregular_grid,
        )
        test_transform = self._build_test_transform(test_splitter) if uses_irregular_grid else test_splitter

        evaluator = Evaluator(num_workers=1)

        control_future = None
        context_forecast_map = None
        gp_mean_context_map = None
        gp_mean_future_map = None
        if self._supports_joint_forecast_control_pass(pl_module):
            try:
                (
                    forecasts,
                    tss,
                    control_future,
                    context_forecast_map,
                    gp_mean_context_map,
                    gp_mean_future_map,
                    time_grid_map,
                ) = self._make_evaluation_predictions_with_controls(
                    pl_module=pl_module,
                    dataset=transformed_testdata,
                    test_transform=test_transform,
                    uses_irregular_grid=uses_irregular_grid,
                )
            except Exception:
                log.exception("[EvalPlot] Joint forecast/control pass failed; falling back to predictor path")
                forecasts, tss, time_grid_map = self._make_standard_evaluation_predictions(
                    pl_module=pl_module,
                    dataset=transformed_testdata,
                    test_transform=test_transform,
                    uses_irregular_grid=uses_irregular_grid,
                )
        elif self._supports_plot_diagnostics_pass(pl_module):
            try:
                forecasts, tss, context_forecast_map, gp_mean_context_map, gp_mean_future_map, time_grid_map = (
                    self._make_evaluation_predictions_with_plot_diagnostics(
                        pl_module=pl_module,
                        dataset=transformed_testdata,
                        test_transform=test_transform,
                        uses_irregular_grid=uses_irregular_grid,
                    )
                )
            except Exception:
                log.exception("[EvalPlot] Joint forecast/diagnostic pass failed; falling back to predictor path")
                forecasts, tss, time_grid_map = self._make_standard_evaluation_predictions(
                    pl_module=pl_module,
                    dataset=transformed_testdata,
                    test_transform=test_transform,
                    uses_irregular_grid=uses_irregular_grid,
                )
        else:
            forecasts, tss, time_grid_map = self._make_standard_evaluation_predictions(
                pl_module=pl_module,
                dataset=transformed_testdata,
                test_transform=test_transform,
                uses_irregular_grid=uses_irregular_grid,
            )

        # ---- Check for NaN values ------------------------------------
        has_nan = any(np.isnan(fc.samples).any() for fc in forecasts)
        if has_nan:
            log.warning("[EvalPlot] Skipping evaluation at epoch %d - forecasts contain NaN values", epoch + 1)
            return

        # ---- metrics -------------------------------------------------
        try:
            metrics, metrics_per_ts = evaluator(tss, forecasts)
        except ValueError as e:
            if "NaN" in str(e):
                log.warning("[EvalPlot] Skipping evaluation at epoch %d - %s", epoch + 1, e)
                return
            raise
        metrics["CRPS"] = metrics["mean_wQuantileLoss"]
        # Keep the full metrics dict only when the diagnostic plots are enabled.
        full_metrics = dict(metrics) if self.enable_quantile_loss_plots else None
        select = ["CRPS", "ND", "NRMSE"]
        metrics = filter_metrics(metrics, select)

        for logger in trainer.loggers:
            logger.log_metrics(
                {f"{self.log_prefix}_{k}": v for k, v in metrics.items()},
                step=trainer.global_step,
            )
        log.info("[EvalPlot] Epoch %d metrics: %s", epoch + 1, metrics)

        # ---- early stopping on validation CRPS -----------------------
        if self.early_stopping_patience_epochs is not None and self.early_stopping_patience_epochs > 0:
            current_crps = float(metrics["CRPS"])
            improved = self._best_crps is None or (
                current_crps < (self._best_crps - self.early_stopping_min_delta)
            )
            if improved:
                self._best_crps = current_crps
                self._best_epoch = epoch + 1
                self._epochs_without_improvement = 0
            else:
                self._epochs_without_improvement += self.eval_every
                if self._epochs_without_improvement >= self.early_stopping_patience_epochs:
                    trainer.should_stop = True
                    log.info(
                        "[EvalPlot] Early stopping triggered at epoch %d "
                        "(best val_CRPS %.6f at epoch %d; no improvement for %d epochs, "
                        "patience=%d epochs).",
                        epoch + 1,
                        self._best_crps if self._best_crps is not None else float("nan"),
                        self._best_epoch if self._best_epoch is not None else -1,
                        self._epochs_without_improvement,
                        self.early_stopping_patience_epochs,
                    )
        else:
            current_crps = float(metrics["CRPS"])
            improved = self._best_crps is None or (
                current_crps < (self._best_crps - self.early_stopping_min_delta)
            )
            if improved:
                self._best_crps = current_crps
                self._best_epoch = epoch + 1

        if improved and self.checkpoint_dir is not None and self.log_prefix == "val":
            ckpt_path = self.checkpoint_dir / "best_checkpoint.ckpt"
            torch.save(pl_module.state_dict(), ckpt_path)
            log.info("[EvalPlot] Saved best checkpoint to %s", ckpt_path)

        # ---- reduce LR on val_CRPS plateau ---------------------------
        if self.reduce_lr_on_plateau and self.reduce_lr_patience_epochs > 0:
            if improved:
                self._lr_plateau_epochs_without_improvement = 0
            else:
                self._lr_plateau_epochs_without_improvement += self.eval_every

            if self._lr_plateau_epochs_without_improvement >= self.reduce_lr_patience_epochs:
                for opt_idx, optimizer in enumerate(trainer.optimizers):
                    old_lrs = [float(pg.get("lr", 0.0)) for pg in optimizer.param_groups]
                    new_lrs = [max(old_lr * self.reduce_lr_factor, self.reduce_lr_min_lr) for old_lr in old_lrs]
                    changed = any(new_lr < old_lr for old_lr, new_lr in zip(old_lrs, new_lrs))
                    if changed:
                        for param_group, new_lr in zip(optimizer.param_groups, new_lrs):
                            param_group["lr"] = new_lr
                        log.info(
                            "[EvalPlot] ReduceLROnPlateau triggered at epoch %d on optimizer %d: "
                            "old_lrs=%s -> new_lrs=%s (best val_CRPS %.6f, plateau_epochs=%d, patience=%d).",
                            epoch + 1,
                            opt_idx,
                            [f"{lr:.8g}" for lr in old_lrs],
                            [f"{lr:.8g}" for lr in new_lrs],
                            self._best_crps if self._best_crps is not None else float("nan"),
                            self._lr_plateau_epochs_without_improvement,
                            self.reduce_lr_patience_epochs,
                        )
                        for logger in trainer.loggers:
                            logger.log_metrics(
                                {f"lr_opt{opt_idx}": new_lrs[0]},
                                step=trainer.global_step,
                            )
                self._lr_plateau_epochs_without_improvement = 0

        default_indices = list(range(min(len(forecasts), self.max_show)))

        # ---- plot ----------------------------------------------------
        try:
            fig = self._plot_forecasts(
                forecasts,
                tss,
                metrics_per_ts,
                epoch,
                control_future=control_future,
                context_forecast_map=context_forecast_map,
                gp_mean_context_map=gp_mean_context_map,
                gp_mean_future_map=gp_mean_future_map,
                time_grid_map=time_grid_map,
                selected_indices=default_indices,
                title_prefix="Validation",
            )
            save_path = self.save_dir / f"val_forecast_epoch_{epoch + 1:04d}.png"
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            log.info("[EvalPlot] Plot saved to %s", save_path)
        except Exception:
            log.exception("[EvalPlot] Failed to generate plot at epoch %d", epoch + 1)

        if self.enable_worst_crps_plot:
            try:
                worst_indices = self._select_worst_crps_indices(metrics_per_ts, len(forecasts))
                if len(worst_indices) > 0:
                    fig_worst = self._plot_forecasts(
                        forecasts,
                        tss,
                        metrics_per_ts,
                        epoch,
                        control_future=control_future,
                        time_grid_map=time_grid_map,
                        selected_indices=worst_indices,
                        title_prefix="Worst-CRPS Validation",
                    )
                    worst_path = self.save_dir / f"val_forecast_worst_crps_epoch_{epoch + 1:04d}.png"
                    fig_worst.savefig(worst_path, dpi=150, bbox_inches="tight")
                    plt.close(fig_worst)
                    log.info("[EvalPlot] Worst-CRPS plot saved to %s", worst_path)
            except Exception:
                log.exception("[EvalPlot] Failed to generate worst-CRPS plot at epoch %d", epoch + 1)

        if self.enable_quantile_loss_plots:
            try:
                fig_ql = self._plot_quantile_loss_distribution(full_metrics, epoch)
                ql_path = self.save_dir / f"val_wQuantileLoss_epoch_{epoch + 1:04d}.png"
                fig_ql.savefig(ql_path, dpi=150, bbox_inches="tight")
                plt.close(fig_ql)
                log.info("[EvalPlot] wQuantileLoss distribution plot saved to %s", ql_path)
            except Exception:
                log.exception("[EvalPlot] Failed to generate wQuantileLoss plot at epoch %d", epoch + 1)

            try:
                fig_ps = self._plot_per_sample_quantile_loss(
                    metrics_per_ts,
                    epoch,
                    forecasts=forecasts,
                    tss=tss,
                )
                ps_path = self.save_dir / f"val_wQuantileLoss_per_sample_epoch_{epoch + 1:04d}.png"
                fig_ps.savefig(ps_path, dpi=150, bbox_inches="tight")
                plt.close(fig_ps)
                log.info("[EvalPlot] Per-sample wQuantileLoss plot saved to %s", ps_path)
            except Exception:
                log.exception("[EvalPlot] Failed to generate per-sample wQuantileLoss plot at epoch %d", epoch + 1)

    def _build_test_transform(self, test_splitter):
        if not (bool(self.model_params.get("uses_irregular_grid", False)) or has_irregular_grid(self.model_params.get("dataset_params", {}) or {})):
            return test_splitter

        spec = get_irregular_grid_spec(self.model_params.get("dataset_params", {}), fallback_freq=self.model_params["freq"])
        return Chain(
            [
                test_splitter,
                IrregularInstanceTransform(
                    irregular_spec=spec,
                    lag_steps=list(self.model_params.get("lags_seq", []) or []),
                ),
            ]
        )

    def _transform_eval_input_entry(self, input_entry, test_transform):
        transformed = list(test_transform.apply([input_entry], is_train=False))
        if not transformed:
            raise ValueError("Validation transform did not yield any entries.")
        return transformed[0]

    def _eval_source_prediction_length(self) -> int:
        return int(self.model_params.get("source_prediction_length", self.model_params["prediction_length"]))

    def _build_irregular_eval_entry(self, input_label):
        input_entry, label_entry = input_label
        dense_past_target = np.asarray(input_entry[FieldName.TARGET], dtype=np.float32)
        dense_future_target = np.asarray(label_entry[FieldName.TARGET], dtype=np.float32)
        full_time_grid = np.asarray(input_entry["time_grid"], dtype=np.float32)
        full_observed = np.asarray(
            input_entry.get(
                "observed_values",
                np.ones((dense_past_target.shape[0] + dense_future_target.shape[0],), dtype=np.float32),
            ),
            dtype=np.float32,
        )
        past_len = int(dense_past_target.shape[0])
        future_len = int(dense_future_target.shape[0])
        dense_entry = {
            "past_target": dense_past_target,
            "future_target": dense_future_target,
            PAST_TIME_GRID_FIELD: full_time_grid[:past_len],
            FUTURE_TIME_GRID_FIELD: full_time_grid[past_len : past_len + future_len],
            "past_observed_values": full_observed[:past_len],
            "future_observed_values": full_observed[past_len : past_len + future_len],
            "mean": np.asarray(input_entry.get("mean"), dtype=np.float32),
            FieldName.ITEM_ID: input_entry.get(FieldName.ITEM_ID, label_entry.get(FieldName.ITEM_ID)),
            FieldName.FORECAST_START: label_entry[FieldName.START],
            FieldName.START: input_entry[FieldName.START],
        }
        if "time_feat" in input_entry:
            full_time_feat = np.asarray(input_entry["time_feat"])
            dense_entry["past_feat_time"] = np.swapaxes(full_time_feat[..., :past_len], 0, -1)
            dense_entry["future_feat_time"] = np.swapaxes(full_time_feat[..., past_len : past_len + future_len], 0, -1)
        transform = IrregularInstanceTransform(
            irregular_spec=get_irregular_grid_spec(self.model_params.get("dataset_params", {}), fallback_freq=self.model_params["freq"]),
            lag_steps=list(self.model_params.get("lags_seq", []) or []),
        )
        return transform.map_transform(dense_entry, is_train=False)

    def _chunked_batchify(self, entries):
        for start in range(0, len(entries), self.batch_size):
            yield batchify(entries[start : start + self.batch_size])

    def _make_standard_evaluation_predictions(self, pl_module, dataset, test_transform, uses_irregular_grid: bool):
        predictor = pl_module.get_predictor(
            input_transform=test_transform,
            batch_size=self.batch_size,
            input_names=maybe_get_irregular_input_names(
                self.model_params.get("dataset_params", {}) or {},
                use_lags=bool(self.model_params.get("use_lags", True)),
            ),
        )

        window_length = self._eval_source_prediction_length() if uses_irregular_grid else self.model_params["prediction_length"]
        _, test_template = split(dataset, offset=-window_length)
        test_data = test_template.generate_instances(window_length)
        test_pairs = list(test_data)
        transformed_entries = [
            self._build_irregular_eval_entry(pair) if uses_irregular_grid else None
            for pair in test_pairs
        ]

        forecast_it, _ = make_evaluation_predictions(
            dataset=dataset,
            predictor=predictor,
            num_samples=self.num_samples,
        )
        forecasts = list(tqdm(forecast_it, total=len(dataset), desc="Eval forecasts"))
        if uses_irregular_grid:
            tss = [self._transformed_entry_to_dataframe(entry) for entry in transformed_entries]
            time_grid_map = {
                idx: self._extract_series_time_grids(entry)
                for idx, entry in enumerate(transformed_entries)
                if entry is not None
            }
        else:
            tss = [self._test_pair_to_dataframe(pair) for pair in test_pairs]
            time_grid_map = None
        return forecasts, tss, time_grid_map

    def _supports_joint_forecast_control_pass(self, pl_module) -> bool:
        predict_fn = getattr(pl_module, "predict_samples_from_past", None)
        if predict_fn is None:
            return False
        try:
            return "return_control" in inspect.signature(predict_fn).parameters
        except (TypeError, ValueError):
            return False

    def _supports_plot_diagnostics_pass(self, pl_module) -> bool:
        diag_fn = getattr(pl_module, "predict_plot_diagnostics_from_past", None)
        return callable(diag_fn)

    def _make_evaluation_predictions_with_controls(self, pl_module, dataset, test_transform, uses_irregular_grid: bool):
        """Run a single model-inference pass that returns forecasts, controls, and diagnostics."""
        window_length = self._eval_source_prediction_length() if uses_irregular_grid else self.model_params["prediction_length"]
        _, test_template = split(dataset, offset=-window_length)
        test_data = test_template.generate_instances(window_length)
        test_pairs = list(test_data)
        if not test_pairs:
            return [], [], None, None, None, None, None

        input_entries = [input_entry for input_entry, _ in test_pairs]
        transformed_entries = [
            self._build_irregular_eval_entry(pair) if uses_irregular_grid else None
            for pair in test_pairs
        ]
        tss = [
            self._transformed_entry_to_dataframe(entry) if uses_irregular_grid else self._test_pair_to_dataframe(pair)
            for entry, pair in zip(transformed_entries, test_pairs)
        ]

        forecasts: list[SampleForecast] = []
        controls: list[np.ndarray] = []
        context_forecast_map: dict[int, np.ndarray] = {}
        gp_mean_context_map: dict[int, np.ndarray] = {}
        gp_mean_future_map: dict[int, np.ndarray] = {}
        time_grid_map: dict[int, dict[str, np.ndarray]] = {}
        cursor = 0
        total_batches = (len(transformed_entries if uses_irregular_grid else input_entries) + self.batch_size - 1) // self.batch_size

        if uses_irregular_grid:
            data_loader = self._chunked_batchify(transformed_entries)
        else:
            data_loader = InferenceDataLoader(
                Cached(input_entries),
                batch_size=self.batch_size,
                stack_fn=batchify,
                transform=test_transform,
            )

        with torch.no_grad():
            for batch in tqdm(data_loader, total=total_batches, desc="Eval forecast/control batches"):
                past_target = torch.as_tensor(batch["past_target"], dtype=torch.float32, device=pl_module.device)
                past_observed_values = batch.get("past_observed_values", None)
                if past_observed_values is not None:
                    past_observed_values = torch.as_tensor(
                        past_observed_values,
                        dtype=torch.float32,
                        device=pl_module.device,
                    )
                mean = batch.get("mean", None)
                if mean is not None:
                    mean = torch.as_tensor(mean, dtype=torch.float32, device=pl_module.device)
                scale = batch.get("scale", None)
                if scale is not None:
                    scale = torch.as_tensor(scale, dtype=torch.float32, device=pl_module.device)
                past_time_grid = batch.get(PAST_TIME_GRID_FIELD, None)
                if past_time_grid is not None:
                    past_time_grid = torch.as_tensor(past_time_grid, dtype=torch.float32, device=pl_module.device)
                future_time_grid = batch.get(FUTURE_TIME_GRID_FIELD, None)
                if future_time_grid is not None:
                    future_time_grid = torch.as_tensor(future_time_grid, dtype=torch.float32, device=pl_module.device)
                lag_features = batch.get("lag_features", None)
                if lag_features is not None:
                    lag_features = torch.as_tensor(lag_features, dtype=torch.float32, device=pl_module.device)
                dense_past_target = batch.get("dense_past_target", None)
                if dense_past_target is not None:
                    dense_past_target = torch.as_tensor(dense_past_target, dtype=torch.float32, device=pl_module.device)
                dense_past_observed_values = batch.get("dense_past_observed_values", None)
                if dense_past_observed_values is not None:
                    dense_past_observed_values = torch.as_tensor(
                        dense_past_observed_values,
                        dtype=torch.float32,
                        device=pl_module.device,
                    )
                dense_past_time_grid = batch.get("dense_past_time_grid", None)
                if dense_past_time_grid is not None:
                    dense_past_time_grid = torch.as_tensor(dense_past_time_grid, dtype=torch.float32, device=pl_module.device)

                forecast_batch, control_batch = pl_module.predict_samples_from_past(
                    past_target=past_target,
                    past_observed_values=past_observed_values,
                    mean=mean,
                    scale=scale,
                    past_time_grid=past_time_grid,
                    future_time_grid=future_time_grid,
                    lag_features=lag_features,
                    dense_past_target=dense_past_target,
                    dense_past_observed_values=dense_past_observed_values,
                    dense_past_time_grid=dense_past_time_grid,
                    num_samples=self.num_samples,
                    return_control=True,
                )
                diagnostics = None
                if hasattr(pl_module, "predict_plot_diagnostics_from_past"):
                    diagnostics = pl_module.predict_plot_diagnostics_from_past(
                        past_target=past_target,
                        past_observed_values=past_observed_values,
                        num_samples=self.num_samples,
                        mean=mean,
                        past_time_grid=past_time_grid,
                        future_time_grid=future_time_grid,
                        lag_features=lag_features,
                        dense_past_target=dense_past_target,
                        dense_past_observed_values=dense_past_observed_values,
                        dense_past_time_grid=dense_past_time_grid,
                    )

                forecast_np = forecast_batch.detach().cpu().numpy()  # (S, B, H, D)
                control_np = control_batch.detach().cpu().numpy()  # (S, B, H, D)
                batch_n = int(forecast_np.shape[1])
                batch_pairs = test_pairs[cursor : cursor + batch_n]

                for idx, (_, label_entry) in enumerate(batch_pairs):
                    samples = forecast_np[:, idx, :, :]
                    if samples.shape[-1] == 1:
                        samples = np.squeeze(samples, axis=-1)
                    forecasts.append(
                        SampleForecast(
                            samples=samples,
                            start_date=label_entry[FieldName.START],
                            item_id=label_entry.get(FieldName.ITEM_ID),
                        )
                    )
                    global_idx = cursor + idx
                    if diagnostics is not None:
                        context_forecast_map[global_idx] = diagnostics["context_forecast_samples"][idx].detach().cpu().numpy()
                        gp_mean_context_map[global_idx] = diagnostics["gp_mean_context"][idx].detach().cpu().numpy()
                        gp_mean_future_map[global_idx] = diagnostics["gp_mean_future"][idx].detach().cpu().numpy()
                    if uses_irregular_grid and transformed_entries[global_idx] is not None:
                        time_grid_map[global_idx] = self._extract_series_time_grids(transformed_entries[global_idx])

                controls.append(control_np)
                cursor += batch_n

        control_all = np.concatenate(controls, axis=1)[:, : len(forecasts), :, :]
        if control_all.shape[-1] > 1:
            control_all = control_all[..., 0]
        else:
            control_all = np.squeeze(control_all, axis=-1)
        return forecasts, tss, control_all, context_forecast_map, gp_mean_context_map, gp_mean_future_map, time_grid_map

    def _make_evaluation_predictions_with_plot_diagnostics(self, pl_module, dataset, test_transform, uses_irregular_grid: bool):
        """Run a single inference pass that returns forecasts plus TSFlow diagnostics."""
        window_length = self._eval_source_prediction_length() if uses_irregular_grid else self.model_params["prediction_length"]
        _, test_template = split(dataset, offset=-window_length)
        test_data = test_template.generate_instances(window_length)
        test_pairs = list(test_data)
        if not test_pairs:
            return [], [], None, None, None, None

        input_entries = [input_entry for input_entry, _ in test_pairs]
        transformed_entries = [
            self._build_irregular_eval_entry(pair) if uses_irregular_grid else None
            for pair in test_pairs
        ]
        tss = [
            self._transformed_entry_to_dataframe(entry) if uses_irregular_grid else self._test_pair_to_dataframe(pair)
            for entry, pair in zip(transformed_entries, test_pairs)
        ]

        forecasts: list[SampleForecast] = []
        context_forecast_map: dict[int, np.ndarray] = {}
        gp_mean_context_map: dict[int, np.ndarray] = {}
        gp_mean_future_map: dict[int, np.ndarray] = {}
        time_grid_map: dict[int, dict[str, np.ndarray]] = {}
        cursor = 0
        total_batches = (len(transformed_entries if uses_irregular_grid else input_entries) + self.batch_size - 1) // self.batch_size
        old_num_samples = getattr(pl_module, "num_samples", None)

        if uses_irregular_grid:
            data_loader = self._chunked_batchify(transformed_entries)
        else:
            data_loader = InferenceDataLoader(
                Cached(input_entries),
                batch_size=self.batch_size,
                stack_fn=batchify,
                transform=test_transform,
            )

        try:
            if old_num_samples is not None:
                pl_module.num_samples = int(self.num_samples)

            with torch.no_grad():
                for batch in tqdm(data_loader, total=total_batches, desc="Eval forecast/diagnostic batches"):
                    past_target = torch.as_tensor(batch["past_target"], dtype=torch.float32, device=pl_module.device)
                    past_observed_values = torch.as_tensor(
                        batch["past_observed_values"],
                        dtype=torch.float32,
                        device=pl_module.device,
                    )
                    mean = batch.get("mean", None)
                    if mean is not None:
                        mean = torch.as_tensor(mean, dtype=torch.float32, device=pl_module.device)
                    past_time_grid = batch.get(PAST_TIME_GRID_FIELD, None)
                    if past_time_grid is not None:
                        past_time_grid = torch.as_tensor(past_time_grid, dtype=torch.float32, device=pl_module.device)
                    future_time_grid = batch.get(FUTURE_TIME_GRID_FIELD, None)
                    if future_time_grid is not None:
                        future_time_grid = torch.as_tensor(future_time_grid, dtype=torch.float32, device=pl_module.device)
                    lag_features = batch.get("lag_features", None)
                    if lag_features is not None:
                        lag_features = torch.as_tensor(lag_features, dtype=torch.float32, device=pl_module.device)
                    dense_past_target = batch.get("dense_past_target", None)
                    if dense_past_target is not None:
                        dense_past_target = torch.as_tensor(dense_past_target, dtype=torch.float32, device=pl_module.device)
                    dense_past_observed_values = batch.get("dense_past_observed_values", None)
                    if dense_past_observed_values is not None:
                        dense_past_observed_values = torch.as_tensor(
                            dense_past_observed_values,
                            dtype=torch.float32,
                            device=pl_module.device,
                        )
                    dense_past_time_grid = batch.get("dense_past_time_grid", None)
                    if dense_past_time_grid is not None:
                        dense_past_time_grid = torch.as_tensor(
                            dense_past_time_grid,
                            dtype=torch.float32,
                            device=pl_module.device,
                        )

                    forecast_batch = pl_module.forward(
                        past_target=past_target,
                        past_observed_values=past_observed_values,
                        mean=mean,
                        past_time_grid=past_time_grid,
                        future_time_grid=future_time_grid,
                        lag_features=lag_features,
                        dense_past_target=dense_past_target,
                        dense_past_observed_values=dense_past_observed_values,
                        dense_past_time_grid=dense_past_time_grid,
                    )
                    diagnostics = pl_module.predict_plot_diagnostics_from_past(
                        past_target=past_target,
                        past_observed_values=past_observed_values,
                        num_samples=self.num_samples,
                        mean=mean,
                        past_time_grid=past_time_grid,
                        future_time_grid=future_time_grid,
                        lag_features=lag_features,
                        dense_past_target=dense_past_target,
                        dense_past_observed_values=dense_past_observed_values,
                        dense_past_time_grid=dense_past_time_grid,
                    )

                    forecast_np = forecast_batch.detach().cpu().numpy()
                    batch_n = int(forecast_np.shape[0])
                    batch_pairs = test_pairs[cursor : cursor + batch_n]

                    for idx, (_, label_entry) in enumerate(batch_pairs):
                        samples = forecast_np[idx]
                        if samples.ndim == 3 and samples.shape[-1] == 1:
                            samples = np.squeeze(samples, axis=-1)
                        forecasts.append(
                            SampleForecast(
                                samples=samples,
                                start_date=label_entry[FieldName.START],
                                item_id=label_entry.get(FieldName.ITEM_ID),
                            )
                        )

                        global_idx = cursor + idx
                        context_forecast_map[global_idx] = diagnostics["context_forecast_samples"][idx].detach().cpu().numpy()
                        gp_mean_context_map[global_idx] = diagnostics["gp_mean_context"][idx].detach().cpu().numpy()
                        gp_mean_future_map[global_idx] = diagnostics["gp_mean_future"][idx].detach().cpu().numpy()
                        if uses_irregular_grid and transformed_entries[global_idx] is not None:
                            time_grid_map[global_idx] = self._extract_series_time_grids(transformed_entries[global_idx])

                    cursor += batch_n
        finally:
            if old_num_samples is not None:
                pl_module.num_samples = old_num_samples

        return forecasts, tss, context_forecast_map, gp_mean_context_map, gp_mean_future_map, time_grid_map

    def _test_pair_to_dataframe(self, input_label) -> pd.DataFrame:
        """Mirror GluonTS backtest._to_dataframe without an extra predictor pass."""
        start = input_label[0][FieldName.START]
        targets = [entry[FieldName.TARGET] for entry in input_label]
        full_target = np.concatenate(targets, axis=-1)
        index = period_index({FieldName.START: start, FieldName.TARGET: full_target})
        return pd.DataFrame(full_target.transpose(), index=index)

    def _transformed_entry_to_dataframe(self, entry) -> pd.DataFrame:
        past_target = np.asarray(entry["past_target"])
        future_target = np.asarray(entry["future_target"])
        full_target = np.concatenate([past_target, future_target], axis=-1)

        forecast_start = entry.get(FieldName.FORECAST_START)
        start = entry[FieldName.START]
        if forecast_start is not None:
            start = forecast_start - int(past_target.shape[-1])

        index = period_index({FieldName.START: start, FieldName.TARGET: full_target})
        return pd.DataFrame(full_target.transpose(), index=index)

    def _extract_series_time_grids(self, entry) -> dict[str, np.ndarray]:
        context_length = int(self.model_params["context_length"])
        past_time_grid = np.asarray(entry[PAST_TIME_GRID_FIELD], dtype=np.float64).reshape(-1)
        future_time_grid = np.asarray(entry[FUTURE_TIME_GRID_FIELD], dtype=np.float64).reshape(-1)
        return {
            "context_time": past_time_grid[-context_length:].copy(),
            "future_time": future_time_grid.copy(),
        }

    # ------------------------------------------------------------------ #
    # Plotting
    # ------------------------------------------------------------------ #

    def _plot_future_controls(self, control_future: np.ndarray, epoch: int):
        """Plot control trajectories over the forecast horizon only.

        Args:
            control_future: (S, N, prediction_length), where S=num_samples, N=num_series.
        """
        n_show = min(control_future.shape[1], self.max_show)
        prediction_length = control_future.shape[2]
        t_h = np.arange(prediction_length)
        cmap = plt.colormaps["tab20"].colors

        n_indiv_rows = (n_show + 1) // 2
        n_rows = 1 + n_indiv_rows
        fig = plt.figure(figsize=(18, 3.2 * n_rows))
        gs = fig.add_gridspec(n_rows, 2, hspace=0.45, wspace=0.30)

        # ---- Row 0: aggregated control mean ± std over shown series ----
        ax_mean = fig.add_subplot(gs[0, :])
        shown = control_future[:, :n_show, :]  # (S, N, H)
        per_series_mean = shown.mean(axis=0)  # (N, H)
        agg_mean = per_series_mean.mean(axis=0)
        agg_std = per_series_mean.std(axis=0)
        ax_mean.plot(t_h, agg_mean, color="tab:blue", lw=2.0, label="Control Mean")
        ax_mean.fill_between(
            t_h,
            agg_mean - agg_std,
            agg_mean + agg_std,
            color="tab:blue",
            alpha=0.25,
            label="Series mean ±1σ",
        )
        ax_mean.set_title(
            f"Future-Region Control (Aggregated) — Epoch {epoch + 1}",
            fontsize=14,
            fontweight="bold",
        )
        ax_mean.set_xlabel("Forecast Horizon")
        ax_mean.set_ylabel("Control Value")
        ax_mean.grid(True, alpha=0.3)
        ax_mean.legend(fontsize=9)

        # ---- Individual series controls (2 cols) ----
        for idx in range(n_show):
            row = 1 + idx // 2
            col = idx % 2
            ax = fig.add_subplot(gs[row, col])

            samples = control_future[:, idx, :]  # (S, H)
            ctrl_mean = samples.mean(axis=0)
            ctrl_q10 = np.quantile(samples, 0.1, axis=0)
            ctrl_q90 = np.quantile(samples, 0.9, axis=0)
            c = cmap[idx % len(cmap)]

            ax.plot(t_h, ctrl_mean, color=c, lw=2.0, label="Control Mean")
            ax.fill_between(t_h, ctrl_q10, ctrl_q90, color=c, alpha=0.25, label="10–90 %")
            ax.set_title(f"Series {idx + 1}", fontsize=10)
            ax.set_xlabel("Forecast Horizon")
            ax.set_ylabel("Control")
            ax.grid(True, alpha=0.3)
            if idx == 0:
                ax.legend(fontsize=8)

        fig.suptitle(f"Validation Future Control — Epoch {epoch + 1}", fontsize=16, fontweight="bold", y=1.01)
        fig.tight_layout()
        return fig

    def _plot_per_sample_quantile_loss(self, metrics_per_ts, epoch: int,
                                       forecasts=None, tss=None):
        """Per-time-step, per-series *weighted* quantile loss.

        Each per-timestep pinball loss is divided by ``abs_target_sum``
        (the same normalisation the GluonTS ``Evaluator`` uses for
        ``wQuantileLoss``), so that the values are comparable across
        series of different scale.

        Layout
        ------
        - **Row 0** (full width): Heatmap of *all* series × horizon steps.
          Each cell is the wQuantileLoss summed over quantile levels for that
          series/time-step.
        - **Rows 1‥N** (2 columns): One subplot per series (up to
          ``max_show``).  Each subplot shows 9 curves (one per quantile level)
          of the weighted pinball loss at every horizon step.  The per-series
          CRPS (= mean over quantiles of ``wQuantileLoss[q]``) is shown in
          the panel title.
        """
        if forecasts is None or tss is None:
            raise ValueError("forecasts and tss must be provided")

        prediction_length = self.model_params["prediction_length"]
        quantile_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        n_series = len(forecasts)
        n_show = min(n_series, self.max_show)

        # ---- helpers ----
        def _ts_values(ts):
            v = ts if isinstance(ts, np.ndarray) else ts.values
            if v.ndim > 1:
                v = v[:, 0]
            return v.astype(np.float64)

        def _fc_samples(fc):
            s = fc.samples
            if s.ndim > 2:
                s = s[:, :, 0]
            return s.astype(np.float64)

        # ---- per-series abs_target_sum from evaluator (for normalisation) ----
        abs_target_sums = np.clip(
            metrics_per_ts["abs_target_sum"].to_numpy(dtype=np.float64),
            1e-12,
            None,
        )

        # ---- compute per-timestep wQuantileLoss for every series ----
        # wql_all: (n_series, n_quantiles, prediction_length)
        # wql_all[i, j, t] = pinball_loss(t, q_j) / abs_target_sum_i
        # so that  sum_t wql_all[i, j, :]  ==  wQuantileLoss[q_j] for series i
        #
        # We use forecast.quantile(q) (nearest-rank) rather than
        # np.quantile(samples, q) (linear interpolation) to stay consistent
        # with GluonTS SampleForecast.quantile / Evaluator.
        wql_all = np.full((n_series, len(quantile_levels), prediction_length), np.nan)
        for i in range(n_series):
            target = _ts_values(tss[i])[-prediction_length:]
            for j, q in enumerate(quantile_levels):
                forecast_q = np.squeeze(forecasts[i].quantile(q)).astype(np.float64)
                raw_ql = 2.0 * np.abs(
                    (forecast_q - target) * ((target <= forecast_q).astype(float) - q)
                )
                # normalise like evaluator: wQuantileLoss = QuantileLoss / abs_target_sum
                wql_all[i, j, :] = raw_ql / abs_target_sums[i]

        # Per-series CRPS = mean over quantiles of wQuantileLoss[q]
        #                  = mean_q( sum_t( wql_all[i, q, :] ) )
        crps_by_series = wql_all.sum(axis=2).mean(axis=1)

        # ---- figure ----
        n_indiv_rows = (n_show + 1) // 2
        n_rows = 1 + n_indiv_rows  # heatmap + individual panels
        fig = plt.figure(figsize=(18, 3.5 * n_rows))
        gs = fig.add_gridspec(n_rows, 2, hspace=0.45, wspace=0.30)
        cmap_q = plt.colormaps["viridis"](np.linspace(0, 1, len(quantile_levels)))
        t_steps = np.arange(prediction_length)

        # ---- Row 0: overview heatmap (all series, summed over quantiles) ----
        ax_heat = fig.add_subplot(gs[0, :])
        # Sum over quantile levels → (n_series, prediction_length)
        wql_sum = wql_all.sum(axis=1)
        im = ax_heat.imshow(wql_sum, aspect="auto", cmap="YlOrRd",
                            interpolation="nearest")
        ax_heat.set_xlabel("Forecast Horizon Step", fontweight="bold")
        ax_heat.set_ylabel("Series Index", fontweight="bold")
        ax_heat.set_title(
            f"wQuantileLoss per Series × Horizon Step (summed over q) — Epoch {epoch + 1}",
            fontsize=13, fontweight="bold",
        )
        cbar = fig.colorbar(im, ax=ax_heat, shrink=0.8)
        cbar.set_label("Σ wQuantileLoss", fontweight="bold")

        # ---- Individual panels: one per series, 2 cols ----
        for idx in range(n_show):
            row = 1 + idx // 2
            col = idx % 2
            ax = fig.add_subplot(gs[row, col])

            for j, q in enumerate(quantile_levels):
                ax.plot(t_steps, wql_all[idx, j, :], lw=1.5, color=cmap_q[j],
                        label=f"q={q:.1f}", alpha=0.85)

            ax.set_xlabel("Horizon Step")
            ax.set_ylabel("wQuantileLoss")
            series_label = idx + 1
            if "item_id" in metrics_per_ts.columns and idx < len(metrics_per_ts):
                item_id = metrics_per_ts["item_id"].iloc[idx]
                if item_id is not None:
                    series_label = item_id
            ax.set_title(f"Series {series_label} | CRPS={crps_by_series[idx]:.4f}", fontsize=10)
            ax.grid(True, alpha=0.3)
            if idx == 0:
                ax.legend(fontsize=7, ncol=3, loc="upper left")

        fig.suptitle(
            f"Per-Value wQuantileLoss — Epoch {epoch + 1}",
            fontsize=15, fontweight="bold", y=1.01,
        )
        fig.tight_layout()
        return fig

    def _plot_quantile_loss_distribution(self, full_metrics: dict, epoch: int):
        """Bar chart of wQuantileLoss at each quantile level.

        Shows the per-quantile weighted quantile loss whose mean equals CRPS
        (``mean_wQuantileLoss``).  A horizontal dashed line marks the mean.
        """
        import re

        # Extract wQuantileLoss[<q>] entries
        pattern = re.compile(r"^wQuantileLoss\[(.+)\]$")
        q_vals, losses = [], []
        for key, val in sorted(full_metrics.items()):
            m = pattern.match(key)
            if m:
                q_vals.append(float(m.group(1)))
                losses.append(val)

        if not q_vals:
            raise ValueError("No wQuantileLoss entries found in metrics")

        mean_wql = full_metrics.get("mean_wQuantileLoss", np.mean(losses))

        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.bar(
            [f"{q:.1f}" for q in q_vals],
            losses,
            color="steelblue",
            edgecolor="white",
            width=0.6,
        )
        ax.axhline(mean_wql, color="crimson", ls="--", lw=1.5,
                   label=f"mean (CRPS) = {mean_wql:.4f}")

        # Annotate each bar with its value
        for bar, loss in zip(bars, losses):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{loss:.4f}",
                ha="center", va="bottom", fontsize=8,
            )

        ax.set_xlabel("Quantile level", fontweight="bold")
        ax.set_ylabel("Weighted Quantile Loss", fontweight="bold")
        ax.set_title(
            f"wQuantileLoss Distribution — Epoch {epoch + 1}",
            fontsize=13, fontweight="bold",
        )
        ax.legend(fontsize=10)
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        return fig

    def _plot_forecasts(
        self,
        forecasts,
        tss,
        metrics_per_ts,
        epoch: Optional[int],
        control_future: Optional[np.ndarray] = None,
        context_forecast_map: Optional[dict[int, np.ndarray]] = None,
        gp_mean_context_map: Optional[dict[int, np.ndarray]] = None,
        gp_mean_future_map: Optional[dict[int, np.ndarray]] = None,
        time_grid_map: Optional[dict[int, dict[str, np.ndarray]]] = None,
        selected_indices: Optional[list[int]] = None,
        title_prefix: str = "Validation",
        show_aggregate_panel: bool = True,
        show_error_panel: bool = True,
        compact_mode: bool = False,
    ):
        """Create a multi-panel forecast visualisation."""
        epoch_suffix = f" — Epoch {epoch + 1}" if epoch is not None else ""
        prediction_length = self.model_params["prediction_length"]
        # Plot only the short context window, never the optional long context.
        plot_context_length = self.model_params["context_length"]
        step_hours = float(get_relative_time_step(str(self.model_params["freq"])))
        if selected_indices is None:
            selected_indices = list(range(min(len(forecasts), self.max_show)))
        selected_indices = [int(i) for i in selected_indices if 0 <= int(i) < len(forecasts)]
        n_show = len(selected_indices)
        if n_show == 0:
            raise ValueError("selected_indices must contain at least one valid series index.")
        cmap = plt.colormaps["tab20"].colors
        fg_color = "#2b2118"
        bg_color = "#f5f1e8" if compact_mode else "white"
        panel_color = "#fffdf8" if compact_mode else "white"
        target_color = "#24180f"
        context_gp_color = "#7a3ef0"
        forecast_color = "#0d6e6e"
        context_forecast_color = "#dd7a00"

        # Per-series CRPS using evaluator outputs:
        # for each row in metrics_per_ts, mean over wQuantileLoss[q], where
        # wQuantileLoss[q] = QuantileLoss[q] / abs_target_sum.
        qloss_cols = sorted(
            [col for col in metrics_per_ts.columns if col.startswith("QuantileLoss[")]
        )
        crps_by_series = None
        if (
            metrics_per_ts is not None
            and "abs_target_sum" in metrics_per_ts.columns
            and len(qloss_cols) > 0
        ):
            denom = np.clip(
                metrics_per_ts["abs_target_sum"].to_numpy(dtype=np.float64),
                1e-12,
                None,
            )
            qloss = metrics_per_ts[qloss_cols].to_numpy(dtype=np.float64)
            wql = qloss / denom[:, None]
            crps_by_series = wql.mean(axis=1)

        n_indiv_rows = (n_show + 1) // 2
        n_rows = n_indiv_rows + int(show_aggregate_panel) + int(show_error_panel)
        if compact_mode:
            fig = plt.figure(figsize=(16, max(4.8, 4.8 * n_rows)), facecolor=bg_color)
        else:
            fig = plt.figure(figsize=(18, 3.5 * n_rows), facecolor=bg_color)
        gs = fig.add_gridspec(n_rows, 2, hspace=0.45, wspace=0.30)

        # ---- helpers: extract arrays from GluonTS objects ----
        def _ts_values(ts, ndim_squeeze: bool = True):
            v = ts if isinstance(ts, np.ndarray) else ts.values
            if ndim_squeeze and v.ndim > 1:
                v = v[:, 0]
            return v.astype(np.float64)

        def _fc_samples(fc):
            s = fc.samples
            if s.ndim > 2:
                s = s[:, :, 0]
            return s.astype(np.float64)

        def _series_samples(values):
            arr = np.asarray(values, dtype=np.float64)
            if arr.ndim == 3:
                arr = arr[..., 0]
            return arr

        def _series_curve(values):
            arr = np.asarray(values, dtype=np.float64)
            if arr.ndim == 2:
                arr = arr[:, 0]
            return arr

        def _series_time_grids(series_idx: int):
            if not time_grid_map or series_idx not in time_grid_map:
                return None, None
            info = time_grid_map[series_idx]
            context_time = np.asarray(info.get("context_time"), dtype=np.float64).reshape(-1)
            future_time = np.asarray(info.get("future_time"), dtype=np.float64).reshape(-1)
            if context_time.size == 0 or future_time.size == 0:
                return None, None
            return context_time, future_time

        def _shared_axis(name: str, series_indices: list[int]):
            if not time_grid_map:
                return None
            ref = None
            for series_idx in series_indices:
                context_time, future_time = _series_time_grids(series_idx)
                if context_time is None or future_time is None:
                    return None
                current = context_time if name == "context" else future_time
                if ref is None:
                    ref = current
                elif current.shape != ref.shape or not np.allclose(current, ref):
                    return None
            return ref

        next_row_offset = 0
        t_pred = _elapsed_time_axis(prediction_length, step_hours)
        aggregate_future_axis = _shared_axis("future", selected_indices)
        aggregate_context_axis = _shared_axis("context", selected_indices)
        if aggregate_context_axis is not None or aggregate_future_axis is not None:
            aggregate_origin = (
                float(aggregate_context_axis[0])
                if aggregate_context_axis is not None and len(aggregate_context_axis) > 0
                else float(aggregate_future_axis[0])
            )
            if aggregate_context_axis is not None:
                aggregate_context_axis = aggregate_context_axis - aggregate_origin
            if aggregate_future_axis is not None:
                aggregate_future_axis = aggregate_future_axis - aggregate_origin
        if show_aggregate_panel:
            ax_mean = fig.add_subplot(gs[0, :])
            ax_mean.set_facecolor(panel_color)

            all_means, all_targets = [], []
            for series_idx in selected_indices:
                tv = _ts_values(tss[series_idx])[-prediction_length:]
                fm = _fc_samples(forecasts[series_idx]).mean(axis=0)
                all_targets.append(tv)
                all_means.append(fm)
            all_targets = np.array(all_targets)
            all_means = np.array(all_means)

            gt_mean, gt_std = all_targets.mean(0), all_targets.std(0)
            fc_mean, fc_std = all_means.mean(0), all_means.std(0)

            agg_x = aggregate_future_axis if aggregate_future_axis is not None else t_pred
            ax_mean.plot(agg_x, gt_mean, lw=2, color=target_color, label="Target Mean")
            ax_mean.plot(agg_x, fc_mean, lw=2, ls="--", color=forecast_color, label="Forecast Mean")
            ax_mean.fill_between(
                agg_x,
                gt_mean - gt_std,
                gt_mean + gt_std,
                alpha=0.2,
                color=target_color,
                label="Target ±1σ",
            )
            ax_mean.fill_between(
                agg_x,
                fc_mean - fc_std,
                fc_mean + fc_std,
                alpha=0.2,
                color=forecast_color,
                label="Forecast ±1σ",
            )
            if context_forecast_map is not None:
                context_forecast_curves = []
                for series_idx in selected_indices:
                    samples = context_forecast_map.get(series_idx)
                    if samples is None:
                        continue
                    context_forecast_curves.append(_series_samples(samples).mean(axis=0))
                if context_forecast_curves:
                    context_forecast_curves = np.array(context_forecast_curves)
                    t_ctx_overlay = aggregate_context_axis if aggregate_context_axis is not None else _elapsed_time_axis(context_forecast_curves.shape[1], step_hours)
                    ctx_fc_mean = context_forecast_curves.mean(axis=0)
                    ctx_fc_std = context_forecast_curves.std(axis=0)
                    ax_mean.plot(
                        t_ctx_overlay,
                        ctx_fc_mean,
                        lw=2,
                        ls="-.",
                        color=context_forecast_color,
                        label="Context Forecast Mean",
                    )
                    ax_mean.fill_between(
                        t_ctx_overlay,
                        ctx_fc_mean - ctx_fc_std,
                        ctx_fc_mean + ctx_fc_std,
                        alpha=0.18,
                        color=context_forecast_color,
                        label="Context Forecast ±1σ",
                    )
            if gp_mean_context_map is not None:
                context_gp_curves = []
                for series_idx in selected_indices:
                    gp_values = gp_mean_context_map.get(series_idx)
                    if gp_values is None:
                        continue
                    context_gp_curves.append(_series_curve(gp_values))
                if context_gp_curves:
                    context_gp_curves = np.array(context_gp_curves)
                    t_ctx_overlay = aggregate_context_axis if aggregate_context_axis is not None else _elapsed_time_axis(context_gp_curves.shape[1], step_hours)
                    ax_mean.plot(
                        t_ctx_overlay,
                        context_gp_curves.mean(axis=0),
                        lw=2,
                        ls=":",
                        color=context_gp_color,
                        label="GP Mean (Context)",
                    )
            if gp_mean_future_map is not None:
                future_gp_curves = []
                for series_idx in selected_indices:
                    gp_values = gp_mean_future_map.get(series_idx)
                    if gp_values is None:
                        continue
                    future_gp_curves.append(_series_curve(gp_values))
                if future_gp_curves:
                    future_gp_curves = np.array(future_gp_curves)
                    gp_future_len = min(len(t_pred), future_gp_curves.shape[1])
                    t_gp_future = agg_x[:gp_future_len]
                    ax_mean.plot(
                        t_gp_future,
                        future_gp_curves.mean(axis=0)[:gp_future_len],
                        lw=2,
                        ls=":",
                        color=context_gp_color,
                        label="GP Mean (Future)",
                    )
            if control_future is not None and control_future.shape[1] > 0:
                ctrl_subset = control_future[:, selected_indices, :]
                ctrl_mean = ctrl_subset.mean(axis=(0, 1))
                ctrl_std = ctrl_subset.std(axis=(0, 1))
                ax_mean.plot(t_pred, ctrl_mean, lw=2, color="green", ls=":", label="Control Mean")
                ax_mean.fill_between(
                    t_pred,
                    ctrl_mean - ctrl_std,
                    ctrl_mean + ctrl_std,
                    alpha=0.15,
                    color="green",
                    label="Control ±1σ",
                )
            ax_mean.set_title(f"{title_prefix} Forecast vs Target{epoch_suffix}", fontsize=14, fontweight="bold", color=fg_color)
            ax_mean.legend(fontsize=9)
            ax_mean.grid(True, alpha=0.22)
            ax_mean.set_ylabel("Value", color=fg_color)
            ax_mean.tick_params(colors=fg_color)
            ax_mean.spines["top"].set_visible(False)
            ax_mean.spines["right"].set_visible(False)
            next_row_offset = 1

        # ---- Middle rows: individual series (2 cols) ----
        for plot_idx, series_idx in enumerate(selected_indices):
            row = next_row_offset + plot_idx // 2
            col = plot_idx % 2
            ax = fig.add_subplot(gs[row, col])
            ax.set_facecolor(panel_color)

            ts_vals = _ts_values(tss[series_idx])
            ctx_start = max(0, len(ts_vals) - prediction_length - plot_context_length)
            context_vals = ts_vals[ctx_start: len(ts_vals) - prediction_length]
            target_vals = ts_vals[-prediction_length:]

            context_time, future_time = _series_time_grids(series_idx)
            show_observation_markers = context_time is not None and future_time is not None
            if context_time is not None and future_time is not None:
                context_time = context_time[-len(context_vals):]
                future_time = future_time[: len(target_vals)]
                local_origin = (
                    float(context_time[0])
                    if len(context_time) > 0
                    else float(future_time[0])
                )
                t_ctx = context_time - local_origin
                t_fc = future_time - local_origin
                split_time = float(future_time[0]) if len(future_time) > 0 else float(context_time[-1]) if len(context_time) > 0 else 0.0
                split_time = split_time - local_origin
                span_start = split_time
                span_end = (float(future_time[-1]) - local_origin) if len(future_time) > 0 else split_time
                x_label = "Elapsed Time [h]"
            else:
                context_duration_hours = float(len(context_vals)) * step_hours
                t_ctx = _elapsed_time_axis(len(context_vals), step_hours)
                t_fc = _elapsed_time_axis(prediction_length, step_hours, start_hours=context_duration_hours)
                split_time = context_duration_hours
                span_start = context_duration_hours
                span_end = t_fc[-1] if len(t_fc) > 0 else context_duration_hours
                x_label = "Elapsed Time [h]"

            target_marker_kwargs = {}
            if show_observation_markers:
                target_marker_kwargs = {
                    "marker": "o",
                    "markersize": 4.0,
                    "markerfacecolor": "#fffdf8" if compact_mode else "white",
                    "markeredgecolor": target_color,
                    "markeredgewidth": 0.9,
                }

            # context
            ax.plot(t_ctx, context_vals, color=target_color, lw=2.2, label="Context Target", **target_marker_kwargs)
            if gp_mean_context_map is not None and series_idx in gp_mean_context_map:
                gp_mean_context = _series_curve(gp_mean_context_map[series_idx])
                gp_context_x = t_ctx if len(t_ctx) == len(gp_mean_context) else _elapsed_time_axis(len(gp_mean_context), step_hours)
                ax.plot(
                    gp_context_x,
                    gp_mean_context,
                    color=context_gp_color,
                    lw=2,
                    ls=":",
                    alpha=0.95,
                    label="GP Mean (Context)",
                )
            if gp_mean_future_map is not None and series_idx in gp_mean_future_map:
                gp_mean_future = _series_curve(gp_mean_future_map[series_idx])
                gp_future_len = min(len(t_fc), len(gp_mean_future))
                ax.plot(
                    t_fc[:gp_future_len],
                    gp_mean_future[:gp_future_len],
                    color=context_gp_color,
                    lw=2,
                    ls=":",
                    alpha=0.95,
                    label="GP Mean (Future)",
                )
            if context_forecast_map is not None and series_idx in context_forecast_map:
                context_forecast_samples = _series_samples(context_forecast_map[series_idx])
                ctx_fc_med = np.median(context_forecast_samples, axis=0)
                ctx_fc_mean = context_forecast_samples.mean(axis=0)
                ctx_fc_std = context_forecast_samples.std(axis=0)
                t_ctx_overlay = t_ctx if len(t_ctx) == context_forecast_samples.shape[1] else _elapsed_time_axis(context_forecast_samples.shape[1], step_hours)
                ax.plot(
                    t_ctx_overlay,
                    ctx_fc_med,
                    color=context_forecast_color,
                    lw=2,
                    ls="-.",
                    label="Context Forecast Median",
                )
                ax.plot(
                    t_ctx_overlay,
                    ctx_fc_mean,
                    color=context_forecast_color,
                    lw=1.8,
                    ls="--",
                    alpha=0.9,
                    label="Context Forecast Mean",
                )
                ax.fill_between(
                    t_ctx_overlay,
                    ctx_fc_mean - ctx_fc_std,
                    ctx_fc_mean + ctx_fc_std,
                    color=context_forecast_color,
                    alpha=0.14,
                    label="Context Forecast ±1σ",
                )
            # ground-truth future
            ax.axvspan(span_start, span_end, color="#eadcc6", alpha=0.22, zorder=0)
            ax.plot(t_fc, target_vals, color=target_color, lw=2.0, ls="--", label="Forecast Target", **target_marker_kwargs)

            # forecast quantiles
            samples = _fc_samples(forecasts[series_idx])
            fc_med = np.median(samples, axis=0)
            fc_mean = samples.mean(axis=0)
            fc_std = samples.std(axis=0)

            c = forecast_color if compact_mode else cmap[plot_idx % len(cmap)]
            ax.plot(t_fc, fc_med, color=c, lw=2.2, label="Forecast Median")
            ax.plot(t_fc, fc_mean, color=c, lw=2.0, ls="-.", alpha=0.85, label="Forecast Mean")
            ax.fill_between(t_fc, fc_mean - fc_std, fc_mean + fc_std, color=c, alpha=0.22, label="Forecast ±1σ")
            if control_future is not None and series_idx < control_future.shape[1]:
                ctrl_samples = control_future[:, series_idx, :]  # (S, H)
                ctrl_mean = ctrl_samples.mean(axis=0)
                ax.plot(t_fc, ctrl_mean, color="green", lw=2, ls=":", alpha=0.7, label="Control")
            ax.axvline(x=split_time, color="#6d655d", ls=":", alpha=0.65, lw=1.4)
            if crps_by_series is not None and series_idx < len(crps_by_series):
                ax.set_title(
                    f"Series {series_idx + 1} | CRPS={crps_by_series[series_idx]:.4f}",
                    fontsize=11,
                    color=fg_color,
                    fontweight="bold" if compact_mode else None,
                )
            else:
                ax.set_title(f"Series {series_idx + 1}", fontsize=11, color=fg_color)
            ax.set_xlabel(x_label, color=fg_color)
            ax.set_ylabel("Value", color=fg_color)
            ax.tick_params(colors=fg_color)
            ax.grid(True, alpha=0.2)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            if plot_idx == 0:
                legend_cols = 2 if compact_mode else 1
                ax.legend(fontsize=8.5 if compact_mode else 8, ncol=legend_cols, frameon=False, loc="upper left")

        if show_error_panel:
            ax_err = fig.add_subplot(gs[next_row_offset + n_indiv_rows, :])
            ax_err.set_facecolor(panel_color)
            for plot_idx, series_idx in enumerate(selected_indices):
                tv = _ts_values(tss[series_idx])[-prediction_length:]
                fm = _fc_samples(forecasts[series_idx]).mean(axis=0)
                err = (fm - tv) ** 2
                err_x = aggregate_future_axis if aggregate_future_axis is not None else t_pred
                ax_err.plot(
                    err_x,
                    err,
                    color=cmap[plot_idx % len(cmap)],
                    alpha=0.6,
                    lw=1.5,
                    label=f"Series {series_idx + 1}",
                )

            ax_err.set_title("Pointwise MSE: Forecast Error Over Horizon", fontsize=14, fontweight="bold", color=fg_color)
            ax_err.set_xlabel("Time [h]" if aggregate_future_axis is not None else "Elapsed Time [h]", fontweight="bold", color=fg_color)
            ax_err.set_ylabel("Squared Error", fontweight="bold", color=fg_color)
            ax_err.legend(fontsize=8, ncol=min(4, n_show))
            ax_err.grid(True, alpha=0.3)
            ax_err.tick_params(colors=fg_color)
            ax_err.spines["top"].set_visible(False)
            ax_err.spines["right"].set_visible(False)

        fig.suptitle(
            f"Epoch {epoch + 1}" if epoch is not None else title_prefix,
            fontsize=16,
            fontweight="bold",
            y=1.01,
            color=fg_color,
        )
        fig.tight_layout()
        return fig

    def _select_worst_crps_indices(self, metrics_per_ts, n_forecasts: int) -> list[int]:
        if metrics_per_ts is None or n_forecasts <= 0:
            return list(range(min(n_forecasts, self.max_show)))

        qloss_cols = sorted([col for col in metrics_per_ts.columns if col.startswith("QuantileLoss[")])
        if "abs_target_sum" not in metrics_per_ts.columns or len(qloss_cols) == 0:
            return list(range(min(n_forecasts, self.max_show)))

        denom = np.clip(
            metrics_per_ts["abs_target_sum"].to_numpy(dtype=np.float64),
            1e-12,
            None,
        )
        qloss = metrics_per_ts[qloss_cols].to_numpy(dtype=np.float64)
        crps_by_series = (qloss / denom[:, None]).mean(axis=1)
        n = min(len(crps_by_series), n_forecasts, self.max_show)
        if n <= 0:
            return []
        return [int(i) for i in np.argsort(crps_by_series)[-n:][::-1]]
