import argparse
import ast
import csv
import json
import re
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass
from itertools import islice
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import torch
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.common import ListDataset
from gluonts.dataset.loader import InferenceDataLoader
from gluonts.dataset.multivariate_grouper import MultivariateGrouper
from gluonts.dataset.split import split
from gluonts.dataset.util import period_index
from gluonts.evaluation import Evaluator, MultivariateEvaluator, make_evaluation_predictions
from gluonts.itertools import Cached
from gluonts.model.forecast import SampleForecast
from gluonts.time_feature import time_features_from_frequency_str
from gluonts.torch.batchify import batchify
from tqdm.auto import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import ticker as mticker
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from execute.inference_efficiency_gluonts import (
    BenchmarkSpec,
    _build_model,
    _build_specs,
    _read_yaml,
    _run_inference_for_horizon,
    _resolve_device,
)
from gslice.callbacks.gluonts_eval_plot_callback import GluonTSEvalPlotCallback
from gslice.dataset import (
    get_dataset_frequency_variants,
    get_dataset_name_from_params,
    get_gts_dataset_from_params,
    infer_target_dim,
)
from gslice.irregular import (
    DENSE_PAST_OBSERVED_FIELD,
    DENSE_PAST_TARGET_FIELD,
    DENSE_PAST_TIME_GRID_FIELD,
    FUTURE_TIME_GRID_FIELD,
    PAST_TIME_GRID_FIELD,
    has_irregular_grid,
    maybe_get_irregular_input_names,
    compute_irregular_lag_features,
    deterministic_rng,
    get_irregular_grid_spec,
    resolve_model_window_lengths,
    resolve_source_window_lengths,
    sample_irregular_indices,
)
from gslice.utils.transforms import IrregularInstanceTransform
from gslice.utils.gaussian_process import GPRegressor, Q0Dist
from gslice.utils import create_transforms
from gslice.utils.util import create_splitter, filter_metrics, temporary_random_seed
from gslice.utils.variables import frequencies_match, get_relative_time_step, get_season_length
from gluonts.transform import Chain


@dataclass
class CrossFrequencyEvalResult:
    label: str
    model_type: str
    checkpoint_path: str
    config_path: str
    train_dataset: str
    train_freq: str
    eval_dataset: str
    eval_freq: str
    setting: str
    device: str
    num_samples: int
    eval_seed: Optional[int]
    plot_path: Optional[str]
    max_eval_instances: Optional[int]
    prediction_length: int
    model_prediction_length: int
    past_length: int
    CRPS: float
    ND: float
    NRMSE: float
    m_sum_CRPS: Optional[float] = None


@dataclass
class FirstSeriesPlotPayload:
    setting: str
    model_params: Dict[str, Any]
    forecasts: Any
    tss: Any
    metrics_per_ts: Any
    eval_dataset_name: str
    eval_freq: str
    prediction_length: int
    num_samples: int
    eval_seed: Optional[int]
    selected_indices: list[int]
    plot_diagnostics: Optional[dict[str, Any]] = None


FINE_TO_COARSE_EVAL_ADAPTER_KEY = "fine_to_coarse_eval_adapter"
SUPPORTED_FINE_TO_COARSE_EVAL_ADAPTERS = {"none", "repeat", "gp_resample"}

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


def create_multivariate_transforms(*args, **kwargs):
    raise NotImplementedError(
        "Multivariate cross-frequency evaluation is not available in the current gslice.utils API."
    )


def _parse_list_arg(value: str, name: str) -> List[str]:
    parsed = ast.literal_eval(value)
    if isinstance(parsed, str):
        return [parsed]
    if not isinstance(parsed, list):
        raise ValueError(f"{name} must be a Python list literal, got {type(parsed).__name__}.")
    return [str(item) for item in parsed]


def _parse_optional_int_list_arg(value: Optional[str], name: str) -> Optional[List[int]]:
    if value is None:
        return None
    parsed = ast.literal_eval(value)
    if isinstance(parsed, int):
        return [int(parsed)]
    if not isinstance(parsed, list):
        raise ValueError(f"{name} must be an int or Python list literal, got {type(parsed).__name__}.")
    return [int(item) for item in parsed]


def _normalize_optional_list(values: Optional[List[Any]], n: int, name: str) -> List[Any]:
    if values is None:
        return [None] * n
    if len(values) == 1 and n > 1:
        return values * n
    if len(values) != n:
        raise ValueError(f"{name} must have length 1 or match eval_datasets ({n}), got {len(values)}.")
    return values


def _resolve_num_samples(spec: BenchmarkSpec, config: Dict[str, Any], override_num_samples: Optional[int]) -> int:
    if override_num_samples is not None:
        return int(override_num_samples)

    model_params = config.get("model_params", {})
    eval_params = config.get("evaluation_params", {})
    if spec.model_type == "lcde" and "num_samples" in eval_params:
        return int(eval_params["num_samples"])
    if "num_samples" in model_params:
        return int(model_params["num_samples"])
    if "num_samples" in eval_params:
        return int(eval_params["num_samples"])
    return 100


def _truncate_dataset(test_data, freq: str, max_eval_instances: Optional[int]):
    if max_eval_instances is None:
        return test_data
    limited = list(islice(test_data, int(max_eval_instances)))
    return ListDataset(limited, freq=freq)


def _normalize_fine_to_coarse_eval_adapter(mode: Optional[str]) -> str:
    normalized = str(mode or "none").strip().lower()
    if normalized not in SUPPORTED_FINE_TO_COARSE_EVAL_ADAPTERS:
        supported = ", ".join(sorted(SUPPORTED_FINE_TO_COARSE_EVAL_ADAPTERS))
        raise ValueError(
            f"{FINE_TO_COARSE_EVAL_ADAPTER_KEY} must be one of {{{supported}}}, got {mode!r}."
        )
    return normalized


def _get_fine_to_coarse_eval_adapter_mode(dataset_params: Dict[str, Any]) -> str:
    return _normalize_fine_to_coarse_eval_adapter(dataset_params.get(FINE_TO_COARSE_EVAL_ADAPTER_KEY, "none"))


def _resolve_fine_to_coarse_ratio(train_freq: str, eval_freq: str) -> Optional[int]:
    train_step_hours = float(get_relative_time_step(str(train_freq)))
    eval_step_hours = float(get_relative_time_step(str(eval_freq)))
    if eval_step_hours <= train_step_hours:
        return None
    ratio = eval_step_hours / train_step_hours
    rounded_ratio = int(round(ratio))
    if not np.isclose(ratio, rounded_ratio):
        raise ValueError(
            f"Evaluation frequency {eval_freq} must be an integer multiple of training frequency {train_freq}; "
            f"got ratio={ratio}."
        )
    if rounded_ratio <= 1:
        return None
    return rounded_ratio


def _build_uniform_time_grid(length: int, step_hours: float, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.arange(length, device=device, dtype=dtype) * float(step_hours)


def _repeat_expand_time_tensor(values: torch.Tensor, repeats: int) -> torch.Tensor:
    return values.repeat_interleave(int(repeats), dim=1)


def _select_eval_grid_indices(
    fine_length: int,
    ratio: int,
    *,
    expected_length: Optional[int] = None,
    device: Optional[torch.device] = None,
):
    if fine_length <= 0:
        raise ValueError(f"fine_length must be > 0, got {fine_length}.")
    if ratio <= 1:
        raise ValueError(f"ratio must be > 1 for fine-to-coarse downsampling, got {ratio}.")
    if fine_length % ratio != 0:
        raise ValueError(
            f"Fine-grid length {fine_length} is not divisible by fine-to-coarse ratio {ratio}."
        )
    indices = torch.arange(ratio - 1, fine_length, ratio, device=device)
    if expected_length is not None and int(indices.numel()) != int(expected_length):
        raise ValueError(
            f"Expected {expected_length} coarse steps after downsampling, got {int(indices.numel())}."
        )
    return indices


def _downsample_fine_time_tensor(
    values: torch.Tensor,
    ratio: int,
    *,
    time_dim: int,
    expected_length: Optional[int] = None,
) -> torch.Tensor:
    indices = _select_eval_grid_indices(
        int(values.shape[time_dim]),
        int(ratio),
        expected_length=expected_length,
        device=values.device,
    )
    return values.index_select(time_dim, indices)


def _build_eval_adapter_gp_regressor(model, model_params: Dict[str, Any]) -> GPRegressor:
    prior_params = dict(model_params.get("prior_params", {}) or {})
    season_length = prior_params.get("season_length", getattr(model, "freq", None))
    period_hours = prior_params.get("period_hours", None)
    return GPRegressor(
        kernel=model.prior,
        gamma=float(prior_params.get("gamma", 1.0)),
        noise=float(prior_params.get("noise", prior_params.get("iso", 1e-4))),
        jitter=float(prior_params.get("jitter", 1e-6)),
        use_data_mean=bool(prior_params.get("use_data_mean", True)),
        season_length=int(season_length) if season_length is not None else None,
        period_hours=float(period_hours) if period_hours is not None else None,
    )


def _gp_expand_past_target_to_train_grid(
    *,
    past_target: torch.Tensor,
    past_observed_values: torch.Tensor,
    model,
    model_params: Dict[str, Any],
    train_freq: str,
    eval_freq: str,
) -> torch.Tensor:
    original_dim = past_target.dim()
    target = past_target if original_dim == 3 else past_target.unsqueeze(-1)
    if past_observed_values.dim() == 3:
        observed = past_observed_values.any(dim=-1)
    else:
        observed = past_observed_values
    observed = observed.to(dtype=torch.bool)

    batch_size, coarse_length, _ = target.shape
    ratio = _resolve_fine_to_coarse_ratio(train_freq, eval_freq)
    if ratio is None:
        raise ValueError("GP expansion requires a finer training grid than evaluation grid.")

    train_step_hours = float(get_relative_time_step(str(train_freq)))
    eval_step_hours = float(get_relative_time_step(str(eval_freq)))
    t_coarse = _build_uniform_time_grid(coarse_length, eval_step_hours, device=target.device, dtype=target.dtype)
    t_fine = _build_uniform_time_grid(coarse_length * ratio, train_step_hours, device=target.device, dtype=target.dtype)

    expanded_samples = []
    for batch_idx in range(batch_size):
        obs_idx = observed[batch_idx].nonzero(as_tuple=True)[0]
        if obs_idx.numel() == 0:
            obs_idx = torch.tensor([coarse_length - 1], device=target.device, dtype=torch.long)
        gp = _build_eval_adapter_gp_regressor(model, model_params).to(target.device)
        gp.fit(
            t_coarse.index_select(0, obs_idx),
            target[batch_idx].index_select(0, obs_idx),
        )
        expanded_samples.append(gp.sample(t_fine, num_samples=1).squeeze(0))

    expanded = torch.stack(expanded_samples, dim=0)
    if original_dim == 2:
        return expanded.squeeze(-1)
    return expanded


def _adapt_tsflow_batch_to_train_grid(
    *,
    batch: Dict[str, torch.Tensor],
    model,
    model_params: Dict[str, Any],
    train_freq: str,
    eval_freq: str,
    adapter_mode: str,
) -> tuple[Dict[str, torch.Tensor], int]:
    normalized_mode = _normalize_fine_to_coarse_eval_adapter(adapter_mode)
    if normalized_mode == "none":
        return dict(batch), 1

    ratio = _resolve_fine_to_coarse_ratio(train_freq, eval_freq)
    if ratio is None:
        raise ValueError(
            f"Fine-grid eval adaptation requires a finer training grid than eval grid, got "
            f"train_freq={train_freq}, eval_freq={eval_freq}."
        )

    adapted_batch = dict(batch)
    past_target = batch["past_target"]
    past_observed_values = batch["past_observed_values"]

    if normalized_mode == "repeat":
        adapted_batch["past_target"] = _repeat_expand_time_tensor(past_target, ratio)
    elif normalized_mode == "gp_resample":
        adapted_batch["past_target"] = _gp_expand_past_target_to_train_grid(
            past_target=past_target,
            past_observed_values=past_observed_values,
            model=model,
            model_params=model_params,
            train_freq=train_freq,
            eval_freq=eval_freq,
        )
    else:
        raise AssertionError(f"Unhandled fine-to-coarse adapter mode: {normalized_mode}.")

    adapted_batch["past_observed_values"] = _repeat_expand_time_tensor(past_observed_values, ratio)
    return adapted_batch, ratio


def _repeat_project_irregular_context(
    source_times: np.ndarray,
    source_values: np.ndarray,
    target_times: np.ndarray,
) -> np.ndarray:
    indices = np.searchsorted(source_times, target_times, side="right") - 1
    indices = np.clip(indices, 0, len(source_times) - 1)
    return source_values[indices]


def _gp_project_irregular_context(
    *,
    source_times: np.ndarray,
    source_values: np.ndarray,
    dense_source_times: np.ndarray | None = None,
    dense_source_values: np.ndarray | None = None,
    dense_source_observed: np.ndarray | None = None,
    model,
    model_params: Dict[str, Any],
    target_times: np.ndarray,
) -> np.ndarray:
    device = next(model.parameters()).device if hasattr(model, "parameters") else torch.device("cpu")
    gp = _build_eval_adapter_gp_regressor(model, model_params).to(device)
    use_dense = dense_source_times is not None and dense_source_values is not None
    fit_times = dense_source_times if use_dense else source_times
    fit_values = dense_source_values if use_dense else source_values
    fit_observed = dense_source_observed
    if use_dense and fit_observed is not None:
        observed_idx = np.asarray(fit_observed, dtype=bool).nonzero()[0]
        if observed_idx.size > 0:
            fit_times = np.asarray(fit_times)[observed_idx]
            fit_values = np.asarray(fit_values)[observed_idx]
    x = torch.as_tensor(np.asarray(fit_times)[:, None], dtype=torch.float32, device=device)
    y = torch.as_tensor(np.asarray(fit_values)[:, None], dtype=torch.float32, device=device)
    gp.fit(x, y)
    projected = gp.predict(torch.as_tensor(target_times[:, None], dtype=torch.float32, device=device))
    return projected.squeeze(-1).detach().cpu().numpy()


def _adapt_tsflow_batch_to_train_irregular_grid(
    *,
    batch: Dict[str, torch.Tensor],
    train_dataset_params: Dict[str, Any],
    eval_dataset_params: Dict[str, Any],
    model,
    model_params: Dict[str, Any],
    adapter_mode: str,
) -> Dict[str, torch.Tensor]:
    normalized_mode = _normalize_fine_to_coarse_eval_adapter(adapter_mode)
    if normalized_mode == "none":
        return dict(batch)

    train_spec = get_irregular_grid_spec(train_dataset_params, fallback_freq=str(model_params["freq"]))
    eval_spec = get_irregular_grid_spec(eval_dataset_params, fallback_freq=str(model_params["freq"]))
    if train_spec.num_context_points != eval_spec.num_context_points:
        raise ValueError("Irregular-grid adapter currently requires matching context point counts.")

    adapted = dict(batch)
    past_target = batch["past_target"].detach().cpu().numpy()
    past_observed = batch["past_observed_values"].detach().cpu().numpy()
    future_time_grid = batch["future_time_grid"].detach().cpu().numpy()
    dense_past_target = batch[DENSE_PAST_TARGET_FIELD].detach().cpu().numpy()
    dense_past_time_grid = batch[DENSE_PAST_TIME_GRID_FIELD].detach().cpu().numpy()
    dense_past_observed = batch[DENSE_PAST_OBSERVED_FIELD].detach().cpu().numpy()

    batch_size = past_target.shape[0]
    adapted_past_target = []
    adapted_past_obs = []
    adapted_past_time = []
    adapted_lag_features = []

    for batch_idx in range(batch_size):
        long_len = int(dense_past_target[batch_idx].shape[0] - train_spec.dense_context_length)
        dense_context_times = dense_past_time_grid[batch_idx][long_len:]
        rng = deterministic_rng(
            item_id=batch_idx,
            forecast_start=float(dense_context_times[-1]),
            seed_offset=train_spec.seed_offset,
            salt="adapter_context",
        )
        target_indices = sample_irregular_indices(
            dense_length=int(train_spec.dense_context_length),
            num_points=int(train_spec.num_context_points),
            gamma_k=float(train_spec.gamma_k),
            rng=rng,
            include_endpoints=bool(train_spec.include_region_endpoints),
        )
        target_context_times = dense_context_times[target_indices]
        source_context_times = batch["past_time_grid"][batch_idx].detach().cpu().numpy()[-eval_spec.num_context_points :]
        source_context_values = past_target[batch_idx][-eval_spec.num_context_points :]

        if normalized_mode == "repeat":
            projected_context_values = _repeat_project_irregular_context(
                source_context_times,
                source_context_values,
                target_context_times,
            )
        elif normalized_mode == "gp_resample":
            projected_context_values = _gp_project_irregular_context(
                source_times=source_context_times,
                source_values=source_context_values,
                dense_source_times=dense_past_time_grid[batch_idx],
                dense_source_values=dense_past_target[batch_idx],
                dense_source_observed=dense_past_observed[batch_idx],
                model=model,
                model_params=model_params,
                target_times=target_context_times,
            )
        else:
            raise AssertionError(f"Unhandled irregular adapter mode: {normalized_mode}.")

        adapted_past_target.append(projected_context_values)
        adapted_past_obs.append(np.ones_like(projected_context_values, dtype=np.bool_))
        adapted_past_time.append(target_context_times)
        query_time = np.concatenate([target_context_times, future_time_grid[batch_idx]], axis=0)
        adapted_lag_features.append(
            compute_irregular_lag_features(
                dense_past_target=dense_past_target[batch_idx],
                dense_past_time_grid=dense_past_time_grid[batch_idx],
                query_time_grid=query_time,
                lag_steps=list(getattr(model, "lags_seq", []) or []),
                step_hours=float(get_relative_time_step(str(train_spec.base_freq))),
            )
        )

    device = batch["past_target"].device
    adapted["past_target"] = torch.as_tensor(np.stack(adapted_past_target), dtype=batch["past_target"].dtype, device=device)
    adapted["past_observed_values"] = torch.as_tensor(
        np.stack(adapted_past_obs),
        dtype=batch["past_observed_values"].dtype,
        device=device,
    )
    adapted["past_time_grid"] = torch.as_tensor(
        np.stack(adapted_past_time),
        dtype=batch["past_time_grid"].dtype,
        device=device,
    )
    adapted["lag_features"] = torch.as_tensor(
        np.stack(adapted_lag_features),
        dtype=torch.float32,
        device=device,
    )
    return adapted


def _resolve_adapter_eval_past_length(
    *,
    model,
    train_freq: str,
    eval_freq: str,
) -> int:
    ratio = _resolve_fine_to_coarse_ratio(train_freq, eval_freq)
    if ratio is None:
        raise ValueError("Adapter eval past length is only defined for finer-train/coarser-eval settings.")
    if hasattr(model, "lags_seq") and hasattr(model, "prior_context_length"):
        max_lag = max(getattr(model, "lags_seq", [0]) or [0])
        fine_past_length = max(
            int(getattr(model, "context_length")),
            int(getattr(model, "context_length")) + int(max_lag),
            int(getattr(model, "prior_context_length")),
        )
    else:
        fine_past_length = int(getattr(model, "context_length"))
    return max(1, int(np.ceil(float(fine_past_length) / float(ratio))))


def _prepare_eval_data(
    config: Dict[str, Any],
    model,
    eval_dataset_params: Dict[str, Any],
    eval_freq: Optional[str],
    eval_prediction_length: Optional[int],
    regenerate: bool,
    past_length_override: Optional[int] = None,
):
    setting = config["setting"]
    model_params = config["model_params"]

    if eval_prediction_length is not None and "physical_prediction_hours" not in eval_dataset_params:
        eval_dataset_params = dict(eval_dataset_params)
        eval_dataset_params["prediction_length"] = int(eval_prediction_length)

    dataset = get_gts_dataset_from_params(
        eval_dataset_params,
        regenerate=regenerate,
    )
    target_dim = infer_target_dim(dataset)
    resolved_eval_freq = str(eval_freq or dataset.metadata.freq)
    (
        resolved_context_length,
        resolved_prediction_length,
        source_context_length,
        source_prediction_length,
    ) = _resolve_eval_window_lengths(
        eval_dataset_params,
        dataset_freq=resolved_eval_freq,
        fallback_prediction_length=int(eval_prediction_length or dataset.metadata.prediction_length),
    )
    eval_dataset_name = get_dataset_name_from_params(eval_dataset_params)

    if not frequencies_match(dataset.metadata.freq, resolved_eval_freq):
        raise ValueError(
            f"Requested eval freq ({resolved_eval_freq}) does not match dataset metadata ({dataset.metadata.freq}) "
            f"for dataset {eval_dataset_name}."
        )
    expected_dataset_prediction_length = source_prediction_length if _uses_irregular_grid(eval_dataset_params) else resolved_prediction_length
    if dataset.metadata.prediction_length != expected_dataset_prediction_length:
        raise ValueError(
            f"Requested eval prediction_length ({expected_dataset_prediction_length}) does not match dataset metadata "
            f"({dataset.metadata.prediction_length}) for dataset {eval_dataset_name}."
        )

    num_rolling_evals = int(len(dataset.test) / len(dataset.train))
    time_features = time_features_from_frequency_str(resolved_eval_freq)

    if setting == "univariate":
        transformation = create_transforms(
            time_features=time_features,
            prediction_length=source_prediction_length,
            freq=get_season_length(resolved_eval_freq),
            train_length=len(dataset.train),
            time_step_hours=float(get_relative_time_step(resolved_eval_freq)),
            include_time_grid=_uses_irregular_grid(eval_dataset_params),
        )
        training_data = dataset.train
        test_data = dataset.test
    elif setting == "multivariate":
        train_grouper = MultivariateGrouper(max_target_dim=target_dim)
        test_grouper = MultivariateGrouper(
            num_test_dates=num_rolling_evals,
            max_target_dim=target_dim,
        )
        transformation = create_multivariate_transforms(
            time_features=time_features,
            prediction_length=source_prediction_length,
            target_dim=target_dim,
            freq=get_season_length(resolved_eval_freq),
            train_length=len(dataset.train),
            time_step_hours=float(get_relative_time_step(resolved_eval_freq)),
            include_time_grid=_uses_irregular_grid(eval_dataset_params),
        )
        training_data = train_grouper(dataset.train)
        test_data = test_grouper(dataset.test)
    else:
        raise ValueError(f"Unknown setting: {setting}")

    for _ in transformation.apply(training_data, is_train=True):
        pass

    if past_length_override is not None:
        past_length = int(past_length_override)
    elif hasattr(model, "lags_seq") and hasattr(model, "prior_context_length"):
        max_lag = max(getattr(model, "lags_seq", [0]) or [0])
        past_length = max(
            int(resolved_context_length) + int(max_lag),
            int(model.prior_context_length),
        )
    else:
        past_length = int(resolved_context_length) + int(model_params.get("long_context_length", 0))

    return (
        test_data,
        transformation,
        past_length,
        resolved_eval_freq,
        resolved_prediction_length,
        resolved_context_length,
        source_prediction_length,
    )


def _resolve_eval_dataset_metadata(
    eval_dataset_params: Dict[str, Any],
    eval_freq: Optional[str],
    eval_prediction_length: Optional[int],
    regenerate: bool,
) -> tuple[str, int]:
    if eval_prediction_length is not None and "physical_prediction_hours" not in eval_dataset_params:
        eval_dataset_params = dict(eval_dataset_params)
        eval_dataset_params["prediction_length"] = int(eval_prediction_length)

    dataset = get_gts_dataset_from_params(
        eval_dataset_params,
        regenerate=regenerate,
    )
    resolved_eval_freq = str(eval_freq or dataset.metadata.freq)
    _, resolved_prediction_length, _, source_prediction_length = _resolve_eval_window_lengths(
        eval_dataset_params,
        dataset_freq=resolved_eval_freq,
        fallback_prediction_length=int(eval_prediction_length or dataset.metadata.prediction_length),
    )
    eval_dataset_name = get_dataset_name_from_params(eval_dataset_params)
    if not frequencies_match(dataset.metadata.freq, resolved_eval_freq):
        raise ValueError(
            f"Requested eval freq ({resolved_eval_freq}) does not match dataset metadata ({dataset.metadata.freq}) "
            f"for dataset {eval_dataset_name}."
        )
    expected_dataset_prediction_length = source_prediction_length if _uses_irregular_grid(eval_dataset_params) else resolved_prediction_length
    if dataset.metadata.prediction_length != expected_dataset_prediction_length:
        raise ValueError(
            f"Requested eval prediction_length ({expected_dataset_prediction_length}) does not match dataset metadata "
            f"({dataset.metadata.prediction_length}) for dataset {eval_dataset_name}."
        )
    return resolved_eval_freq, resolved_prediction_length


def _to_dataframe(input_label) -> pd.DataFrame:
    start = input_label[0][FieldName.START]
    targets = [entry[FieldName.TARGET] for entry in input_label]
    full_target = np.concatenate(targets, axis=-1)
    index = period_index({FieldName.START: start, FieldName.TARGET: full_target})
    return pd.DataFrame(full_target.transpose(), index=index)


def _build_irregular_eval_entry(
    input_entry,
    label_entry,
    *,
    dataset_params: Dict[str, Any],
    freq: str,
    model_params: Dict[str, Any],
):
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
        "past_time_grid": full_time_grid[:past_len],
        "future_time_grid": full_time_grid[past_len : past_len + future_len],
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
        irregular_spec=get_irregular_grid_spec(dataset_params, fallback_freq=freq),
        lag_steps=(
            list(model_params.get("lags_seq", []) or [])
            if bool(model_params.get("use_lags", True))
            else []
        ),
    )
    return transform.map_transform(dense_entry, is_train=False)


def _irregular_entry_to_dataframe(entry) -> pd.DataFrame:
    past_target = np.asarray(entry["past_target"])
    future_target = np.asarray(entry["future_target"])
    full_target = np.concatenate([past_target, future_target], axis=-1)

    forecast_start = entry.get(FieldName.FORECAST_START)
    start = entry[FieldName.START]
    if forecast_start is not None:
        start = forecast_start - int(past_target.shape[-1])

    index = period_index({FieldName.START: start, FieldName.TARGET: full_target})
    df = pd.DataFrame(full_target.transpose(), index=index)
    df.attrs["irregular_time_grids"] = {
        "context_time": np.asarray(entry[PAST_TIME_GRID_FIELD], dtype=np.float64).reshape(-1).copy(),
        "future_time": np.asarray(entry[FUTURE_TIME_GRID_FIELD], dtype=np.float64).reshape(-1).copy(),
        "origin_start": entry[FieldName.START],
        "forecast_start": entry.get(FieldName.FORECAST_START),
    }
    return df


def _slugify(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def _load_eval_dataset_params(eval_dataset_name: Optional[str], eval_dataset_config: Optional[str]) -> Dict[str, Any]:
    if eval_dataset_config is not None:
        eval_config = _read_yaml(eval_dataset_config)
        if "dataset_params" in eval_config:
            dataset_params = dict(eval_config["dataset_params"])
        else:
            dataset_params = dict(eval_config)
        if "dataset" not in dataset_params and "base_dataset" not in dataset_params:
            raise ValueError(
                f"Evaluation dataset config {eval_dataset_config} must contain dataset_params "
                "or top-level dataset/base_dataset fields."
            )
        return dataset_params

    if eval_dataset_name is None:
        raise ValueError("Either eval_dataset_name or eval_dataset_config must be provided.")
    return {"dataset": str(eval_dataset_name)}


def _apply_dataset_param_overrides(
    config: Dict[str, Any],
    *,
    fine_to_coarse_eval_adapter_override: Optional[str],
) -> Dict[str, Any]:
    if fine_to_coarse_eval_adapter_override is None:
        return config

    updated = dict(config)
    dataset_params = dict(updated.get("dataset_params", {}))
    dataset_params[FINE_TO_COARSE_EVAL_ADAPTER_KEY] = _normalize_fine_to_coarse_eval_adapter(
        fine_to_coarse_eval_adapter_override
    )
    updated["dataset_params"] = dataset_params
    return updated


def _uses_irregular_grid(dataset_params: Dict[str, Any]) -> bool:
    return has_irregular_grid(dataset_params)


def _should_use_irregular_grid_eval_adapter(
    *,
    model_type: str,
    native_in_distribution_eval: bool,
    train_dataset_params: Dict[str, Any],
    eval_dataset_params: Dict[str, Any],
) -> bool:
    del native_in_distribution_eval
    return (
        model_type == "tsflow"
        and _uses_irregular_grid(train_dataset_params)
        and _uses_irregular_grid(eval_dataset_params)
    )


def _resolve_eval_window_lengths(
    dataset_params: Dict[str, Any],
    *,
    dataset_freq: str,
    fallback_prediction_length: int,
) -> tuple[int, int, int, int]:
    if not _uses_irregular_grid(dataset_params):
        prediction_length = int(fallback_prediction_length)
        return prediction_length, prediction_length, prediction_length, prediction_length
    context_length, prediction_length = resolve_model_window_lengths(dataset_params, fallback_freq=dataset_freq)
    source_context_length, source_prediction_length = resolve_source_window_lengths(
        dataset_params,
        fallback_freq=dataset_freq,
    )
    return (
        int(context_length),
        int(prediction_length),
        int(source_context_length),
        int(source_prediction_length),
    )


def _build_eval_instance_transform(
    *,
    splitter,
    dataset_params: Dict[str, Any],
    lag_steps: list[int],
    freq: str,
):
    if not _uses_irregular_grid(dataset_params):
        return splitter
    from gslice.irregular import get_irregular_grid_spec

    return Chain(
        [
            splitter,
            IrregularInstanceTransform(
                irregular_spec=get_irregular_grid_spec(dataset_params, fallback_freq=freq),
                lag_steps=list(lag_steps or []),
            ),
        ]
    )


@contextmanager
def _temporary_model_context_length(model, context_length: int):
    old_context_length = getattr(model, "context_length", None)
    if old_context_length is None:
        yield
        return
    model.context_length = int(context_length)
    try:
        yield
    finally:
        model.context_length = old_context_length


@contextmanager
def _temporary_model_prediction_length(model, prediction_length: int, *, enabled: bool):
    if not enabled:
        yield
        return

    old_prediction_length = getattr(model, "prediction_length", None)
    if old_prediction_length is None:
        yield
        return

    new_prediction_length = int(prediction_length)
    if int(old_prediction_length) == new_prediction_length:
        yield
        return

    old_q0 = getattr(model, "q0", None)
    old_prior_context_length = getattr(model, "prior_context_length", None)

    model.prediction_length = new_prediction_length
    try:
        if old_q0 is not None and old_prior_context_length is not None and hasattr(model, "prior_params"):
            # Keep the original prior-context span when we change the forecast horizon.
            # The eval-time GP adaptation context manager can then remap that span into
            # evaluation-frequency units without first collapsing it to the new chunk size.
            context_freqs = max(1, int(np.ceil(float(old_prior_context_length) / float(new_prediction_length))))
            q0_kwargs = dict(
                kernel=model.prior,
                context_freqs=context_freqs,
                prediction_length=new_prediction_length,
                freq=int(getattr(old_q0, "freq", getattr(model, "freq", 1))),
                gamma=float(model.prior_params.get("gamma", 1.0)),
                iso=float(old_q0.iso.item()) if hasattr(old_q0, "iso") else float(model.prior_params.get("iso", 0.0)),
                use_seasonal_mean=bool(getattr(old_q0, "use_seasonal_mean", True)),
            )
            model.q0 = Q0Dist(**q0_kwargs).to(next(model.parameters()).device)
            model.prior_context_length = context_freqs * new_prediction_length
        yield
    finally:
        model.prediction_length = old_prediction_length
        if old_q0 is not None:
            model.q0 = old_q0
        if old_prior_context_length is not None:
            model.prior_context_length = old_prior_context_length


@contextmanager
def _temporary_model_frequency_attrs(
    model,
    *,
    eval_freq: str,
    context_length: int,
    enabled: bool,
):
    if not enabled:
        yield
        return

    old_frequency = getattr(model, "frequency", None)
    old_freq = getattr(model, "freq", None)
    old_relative_time_step = getattr(model, "relative_time_step", None)
    if old_frequency is not None:
        model.frequency = str(eval_freq)
    if old_freq is not None:
        model.freq = int(get_season_length(str(eval_freq)))
    if old_relative_time_step is not None:
        model.relative_time_step = float(get_relative_time_step(str(eval_freq)))

    try:
        yield
    finally:
        if old_frequency is not None:
            model.frequency = old_frequency
        if old_freq is not None:
            model.freq = old_freq
        if old_relative_time_step is not None:
            model.relative_time_step = old_relative_time_step


def _resolve_adapted_context_freqs(
    *,
    train_freq: str,
    eval_freq: str,
    model_prediction_length: int,
    prior_context_length: int,
) -> int:
    train_step_hours = float(get_relative_time_step(train_freq))
    eval_step_hours = float(get_relative_time_step(eval_freq))
    train_prior_hours = float(prior_context_length) * train_step_hours
    eval_chunk_hours = float(model_prediction_length) * eval_step_hours
    if eval_chunk_hours <= 0:
        return 1
    return max(1, int(round(train_prior_hours / eval_chunk_hours)))


@contextmanager
def _temporary_eval_adapted_gp(
    *,
    spec: BenchmarkSpec,
    config: Dict[str, Any],
    model,
    eval_freq: str,
    enabled: bool,
):
    if not enabled:
        yield
        return

    model_params = config["model_params"]
    prior_params = dict(model_params.get("prior_params", {}) or {})
    train_freq = str(model_params["freq"])
    model_prediction_length = int(getattr(model, "prediction_length"))
    adapted_context_freqs = _resolve_adapted_context_freqs(
        train_freq=train_freq,
        eval_freq=str(eval_freq),
        model_prediction_length=model_prediction_length,
        prior_context_length=int(getattr(model, "prior_context_length")),
    )
    eval_season_length = int(get_season_length(str(eval_freq)))
    device = next(model.parameters()).device

    if spec.model_type == "tsflow" and hasattr(model, "q0"):
        old_q0 = model.q0
        old_prior_context_length = int(model.prior_context_length)
        iso_default = 1e-1 if str(prior_params.get("kernel", "")).lower() != "iso" else 0.0
        model.q0 = Q0Dist(
            kernel=prior_params["kernel"],
            context_freqs=adapted_context_freqs,
            prediction_length=model_prediction_length,
            freq=eval_season_length,
            gamma=float(prior_params.get("gamma", 1.0)),
            iso=float(prior_params.get("iso", iso_default)),
        ).to(device)
        model.prior_context_length = adapted_context_freqs * model_prediction_length
        try:
            yield
        finally:
            model.q0 = old_q0
            model.prior_context_length = old_prior_context_length
        return

    if spec.model_type == "lcde" and hasattr(model, "q0_dist"):
        old_q0_dist = model.q0_dist
        old_prior_context_length = int(model.prior_context_length)
        iso_default = 1e-1 if str(prior_params.get("kernel", "")).lower() != "iso" else 0.0
        model.q0_dist = Q0Dist(
            kernel=prior_params.get("kernel", "se"),
            context_freqs=adapted_context_freqs,
            prediction_length=model_prediction_length,
            freq=eval_season_length,
            gamma=float(prior_params.get("gamma", 1.0)),
            iso=float(prior_params.get("iso", iso_default)),
        ).to(device)
        model.prior_context_length = adapted_context_freqs * model_prediction_length
        try:
            yield
        finally:
            model.q0_dist = old_q0_dist
            model.prior_context_length = old_prior_context_length
        return

    yield


def _collect_first_series_plot_diagnostics(
    *,
    model,
    dataset,
    test_splitter,
    resolved_prediction_length: int,
    model_prediction_length: int,
    num_samples: int,
    eval_seed: Optional[int],
    max_examples: int = 2,
    fine_to_coarse_eval_adapter_mode: str = "none",
    train_freq: Optional[str] = None,
    eval_freq: Optional[str] = None,
    model_params: Optional[Dict[str, Any]] = None,
):
    predict_diagnostics = getattr(model, "predict_plot_diagnostics_from_past", None)
    if predict_diagnostics is None:
        return None

    if resolved_prediction_length == model_prediction_length:
        input_entries = list(islice(iter(dataset), int(max_examples)))
    else:
        _, test_template = split(dataset, offset=-int(resolved_prediction_length))
        test_instances = test_template.generate_instances(int(resolved_prediction_length))
        input_entries = [pair[0] for pair in islice(iter(test_instances), int(max_examples))]

    if not input_entries:
        return None

    device = next(model.parameters()).device
    data_loader = InferenceDataLoader(
        Cached(input_entries),
        batch_size=len(input_entries),
        stack_fn=lambda data: batchify(data, device),
        transform=test_splitter,
    )
    try:
        first_batch = next(iter(data_loader))
    except StopIteration:
        return None

    coarse_past_target = first_batch["past_target"].detach().cpu().numpy()
    ratio = 1
    adapter_mode = _normalize_fine_to_coarse_eval_adapter(fine_to_coarse_eval_adapter_mode)
    fine_past_target = None
    if adapter_mode != "none":
        if model_params is None or train_freq is None or eval_freq is None:
            raise ValueError("Adapted plot diagnostics require model_params, train_freq, and eval_freq.")
        first_batch, ratio = _adapt_tsflow_batch_to_train_grid(
            batch=first_batch,
            model=model,
            model_params=model_params,
            train_freq=train_freq,
            eval_freq=eval_freq,
            adapter_mode=adapter_mode,
        )
        fine_past_target = first_batch["past_target"].detach().cpu().numpy()

    with temporary_random_seed(eval_seed):
        dense_past_target = torch.as_tensor(
            first_batch.get(DENSE_PAST_TARGET_FIELD),
            dtype=torch.float32,
            device=device,
        ) if DENSE_PAST_TARGET_FIELD in first_batch else None
        dense_past_observed_values = torch.as_tensor(
            first_batch.get(DENSE_PAST_OBSERVED_FIELD),
            dtype=torch.float32,
            device=device,
        ) if DENSE_PAST_OBSERVED_FIELD in first_batch else None
        dense_past_time_grid = torch.as_tensor(
            first_batch.get(DENSE_PAST_TIME_GRID_FIELD),
            dtype=torch.float32,
            device=device,
        ) if DENSE_PAST_TIME_GRID_FIELD in first_batch else None
        diagnostics = predict_diagnostics(
            past_target=torch.as_tensor(first_batch["past_target"], dtype=torch.float32, device=device),
            past_observed_values=torch.as_tensor(
                first_batch["past_observed_values"],
                dtype=torch.float32,
                device=device,
            ),
            mean=torch.as_tensor(first_batch["mean"], dtype=torch.float32, device=device),
            scale=torch.as_tensor(first_batch["scale"], dtype=torch.float32, device=device)
            if "scale" in first_batch
            else None,
            num_samples=int(num_samples),
            past_time_grid=torch.as_tensor(first_batch["past_time_grid"], dtype=torch.float32, device=device)
            if "past_time_grid" in first_batch
            else None,
            future_time_grid=torch.as_tensor(first_batch["future_time_grid"], dtype=torch.float32, device=device)
            if "future_time_grid" in first_batch
            else None,
            lag_features=torch.as_tensor(first_batch["lag_features"], dtype=torch.float32, device=device)
            if "lag_features" in first_batch
            else None,
            dense_past_target=dense_past_target,
            dense_past_observed_values=dense_past_observed_values,
            dense_past_time_grid=dense_past_time_grid,
        )

    if ratio > 1:
        diagnostics["fine_past_target"] = fine_past_target
        diagnostics["coarse_past_target"] = coarse_past_target
        diagnostics["fine_to_coarse_ratio"] = ratio
        if diagnostics.get("future_forecast_samples") is not None:
            diagnostics["fine_future_forecast_samples"] = diagnostics["future_forecast_samples"].detach().cpu().clone()
        if diagnostics.get("gp_mean_context") is not None:
            diagnostics["fine_gp_mean_context"] = diagnostics["gp_mean_context"].detach().cpu().clone()
        if diagnostics.get("gp_mean_future") is not None:
            diagnostics["fine_gp_mean_future"] = diagnostics["gp_mean_future"].detach().cpu().clone()
        if diagnostics.get("context_forecast_samples") is not None:
            diagnostics["context_forecast_samples"] = _downsample_fine_time_tensor(
                diagnostics["context_forecast_samples"],
                ratio,
                time_dim=2,
                expected_length=int(resolved_prediction_length),
            )
        if diagnostics.get("gp_mean_context") is not None:
            diagnostics["gp_mean_context"] = _downsample_fine_time_tensor(
                diagnostics["gp_mean_context"],
                ratio,
                time_dim=1,
                expected_length=int(resolved_prediction_length),
            )
        if diagnostics.get("gp_mean_future") is not None:
            diagnostics["gp_mean_future"] = _downsample_fine_time_tensor(
                diagnostics["gp_mean_future"],
                ratio,
                time_dim=1,
                expected_length=int(resolved_prediction_length),
            )

    batch_size = len(input_entries)

    def _to_numpy(values):
        if torch.is_tensor(values):
            return values.detach().cpu().numpy()
        return np.asarray(values)

    def _split_sample_batches(values):
        arr = _to_numpy(values)
        if arr.ndim < 3:
            raise ValueError(f"Expected at least 3 dims for sampled diagnostics, got shape {arr.shape}.")
        if arr.shape[0] == batch_size:
            return {idx: arr[idx] for idx in range(batch_size)}
        if arr.shape[1] == batch_size:
            return {idx: arr[:, idx] for idx in range(batch_size)}
        raise ValueError(f"Could not infer batch axis for sampled diagnostics with shape {arr.shape}.")

    def _split_curves(values):
        arr = _to_numpy(values)
        if arr.ndim < 2:
            raise ValueError(f"Expected at least 2 dims for curve diagnostics, got shape {arr.shape}.")
        if arr.shape[0] != batch_size:
            raise ValueError(f"Expected batch axis of size {batch_size}, got shape {arr.shape}.")
        return {idx: arr[idx] for idx in range(batch_size)}

    context_forecast = diagnostics.get("context_forecast_samples")
    gp_mean_context = diagnostics.get("gp_mean_context")
    gp_mean_future = diagnostics.get("gp_mean_future")
    result = {
        "context_forecast_samples": None if context_forecast is None else _split_sample_batches(context_forecast),
        "gp_mean_context": None if gp_mean_context is None else _split_curves(gp_mean_context),
        "gp_mean_future": None if gp_mean_future is None else _split_curves(gp_mean_future),
    }
    if ratio > 1:
        result["fine_past_target"] = _split_curves(diagnostics["fine_past_target"])
        result["coarse_past_target"] = _split_curves(diagnostics["coarse_past_target"])
        if diagnostics.get("fine_future_forecast_samples") is not None:
            result["fine_future_forecast_samples"] = _split_sample_batches(diagnostics["fine_future_forecast_samples"])
        if diagnostics.get("fine_gp_mean_context") is not None:
            result["fine_gp_mean_context"] = _split_curves(diagnostics["fine_gp_mean_context"])
        if diagnostics.get("fine_gp_mean_future") is not None:
            result["fine_gp_mean_future"] = _split_curves(diagnostics["fine_gp_mean_future"])
        result["fine_to_coarse_ratio"] = ratio
    return result


def _render_first_series_plot_figure(
    *,
    spec: BenchmarkSpec,
    payload: FirstSeriesPlotPayload,
):
    if not payload.forecasts or not payload.tss:
        return None

    plot_model_params = dict(payload.model_params)
    plot_model_params["prediction_length"] = int(payload.prediction_length)
    plotter = GluonTSEvalPlotCallback(
        test_dataset=[],
        transformation=None,
        model_params=plot_model_params,
        setting=payload.setting,
        max_show=max(2, len(payload.selected_indices)),
        save_dir=".",
    )
    left_idx = payload.selected_indices[0] if payload.selected_indices else 0
    right_idx = payload.selected_indices[1] if len(payload.selected_indices) > 1 else left_idx

    coarse_fig = plotter._plot_forecasts(
        payload.forecasts,
        payload.tss,
        payload.metrics_per_ts,
        epoch=None,
        context_forecast_map=None
        if payload.plot_diagnostics is None or payload.plot_diagnostics.get("context_forecast_samples") is None
        else payload.plot_diagnostics["context_forecast_samples"],
        gp_mean_context_map=None
        if payload.plot_diagnostics is None or payload.plot_diagnostics.get("gp_mean_context") is None
        else payload.plot_diagnostics["gp_mean_context"],
        gp_mean_future_map=None
        if payload.plot_diagnostics is None or payload.plot_diagnostics.get("gp_mean_future") is None
        else payload.plot_diagnostics["gp_mean_future"],
        time_grid_map=None
        if payload.plot_diagnostics is None or payload.plot_diagnostics.get("time_grid_map") is None
        else payload.plot_diagnostics["time_grid_map"],
        selected_indices=[left_idx],
        title_prefix=f"{spec.label} on {payload.eval_dataset_name}",
        show_aggregate_panel=False,
        show_error_panel=False,
        compact_mode=True,
    )
    if payload.plot_diagnostics is None or payload.plot_diagnostics.get("fine_past_target") is None:
        return coarse_fig

    coarse_image = _figure_to_rgb_array(coarse_fig)
    plt.close(coarse_fig)

    fine_map = payload.plot_diagnostics.get("fine_past_target", {})
    coarse_map = payload.plot_diagnostics.get("coarse_past_target", {})
    fine_future_forecast_map = payload.plot_diagnostics.get("fine_future_forecast_samples", {})
    fine_gp_context_map = payload.plot_diagnostics.get("fine_gp_mean_context", {})
    fine_gp_future_map = payload.plot_diagnostics.get("fine_gp_mean_future", {})
    ratio = int(payload.plot_diagnostics.get("fine_to_coarse_ratio", 1))

    def _curve(values: np.ndarray) -> np.ndarray:
        arr = np.asarray(values)
        if arr.ndim == 1:
            return arr
        if arr.ndim == 2:
            return arr[:, 0]
        raise ValueError(f"Expected 1D or 2D curve array, got shape {arr.shape}.")

    def _samples(values: np.ndarray) -> np.ndarray:
        arr = np.asarray(values)
        if arr.ndim == 1:
            return arr[None, :]
        if arr.ndim == 2:
            return arr
        if arr.ndim == 3 and arr.shape[-1] == 1:
            return arr[..., 0]
        raise ValueError(f"Expected 1D/2D sample array, got shape {arr.shape}.")

    eval_step_hours = float(get_relative_time_step(str(payload.eval_freq)))
    fine_step_hours = eval_step_hours / float(ratio)

    fine_control_raw = _curve(fine_map[right_idx])
    coarse_context_raw = _curve(coarse_map[right_idx])
    fine_gp_context = _curve(fine_gp_context_map[right_idx]) if right_idx in fine_gp_context_map else None
    fine_gp_future = _curve(fine_gp_future_map[right_idx]) if right_idx in fine_gp_future_map else None
    fine_future_samples = _samples(fine_future_forecast_map[right_idx]) if right_idx in fine_future_forecast_map else None

    if fine_gp_context is not None:
        fine_context_len = len(fine_gp_context)
    else:
        fine_context_len = int(round(float(payload.prediction_length) * float(ratio)))
    coarse_context_len = int(round(float(fine_context_len) / float(ratio)))

    fine_control = fine_control_raw[-fine_context_len:]
    coarse_target_context = coarse_context_raw[-coarse_context_len:]

    target_series = payload.tss[right_idx]
    target_values = target_series.iloc[:, 0].to_numpy() if hasattr(target_series, "iloc") else np.asarray(target_series)
    coarse_future_len = int(plot_model_params["prediction_length"])
    coarse_future = target_values[-coarse_future_len:]

    fine_context_x = (np.arange(-len(fine_control), 0, dtype=float) * fine_step_hours)
    fine_future_start = 0.0
    coarse_context_x = (np.arange(-len(coarse_target_context), 0, dtype=float) * eval_step_hours)
    coarse_future_x = (np.arange(0, len(coarse_future), dtype=float) * eval_step_hours)
    ground_truth_x = np.concatenate([coarse_context_x, coarse_future_x])
    ground_truth_y = np.concatenate([coarse_target_context, coarse_future])

    fig, axes = plt.subplots(1, 2, figsize=(22, 7), facecolor="#f5f1e8")
    axes[0].set_facecolor("#f5f1e8")
    axes[0].imshow(coarse_image)
    axes[0].axis("off")
    axes[0].set_title(f"Coarse Evaluation View (sample {left_idx})", fontsize=15, fontweight="bold", color="#2b2118", pad=10)

    axes[1].set_facecolor("#f5f1e8")
    axes[1].plot(
        fine_context_x,
        fine_control,
        color="#0f5c4d",
        linewidth=1.8,
        drawstyle="steps-post",
        label="Repeated control",
    )
    if fine_gp_context is not None:
        axes[1].plot(
            np.arange(-len(fine_gp_context), 0, dtype=float) * fine_step_hours,
            fine_gp_context,
            color="#2364aa",
            linewidth=1.3,
            linestyle="--",
            label="Fine GP context mean",
        )
    if fine_future_samples is not None:
        future_x = np.arange(0, fine_future_samples.shape[1], dtype=float) * fine_step_hours
        n_show = min(6, fine_future_samples.shape[0])
        for sample_idx in range(n_show):
            axes[1].plot(
                future_x,
                fine_future_samples[sample_idx],
                color="#d1495b",
                linewidth=1.0,
                alpha=0.35,
                label="Fine model samples" if sample_idx == 0 else None,
            )
        axes[1].plot(
            future_x,
            fine_future_samples.mean(axis=0),
            color="#8f1d2c",
            linewidth=2.0,
            label="Fine model mean",
        )
    if fine_gp_future is not None:
        future_x = np.arange(0, len(fine_gp_future), dtype=float) * fine_step_hours
        axes[1].plot(
            future_x,
            fine_gp_future,
            color="#2364aa",
            linewidth=1.3,
            linestyle=":",
            label="Fine GP future mean",
        )
    axes[1].axvline(0.0, color="#2b2118", linewidth=1.0, alpha=0.5)
    axes[1].scatter(
        ground_truth_x,
        ground_truth_y,
        color="#111111",
        s=28,
        marker="o",
        zorder=4,
        label="Ground truth (coarse grid)",
    )
    axes[1].set_title(
        f"Fine-Grid Control/GP/Model, Coarse Target (sample {right_idx})",
        fontsize=15,
        fontweight="bold",
        color="#2b2118",
        pad=10,
    )
    axes[1].set_xlabel("Hours Relative to Forecast Start")
    axes[1].set_ylabel("Value")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(loc="best")

    fig.suptitle(
        f"{spec.label} on {payload.eval_dataset_name}",
        fontsize=18,
        fontweight="bold",
        y=0.98,
        color="#2b2118",
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    return fig


def _figure_to_rgb_array(fig) -> np.ndarray:
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    rgba = buf.reshape(height, width, 4)
    return rgba[..., :3].copy()


def _frequency_to_plain_english(freq: str) -> str:
    value = str(freq).strip()
    match = re.fullmatch(r"(\d+)(min|h|D)", value)
    if match is None:
        return value
    amount = int(match.group(1))
    unit = match.group(2)
    if unit == "min":
        suffix = "minute" if amount == 1 else "minute"
    elif unit == "h":
        suffix = "hour"
    else:
        suffix = "day"
    return f"{amount}-{suffix}"


def _display_model_name_plain(spec: BenchmarkSpec) -> str:
    label = str(spec.label).lower()
    if "slice" in label:
        return "SLiCE"
    if "tsflow" in label or spec.model_type == "tsflow":
        return "TSFlow"
    if "lcde" in label or spec.model_type == "lcde":
        return "LCDE"
    return str(spec.model_type).upper()


def _coerce_plot_index(values: Any) -> tuple[np.ndarray | pd.DatetimeIndex, bool]:
    if values is None:
        return np.array([], dtype=float), False
    if isinstance(values, pd.PeriodIndex):
        return values.to_timestamp(), True
    if isinstance(values, pd.DatetimeIndex):
        return values, True
    if isinstance(values, pd.Index):
        inferred = pd.Index(values)
        if isinstance(inferred, pd.PeriodIndex):
            return inferred.to_timestamp(), True
        if isinstance(inferred, pd.DatetimeIndex):
            return inferred, True
        return inferred.to_numpy(), False
    arr = np.asarray(values)
    if arr.size == 0:
        return arr.astype(float), False
    if np.issubdtype(arr.dtype, np.datetime64):
        return pd.to_datetime(arr), True
    if arr.dtype == object and len(arr) > 0 and isinstance(arr[0], pd.Period):
        return pd.PeriodIndex(arr).to_timestamp(), True
    return arr, False


def _extract_target_series_1d(target_series: Any) -> tuple[np.ndarray | pd.DatetimeIndex, np.ndarray]:
    if isinstance(target_series, pd.DataFrame):
        if target_series.shape[1] == 0:
            return np.array([], dtype=float), np.array([], dtype=float)
        values = target_series.iloc[:, 0].to_numpy(dtype=float)
        index, _ = _coerce_plot_index(target_series.index)
        return index, values
    if isinstance(target_series, pd.Series):
        values = target_series.to_numpy(dtype=float)
        index, _ = _coerce_plot_index(target_series.index)
        return index, values
    arr = np.asarray(target_series, dtype=float)
    return np.arange(arr.shape[0], dtype=float), arr


def _extract_forecast_samples_2d(forecast: Any) -> np.ndarray:
    if hasattr(forecast, "samples"):
        samples = np.asarray(forecast.samples)
    else:
        samples = np.asarray(forecast)
    if samples.ndim == 1:
        return samples[None, :]
    if samples.ndim == 2:
        return samples
    if samples.ndim == 3 and samples.shape[-1] == 1:
        return samples[..., 0]
    raise ValueError(f"Expected forecast samples with 1-3 dims, got shape {samples.shape}.")


def _build_paper_plot_panel_data(
    payload: FirstSeriesPlotPayload,
    *,
    series_index: int,
):
    if series_index >= len(payload.forecasts) or series_index >= len(payload.tss):
        return None

    target_index, target_values = _extract_target_series_1d(payload.tss[series_index])
    prediction_length = int(payload.prediction_length)
    if target_values.shape[0] <= prediction_length:
        return None

    forecast_samples = _extract_forecast_samples_2d(payload.forecasts[series_index])
    if forecast_samples.shape[1] != prediction_length:
        prediction_length = min(prediction_length, forecast_samples.shape[1], target_values.shape[0] - 1)
        forecast_samples = forecast_samples[:, :prediction_length]
    if prediction_length <= 0:
        return None

    context_length = int(payload.model_params.get("context_length", prediction_length))
    history_values = target_values[:-prediction_length]
    future_truth = target_values[-prediction_length:]
    history_start = max(0, history_values.shape[0] - context_length)
    visible_values = np.concatenate([history_values[history_start:], future_truth], axis=0)

    if isinstance(target_index, (pd.DatetimeIndex, pd.Index)) and len(target_index) == len(target_values):
        visible_index = target_index[history_start:]
        forecast_x = target_index[-prediction_length:]
        forecast_start = forecast_x[0]
    else:
        visible_index = np.arange(history_start, target_values.shape[0], dtype=float)
        forecast_x = np.arange(target_values.shape[0] - prediction_length, target_values.shape[0], dtype=float)
        forecast_start = float(target_values.shape[0] - prediction_length)

    forecast_mean = forecast_samples.mean(axis=0)
    forecast_lo = np.quantile(forecast_samples, 0.025, axis=0)
    forecast_hi = np.quantile(forecast_samples, 0.975, axis=0)

    _, uses_datetime = _coerce_plot_index(forecast_x)
    return {
        "visible_index": visible_index,
        "visible_values": visible_values,
        "forecast_x": forecast_x,
        "forecast_mean": forecast_mean,
        "forecast_lo": forecast_lo,
        "forecast_hi": forecast_hi,
        "forecast_start": forecast_start,
        "uses_datetime": uses_datetime,
        "panel_title": f"Evaluated on {_frequency_to_plain_english(payload.eval_freq)} data",
    }


def _to_mathtext_words(text: str) -> str:
    escaped = text.replace(" ", r"\ ")
    return rf"$\mathrm{{{escaped}}}$"


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


def _format_frequency_mathtext(freq: str) -> str:
    return rf"\mathrm{{{str(freq)}}}"


def _frequency_panel_label(eval_freq: str, train_freq: str) -> str:
    eval_math = _format_frequency_mathtext(str(eval_freq))
    if frequencies_match(str(eval_freq), str(train_freq)):
        return rf"$f_{{\mathrm{{train/test}}}}={eval_math}$"
    return rf"$f_{{\mathrm{{test}}}}={eval_math}$"


def _draw_publication_panel(
    *,
    ax,
    panel: dict[str, Any],
    panel_label: str,
    show_x_labels: bool,
    show_y_labels: bool,
    show_panel_label: bool = True,
    panel_label_position: str = "auto",
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
            panel_label,
            position=panel_label_position,
            fontsize=30,
        )
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5 if y_scale == "symlog" else 4, symmetric=(y_scale == "symlog")))
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
    arrays: list[np.ndarray] = []
    for panel in panels:
        arrays.extend(
            [
                np.asarray(panel["visible_values"], dtype=float),
                np.asarray(panel["forecast_mean"], dtype=float),
                np.asarray(panel["forecast_lo"], dtype=float),
                np.asarray(panel["forecast_hi"], dtype=float),
            ]
        )
    finite_arrays = [arr[np.isfinite(arr)] for arr in arrays if arr.size > 0]
    if not finite_arrays:
        return -1.0, 1.0, 1.0
    finite = np.concatenate(finite_arrays, axis=0)
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
    panel_label: str,
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
        panel_label,
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


def _render_checkpoint_paper_plot(
    *,
    spec: BenchmarkSpec,
    plot_payloads: List[FirstSeriesPlotPayload],
):
    if not plot_payloads:
        return None

    _set_plot_theme()
    ordered_payloads = sorted(
        plot_payloads,
        key=lambda payload: (float(get_relative_time_step(str(payload.eval_freq))), str(payload.eval_dataset_name)),
    )
    series_index = ordered_payloads[0].selected_indices[0] if ordered_payloads[0].selected_indices else 0
    train_freq = str(ordered_payloads[0].model_params.get("freq", ordered_payloads[0].eval_freq))

    panels: list[tuple[str, dict[str, Any]]] = []
    for payload in ordered_payloads:
        panel = _build_paper_plot_panel_data(payload, series_index=series_index)
        if panel is not None:
            panels.append((_frequency_panel_label(str(payload.eval_freq), train_freq), panel))
    if not panels:
        return None

    n_panels = len(panels)
    fig_height = max(4.8, 2.55 * n_panels)
    fig, axes = plt.subplots(
        n_panels,
        1,
        figsize=(5.25, fig_height),
        sharex=True,
        facecolor="white",
    )
    if n_panels == 1:
        axes = [axes]

    all_datetime = all(panel["uses_datetime"] for _, panel in panels)

    for idx, (ax, (panel_label, panel)) in enumerate(zip(axes, panels)):
        _draw_publication_panel(
            ax=ax,
            panel=panel,
            panel_label=panel_label,
            show_x_labels=(idx == n_panels - 1),
            show_y_labels=True,
        )

    if all_datetime:
        locator = mdates.AutoDateLocator(minticks=3, maxticks=4)
        axes[-1].xaxis.set_major_locator(locator)
        axes[-1].xaxis.set_major_formatter(
            mticker.FuncFormatter(
                lambda value, _: rf"$\mathdefault{{{mdates.num2date(value).strftime('%H:%M')}}}$"
            )
        )

    fig.legend(
        handles=_make_publication_legend_handles(),
        loc="upper center",
        bbox_to_anchor=(0.56, 0.968),
        ncol=3,
        frameon=False,
        fontsize=20,
        handlelength=1.0,
        handletextpad=0.7,
        columnspacing=1.7,
    )
    fig.subplots_adjust(left=0.12, right=0.99, bottom=0.11, top=0.86, hspace=0.16)
    return fig


def _comparison_model_key(spec: BenchmarkSpec) -> str | None:
    label = str(spec.label).lower()
    if "slice" in label:
        return "slice"
    if "tsflow" in label or spec.model_type == "tsflow":
        return "tsflow"
    return None


def _render_checkpoint_comparison_plot(
    *,
    train_freq: str,
    slice_payloads: List[FirstSeriesPlotPayload],
    tsflow_payloads: List[FirstSeriesPlotPayload],
    use_symlog: bool = False,
):
    _set_plot_theme()

    def _panel_map(payloads: List[FirstSeriesPlotPayload]) -> dict[tuple[float, str], tuple[str, dict[str, Any]]]:
        ordered_payloads = sorted(
            payloads,
            key=lambda payload: (float(get_relative_time_step(str(payload.eval_freq))), str(payload.eval_dataset_name)),
        )
        series_index = ordered_payloads[0].selected_indices[0] if ordered_payloads and ordered_payloads[0].selected_indices else 0
        panels: dict[tuple[float, str], tuple[str, dict[str, Any]]] = {}
        for payload in ordered_payloads:
            panel = _build_paper_plot_panel_data(payload, series_index=series_index)
            if panel is None:
                continue
            key = (float(get_relative_time_step(str(payload.eval_freq))), str(payload.eval_dataset_name))
            panels[key] = (_frequency_panel_label(str(payload.eval_freq), train_freq), panel)
        return panels

    panels_by_model = {
        "slice": _panel_map(slice_payloads),
        "tsflow": _panel_map(tsflow_payloads),
    }
    row_keys = sorted(set(panels_by_model["slice"]) | set(panels_by_model["tsflow"]))
    if not row_keys:
        return None

    n_rows = len(row_keys)
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
    row_symlog_limits: dict[tuple[float, str], tuple[float, float, float]] = {}
    if use_symlog:
        for row_key in row_keys:
            row_panels = [
                model_panels[row_key][1]
                for model_panels in panels_by_model.values()
                if row_key in model_panels
            ]
            if row_panels:
                row_symlog_limits[row_key] = _symmetric_symlog_limits(row_panels)

    for col_idx, (model_key, _) in enumerate(model_columns):
        model_panels = panels_by_model[model_key]
        for row_idx, row_key in enumerate(row_keys):
            ax = axes[row_idx, col_idx]
            panel_entry = model_panels.get(row_key)
            if panel_entry is None:
                ax.axis("off")
                continue
            panel_label, panel = panel_entry
            symlog_limits = row_symlog_limits.get(row_key)
            _draw_publication_panel(
                ax=ax,
                panel=panel,
                panel_label=panel_label,
                show_x_labels=(row_idx == n_rows - 1),
                show_y_labels=True,
                show_panel_label=True,
                panel_label_position="lower_left",
                y_limits=(symlog_limits[:2] if symlog_limits is not None else None),
                y_scale=("symlog" if symlog_limits is not None else "linear"),
                symlog_linthresh=(symlog_limits[2] if symlog_limits is not None else None),
            )

    for col_idx, (model_key, _) in enumerate(model_columns):
        model_panels = panels_by_model[model_key]
        ordered_model_panels = [model_panels[row_key] for row_key in row_keys if row_key in model_panels]
        if ordered_model_panels and all(panel["uses_datetime"] for _, panel in ordered_model_panels):
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
        bbox_to_anchor=(0.55, 0.988),
        ncol=3,
        frameon=False,
        fontsize=25,
        handlelength=2.0,
        handletextpad=0.7,
        columnspacing=1.7,
    )
    fig.text(0.29, 0.00, "TSFlow", ha="center", va="center", fontsize=30, color="black")
    fig.text(0.77, 0.00, "G-SLiCE", ha="center", va="center", fontsize=30, color="black")
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.12, top=0.86, hspace=0.16, wspace=0.18)
    return fig


def _render_overlay_checkpoint_comparison_plot(
    *,
    train_freq: str,
    slice_payloads: List[FirstSeriesPlotPayload],
    tsflow_payloads: List[FirstSeriesPlotPayload],
):
    _set_plot_theme()

    def _panel_map(payloads: List[FirstSeriesPlotPayload]) -> dict[tuple[float, str], tuple[str, dict[str, Any]]]:
        ordered_payloads = sorted(
            payloads,
            key=lambda payload: (float(get_relative_time_step(str(payload.eval_freq))), str(payload.eval_dataset_name)),
        )
        series_index = ordered_payloads[0].selected_indices[0] if ordered_payloads and ordered_payloads[0].selected_indices else 0
        panels: dict[tuple[float, str], tuple[str, dict[str, Any]]] = {}
        for payload in ordered_payloads:
            panel = _build_paper_plot_panel_data(payload, series_index=series_index)
            if panel is None:
                continue
            key = (float(get_relative_time_step(str(payload.eval_freq))), str(payload.eval_dataset_name))
            panels[key] = (_frequency_panel_label(str(payload.eval_freq), train_freq), panel)
        return panels

    panels_by_model = {
        "slice": _panel_map(slice_payloads),
        "tsflow": _panel_map(tsflow_payloads),
    }
    row_keys = sorted(set(panels_by_model["slice"]) & set(panels_by_model["tsflow"]))
    if not row_keys:
        return None

    fig, axes = plt.subplots(
        len(row_keys),
        1,
        figsize=(6.4, max(4.8, 2.8 * len(row_keys))),
        sharex=False,
        facecolor="white",
    )
    if len(row_keys) == 1:
        axes = [axes]

    for row_idx, row_key in enumerate(row_keys):
        panel_label, slice_panel = panels_by_model["slice"][row_key]
        _, tsflow_panel = panels_by_model["tsflow"][row_key]
        _draw_overlay_comparison_panel(
            ax=axes[row_idx],
            panel_label=panel_label,
            slice_panel=slice_panel,
            tsflow_panel=tsflow_panel,
            show_x_labels=(row_idx == len(row_keys) - 1),
            show_y_labels=True,
        )

    if all(
        panel["uses_datetime"]
        for row_key in row_keys
        for _, panel in (panels_by_model["slice"][row_key], panels_by_model["tsflow"][row_key])
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


def _save_comparison_plots(
    *,
    comparison_dir: Path,
    spec_plot_payloads: list[tuple[BenchmarkSpec, List[FirstSeriesPlotPayload]]],
    overlay_models_symlog: bool,
) -> list[str]:
    grouped: dict[str, dict[str, List[FirstSeriesPlotPayload]]] = {}
    seeds_by_freq: dict[str, Optional[int]] = {}
    num_samples_by_freq: dict[str, int] = {}

    for spec, plot_payloads in spec_plot_payloads:
        model_key = _comparison_model_key(spec)
        if model_key not in {"slice", "tsflow"} or not plot_payloads:
            continue
        spec_config = _read_yaml(spec.config_path)
        train_freq = str(spec_config.get("model_params", {}).get("freq", plot_payloads[0].eval_freq))
        grouped.setdefault(train_freq, {})[model_key] = plot_payloads
        seeds_by_freq[train_freq] = plot_payloads[0].eval_seed
        num_samples_by_freq[train_freq] = int(plot_payloads[0].num_samples)

    saved_paths: list[str] = []
    comparison_dir.mkdir(parents=True, exist_ok=True)
    for train_freq, payloads_by_model in sorted(
        grouped.items(),
        key=lambda item: (float(get_relative_time_step(str(item[0]))), str(item[0])),
    ):
        if "slice" not in payloads_by_model or "tsflow" not in payloads_by_model:
            continue
        fig = _render_checkpoint_comparison_plot(
            train_freq=train_freq,
            slice_payloads=payloads_by_model["slice"],
            tsflow_payloads=payloads_by_model["tsflow"],
            use_symlog=overlay_models_symlog,
        )
        if fig is None:
            continue
        stem = (
            f"comparison__trainfreq_{_slugify(train_freq)}__ns{num_samples_by_freq[train_freq]}__seed"
            f"{'none' if seeds_by_freq[train_freq] is None else int(seeds_by_freq[train_freq])}"
            f"{'__symlog' if overlay_models_symlog else ''}"
        )
        png_path = comparison_dir / f"{stem}.png"
        pdf_path = comparison_dir / f"{stem}.pdf"
        fig.savefig(
            png_path,
            dpi=600,
            bbox_inches="tight",
            pad_inches=0.03,
            facecolor=fig.get_facecolor(),
        )
        fig.savefig(
            pdf_path,
            bbox_inches="tight",
            pad_inches=0.03,
            facecolor=fig.get_facecolor(),
        )
        plt.close(fig)
        saved_paths.append(str(png_path))
    return saved_paths


def _save_checkpoint_summary_plot(
    *,
    plot_dir: Optional[Path],
    spec: BenchmarkSpec,
    plot_payloads: List[FirstSeriesPlotPayload],
) -> Optional[str]:
    if plot_dir is None or not plot_payloads:
        return None

    plot_dir.mkdir(parents=True, exist_ok=True)
    filename = (
        f"{_slugify(spec.label)}__all_eval_datasets__ns{int(plot_payloads[0].num_samples)}__seed"
        f"{'none' if plot_payloads[0].eval_seed is None else int(plot_payloads[0].eval_seed)}.png"
    )
    plot_path = plot_dir / filename

    paper_fig = _render_checkpoint_paper_plot(spec=spec, plot_payloads=plot_payloads)
    primary_plot_created = False
    if paper_fig is not None:
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plot_path_pdf = plot_dir / f"{plot_path.stem}.pdf"
        compatibility_png = plot_dir / f"{plot_path.stem}__paper.png"
        compatibility_pdf = plot_dir / f"{plot_path.stem}__paper.pdf"
        paper_fig.savefig(
            plot_path,
            dpi=600,
            bbox_inches="tight",
            pad_inches=0.03,
            facecolor=paper_fig.get_facecolor(),
        )
        paper_fig.savefig(
            plot_path_pdf,
            bbox_inches="tight",
            pad_inches=0.03,
            facecolor=paper_fig.get_facecolor(),
        )
        paper_fig.savefig(
            compatibility_png,
            dpi=600,
            bbox_inches="tight",
            pad_inches=0.03,
            facecolor=paper_fig.get_facecolor(),
        )
        paper_fig.savefig(
            compatibility_pdf,
            bbox_inches="tight",
            pad_inches=0.03,
            facecolor=paper_fig.get_facecolor(),
        )
        plt.close(paper_fig)
        primary_plot_created = True

    rendered_images: list[tuple[str, np.ndarray]] = []
    for payload in plot_payloads:
        fig = _render_first_series_plot_figure(spec=spec, payload=payload)
        if fig is None:
            continue
        rendered_images.append((payload.eval_dataset_name, _figure_to_rgb_array(fig)))
        plt.close(fig)

    if rendered_images:
        diagnostic_path = plot_path if not primary_plot_created else plot_dir / f"{plot_path.stem}__diagnostic.png"
        n_panels = len(rendered_images)
        fig_height = max(5.5, 5.25 * n_panels)
        summary_fig, axes = plt.subplots(n_panels, 1, figsize=(18, fig_height), facecolor="#f5f1e8")
        if n_panels == 1:
            axes = [axes]

        for ax, (dataset_name, image) in zip(axes, rendered_images):
            ax.set_facecolor("#f5f1e8")
            ax.imshow(image)
            ax.set_title(dataset_name, fontsize=15, fontweight="bold", pad=10, color="#2b2118")
            ax.axis("off")

        summary_fig.suptitle(
            f"{spec.label} cross-frequency diagnostics",
            fontsize=18,
            fontweight="bold",
            y=0.995,
            color="#2b2118",
        )
        summary_fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.985))
        summary_fig.savefig(
            diagnostic_path,
            dpi=240,
            bbox_inches="tight",
            facecolor=summary_fig.get_facecolor(),
        )
        plt.close(summary_fig)
        primary_plot_created = True

    if not primary_plot_created:
        return None

    return str(plot_path)


@torch.no_grad()
def _make_evaluation_predictions_with_dataset_horizon(
    *,
    spec: BenchmarkSpec,
    model,
    dataset,
    test_splitter,
    prediction_length: int,
    num_samples: int,
    batch_size: int,
    device: torch.device,
):
    window_length = int(prediction_length)
    _, test_template = split(dataset, offset=-window_length)
    test_instances = test_template.generate_instances(window_length)
    test_pairs = list(test_instances)
    if not test_pairs:
        return [], []

    # keep only one pair per series (last window) so we feed a single contiguous input to the model
    grouped: dict[str | int, tuple[dict, dict]] = {}
    for idx, (input_entry, label_entry) in enumerate(test_pairs):
        item_id = label_entry.get(FieldName.ITEM_ID, None)
        key = item_id if item_id is not None else idx
        start = label_entry[FieldName.START]
        existing = grouped.get(key)
        if existing is None or start >= existing[1][FieldName.START]:
            grouped[key] = (input_entry, label_entry)
    test_pairs = list(grouped.values())

    if not test_pairs:
        return [], []

    input_entries = [input_entry for input_entry, _ in test_pairs]
    tss = [_to_dataframe(pair) for pair in test_pairs]

    data_loader = InferenceDataLoader(
        Cached(input_entries),
        batch_size=int(batch_size),
        stack_fn=lambda data: batchify(data, device),
        transform=test_splitter,
    )

    forecasts: list[SampleForecast] = []
    total_batches = (len(input_entries) + int(batch_size) - 1) // int(batch_size)
    cursor = 0

    for batch in tqdm(data_loader, total=total_batches, desc=f"{spec.label}:rollout"):
        output = _run_inference_for_horizon(
            spec.model_type,
            model,
            batch,
            int(num_samples),
            int(prediction_length),
        )
        output_np = output.detach().cpu().numpy()  # (B, S, T, D)
        batch_n = int(output_np.shape[0])
        batch_pairs = test_pairs[cursor : cursor + batch_n]

        for idx, (_, label_entry) in enumerate(batch_pairs):
            samples = output_np[idx]
            if samples.shape[-1] == 1:
                samples = np.squeeze(samples, axis=-1)
            forecasts.append(
                SampleForecast(
                    samples=samples,
                    start_date=label_entry[FieldName.START],
                    item_id=label_entry.get(FieldName.ITEM_ID),
                )
            )

        cursor += batch_n

    return forecasts, tss


@torch.no_grad()
def _make_evaluation_predictions_with_fine_grid_adapter(
    *,
    spec: BenchmarkSpec,
    model,
    dataset,
    test_splitter,
    coarse_prediction_length: int,
    num_samples: int,
    batch_size: int,
    device: torch.device,
    model_params: Dict[str, Any],
    train_freq: str,
    eval_freq: str,
    adapter_mode: str,
):
    ratio = _resolve_fine_to_coarse_ratio(train_freq, eval_freq)
    if ratio is None:
        raise ValueError("Fine-grid adapter predictions require a finer training grid than evaluation grid.")

    native_prediction_length = int(getattr(model, "prediction_length"))
    expected_coarse_length = native_prediction_length / float(ratio)
    rounded_coarse_length = int(round(expected_coarse_length))
    if not np.isclose(expected_coarse_length, rounded_coarse_length):
        raise ValueError(
            f"Native model prediction_length={native_prediction_length} is not divisible by fine-to-coarse "
            f"ratio={ratio}."
        )
    if rounded_coarse_length != int(coarse_prediction_length):
        raise ValueError(
            f"Fine-grid adapter expected coarse prediction length {rounded_coarse_length}, "
            f"got evaluation prediction length {coarse_prediction_length}."
        )

    window_length = int(coarse_prediction_length)
    _, test_template = split(dataset, offset=-window_length)
    test_instances = test_template.generate_instances(window_length)
    test_pairs = list(test_instances)
    if not test_pairs:
        return [], []

    grouped: dict[str | int, tuple[dict, dict]] = {}
    for idx, (input_entry, label_entry) in enumerate(test_pairs):
        item_id = label_entry.get(FieldName.ITEM_ID, None)
        key = item_id if item_id is not None else idx
        start = label_entry[FieldName.START]
        existing = grouped.get(key)
        if existing is None or start >= existing[1][FieldName.START]:
            grouped[key] = (input_entry, label_entry)
    test_pairs = list(grouped.values())
    if not test_pairs:
        return [], []

    input_entries = [input_entry for input_entry, _ in test_pairs]
    tss = [_to_dataframe(pair) for pair in test_pairs]
    data_loader = InferenceDataLoader(
        Cached(input_entries),
        batch_size=int(batch_size),
        stack_fn=lambda data: batchify(data, device),
        transform=test_splitter,
    )

    forecasts: list[SampleForecast] = []
    total_batches = (len(input_entries) + int(batch_size) - 1) // int(batch_size)
    cursor = 0

    for batch in tqdm(data_loader, total=total_batches, desc=f"{spec.label}:adapted-rollout"):
        adapted_batch, adapted_ratio = _adapt_tsflow_batch_to_train_grid(
            batch=batch,
            model=model,
            model_params=model_params,
            train_freq=train_freq,
            eval_freq=eval_freq,
            adapter_mode=adapter_mode,
        )
        output = _run_inference_for_horizon(
            spec.model_type,
            model,
            adapted_batch,
            int(num_samples),
            native_prediction_length,
        )
        output = _downsample_fine_time_tensor(
            output,
            adapted_ratio,
            time_dim=2,
            expected_length=int(coarse_prediction_length),
        )
        output_np = output.detach().cpu().numpy()
        batch_n = int(output_np.shape[0])
        batch_pairs = test_pairs[cursor : cursor + batch_n]

        for idx, (_, label_entry) in enumerate(batch_pairs):
            samples = output_np[idx]
            if samples.shape[-1] == 1:
                samples = np.squeeze(samples, axis=-1)
            forecasts.append(
                SampleForecast(
                    samples=samples,
                    start_date=label_entry[FieldName.START],
                    item_id=label_entry.get(FieldName.ITEM_ID),
                )
            )

        cursor += batch_n

    return forecasts, tss


def _make_evaluation_predictions_with_irregular_adapter(
    *,
    spec: BenchmarkSpec,
    model,
    dataset,
    transformation,
    num_samples: int,
    batch_size: int,
    train_dataset_params: Dict[str, Any],
    eval_dataset_params: Dict[str, Any],
    model_params: Dict[str, Any],
    eval_freq: str,
):
    window_length = int(get_irregular_grid_spec(eval_dataset_params, fallback_freq=eval_freq).dense_prediction_length)
    _, test_template = split(dataset, offset=-window_length)
    test_pairs = list(test_template.generate_instances(window_length))
    if not test_pairs:
        return [], []

    input_entries = [
        next(iter(transformation.apply([input_entry], is_train=False)))
        for input_entry, _ in test_pairs
    ]
    transformed_entries = [
        _build_irregular_eval_entry(
            input_entry=input_entry,
            label_entry=label_entry,
            dataset_params=eval_dataset_params,
            freq=eval_freq,
            model_params=model_params,
        )
        for input_entry, (_, label_entry) in zip(input_entries, test_pairs)
    ]
    tss = [_irregular_entry_to_dataframe(entry) for entry in transformed_entries]
    device = next(model.parameters()).device

    forecasts: list[SampleForecast] = []
    total_batches = (len(transformed_entries) + int(batch_size) - 1) // int(batch_size)
    cursor = 0

    for batch_start in tqdm(
        range(0, len(transformed_entries), int(batch_size)),
        total=total_batches,
        desc=f"{spec.label}:irregular-adapted-rollout",
    ):
        batch_entries = transformed_entries[batch_start : batch_start + int(batch_size)]
        batch = batchify(batch_entries, device)
        adapted_batch = _adapt_tsflow_batch_to_train_irregular_grid(
            batch=batch,
            train_dataset_params=train_dataset_params,
            eval_dataset_params=eval_dataset_params,
            model=model,
            model_params=model_params,
            adapter_mode=_get_fine_to_coarse_eval_adapter_mode(train_dataset_params),
        )
        output = _run_inference_for_horizon(
            spec.model_type,
            model,
            adapted_batch,
            int(num_samples),
            int(getattr(model, "prediction_length")),
        )
        output_np = output.detach().cpu().numpy()
        batch_n = int(output_np.shape[0])
        batch_entries = transformed_entries[cursor : cursor + batch_n]
        for idx, entry in enumerate(batch_entries):
            samples = output_np[idx]
            if samples.shape[-1] == 1:
                samples = np.squeeze(samples, axis=-1)
            forecasts.append(
                SampleForecast(
                    samples=samples,
                    start_date=entry[FieldName.FORECAST_START],
                    item_id=entry.get(FieldName.ITEM_ID),
                )
            )
        cursor += batch_n

    return forecasts, tss


def _make_evaluation_predictions_training_path(
    *,
    spec: BenchmarkSpec,
    model,
    dataset,
    test_splitter,
    num_samples: int,
    batch_size: int,
    dataset_params: Optional[Dict[str, Any]] = None,
):
    if spec.model_type == "tsflow":
        model.num_samples = int(num_samples)
        predictor_device = getattr(model, "device", None)
        if predictor_device is not None:
            predictor_device = str(predictor_device)
        predictor = model.get_predictor(
            test_splitter,
            batch_size=batch_size,
            device=predictor_device,
            input_names=maybe_get_irregular_input_names(
                dataset_params or {},
                use_lags=bool(getattr(model, "use_lags", True)),
            ),
        )
    elif spec.model_type == "lcde":
        predictor = model.get_predictor(
            input_transform=test_splitter,
            batch_size=batch_size,
            input_names=maybe_get_irregular_input_names(
                dataset_params or {},
                use_lags=bool(getattr(model, "use_lags", True)),
            ),
        )
    else:
        raise ValueError(f"Unknown model type: {spec.model_type}")

    forecast_it, ts_it = make_evaluation_predictions(
        dataset=dataset,
        predictor=predictor,
        num_samples=int(num_samples),
    )
    forecasts = list(tqdm(forecast_it, total=len(dataset), desc=f"{spec.label}:predict"))
    tss = list(ts_it)
    return forecasts, tss


def _is_native_in_distribution_eval(
    *,
    train_dataset_params: Dict[str, Any],
    eval_dataset_params: Dict[str, Any],
    train_freq: str,
    eval_freq: str,
    train_prediction_length: int,
    eval_prediction_length: int,
) -> bool:
    return (
        dict(train_dataset_params) == dict(eval_dataset_params)
        and frequencies_match(str(train_freq), str(eval_freq))
        and int(train_prediction_length) == int(eval_prediction_length)
    )


def _resolve_univariate_eval_batch_size(
    *,
    spec: BenchmarkSpec,
    native_in_distribution_eval: bool,
    num_samples: int,
    dataset_params: Dict[str, Any],
    batch_size_override: Optional[int],
) -> int:
    if batch_size_override is not None:
        return int(batch_size_override)

    if spec.model_type == "tsflow" and native_in_distribution_eval:
        return max(1, (1024 * 64) // int(num_samples))

    return int(dataset_params.get("test_batch_size", dataset_params["batch_size"]))


def _collect_native_tsflow_plot_payload(
    *,
    setting: str,
    model_params: Dict[str, Any],
    model,
    test_data,
    transformation,
    transformed_testdata,
    test_splitter,
    resolved_eval_freq: str,
    resolved_prediction_length: int,
    batch_size: int,
    num_samples: int,
    eval_seed: Optional[int],
    eval_dataset_name: str,
):
    callback = GluonTSEvalPlotCallback(
        test_dataset=test_data,
        transformation=transformation,
        model_params={
            **dict(model_params),
            "freq": str(resolved_eval_freq),
            "context_length": int(resolved_prediction_length),
            "prediction_length": int(resolved_prediction_length),
        },
        setting=setting,
        eval_every=1,
        num_samples=int(num_samples),
        batch_size=int(batch_size),
        max_show=2,
        save_dir=".",
        eval_seed=eval_seed,
    )
    with temporary_random_seed(eval_seed):
        forecasts, tss, context_forecast_map, gp_mean_context_map, gp_mean_future_map, time_grid_map = (
            callback._make_evaluation_predictions_with_plot_diagnostics(
                pl_module=model,
                dataset=transformed_testdata,
                test_transform=test_splitter,
                uses_irregular_grid=False,
            )
        )

    return forecasts, tss, {
        "context_forecast_samples": context_forecast_map,
        "gp_mean_context": gp_mean_context_map,
        "gp_mean_future": gp_mean_future_map,
        "time_grid_map": time_grid_map,
    }


def _evaluate_checkpoint_on_dataset(
    spec: BenchmarkSpec,
    config: Dict[str, Any],
    model,
    eval_dataset_name: str,
    eval_dataset_params: Dict[str, Any],
    eval_freq: Optional[str],
    eval_prediction_length: Optional[int],
    num_samples: int,
    eval_seed: Optional[int],
    plot_dir: Optional[Path],
    max_eval_instances: Optional[int],
    batch_size_override: Optional[int],
    regenerate: bool,
    adapt_gp_to_eval_dataset: bool,
) -> tuple[CrossFrequencyEvalResult, Optional[FirstSeriesPlotPayload]]:
    setting = config["setting"]
    dataset_params = config["dataset_params"]
    model_params = config["model_params"]
    native_model_prediction_length = int(model.prediction_length)
    resolved_eval_freq, resolved_prediction_length = _resolve_eval_dataset_metadata(
        eval_dataset_params=eval_dataset_params,
        eval_freq=eval_freq,
        eval_prediction_length=eval_prediction_length,
        regenerate=regenerate,
    )
    eval_context_length = int(resolved_prediction_length)
    native_in_distribution_eval = _is_native_in_distribution_eval(
        train_dataset_params=dataset_params,
        eval_dataset_params=eval_dataset_params,
        train_freq=str(model_params["freq"]),
        eval_freq=str(resolved_eval_freq),
        train_prediction_length=int(model_params["prediction_length"]),
        eval_prediction_length=int(resolved_prediction_length),
    )
    fine_to_coarse_eval_adapter_mode = _get_fine_to_coarse_eval_adapter_mode(dataset_params)
    fine_to_coarse_ratio = _resolve_fine_to_coarse_ratio(str(model_params["freq"]), str(resolved_eval_freq))
    use_irregular_grid_eval_adapter = _should_use_irregular_grid_eval_adapter(
        model_type=spec.model_type,
        native_in_distribution_eval=native_in_distribution_eval,
        train_dataset_params=dataset_params,
        eval_dataset_params=eval_dataset_params,
    )
    use_fine_grid_eval_adapter = (
        spec.model_type == "tsflow"
        and not native_in_distribution_eval
        and fine_to_coarse_eval_adapter_mode != "none"
        and fine_to_coarse_ratio is not None
    )
    adapter_past_length = (
        _resolve_adapter_eval_past_length(
            model=model,
            train_freq=str(model_params["freq"]),
            eval_freq=str(resolved_eval_freq),
        )
        if use_fine_grid_eval_adapter
        else None
    )

    with _temporary_model_prediction_length(
        model,
        int(resolved_prediction_length),
        enabled=spec.model_type == "tsflow" and not native_in_distribution_eval and not use_fine_grid_eval_adapter and not use_irregular_grid_eval_adapter,
    ):
        with _temporary_eval_adapted_gp(
            spec=spec,
            config=config,
            model=model,
            eval_freq=resolved_eval_freq,
            enabled=adapt_gp_to_eval_dataset and not native_in_distribution_eval and not use_fine_grid_eval_adapter and not use_irregular_grid_eval_adapter,
        ):
            with _temporary_model_frequency_attrs(
                model,
                eval_freq=str(resolved_eval_freq),
                context_length=int(eval_context_length),
                enabled=spec.model_type == "tsflow" and not native_in_distribution_eval and not use_fine_grid_eval_adapter and not use_irregular_grid_eval_adapter,
            ):
                (
                    test_data,
                    transformation,
                    past_length,
                    resolved_eval_freq,
                    resolved_prediction_length,
                    resolved_context_length,
                    source_prediction_length,
                ) = _prepare_eval_data(
                    config=config,
                    model=model,
                    eval_dataset_params=eval_dataset_params,
                    eval_freq=resolved_eval_freq,
                    eval_prediction_length=resolved_prediction_length,
                    regenerate=regenerate,
                    past_length_override=adapter_past_length,
                )
                test_data = _truncate_dataset(test_data, resolved_eval_freq, max_eval_instances)

                transformed_testdata = transformation.apply(test_data, is_train=False)
                test_splitter = create_splitter(
                    past_length=past_length,
                    future_length=int(source_prediction_length),
                    mode="test",
                    include_time_grid=_uses_irregular_grid(eval_dataset_params),
                )
                test_splitter = _build_eval_instance_transform(
                    splitter=test_splitter,
                    dataset_params=eval_dataset_params,
                    lag_steps=list(getattr(model, "lags_seq", []) or []),
                    freq=str(resolved_eval_freq),
                )

                if setting == "univariate":
                    batch_size = _resolve_univariate_eval_batch_size(
                        spec=spec,
                        native_in_distribution_eval=native_in_distribution_eval,
                        num_samples=int(num_samples),
                        dataset_params=dataset_params,
                        batch_size_override=batch_size_override,
                    )
                    # Irregular adapted evals are diagnostic-heavy and otherwise often
                    # collapse the whole test set into a single opaque 0/1 batch.
                    if use_irregular_grid_eval_adapter and batch_size_override is None:
                        batch_size = 1
                    evaluator = Evaluator(num_workers=1)
                else:
                    batch_size = 1
                    evaluator = MultivariateEvaluator(target_agg_funcs={"sum": np.sum})

                eval_model_params = dict(model_params)
                eval_model_params["freq"] = str(resolved_eval_freq)
                eval_model_params["context_length"] = int(resolved_context_length)
                eval_model_params["prediction_length"] = int(resolved_prediction_length)
                with (
                    nullcontext()
                    if use_fine_grid_eval_adapter
                    else _temporary_model_context_length(model, int(resolved_context_length))
                ):
                    with temporary_random_seed(eval_seed):
                        if use_fine_grid_eval_adapter:
                            forecasts, tss = _make_evaluation_predictions_with_fine_grid_adapter(
                                spec=spec,
                                model=model,
                                dataset=transformed_testdata,
                                test_splitter=test_splitter,
                                coarse_prediction_length=int(resolved_prediction_length),
                                num_samples=int(num_samples),
                                batch_size=batch_size,
                                device=next(model.parameters()).device,
                                model_params=model_params,
                                train_freq=str(model_params["freq"]),
                                eval_freq=str(resolved_eval_freq),
                                adapter_mode=fine_to_coarse_eval_adapter_mode,
                            )
                        elif use_irregular_grid_eval_adapter:
                            forecasts, tss = _make_evaluation_predictions_with_irregular_adapter(
                                spec=spec,
                                model=model,
                                dataset=test_data,
                                transformation=transformation,
                                num_samples=int(num_samples),
                                batch_size=batch_size,
                                train_dataset_params=dataset_params,
                                eval_dataset_params=eval_dataset_params,
                                model_params=model_params,
                                eval_freq=str(resolved_eval_freq),
                            )
                        elif resolved_prediction_length == native_model_prediction_length:
                            forecasts, tss = _make_evaluation_predictions_training_path(
                                spec=spec,
                                model=model,
                                dataset=transformed_testdata,
                                test_splitter=test_splitter,
                                num_samples=int(num_samples),
                                batch_size=batch_size,
                                dataset_params=eval_dataset_params,
                            )
                        else:
                            forecasts, tss = _make_evaluation_predictions_with_dataset_horizon(
                                spec=spec,
                                model=model,
                                dataset=transformed_testdata,
                                test_splitter=test_splitter,
                                prediction_length=resolved_prediction_length,
                                num_samples=int(num_samples),
                                batch_size=batch_size,
                                device=next(model.parameters()).device,
                            )
                        metrics, metrics_per_ts = evaluator(tss, forecasts)
                metrics["CRPS"] = metrics["mean_wQuantileLoss"]
                select = ["CRPS", "ND", "NRMSE"]
                if setting == "multivariate":
                    metrics["m_sum_CRPS"] = metrics["m_sum_mean_wQuantileLoss"]
                    select.append("m_sum_CRPS")
                metrics = filter_metrics(metrics, select)

                with (
                    nullcontext()
                    if use_fine_grid_eval_adapter
                    else _temporary_model_context_length(model, int(resolved_context_length))
                ):
                    if (
                        plot_dir is not None
                        and spec.model_type == "tsflow"
                        and native_in_distribution_eval
                        and not _uses_irregular_grid(eval_dataset_params)
                    ):
                        plot_forecasts, plot_tss, plot_diagnostics = _collect_native_tsflow_plot_payload(
                            setting=setting,
                            model_params=eval_model_params,
                            model=model,
                            test_data=test_data,
                            transformation=transformation,
                            transformed_testdata=transformed_testdata,
                            test_splitter=test_splitter,
                            resolved_eval_freq=resolved_eval_freq,
                            resolved_prediction_length=int(resolved_prediction_length),
                            batch_size=int(batch_size),
                            num_samples=int(num_samples),
                            eval_seed=eval_seed,
                            eval_dataset_name=eval_dataset_name,
                        )
                    else:
                        plot_forecasts, plot_tss = forecasts, tss
                        plot_diagnostics = None if use_irregular_grid_eval_adapter else _collect_first_series_plot_diagnostics(
                            model=model,
                            dataset=transformed_testdata,
                            test_splitter=test_splitter,
                            resolved_prediction_length=int(resolved_prediction_length),
                            model_prediction_length=int(native_model_prediction_length),
                            num_samples=int(num_samples),
                            eval_seed=eval_seed,
                            fine_to_coarse_eval_adapter_mode=fine_to_coarse_eval_adapter_mode
                            if use_fine_grid_eval_adapter
                            else "none",
                            train_freq=str(model_params["freq"]),
                            eval_freq=str(resolved_eval_freq),
                            model_params=model_params,
                        )
                    plot_payload = FirstSeriesPlotPayload(
                        setting=setting,
                        model_params=eval_model_params,
                        forecasts=plot_forecasts,
                        tss=plot_tss,
                        metrics_per_ts=metrics_per_ts,
                        eval_dataset_name=eval_dataset_name,
                        eval_freq=resolved_eval_freq,
                        prediction_length=int(resolved_prediction_length),
                        num_samples=int(num_samples),
                        eval_seed=None if eval_seed is None else int(eval_seed),
                        selected_indices=list(range(min(2, len(plot_forecasts)))),
                        plot_diagnostics=plot_diagnostics,
                    )

                return CrossFrequencyEvalResult(
                    label=spec.label,
                    model_type=spec.model_type,
                    checkpoint_path=spec.checkpoint_path,
                    config_path=spec.config_path,
                    train_dataset=get_dataset_name_from_params(dataset_params),
                    train_freq=str(model_params["freq"]),
                    eval_dataset=eval_dataset_name,
                    eval_freq=resolved_eval_freq,
                    setting=setting,
                    device=str(next(model.parameters()).device),
                    num_samples=int(num_samples),
                    eval_seed=None if eval_seed is None else int(eval_seed),
                    plot_path=None,
                    max_eval_instances=None if max_eval_instances is None else int(max_eval_instances),
                    prediction_length=int(resolved_prediction_length),
                    model_prediction_length=native_model_prediction_length,
                    past_length=int(past_length),
                    CRPS=float(metrics["CRPS"]),
                    ND=float(metrics["ND"]),
                    NRMSE=float(metrics["NRMSE"]),
                    m_sum_CRPS=float(metrics["m_sum_CRPS"]) if "m_sum_CRPS" in metrics else None,
                ), plot_payload


def main():
    parser = argparse.ArgumentParser(
        description="Load TSFlow/LCDE checkpoints and evaluate them on GluonTS datasets with different frequencies."
    )
    parser.add_argument(
        "--checkpoint_paths",
        type=str,
        required=False,
        help="Python list of checkpoint paths.",
    )
    parser.add_argument(
        "--config_paths",
        type=str,
        default=None,
        help="Optional Python list of config paths. If omitted, inferred from checkpoint directory.",
    )
    parser.add_argument(
        "--model_types",
        type=str,
        default=None,
        help="Optional Python list with entries in {'tsflow','lcde','auto'}. Defaults to auto-detect.",
    )
    parser.add_argument(
        "--labels",
        type=str,
        default=None,
        help="Optional Python list of labels for reporting.",
    )
    parser.add_argument(
        "--eval_datasets",
        type=str,
        required=False,
        help="Python list of GluonTS evaluation datasets, e.g. \"['ett_small_15min','ett_small_1h']\".",
    )
    parser.add_argument(
        "--eval_dataset_configs",
        type=str,
        default=None,
        help="Optional Python list of config paths whose dataset_params define the evaluation datasets.",
    )
    parser.add_argument(
        "--eval_freqs",
        type=str,
        default=None,
        help="Optional Python list of eval frequencies. Length 1 is broadcast across eval datasets.",
    )
    parser.add_argument(
        "--eval_prediction_lengths",
        type=str,
        default=None,
        help="Optional int or Python list of eval prediction lengths. Length 1 is broadcast across eval datasets.",
    )
    parser.add_argument("--batch_size", type=int, default=None, help="Optional evaluation batch size override.")
    parser.add_argument("--num_samples", type=int, default=None, help="Optional sample count override.")
    parser.add_argument("--seed", type=int, default=None, help="Optional RNG seed for deterministic evaluation.")
    parser.add_argument("--plot_dir", type=str, default=None, help="Optional directory for first-series forecast plots.")
    parser.add_argument(
        "--save_comparison_plots_dir",
        type=str,
        default=None,
        help="Optional directory for TSFlow vs G-SLiCE comparison plots when both models are evaluated together.",
    )
    parser.add_argument(
        "--overlay_comparison_models_symlog",
        action="store_true",
        help="Use aligned symmetric symlog y-axes for the TSFlow/G-SLiCE comparison plots.",
    )
    parser.add_argument(
        "--skip_individual_summary_plots",
        action="store_true",
        help="Skip per-checkpoint summary plot generation.",
    )
    parser.add_argument(
        "--no_adapt_gp_to_eval_dataset",
        action="store_true",
        help="Disable the default eval-time GP adaptation to the evaluation dataset frequency.",
    )
    parser.add_argument(
        "--fine_to_coarse_eval_adapter_override",
        type=str,
        default=None,
        help="Optional override for dataset_params.fine_to_coarse_eval_adapter, e.g. repeat or gp_resample.",
    )
    parser.add_argument(
        "--only_coarser_eval_inputs",
        action="store_true",
        help="Only evaluate each checkpoint on eval datasets whose frequency is strictly coarser than the train frequency.",
    )
    parser.add_argument(
        "--only_not_finer_eval_inputs",
        action="store_true",
        help="Only evaluate each checkpoint on eval datasets whose frequency is equal to or coarser than the train frequency.",
    )
    parser.add_argument("--max_eval_instances", type=int, default=None, help="Optional cap on evaluated test items.")
    parser.add_argument("--device", type=str, default="cuda", help="Device, e.g. cuda, cuda:0, cpu.")
    parser.add_argument("--regenerate", action="store_true", help="Regenerate eval datasets before loading them.")
    parser.add_argument("--save_json", type=str, default=None, help="Optional path to save results as JSON.")
    parser.add_argument("--save_csv", type=str, default=None, help="Optional path to save results as CSV.")
    parser.add_argument(
        "--list_frequency_variants",
        action="store_true",
        help="Print the built-in GluonTS frequency-variant dataset map and exit.",
    )
    args = parser.parse_args()

    if args.list_frequency_variants:
        print(json.dumps(get_dataset_frequency_variants(), indent=2, sort_keys=True))
        return

    if args.checkpoint_paths is None or (args.eval_datasets is None and args.eval_dataset_configs is None):
        raise ValueError(
            "--checkpoint_paths and one of --eval_datasets/--eval_dataset_configs are required "
            "unless --list_frequency_variants is used."
        )

    checkpoint_paths = _parse_list_arg(args.checkpoint_paths, "checkpoint_paths")
    config_paths = _parse_list_arg(args.config_paths, "config_paths") if args.config_paths is not None else None
    model_types = _parse_list_arg(args.model_types, "model_types") if args.model_types is not None else None
    labels = _parse_list_arg(args.labels, "labels") if args.labels is not None else None
    eval_datasets = _parse_list_arg(args.eval_datasets, "eval_datasets") if args.eval_datasets is not None else None
    eval_dataset_configs = (
        _parse_list_arg(args.eval_dataset_configs, "eval_dataset_configs")
        if args.eval_dataset_configs is not None
        else None
    )
    eval_freqs = _parse_list_arg(args.eval_freqs, "eval_freqs") if args.eval_freqs is not None else None
    eval_prediction_lengths = _parse_optional_int_list_arg(args.eval_prediction_lengths, "eval_prediction_lengths")

    n_eval_specs = len(eval_dataset_configs) if eval_dataset_configs is not None else len(eval_datasets or [])
    if eval_datasets is None:
        eval_datasets = [None] * n_eval_specs
    else:
        eval_datasets = _normalize_optional_list(eval_datasets, n_eval_specs, "eval_datasets")
    eval_dataset_configs = _normalize_optional_list(eval_dataset_configs, n_eval_specs, "eval_dataset_configs")
    eval_freqs = _normalize_optional_list(eval_freqs, n_eval_specs, "eval_freqs")
    eval_prediction_lengths = _normalize_optional_list(
        eval_prediction_lengths,
        n_eval_specs,
        "eval_prediction_lengths",
    )

    specs = _build_specs(
        checkpoint_paths=checkpoint_paths,
        config_paths=config_paths,
        model_types=model_types,
        labels=labels,
    )

    device = _resolve_device(args.device)
    plot_dir = None
    if args.plot_dir is not None:
        plot_dir = Path(args.plot_dir)
    elif args.save_json is not None:
        save_json_path = Path(args.save_json)
        plot_dir = save_json_path.parent / f"{save_json_path.stem}_plots"
    results: List[CrossFrequencyEvalResult] = []
    comparison_plot_inputs: list[tuple[BenchmarkSpec, List[FirstSeriesPlotPayload]]] = []

    for spec in specs:
        config = _apply_dataset_param_overrides(
            _read_yaml(spec.config_path),
            fine_to_coarse_eval_adapter_override=args.fine_to_coarse_eval_adapter_override,
        )
        num_samples = _resolve_num_samples(spec, config, args.num_samples)
        eval_seed = args.seed
        if eval_seed is None:
            eval_params = config.get("evaluation_params", {})
            eval_seed = eval_params.get("test_eval_seed", eval_params.get("eval_seed", config.get("seed")))
        spec_results: List[CrossFrequencyEvalResult] = []
        spec_plot_payloads: List[FirstSeriesPlotPayload] = []

        for eval_dataset_name, eval_dataset_config, eval_freq, eval_prediction_length in zip(
            eval_datasets,
            eval_dataset_configs,
            eval_freqs,
            eval_prediction_lengths,
        ):
            eval_dataset_params = _load_eval_dataset_params(eval_dataset_name, eval_dataset_config)
            if args.only_coarser_eval_inputs or args.only_not_finer_eval_inputs:
                resolved_eval_freq, _ = _resolve_eval_dataset_metadata(
                    eval_dataset_params=eval_dataset_params,
                    eval_freq=eval_freq,
                    eval_prediction_length=eval_prediction_length,
                    regenerate=args.regenerate,
                )
                train_step_hours = float(get_relative_time_step(str(config["model_params"]["freq"])))
                eval_step_hours = float(get_relative_time_step(str(resolved_eval_freq)))
                should_skip = (
                    eval_step_hours <= train_step_hours
                    if args.only_coarser_eval_inputs
                    else eval_step_hours < train_step_hours
                )
                if should_skip:
                    print(
                        f"Skipping {spec.label} on {get_dataset_name_from_params(eval_dataset_params)} "
                        f"because eval_freq={resolved_eval_freq} is "
                        f"{'not coarser than' if args.only_coarser_eval_inputs else 'finer than'} "
                        f""
                        f"train_freq={config['model_params']['freq']}.",
                        flush=True,
                    )
                    continue

            model = _build_model(spec, config, device)
            resolved_eval_dataset_name = get_dataset_name_from_params(eval_dataset_params)
            result, plot_payload = _evaluate_checkpoint_on_dataset(
                spec=spec,
                config=config,
                model=model,
                eval_dataset_name=resolved_eval_dataset_name,
                eval_dataset_params=eval_dataset_params,
                eval_freq=eval_freq,
                eval_prediction_length=eval_prediction_length,
                num_samples=num_samples,
                eval_seed=eval_seed,
                plot_dir=plot_dir,
                max_eval_instances=args.max_eval_instances,
                batch_size_override=args.batch_size,
                regenerate=args.regenerate,
                adapt_gp_to_eval_dataset=not args.no_adapt_gp_to_eval_dataset,
            )
            spec_results.append(result)
            if plot_payload is not None:
                spec_plot_payloads.append(plot_payload)
            del model

        plot_path = None
        if not args.skip_individual_summary_plots:
            plot_path = _save_checkpoint_summary_plot(
                plot_dir=plot_dir,
                spec=spec,
                plot_payloads=spec_plot_payloads,
            )
        comparison_plot_inputs.append((spec, list(spec_plot_payloads)))
        for result in spec_results:
            result.plot_path = plot_path
            results.append(result)
            print(json.dumps(asdict(result), indent=2, sort_keys=True))

    if args.save_comparison_plots_dir is not None:
        comparison_dir = Path(args.save_comparison_plots_dir)
        saved_paths = _save_comparison_plots(
            comparison_dir=comparison_dir,
            spec_plot_payloads=comparison_plot_inputs,
            overlay_models_symlog=bool(args.overlay_comparison_models_symlog),
        )
        for saved_path in saved_paths:
            print(f"Saved comparison plot to: {saved_path}")

    if args.save_json is not None:
        save_json_path = Path(args.save_json)
        save_json_path.parent.mkdir(parents=True, exist_ok=True)
        save_json_path.write_text(json.dumps([asdict(result) for result in results], indent=2))
        print(f"Saved JSON results to: {save_json_path}")

    if args.save_csv is not None:
        save_csv_path = Path(args.save_csv)
        save_csv_path.parent.mkdir(parents=True, exist_ok=True)
        rows = [asdict(result) for result in results]
        if rows:
            with save_csv_path.open("w", newline="") as fp:
                writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
        print(f"Saved CSV results to: {save_csv_path}")


if __name__ == "__main__":
    main()
