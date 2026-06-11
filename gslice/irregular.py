import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np

from gslice.utils.variables import get_relative_time_step

TIME_GRID_FIELD = "time_grid"
PAST_TIME_GRID_FIELD = "past_time_grid"
FUTURE_TIME_GRID_FIELD = "future_time_grid"
LAG_FEATURE_FIELD = "lag_features"
DENSE_PAST_TARGET_FIELD = "dense_past_target"
DENSE_PAST_TIME_GRID_FIELD = "dense_past_time_grid"
DENSE_PAST_OBSERVED_FIELD = "dense_past_observed_values"


@dataclass(frozen=True)
class IrregularGridSpec:
    base_freq: str
    context_physical_hours: float
    prediction_physical_hours: float
    num_context_points: int
    num_prediction_points: int
    gamma_k: float
    include_region_endpoints: bool = True
    per_instance: bool = True
    seed_offset: int = 0
    eval_context_sampling: str = "irregular"
    eval_future_sampling: str = "irregular"

    @property
    def base_step_hours(self) -> float:
        return float(get_relative_time_step(self.base_freq))

    @property
    def dense_context_length(self) -> int:
        return _resolve_dense_length(self.context_physical_hours, self.base_step_hours)

    @property
    def dense_prediction_length(self) -> int:
        return _resolve_dense_length(self.prediction_physical_hours, self.base_step_hours)


def has_irregular_grid(dataset_params: dict[str, Any]) -> bool:
    return bool(dataset_params.get("irregular_grid_params"))


def get_irregular_grid_spec(dataset_params: dict[str, Any], *, fallback_freq: str | None = None) -> IrregularGridSpec:
    params = dict(dataset_params.get("irregular_grid_params", {}) or {})
    if not params:
        raise ValueError("dataset_params.irregular_grid_params is required for irregular-grid utilities.")

    base_freq = str(params.get("base_freq", fallback_freq or dataset_params.get("base_freq") or dataset_params.get("freq")))
    if not base_freq:
        raise ValueError("irregular_grid_params.base_freq is required.")

    return IrregularGridSpec(
        base_freq=base_freq,
        context_physical_hours=float(params["context_physical_hours"]),
        prediction_physical_hours=float(params["prediction_physical_hours"]),
        num_context_points=int(params["num_context_points"]),
        num_prediction_points=int(params["num_prediction_points"]),
        gamma_k=float(params["gamma_k"]),
        include_region_endpoints=bool(params.get("include_region_endpoints", True)),
        per_instance=bool(params.get("per_instance", True)),
        seed_offset=int(params.get("seed_offset", 0)),
        eval_context_sampling=str(params.get("eval_context_sampling", "irregular")).lower(),
        eval_future_sampling=str(params.get("eval_future_sampling", "irregular")).lower(),
    )


def resolve_model_window_lengths(dataset_params: dict[str, Any], *, fallback_freq: str | None = None) -> tuple[int, int]:
    if not has_irregular_grid(dataset_params):
        prediction_length = int(dataset_params.get("prediction_length")) if dataset_params.get("prediction_length") is not None else None
        if prediction_length is None:
            raise ValueError("prediction_length is required for regular datasets.")
        return prediction_length, prediction_length
    spec = get_irregular_grid_spec(dataset_params, fallback_freq=fallback_freq)
    return int(spec.num_context_points), int(spec.num_prediction_points)


def resolve_source_window_lengths(dataset_params: dict[str, Any], *, fallback_freq: str | None = None) -> tuple[int, int]:
    if not has_irregular_grid(dataset_params):
        prediction_length = int(dataset_params.get("prediction_length")) if dataset_params.get("prediction_length") is not None else None
        if prediction_length is None:
            raise ValueError("prediction_length is required for regular datasets.")
        return prediction_length, prediction_length
    spec = get_irregular_grid_spec(dataset_params, fallback_freq=fallback_freq)
    return int(spec.dense_context_length), int(spec.dense_prediction_length)


def build_irregular_dataset_name(dataset_params: dict[str, Any], *, fallback_freq: str | None = None) -> str:
    spec = get_irregular_grid_spec(dataset_params, fallback_freq=fallback_freq)
    base_dataset = str(dataset_params.get("base_dataset") or dataset_params.get("dataset"))
    if spec.eval_context_sampling == "regular" and spec.eval_future_sampling == "regular":
        return (
            f"{base_dataset}__regularized__"
            f"ctx{spec.num_context_points}_pred{spec.num_prediction_points}"
        )
    return (
        f"{base_dataset}__irregular__"
        f"ctx{spec.num_context_points}_pred{spec.num_prediction_points}_"
        f"k{_slug_number(spec.gamma_k)}"
    )


def deterministic_rng(*, item_id: Any, forecast_start: Any, seed_offset: int = 0, salt: str = "") -> np.random.RandomState:
    digest = hashlib.sha256(
        f"{item_id}|{forecast_start}|{seed_offset}|{salt}".encode("utf-8")
    ).hexdigest()
    seed = int(digest[:16], 16) % (2**32 - 1)
    return np.random.RandomState(seed)


def sample_irregular_indices(
    *,
    dense_length: int,
    num_points: int,
    gamma_k: float,
    rng: np.random.RandomState,
    include_endpoints: bool = True,
) -> np.ndarray:
    dense_length = int(dense_length)
    num_points = int(num_points)
    if dense_length <= 0:
        raise ValueError(f"dense_length must be > 0, got {dense_length}.")
    if num_points <= 0:
        raise ValueError(f"num_points must be > 0, got {num_points}.")
    if num_points > dense_length:
        raise ValueError(f"num_points={num_points} exceeds dense_length={dense_length}.")

    if dense_length == num_points:
        return np.arange(dense_length, dtype=np.int64)

    if include_endpoints:
        if num_points < 2:
            raise ValueError("num_points must be >= 2 when include_endpoints=True.")
        interior_count = num_points - 2
        if interior_count == 0:
            return np.array([0, dense_length - 1], dtype=np.int64)
        raw = rng.gamma(shape=float(gamma_k), scale=1.0, size=interior_count + 1)
        cumulative = np.cumsum(raw[:-1]) / np.sum(raw)
        candidate = np.rint(cumulative * float(dense_length - 1)).astype(np.int64)
        candidate = np.clip(candidate, 1, dense_length - 2)
        selected = _unique_monotone_fill(candidate, required=interior_count, lower=1, upper=dense_length - 2)
        return np.concatenate([np.array([0]), selected, np.array([dense_length - 1])]).astype(np.int64)

    raw = rng.gamma(shape=float(gamma_k), scale=1.0, size=num_points)
    cumulative = np.cumsum(raw) / np.sum(raw)
    candidate = np.rint((cumulative - cumulative[0]) / max(1e-12, (cumulative[-1] - cumulative[0])) * (dense_length - 1))
    candidate = np.clip(candidate.astype(np.int64), 0, dense_length - 1)
    return _unique_monotone_fill(candidate, required=num_points, lower=0, upper=dense_length - 1)


def select_regular_indices(
    *,
    dense_length: int,
    num_points: int,
    include_endpoints: bool = True,
) -> np.ndarray:
    dense_length = int(dense_length)
    num_points = int(num_points)
    if dense_length <= 0:
        raise ValueError(f"dense_length must be > 0, got {dense_length}.")
    if num_points <= 0:
        raise ValueError(f"num_points must be > 0, got {num_points}.")
    if num_points > dense_length:
        raise ValueError(f"num_points={num_points} exceeds dense_length={dense_length}.")

    if dense_length == num_points:
        return np.arange(dense_length, dtype=np.int64)

    if include_endpoints:
        if num_points < 2:
            raise ValueError("num_points must be >= 2 when include_endpoints=True.")
        candidate = np.rint(np.linspace(0, dense_length - 1, num_points)).astype(np.int64)
        candidate = np.clip(candidate, 0, dense_length - 1)
        return _unique_monotone_fill(candidate, required=num_points, lower=0, upper=dense_length - 1)

    offsets = np.linspace(0.0, 1.0, num_points + 2, dtype=np.float64)[1:-1]
    candidate = np.rint(offsets * float(dense_length - 1)).astype(np.int64)
    candidate = np.clip(candidate, 0, dense_length - 1)
    return _unique_monotone_fill(candidate, required=num_points, lower=0, upper=dense_length - 1)


def compute_irregular_lag_features(
    *,
    dense_past_target: np.ndarray,
    dense_past_time_grid: np.ndarray,
    query_time_grid: np.ndarray,
    lag_steps: list[int],
    step_hours: float,
) -> np.ndarray:
    dense_target = np.asarray(dense_past_target)
    dense_time = np.asarray(dense_past_time_grid, dtype=np.float32).reshape(-1)
    query_time = np.asarray(query_time_grid, dtype=np.float32).reshape(-1)
    if dense_target.ndim == 1:
        dense_target = dense_target[:, None]
    if dense_target.ndim != 2:
        raise NotImplementedError("Irregular lag features currently support only 1D/univariate targets.")

    lag_hours = np.asarray(lag_steps, dtype=np.float32) * float(step_hours)
    lag_targets = query_time[:, None] - lag_hours[None, :]

    indices = np.searchsorted(dense_time, lag_targets, side="left")
    if np.any(indices >= dense_time.shape[0]):
        raise ValueError("Dense past window is too short for requested irregular lag features.")

    matched_times = dense_time[indices]
    if not np.allclose(matched_times, lag_targets, atol=1e-5, rtol=1e-5):
        raise ValueError("Irregular lag targets do not align with the dense source time grid.")

    gathered = dense_target[indices]
    if gathered.shape[-1] == 1:
        return gathered[..., 0]
    return np.transpose(gathered, (0, 2, 1))


def maybe_get_irregular_input_names(dataset_params: dict[str, Any], *, use_lags: bool = True) -> list[str]:
    base = ["past_target", "past_observed_values", "mean"]
    if has_irregular_grid(dataset_params):
        names = base + [PAST_TIME_GRID_FIELD, FUTURE_TIME_GRID_FIELD]
        if use_lags:
            names.append(LAG_FEATURE_FIELD)
        return names
    return base


def _resolve_dense_length(physical_hours: float, step_hours: float) -> int:
    raw = float(physical_hours) / float(step_hours)
    rounded = int(round(raw))
    if not np.isclose(raw, rounded):
        raise ValueError(
            f"physical_hours={physical_hours} is not divisible by step_hours={step_hours}."
        )
    if rounded <= 0:
        raise ValueError(f"Dense window length must be > 0, got {rounded}.")
    return rounded


def _unique_monotone_fill(values: np.ndarray, *, required: int, lower: int, upper: int) -> np.ndarray:
    if required <= 0:
        return np.empty((0,), dtype=np.int64)

    seen: set[int] = set()
    selected: list[int] = []
    for value in np.asarray(values, dtype=np.int64).tolist():
        if lower <= value <= upper and value not in seen:
            selected.append(value)
            seen.add(value)
        if len(selected) == required:
            break

    if len(selected) == required:
        return np.asarray(sorted(selected), dtype=np.int64)

    center = (lower + upper) / 2.0
    candidates = sorted(
        [value for value in range(lower, upper + 1) if value not in seen],
        key=lambda value: (abs(value - center), value),
    )
    for value in candidates:
        selected.append(value)
        if len(selected) == required:
            return np.asarray(sorted(selected), dtype=np.int64)

    raise ValueError("Could not construct a strictly increasing irregular index set.")


def _slug_number(value: float) -> str:
    text = f"{float(value):g}"
    return text.replace("-", "m").replace(".", "p")
