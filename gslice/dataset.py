import os
import tarfile
from pathlib import Path
from typing import Any, Dict, List
from urllib import request

import numpy as np
import pandas as pd
from gluonts.dataset.common import MetaData, TrainDatasets, load_datasets
from gluonts.dataset.repository.datasets import get_dataset, get_download_path

from gslice.irregular import build_irregular_dataset_name, has_irregular_grid

default_dataset_path: Path = get_download_path() / "datasets"
wiki2k_download_link: str = (
    "https://github.com/awslabs/gluonts/raw/b89f203595183340651411a41eeb0ee60570a4d9/datasets/wiki2000_nips.tar.gz"  # noqa: E501
)

# Datasets in or adjacent to the current experiment suite that come in multiple
# sampling rates in the GluonTS repository.
DATASET_FREQUENCY_VARIANTS: Dict[str, List[str]] = {
    "electricity_nips": ["electricity_weekly"],
    "m4_hourly": ["m4_daily", "m4_weekly", "m4_monthly", "m4_quarterly", "m4_yearly"],
    "solar_nips": ["solar_10_minutes", "solar_weekly"],
    "uber_tlc_hourly": ["uber_tlc_daily"],
    # Useful 15min <-> 1h pair for explicit cross-frequency experiments.
    "ett_small_15min": ["ett_small_1h"],
    "ett_small_1h": ["ett_small_15min"],
}


def get_dataset_frequency_variants() -> dict[str, list[str]]:
    return {name: list(variants) for name, variants in DATASET_FREQUENCY_VARIANTS.items()}


def _dataset_artifact_paths(dataset_root: Path) -> tuple[Path, Path, Path] | None:
    metadata_dir = dataset_root / "metadata"
    metadata_root = metadata_dir if metadata_dir.exists() else dataset_root
    if not metadata_dir.exists() and not (dataset_root / "metadata.json").exists():
        return None

    train_path = dataset_root / "train"
    test_path = dataset_root / "test"

    if train_path.exists() and test_path.exists():
        return metadata_root, train_path, test_path
    return None


def _find_local_dataset_path(dataset_name: str) -> Path | None:
    candidates: list[Path] = []

    for root in {
        default_dataset_path,
        Path.home() / ".gluonts" / "datasets",
        Path(os.environ.get("HOME", "")) / ".gluonts" / "datasets",
    }:
        if str(root):
            candidates.append(root / dataset_name)

    home_root = Path("/home")
    if home_root.exists():
        candidates.extend(home_root.glob(f"*/.gluonts/datasets/{dataset_name}"))

    seen: set[Path] = set()
    for candidate in candidates:
        try:
            candidate = candidate.expanduser().resolve()
        except FileNotFoundError:
            candidate = candidate.expanduser()
        if candidate in seen:
            continue
        seen.add(candidate)
        if _dataset_artifact_paths(candidate) is not None:
            return candidate
    return None


def infer_target_dim(dataset, max_target_dim: int = 2000) -> int:
    feat_static_cat = getattr(dataset.metadata, "feat_static_cat", None)
    if feat_static_cat:
        cardinality = getattr(feat_static_cat[0], "cardinality", None)
        if cardinality is not None:
            return min(max_target_dim, int(cardinality))

    train_data = dataset.train
    try:
        train_length = len(train_data)
    except TypeError:
        train_length = None

    if train_length and train_length > 1:
        return min(max_target_dim, int(train_length))

    try:
        first_entry = next(iter(train_data))
    except StopIteration:
        return 1

    target = first_entry.get("target")
    shape = getattr(target, "shape", None)
    if shape and len(shape) > 1:
        return min(max_target_dim, int(shape[0]))

    if train_length is not None and train_length > 0:
        return min(max_target_dim, int(train_length))

    return 1


def _metadata_kwargs(metadata: MetaData) -> dict[str, Any]:
    kwargs = metadata.dict()
    kwargs["feat_static_cat"] = list(kwargs.get("feat_static_cat", []) or [])
    kwargs["feat_static_real"] = list(kwargs.get("feat_static_real", []) or [])
    kwargs["feat_dynamic_real"] = list(kwargs.get("feat_dynamic_real", []) or [])
    kwargs["feat_dynamic_cat"] = list(kwargs.get("feat_dynamic_cat", []) or [])
    return kwargs


def _dataset_start_timestamp(start: Any, freq: str) -> pd.Timestamp:
    if isinstance(start, pd.Period):
        return start.to_timestamp(how="start")
    return pd.Period(start, freq=freq).to_timestamp(how="start")


def _subsample_mask(
    index: pd.DatetimeIndex,
    *,
    base_freq: str,
    target_freq: str,
    offset_steps: int,
    selector: dict[str, Any],
) -> np.ndarray:
    anchor_mask = np.ones(len(index), dtype=bool)

    minute = selector.get("minute", None)
    if minute is not None:
        anchor_mask &= index.minute == int(minute)

    hour = selector.get("hour", None)
    if hour is not None:
        anchor_mask &= index.hour == int(hour)

    day_of_week = selector.get("day_of_week", None)
    if day_of_week is not None:
        anchor_mask &= index.dayofweek == int(day_of_week)

    day_of_month = selector.get("day_of_month", None)
    if day_of_month is not None:
        anchor_mask &= index.day == int(day_of_month)

    month_of_year = selector.get("month_of_year", None)
    if month_of_year is not None:
        anchor_mask &= index.month == int(month_of_year)

    if len(index) == 0:
        return anchor_mask

    if selector:
        anchor_positions = np.flatnonzero(anchor_mask)
        if len(anchor_positions) == 0:
            return anchor_mask
        if offset_steps >= len(anchor_positions):
            raise ValueError(
                f"offset_steps={offset_steps} is too large for the matched anchor set of size "
                f"{len(anchor_positions)}."
            )
        anchor_position = int(anchor_positions[offset_steps])
    else:
        if offset_steps >= len(index):
            raise ValueError(
                f"offset_steps={offset_steps} is too large for a series of length {len(index)}."
            )
        anchor_position = int(offset_steps)

    from gslice.utils.variables import get_relative_time_step

    base_step_hours = float(get_relative_time_step(str(base_freq)))
    target_step_hours = float(get_relative_time_step(str(target_freq)))
    if target_step_hours + 1e-12 < base_step_hours:
        raise ValueError(
            f"Subsampling target_freq={target_freq} must not be finer than base_freq={base_freq}."
        )
    step_ratio = target_step_hours / base_step_hours
    rounded_ratio = int(round(step_ratio))
    if not np.isclose(step_ratio, rounded_ratio):
        raise ValueError(
            f"Subsampling target_freq={target_freq} must be an integer multiple of base_freq={base_freq}; "
            f"got ratio={step_ratio}."
        )

    anchor_timestamp = index[anchor_position]
    selected_timestamps = pd.date_range(start=anchor_timestamp, end=index[-1], freq=target_freq)
    return index.isin(selected_timestamps)


def _subsample_series(
    *,
    entry: dict[str, Any],
    base_freq: str,
    target_freq: str,
    selector: dict[str, Any],
    offset_steps: int,
) -> tuple[np.ndarray, pd.Period]:
    target = np.asarray(entry["target"])
    if target.ndim > 1:
        raise NotImplementedError("Subsampled dataset views currently only support 1D targets.")

    index = pd.date_range(
        start=_dataset_start_timestamp(entry["start"], base_freq),
        periods=len(target),
        freq=base_freq,
    )
    mask = _subsample_mask(
        index,
        base_freq=base_freq,
        target_freq=target_freq,
        offset_steps=offset_steps,
        selector=selector,
    )
    selected_positions = np.flatnonzero(mask)
    if len(selected_positions) == 0:
        raise ValueError("Subsample specification selected no timestamps from the base series.")

    selected_target = target[selected_positions]
    selected_start = pd.Period(index[selected_positions[0]], freq=target_freq)
    return selected_target, selected_start


def _resolve_subsample_prediction_length(
    *,
    dataset_params: dict[str, Any],
    target_freq: str,
) -> int:
    physical_prediction_hours = dataset_params.get("physical_prediction_hours", None)
    if physical_prediction_hours is not None:
        from gslice.utils.variables import get_relative_time_step

        target_step_hours = float(get_relative_time_step(str(target_freq)))
        prediction_length = float(physical_prediction_hours) / target_step_hours
        rounded = int(round(prediction_length))
        if not np.isclose(prediction_length, rounded):
            raise ValueError(
                f"physical_prediction_hours={physical_prediction_hours} is not divisible by "
                f"target_freq={target_freq} (step_hours={target_step_hours})."
            )
        if rounded <= 0:
            raise ValueError(f"Derived prediction_length must be > 0, got {rounded}.")
        return rounded

    prediction_length = dataset_params.get("prediction_length", None)
    if prediction_length is None:
        raise ValueError(
            "Subsampled datasets require either dataset_params.physical_prediction_hours "
            "or dataset_params.prediction_length."
        )
    return int(prediction_length)


def _subsample_dataset_name(dataset_params: dict[str, Any]) -> str:
    subsample_params = dict(dataset_params.get("subsample_params", {}) or {})
    base_dataset = str(dataset_params["base_dataset"])
    target_freq = str(subsample_params["target_freq"])
    selector_parts = []
    for key in ["minute", "hour", "day_of_week", "day_of_month", "month_of_year"]:
        if key in subsample_params:
            selector_parts.append(f"{key}-{subsample_params[key]}")
    if "offset_steps" in subsample_params:
        selector_parts.append(f"offset-{subsample_params['offset_steps']}")
    suffix = "__".join([target_freq.replace("/", "_")] + selector_parts) if selector_parts else target_freq.replace("/", "_")
    return f"{base_dataset}__subsample__{suffix}"


def get_dataset_name_from_params(dataset_params: dict[str, Any]) -> str:
    if has_irregular_grid(dataset_params):
        return build_irregular_dataset_name(dataset_params)
    if dataset_params.get("subsample_params", None):
        return _subsample_dataset_name(dataset_params)
    if "dataset" in dataset_params:
        return str(dataset_params["dataset"])
    if "base_dataset" in dataset_params:
        return str(dataset_params["base_dataset"])
    raise KeyError("dataset_params must contain either dataset or base_dataset.")


def get_dataset_family_from_params(dataset_params: dict[str, Any]) -> str:
    if has_irregular_grid(dataset_params):
        return str(dataset_params.get("base_dataset", dataset_params.get("dataset")))
    if dataset_params.get("subsample_params", None):
        return str(dataset_params["base_dataset"])
    if "dataset" in dataset_params:
        return str(dataset_params["dataset"])
    if "base_dataset" in dataset_params:
        return str(dataset_params["base_dataset"])
    raise KeyError("dataset_params must contain either dataset or base_dataset.")


def _build_subsampled_dataset(
    base_dataset: TrainDatasets,
    *,
    dataset_params: dict[str, Any],
) -> TrainDatasets:
    subsample_params = dict(dataset_params.get("subsample_params", {}) or {})
    target_freq = str(subsample_params["target_freq"])
    selector = {
        key: subsample_params[key]
        for key in ["minute", "hour", "day_of_week", "day_of_month", "month_of_year"]
        if key in subsample_params
    }
    offset_steps = int(subsample_params.get("offset_steps", 0))
    prediction_length = _resolve_subsample_prediction_length(
        dataset_params=dataset_params,
        target_freq=target_freq,
    )

    train_entries = list(base_dataset.train)
    test_entries = list(base_dataset.test)
    base_freq = str(base_dataset.metadata.freq)
    subsampled_train: list[dict[str, Any]] = []
    subsampled_test: list[dict[str, Any]] = []

    for train_entry in train_entries:
        selected_target, selected_start = _subsample_series(
            entry=train_entry,
            base_freq=base_freq,
            target_freq=target_freq,
            selector=selector,
            offset_steps=offset_steps,
        )
        if len(selected_target) <= prediction_length:
            raise ValueError(
                f"Subsampled train series for item_id={train_entry.get('item_id')} has length {len(selected_target)} "
                f"which is not greater than prediction_length={prediction_length}."
            )
        shared_fields = {
            key: value
            for key, value in train_entry.items()
            if key not in {"target", "start"}
        }
        subsampled_train.append(
            {
                "start": selected_start,
                "target": selected_target,
                **shared_fields,
            }
        )

    for test_entry in test_entries:
        selected_target, selected_start = _subsample_series(
            entry=test_entry,
            base_freq=base_freq,
            target_freq=target_freq,
            selector=selector,
            offset_steps=offset_steps,
        )
        if len(selected_target) <= prediction_length:
            raise ValueError(
                f"Subsampled test series for item_id={test_entry.get('item_id')} has length {len(selected_target)} "
                f"which is not greater than prediction_length={prediction_length}."
            )
        shared_fields = {
            key: value
            for key, value in test_entry.items()
            if key not in {"target", "start"}
        }
        subsampled_test.append(
            {
                "start": selected_start,
                "target": selected_target,
                **shared_fields,
            }
        )

    metadata_kwargs = _metadata_kwargs(base_dataset.metadata)
    metadata_kwargs["freq"] = target_freq
    metadata_kwargs["prediction_length"] = int(prediction_length)
    metadata = MetaData(**metadata_kwargs)
    return TrainDatasets(metadata=metadata, train=subsampled_train, test=subsampled_test)


def get_gts_dataset_from_params(
    dataset_params: dict[str, Any],
    *,
    regenerate: bool = False,
):
    if has_irregular_grid(dataset_params):
        base_dataset_name = str(dataset_params.get("base_dataset", dataset_params.get("dataset")))
        if not base_dataset_name:
            raise ValueError("Irregular-grid datasets require dataset_params.base_dataset or dataset.")
        return get_gts_dataset(
            base_dataset_name,
            regenerate=bool(dataset_params.get("base_regenerate", regenerate)),
            prediction_length=dataset_params.get("prediction_length", None),
        )

    if dataset_params.get("subsample_params", None):
        base_dataset_name = str(dataset_params["base_dataset"])
        base_dataset = get_gts_dataset(
            base_dataset_name,
            regenerate=bool(dataset_params.get("base_regenerate", regenerate)),
        )
        return _build_subsampled_dataset(base_dataset, dataset_params=dataset_params)

    return get_gts_dataset(
        str(dataset_params["dataset"]),
        regenerate=regenerate,
        prediction_length=dataset_params.get("prediction_length", None),
    )


def get_gts_dataset(
    dataset_name,
    *,
    regenerate: bool = False,
    prediction_length: int | None = None,
):
    local_dataset_path = _find_local_dataset_path(dataset_name) or (default_dataset_path / dataset_name)

    if dataset_name == "wiki2000_nips":
        wiki_dataset_path = local_dataset_path
        Path(default_dataset_path).mkdir(parents=True, exist_ok=True)
        if not wiki_dataset_path.exists():
            tar_file_path = wiki_dataset_path.parent / f"{dataset_name}.tar.gz"
            request.urlretrieve(
                wiki2k_download_link,
                tar_file_path,
            )

            with tarfile.open(tar_file_path) as tar:
                tar.extractall(path=wiki_dataset_path.parent)

            os.remove(tar_file_path)
        dataset_paths = _dataset_artifact_paths(wiki_dataset_path)
        if dataset_paths is None:
            raise FileNotFoundError(f"Incomplete local dataset at {wiki_dataset_path}.")
        return load_datasets(
            metadata=dataset_paths[0],
            train=dataset_paths[1],
            test=dataset_paths[2],
        )
    dataset_paths = _dataset_artifact_paths(local_dataset_path)
    if not regenerate and dataset_paths is not None:
        return load_datasets(
            metadata=dataset_paths[0],
            train=dataset_paths[1],
            test=dataset_paths[2],
        )
    try:
        return get_dataset(
            dataset_name,
            regenerate=regenerate,
            prediction_length=prediction_length,
        )
    except (AssertionError, FileNotFoundError):
        if dataset_paths is not None:
            return load_datasets(
                metadata=dataset_paths[0],
                train=dataset_paths[1],
                test=dataset_paths[2],
            )
        raise
