import math
from typing import Type

import numpy as np
from gluonts.core.component import validated
from gluonts.dataset import DataEntry
from gluonts.dataset.field_names import FieldName
from gluonts.transform import (
    AddObservedValuesIndicator,
    AddTimeFeatures,
    AsNumpyArray,
    Chain,
    MapTransformation,
)

from gslice.irregular import (
    DENSE_PAST_OBSERVED_FIELD,
    DENSE_PAST_TARGET_FIELD,
    DENSE_PAST_TIME_GRID_FIELD,
    FUTURE_TIME_GRID_FIELD,
    LAG_FEATURE_FIELD,
    PAST_TIME_GRID_FIELD,
    TIME_GRID_FIELD,
    IrregularGridSpec,
    compute_irregular_lag_features,
    deterministic_rng,
    sample_irregular_indices,
    select_regular_indices,
)

def _lookup_cached_stat(train_means: dict, item_id, train_length: int):
    if item_id in train_means:
        return train_means[item_id]

    candidates = []
    if isinstance(item_id, str):
        candidates.append(item_id)

    try:
        numeric_id = int(item_id)
    except (TypeError, ValueError):
        numeric_id = None

    if numeric_id is not None:
        candidates.extend(
            [
                numeric_id,
                str(numeric_id),
                numeric_id % int(train_length),
                str(numeric_id % int(train_length)),
            ]
        )

    for candidate in candidates:
        if candidate in train_means:
            return train_means[candidate]
    return None


def create_transforms(
    time_features,
    prediction_length,
    freq,
    train_length,
    time_step_hours: float | None = None,
    include_time_grid: bool = False,
):
    transforms = [
        AsNumpyArray(
            field=FieldName.TARGET,
            expected_ndim=1,
        ),
        AddTimeFeatures(
            start_field=FieldName.START,
            target_field=FieldName.TARGET,
            output_field=FieldName.FEAT_TIME,
            time_features=time_features,
            pred_length=prediction_length,
        ),
    ]
    if include_time_grid:
        transforms.append(
            AddTimeGrid(
                target_field=FieldName.TARGET,
                output_field=TIME_GRID_FIELD,
                step_hours=float(time_step_hours if time_step_hours is not None else 1.0),
            )
        )
    transforms.extend(
        [
            AddMeanFeature(
                target_field=FieldName.TARGET,
                output_field="mean",
                train_length=train_length,
            ),
            AddMeanAndStdFeature(
                target_field=FieldName.TARGET,
                output_field="stats",
            ),
            AddObservedValuesIndicator(
                target_field=FieldName.TARGET,
                output_field=FieldName.OBSERVED_VALUES,
            ),
        ]
    )
    return Chain(transforms)


class AddTimeGrid(MapTransformation):
    @validated()
    def __init__(self, target_field: str, output_field: str, step_hours: float, dtype: Type = np.float32) -> None:
        self.target_field = target_field
        self.output_field = output_field
        self.step_hours = float(step_hours)
        self.dtype = dtype

    def map_transform(self, data: DataEntry, is_train: bool) -> DataEntry:
        del is_train
        target = np.asarray(data[self.target_field])
        length = int(target.shape[-1])
        data[self.output_field] = (np.arange(length, dtype=self.dtype) * self.step_hours).astype(self.dtype)
        return data


class IrregularInstanceTransform(MapTransformation):
    @validated()
    def __init__(
        self,
        irregular_spec: IrregularGridSpec,
        lag_steps: list[int] | None = None,
        item_id_field: str = FieldName.ITEM_ID,
        forecast_start_field: str = FieldName.FORECAST_START,
    ) -> None:
        self.irregular_spec = irregular_spec
        self.lag_steps = [int(lag) for lag in (lag_steps or [])]
        self.item_id_field = item_id_field
        self.forecast_start_field = forecast_start_field

    def _future_indices(
        self,
        *,
        dense_prediction_len: int,
        rng: np.random.RandomState,
        is_train: bool,
    ) -> np.ndarray:
        num_prediction_points = int(self.irregular_spec.num_prediction_points)
        include_endpoints = bool(self.irregular_spec.include_region_endpoints)
        sampling_mode = "irregular" if is_train else str(self.irregular_spec.eval_future_sampling).lower()
        future_dense_length = int(dense_prediction_len)

        if include_endpoints:
            if future_dense_length < 2:
                raise ValueError(
                    "dense_prediction_len must be >= 2 when include_region_endpoints=True "
                    "so the first future point can be excluded."
                )
            requested_points = num_prediction_points + 1
            if requested_points > future_dense_length:
                raise ValueError(
                    "num_prediction_points exceeds the available future points once the "
                    "first future timestamp is excluded."
                )
        else:
            requested_points = num_prediction_points

        if sampling_mode == "irregular":
            future_indices = sample_irregular_indices(
                dense_length=future_dense_length,
                num_points=requested_points,
                gamma_k=float(self.irregular_spec.gamma_k),
                rng=rng,
                include_endpoints=include_endpoints,
            )
        elif sampling_mode == "regular":
            future_indices = select_regular_indices(
                dense_length=future_dense_length,
                num_points=requested_points,
                include_endpoints=include_endpoints,
            )
        else:
            raise ValueError(
                "IrregularInstanceTransform requires eval_future_sampling to be "
                f"'irregular' or 'regular', got {self.irregular_spec.eval_future_sampling!r}."
            )

        if include_endpoints:
            return future_indices[1:]
        return future_indices

    def _context_indices(
        self,
        *,
        dense_context_len: int,
        rng: np.random.RandomState,
        is_train: bool,
    ) -> np.ndarray:
        sampling_mode = "irregular" if is_train else str(self.irregular_spec.eval_context_sampling).lower()
        if sampling_mode == "irregular":
            return sample_irregular_indices(
                dense_length=dense_context_len,
                num_points=int(self.irregular_spec.num_context_points),
                gamma_k=float(self.irregular_spec.gamma_k),
                rng=rng,
                include_endpoints=bool(self.irregular_spec.include_region_endpoints),
            )
        if sampling_mode == "regular":
            return select_regular_indices(
                dense_length=dense_context_len,
                num_points=int(self.irregular_spec.num_context_points),
                include_endpoints=bool(self.irregular_spec.include_region_endpoints),
            )
        raise ValueError(
            "IrregularInstanceTransform requires eval_context_sampling to be "
            f"'irregular' or 'regular', got {self.irregular_spec.eval_context_sampling!r}."
        )

    def map_transform(self, data: DataEntry, is_train: bool) -> DataEntry:
        if PAST_TIME_GRID_FIELD not in data or FUTURE_TIME_GRID_FIELD not in data:
            raise KeyError("IrregularInstanceTransform requires past_time_grid and future_time_grid.")

        dense_past_target = np.asarray(data["past_target"])
        dense_future_target = np.asarray(data["future_target"])
        dense_past_time = np.asarray(data[PAST_TIME_GRID_FIELD], dtype=np.float32)
        dense_future_time = np.asarray(data[FUTURE_TIME_GRID_FIELD], dtype=np.float32)

        if dense_past_target.ndim > 1 or dense_future_target.ndim > 1:
            raise NotImplementedError("IrregularInstanceTransform currently supports only univariate targets.")

        # Rebase split-window timestamps so every irregular example uses
        # window-local coordinates, independent of where the window came from
        # in the original series.
        window_start = float(dense_past_time[0]) if dense_past_time.shape[0] > 0 else 0.0
        dense_past_time = dense_past_time - window_start
        if dense_future_time.shape[0] > 0:
            dense_future_time = dense_future_time - window_start

        dense_context_len = int(self.irregular_spec.dense_context_length)
        dense_prediction_len = int(self.irregular_spec.dense_prediction_length)
        if dense_past_target.shape[0] < dense_context_len:
            raise ValueError("Dense past window is shorter than the irregular context window.")
        has_dense_future_targets = dense_future_target.shape[0] > 0
        if dense_future_target.shape[0] == 0:
            dense_future_target = np.zeros((dense_prediction_len,), dtype=dense_past_target.dtype)
        if dense_future_time.shape[0] == 0:
            start_time = float(dense_past_time[-1]) if dense_past_time.shape[0] > 0 else 0.0
            step = float(self.irregular_spec.base_step_hours)
            dense_future_time = start_time + (np.arange(dense_prediction_len, dtype=np.float32) + 1.0) * step
        if dense_future_target.shape[0] != dense_prediction_len:
            raise ValueError(
                f"Expected dense future length {dense_prediction_len}, got {dense_future_target.shape[0]}."
            )
        if dense_future_time.shape[0] != dense_prediction_len:
            raise ValueError(
                f"Expected dense future time-grid length {dense_prediction_len}, got {dense_future_time.shape[0]}."
            )

        item_id = data.get(self.item_id_field, data.get("item_id", "unknown"))
        forecast_start = data.get(self.forecast_start_field, "unknown")
        rng_context = deterministic_rng(
            item_id=item_id,
            forecast_start=forecast_start,
            seed_offset=self.irregular_spec.seed_offset,
            salt="context",
        )
        rng_future = deterministic_rng(
            item_id=item_id,
            forecast_start=forecast_start,
            seed_offset=self.irregular_spec.seed_offset,
            salt="future",
        )

        context_indices = self._context_indices(
            dense_context_len=dense_context_len,
            rng=rng_context,
            is_train=is_train,
        )
        future_indices = self._future_indices(
            dense_prediction_len=dense_prediction_len,
            rng=rng_future,
            is_train=is_train,
        )

        long_past_len = dense_past_target.shape[0] - dense_context_len
        dense_context_target = dense_past_target[long_past_len:]
        dense_context_time = dense_past_time[long_past_len:]

        dense_past_observed = np.asarray(data.get("past_observed_values", np.ones_like(dense_past_target, dtype=np.bool_)))
        dense_context_observed = dense_past_observed[long_past_len:]

        selected_context_target = dense_context_target[context_indices]
        selected_context_time = dense_context_time[context_indices]
        selected_context_observed = dense_context_observed[context_indices]

        selected_future_target = dense_future_target[future_indices]
        selected_future_time = dense_future_time[future_indices]

        data[DENSE_PAST_TARGET_FIELD] = dense_past_target
        data[DENSE_PAST_TIME_GRID_FIELD] = dense_past_time
        data[DENSE_PAST_OBSERVED_FIELD] = dense_past_observed

        data["past_target"] = selected_context_target
        data["future_target"] = selected_future_target
        data[PAST_TIME_GRID_FIELD] = selected_context_time.astype(np.float32)
        data[FUTURE_TIME_GRID_FIELD] = selected_future_time.astype(np.float32)
        data["past_observed_values"] = selected_context_observed
        if "future_observed_values" in data:
            dense_future_observed = np.asarray(data["future_observed_values"])
            if dense_future_observed.shape[0] == 0:
                dense_future_observed = np.ones((dense_prediction_len,), dtype=dense_past_observed.dtype)
            data["future_observed_values"] = dense_future_observed[future_indices]
        elif has_dense_future_targets:
            data["future_observed_values"] = np.ones((len(future_indices),), dtype=dense_past_observed.dtype)
        if "past_is_pad" in data:
            dense_past_is_pad = np.asarray(data["past_is_pad"])
            data["past_is_pad"] = dense_past_is_pad[long_past_len:][context_indices]
        if "past_feat_time" in data:
            dense_past_feat_time = np.asarray(data["past_feat_time"])
            data["past_feat_time"] = dense_past_feat_time[long_past_len:][context_indices]
        if "future_feat_time" in data:
            data["future_feat_time"] = np.asarray(data["future_feat_time"])[future_indices]

        if self.lag_steps:
            query_time = np.concatenate([selected_context_time, selected_future_time], axis=0)
            data[LAG_FEATURE_FIELD] = compute_irregular_lag_features(
                dense_past_target=dense_past_target,
                dense_past_time_grid=dense_past_time,
                query_time_grid=query_time,
                lag_steps=self.lag_steps,
                step_hours=float(self.irregular_spec.base_step_hours),
            ).astype(np.float32)

        return data


class AddMeanAndStdFeature(MapTransformation):
    @validated()
    def __init__(
        self,
        target_field: str,
        output_field: str,
        dtype: Type = np.float32,
    ) -> None:
        self.target_field = target_field
        self.feature_name = output_field
        self.dtype = dtype

    def map_transform(self, data: DataEntry, is_train: bool) -> DataEntry:
        target = np.asarray(data[self.target_field], dtype=np.float64)
        data[self.feature_name] = np.array([np.nanmean(target), np.nanstd(target)])

        return data


class AddIDFeature(MapTransformation):
    @validated()
    def __init__(
        self,
        output_field: str,
        train_length: int,
    ) -> None:
        self.feature_name = output_field
        self.train_length = train_length

    def map_transform(self, data: DataEntry, is_train: bool) -> DataEntry:
        id = data["feat_static_cat"][0]
        if not is_train:
            id = id % self.train_length
        data[self.feature_name] = id
        return data


class AddMeanFeature(MapTransformation):
    @validated()
    def __init__(
        self,
        target_field: str,
        output_field: str,
        train_length: int,
        dtype: Type = np.float32,
        minimum_scale: float = 0.01,
    ) -> None:
        self.target_field = target_field
        self.feature_name = output_field
        self.dtype = dtype
        self.minimum_scale = minimum_scale
        self.train_means = {}
        self.train_length = train_length

    def map_transform(self, data: DataEntry, is_train: bool) -> DataEntry:
        if "item_id" in data:
            id = data["item_id"]
        else:
            id = data["feat_static_cat"][0]
        if is_train:
            if id in self.train_means.keys():
                scale = self.train_means[id]
            else:
                scale = np.array([np.nanmean(data[self.target_field], axis=-1)])
                scale = np.clip(scale, a_min=self.minimum_scale, a_max=np.inf)
                self.train_means[id] = scale
        else:
            scale = _lookup_cached_stat(self.train_means, id, self.train_length)
            if scale is None:
                scale = np.array([np.nanmean(data[self.target_field], axis=-1)])
                scale = np.clip(scale, a_min=self.minimum_scale, a_max=np.inf)
                self.train_means[id] = scale

        data[self.feature_name] = scale
        return data


class AddStdFeature(MapTransformation):
    @validated()
    def __init__(
        self,
        target_field: str,
        output_field: str,
        train_length: int,
        dtype: Type = np.float32,
        minimum_scale: float = 0.1,
    ) -> None:
        self.target_field = target_field
        self.feature_name = output_field
        self.dtype = dtype
        self.minimum_scale = minimum_scale
        self.train_means = {}
        self.train_length = train_length

    def map_transform(self, data: DataEntry, is_train: bool) -> DataEntry:
        if "item_id" in data:
            id = data["item_id"]
        else:
            id = data["feat_static_cat"][0]
        if is_train:
            if id in self.train_means.keys():
                scale = self.train_means[id]
            else:
                scale = np.array([np.nanstd(data[self.target_field], axis=-1)])
                scale = np.clip(scale, a_min=self.minimum_scale, a_max=np.inf)
                self.train_means[id] = scale
        else:
            scale = _lookup_cached_stat(self.train_means, id, self.train_length)
            if scale is None:
                scale = np.array([np.nanstd(data[self.target_field], axis=-1)])
                scale = np.clip(scale, a_min=self.minimum_scale, a_max=np.inf)
                self.train_means[id] = scale

        data[self.feature_name] = scale
        return data


class AddStdFreqFeature(MapTransformation):
    @validated()
    def __init__(
        self,
        target_field: str,
        output_field: str,
        freq: int,
        dtype: Type = np.float32,
        minimum_scale: float = 1e-10,
    ) -> None:
        self.target_field = target_field
        self.feature_name = output_field
        self.freq = freq
        self.dtype = dtype
        self.minimum_scale = minimum_scale

    def map_transform(self, data: DataEntry, is_train: bool) -> DataEntry:
        length = data[self.target_field].shape[0]
        stds = np.nanstd(
            data[self.target_field][: ((length // self.freq) * self.freq)].reshape(-1, self.freq),
            axis=0,
        )
        stds = np.tile(stds, math.ceil(length / self.freq))[:length]
        data[self.feature_name] = stds
        return data


class ScaleAndAddMeanFeature(MapTransformation):
    def __init__(self, target_field: str, output_field: str, prediction_length: int) -> None:
        """Scale the time series using mean scaler and
        add the scale to `output_field`.

        Parameters
        ----------
        target_field
            Key for target time series
        output_field
            Key for the mean feature
        prediction_length
            prediction length, only the time series before the
            last `prediction_length` timesteps is used for
            scale computation
        """
        self.target_field = target_field
        self.feature_name = output_field
        self.prediction_length = prediction_length

    def map_transform(self, data, is_train: bool):
        scale = np.nanmean(
            np.abs(data[self.target_field][..., : -self.prediction_length]),
            axis=-1,
            keepdims=True,
        )
        scale = np.maximum(scale, 1e-7)
        scaled_target = data[self.target_field] / scale
        data[self.target_field] = scaled_target
        data[self.feature_name] = scale

        return data


class ScaleAndAddLongMeanFeature(MapTransformation):
    def __init__(self, target_field: str, output_field: str, prediction_length: int) -> None:
        """Scale the time series using mean scaler and
        add the scale to `output_field`.

        Parameters
        ----------
        target_field
            Key for target time series
        output_field
            Key for the mean feature
        prediction_length
            prediction length, only the time series before the
            last `prediction_length` timesteps is used for
            scale computation
        """
        self.target_field = target_field
        self.feature_name = output_field
        self.prediction_length = prediction_length

    def map_transform(self, data, is_train: bool):
        scaled_target = data[self.target_field] / data["mean"]
        data[self.target_field] = scaled_target
        data[self.feature_name] = data["mean"]
        return data


class ScaleAndAddMinMaxFeature(MapTransformation):
    def __init__(self, target_field: str, output_field: str, prediction_length: int) -> None:
        """Scale the time series using min-max scaler and
        add the scale to `output_field`.

        Parameters
        ----------
        target_field
            Key for target time series
        output_field
            Key for the min-max feature
        prediction_length
            prediction length, only the time series before the
            last `prediction_length` timesteps is used for
            scale computation
        """
        self.target_field = target_field
        self.feature_name = output_field
        self.prediction_length = prediction_length

    def map_transform(self, data, is_train: bool):
        full_seq = data[self.target_field][..., : -self.prediction_length]
        min_val = np.nanmin(full_seq, axis=-1, keepdims=True)
        max_val = np.nanmax(full_seq, axis=-1, keepdims=True)
        loc = min_val
        scale = np.maximum(max_val - min_val, 1e-7)
        scaled_target = (full_seq - loc) / scale
        data[self.target_field] = scaled_target
        data[self.feature_name] = (loc, scale)

        return data
