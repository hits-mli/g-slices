from typing import Tuple

import torch
from einops import rearrange
from ema_pytorch import EMA
from gluonts.torch.model.predictor import PyTorchPredictor
from gluonts.torch.util import lagged_sequence_values
from gluonts.transform import Transformation
from gluonts.transform.split import InstanceSplitter
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

from gslice.arch import BackboneModel
from gslice.irregular import (
    DENSE_PAST_OBSERVED_FIELD,
    DENSE_PAST_TARGET_FIELD,
    DENSE_PAST_TIME_GRID_FIELD,
    FUTURE_TIME_GRID_FIELD,
    LAG_FEATURE_FIELD,
    PAST_TIME_GRID_FIELD,
)
from gslice.model._base import PREDICTION_INPUT_NAMES, TSFlowBase
from gslice.utils.gaussian_process import GPRegressor, Q0Dist
from gslice.utils.util import LongScaler
from gslice.utils.variables import Prior, Setting, get_relative_time_step

patch_typeguard()


def _scale_provided_lag_features(
    lag_features: torch.Tensor,
    *,
    loc: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    scaled_loc = loc.to(device=lag_features.device, dtype=lag_features.dtype)
    scaled_scale = scale.to(device=lag_features.device, dtype=lag_features.dtype)
    while scaled_loc.dim() < lag_features.dim():
        scaled_loc = scaled_loc.unsqueeze(-1)
    while scaled_scale.dim() < lag_features.dim():
        scaled_scale = scaled_scale.unsqueeze(-1)
    return (lag_features - scaled_loc) / scaled_scale


class TSFlowCond(TSFlowBase):
    def __init__(
        self,
        setting: str,
        target_dim: int,
        context_length: int,
        prediction_length: int,
        backbone_params: dict,
        prior_params: dict,
        optimizer_params: dict,
        ema_params: dict,
        frequency: str,
        normalization: str | None = None,
        use_lags: bool = True,
        use_ema: bool = False,
        num_steps: int = 16,
        solver: str = "euler",
        matching: str = "random",
        gp_time_mode: str = "discrete",
        num_samples: int = 1,
        lags_seq: list[int] | None = None,
        prior_context_length_override: int | None = None,
        gp_fit_context_only: bool = False,
    ):
        gp_fit_context_only = bool(gp_fit_context_only)
        if gp_fit_context_only:
            prior_context_length_override = int(context_length)
        super().__init__(
            context_length=context_length,
            prediction_length=prediction_length,
            prior_params=prior_params,
            optimizer_params=optimizer_params,
            frequency=frequency,
            lags_seq=lags_seq,
            prior_context_length_override=prior_context_length_override,
            normalization=normalization,
            use_lags=use_lags,
            use_ema=use_ema,
            num_steps=num_steps,
            solver=solver,
            matching=matching,
            num_samples=num_samples,
        )
        if setting != Setting.UNIVARIATE:
            raise ValueError(f"Unsupported setting {setting!r}; only 'univariate' is supported.")
        num_features = 2 + (len(self.lags_seq) if use_lags else 0)
        self.backbone = BackboneModel(
            **backbone_params,
            num_features=num_features,
            target_dim=1,
        )
        self.ema_backbone = EMA(self.backbone, **ema_params)
        self.setting = setting
        self.guidance_scale = 0
        self.sigmax = self.sigmin
        self.gp_time_mode = str(gp_time_mode).lower()
        if self.gp_time_mode not in {"discrete", "continuous"}:
            raise ValueError(
                f"gp_time_mode must be 'discrete' or 'continuous', got {gp_time_mode!r}."
            )
        self.gp_fit_context_only = gp_fit_context_only
        self.relative_time_step = float(get_relative_time_step(frequency))
        q0_context_freqs = max(1, int(self.prior_context_length // prediction_length))
        q0_params = {
            "kernel": self.prior,
            "context_freqs": q0_context_freqs,
            "prediction_length": prediction_length,
            "freq": self.freq,
            "gamma": float(self.prior_params.get("gamma", 1.0)),
            "iso": 1e-1 if self.prior != Prior.ISO else 0,
        }
        self.q0 = Q0Dist(
            **q0_params,
        )
        self._continuous_gp_regressor = self._build_continuous_gp_regressor()
        self._discrete_q0_cache: dict[tuple[int, int], Q0Dist] = {}

    def _build_indexed_time_grid(self, start_index: int, length: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        t_vals = torch.arange(start_index, start_index + length, device=device, dtype=dtype) * self.relative_time_step
        return t_vals.view(1, length, 1)

    def _build_continuous_gp_regressor(self) -> GPRegressor:
        season_length = self.prior_params.get("season_length", self.freq)
        period_hours = self.prior_params.get("period_hours", None)
        if period_hours is None and self.prior == Prior.PE:
            period_hours = float(self.relative_time_step * float(season_length))
        return GPRegressor(
            kernel=self.prior,
            gamma=float(self.prior_params.get("gamma", 1.0)),
            noise=float(self.prior_params.get("noise", self.prior_params.get("iso", 1e-4))),
            jitter=float(self.prior_params.get("jitter", 1e-6)),
            use_data_mean=bool(self.prior_params.get("use_data_mean", True)),
            season_length=int(season_length) if season_length is not None else None,
            period_hours=float(period_hours) if period_hours is not None else None,
        )

    def _sample_continuous_gp_posterior(
        self,
        prior_context: torch.Tensor,
        prior_observed: torch.Tensor,
        context_length: int,
        future_length: int,
        prior_time_grid: torch.Tensor | None = None,
        query_time_grid: torch.Tensor | None = None,
        dense_prior_context: torch.Tensor | None = None,
        dense_prior_observed: torch.Tensor | None = None,
        dense_prior_time_grid: torch.Tensor | None = None,
        repeat_groups: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        source_context = dense_prior_context if dense_prior_context is not None else prior_context
        source_observed = dense_prior_observed if dense_prior_observed is not None else prior_observed
        source_time_grid = dense_prior_time_grid if dense_prior_time_grid is not None else prior_time_grid

        batch_size, source_length, target_dim = source_context.shape
        device = source_context.device
        dtype = source_context.dtype

        source_t = (
            self._build_indexed_time_grid(0, source_length, device, dtype).expand(batch_size, -1, -1)
            if source_time_grid is None
            else source_time_grid.to(device=device, dtype=dtype)
        )
        if query_time_grid is None:
            query_start = source_length - context_length
            query_length = context_length + future_length
            query_t = self._build_indexed_time_grid(query_start, query_length, device, dtype).expand(batch_size, -1, -1)
        else:
            query_t = query_time_grid.to(device=device, dtype=dtype)
        future_t = query_t[:, context_length:, :]

        gp_means = []
        future_samples = []
        repeat_groups = max(1, int(repeat_groups))
        gp = self._continuous_gp_regressor.to(device)
        with torch.no_grad():
            if repeat_groups > 1 and batch_size % repeat_groups == 0:
                grouped_batch_size = batch_size // repeat_groups
                source_context = source_context[::repeat_groups]
                source_observed = source_observed[::repeat_groups]
                source_t = source_t[::repeat_groups]
                query_t = query_t[::repeat_groups]
                future_t = future_t[::repeat_groups]

                for batch_idx in range(grouped_batch_size):
                    obs_idx = source_observed[batch_idx].nonzero(as_tuple=True)[0]
                    if obs_idx.numel() == 0:
                        obs_idx = torch.tensor([source_length - 1], device=device, dtype=torch.long)

                    gp.fit(
                        source_t[batch_idx, obs_idx, :],
                        source_context[batch_idx, obs_idx, :],
                    )
                    gp_means.append(gp.predict(query_t[batch_idx]))
                    future_samples.append(gp.sample(future_t[batch_idx], num_samples=repeat_groups))

                gp_mean = torch.stack(gp_means, dim=0).repeat_interleave(repeat_groups, dim=0)
                sampled_future = torch.stack(future_samples, dim=0).reshape(batch_size, future_length, target_dim)
                return sampled_future, gp_mean

            for batch_idx in range(batch_size):
                obs_idx = source_observed[batch_idx].nonzero(as_tuple=True)[0]
                if obs_idx.numel() == 0:
                    obs_idx = torch.tensor([source_length - 1], device=device, dtype=torch.long)

                gp.fit(
                    source_t[batch_idx, obs_idx, :],
                    source_context[batch_idx, obs_idx, :],
                )
                gp_means.append(gp.predict(query_t[batch_idx]))
                future_samples.append(gp.sample(future_t[batch_idx], num_samples=1).squeeze(0))

        gp_mean = torch.stack(gp_means, dim=0)
        sampled_future = torch.stack(future_samples, dim=0)
        if sampled_future.shape != (batch_size, future_length, target_dim):
            sampled_future = sampled_future.reshape(batch_size, future_length, target_dim)
        return sampled_future, gp_mean

    def _get_discrete_q0(self, *, context_freqs: int, prediction_length: int) -> Q0Dist:
        key = (int(context_freqs), int(prediction_length))
        module = self._discrete_q0_cache.get(key)
        if module is None:
            q0_params = {
                "kernel": self.prior,
                "context_freqs": int(context_freqs),
                "prediction_length": int(prediction_length),
                "freq": self.freq,
                "gamma": float(self.prior_params.get("gamma", 1.0)),
                "iso": 1e-1 if self.prior != Prior.ISO else 0,
            }
            module = Q0Dist(**q0_params)
            self._discrete_q0_cache[key] = module
        return module.to(self.device)

    def _sample_discrete_gp_posterior_irregular(
        self,
        *,
        scaled_context: torch.Tensor,
        loc: torch.Tensor,
        scale: torch.Tensor,
        dense_past_target: torch.Tensor,
        past_time_grid: torch.Tensor,
        future_time_grid: torch.Tensor,
        anchor_time_grid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        step = float(self.relative_time_step)
        if step <= 0:
            raise ValueError(f"relative_time_step must be > 0, got {step}.")

        dense_context_offsets = torch.round(
            (past_time_grid[:, -1, 0] - past_time_grid[:, 0, 0]) / step
        ).to(dtype=torch.long)
        dense_context_length = int(dense_context_offsets.max().item()) + 1
        if dense_context_length <= 0:
            return None

        future_offsets = torch.round(
            (future_time_grid[..., 0] - anchor_time_grid[:, -1:, 0]) / step
        ).to(dtype=torch.long)
        future_indices = future_offsets - 1
        if future_indices.numel() == 0:
            return None
        if int(future_indices.min().item()) < 0:
            raise ValueError("Future time grid must start after the anchor time for discrete irregular GP.")

        dense_future_length = int(future_indices.max().item()) + 1
        if dense_past_target.shape[1] < dense_context_length or dense_context_length < dense_future_length:
            return None

        context_freqs = max(1, dense_context_length // dense_future_length)
        usable_context_length = int(context_freqs * dense_future_length)
        scaled_dense_context = (dense_past_target[:, -usable_context_length:] - loc) / scale
        _, _, c = scaled_dense_context.shape

        dist = self._get_discrete_q0(
            context_freqs=context_freqs,
            prediction_length=dense_future_length,
        ).gp_regression(
            rearrange(scaled_dense_context, "b l c -> (b c) l"),
            dense_future_length,
        )
        dense_fut = rearrange(dist.sample(), "(b c) l -> b l c", c=c)
        dense_fut_mean = rearrange(dist.mean, "(b c) l -> b l c", c=c)

        gather_index = future_indices.unsqueeze(-1).expand(-1, -1, c)
        fut = torch.gather(dense_fut, 1, gather_index)
        fut_mean = torch.gather(dense_fut_mean, 1, gather_index)
        gp_mean = torch.cat([scaled_context, fut_mean], dim=-2)
        return fut, gp_mean

    @typechecked
    def _extract_features(
        self, data: dict, gp_sample_repeats: int = 1
    ) -> Tuple[
        TensorType[float, "batch", "length", "num_series"],
        TensorType[float, "batch", "length", "num_series"],
        TensorType[float, "batch", "length", "num_series"],
        TensorType[float, "batch", 1, "num_series"],
        TensorType[float, "batch", 1, "num_series"],
        TensorType[float, "batch", "length", "num_series", "num_features"],
        TensorType[float, "batch", "length", 1] | None,
    ]:
        past = data["past_target"]
        future = data["future_target"]
        context_observed = data["past_observed_values"]
        mean = data["mean"]
        past_time_grid = data.get(PAST_TIME_GRID_FIELD)
        future_time_grid = data.get(FUTURE_TIME_GRID_FIELD)
        lag_features = data.get(LAG_FEATURE_FIELD)
        dense_past_target = data.get(DENSE_PAST_TARGET_FIELD)
        dense_past_observed = data.get(DENSE_PAST_OBSERVED_FIELD)
        dense_past_time_grid = data.get(DENSE_PAST_TIME_GRID_FIELD)
        # id = data["id"]
        if self.setting == Setting.UNIVARIATE:
            past = rearrange(past, "... -> ... 1")
            future = rearrange(future, "... -> ... 1")
            context_observed = rearrange(context_observed, "... -> ... 1")
            mean = rearrange(data["mean"], "... -> ... 1")
            if lag_features is not None and lag_features.dim() == 3:
                lag_features = lag_features.unsqueeze(2)
            if dense_past_target is not None and dense_past_target.dim() == 2:
                dense_past_target = dense_past_target.unsqueeze(-1)
            if dense_past_observed is not None and dense_past_observed.dim() == 1:
                dense_past_observed = dense_past_observed.unsqueeze(0)
            if dense_past_time_grid is not None and dense_past_time_grid.dim() == 2:
                dense_past_time_grid = dense_past_time_grid.unsqueeze(-1)

        context = past[:, -self.context_length :]
        long_context = past[:, : -self.context_length]
        prior_context = past[:, -self.prior_context_length :]

        if isinstance(self.scaler, LongScaler):
            scaled_context, loc, scale = self.scaler(context, scale=mean)
        else:
            _, loc, scale = self.scaler(past, context_observed)
            scaled_context = context / scale
        scaled_long_context = (long_context - loc) / scale
        scaled_prior_context = (prior_context - loc) / scale
        scaled_future = (future - loc) / scale

        x1 = torch.cat([scaled_context, scaled_future], dim=-2)
        batch_size, length, c = x1.shape

        observation_mask = torch.zeros_like(x1)
        observation_mask[:, : -self.prediction_length] = context_observed[:, -self.context_length :]
        stage2_time_grid = None
        prior_time_grid = None
        if past_time_grid is not None and future_time_grid is not None:
            if past_time_grid.dim() == 2:
                past_time_grid = past_time_grid.unsqueeze(-1)
            if future_time_grid.dim() == 2:
                future_time_grid = future_time_grid.unsqueeze(-1)
            stage2_time_grid = torch.cat([past_time_grid[:, -self.context_length :], future_time_grid], dim=1)
            prior_time_grid = past_time_grid[:, -self.prior_context_length :]

        features = []
        if self.use_lags:
            if lag_features is not None:
                lags = lag_features.to(x1.device, dtype=x1.dtype)
                lags = _scale_provided_lag_features(
                    lags,
                    loc=loc,
                    scale=scale,
                )
            else:
                lags = lagged_sequence_values(
                    self.lags_seq,
                    scaled_long_context,
                    x1,
                    dim=1,
                )
            features.append(lags)

        if context_observed.dim() == 3:
            prior_observed = context_observed[:, -self.prior_context_length :].any(dim=-1).to(dtype=torch.bool)
            context_only_observed = context_observed[:, -self.context_length :].any(dim=-1).to(dtype=torch.bool)
        else:
            prior_observed = context_observed[:, -self.prior_context_length :].to(dtype=torch.bool)
            context_only_observed = context_observed[:, -self.context_length :].to(dtype=torch.bool)

        # `gp_time_mode` should control the GP branch explicitly. Irregular experiments
        # may still provide time grids for downstream backbones (for example SLiCE),
        # but `gp_time_mode="discrete"` must keep the GP on the discrete prior path.
        if self.gp_time_mode == "continuous":
            dense_source_context = None
            dense_source_observed = None
            dense_source_time_grid = None
            gp_context = scaled_prior_context
            gp_observed = prior_observed
            gp_prior_time_grid = prior_time_grid
            if self.gp_fit_context_only:
                gp_context = scaled_context
                gp_observed = context_only_observed
                gp_prior_time_grid = past_time_grid[:, -self.context_length :] if past_time_grid is not None else None
            elif dense_past_target is not None and dense_past_observed is not None:
                dense_source_context = (dense_past_target[:, -self.prior_context_length :] - loc) / scale
                dense_source_observed = dense_past_observed[:, -self.prior_context_length :].to(dtype=torch.bool)
                if dense_past_time_grid is not None:
                    dense_source_time_grid = dense_past_time_grid[:, -self.prior_context_length :]
            fut, gp_mean = self._sample_continuous_gp_posterior(
                gp_context,
                gp_observed,
                context_length=self.context_length,
                future_length=self.prediction_length,
                prior_time_grid=gp_prior_time_grid,
                query_time_grid=stage2_time_grid,
                dense_prior_context=dense_source_context,
                dense_prior_observed=dense_source_observed,
                dense_prior_time_grid=dense_source_time_grid,
                repeat_groups=gp_sample_repeats,
            )
        else:
            discrete_irregular = None
            if stage2_time_grid is not None and dense_past_target is not None and future_time_grid is not None:
                anchor_time_grid = dense_past_time_grid if dense_past_time_grid is not None else past_time_grid
                if anchor_time_grid is not None:
                    discrete_irregular = self._sample_discrete_gp_posterior_irregular(
                        scaled_context=scaled_context,
                        loc=loc,
                        scale=scale,
                        dense_past_target=dense_past_target,
                        past_time_grid=past_time_grid,
                        future_time_grid=future_time_grid,
                        anchor_time_grid=anchor_time_grid,
                    )
            if discrete_irregular is not None:
                fut, gp_mean = discrete_irregular
            else:
                dist = self.q0.gp_regression(rearrange(scaled_prior_context, "b l c -> (b c) l"), self.prediction_length)
                fut = rearrange(dist.sample(), "(b c) l -> b l c", c=c)
                fut_mean = rearrange(dist.mean, "(b c) l -> b l c", c=c)
                gp_mean = torch.cat([scaled_context, fut_mean], dim=-2)

        features.append(gp_mean.unsqueeze(-1))
        features.append(observation_mask.unsqueeze(-1))
        x0 = torch.cat([scaled_context, fut], dim=-2)

        features = torch.cat(features, dim=-1)
        return x1, x0, observation_mask, loc, scale, features, stage2_time_grid

    @typechecked
    def training_step(self, data: dict, idx: int) -> dict:
        assert self.training is True
        data = self._expand_training_batch(data)
        x1, x0, _, _, _, features, time_grid = self._extract_features(
            data,
            gp_sample_repeats=self.train_num_samples,
        )
        t = torch.rand((x1.shape[0], 1), device=self.device)
        loss = self.p_losses(x1, x0, t, features, time_grid=time_grid)
        self.log(
            "train_loss",
            loss,
            on_step=False,
            batch_size=x1.shape[0],
            on_epoch=True,
            logger=True,
        )
        return {"loss": loss}

    @typechecked
    def forward(
        self,
        past_target: TensorType[float, "batch", "length"] | TensorType[float, "batch", "length", "num_series"],
        past_observed_values: TensorType[float, "batch", "length"] | TensorType[float, "batch", "length", "num_series"],
        mean: TensorType[float, "batch", 1] | TensorType[float, "batch", 1, "num_series"] = None,
        past_time_grid: torch.Tensor | None = None,
        future_time_grid: torch.Tensor | None = None,
        lag_features: torch.Tensor | None = None,
        dense_past_target: torch.Tensor | None = None,
        dense_past_observed_values: torch.Tensor | None = None,
        dense_past_time_grid: torch.Tensor | None = None,
    ) -> (
        TensorType[float, "batch", "num_samples", "prediction_length"]
        | TensorType[float, "batch", "num_samples", "prediction_length", "num_series"]
    ):
        # This is only used during prediction
        past_target = past_target.to(self.device).repeat_interleave(self.num_samples, dim=0)
        past_observed_values = past_observed_values.to(self.device).repeat_interleave(self.num_samples, dim=0)
        mean = mean.to(self.device).repeat_interleave(self.num_samples, dim=0)
        if past_time_grid is not None:
            past_time_grid = past_time_grid.to(self.device).repeat_interleave(self.num_samples, dim=0)
        if future_time_grid is not None:
            future_time_grid = future_time_grid.to(self.device).repeat_interleave(self.num_samples, dim=0)
        if lag_features is not None:
            lag_features = lag_features.to(self.device).repeat_interleave(self.num_samples, dim=0)
        if dense_past_target is not None:
            dense_past_target = dense_past_target.to(self.device).repeat_interleave(self.num_samples, dim=0)
        if dense_past_observed_values is not None:
            dense_past_observed_values = dense_past_observed_values.to(self.device).repeat_interleave(self.num_samples, dim=0)
        if dense_past_time_grid is not None:
            dense_past_time_grid = dense_past_time_grid.to(self.device).repeat_interleave(self.num_samples, dim=0)
        future_target = torch.zeros_like(past_target[:, -self.prediction_length :])
        data = dict(
            past_target=past_target,
            past_observed_values=past_observed_values,
            mean=mean,
            future_target=future_target,
            past_time_grid=past_time_grid,
            future_time_grid=future_time_grid,
            lag_features=lag_features,
            dense_past_target=dense_past_target,
            dense_past_observed_values=dense_past_observed_values,
            dense_past_time_grid=dense_past_time_grid,
        )
        observation, x0, observation_mask, loc, scale, features, time_grid = self._extract_features(
            data,
            gp_sample_repeats=self.num_samples,
        )
        x0 = x0 + self.sigmax * torch.randn_like(x0)
        pred = self.sample(
            x0.to(self.device),
            features=features,
            observation=observation,
            observation_mask=observation_mask,
            guidance_scale=self.guidance_scale,
            time_grid=time_grid,
        )
        if self.setting == Setting.UNIVARIATE:
            pred = rearrange(pred * scale + loc, "(b n) l 1 -> b n l", n=self.num_samples)
        else:
            pred = rearrange(pred * scale + loc, "(b n) l k -> b n l k", n=self.num_samples)
        return pred[:, :, observation.shape[1] - self.prediction_length :]

    @typechecked
    def get_predictor(
        self,
        input_transform: Transformation,
        batch_size: int = 40,
        device: str | torch.device = None,
        input_names: list[str] | None = None,
    ):
        if input_names is None:
            input_names = PREDICTION_INPUT_NAMES
        # Use self.device if device parameter is None
        if device is None:
            device = self.device
        return PyTorchPredictor(
            prediction_length=self.prediction_length,
            input_names=input_names,
            prediction_net=self,
            batch_size=batch_size,
            input_transform=input_transform,
            device=device,
        )

    @torch.no_grad()
    def predict_plot_diagnostics_from_past(
        self,
        past_target,
        past_observed_values,
        *,
        num_samples: int | None = None,
        scale=None,
        mean=None,
        past_time_grid=None,
        future_time_grid=None,
        lag_features=None,
        dense_past_target=None,
        dense_past_observed_values=None,
        dense_past_time_grid=None,
    ) -> dict[str, torch.Tensor]:
        del scale
        nsamples = int(num_samples if num_samples is not None else self.num_samples)

        past_target = past_target.to(self.device)
        past_observed_values = past_observed_values.to(self.device)
        mean = mean.to(self.device)

        repeated_past = past_target.repeat_interleave(nsamples, dim=0)
        repeated_obs = past_observed_values.repeat_interleave(nsamples, dim=0)
        repeated_mean = mean.repeat_interleave(nsamples, dim=0)
        repeated_past_time = None if past_time_grid is None else past_time_grid.to(self.device).repeat_interleave(nsamples, dim=0)
        repeated_future_time = None if future_time_grid is None else future_time_grid.to(self.device).repeat_interleave(nsamples, dim=0)
        repeated_lags = None if lag_features is None else lag_features.to(self.device).repeat_interleave(nsamples, dim=0)
        repeated_dense_past_target = None if dense_past_target is None else dense_past_target.to(self.device).repeat_interleave(nsamples, dim=0)
        repeated_dense_past_observed = None if dense_past_observed_values is None else dense_past_observed_values.to(self.device).repeat_interleave(nsamples, dim=0)
        repeated_dense_past_time = None if dense_past_time_grid is None else dense_past_time_grid.to(self.device).repeat_interleave(nsamples, dim=0)
        future_target = torch.zeros_like(repeated_past[:, -self.prediction_length :])
        data = dict(
            past_target=repeated_past,
            past_observed_values=repeated_obs,
            mean=repeated_mean,
            future_target=future_target,
            past_time_grid=repeated_past_time,
            future_time_grid=repeated_future_time,
            lag_features=repeated_lags,
            dense_past_target=repeated_dense_past_target,
            dense_past_observed_values=repeated_dense_past_observed,
            dense_past_time_grid=repeated_dense_past_time,
        )

        observation, x0, observation_mask, loc, scale, features, time_grid = self._extract_features(
            data,
            gp_sample_repeats=nsamples,
        )
        x0 = x0 + self.sigmax * torch.randn_like(x0)
        pred = self.sample(
            x0,
            features=features,
            observation=observation,
            observation_mask=observation_mask,
            guidance_scale=self.guidance_scale,
            time_grid=time_grid,
        )
        gp_mean = features[..., -2]

        if self.setting == Setting.UNIVARIATE:
            pred = rearrange(pred * scale + loc, "(b n) l 1 -> b n l", n=nsamples)
            gp_mean = rearrange(gp_mean * scale + loc, "(b n) l 1 -> b n l", n=nsamples)
        else:
            pred = rearrange(pred * scale + loc, "(b n) l k -> b n l k", n=nsamples)
            gp_mean = rearrange(gp_mean * scale + loc, "(b n) l k -> b n l k", n=nsamples)

        context_length = int(observation.shape[1] - self.prediction_length)
        return {
            "context_forecast_samples": pred[:, :, :context_length],
            "future_forecast_samples": pred[:, :, context_length:],
            "gp_mean_context": gp_mean[:, 0, :context_length],
            "gp_mean_future": gp_mean[:, 0, context_length:],
        }
