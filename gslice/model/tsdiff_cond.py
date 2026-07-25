"""DSPD (tsdiff) drop-in replacement for TSFlowCond.

TSDiffCond swaps the generative core of TSFlowCond — flow matching with a GP
prior — for discrete stochastic process diffusion (DSPD) from Biloš et al.,
"Modeling Temporal Data as Continuous Functions with Stochastic Process
Diffusion" (ICML 2023). The forward kernel, epsilon-prediction loss, ancestral
sampler, and GP/OU noise covariances execute the upstream code vendored
byte-identical under ``gslice/vendor/tsdiff``. Everything else — the data
pipeline, conditioning features, scalers, lags, BackboneModel denoiser, EMA,
predictor surface — is inherited unchanged from TSFlowCond, so the two models
differ only in the generative process.

Deviations from upstream (all local to this file, vendor untouched):
- ``betas``/``alphas`` are moved to the data device before vendored calls
  (upstream keeps them as plain CPU attributes, which breaks CUDA indexing in
  ``get_loss``; keeping them out of buffers preserves state-dict parity with
  TSFlowCond).
- Covariance/Cholesky factors are cached per unique time grid with a jitter
  escalation ladder; OU draws use the cached factor instead of upstream's
  sequential recursion (identical Gaussian law — stationary OU is a GP with a
  Matérn-1/2 kernel).
- The epsilon loss is averaged over observed context rows plus all future
  rows, mirroring upstream forecasting's masked ``weighted_average``
  (disable via ``diffusion_params.loss_mask_unobserved: false``).
"""

import hashlib
import warnings
from collections import OrderedDict
from typing import Tuple

import torch
import torch.nn as nn
from ema_pytorch import EMA
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

from gslice.model.tsflow_cond import TSFlowCond
from gslice.vendor.tsdiff import (
    BetaLinear,
    DiscreteDiffusion,
    GaussianProcess,
    Normal,
    OrnsteinUhlenbeck,
)

patch_typeguard()


class _CachedCholeskyMixin:
    """Per-unique-grid covariance/Cholesky cache with jitter escalation.

    ``covariance()`` returns the same matrices as the upstream class (the
    escalated jitter kicks in only when the factorization fails, and is
    reported); draws use ``L @ eps`` with the cached factor. Grids duplicated
    by ``repeat_interleave`` (training num_samples, eval sample fan-out)
    factorize once.
    """

    _EXTRA_JITTER = (0.0, 1e-3, 1e-2)

    def _init_cache(self, cache_size: int) -> None:
        self._chol_cache: OrderedDict[tuple, tuple[torch.Tensor, torch.Tensor]] = OrderedDict()
        self._cache_size = int(cache_size)
        self._logged_corr = False

    def _param_key(self) -> tuple:
        raise NotImplementedError

    def _maybe_log_adjacent_corr(self, cov: torch.Tensor) -> None:
        if self._logged_corr:
            return
        self._logged_corr = True
        adjacent = float(cov.diagonal(offset=1, dim1=-2, dim2=-1).mean())
        print(
            f"[TSDiffCond] {type(self).__name__}{self._param_key()}: effective adjacent-step "
            f"noise correlation ~ {adjacent:.4f} "
            "(near 0: degenerates to white-noise DDPM; near 1: near-singular covariance)"
        )

    def _factors(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        grids = t.reshape(-1, t.shape[-2])
        unique_grids, inverse = torch.unique(grids, dim=0, return_inverse=True)
        covs, chols = [], []
        for row in unique_grids:
            key = (
                int(row.shape[-1]),
                str(row.device),
                str(row.dtype),
                self._param_key(),
                hashlib.sha1(row.detach().cpu().numpy().tobytes()).hexdigest(),
            )
            hit = self._chol_cache.get(key)
            if hit is None:
                cov = super().covariance(row.view(-1, 1))
                chol = None
                for extra in self._EXTRA_JITTER:
                    candidate = cov if extra == 0.0 else cov + extra * torch.eye(
                        cov.shape[-1], device=cov.device, dtype=cov.dtype
                    )
                    try:
                        chol = torch.linalg.cholesky(candidate)
                    except RuntimeError:
                        continue
                    if extra > 0.0:
                        warnings.warn(
                            f"TSDiffCond: covariance factorization needed extra jitter {extra:g} "
                            f"on top of the upstream 1e-4 (clustered irregular grid?)."
                        )
                    cov = candidate
                    break
                if chol is None:
                    raise RuntimeError(
                        "TSDiffCond: covariance not factorizable even with extra jitter "
                        f"{self._EXTRA_JITTER[-1]:g}; check kernel hyperparameters vs the time grid."
                    )
                self._maybe_log_adjacent_corr(cov)
                hit = (cov, chol)
                self._chol_cache[key] = hit
                if len(self._chol_cache) > self._cache_size:
                    self._chol_cache.popitem(last=False)
            else:
                self._chol_cache.move_to_end(key)
            covs.append(hit[0])
            chols.append(hit[1])
        return torch.stack(covs), torch.stack(chols), inverse

    def _expand(self, unique_mats: torch.Tensor, inverse: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        out = unique_mats if unique_mats.shape[0] == 1 else unique_mats[inverse]
        if t.dim() == 2:
            out = out[0]
        return out

    def covariance(self, t: torch.Tensor = None, **kwargs) -> torch.Tensor:
        cov_u, _, inverse = self._factors(t)
        return self._expand(cov_u, inverse, t)

    def covariance_cholesky(self, t: torch.Tensor = None, **kwargs) -> torch.Tensor:
        _, chol_u, inverse = self._factors(t)
        return self._expand(chol_u, inverse, t)

    def forward(self, *args, t: torch.Tensor = None, **kwargs) -> torch.Tensor:
        chol = self.covariance_cholesky(t)
        eps = torch.randn(*t.shape[:-1], self.dim, device=t.device, dtype=t.dtype)
        return chol @ eps


class _CachedGaussianProcess(_CachedCholeskyMixin, GaussianProcess):
    def __init__(self, dim: int, sigma: float, cache_size: int = 64):
        super().__init__(dim=dim, sigma=sigma)
        self._init_cache(cache_size)

    def _param_key(self) -> tuple:
        return ("gp", float(self.sigma))


class _CachedOrnsteinUhlenbeck(_CachedCholeskyMixin, OrnsteinUhlenbeck):
    def __init__(self, dim: int, theta: float, cache_size: int = 64):
        super().__init__(dim=dim, theta=theta)
        self._init_cache(cache_size)

    def _param_key(self) -> tuple:
        return ("ou", float(self.theta))


class _TsdiffRNNDenoiser(nn.Module):
    """Upstream denoiser with upstream conditioning.

    Wraps the vendored synthetic ``RNNModel`` — the denoiser Biloš et al.
    validate on irregular grids (their Table 6; the feedforward variant is the
    one that fails there) — with the channel-concat conditioning pattern of
    their CSDI experiment: input channels are ``[noisy window, clean scaled
    context (zero on future/unobserved rows), observation mask]``. The
    GP-posterior mean and lag features are deliberately NOT fed to this
    network: the published method has neither.

    ``RNNModel`` embeds observation time with ``PositionalEncoding(max_value=1)``,
    i.e. it expects times in ``[0, 1]`` (upstream normalized by a global
    constant). Every window within an experiment family spans the same
    physical duration, so per-window span normalization here is that same
    constant rescaling. This affects only the network's time embedding — the
    noise covariance keeps physical hours.
    """

    def __init__(
        self,
        *,
        num_steps: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        bidirectional: bool = True,
    ):
        super().__init__()
        from gslice.vendor.tsdiff.synthetic.diffusion_model import RNNModel

        self.net = RNNModel(
            dim=3,
            hidden_dim=int(hidden_dim),
            max_i=max(int(num_steps), 1),
            num_layers=int(num_layers),
            bidirectional=bool(bidirectional),
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        i: torch.Tensor,
        features: torch.Tensor,
        time_grid: torch.Tensor,
    ) -> torch.Tensor:
        if features is None:
            raise ValueError("The tsdiff_rnn denoiser requires conditioning features.")
        mask = features[..., -1]
        cond = features[..., -2] * mask
        stacked = torch.cat([x, cond, mask], dim=-1)
        start = time_grid[..., :1, :]
        span = (time_grid[..., -1:, :] - start).clamp_min(torch.finfo(time_grid.dtype).eps)
        tau = (time_grid - start) / span
        return self.net(stacked, t=tau, i=i)[..., :1]


class _DenoiserClosure(nn.Module):
    """Adapts the denoiser to the callable the vendored diffusion expects.

    Conditioning (features, observation time grid) is closed over, so the
    vendored code only ever passes the noisy input and the step index. The
    step index arrives as an all-equal ``(B, L, 1)`` tensor in ``[0, N-1]``.
    For the gslices backbone it is normalized to ``[0, 1]`` (the scale its
    sinusoidal step embedding sees during flow-matching training); the
    upstream denoiser takes it raw, as its step embedding is built with
    ``max_value = N``.
    """

    def __init__(self, net, features, time_grid, num_steps: int, style: str = "gslices"):
        super().__init__()
        self._net = net
        self._features = features
        self._time_grid = time_grid
        self._num_steps = int(num_steps)
        self._style = style

    def forward(self, x: torch.Tensor, i: torch.Tensor = None, t: torch.Tensor = None, **kwargs) -> torch.Tensor:
        del t, kwargs
        if self._style == "tsdiff":
            return self._net(x, i=i, features=self._features, time_grid=self._time_grid)
        step = i[:, :1, :1].to(dtype=x.dtype) / float(max(self._num_steps - 1, 1))
        return self._net(step, x, self._features, time_grid=self._time_grid)


class TSDiffCond(TSFlowCond):
    """TSFlowCond with the flow-matching core replaced by DSPD diffusion.

    Constructor signature is TSFlowCond's plus a trailing ``diffusion_params``
    dict (required — a config without it must build TSFlowCond instead).
    Reinterpreted kwargs: ``num_steps`` is the number of diffusion steps N;
    ``solver`` must be ``"ancestral"`` (legacy ``"euler"`` is accepted with a
    warning); ``matching`` is ignored. ``prior_params`` keeps driving the
    GP-posterior conditioning features exactly as in TSFlowCond.

    diffusion_params keys (all optional):
      noise: "gp" | "ou" | "normal"          [gp]
      gp_sigma: float, hours                 [1/sqrt(prior gamma)]
      ou_theta: float, 1/hours               [prior gamma]
      beta_start: float                      [1e-4]
      beta_end: float                        [20/num_steps, upstream convention]
      predict_gaussian_noise: bool           [true]
      noise_around: "zero" | "gp_mean"       [zero — published tsdiff behavior]
      time_normalization: "physical"|"span"  [physical — kernel params in hours]
      loss_mask_unobserved: bool             [true]
      cache_size: int                        [64]
      family: "discrete"                     [discrete; CSPD not implemented]
      denoiser: "gslices"|"tsdiff_rnn"       [gslices; the shipped configs set
                                              tsdiff_rnn = upstream RNN denoiser
                                              with CSDI-style conditioning]
      denoiser_params: dict                  [{hidden_dim: 128, num_layers: 2,
                                              bidirectional: true} — upstream
                                              synthetic defaults]
    """

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
        num_steps: int = 100,
        solver: str = "ancestral",
        matching: str = "random",
        gp_time_mode: str = "discrete",
        num_samples: int = 1,
        lags_seq: list[int] | None = None,
        prior_context_length_override: int | None = None,
        gp_fit_context_only: bool = False,
        diffusion_params: dict | None = None,
    ):
        if diffusion_params is None:
            raise ValueError(
                "TSDiffCond requires model_params.diffusion_params; a config without it "
                "must construct TSFlowCond instead (see create_model dispatch)."
            )
        solver_normalized = str(solver).lower()
        if solver_normalized == "euler":
            warnings.warn(
                "TSDiffCond: solver='euler' is a flow-matching setting; using 'ancestral'."
            )
            solver_normalized = "ancestral"
        if solver_normalized != "ancestral":
            raise ValueError(
                f"TSDiffCond supports solver='ancestral' only, got {solver!r}."
            )
        super().__init__(
            setting=setting,
            target_dim=target_dim,
            context_length=context_length,
            prediction_length=prediction_length,
            backbone_params=backbone_params,
            prior_params=prior_params,
            optimizer_params=optimizer_params,
            ema_params=ema_params,
            frequency=frequency,
            normalization=normalization,
            use_lags=use_lags,
            use_ema=use_ema,
            num_steps=num_steps,
            solver=solver_normalized,
            matching=matching,
            gp_time_mode=gp_time_mode,
            num_samples=num_samples,
            lags_seq=lags_seq,
            prior_context_length_override=prior_context_length_override,
            gp_fit_context_only=gp_fit_context_only,
        )
        if "diffusion_params" not in self.hparams:
            self.hparams["diffusion_params"] = dict(diffusion_params)

        if str(matching).lower() != "random":
            warnings.warn(
                f"TSDiffCond ignores matching={matching!r}; OT coupling has no diffusion counterpart."
            )

        dp = dict(diffusion_params)
        self.denoiser_style = str(dp.get("denoiser", "gslices")).lower()
        if self.denoiser_style not in ("gslices", "tsdiff_rnn"):
            raise ValueError(
                f"diffusion_params.denoiser must be gslices|tsdiff_rnn, got {self.denoiser_style!r}."
            )
        if self.denoiser_style == "gslices":
            if bool(backbone_params.get("residual_to_gp_mean", False)):
                raise ValueError(
                    "TSDiffCond: backbone_params.residual_to_gp_mean=True adds the GP mean to the "
                    "network output, which corrupts the epsilon-prediction target."
                )
            if bool(backbone_params.get("init_skip", True)):
                warnings.warn(
                    "TSDiffCond: backbone_params.init_skip=True reparametrizes the output as "
                    "prediction-minus-input; set init_skip: false for a clean epsilon head."
                )
        family = str(dp.get("family", "discrete")).lower()
        if family != "discrete":
            raise NotImplementedError(
                f"TSDiffCond implements family='discrete' (DSPD); got {family!r}. "
                "CSPD needs torchdiffeq/torchsde and is intentionally not wired up."
            )
        noise_kind = str(dp.get("noise", "gp")).lower()
        if noise_kind not in ("gp", "ou", "normal"):
            raise ValueError(f"diffusion_params.noise must be gp|ou|normal, got {noise_kind!r}.")
        self.noise_kind = noise_kind
        self.noise_around = str(dp.get("noise_around", "zero")).lower()
        if self.noise_around not in ("zero", "gp_mean"):
            raise ValueError(
                f"diffusion_params.noise_around must be zero|gp_mean, got {self.noise_around!r}."
            )
        self.time_normalization = str(dp.get("time_normalization", "physical")).lower()
        if self.time_normalization not in ("physical", "span"):
            raise ValueError(
                f"diffusion_params.time_normalization must be physical|span, "
                f"got {self.time_normalization!r}."
            )
        self.loss_mask_unobserved = bool(dp.get("loss_mask_unobserved", True))
        self.predict_gaussian_noise = bool(dp.get("predict_gaussian_noise", True))

        # Kernel hyperparameters live in the same physical time unit (hours) as the
        # observation grids and the Q0/GPRegressor prior, and default to the prior's
        # gamma so diffusion noise and conditioning share the same prior knowledge:
        # gslices OU kernel exp(-gamma|dt|) == upstream exp(-theta|dt|), and
        # gslices SE kernel exp(-gamma dt^2) == upstream exp(-(dt/sigma)^2).
        gamma = float(self.prior_params.get("gamma", 1.0))
        gp_sigma = float(dp["gp_sigma"]) if dp.get("gp_sigma") is not None else gamma ** -0.5
        ou_theta = float(dp["ou_theta"]) if dp.get("ou_theta") is not None else gamma
        beta_start = float(dp.get("beta_start", 1e-4))
        num_steps = int(num_steps)
        beta_end = (
            float(dp["beta_end"])
            if dp.get("beta_end") is not None
            else 20.0 / max(num_steps, 1)
        )
        if beta_end >= 1.0:
            raise ValueError(
                f"beta_end={beta_end:g} >= 1 gives an invalid schedule; with the upstream "
                "beta_end=20/N convention this means num_steps is too small (use N >= 21 "
                "or set diffusion_params.beta_end explicitly)."
            )
        cache_size = int(dp.get("cache_size", 64))
        if noise_kind == "gp":
            noise_fn = _CachedGaussianProcess(dim=1, sigma=gp_sigma, cache_size=cache_size)
        elif noise_kind == "ou":
            noise_fn = _CachedOrnsteinUhlenbeck(dim=1, theta=ou_theta, cache_size=cache_size)
        else:
            noise_fn = Normal(1)
        self.diffusion = DiscreteDiffusion(
            dim=1,
            num_steps=num_steps,
            beta_fn=BetaLinear(beta_start, beta_end),
            noise_fn=noise_fn,
            is_time_series=noise_kind != "normal",
            predict_gaussian_noise=self.predict_gaussian_noise,
        )
        if num_steps > 0:
            terminal_alpha = float(self.diffusion.alphas[-1])
            if terminal_alpha > 1e-3:
                raise ValueError(
                    f"alpha_bar_N = {terminal_alpha:.2e} > 1e-3: the terminal marginal leaks "
                    "data into X_N. Increase num_steps or beta_end (upstream: beta_end=20/N)."
                )
        if self.denoiser_style == "tsdiff_rnn":
            # Replace the SLiCE/S4 backbone (already built by TSFlowCond.__init__)
            # with the upstream denoiser; EMA must wrap the new module.
            self.backbone = _TsdiffRNNDenoiser(
                num_steps=num_steps,
                **dict(dp.get("denoiser_params") or {}),
            )
            self.ema_backbone = EMA(self.backbone, **ema_params)
        self._warned_guidance = False
        self._warned_num_steps = False

    # --- diffusion plumbing -------------------------------------------------

    @property
    def _closure_style(self) -> str:
        return "tsdiff" if self.denoiser_style == "tsdiff_rnn" else "gslices"

    def _sync_schedule(self, ref: torch.Tensor) -> None:
        # betas/alphas are plain attributes in the vendored module; keeping them
        # out of buffers preserves state-dict parity with TSFlowCond, so they are
        # moved manually instead of via Module.to().
        if self.diffusion.alphas.device != ref.device:
            self.diffusion.alphas = self.diffusion.alphas.to(ref.device)
            self.diffusion.betas = self.diffusion.betas.to(ref.device)

    def _diffusion_tau(self, grid: torch.Tensor) -> torch.Tensor:
        # Stationary kernels depend only on time differences, so each window is
        # shifted to start at zero (keeps float32 covariances well-scaled without
        # changing the law). "span" additionally rescales to [0, 1] per window —
        # upstream-parity ablation only: it changes the effective correlation
        # length whenever the window span changes (e.g. cross-frequency eval).
        tau = grid - grid[..., :1, :]
        if self.time_normalization == "span":
            span = tau[..., -1:, :].clamp_min(torch.finfo(tau.dtype).eps)
            tau = tau / span
        return tau

    def _anchor(self, features: torch.Tensor | None) -> torch.Tensor | None:
        if self.noise_around != "gp_mean":
            return None
        if features is None:
            raise ValueError("noise_around='gp_mean' requires conditioning features.")
        return features[..., -2]

    # --- generative core (the only overrides that change behavior) -----------

    @typechecked
    def p_losses(
        self,
        x1: TensorType[float, "batch", "length", "num_series"],
        x0: TensorType[float, "batch", "length", "num_series"],
        t: TensorType[float, "batch", 1],
        features: TensorType[float, "batch", "length", "num_series", "num_features"] | None = None,
        time_grid: TensorType[float, "batch", "length", 1] | None = None,
    ) -> TensorType[float]:
        # x0 (GP-posterior draw) and t (flow time) are flow-matching inputs; the
        # vendored get_loss draws its own uniform diffusion step per batch row.
        del x0, t
        grid = self._resolve_time_grid(x1, time_grid)
        tau = self._diffusion_tau(grid)
        self._sync_schedule(x1)
        anchor = self._anchor(features)
        target = x1 - anchor if anchor is not None else x1
        closure = _DenoiserClosure(
            self.backbone, features, grid, self.diffusion.num_steps, style=self._closure_style
        )
        raw = self.diffusion.get_loss(closure, target, t=tau)
        if self.loss_mask_unobserved and features is not None:
            mask = features[..., -1].clone()
            mask[:, -self.prediction_length :] = 1.0
            return (raw * mask).sum() / mask.sum().clamp_min(1.0)
        return raw.mean()

    @typechecked
    @torch.no_grad()
    def sample(
        self,
        noise: TensorType[float, "batch", "length", "num_series"],
        features: TensorType[float, "batch", "length", "num_series", "num_features"] | None = None,
        observation: TensorType[float, "batch", "length", "num_series"] | None = None,
        observation_mask: TensorType[float, "batch", "length", "num_series"] | None = None,
        guidance_scale: float = 0,
        time_grid: TensorType[float, "batch", "length", 1] | None = None,
    ) -> TensorType[float, "batch", "length", "num_series"]:
        del observation, observation_mask
        if self.num_steps == 0:
            # Prior-only baseline: return the GP-posterior init unchanged.
            return noise.to(self.device)
        if guidance_scale > 0 and not self._warned_guidance:
            self._warned_guidance = True
            warnings.warn("TSDiffCond: guidance_scale > 0 is not supported and is ignored.")
        if self.num_steps != self.diffusion.num_steps and not self._warned_num_steps:
            self._warned_num_steps = True
            warnings.warn(
                f"TSDiffCond: num_steps was changed to {self.num_steps} after training, but the "
                f"DSPD schedule is fixed at N={self.diffusion.num_steps}; sampling uses the "
                "trained schedule (DDIM subsequencing is not implemented)."
            )
        noise = noise.to(self.device)
        grid = self._resolve_time_grid(noise, time_grid)
        tau = self._diffusion_tau(grid)
        self._sync_schedule(noise)
        net = self.ema_backbone if self.use_ema else self.backbone
        closure = _DenoiserClosure(
            net, features, grid, self.diffusion.num_steps, style=self._closure_style
        )
        prediction = self.diffusion.sample(
            closure,
            num_samples=tuple(noise.shape[:-1]),
            device=noise.device,
            t=tau,
        )
        anchor = self._anchor(features)
        if anchor is not None:
            prediction = prediction + anchor
        return prediction

    # --- inherited flow-matching methods that must never run -----------------

    def forward_path(self, *args, **kwargs):
        raise NotImplementedError(
            "TSDiffCond has no flow-matching interpolant; the DSPD forward kernel lives in "
            "the vendored DiscreteDiffusion."
        )

    def get_vf(self, *args, **kwargs):
        raise NotImplementedError("TSDiffCond has no flow-matching vector field.")

    def fast_denoise(self, *args, **kwargs):
        raise NotImplementedError("TSDiffCond does not implement fast_denoise.")

    def fast_noise(self, *args, **kwargs):
        raise NotImplementedError("TSDiffCond does not implement fast_noise.")

    def sample_n(self, *args, **kwargs):
        raise NotImplementedError("TSDiffCond does not implement unconditional sample_n.")
