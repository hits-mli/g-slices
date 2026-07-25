"""Contract tests for TSDiffCond, the DSPD (tsdiff) drop-in for TSFlowCond."""

import hashlib
import re
from pathlib import Path

import pytest
import torch

from gslice.model import TSDiffCond, TSFlowCond
from gslice.model.tsdiff_cond import (
    _CachedGaussianProcess,
    _CachedOrnsteinUhlenbeck,
    _DenoiserClosure,
)
from gslice.vendor.tsdiff import (
    BetaLinear,
    DiscreteDiffusion,
    GaussianProcess,
    Normal,
    OrnsteinUhlenbeck,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

CONTEXT_LENGTH = 24
PREDICTION_LENGTH = 8
NUM_STEPS = 100


def model_kwargs(**overrides):
    kwargs = dict(
        setting="univariate",
        target_dim=1,
        context_length=CONTEXT_LENGTH,
        prediction_length=PREDICTION_LENGTH,
        backbone_params=dict(
            input_dim=1,
            output_dim=1,
            step_emb=16,
            num_residual_blocks=1,
            residual_block="slice",
            bidirectional=False,
            hidden_dim=8,
            dropout=0.0,
            init_skip=False,
            feature_skip=True,
            slice_block_params=dict(block_size=1, diagonal_dense=False, bound_norm=True),
        ),
        prior_params=dict(kernel="ou", gamma=1.0),
        optimizer_params=dict(lr=1e-3, weight_decay=0.0),
        ema_params=dict(beta=0.9999, update_after_step=1, update_every=1),
        frequency="H",
        normalization="mean",
        use_lags=False,
        use_ema=True,
        num_steps=NUM_STEPS,
        solver="ancestral",
        matching="random",
        gp_time_mode="discrete",
        num_samples=1,
    )
    kwargs.update(overrides)
    return kwargs


def make_tsdiff(diffusion_params=None, **overrides):
    diffusion_params = dict(diffusion_params or {"noise": "gp"})
    return TSDiffCond(**model_kwargs(**overrides), diffusion_params=diffusion_params)


def make_tsflow(**overrides):
    overrides.setdefault("solver", "euler")
    return TSFlowCond(**model_kwargs(**overrides))


def irregular_grid(batch, length, seed=0):
    torch.manual_seed(seed)
    deltas = torch.rand(batch, length, 1) * 2.0 + 0.1
    return deltas.cumsum(dim=1)


# --- construction guards ------------------------------------------------------


def test_requires_diffusion_params():
    with pytest.raises(ValueError, match="diffusion_params"):
        TSDiffCond(**model_kwargs(), diffusion_params=None)


def test_alpha_bar_guard_rejects_leaky_schedule():
    with pytest.raises(ValueError, match="alpha_bar_N"):
        make_tsdiff({"noise": "gp", "beta_end": 0.01}, num_steps=30)


def test_beta_end_convention_rejects_tiny_num_steps():
    with pytest.raises(ValueError, match="beta_end"):
        make_tsdiff({"noise": "gp"}, num_steps=16)


def test_euler_solver_is_mapped_to_ancestral():
    with pytest.warns(UserWarning, match="euler"):
        model = make_tsdiff(solver="euler")
    assert model.solver == "ancestral"


def test_residual_to_gp_mean_rejected():
    kwargs = model_kwargs()
    kwargs["backbone_params"]["residual_to_gp_mean"] = True
    with pytest.raises(ValueError, match="residual_to_gp_mean"):
        TSDiffCond(**kwargs, diffusion_params={"noise": "gp"})


def test_kernel_defaults_follow_prior_gamma():
    model = make_tsdiff({"noise": "ou"}, prior_params=dict(kernel="ou", gamma=2.5))
    assert model.diffusion.noise.theta == pytest.approx(2.5)
    model = make_tsdiff({"noise": "gp"}, prior_params=dict(kernel="se", gamma=4.0))
    assert model.diffusion.noise.sigma == pytest.approx(0.5)


def test_hparams_carry_diffusion_params():
    model = make_tsdiff({"noise": "ou", "ou_theta": 3.0})
    assert "diffusion_params" in model.hparams
    assert model.hparams["diffusion_params"]["ou_theta"] == 3.0


# --- state-dict contract ------------------------------------------------------


def test_state_dict_key_parity_with_tsflow():
    tsdiff_keys = set(make_tsdiff().state_dict().keys())
    tsflow_keys = set(make_tsflow().state_dict().keys())
    assert tsdiff_keys == tsflow_keys


def test_strict_state_dict_roundtrip():
    model_a = make_tsdiff()
    state = model_a.state_dict()
    model_b = make_tsdiff()
    model_b.load_state_dict(state, strict=True)


def test_tsdiff_checkpoint_strict_loads_into_tsflow():
    # This is the documented hazard that makes the create_model dispatch
    # mandatory: the key sets are identical, so nothing at load time
    # distinguishes the two families. If this test ever fails, the dispatch
    # requirement can be relaxed.
    state = make_tsdiff().state_dict()
    make_tsflow().load_state_dict(state, strict=True)


# --- dispatch -----------------------------------------------------------------


def test_create_model_dispatch():
    train_model = pytest.importorskip("bin.train_model")
    base_params = dict(
        context_length=CONTEXT_LENGTH,
        prediction_length=PREDICTION_LENGTH,
        backbone_params=model_kwargs()["backbone_params"],
        prior_params=dict(kernel="ou", gamma=1.0),
        optimizer_params=dict(lr=1e-3, weight_decay=0.0),
        ema_params=dict(beta=0.9999, update_after_step=1, update_every=1),
        freq="H",
        normalization="mean",
        use_lags=False,
        use_ema=True,
        num_steps=NUM_STEPS,
        solver="ancestral",
        matching="random",
        device="cpu",
    )
    tsdiff_params = dict(base_params, diffusion_params={"noise": "gp"})
    model = train_model.create_model("univariate", 1, tsdiff_params)
    assert isinstance(model, TSDiffCond)

    flow_params = dict(base_params, solver="euler")
    model = train_model.create_model("univariate", 1, flow_params)
    assert isinstance(model, TSFlowCond)
    assert not isinstance(model, TSDiffCond)

    null_params = dict(base_params, solver="euler", diffusion_params=None)
    model = train_model.create_model("univariate", 1, null_params)
    assert not isinstance(model, TSDiffCond)


def test_eval_reconstruction_dispatch():
    inference = pytest.importorskip("execute.inference_efficiency_gluonts")
    model_params = dict(
        context_length=CONTEXT_LENGTH,
        prediction_length=PREDICTION_LENGTH,
        backbone_params=model_kwargs()["backbone_params"],
        prior_params=dict(kernel="ou", gamma=1.0),
        freq="H",
        normalization="mean",
        use_lags=False,
        num_steps=NUM_STEPS,
        solver="ancestral",
    )
    config = dict(setting="univariate", model_params=dict(model_params, diffusion_params={"noise": "gp"}))
    model = inference._create_tsflow_model(config, target_dim=1)
    assert isinstance(model, TSDiffCond)

    config = dict(setting="univariate", model_params=dict(model_params, solver="euler"))
    model = inference._create_tsflow_model(config, target_dim=1)
    assert not isinstance(model, TSDiffCond)


# --- vendored code integrity ----------------------------------------------------


def test_vendored_files_match_provenance_checksums():
    vendor_dir = REPO_ROOT / "gslice" / "vendor" / "tsdiff"
    provenance = (vendor_dir / "PROVENANCE.md").read_text()
    recorded = dict(re.findall(r"`([\w/]+\.py)` \| `tsdiff/[\w/]+\.py` \| `([0-9a-f]{64})`", provenance))
    assert len(recorded) == 7
    for name, expected in recorded.items():
        actual = hashlib.sha256((vendor_dir / name).read_bytes()).hexdigest()
        assert actual == expected, f"{name} differs from the vendored pinned version"


# --- cached noise wrappers ------------------------------------------------------


def test_cached_gp_covariance_matches_upstream():
    grid = irregular_grid(3, 12)
    cached = _CachedGaussianProcess(dim=1, sigma=0.7)
    upstream = GaussianProcess(dim=1, sigma=0.7)
    torch.testing.assert_close(cached.covariance(t=grid), upstream.covariance(grid))


def test_cached_ou_covariance_matches_upstream():
    grid = irregular_grid(3, 12, seed=1)
    cached = _CachedOrnsteinUhlenbeck(dim=1, theta=2.0)
    upstream = OrnsteinUhlenbeck(dim=1, theta=2.0)
    torch.testing.assert_close(cached.covariance(t=grid), upstream.covariance(grid))


def test_cache_deduplicates_repeated_grids():
    base = irregular_grid(2, 10)
    repeated = base.repeat_interleave(50, dim=0)  # forward()-style sample fan-out
    cached = _CachedGaussianProcess(dim=1, sigma=0.7)
    cov = cached.covariance(t=repeated)
    assert cov.shape == (100, 10, 10)
    assert len(cached._chol_cache) == 2
    cached.covariance(t=repeated)
    assert len(cached._chol_cache) == 2


def test_cached_draws_have_kernel_covariance():
    torch.manual_seed(0)
    grid = irregular_grid(1, 6).expand(40000, -1, -1)
    cached = _CachedOrnsteinUhlenbeck(dim=1, theta=1.0)
    draws = cached(t=grid)[..., 0]
    empirical = draws.T.cov()
    expected = OrnsteinUhlenbeck(dim=1, theta=1.0).covariance(grid[0])
    torch.testing.assert_close(empirical, expected, atol=0.06, rtol=0.0)


# --- diffusion math against the published mechanism ------------------------------


def test_forward_marginal_covariance_on_irregular_grid():
    torch.manual_seed(0)
    num_draws, length, step = 40000, 6, 60
    grid = irregular_grid(1, length).expand(num_draws, -1, -1)
    diffusion = DiscreteDiffusion(
        dim=1,
        num_steps=NUM_STEPS,
        beta_fn=BetaLinear(1e-4, 0.2),
        noise_fn=_CachedGaussianProcess(dim=1, sigma=1.0),
        is_time_series=True,
        predict_gaussian_noise=True,
    )
    x0 = torch.zeros(num_draws, length, 1)
    i = torch.full((num_draws, length, 1), float(step))
    noisy, _ = diffusion.forward(x0, i, t=grid)
    empirical = noisy[..., 0].T.cov()
    expected = (1 - diffusion.alphas[step]) * GaussianProcess(dim=1, sigma=1.0).covariance(grid[0])
    torch.testing.assert_close(empirical, expected, atol=0.06, rtol=0.0)


def test_dspd_gauss_matches_handrolled_ddpm_sampler():
    batch, length, steps = 4, 12, 25
    diffusion = DiscreteDiffusion(
        dim=1,
        num_steps=steps,
        beta_fn=BetaLinear(1e-4, 20.0 / steps),
        noise_fn=Normal(1),
        is_time_series=False,
        predict_gaussian_noise=True,
    )

    def stub_model(x, i=None, **kwargs):
        return 0.1 * x

    torch.manual_seed(7)
    vendored = diffusion.sample(stub_model, num_samples=(batch, length), device="cpu")

    torch.manual_seed(7)
    x = torch.randn(batch, length, 1)
    for step in reversed(range(steps)):
        alpha = diffusion.alphas[step]
        beta = diffusion.betas[step]
        z = torch.randn(batch, length, 1) if step > 0 else torch.zeros_like(x)
        eps = 0.1 * x
        x = (x - beta * eps / (1 - alpha).sqrt()) / (1 - beta).sqrt() + beta.sqrt() * z
    torch.testing.assert_close(vendored, x)


# --- model-level generative core -------------------------------------------------


def fake_batch(model, batch=3, seed=0):
    torch.manual_seed(seed)
    length = CONTEXT_LENGTH + PREDICTION_LENGTH
    x1 = torch.randn(batch, length, 1)
    x0 = torch.randn(batch, length, 1)
    t = torch.rand(batch, 1)
    gp_mean = torch.randn(batch, length, 1, 1)
    mask = torch.ones(batch, length, 1, 1)
    mask[:, -PREDICTION_LENGTH:] = 0.0
    features = torch.cat([gp_mean, mask], dim=-1)
    grid = irregular_grid(batch, length, seed=seed)
    return x1, x0, t, features, grid


def test_p_losses_runs_and_matches_manual_masking():
    model = make_tsdiff()
    x1, x0, t, features, grid = fake_batch(model)
    features[..., -1][:, :4] = 0.0  # unobserved context rows

    torch.manual_seed(11)
    loss = model.p_losses(x1, x0, t, features=features, time_grid=grid)

    torch.manual_seed(11)
    tau = model._diffusion_tau(model._resolve_time_grid(x1, grid))
    closure = _DenoiserClosure(model.backbone, features, grid, model.diffusion.num_steps)
    raw = model.diffusion.get_loss(closure, x1, t=tau)
    mask = features[..., -1].clone()
    mask[:, -PREDICTION_LENGTH:] = 1.0
    expected = (raw * mask).sum() / mask.sum()
    torch.testing.assert_close(loss, expected)
    assert torch.isfinite(loss)


def test_unmasked_loss_flag():
    model = make_tsdiff({"noise": "gp", "loss_mask_unobserved": False})
    x1, x0, t, features, grid = fake_batch(model)
    torch.manual_seed(11)
    loss = model.p_losses(x1, x0, t, features=features, time_grid=grid)
    assert torch.isfinite(loss)


@pytest.mark.parametrize("noise_kind", ["gp", "ou", "normal"])
def test_sample_contract(noise_kind):
    model = make_tsdiff({"noise": noise_kind}, num_steps=25)
    model.eval()
    x1, _, _, features, grid = fake_batch(model, batch=2)
    prediction = model.sample(torch.randn_like(x1), features=features, time_grid=grid)
    assert prediction.shape == x1.shape
    assert torch.isfinite(prediction).all()


def test_sample_num_steps_zero_returns_prior_init():
    model = make_tsdiff()
    model.num_steps = 0
    x1, _, _, features, grid = fake_batch(model, batch=2)
    init = torch.randn_like(x1)
    torch.testing.assert_close(model.sample(init, features=features, time_grid=grid), init)


def test_gp_mean_anchor_added_to_samples(monkeypatch):
    x1 = torch.randn(2, CONTEXT_LENGTH + PREDICTION_LENGTH, 1)

    def stubbed_sample(*args, **kwargs):
        return torch.zeros_like(x1)

    anchored = make_tsdiff({"noise": "gp", "noise_around": "gp_mean"}, num_steps=25)
    anchored.eval()
    _, _, _, features, grid = fake_batch(anchored, batch=2)
    monkeypatch.setattr(anchored.diffusion, "sample", stubbed_sample)
    prediction = anchored.sample(torch.randn_like(x1), features=features, time_grid=grid)
    torch.testing.assert_close(prediction, features[..., -2])

    plain = make_tsdiff({"noise": "gp", "noise_around": "zero"}, num_steps=25)
    plain.eval()
    monkeypatch.setattr(plain.diffusion, "sample", stubbed_sample)
    prediction = plain.sample(torch.randn_like(x1), features=features, time_grid=grid)
    torch.testing.assert_close(prediction, torch.zeros_like(x1))


def test_flow_matching_methods_fail_loud():
    model = make_tsdiff()
    for method in ("forward_path", "get_vf", "fast_denoise", "fast_noise", "sample_n"):
        with pytest.raises(NotImplementedError):
            getattr(model, method)()


# --- upstream (their-backbone) denoiser variant -----------------------------------


THEIR_NET = {
    "noise": "gp",
    "denoiser": "tsdiff_rnn",
    "denoiser_params": {"hidden_dim": 32, "num_layers": 1, "bidirectional": True},
}


def test_their_backbone_replaces_slice_backbone():
    from gslice.model.tsdiff_cond import _TsdiffRNNDenoiser

    model = make_tsdiff(THEIR_NET)
    assert isinstance(model.backbone, _TsdiffRNNDenoiser)
    assert model.ema_backbone.online_model is model.backbone
    # Different network → different key set than TSFlowCond: a their-backbone
    # checkpoint can never silently strict-load into a flow model.
    assert set(model.state_dict().keys()) != set(make_tsflow().state_dict().keys())


def test_their_backbone_p_losses_and_sample():
    model = make_tsdiff(dict(THEIR_NET, noise="ou"), num_steps=25)
    x1, x0, t, features, grid = fake_batch(model)
    loss = model.p_losses(x1, x0, t, features=features, time_grid=grid)
    assert torch.isfinite(loss)
    model.eval()
    prediction = model.sample(torch.randn_like(x1), features=features, time_grid=grid)
    assert prediction.shape == x1.shape
    assert torch.isfinite(prediction).all()


def test_their_backbone_state_dict_roundtrip():
    model_a = make_tsdiff(THEIR_NET)
    model_b = make_tsdiff(THEIR_NET)
    model_b.load_state_dict(model_a.state_dict(), strict=True)


def test_their_backbone_does_not_see_gp_extrapolation():
    # The conditioning channel must be gp_mean * mask: clean context on observed
    # context rows, exactly zero on future rows regardless of the GP posterior.
    from gslice.model.tsdiff_cond import _TsdiffRNNDenoiser

    torch.manual_seed(0)
    denoiser = _TsdiffRNNDenoiser(num_steps=25, hidden_dim=16, num_layers=1)
    length = CONTEXT_LENGTH + PREDICTION_LENGTH
    x = torch.randn(2, length, 1)
    i = torch.full((2, length, 1), 3.0)
    grid = irregular_grid(2, length)
    gp_mean = torch.randn(2, length, 1, 1)
    mask = torch.ones(2, length, 1, 1)
    mask[:, -PREDICTION_LENGTH:] = 0.0
    features = torch.cat([gp_mean, mask], dim=-1)

    out_a = denoiser(x, i=i, features=features, time_grid=grid)
    perturbed = features.clone()
    perturbed[..., -2][:, -PREDICTION_LENGTH:] += 1000.0  # future GP extrapolation
    out_b = denoiser(x, i=i, features=perturbed, time_grid=grid)
    torch.testing.assert_close(out_a, out_b)

    perturbed_context = features.clone()
    perturbed_context[..., -2][:, :CONTEXT_LENGTH] += 1.0  # observed context values
    out_c = denoiser(x, i=i, features=perturbed_context, time_grid=grid)
    assert not torch.allclose(out_a, out_c)


def test_invalid_denoiser_rejected():
    with pytest.raises(ValueError, match="denoiser"):
        make_tsdiff({"noise": "gp", "denoiser": "epsilon_theta"})
