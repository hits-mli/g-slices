import torch

from gslice.utils.gaussian_process import GPRegressor


def generate_time_windows(batch_size, seq_length):
    """Generate time grid for a batch of sequences."""
    t = torch.linspace(0, 1, seq_length).unsqueeze(0).repeat(batch_size, 1).unsqueeze(-1)
    return t


def gp_regression(t_train, y_train, t_forecast=None, prior_params=None):
    """Fit a Gaussian process regressor to training data."""
    gp = GPRegressor(**prior_params) if prior_params is not None else GPRegressor()
    dist = gp.fit(t_train, y_train)
    if t_forecast is not None:
        forecast = dist.sample(t_forecast)
        return gp, forecast
    return gp


def center_time(t, t_now=0.0):
    """Center time values around a reference point."""
    t_vals = t[..., 0]
    t_now = torch.tensor(t_now, device=t.device, dtype=t.dtype)
    t_centered_vals = t_vals - t_now
    return t_centered_vals.unsqueeze(-1)


def shift_time_by_portion(t, shift_portion=0.0):
    """Shift time values by a mask-based reference point."""
    t_vals = t[..., 0]
    if shift_portion > 0.0:
        mask_tensor = torch.tensor(shift_portion, device=t.device, dtype=t.dtype).view(1)
        t_now = t_vals.min(dim=1).values + mask_tensor * (t_vals.max(dim=1).values - t_vals.min(dim=1).values)
        t_shifted_vals = t_vals - t_now[:, None]
        return t_shifted_vals.unsqueeze(-1)
    return t


def mask_time(t, mask_percentage=0.0):
    """Mask time values by splitting at a percentile point."""
    t_vals = t[..., 0]
    if mask_percentage > 0.0:
        mask_tensor = torch.tensor(mask_percentage, device=t.device, dtype=t.dtype).view(1)
        t_now = t_vals.min(dim=1).values + mask_tensor * (t_vals.max(dim=1).values - t_vals.min(dim=1).values)
        idx = (t_vals <= t_now[:, None]).sum(dim=1).clamp_min(1)
        k = idx.min().item()
        return t[:, :k, :], k
    return t, t.shape[1]


def finite_difference_control(samples: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Compute finite-difference derivatives of sample paths."""
    if t.dim() == 3:
        base_t = t[0, :, 0]
    elif t.dim() == 2:
        base_t = t[0]
    else:
        base_t = t

    dt = base_t[1:] - base_t[:-1]
    dt = dt.clamp(min=1e-8)

    view_shape = (1,) * (samples.dim() - 2) + (dt.shape[0], 1)
    dx = samples[..., 1:, :] - samples[..., :-1, :]
    deriv = dx / dt.view(*view_shape)

    first_step = deriv[..., :1, :]
    deriv = torch.cat([first_step, deriv], dim=-2)
    return deriv


class FiniteDiffControl:
    """Control object compatible with the prior spline interface using finite differences."""

    def __init__(self, samples: torch.Tensor, t: torch.Tensor):
        self.samples = samples
        self.t = t
        self._deriv = finite_difference_control(samples, t)

    def derivative(self, t_grid: torch.Tensor) -> torch.Tensor:
        del t_grid
        return self._deriv

    def evaluate(self, t0: torch.Tensor | float) -> torch.Tensor:
        del t0
        return self.samples[..., 0, :]


def fit_spline(samples, t):
    """Fit a spline-like control object to sample paths."""
    return FiniteDiffControl(samples, t)
