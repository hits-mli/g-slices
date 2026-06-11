import torch
from torch.distributions import MultivariateNormal as MV
from collections import OrderedDict
from torchtyping import TensorType

from gslice.utils.variables import Prior

_KERNEL_CACHE_SIZE = 64
_kernel_cache: "OrderedDict[tuple, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]" = OrderedDict()
_MAX_CACHED_UNBATCHED_KERNEL_LENGTH = 4096


def _make_kernel_key(
    kernel: Prior,
    gamma: torch.Tensor,
    noise: torch.Tensor,
    jitter: torch.Tensor,
    season_length: int | None,
    period_hours: float | None,
    t_grid: torch.Tensor,
) -> tuple:
    flat = t_grid.detach().cpu().numpy().tobytes()
    return (
        kernel.value,
        float(gamma),
        float(noise),
        float(jitter),
        season_length,
        period_hours,
        t_grid.shape,
        hash(flat),
    )


def _should_cache_kernel_matrices(t_grid: torch.Tensor) -> bool:
    if t_grid.dim() != 1:
        return False
    return int(t_grid.numel()) <= int(_MAX_CACHED_UNBATCHED_KERNEL_LENGTH)


def isotropic_kernel(
    gamma: TensorType[float], t: TensorType[float, "length"], u: TensorType[float, "length"]
) -> TensorType[float, "length", "length"]:
    return gamma * torch.eye(t.size(0), device=t.device, dtype=t.dtype)


def radial_basis_kernel(
    gamma: TensorType[float], t: TensorType[float, "length"], u: TensorType[float, "length"]
) -> TensorType[float, "length", "length"]:
    return torch.exp(-gamma * (t - u) ** 2)


def ornstein_uhlenbeck(
    gamma: TensorType[float], t: TensorType[float, "length"], u: TensorType[float, "length"]
) -> TensorType[float, "length", "length"]:
    return torch.exp(-gamma * torch.abs(t - u))


def periodic_kernel(
    gamma: TensorType[float], t: TensorType[float, "length"], u: TensorType[float, "length"]
) -> TensorType[float, "length", "length"]:
    return torch.exp(-gamma * torch.sin(t - u) ** 2)


kernel_function_map = {
    Prior.ISO: isotropic_kernel,
    Prior.SE: radial_basis_kernel,
    Prior.OU: ornstein_uhlenbeck,
    Prior.PE: periodic_kernel,
}


def get_gp(kernel: Prior, gamma: TensorType[float], length: int, freq: int) -> TensorType[float, "length", "length"]:
    """
    Generate a Gaussian process covariance matrix using the specified kernel.

    Args:
        kernel (Prior): The type of kernel to use.
        gamma (TensorType[float]): Kernel parameter.
        length (int): The number of time points (size of the covariance matrix).
        freq (int): Frequency parameter used to determine the time scale.
    """
    t = torch.arange(length, device=gamma.device, dtype=gamma.dtype) * (torch.pi / freq)
    cov = kernel_function_map[kernel](gamma, t, t.unsqueeze(1))
    return cov


class Q0Dist(torch.nn.Module):
    def __init__(
        self,
        kernel: Prior,
        context_freqs: int,
        prediction_length: int,
        freq: int = 24,
        gamma: float = 1.0,
        iso: float = 1e-4,
        use_seasonal_mean: bool = True,
    ):
        """
        Initialize the GP distribution.

        Args:
            kernel (Prior): The kernel type (e.g. Prior.SE, Prior.OU, Prior.PE, etc.).
            context_freqs (int): The number of frequency groups in the context.
            prediction_length (int): The prediction horizon.
            freq (int, optional): Frequency parameter for time computations. Defaults to 24.
            gamma (float, optional): The kernel hyperparameter. Defaults to 1.0.
            iso (float, optional): The isotropic noise level. Defaults to 1e-4.
            use_seasonal_mean (bool, optional): Whether to use a seasonal mean. Defaults to True.
        """
        super().__init__()
        context_length = context_freqs * prediction_length
        window_length = context_length + prediction_length

        # Convert gamma to a tensor for consistency.
        self.gamma = torch.tensor(gamma, dtype=torch.float64)
        self.iso = torch.tensor(iso, dtype=torch.float64)
        self.kernel = kernel
        self.freq = freq
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.use_seasonal_mean = use_seasonal_mean

        mean = torch.zeros(window_length)
        cov = get_gp(kernel, self.gamma, window_length, freq) + self.iso * torch.eye(window_length)

        # GP reg
        t_obs = torch.cat([torch.ones(context_length), torch.zeros(prediction_length)]) == 1
        t_new = ~t_obs

        # Extract the relevant submatrices from the covariance matrix.
        K = cov[t_obs][:, t_obs]
        K_star = cov[t_obs][:, t_new]
        K_star_star = cov[t_new][:, t_new]

        # Add a small amount of noise to the diagonal to ensure invertibility.
        K += 1e-4 * torch.eye(K.size(0))
        K_inv = torch.linalg.inv(K)
        K_inv_x_K_star = K_inv @ K_star
        cov_reg = K_star_star - K_star.T @ K_inv_x_K_star

        # Register buffers
        self.register_buffer("cov_inv", torch.linalg.inv(cov).float(), persistent=False)
        self.register_buffer("K_inv_x_K_star", K_inv_x_K_star.float(), persistent=False)
        self.register_buffer("cov_reg", cov_reg.float(), persistent=False)
        self.register_buffer(
            "context_phase_index",
            (torch.arange(context_length, dtype=torch.long) % max(1, int(freq))),
            persistent=False,
        )
        self.register_buffer(
            "future_phase_index",
            ((torch.arange(prediction_length, dtype=torch.long) + context_length) % max(1, int(freq))),
            persistent=False,
        )

        self.dist = MV(loc=mean.float(), covariance_matrix=cov.float())

    def _apply(self, fn):
        """
        Override _apply so that when the module is moved to a new device,
        we update our precomputed distribution (self.dist) accordingly.
        """
        super()._apply(fn)
        self.dist = MV(
            loc=self.dist.loc.to(self.cov_inv.device),
            scale_tril=self.dist.scale_tril.to(self.cov_inv.device),
        )
        return self

    def log_likelihood(self, x: TensorType[float, "batch_size", "length"]) -> TensorType[float]:
        """
        Compute the log likelihood of the data given the GP distribution.
        """
        x = x.to(self.cov_inv.device)
        return -self.dist.log_prob(x)

    def forward(self, num_samples: int) -> TensorType[float, "num_samples", "length"]:
        samples = self.dist.sample(torch.Size([num_samples]))
        return samples

    def gp_regression(self, x: TensorType[float], prediction_length: int) -> MV:
        """
        Perform GP regression on input data.

        Args:
            x (torch.Tensor): Input data tensor of shape [batch_size, window_length].
            prediction_length (int): Number of prediction points.

        Returns:
            MultivariateNormal: GP regression posterior.
        """
        batch_size = x.size(0)
        phase_index = self.context_phase_index.to(device=x.device)
        phase_sum = x.new_zeros(batch_size, self.freq)
        phase_sum.scatter_add_(1, phase_index.unsqueeze(0).expand(batch_size, -1), x)

        phase_count = x.new_zeros(self.freq)
        phase_count.scatter_add_(0, phase_index, x.new_ones(self.context_length))
        global_mean = x.mean(dim=1, keepdim=True)
        loc_freq = phase_sum / phase_count.clamp_min(1.0).unsqueeze(0)
        loc_freq = torch.where(phase_count.unsqueeze(0) > 0, loc_freq, global_mean)
        if self.use_seasonal_mean and self.context_length >= self.freq and self.context_length % self.freq == 0:
            # Only use full seasonal blocks when the available context spans an integer
            # number of periods. Otherwise fall back to the partial-period estimate above.
            x_reshaped = x.reshape(batch_size, -1, self.freq)
            loc_freq = x_reshaped.mean(1, keepdims=False)
        elif not self.use_seasonal_mean:
            # Compute a single global mean per batch element
            loc_freq = x.mean(-1, keepdims=True).expand(-1, self.freq)

        # For periodic kernels, set location bias to zero.
        if self.kernel == Prior.PE:
            loc_freq = torch.zeros_like(loc_freq)

        # Remove the location bias.
        context_loc = loc_freq.gather(1, phase_index.unsqueeze(0).expand(batch_size, -1))
        x_centered = x - context_loc

        # Compute the regression mean.
        mean = x_centered @ self.K_inv_x_K_star
        if prediction_length <= self.prediction_length:
            future_phase_index = self.future_phase_index[:prediction_length]
        else:
            future_phase_index = (
                (torch.arange(prediction_length, device=x.device, dtype=torch.long) + self.context_length) % self.freq
            )
        future_phase_index = future_phase_index.to(device=x.device)
        future_loc = loc_freq.gather(1, future_phase_index.unsqueeze(0).expand(batch_size, -1))
        mean = mean + future_loc

        # Scale the regression covariance.
        cov = self.cov_reg
        return MV(mean, cov)


class GPRegressor(torch.nn.Module):
    """
    A flexible Gaussian Process regressor that allows querying at arbitrary time points.
    
    Supports multi-channel time series and batched inputs.
    
    Expected input shapes:
        - t_train: (B, L, 1) or (L,) for unbatched
        - y_train: (B, L, C) or (L, C) or (L,) for unbatched single-channel
        - t_test: (B, M, 1) or (M,) for unbatched
    
    Output shapes:
        - mean: (B, M, C) or (M, C) matching input batch/channel structure
        - std: (B, M, C) or (M, C) if return_std=True
        - cov: (B, C, M, M) or (C, M, M) if return_cov=True (per-channel covariance)
    """
    
    def __init__(
        self,
        kernel: Prior = Prior.SE,
        gamma: float = 1.0,
        noise: float = 1e-2,
        jitter: float = 1e-6,
        use_data_mean: bool = True,
        season_length: int | None = None,
        period_hours: float | None = None,
    ):
        """
        Initialize the GP regressor.
        
        Args:
            kernel (Prior): Kernel type (Prior.SE, Prior.OU, Prior.PE, Prior.ISO).
            gamma (float): Kernel length-scale parameter.
            noise (float): Observation noise variance (added to diagonal).
            jitter (float): Numerical jitter for matrix stability.
            use_data_mean (bool): If True (default), use the training data mean as the
                mean function. Predictions far from training data will revert to this
                mean rather than zero. If False, use zero mean.
            season_length (int | None): If set, use a periodic/seasonal mean function
                instead of a simple global mean. The training data is folded into
                periods of this length and averaged to extract a per-phase seasonal
                profile (like Q0Dist's loc_freq). Requires the context to contain
                at least one complete period. Falls back to simple mean otherwise.
        """
        super().__init__()
        self.kernel = kernel
        self.use_data_mean = use_data_mean
        self.season_length = season_length
        self.period_hours = float(period_hours) if period_hours is not None else None
        self.register_buffer("gamma", torch.tensor(gamma, dtype=torch.float32))
        self.register_buffer("noise", torch.tensor(noise, dtype=torch.float32))
        # Numerical jitter used only for matrix factorization stability (does not model observation noise).
        self.register_buffer("jitter", torch.tensor(jitter, dtype=torch.float32))
        
        # Storage for fitted data
        self.t_train: torch.Tensor | None = None
        self.y_train: torch.Tensor | None = None
        self.K_inv: torch.Tensor | None = None
        self.alpha: torch.Tensor | None = None  # K_inv @ (y - mean) for efficient prediction
        self.batch_size: int = 0
        self.n_channels: int = 1
        self.K_train: torch.Tensor | None = None  # Clean training kernel without noise
        self.K_train_noisy: torch.Tensor | None = None  # Kernel used for inversion (noise + jitter)
        self.y_mean: torch.Tensor | None = None  # Data-driven mean function
        self.y_seasonal: torch.Tensor | None = None  # Per-phase seasonal profile
        self._t0: torch.Tensor | None = None  # First training time (for phase mapping)
        self._dt: torch.Tensor | float | None = None  # Time step (for phase mapping)

    def _make_psd(self, mat: torch.Tensor) -> torch.Tensor:
        """
        Make a matrix positive semi-definite by clipping negative eigenvalues.
        
        Args:
            mat: Matrix to make positive semi-definite.
            
        Returns:
            Positive semi-definite matrix.
        """
        orig_dtype = mat.dtype
        mat = mat.float()  # eigh requires float32+
        mat = 0.5 * (mat + mat.transpose(-1, -2))
        eigenvalues, eigenvectors = torch.linalg.eigh(mat)
        eigenvalues = torch.clamp(eigenvalues, min=1e-6)
        result = eigenvectors @ torch.diag_embed(eigenvalues) @ eigenvectors.transpose(-1, -2)
        return result.to(orig_dtype)
    
    def _is_train_query(self, t_test: torch.Tensor) -> bool:
        """
        Return True when the query locations match the stored training grid.
        This lets us use the same noisy kernel that was inverted, ensuring
        interpolation at the training points even when a small noise term is set.
        """
        if self.t_train is None:
            return False
        if t_test.shape != self.t_train.shape or t_test.dim() != self.t_train.dim():
            return False
        return torch.allclose(t_test, self.t_train)

    def _match_train_indices(
        self, t_test: torch.Tensor, atol: float = 1e-6
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Vectorized match of test times to training times.

        Returns:
            match_mask: bool tensor of shape (M,) or (B, M) — True where test point matches a train point.
            train_idx: int tensor of same shape — index into training grid for matched points (0 where unmatched).
        """
        # t_test: (M,) or (B, M), t_train: (L,) or (B, L)
        if t_test.dim() == 1 and self.t_train.dim() == 1:
            # Unbatched: pairwise distances (M, L)
            diff = (t_test.unsqueeze(-1) - self.t_train.unsqueeze(0)).abs()  # (M, L)
            min_dist, train_idx = diff.min(dim=-1)  # (M,), (M,)
            match_mask = min_dist < atol
        else:
            # Batched: (B, M) vs (B, L) -> (B, M, L)
            diff = (t_test.unsqueeze(-1) - self.t_train.unsqueeze(-2)).abs()  # (B, M, L)
            min_dist, train_idx = diff.min(dim=-1)  # (B, M), (B, M)
            match_mask = min_dist < atol
        return match_mask, train_idx
    
    def _normalize_t(self, t: torch.Tensor) -> torch.Tensor:
        """
        Normalize time tensor to shape (B, L) for internal processing.
        
        Args:
            t: Time tensor of shape (B, L, 1), (B, L), (L, 1), or (L,)
            
        Returns:
            Time tensor of shape (B, L) or (L,)
        """
        if t.dim() == 3:
            return t.squeeze(-1)  # (B, L, 1) -> (B, L)
        elif t.dim() == 2 and t.shape[-1] == 1:
            return t.squeeze(-1)  # (L, 1) -> (L,)
        return t
    
    def _normalize_y(self, y: torch.Tensor) -> tuple[torch.Tensor, int]:
        """
        Normalize y tensor and extract channel count.
        
        Args:
            y: Observation tensor of shape (B, L, C), (B, L), (L, C), or (L,)
            
        Returns:
            Tuple of (normalized y of shape (B, L, C) or (L, C), n_channels)
        """
        if y.dim() == 1:
            return y.unsqueeze(-1), 1  # (L,) -> (L, 1)
        elif y.dim() == 2:
            if y.shape[0] == self.batch_size and self.batch_size > 0:
                # Likely (B, L) with single channel
                return y.unsqueeze(-1), 1
            else:
                # Likely (L, C)
                return y, y.shape[-1]
        elif y.dim() == 3:
            return y, y.shape[-1]  # (B, L, C)
        return y, y.shape[-1] if y.dim() > 1 else 1
    
    def _get_seasonal_mean(self, t: torch.Tensor) -> torch.Tensor:
        """
        Map time points to their seasonal mean values using the stored profile.
        
        Uses the time grid spacing from training data to determine the seasonal
        phase of each time point, then looks up the corresponding seasonal value.
        
        Args:
            t: Time tensor of shape (L,) or (B, L) (already normalized, no trailing dim).
            
        Returns:
            Seasonal mean at each time point: (L, C) or (B, L, C).
        """
        sl = self.season_length
        if t.dim() == 1:
            # Unbatched
            indices = torch.round((t - self._t0) / self._dt).long()
            phases = indices % sl
            return self.y_seasonal[phases]  # (L, C) indexed from (sl, C)
        else:
            # Batched: t is (B, L), self._t0 is (B,)
            indices = torch.round(
                (t - self._t0.unsqueeze(-1)) / self._dt
            ).long()
            phases = indices % sl  # (B, L)
            # Gather from y_seasonal: (B, sl, C) -> (B, L, C)
            B, L = phases.shape
            C = self.y_seasonal.shape[-1]
            phases_expanded = phases.unsqueeze(-1).expand(B, L, C)
            return torch.gather(self.y_seasonal, 1, phases_expanded)  # (B, L, C)
    
    def _get_mean_correction(self, t_test: torch.Tensor) -> torch.Tensor | None:
        """
        Get the mean function value at test time points.
        
        Returns the seasonal profile at the test points if a seasonal mean was
        fitted, otherwise the simple global mean, or None if no mean function.
        
        Args:
            t_test: Normalized test time tensor of shape (M,) or (B, M).
            
        Returns:
            Mean correction tensor or None.
        """
        if self.y_seasonal is not None:
            t = t_test
            # If t_test is unbatched but training data was batched, expand
            if t.dim() == 1 and self.t_train is not None and self.t_train.dim() == 2:
                t = t.unsqueeze(0).expand(self.batch_size, -1)
            return self._get_seasonal_mean(t)
        return self.y_mean
    
    def _compute_kernel(
        self,
        t1: torch.Tensor,
        t2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the kernel matrix between two sets of time points.
        
        Args:
            t1: First set of time points, shape (n,) or (batch, n).
            t2: Second set of time points, shape (m,) or (batch, m).
            
        Returns:
            Kernel matrix of shape (n, m) or (batch, n, m).
        """
        # Handle batched inputs
        if t1.dim() == 1 and t2.dim() == 1:
            diff = t1.unsqueeze(-1) - t2.unsqueeze(0)  # (n, m)
        else:
            diff = t1.unsqueeze(-1) - t2.unsqueeze(-2)  # (batch, n, m)
        
        if self.kernel == Prior.ISO:
            K = self.gamma * (diff.abs() < 1e-8).float()
        elif self.kernel == Prior.SE:
            K = torch.exp(-self.gamma * diff ** 2)
        elif self.kernel == Prior.OU:
            K = torch.exp(-self.gamma * diff.abs())
        elif self.kernel == Prior.PE:
            if self.period_hours is None:
                K = torch.exp(-self.gamma * torch.sin(diff) ** 2)
            else:
                K = torch.exp(-self.gamma * torch.sin(torch.pi * diff / self.period_hours) ** 2)
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")
        
        return K
    
    def fit(
        self,
        t_train: torch.Tensor,
        y_train: torch.Tensor,
    ) -> "GPRegressor":
        """
        Fit the GP to observed data.
        
        Args:
            t_train: Training time points, shape (B, L, 1), (B, L), (L, 1), or (L,).
            y_train: Training observations, shape (B, L, C), (B, L), (L, C), or (L,).
            
        Returns:
            self (for chaining).
        """
        # Normalize inputs
        t_train = self._normalize_t(t_train)
        
        # Determine batch size
        if t_train.dim() == 2:
            self.batch_size = t_train.shape[0]
        else:
            self.batch_size = 0
        
        # Normalize y and get channel count
        y_train, self.n_channels = self._normalize_y(y_train)
        
        self.t_train = t_train
        self.y_train = y_train
        
        # Compute mean function (seasonal or simple global mean)
        self.y_seasonal = None
        if self.use_data_mean:
            used_seasonal = False
            season_length = self.season_length
            if self.period_hours is not None:
                if t_train.dim() == 1 and t_train.shape[0] > 1:
                    dt_hours = float((t_train[1] - t_train[0]).item())
                elif t_train.dim() == 2 and t_train.shape[1] > 1:
                    dt_hours = float((t_train[0, 1] - t_train[0, 0]).item())
                else:
                    dt_hours = None
                if dt_hours is not None and dt_hours > 0:
                    season_length = max(1, int(round(self.period_hours / dt_hours)))
            self.season_length = season_length
            if season_length is not None:
                L_time = y_train.shape[-2]
                sl = season_length
                n_complete = L_time // sl
                if n_complete >= 1:
                    # Store time grid info for phase mapping
                    if t_train.dim() == 1:
                        self._t0 = t_train[0]
                        self._dt = (t_train[-1] - t_train[0]) / max(t_train.shape[0] - 1, 1)
                    else:
                        self._t0 = t_train[:, 0]  # (B,)
                        self._dt = (
                            (t_train[0, 1] - t_train[0, 0])
                            if t_train.shape[1] > 1
                            else torch.ones(1, device=t_train.device, dtype=t_train.dtype)
                        )
                    # Compute per-phase seasonal profile from complete periods
                    y_crop = y_train[..., :n_complete * sl, :]
                    if y_crop.dim() == 2:
                        self.y_seasonal = y_crop.reshape(n_complete, sl, -1).mean(0)  # (sl, C)
                    else:
                        self.y_seasonal = y_crop.reshape(
                            self.batch_size, n_complete, sl, -1
                        ).mean(1)  # (B, sl, C)
                    self.y_mean = None
                    used_seasonal = True
            if not used_seasonal:
                # Simple global mean (original behaviour)
                if y_train.dim() == 2:
                    self.y_mean = y_train.mean(dim=0, keepdim=True)  # (1, C)
                else:
                    self.y_mean = y_train.mean(dim=1, keepdim=True)  # (B, 1, C)
        else:
            self.y_mean = None
        
        # Center the training data using the chosen mean function
        mean_corr = self._get_mean_correction(t_train)
        y_centered = y_train - mean_corr if mean_corr is not None else y_train
        
        # Build or reuse kernel matrices for the current grid
        key = (
            _make_kernel_key(
                self.kernel,
                self.gamma,
                self.noise,
                self.jitter,
                self.season_length,
                self.period_hours,
                t_train,
            )
            if _should_cache_kernel_matrices(t_train)
            else None
        )
        cached = _kernel_cache.get(key) if key is not None else None
        if cached is not None:
            device = t_train.device
            self.K_train, self.K_train_noisy, self.K_inv = (
                cached_tensor.to(device) for cached_tensor in cached
            )
        else:
            # Compute kernel matrix for training points
            K = self._compute_kernel(t_train, t_train)

            # Add noise to diagonal
            if t_train.dim() == 1:
                eye = torch.eye(K.size(0), device=K.device, dtype=K.dtype)
            else:
                batch_size, n = t_train.shape
                eye = torch.eye(n, device=K.device, dtype=K.dtype).unsqueeze(0)

            # Store clean and noisy kernels separately so we can reuse them during prediction
            self.K_train = K
            K_noisy = K + (self.noise + self.jitter) * eye
            self.K_train_noisy = K_noisy

            # Compute inverse using Cholesky for numerical stability
            # Upcast to float32 — cholesky/inv are not implemented for bf16
            K_noisy_f32 = K_noisy.float()
            try:
                L = torch.linalg.cholesky(K_noisy_f32)
                self.K_inv = torch.cholesky_inverse(L).to(K_noisy.dtype)
            except RuntimeError:
                if K_noisy_f32.dim() == 2:
                    K_noisy_f32 = K_noisy_f32 + 1e-4 * torch.eye(K_noisy_f32.size(-1), device=K_noisy_f32.device)
                else:
                    K_noisy_f32 = K_noisy_f32 + 1e-4 * torch.eye(K_noisy_f32.size(-1), device=K_noisy_f32.device).unsqueeze(0)
                self.K_train_noisy = K_noisy_f32.to(K_noisy.dtype)
                self.K_inv = torch.linalg.inv(K_noisy_f32).to(K_noisy.dtype)

            if key is not None:
                _kernel_cache[key] = (self.K_train.cpu(), self.K_train_noisy.cpu(), self.K_inv.cpu())
                while len(_kernel_cache) > _KERNEL_CACHE_SIZE:
                    _kernel_cache.popitem(last=False)
        
        # Precompute alpha = K_inv @ (y - mean) for each channel
        # y_centered shape: (L, C) or (B, L, C)
        # K_inv shape: (L, L) or (B, L, L)
        if t_train.dim() == 1:
            # Unbatched: K_inv (L, L), y_centered (L, C)
            self.alpha = self.K_inv @ y_centered  # (L, C)
        else:
            # Batched: K_inv (B, L, L), y_centered (B, L, C)
            self.alpha = torch.bmm(self.K_inv, y_centered)  # (B, L, C)
        
        return self
    
    def predict(
        self,
        t_test: torch.Tensor,
        return_std: bool = False,
        return_cov: bool = False,
    ) -> tuple[torch.Tensor, ...] | torch.Tensor:
        """
        Predict at arbitrary test time points.
        
        Args:
            t_test: Test time points, shape (B, M, 1), (B, M), (M, 1), or (M,).
            return_std: If True, return standard deviations.
            return_cov: If True, return full covariance matrix (per channel).
            
        Returns:
            mean: Predicted means, shape (B, M, C) or (M, C).
            std (optional): Predicted standard deviations, shape (B, M, C) or (M, C).
            cov (optional): Predicted covariance, shape (B, C, M, M) or (C, M, M).
        """
        if self.t_train is None:
            raise RuntimeError("Must call fit() before predict()")
        
        # Normalize test times
        t_test = self._normalize_t(t_test)

        # Cross-covariance between test and train points
        # If we are asked to predict on the training grid, reuse the noisy kernel we inverted
        # so the predictive mean interpolates the training targets.
        if self._is_train_query(t_test):
            K_star = self.K_train_noisy  # (L, L) or (B, L, L)
        else:
            # Vectorized pointwise match: find test times that coincide with train times
            match_mask, train_idx = self._match_train_indices(t_test)
            K_cross = self._compute_kernel(t_test, self.t_train)  # (M, L) or (B, M, L)

            if match_mask.any():
                # Build K_star by substituting rows from K_train_noisy where matched
                if t_test.dim() == 1:
                    # Unbatched: K_cross (M, L), K_train_noisy (L, L)
                    interp_rows = self.K_train_noisy[train_idx]  # (M, L) indexed by train_idx
                    K_star = torch.where(match_mask.unsqueeze(-1), interp_rows, K_cross)
                else:
                    # Batched: K_cross (B, M, L), K_train_noisy (B, L, L)
                    B, M = t_test.shape
                    L = self.t_train.shape[-1]
                    # Gather rows: for each (b, m), pick K_train_noisy[b, train_idx[b,m], :]
                    batch_idx = torch.arange(B, device=t_test.device).unsqueeze(-1).expand(B, M)
                    interp_rows = self.K_train_noisy[batch_idx, train_idx]  # (B, M, L)
                    K_star = torch.where(match_mask.unsqueeze(-1), interp_rows, K_cross)
            else:
                K_star = K_cross
        
        # Predictive mean: K_star @ alpha + mean_function
        if t_test.dim() == 1 and self.t_train.dim() == 1:
            # Unbatched: K_star (M, L), alpha (L, C)
            mean = K_star @ self.alpha  # (M, C)
        elif t_test.dim() == 1 and self.t_train.dim() == 2:
            # Test unbatched, train batched: expand test
            K_star_expanded = self._compute_kernel(
                t_test.unsqueeze(0).expand(self.batch_size, -1),
                self.t_train
            )  # (B, M, L)
            mean = torch.bmm(K_star_expanded, self.alpha)  # (B, M, C)
        else:
            # Both batched: K_star (B, M, L), alpha (B, L, C)
            mean = torch.bmm(K_star, self.alpha)  # (B, M, C)
        
        # Add back the mean function (seasonal or simple)
        mean_corr = self._get_mean_correction(t_test)
        if mean_corr is not None:
            mean = mean + mean_corr
        
        results = [mean]
        
        if return_std or return_cov:
            # Self-covariance at test points
            if self._is_train_query(t_test) and self.K_train is not None:
                K_star_star = self.K_train  # Use clean kernel at training grid
            else:
                K_star_star = self._compute_kernel(t_test, t_test)  # (M, M) or (B, M, M)
            
            # Predictive covariance: K_star_star - K_star @ K_inv @ K_star.T
            if t_test.dim() == 1 and self.t_train.dim() == 1:
                cov = K_star_star - K_star @ self.K_inv @ K_star.T  # (M, M)
            elif t_test.dim() == 1 and self.t_train.dim() == 2:
                K_star_expanded = self._compute_kernel(
                    t_test.unsqueeze(0).expand(self.batch_size, -1),
                    self.t_train
                )
                K_star_star_expanded = self._compute_kernel(
                    t_test.unsqueeze(0).expand(self.batch_size, -1),
                    t_test.unsqueeze(0).expand(self.batch_size, -1)
                )
                cov = K_star_star_expanded - torch.bmm(
                    K_star_expanded,
                    torch.bmm(self.K_inv, K_star_expanded.transpose(-1, -2))
                )  # (B, M, M)
            else:
                cov = K_star_star - torch.bmm(
                    K_star,
                    torch.bmm(self.K_inv, K_star.transpose(-1, -2))
                )  # (B, M, M)
            
            # Ensure positive semi-definiteness with stronger regularization
            jitter = 1e-4
            if cov.dim() == 2:
                cov = cov + jitter * torch.eye(cov.size(-1), device=cov.device, dtype=cov.dtype)
            else:
                cov = cov + jitter * torch.eye(cov.size(-1), device=cov.device, dtype=cov.dtype).unsqueeze(0)
            
            # Clamp diagonal to be positive and symmetrize
            cov = 0.5 * (cov + cov.transpose(-1, -2))  # Ensure symmetry
            
            if return_std:
                # Diagonal of covariance gives variance per time point
                # Same std for all channels (shared kernel)
                var = torch.diagonal(cov, dim1=-2, dim2=-1)  # (M,) or (B, M)
                std = torch.sqrt(torch.clamp(var, min=1e-8))
                # Expand to match channel dimension
                if std.dim() == 1:
                    std = std.unsqueeze(-1).expand(-1, self.n_channels)  # (M, C)
                else:
                    std = std.unsqueeze(-1).expand(-1, -1, self.n_channels)  # (B, M, C)
                results.append(std)
            
            if return_cov:
                # Expand cov to per-channel: (M, M) -> (C, M, M) or (B, M, M) -> (B, C, M, M)
                if cov.dim() == 2:
                    cov = cov.unsqueeze(0).expand(self.n_channels, -1, -1)  # (C, M, M)
                else:
                    cov = cov.unsqueeze(1).expand(-1, self.n_channels, -1, -1)  # (B, C, M, M)
                results.append(cov)
        
        return tuple(results) if len(results) > 1 else results[0]
    
    def posterior(
        self,
        t_test: torch.Tensor,
        channel: int = 0,
    ) -> MV:
        """
        Return the full posterior distribution at test points for a specific channel.
        
        Args:
            t_test: Test time points, shape (B, M, 1), (B, M), (M, 1), or (M,).
            channel: Which channel to return the distribution for.
            
        Returns:
            MultivariateNormal distribution over test points.
        """
        mean, cov = self.predict(t_test, return_cov=True)
        
        # Extract specific channel
        if mean.dim() == 2:
            mean_ch = mean[:, channel]
            cov_ch = cov[channel]
        else:
            mean_ch = mean[:, :, channel]
            cov_ch = cov[:, channel]
        
        # Ensure positive definiteness and compute Cholesky
        cov_ch = self._make_psd(cov_ch)
        scale_tril = self._safe_cholesky(cov_ch)
        
        return MV(mean_ch, scale_tril=scale_tril)
    
    def sample(
        self,
        t_test: torch.Tensor,
        num_samples: int = 1,
    ) -> torch.Tensor:
        """
        Draw samples from the posterior at test points for all channels simultaneously.
        
        Since all channels share the same kernel, they share the same covariance structure.
        This method samples all channels efficiently in one go.
        
        Args:
            t_test: Test time points.
            num_samples: Number of samples to draw.
            
        Returns:
            Samples of shape (num_samples, M, C) or (num_samples, B, M, C).
        """
        mean, cov = self.predict(t_test, return_cov=True)
        
        # cov shape: (C, M, M) or (B, C, M, M)
        # Since all channels share the same kernel, all channel covariances are identical.
        # Extract just the first channel's covariance for efficiency.
        if cov.dim() == 3:
            cov_single = cov[0]  # (M, M)
        else:
            cov_single = cov[:, 0, :, :]  # (B, M, M)
        
        # Ensure covariance is positive definite
        cov_single = self._make_psd(cov_single)
        
        # Compute Cholesky decomposition with jitter fallback
        L = self._safe_cholesky(cov_single)
        
        # Generate samples from standard normal and apply Cholesky
        samples = self._sample_from_cholesky(mean, L, num_samples)
        
        return samples
    
    def _safe_cholesky(self, mat: torch.Tensor) -> torch.Tensor:
        """
        Compute Cholesky decomposition with jitter fallback for numerical stability.
        
        Args:
            mat: Matrix to decompose, shape (M, M), (C, M, M), or (B, C, M, M).
            
        Returns:
            Lower triangular Cholesky factor.
        """
        orig_dtype = mat.dtype
        mat = mat.float()  # cholesky requires float32+
        eye = self._create_eye(mat)
        
        for jitter in (1e-5, 1e-4, 1e-3):
            try:
                return torch.linalg.cholesky(mat + jitter * eye).to(orig_dtype)
            except RuntimeError:
                continue
        # Last resort
        return torch.linalg.cholesky(mat + 1e-2 * eye).to(orig_dtype)
    
    def _create_eye(self, mat: torch.Tensor) -> torch.Tensor:
        """
        Create identity matrix with same shape as mat (except last 2 dimensions).
        
        Args:
            mat: Reference matrix for shape and device.
            
        Returns:
            Identity matrix with appropriate shape.
        """
        n = mat.size(-1)
        device = mat.device
        dtype = mat.dtype
        
        if mat.dim() == 2:
            return torch.eye(n, device=device, dtype=dtype)
        elif mat.dim() == 3:
            return torch.eye(n, device=device, dtype=dtype).unsqueeze(0)
        else:
            return torch.eye(n, device=device, dtype=dtype).unsqueeze(0).unsqueeze(0)
    
    def _sample_from_cholesky(
        self,
        mean: torch.Tensor,
        L: torch.Tensor,
        num_samples: int,
    ) -> torch.Tensor:
        """
        Generate samples using Cholesky decomposition.
        
        Since all channels share the same covariance, L is the Cholesky factor of
        the shared covariance (not per-channel). We sample from N(0, LL^T) and
        add the channel-specific means.
        
        Args:
            mean: Mean of distribution, shape (M, C) or (B, M, C).
            L: Cholesky factor, shape (M, M) or (B, M, M).
            num_samples: Number of samples to draw.
            
        Returns:
            Samples of shape (num_samples, M, C) or (num_samples, B, M, C).
        """
        if mean.dim() == 2:
            return self._sample_unbatched(mean, L, num_samples)
        else:
            return self._sample_batched(mean, L, num_samples)
    
    def _sample_unbatched(
        self,
        mean: torch.Tensor,
        L: torch.Tensor,
        num_samples: int,
    ) -> torch.Tensor:
        """
        Generate samples for unbatched case.
        
        All channels share the same covariance, so we sample once and broadcast.
        
        Args:
            mean: Mean of shape (M, C).
            L: Cholesky factor of shape (M, M) - shared across channels.
            num_samples: Number of samples.
            
        Returns:
            Samples of shape (num_samples, M, C).
        """
        M, C = mean.shape
        # Sample z for all channels: (num_samples, M, C)
        z = torch.randn(num_samples, M, C, device=mean.device, dtype=mean.dtype)
        # Apply Cholesky: L @ z for each channel
        # L: (M, M), z: (num_samples, M, C)
        # We need (L @ z_c) for each channel c, which is L @ z reshaped
        # torch.matmul broadcasts: (M, M) @ (num_samples, M, C) doesn't work directly
        # Use einsum: samples[s, m, c] = sum_k L[m, k] * z[s, k, c]
        samples = torch.einsum('mk,skc->smc', L, z)
        samples = samples + mean.unsqueeze(0)
        return samples
    
    def _sample_batched(
        self,
        mean: torch.Tensor,
        L: torch.Tensor,
        num_samples: int,
    ) -> torch.Tensor:
        """
        Generate samples for batched case.
        
        All channels share the same covariance per batch element.
        
        Args:
            mean: Mean of shape (B, M, C).
            L: Cholesky factor of shape (B, M, M) - shared across channels per batch.
            num_samples: Number of samples.
            
        Returns:
            Samples of shape (num_samples, B, M, C).
        """
        B, M, C = mean.shape
        # Sample z for all batch elements and channels: (num_samples, B, M, C)
        z = torch.randn(num_samples, B, M, C, device=mean.device, dtype=mean.dtype)
        # Apply Cholesky: L[b] @ z[s, b, :, c] for each batch b and channel c
        # Use einsum: samples[s, b, m, c] = sum_k L[b, m, k] * z[s, b, k, c]
        samples = torch.einsum('bmk,sbkc->sbmc', L, z)
        samples = samples + mean.unsqueeze(0)
        return samples


class Q0Linear(torch.nn.Module):
    def __init__(
        self,
        context_freqs,
        prediction_length,
        freq=24,
        **kwargs,
    ):
        super().__init__()
        context_length = context_freqs * prediction_length
        self.context_length = context_length
        self.freq = freq
        self.mean_linear = torch.nn.Sequential(
            torch.nn.Linear(context_length + 2 * freq, prediction_length),
            torch.nn.ReLU(),
            torch.nn.Linear(prediction_length, prediction_length),
            torch.nn.ReLU(),
            torch.nn.Linear(prediction_length, prediction_length),
        )
        self.cov_linear = torch.nn.Sequential(
            torch.nn.Linear(context_length + 2 * freq, prediction_length),
            torch.nn.ReLU(),
            torch.nn.Linear(prediction_length, prediction_length),
            torch.nn.ReLU(),
            torch.nn.Linear(prediction_length, prediction_length),
            torch.nn.Softplus(),
        )

    def regression(self, x, prediction_length):
        print(x.shape)
        loc_freq = x.reshape(x.shape[0], -1, self.freq).mean(1, keepdims=False)
        x = x - loc_freq.repeat(1, self.context_length // self.freq)
        std_freq = (x.reshape(x.shape[0], -1, self.freq).std(1, keepdims=False)) + 1e-4
        x = x / std_freq.repeat(1, self.context_length // self.freq)
        mean = self.mean_linear(torch.cat([x, loc_freq, std_freq], 1)) + loc_freq.repeat(
            1, prediction_length // self.freq
        )
        cov = self.cov_linear(torch.cat([x, loc_freq, std_freq], 1))

        scalar = std_freq.repeat(1, prediction_length // self.freq)
        cov = cov[:, None] * torch.eye(prediction_length, device=x.device) * scalar.unsqueeze(-1)  # + 1e-3 * torch.eye(
        return mean, cov


if __name__ == "__main__":
    a = 3
    q0 = Q0Dist(kernel=Prior.SE, context_freqs=5, prediction_length=5, gamma=1.0)
    breakpoint()
    a = 5
