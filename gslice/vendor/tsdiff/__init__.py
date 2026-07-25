"""Vendored diffusion core of tsdiff (Stochastic Process Diffusion).

Bilos et al., "Modeling Temporal Data as Continuous Functions with
Stochastic Process Diffusion", ICML 2023. Source: Morgan Stanley MSML
monorepo, papers/Stochastic_Process_Diffusion/tsdiff/diffusion/, Apache-2.0.
See PROVENANCE.md for the pinned commit and per-file checksums; verify with
tools/vendor_tsdiff.sh. The vendored .py files must never be edited.

This __init__ is ours. The vendored modules import each other through the
absolute package path `tsdiff.diffusion.*`, so the self-contained submodules
are loaded first and aliased into sys.modules under those names, keeping the
vendored files byte-identical.

`continuous_diffusion` (CSPD) requires torchdiffeq/torchsde and is not
imported eagerly; load it explicitly via
`importlib.import_module("gslice.vendor.tsdiff.continuous_diffusion")`.
"""

import sys
import types

from gslice.vendor.tsdiff import beta_scheduler as _beta_scheduler
from gslice.vendor.tsdiff import noise as _noise
from gslice.vendor.tsdiff.utils import feedforward as _feedforward
from gslice.vendor.tsdiff.utils import positional_encoding as _positional_encoding

_tsdiff_pkg = sys.modules.setdefault("tsdiff", types.ModuleType("tsdiff"))
_diffusion_pkg = sys.modules.setdefault("tsdiff.diffusion", types.ModuleType("tsdiff.diffusion"))
sys.modules.setdefault("tsdiff.diffusion.noise", _noise)
sys.modules.setdefault("tsdiff.diffusion.beta_scheduler", _beta_scheduler)
_tsdiff_pkg.diffusion = _diffusion_pkg
_diffusion_pkg.noise = _noise
_diffusion_pkg.beta_scheduler = _beta_scheduler

# The vendored synthetic denoisers do `from tsdiff.utils import PositionalEncoding,
# FeedForward`; upstream's utils/__init__ additionally imports a pts-bound trainer,
# so a minimal stand-in module is registered instead of vendoring it.
_utils_pkg = sys.modules.setdefault("tsdiff.utils", types.ModuleType("tsdiff.utils"))
_utils_pkg.FeedForward = _feedforward.FeedForward
_utils_pkg.PositionalEncoding = _positional_encoding.PositionalEncoding
_tsdiff_pkg.utils = _utils_pkg

from gslice.vendor.tsdiff.beta_scheduler import BetaLinear, get_beta_scheduler
from gslice.vendor.tsdiff.discrete_diffusion import (
    DiscreteDiffusion,
    GaussianDiffusion,
    GPDiffusion,
    OUDiffusion,
)
from gslice.vendor.tsdiff.noise import GaussianProcess, Normal, OrnsteinUhlenbeck, Wiener

__all__ = [
    "BetaLinear",
    "get_beta_scheduler",
    "DiscreteDiffusion",
    "GaussianDiffusion",
    "GPDiffusion",
    "OUDiffusion",
    "GaussianProcess",
    "Normal",
    "OrnsteinUhlenbeck",
    "Wiener",
]
