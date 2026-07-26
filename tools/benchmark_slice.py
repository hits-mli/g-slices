#!/usr/bin/env python
"""Benchmark the SLiCE backbone: certified operator-norm bounds and torch.compile.

Covers the two changes on the ``slice-operator-norm-bounds`` branch:

  1. ``gslice/arch/slices.py``  -- certified operator-norm bounds (Hoelder +
     logarithmic norm) replacing the power-iteration estimate. Power iteration
     converges to the largest singular value from *below*, so it could leave
     expansive transitions unscaled and it was non-deterministic (random restart).
  2. ``bin/train_model.py``     -- opt-in ``model_params['compile']`` that compiles
     the backbone in place (``mode='reduce-overhead'`` on regular grids).

Custom Triton scan kernels were also prototyped during development; they were
*not* adopted because ``torch.compile(mode='reduce-overhead')`` (CUDA graphs)
outperformed them at the backbone level, which is what this benchmark measures.

Run from the repo root:

    python tools/benchmark_slice.py --sections all --model both

The ``bounds`` section runs on CPU; the ``throughput`` section uses CUDA when
available (``reduce-overhead`` / CUDA graphs require a GPU).
"""
import argparse
import statistics
import time

import torch

from gslice.arch.backbones import BackboneModel
from gslice.arch.slices import SLiCE


# --------------------------------------------------------------------------- #
# timing helpers
# --------------------------------------------------------------------------- #
def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def timeit(fn, device: torch.device, warmup: int = 10, reps: int = 50) -> float:
    """Median wall-clock of ``fn`` in milliseconds (with CUDA sync)."""
    for _ in range(warmup):
        fn()
    _sync(device)
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        _sync(device)
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts) * 1e3


# --------------------------------------------------------------------------- #
# section 1: operator-norm bounds  (pairs with slices.py)
# --------------------------------------------------------------------------- #
def _power_iter_bound(M: torch.Tensor, iterations: int = 5, max_norm: float = 1.0) -> torch.Tensor:
    """The *previous* approach: power-iteration estimate of sigma_max, kept only so
    the benchmark can report the cost the Hoelder bound replaced."""
    h = M.shape[-1]
    Mf = M.reshape(-1, h, h)
    v = torch.nn.functional.normalize(
        torch.randn(Mf.shape[0], h, 1, device=M.device, dtype=M.dtype), dim=1
    )
    for _ in range(iterations):
        u = torch.bmm(Mf, v)
        v = torch.nn.functional.normalize(torch.bmm(Mf.transpose(1, 2), u), dim=1)
    sigma = torch.bmm(Mf, v).norm(dim=1, keepdim=True).view(*M.shape[:-2], 1, 1)
    return M / torch.clamp(sigma / max_norm, min=1.0)


def bench_bounds(device: torch.device, block_size: int = 16, n: int = 20000, reps: int = 50) -> None:
    print("== operator-norm bounds (pairs with slices.py) ==")
    torch.manual_seed(0)
    sl = SLiCE(input_dim=64, block_size=block_size, bound_norm=True).to(device)

    # Hoelder bound is a certified upper bound: nothing it leaves unscaled exceeds 1.
    A = torch.randn(n, block_size, block_size, device=device) * 0.3
    M = torch.eye(block_size, device=device) + A
    smax = torch.linalg.svdvals(sl._bound_operator_norm(M))[..., 0].max().item()
    print(f"  Hoelder:  max sigma after bound = {smax:.4f}  "
          f"({'certified <= 1' if smax <= 1 + 1e-5 else 'VIOLATION'})")

    # exact for diagonal matrices, and deterministic (unlike power iteration)
    D = torch.diag_embed(torch.randn(n, block_size, device=device) * 2)
    sig = torch.linalg.svdvals(D)[..., 0]
    D_exact = D / torch.clamp(sig, min=1.0)[..., None, None]
    exact_res = (sl._bound_operator_norm(D) - D_exact).abs().max().item()
    det = (sl._bound_operator_norm(M) - sl._bound_operator_norm(M)).abs().max().item()
    print(f"  Hoelder:  diagonal-exactness residual = {exact_res:.2e}; "
          f"call-to-call diff = {det:.2e} (deterministic)")

    # log-norm shift certifies ||exp(A)||_2 <= 1 for the matrix_exp path
    enorm = torch.linalg.svdvals(torch.matrix_exp(sl._lognorm_shift(A)))[..., 0].max().item()
    print(f"  log-norm: max ||exp(A)|| after shift = {enorm:.4f}  "
          f"({'certified <= 1' if enorm <= 1 + 1e-5 else 'VIOLATION'})")

    # cost: Hoelder vs the previous power iteration on a representative batch
    Mb = torch.eye(block_size, device=device) + 0.05 * torch.randn(
        64 * 60, block_size, block_size, device=device
    )
    t_hold = timeit(lambda: sl._bound_operator_norm(Mb), device, reps=reps)
    t_pow = timeit(lambda: _power_iter_bound(Mb), device, reps=reps)
    print(f"  cost ({block_size}x{block_size} blocks, {Mb.shape[0]} mats): "
          f"Hoelder {t_hold:.3f} ms vs power-iter {t_pow:.3f} ms "
          f"({t_pow / max(t_hold, 1e-9):.1f}x cheaper)")


# --------------------------------------------------------------------------- #
# section 2: backbone throughput  (pairs with the compile flag in train_model.py)
# --------------------------------------------------------------------------- #
SLICE_CFG = dict(
    input_dim=1, hidden_dim=128, output_dim=1, step_emb=64, num_residual_blocks=5,
    num_features=0, target_dim=1, residual_block="slice", dropout=0.0,
    bidirectional=True, init_skip=False, feature_skip=True,
    slice_block_params={"block_size": 16, "diagonal_dense": True, "bound_norm": True},
)
S4_CFG = dict(
    input_dim=1, hidden_dim=64, output_dim=1, step_emb=64, num_residual_blocks=3,
    num_features=0, target_dim=1, residual_block="s4", dropout=0.0,
    bidirectional=True, init_skip=False, feature_skip=True,
)


def _build(cfg: dict, device: torch.device) -> BackboneModel:
    torch.manual_seed(0)
    return BackboneModel(**cfg).to(device)


def bench_throughput(name: str, cfg: dict, device: torch.device, B: int, L: int,
                     reps: int, modes: tuple = ("reduce-overhead", "max-autotune")) -> dict:
    x = torch.randn(B, L, 1, device=device)
    tg = torch.linspace(0, 1, L, device=device).view(1, L, 1).expand(B, L, 1).contiguous()
    t = torch.tensor(0.5, device=device)

    def fwd(m):
        with torch.no_grad():
            m(t, x, None, time_grid=tg)

    def fwd_bwd(m, xg):
        m(t, xg, None, time_grid=tg).sum().backward()
        xg.grad = None
        m.zero_grad(set_to_none=True)

    def measure(make):
        m = make()
        tf = timeit(lambda: fwd(m), device, reps=reps)
        m2 = make()
        xg = x.clone().requires_grad_(True)
        tfb = timeit(lambda: fwd_bwd(m2, xg), device, reps=reps)
        return tf, tfb

    print(f"== {name} throughput (B={B}, L={L}) ==")
    results: dict = {"eager": measure(lambda: _build(cfg, device))}
    for mode in modes:
        if device.type != "cuda" and mode in ("reduce-overhead", "max-autotune"):
            print(f"  {'compile ' + mode:26s} skipped (needs CUDA)")
            continue
        try:
            def make(mode=mode):
                m = _build(cfg, device)
                m.compile(mode=mode)  # in-place, matches bin/train_model.py
                return m
            results[mode] = measure(make)
        except Exception as e:  # e.g. S4 does not compile
            results[mode] = None
            print(f"  {'compile ' + mode:26s} FAIL: {type(e).__name__}")

    for k, v in results.items():
        if v is None:
            continue
        tf, tfb = v
        print(f"  {k:26s} fwd {tf:7.2f} ms ({B / (tf / 1e3):6.0f} seq/s) | "
              f"fwd+bwd {tfb:7.2f} ms ({B / (tfb / 1e3):6.0f} seq/s)")
    base = results.get("eager")
    best = min((v for v in results.values() if v), key=lambda v: v[1], default=None)
    if base and best and best is not base:
        print(f"  best speedup vs eager: fwd {base[0] / best[0]:.2f}x | "
              f"fwd+bwd {base[1] / best[1]:.2f}x")
    return results


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--sections", choices=["bounds", "throughput", "all"], default="all")
    ap.add_argument("--model", choices=["slice", "s4", "both"], default="both")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--length", type=int, default=60)
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--modes", default="reduce-overhead,max-autotune",
                    help="comma-separated torch.compile modes to sweep "
                         "(max-autotune is the training default; drop it for quick runs)")
    args = ap.parse_args()
    device = torch.device(args.device)
    print(f"device={device}; torch={torch.__version__}\n")

    if args.sections in ("bounds", "all"):
        bench_bounds(device, reps=args.reps)
        print()

    if args.sections in ("throughput", "all"):
        res = {}
        modes = tuple(m.strip() for m in args.modes.split(",") if m.strip())
        if args.model in ("slice", "both"):
            res["slice"] = bench_throughput(
                "SLiCE (wiki-like)", SLICE_CFG, device, args.batch, args.length,
                args.reps, modes=modes,
            )
            print()
        if args.model in ("s4", "both"):
            res["s4"] = bench_throughput(
                "S4 (tsflow_s4)", S4_CFG, device, args.batch, args.length,
                args.reps, modes=modes,
            )
            print()
        if "slice" in res and "s4" in res:
            sbest = min((v for v in res["slice"].values() if v), key=lambda v: v[1])
            fbest = min((v for v in res["s4"].values() if v), key=lambda v: v[1])
            print("== head-to-head (best-available each) ==")
            print(f"  S4 vs SLiCE: fwd {sbest[0] / fbest[0]:.2f}x | "
                  f"fwd+bwd {sbest[1] / fbest[1]:.2f}x (>1 => S4 faster)")


if __name__ == "__main__":
    main()
