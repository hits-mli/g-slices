"""Equivalence tests for the SLiCE fast paths.

1. The fused diagonal-dense projection (one row-concatenated GEMM) must produce
   the same transforms as the three separate projections it replaced.
2. SliceLayer's batched-bidirectional scan must match the sequential
   per-direction path in outputs AND gradients, and must fall back cleanly when
   its guard conditions are not met.
"""
import torch

from gslice.arch.backbones import SliceLayer
from gslice.arch.slices import SLiCE

TOL = dict(rtol=1e-5, atol=1e-6)


def _rel(a, b):
    return ((a - b).norm() / (b.norm() + 1e-12)).item()


def test_fused_projection_matches_separate():
    torch.manual_seed(0)
    sl = SLiCE(input_dim=32, block_size=8, diagonal_dense=True, bound_norm=True)
    inp = torch.randn(4, 10, sl.augmented_input_dim)
    M_diag, M_dense, b_diag, b_dense = sl._build_diagonal_dense_transform(inp)

    bsz = sl.block_size
    hdiag = sl.hidden_dim - bsz
    ref_Md = sl._discretize_diagonal(sl.vf_A_diag(inp))
    ref_Me = sl._discretize_matrix(sl.vf_A_dense(inp).view(4, 10, bsz, bsz))
    ref_B = sl.vf_B(inp)
    assert torch.allclose(M_diag, ref_Md, **TOL)
    assert torch.allclose(M_dense, ref_Me, **TOL)
    assert torch.allclose(b_diag, ref_B[..., :hdiag], **TOL)
    assert torch.allclose(b_dense, ref_B[..., hdiag:], **TOL)


def _make_layer(**kw):
    torch.manual_seed(0)
    params = {"block_size": 8, "diagonal_dense": True, "bound_norm": True}
    params.update(kw)
    return SliceLayer(d_model=32, dropout=0.0, bidirectional=True,
                      slice_block_params=params)


def test_batched_bidirectional_matches_sequential():
    lay = _make_layer()
    assert lay._can_batch_directions(torch.zeros(1, 20, 32))

    x = torch.randn(3, 32, 20, requires_grad=True)  # (B, D, L)
    tg = torch.linspace(0, 1, 20).view(1, 20, 1).expand(3, 20, 1).contiguous()
    w = torch.randn(3, 32, 20)

    out_fast, _ = lay(x, tg)
    (out_fast * w).sum().backward()
    grads_fast = {n: p.grad.clone() for n, p in lay.named_parameters()}
    x_grad_fast = x.grad.clone()
    lay.zero_grad(set_to_none=True)
    x.grad = None

    # force the sequential fallback on the same module/weights
    orig_guard = SliceLayer._can_batch_directions
    SliceLayer._can_batch_directions = lambda self, z: False
    try:
        out_ref, _ = lay(x, tg)
        (out_ref * w).sum().backward()
    finally:
        SliceLayer._can_batch_directions = orig_guard

    assert torch.allclose(out_fast, out_ref, **TOL)
    assert _rel(x_grad_fast, x.grad) < 1e-5
    for n, p in lay.named_parameters():
        if p.grad is not None and n in grads_fast:
            assert _rel(grads_fast[n], p.grad) < 1e-4, n


def test_guard_falls_back_when_unsupported():
    # block_size=1 collapses diagonal_dense -> elementwise path: guard must refuse
    lay = _make_layer(block_size=1)
    assert not lay._can_batch_directions(torch.zeros(1, 20, 32))
    # sequence longer than one chunk: guard must refuse
    lay2 = _make_layer(chunk_size=16)
    assert not lay2._can_batch_directions(torch.zeros(1, 20, 32))
    # both still produce valid output through the fallback
    x = torch.randn(2, 32, 20)
    tg = torch.linspace(0, 1, 20).view(1, 20, 1).expand(2, 20, 1).contiguous()
    for module in (lay, lay2):
        out, _ = module(x, tg)
        assert out.shape == x.shape
        assert torch.isfinite(out).all()
