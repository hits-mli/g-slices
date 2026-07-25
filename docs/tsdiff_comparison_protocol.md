# DSPD (tsdiff) baseline — fair-comparison protocol

Pre-registered protocol for comparing G-SLiCEs against DSPD (Biloš et al.,
*Modeling Temporal Data as Continuous Functions with Stochastic Process
Diffusion*, ICML 2023) on the irregular-grid and cross-frequency
generalisation experiments. Written before the headline runs; deviations from
this document should be recorded here with a reason.

## 1. Models compared

| Family dir | Model | Generative core | Denoiser | Role |
|---|---|---|---|---|
| `slice` | G-SLiCEs | flow matching, GP-posterior prior | SLiCE | proposed method |
| `tsflow` | TSFlow | flow matching, GP-posterior prior | S4 | prior-work baseline |
| `tsdiff_gp` | DSPD-GP | DSPD, RBF-kernel noise | upstream RNN | published-method baseline (their headline variant) |
| `tsdiff_ou` | DSPD-OU | DSPD, OU-kernel noise | upstream RNN | published-method baseline (prior-matched kernel family) |
| `tsdiff_gauss` | DSPD-Gauss | DDPM, i.i.d. noise | upstream RNN | internal control: isolates the effect of correlated noise |

The DSPD models execute the upstream diffusion code and upstream `RNNModel`
denoiser byte-identical (vendored, `gslice/vendor/tsdiff/`, pinned commit
`1d84a95`, pristineness provable via `tools/vendor_tsdiff.sh` and the pytest
checksum test). A shared-SLiCE-backbone variant
(`diffusion_params.denoiser: gslices`) exists as an optional appendix
ablation isolating the generative core from the architecture; it is not part
of the headline comparison.

## 2. What is held identical across all models

- Data pipeline: same GluonTS datasets, windows, splits, instance transforms,
  irregular grid sampling (same `seed_offset`), `longmean` scaler,
  observation masks, physical time grids.
- Training budget: 400 epochs, batch 64, 128 batches/epoch, EMA
  (0.9999 / 128 / 1), gradient clip 0.5, AdamW weight decay 0, seeds
  6432–6436. (Deviation from upstream's 100 epochs + patience 20 — equal
  budget across models takes precedence; more budget can only help the
  baseline.)
- Evaluation: identical scripts (`submit/evaluate_grid_generalisation_*`),
  100 samples, seed 6432 via `temporary_random_seed`, best checkpoint,
  CRPS/ND/NRMSE, mean ± std over 5 seeds. DSPD sampling always uses the
  trained N=100 ancestral schedule (num_steps mutation is warned and ignored).

## 3. Documented asymmetries (published-method fidelity over symmetry)

The DSPD baseline conditions the way upstream does (CSDI-style channel
concat: noisy window, clean scaled context, observation mask) and therefore
does NOT receive: the GP-posterior mean/sample features, lag features, or a
GP-posterior-anchored x_N (`noise_around: zero`). The published method has
none of these. If a reviewer challenges this, the controlled variants exist
as flags (`noise_around: gp_mean`, `denoiser: gslices`) and can be run as
ablations; they are not the headline.

TSFlow/SLiCE use lags on the subsample family (`use_lags: true`); DSPD does
not. This is a real information asymmetry in TSFlow's favor and must be
stated in the paper.

## 4. Hyperparameters

Adopted from upstream verbatim (unit-free): N=100 diffusion steps,
BetaLinear(1e-4, 20/N = 0.2), `predict_gaussian_noise: true`, RNN denoiser
128 hidden / 2 layers / bidirectional, lr 1e-3, observed-mask loss
weighting.

Noise-kernel scales are the one unit-dependent knob. Upstream used three
different time axes across its experiments (integer indices for forecasting,
globally normalized [0,1] for synthetic, real timestamps for CSDI), so raw
transplantation is meaningless. Protocol: kernels live in physical hours
(`time_normalization: physical`; covariance built on true observation times,
as in upstream's neural-process/CSDI experiments) and the scale is selected
by sweep, where the sweep includes the upstream published values converted
to the physical axis of the 48 h windows.

### 4.1 Irregular family sweep (pre-registered)

On `irregular/ett_15min_k10_24h`, seed 6432, full 400-epoch budget:

- DSPD-GP `gp_sigma` ∈ {2, 4.8, 8} h — 4.8 h = upstream synthetic σ=0.1 × 48 h.
  Adjacent-step correlation at the 2 h nominal spacing: 0.37 / 0.84 / 0.94.
- DSPD-OU `ou_theta` ∈ {0.5, 0.25, 0.125, 0.0104} h⁻¹ — 0.25 ≈ upstream
  forecasting θ=0.5 per step at 2 h spacing; 0.0104 = upstream synthetic
  θ=0.5 / 48 h. Correlations: 0.37 / 0.61 / 0.78 / 0.98.

Selection: lowest test CRPS on the sweep dataset, one value per family, then
frozen for all k ∈ {1, 3, 5, 10, 25, 50, 100, 1000} and all seeds. The
winning k10/seed-6432 run is reused as that cell of the final matrix.
Per-dataset tuning matches the repo's existing `slice_best` precedent.

### 4.2 Cross-frequency family rule

Kernel scale per training variant, by rule: `gp_sigma = 2 × Δt_train`,
`ou_theta = 0.5 / Δt_train` (adjacent-step correlation 0.78 / 0.61 at the
training grid). Constants validated once by a mini-sweep on `ett_1h_native`
(σ ∈ {1, 2, 4} h) before launching the family. The kernel stays fixed in
physical time within each run, so cross-frequency evaluation changes only
the sampling of the same noise process.

## 5. Guardrails (checked on every run log)

- `[TSDiffCond] ... adjacent-step noise correlation` must be within
  ~[0.3, 0.95] for GP/OU headline runs (guards against the upstream
  forecasting failure mode where σ=0.05 on an integer grid made DSPD-GP
  numerically identical to white-noise DDPM). Absent for Gauss runs.
- `alpha_bar_N < 1e-3` asserted at init; Cholesky jitter escalations warned.
- Vendor pristineness + dispatch + state-dict tests green
  (`python -m pytest tests/test_tsdiff_cond.py`).
- DSPD-Gauss is always trained and reported alongside GP/OU.

## 6. Runs matrix

1. Sweep: 7 runs (Section 4.1) → `results/tsdiff_sweep/k10/`.
2. Irregular headline: 8 k-values × {tsdiff_gp, tsdiff_ou, tsdiff_gauss} ×
   seeds 6432–6436 → `results/irregular_generalisation/<family>/k<k>[_s<seed>]`,
   alongside existing `slice`/`tsflow` runs.
3. Cross-frequency: per-variant grids × 3 DSPD families × seeds →
   `results/subsample_generalisation/<family>/<grid>`.
4. Evaluation via the existing wrappers (families are auto-discovered).

## 7. Paper caveats (mandatory text)

(i) Baseline runs the authors' diffusion and denoiser code verbatim
(vendored, pinned, checksummed); conditioning is adapted to univariate
conditional forecasting using their own CSDI channel-concat pattern, because
their forecasting conditioning (multivariate TimeGrad-RNN on pts) is
multivariate-only and dependency-incompatible. (ii) Noise covariance is
built on true observation times — their neural-process/CSDI treatment — 
unlike their published forecasting tables, which used integer index grids
(rendering the GP covariance near-diagonal). (iii) Exact parity with their
published numbers is infeasible (gluonts 0.9/pts, multivariate, different
datasets); the supported reproduction claims are vendored-code identity and
the internal DSPD-GP/OU vs DSPD-Gauss ordering. (iv) Kernel scales were
selected by CRPS on one dataset from a sweep containing the upstream values
unit-converted to physical time; selected values reported. (v) DSPD receives
neither lag features nor GP-posterior conditioning (the published method has
neither); the shared-backbone/anchored variants exist as flags.

## 8. Recorded deviations

- **2026-07-24, GP sweep boundary extension.** Sweep outcome (k10, seed 6432):
  GP σ ∈ {2, 4.8, 8} h → CRPS {0.1314, 0.1294, 0.1285}; OU θ ∈ {0.5, 0.25,
  0.125, 0.0104} h⁻¹ → CRPS {0.1299, 0.1255, 0.1195, 0.1208}. The OU optimum
  is interior (θ=0.125, frozen). The GP optimum fell on the sweep boundary
  (σ=8) with a monotone trend, so a single boundary-extension point σ=16 h
  (adjacent correlation 0.98) was added. σ=16 came back worse (CRPS 0.1370),
  so **σ=8 is an interior optimum** (CRPS 0.1285, corr 0.93) and is frozen; no
  further extension. Measured adjacent-step correlations matched predictions
  within 0.02 everywhere. **Frozen irregular kernels: DSPD-OU θ=0.125,
  DSPD-GP σ=8 h.** OU beats GP on this dataset across CRPS/ND/NRMSE, consistent
  with rough ETT paths favouring the Matérn-1/2 (OU) kernel over the smooth
  RBF (GP) kernel.

## 9. Outcome interpretation (pre-registered)

- DSPD-GP/OU ≈ DSPD-Gauss after tuning → grid-awareness placed only in the
  noise distribution is insufficient; supports the dynamics-based claim.
- DSPD competitive in-distribution but degrading across grids/frequencies
  faster than SLiCE → the transfer claim.
- DSPD-OU ≥ DSPD-GP on rough series (and vice versa on smooth) → kernel
  choice matters; report both.
