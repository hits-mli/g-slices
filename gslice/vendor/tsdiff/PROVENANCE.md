# Vendored tsdiff diffusion core and denoisers

- **Upstream**: https://github.com/morganstanley/MSML — `papers/Stochastic_Process_Diffusion/tsdiff/`
- **Paper**: Biloš, Rasul, Schneider, Nevmyvaka, Günnemann. *Modeling Temporal Data as Continuous Functions with Stochastic Process Diffusion.* ICML 2023. (arXiv:2211.02590)
- **Pinned commit**: `1d84a95d2db2a0866a936d23559dd35bd1bbde9a` (2025-08-12)
- **License**: Apache-2.0 (see `LICENSE` in this directory, copied from the upstream paper directory)

The `.py` files below are byte-identical copies of the upstream sources and must never be edited.
All adaptation (device sync, Cholesky caching, jitter escalation, the denoiser adapter) lives in
`gslice/model/tsdiff_cond.py` as subclasses/wrappers. The `__init__.py` files and this file are ours
(upstream's `utils/__init__.py` is deliberately not vendored — it imports a pts-bound trainer).

Verify pristineness against the pinned commit with:

```bash
bash tools/vendor_tsdiff.sh
```

## SHA-256 checksums

| File | Upstream path | SHA-256 |
|---|---|---|
| `beta_scheduler.py` | `tsdiff/diffusion/beta_scheduler.py` | `8b3f34936e047f8a1c56a12b3b39e19a16cdf6107f228ce18cb3a0beea4f2539` |
| `continuous_diffusion.py` | `tsdiff/diffusion/continuous_diffusion.py` | `c61f24459bfd66412f432ed7a95c7369be19b41d6253c32b238dd3cfe059d249` |
| `discrete_diffusion.py` | `tsdiff/diffusion/discrete_diffusion.py` | `f283b36da0f57402271bd93c2bad27c5452e2fecc9e1d93d0c23c0499ac3c10d` |
| `noise.py` | `tsdiff/diffusion/noise.py` | `537efeb2097e22e54c08b1c5fcd64aaf57fd2a90ac117ff83644433c429326b3` |
| `utils/feedforward.py` | `tsdiff/utils/feedforward.py` | `5fbfb5e73fdfa484a662fc458464ffe5fc8f57ae925b578004012a3ab98d087d` |
| `utils/positional_encoding.py` | `tsdiff/utils/positional_encoding.py` | `9bc6bebc535a9dced032d3fd099ea900c8ae695d7833f69802fd2317c404a862` |
| `synthetic/diffusion_model.py` | `tsdiff/synthetic/diffusion_model.py` | `4134299affd8f5457519de35a0fdc963e687229ee0a66421570f4bdb50cbdc98` |
