## Reproducing the G-SLiCE results

This document explains how to reproduce the main paper results. For general setup instructions and environment details, see `README.md`.

### 1) Probabilistic forecasting

The best GluonTS slice configurations live in `configs/model/slice_best`. Each file in that directory is a dataset-specific override on top of the shared GluonTS training setup. Table 1 in the paper is reproduced by running the same model family across the listed datasets and collecting the test metrics from the resulting run directories.

```bash
python bin/train.py -m \
    experiment=gluonts_base \
    dataset=gluonts/electricity_nips \
    model=slice_best/electricity_nips \
    seed=6432,6433,6434,6435,6436 \
    'runtime.logdir=results/slice_best_seed_sweep/electricity_nips/seed_${seed}'
```

Repeat this command for each dataset in `electricity_nips,exchange_rate_nips, kdd_cup_2018_without_missing, m4_hourly, solar_nips, traffic_nips, uber_tlc_hourly, wiki2000_nips`, changing the `dataset={}`, `model=slice_best/{}` and `runtime.logdir={}` fragments.

### 2) Unconditional generation

TBD

### 3) Generalisation across sampling grids

This family of experiments studies whether models trained on one sampling grid transfer to other grids. The two branches below mirror the two generalisation settings in the paper: cross-frequency evaluation on regular subsampling, and irregular-grid evaluation on gamma-sampled time grids.

#### 3.1) Cross-frequency generalisation

For a single TSFlow training run, use the regular subsample experiment and write the outputs into the matching results tree:

```bash
python bin/train.py \
    experiment=subsample \
    dataset=subsample/ett_15min_native_24h \
    model=tsflow_s4_subsample \
    seed=6432 \
    runtime.logdir=results/subsample_generalisation/tsflow/15min_15min
```

For the slice backbone, keep the dataset and seed identical, and switch only the model family:

```bash
python bin/train.py \
    experiment=subsample \
    dataset=subsample/ett_15min_native_24h \
    model=slice_subsample \
    seed=6432 \
    runtime.logdir=results/subsample_generalisation/slice/15min_15min
```

To cover the full cross-frequency sweep, repeat the same pattern for `subsample/ett_15min_native_24h, subsample/ett_15min_hourly_24h, subsample/ett_15min_6h_24h, subsample/ett_12h_hourly_24h`. The important part is that the dataset name and `runtime.logdir` stay aligned, because the evaluation scripts discover runs from the directory structure.

Once the training runs are finished, evaluate all checkpoints on the relevant grids with:

```bash
bash submit/evaluate_grid_generalisation_cross_frequency.sh \
    --results_root results/subsample_generalisation \
    --output_dir results/subsample_generalisation/eval \
    --device cuda:0 \
    --num_samples 100 \
    --seed 6432
```

This writes the aggregated metrics for Table 4 to `results/subsample_generalisation/eval/values.md` and the corresponding comparison plots for reproducing Figure 3 to `results/subsample_generalisation/eval/comparison`.


#### 3.2) Irregular-grid generalisation

The irregular setup follows the same pattern, but now the model is trained on gamma-sampled irregular grids instead of deterministic subsampling. The `k` value in the dataset name controls how concentrated the sampling is.

For a single TSFlow run, the command is:

```bash
python bin/train.py \
    experiment=irregular \
    dataset=irregular/ett_15min_k1_24h \
    model=tsflow_s4_irregular \
    seed=6432 \
    runtime.logdir=results/irregular_generalisation/tsflow/k1
```

For the slice backbone, keep the irregular dataset fixed and switch the model:

```bash
python bin/train.py \
    experiment=irregular \
    dataset=irregular/ett_15min_k1_24h \
    model=slice_irregular \
    seed=6432 \
    runtime.logdir=results/irregular_generalisation/slice/k1
```

To run the other irregular settings, repeat the same command family for the remaining `k` values and keep the `runtime.logdir` in sync with the dataset you launched. The evaluation scripts rely on this structure when they discover checkpoints.

After the runs complete, cross-evaluate the checkpoints on each grid with:

```bash
bash submit/evaluate_grid_generalisation_irregular.sh \
    --results_root results/irregular_generalisation \
    --output_dir results/irregular_generalisation/eval \
    --device cuda:0 \
    --num_samples 100 \
    --seed 6432 \
    --checkpoint_variant best
```

This produces the grid-comparison metrics for Tables 5 and 7 under `results/irregular_generalisation/eval` and uses the best checkpoint per run by default.

To generate the plots in Figures 4 and 5, specify the `k` values you want:

```bash
python execute/plot_irregular_lowest_seed_checkpoints.py \
    --results_root results/irregular_generalisation \
    --ks 1 3 10 25 50 100 1000 \
    --output_dir results/irregular_generalisation/checkpoint_plots \
    --device cuda:0 \
    --num_samples 100 \
    --max_eval_instances 16
```

This plot step is useful when you want a compact visual comparison across seeds and `k` values. It reuses the stored checkpoints, selects the lowest-seed run for each `k`, and writes the publication-style figures into the requested output directory.
