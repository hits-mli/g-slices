# Expressivity Example Reproduction

This directory contains two entry points:

- `expressivity_example/plot_expressivity_gap.py` for the paper-style figures.
- `expressivity_example/quantify_expressivity_gap.py` for the table-style metrics.

## Figures

To reproduce Figure 2 in the main text and Figures 6 and 7 in the appendix, run:

```bash
python expressivity_example/plot_expressivity_gap.py
```

By default, this writes eight files under `figures/`:

- `figures/slices_expressivity_trace_dense_ssm.png`
- `figures/slices_expressivity_trace_dense_ssm.pdf`
- `figures/slices_expressivity_cumsum_hist_dense_ssm.png`
- `figures/slices_expressivity_cumsum_hist_dense_ssm.pdf`
- `figures/slices_expressivity_trace_diagonal_ssm.png`
- `figures/slices_expressivity_trace_diagonal_ssm.pdf`
- `figures/slices_expressivity_cumsum_hist_diagonal_ssm.png`
- `figures/slices_expressivity_cumsum_hist_diagonal_ssm.pdf`


## Tables

To reproduce Table 8, run:

```bash
python expressivity_example/quantify_expressivity_gap.py
```

This prints three tables to stdout and saves three CSV files:

- `figures/hardcore_random_tables_exact_accuracy.csv`
- `figures/hardcore_random_tables_valid_ratio.csv`
- `figures/hardcore_random_tables_mse.csv`

The printed tables are:

- `Table 1: Exact accuracy vs hard-core ground truth`
- `Table 2: Valid-sequence ratio`
- `Table 3: Mean squared error vs hard-core ground truth`

If you want to compare against the paper, start with the defaults and change one flag at a time.
