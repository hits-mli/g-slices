#!/usr/bin/env bash
set -euo pipefail

export PYTHONIOENCODING="utf-8"

RESULTS_ROOT="results"
OUTPUT_DIR="results/cross_frequency_auto_gp_resample"
DEVICE="cuda:0"
NUM_SAMPLES="100"
SEED="6432"
CONDA_ENV="/home/farjales/miniconda3/envs/cde_torch"
DRY_RUN=0
FAIL_FAST=0

while [[ $# -gt 0 ]]; do
    case $1 in
        --results_root)
            RESULTS_ROOT="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --num_samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --conda_env)
            CONDA_ENV="$2"
            shift 2
            ;;
        --dry_run)
            DRY_RUN=1
            shift 1
            ;;
        --fail_fast)
            FAIL_FAST=1
            shift 1
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--results_root <path>] [--output_dir <path>] [--device <device>] [--num_samples <int>] [--seed <int>] [--conda_env <path>] [--dry_run] [--fail_fast]"
            exit 1
            ;;
    esac
done

CONDA_ROOT="${CONDA_ENV%/envs/*}"
CONDA_BIN="${CONDA_ROOT}/bin/conda"
if [[ ! -x "${CONDA_BIN}" ]]; then
    CONDA_BIN="$(command -v conda || true)"
fi
if [[ -z "${CONDA_BIN}" ]]; then
    echo "Could not find conda executable. Checked ${CONDA_ROOT}/bin/conda and PATH."
    exit 1
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

CMD=(
    "${CONDA_BIN}" run --no-capture-output -p "${CONDA_ENV}"
    python execute/evaluate_grid_generalisation_cross_frequency.py
    --results_root "${RESULTS_ROOT}"
    --output_dir "${OUTPUT_DIR}"
    --device "${DEVICE}"
    --num_samples "${NUM_SAMPLES}"
    --seed "${SEED}"
    --allowed_model_types "tsflow,slice,tsdiff_gp,tsdiff_ou,tsdiff_gauss"
    --fine_to_coarse_eval_adapter_override "gp_resample"
    --only_not_finer_eval_inputs
)

if [[ "${DRY_RUN}" -eq 1 ]]; then
    CMD+=(--dry_run)
fi

if [[ "${FAIL_FAST}" -eq 1 ]]; then
    CMD+=(--fail_fast)
fi

echo "=================================="
echo "GP-Resample Coarse-Input Cross-Frequency Eval"
echo "=================================="
echo "Results Root: ${RESULTS_ROOT}"
echo "Output Dir:   ${OUTPUT_DIR}"
echo "Device:       ${DEVICE}"
echo "Num Samples:  ${NUM_SAMPLES}"
echo "Seed:         ${SEED}"
echo "Conda Env:    ${CONDA_ENV}"
echo "Model Types:  tsflow,slice"
echo "Adapter:      gp_resample"
echo "Eval Filter:  equal-or-coarser inputs only"
echo "Dry Run:      ${DRY_RUN}"
echo "Fail Fast:    ${FAIL_FAST}"
echo "=================================="

"${CMD[@]}"
