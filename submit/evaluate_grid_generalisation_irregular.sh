#!/usr/bin/env bash
set -euo pipefail

export PYTHONIOENCODING="utf-8"

RESULTS_ROOT="results/irregular_generalisation"
OUTPUT_DIR="results/irregular_generalisation_eval"
DEVICE="cuda:0"
NUM_SAMPLES="100"
SEED="6432"
ADAPTER_MODE="none"
CHECKPOINT_VARIANT="best"
CONDA_ENV="/home/farjales/miniconda3/envs/cde_torch"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case $1 in
        --results_root) RESULTS_ROOT="$2"; shift 2 ;;
        --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
        --num_samples) NUM_SAMPLES="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --adapter_mode) ADAPTER_MODE="$2"; shift 2 ;;
        --checkpoint_variant) CHECKPOINT_VARIANT="$2"; shift 2 ;;
        --conda_env) CONDA_ENV="$2"; shift 2 ;;
        --dry_run) DRY_RUN=1; shift 1 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

CONDA_ROOT="${CONDA_ENV%/envs/*}"
CONDA_BIN="${CONDA_ROOT}/bin/conda"
if [[ ! -x "${CONDA_BIN}" ]]; then
    CONDA_BIN="$(command -v conda || true)"
fi
if [[ -z "${CONDA_BIN}" ]]; then
    echo "Could not find conda."
    exit 1
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

CMD=(
    "${CONDA_BIN}" run --no-capture-output -p "${CONDA_ENV}"
    python execute/evaluate_grid_generalisation_irregular.py
    --results_root "${RESULTS_ROOT}"
    --output_dir "${OUTPUT_DIR}"
    --device "${DEVICE}"
    --num_samples "${NUM_SAMPLES}"
    --adapter_mode "${ADAPTER_MODE}"
    --checkpoint_variant "${CHECKPOINT_VARIANT}"
)
if [[ -n "${SEED}" ]]; then
    CMD+=(--seed "${SEED}")
fi
if [[ "${DRY_RUN}" -eq 1 ]]; then
    CMD+=(--dry_run)
fi

"${CMD[@]}"
