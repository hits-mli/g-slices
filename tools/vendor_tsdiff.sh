#!/usr/bin/env bash
# Verify (or re-download) the vendored tsdiff diffusion core against the pinned
# upstream commit. Run from the repo root. With --refresh, overwrites the local
# copies instead of diffing (update PROVENANCE.md checksums afterwards).
set -euo pipefail

COMMIT="1d84a95d2db2a0866a936d23559dd35bd1bbde9a"
BASE="https://raw.githubusercontent.com/morganstanley/MSML/${COMMIT}/papers/Stochastic_Process_Diffusion"
VENDOR_DIR="gslice/vendor/tsdiff"
# local path (relative to VENDOR_DIR) -> upstream path (relative to tsdiff/)
FILES=(
    "beta_scheduler.py:diffusion/beta_scheduler.py"
    "discrete_diffusion.py:diffusion/discrete_diffusion.py"
    "continuous_diffusion.py:diffusion/continuous_diffusion.py"
    "noise.py:diffusion/noise.py"
    "utils/feedforward.py:utils/feedforward.py"
    "utils/positional_encoding.py:utils/positional_encoding.py"
    "synthetic/diffusion_model.py:synthetic/diffusion_model.py"
)

refresh=0
[[ "${1:-}" == "--refresh" ]] && refresh=1

status=0
for entry in "${FILES[@]}"; do
    local_path="${entry%%:*}"
    upstream_path="${entry#*:}"
    if [[ "$refresh" == 1 ]]; then
        curl -sf "${BASE}/tsdiff/${upstream_path}" -o "${VENDOR_DIR}/${local_path}"
        echo "refreshed ${local_path}"
    elif curl -sf "${BASE}/tsdiff/${upstream_path}" | diff -q - "${VENDOR_DIR}/${local_path}" > /dev/null; then
        echo "OK ${local_path} (byte-identical to upstream @ ${COMMIT})"
    else
        echo "MISMATCH ${local_path} — vendored copy differs from upstream @ ${COMMIT}" >&2
        status=1
    fi
done
exit "$status"
