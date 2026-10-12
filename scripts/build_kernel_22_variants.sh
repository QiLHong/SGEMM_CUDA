#!/usr/bin/env bash

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/benchmark_results}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "${OUTPUT_DIR}"

for variant in 1 2 3; do
  build="${ROOT_DIR}/build-k22-v${variant}"
  log="${OUTPUT_DIR}/kernel_22_build_v${variant}_${RUN_TAG}.log"
  [[ ! -e "${log}" ]] || { printf 'Refusing to overwrite %s\n' "${log}" >&2; exit 1; }
  cmake -S "${ROOT_DIR}" -B "${build}" -DCMAKE_BUILD_TYPE=Release \
    -DSGEMM_K22_VARIANT="${variant}" -DCMAKE_CUDA_FLAGS='--ptxas-options=-v' 2>&1 \
    | tee "${log}"
  cmake --build "${build}" --parallel "${JOBS:-2}" 2>&1 | tee -a "${log}"
done
