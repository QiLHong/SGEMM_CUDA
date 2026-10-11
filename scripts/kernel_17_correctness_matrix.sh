#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build-k17}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/benchmark_results}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d-%H%M%S)}"
LOG="${OUTPUT_DIR}/kernel_17_correctness_${RUN_TAG}.log"

mkdir -p "${OUTPUT_DIR}"
if [[ -e "${LOG}" ]]; then
  printf 'Refusing to overwrite existing result: %s\n' "${LOG}" >&2
  exit 1
fi
: >"${LOG}"

for beta in 0 1 3; do
  printf '\n[kernel=17 beta=%s]\n' "${beta}" | tee -a "${LOG}"
  env -u NVIDIA_TF32_OVERRIDE DEVICE="${DEVICE:-0}" \
    SGEMM_WARMUP=1 SGEMM_REPEAT=1 SGEMM_SEED="${SEED:-20261011}" \
    SGEMM_RESET_C_EACH_REPEAT=1 SGEMM_BETA="${beta}" \
    "${BUILD_DIR}/sgemm" 17 | tee -a "${LOG}"
done

printf 'Log: %s\n' "${LOG}"
