#!/usr/bin/env bash

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/benchmark_results}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d-%H%M%S)}"
CSV="${OUTPUT_DIR}/kernel_21_memcheck_${RUN_TAG}.csv"
mkdir -p "${OUTPUT_DIR}"
[[ ! -e "${CSV}" ]] || { printf 'Refusing to overwrite %s\n' "${CSV}" >&2; exit 1; }
printf 'variant,size,result,log\n' >"${CSV}"

for variant in 1 2; do
  for size in 128 256; do
    log="${OUTPUT_DIR}/kernel_21_memcheck_v${variant}_${size}_${RUN_TAG}.log"
    env -u NVIDIA_TF32_OVERRIDE DEVICE="${DEVICE:-0}" SGEMM_BENCH_SIZE="${size}" \
      SGEMM_WARMUP=0 SGEMM_REPEAT=1 SGEMM_SEED="${SEED:-20261012}" \
      SGEMM_RESET_C_EACH_REPEAT=1 SGEMM_BETA=3 \
      compute-sanitizer --tool memcheck --error-exitcode 99 \
      "${ROOT_DIR}/build-k21-v${variant}/sgemm" 21 >"${log}" 2>&1
    result=FAIL
    grep -q 'ERROR SUMMARY: 0 errors' "${log}" && result=PASS
    printf '%s,%s,%s,%s\n' "${variant}" "${size}" "${result}" "${log}" >>"${CSV}"
    [[ "${result}" == PASS ]] || exit 1
  done
done
printf 'CSV: %s\n' "${CSV}"
