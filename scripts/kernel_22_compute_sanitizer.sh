#!/usr/bin/env bash

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/benchmark_results}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d-%H%M%S)}"
CSV="${OUTPUT_DIR}/kernel_22_memcheck_${RUN_TAG}.csv"
mkdir -p "${OUTPUT_DIR}"
[[ ! -e "${CSV}" ]] || { printf 'Refusing to overwrite %s\n' "${CSV}" >&2; exit 1; }
printf 'variant,size,result,log\n' >"${CSV}"

for variant in 1 2 3; do
  for size in 128 256; do
    log="${OUTPUT_DIR}/kernel_22_memcheck_v${variant}_${size}_${RUN_TAG}.log"
    if env -u NVIDIA_TF32_OVERRIDE DEVICE="${DEVICE:-0}" SGEMM_BENCH_SIZE="${size}" \
      SGEMM_WARMUP=0 SGEMM_REPEAT=1 SGEMM_SEED="${SEED:-20261012}" \
      SGEMM_RESET_C_EACH_REPEAT=1 SGEMM_BETA=3 \
      compute-sanitizer --tool memcheck --error-exitcode=99 \
      "${ROOT_DIR}/build-k22-v${variant}/sgemm" 22 >"${log}" 2>&1; then
      result=PASS
    else
      result=FAIL
    fi
    printf '%s,%s,%s,%s\n' "${variant}" "${size}" "${result}" "${log#${ROOT_DIR}/}" >>"${CSV}"
  done
done
fails="$(awk -F, 'NR>1 && $3!="PASS" {n++} END {print n+0}' "${CSV}")"
[[ "${fails}" == 0 ]] || { printf 'Compute Sanitizer failures: %s\n' "${fails}" >&2; exit 1; }
printf 'Compute Sanitizer: 6/6 PASS\nCSV: %s\n' "${CSV}"
