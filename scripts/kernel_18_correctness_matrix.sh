#!/usr/bin/env bash

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/benchmark_results}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d-%H%M%S)}"
CSV="${OUTPUT_DIR}/kernel_18_correctness_${RUN_TAG}.csv"
LOG="${OUTPUT_DIR}/kernel_18_correctness_${RUN_TAG}.log"
mkdir -p "${OUTPUT_DIR}"
if [[ -e "${CSV}" || -e "${LOG}" ]]; then
  printf 'Refusing to overwrite %s or %s\n' "${CSV}" "${LOG}" >&2
  exit 1
fi
printf 'variant,beta,size,result\n' >"${CSV}"
: >"${LOG}"

for variant in 0 1 2 3 4 5; do
  binary="${ROOT_DIR}/build-layout-v${variant}/sgemm"
  for beta in 0 1 3; do
    printf '\n[variant=%s beta=%s]\n' "${variant}" "${beta}" | tee -a "${LOG}"
    output="$(env -u NVIDIA_TF32_OVERRIDE DEVICE="${DEVICE:-0}" \
      SGEMM_WARMUP=1 SGEMM_REPEAT=1 SGEMM_SEED="${SEED:-20261011}" \
      SGEMM_RESET_C_EACH_REPEAT=1 SGEMM_BETA="${beta}" \
      "${binary}" 18 2>&1)"
    printf '%s\n' "${output}" | tee -a "${LOG}"
    awk -v variant="${variant}" -v beta="${beta}" '
      /dimensions\(m=n=k\)/ {gsub(/,/, "", $2); size=$2}
      /Correctness vs strict FP32 cuBLAS:/ {print variant "," beta "," size "," $NF}
    ' <<<"${output}" >>"${CSV}"
  done
done
printf 'CSV: %s\nLog: %s\n' "${CSV}" "${LOG}"
