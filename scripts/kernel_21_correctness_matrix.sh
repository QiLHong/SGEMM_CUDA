#!/usr/bin/env bash

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/benchmark_results}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d-%H%M%S)}"
CSV="${OUTPUT_DIR}/kernel_21_correctness_${RUN_TAG}.csv"
LOG="${OUTPUT_DIR}/kernel_21_correctness_${RUN_TAG}.log"
mkdir -p "${OUTPUT_DIR}"
[[ ! -e "${CSV}" && ! -e "${LOG}" ]] || { printf 'Refusing to overwrite output\n' >&2; exit 1; }
printf 'variant,beta,size,result\n' >"${CSV}"
: >"${LOG}"

for variant in 1 2; do
  binary="${ROOT_DIR}/build-k21-v${variant}/sgemm"
  for beta in 0 1 3; do
    output="$(env -u NVIDIA_TF32_OVERRIDE DEVICE="${DEVICE:-0}" \
      SGEMM_WARMUP=1 SGEMM_REPEAT=1 SGEMM_SEED="${SEED:-20261012}" \
      SGEMM_RESET_C_EACH_REPEAT=1 SGEMM_BETA="${beta}" "${binary}" 21 2>&1)"
    printf '[variant=%s beta=%s]\n%s\n' "${variant}" "${beta}" "${output}" | tee -a "${LOG}"
    awk -v variant="${variant}" -v beta="${beta}" '
      /dimensions\(m=n=k\)/ {gsub(/,/, "", $2); size=$2}
      /Correctness vs strict FP32 cuBLAS:/ {print variant "," beta "," size "," $NF}
    ' <<<"${output}" >>"${CSV}"
  done
done
passes="$(awk -F, 'NR>1 && $4=="PASS" {n++} END {print n+0}' "${CSV}")"
[[ "${passes}" == 36 ]] || { printf 'Correctness failed: %s/36 PASS\n' "${passes}" >&2; exit 1; }
printf 'Correctness: %s/36 PASS\nCSV: %s\nLog: %s\n' "${passes}" "${CSV}" "${LOG}"
