#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/benchmark_results}"
ROUNDS="${ROUNDS:-3}"
REPEAT="${REPEAT:-50}"
WARMUP="${WARMUP:-10}"
SEED="${SEED:-20261011}"
BETA="${BETA:-3}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d-%H%M%S)}"
CSV="${OUTPUT_DIR}/kernel_16_17_fair_${RUN_TAG}.csv"
LOG="${OUTPUT_DIR}/kernel_16_17_fair_${RUN_TAG}.log"
K16_BUILD="${K16_BUILD:-${ROOT_DIR}/build-xor-m1-s3}"
K17_BUILD="${K17_BUILD:-${ROOT_DIR}/build-k17}"
MODES=(strict k10 k16 k17)

mkdir -p "${OUTPUT_DIR}"
if [[ -e "${CSV}" || -e "${LOG}" ]]; then
  printf 'Refusing to overwrite existing result: %s or %s\n' "${CSV}" "${LOG}" >&2
  exit 1
fi
: >"${LOG}"
printf '%s\n' \
  'round,position,mode,kernel,gflops,kernel_time_ms,correctness,temp_before_c,temp_after_c,sm_clock_before_mhz,sm_clock_after_mhz,mem_clock_before_mhz,mem_clock_after_mhz,beta,c_policy,seed' \
  >"${CSV}"

gpu_snapshot() {
  nvidia-smi --query-gpu=temperature.gpu,clocks.current.sm,clocks.current.memory \
    --format=csv,noheader,nounits | head -n 1 | tr -d ' '
}

for ((round = 1; round <= ROUNDS; round++)); do
  start=$(((round - 1) % ${#MODES[@]}))
  for ((position = 0; position < ${#MODES[@]}; position++)); do
    index=$(((start + position) % ${#MODES[@]}))
    mode="${MODES[index]}"
    case "${mode}" in
      strict) kernel=13; binary="${K17_BUILD}/sgemm" ;;
      k10) kernel=10; binary="${K17_BUILD}/sgemm" ;;
      k16) kernel=16; binary="${K16_BUILD}/sgemm" ;;
      k17) kernel=17; binary="${K17_BUILD}/sgemm" ;;
    esac
    before="$(gpu_snapshot)"
    printf '\n[round=%d position=%d mode=%s kernel=%s]\n' \
      "${round}" "$((position + 1))" "${mode}" "${kernel}" | tee -a "${LOG}"
    output="$(env -u NVIDIA_TF32_OVERRIDE DEVICE="${DEVICE:-0}" \
      SGEMM_BENCH_SIZE=4096 SGEMM_WARMUP="${WARMUP}" \
      SGEMM_REPEAT="${REPEAT}" SGEMM_SEED="${SEED}" \
      SGEMM_RESET_C_EACH_REPEAT=1 SGEMM_BETA="${BETA}" \
      "${binary}" "${kernel}" 2>&1)"
    after="$(gpu_snapshot)"
    printf '%s\n' "${output}" | tee -a "${LOG}"
    seconds="$(awk -F'[()]' '/Average elapsed time:/ {value=$2} END {gsub(/[^0-9.]/, "", value); print value}' <<<"${output}")"
    gflops="$(awk -F'[()]' '/performance:/ {value=$4} END {gsub(/[^0-9.]/, "", value); print value}' <<<"${output}")"
    milliseconds="$(awk -v seconds="${seconds}" 'BEGIN {printf "%.4f", seconds * 1000}')"
    correctness=reference
    if [[ "${kernel}" != 13 ]]; then
      correctness="$(grep -q 'Correctness vs strict FP32 cuBLAS: PASS' <<<"${output}" && printf PASS || printf FAIL)"
    fi
    IFS=',' read -r temp_before sm_before mem_before <<<"${before}"
    IFS=',' read -r temp_after sm_after mem_after <<<"${after}"
    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,reset-each-repeat,%s\n' \
      "${round}" "$((position + 1))" "${mode}" "${kernel}" "${gflops}" \
      "${milliseconds}" "${correctness}" "${temp_before}" "${temp_after}" \
      "${sm_before}" "${sm_after}" "${mem_before}" "${mem_after}" \
      "${BETA}" "${SEED}" >>"${CSV}"
  done
done

printf 'CSV: %s\nLog: %s\n' "${CSV}" "${LOG}"
