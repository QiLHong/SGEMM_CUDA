#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/benchmark_results}"
ROUNDS="${ROUNDS:-3}"
REPEAT="${REPEAT:-50}"
WARMUP="${WARMUP:-10}"
SEED="${SEED:-20261010}"
BETA="${BETA:-3}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d-%H%M%S)}"
CSV="${OUTPUT_DIR}/kernel_15_grouped_swizzle_${RUN_TAG}.csv"
LOG="${OUTPUT_DIR}/kernel_15_grouped_swizzle_${RUN_TAG}.log"
GROUP_M_VALUES=(1 2 4 8 16)
MODES=(strict k10 g1 g2 g4 g8 g16)

mkdir -p "${OUTPUT_DIR}"
if [[ -e "${CSV}" || -e "${LOG}" ]]; then
  printf 'Refusing to overwrite existing result: %s or %s\n' "${CSV}" "${LOG}" >&2
  exit 1
fi
: >"${LOG}"

for group in "${GROUP_M_VALUES[@]}"; do
  build_dir="${ROOT_DIR}/build-swizzle-g${group}"
  cmake -S "${ROOT_DIR}" -B "${build_dir}" -G "Unix Makefiles" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_FLAGS_RELEASE="-O3 -DNDEBUG" \
    -DSGEMM_K15_GROUP_M="${group}" | tee -a "${LOG}"
  cmake --build "${build_dir}" --target sgemm --parallel 2 | tee -a "${LOG}"
done

printf '%s\n' \
  'round,position,mode,kernel,group_m,gflops,correctness,temp_before_c,temp_after_c,sm_clock_before_mhz,sm_clock_after_mhz,mem_clock_before_mhz,mem_clock_after_mhz,power_before_w,power_after_w,beta,c_policy,seed' \
  >"${CSV}"

gpu_snapshot() {
  nvidia-smi --query-gpu=temperature.gpu,clocks.current.sm,clocks.current.memory,power.draw \
    --format=csv,noheader,nounits | head -n 1 | tr -d ' '
}

kernel_for_mode() {
  case "$1" in
  strict) printf '13\n' ;;
  k10) printf '10\n' ;;
  g*) printf '15\n' ;;
  *) return 1 ;;
  esac
}

group_for_mode() {
  case "$1" in
  strict | k10) printf '1\n' ;;
  g*) printf '%s\n' "${1#g}" ;;
  *) return 1 ;;
  esac
}

for ((round = 1; round <= ROUNDS; round++)); do
  start=$(((round - 1) % ${#MODES[@]}))
  for ((position = 0; position < ${#MODES[@]}; position++)); do
    index=$(((start + position) % ${#MODES[@]}))
    mode="${MODES[index]}"
    kernel="$(kernel_for_mode "${mode}")"
    group="$(group_for_mode "${mode}")"
    build_dir="${ROOT_DIR}/build-swizzle-g${group}"
    before="$(gpu_snapshot)"
    printf '\n[round=%d position=%d mode=%s kernel=%s group_m=%s]\n' \
      "${round}" "$((position + 1))" "${mode}" "${kernel}" "${group}" | tee -a "${LOG}"
    output="$(env -u NVIDIA_TF32_OVERRIDE DEVICE="${DEVICE:-0}" \
      SGEMM_BENCH_SIZE=4096 SGEMM_WARMUP="${WARMUP}" \
      SGEMM_REPEAT="${REPEAT}" SGEMM_SEED="${SEED}" \
      SGEMM_RESET_C_EACH_REPEAT=1 SGEMM_BETA="${BETA}" \
      "${build_dir}/sgemm" "${kernel}" 2>&1)"
    after="$(gpu_snapshot)"
    printf '%s\n' "${output}" | tee -a "${LOG}"

    gflops="$(awk -F'[()]' '/performance:/ {value=$4} END {gsub(/[^0-9.]/, "", value); print value}' <<<"${output}")"
    correctness="reference"
    if [[ "${kernel}" != 13 ]]; then
      correctness="$(grep -q 'Correctness vs strict FP32 cuBLAS: PASS' <<<"${output}" && printf 'PASS' || printf 'FAIL')"
    fi
    IFS=',' read -r temp_before sm_before mem_before power_before <<<"${before}"
    IFS=',' read -r temp_after sm_after mem_after power_after <<<"${after}"
    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,reset-each-repeat,%s\n' \
      "${round}" "$((position + 1))" "${mode}" "${kernel}" "${group}" \
      "${gflops}" "${correctness}" "${temp_before}" "${temp_after}" \
      "${sm_before}" "${sm_after}" "${mem_before}" "${mem_after}" \
      "${power_before}" "${power_after}" "${BETA}" "${SEED}" >>"${CSV}"
  done
done

printf 'CSV: %s\nLog: %s\n' "${CSV}" "${LOG}"
