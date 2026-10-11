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
CSV="${OUTPUT_DIR}/kernel_16_xor_swizzle_${RUN_TAG}.csv"
LOG="${OUTPUT_DIR}/kernel_16_xor_swizzle_${RUN_TAG}.log"
CONFIGS=(m1s1 m1s2 m1s3 m3s1 m3s2 m3s3 m7s1 m7s2 m7s3)
MODES=(strict k10 "${CONFIGS[@]}")

mkdir -p "${OUTPUT_DIR}"
if [[ -e "${CSV}" || -e "${LOG}" ]]; then
  printf 'Refusing to overwrite existing result: %s or %s\n' "${CSV}" "${LOG}" >&2
  exit 1
fi
: >"${LOG}"

for mask in 1 3 7; do
  for shift in 1 2 3; do
    build_dir="${ROOT_DIR}/build-xor-m${mask}-s${shift}"
    cmake -S "${ROOT_DIR}" -B "${build_dir}" -G "Unix Makefiles" \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_FLAGS_RELEASE="-O3 -DNDEBUG" \
      -DSGEMM_K16_XOR_MASK="${mask}" \
      -DSGEMM_K16_XOR_SHIFT="${shift}" | tee -a "${LOG}"
    cmake --build "${build_dir}" --target sgemm --parallel 2 | tee -a "${LOG}"
  done
done

printf '%s\n' \
  'round,position,mode,kernel,mask,shift,gflops,correctness,temp_before_c,temp_after_c,sm_clock_before_mhz,sm_clock_after_mhz,mem_clock_before_mhz,mem_clock_after_mhz,beta,c_policy,seed' \
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
    kernel=16
    mask=1
    shift=1
    if [[ "${mode}" == strict ]]; then
      kernel=13
    elif [[ "${mode}" == k10 ]]; then
      kernel=10
    else
      mask="${mode:1:1}"
      shift="${mode:3:1}"
    fi
    build_dir="${ROOT_DIR}/build-xor-m${mask}-s${shift}"
    before="$(gpu_snapshot)"
    printf '\n[round=%d position=%d mode=%s kernel=%s mask=%s shift=%s]\n' \
      "${round}" "$((position + 1))" "${mode}" "${kernel}" "${mask}" \
      "${shift}" | tee -a "${LOG}"
    output="$(env -u NVIDIA_TF32_OVERRIDE DEVICE="${DEVICE:-0}" \
      SGEMM_BENCH_SIZE=4096 SGEMM_WARMUP="${WARMUP}" \
      SGEMM_REPEAT="${REPEAT}" SGEMM_SEED="${SEED}" \
      SGEMM_RESET_C_EACH_REPEAT=1 SGEMM_BETA="${BETA}" \
      "${build_dir}/sgemm" "${kernel}" 2>&1)"
    after="$(gpu_snapshot)"
    printf '%s\n' "${output}" | tee -a "${LOG}"
    gflops="$(awk -F'[()]' '/performance:/ {value=$4} END {gsub(/[^0-9.]/, "", value); print value}' <<<"${output}")"
    correctness=reference
    if [[ "${kernel}" != 13 ]]; then
      correctness="$(grep -q 'Correctness vs strict FP32 cuBLAS: PASS' <<<"${output}" && printf PASS || printf FAIL)"
    fi
    IFS=',' read -r temp_before sm_before mem_before <<<"${before}"
    IFS=',' read -r temp_after sm_after mem_after <<<"${after}"
    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,reset-each-repeat,%s\n' \
      "${round}" "$((position + 1))" "${mode}" "${kernel}" "${mask}" \
      "${shift}" "${gflops}" "${correctness}" "${temp_before}" \
      "${temp_after}" "${sm_before}" "${sm_after}" "${mem_before}" \
      "${mem_after}" "${BETA}" "${SEED}" >>"${CSV}"
  done
done

printf 'CSV: %s\nLog: %s\n' "${CSV}" "${LOG}"
