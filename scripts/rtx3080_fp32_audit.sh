#!/usr/bin/env bash

set -euo pipefail

# 公平比较 legacy cuBLAS、pedantic FP32、显式 TF32 与 CUDA-core K10。
# 每次 GEMM 前恢复同一份 C；D2D 恢复不计入 CUDA event 时间。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build-audit}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/benchmark_results}"
ROUNDS="${ROUNDS:-3}"
REPEAT="${REPEAT:-50}"
WARMUP="${WARMUP:-10}"
SEED="${SEED:-20261010}"
BETA="${BETA:-3}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d-%H%M%S)}"
CSV="${OUTPUT_DIR}/rtx3080_fp32_audit_${RUN_TAG}.csv"
LOG="${OUTPUT_DIR}/rtx3080_fp32_audit_${RUN_TAG}.log"

mkdir -p "${OUTPUT_DIR}"
if [[ -e "${CSV}" || -e "${LOG}" ]]; then
  printf 'Refusing to overwrite existing result: %s or %s\n' "${CSV}" "${LOG}" >&2
  exit 1
fi
printf 'round,position,mode,kernel,beta,gflops,max_abs_vs_strict,elements_over_0_01,correctness,c_policy,seed\n' >"${CSV}"
: >"${LOG}"

MODES=(legacy strict k10 tf32)

kernel_for_mode() {
  case "$1" in
  legacy) printf '0\n' ;;
  strict) printf '13\n' ;;
  k10) printf '10\n' ;;
  tf32) printf '14\n' ;;
  *) return 1 ;;
  esac
}

for ((round = 1; round <= ROUNDS; round++)); do
  start=$(((round - 1) % ${#MODES[@]}))
  for ((position = 0; position < ${#MODES[@]}; position++)); do
    index=$(((start + position) % ${#MODES[@]}))
    mode="${MODES[index]}"
    kernel="$(kernel_for_mode "${mode}")"
    printf '\n[round=%d position=%d mode=%s kernel=%s]\n' \
      "${round}" "$((position + 1))" "${mode}" "${kernel}" | tee -a "${LOG}"
    output="$(env -u NVIDIA_TF32_OVERRIDE DEVICE="${DEVICE:-0}" \
      SGEMM_BENCH_SIZE=4096 SGEMM_WARMUP="${WARMUP}" \
      SGEMM_REPEAT="${REPEAT}" SGEMM_SEED="${SEED}" \
      SGEMM_RESET_C_EACH_REPEAT=1 SGEMM_BETA="${BETA}" \
      "${BUILD_DIR}/sgemm" "${kernel}" 2>&1)"
    printf '%s\n' "${output}" | tee -a "${LOG}"

    gflops="$(awk -F'[()]' '/performance:/ {value=$4} END {gsub(/[^0-9.]/, "", value); print value}' <<<"${output}")"
    diff_line="$(awk '/Difference vs strict FP32:/ {line=$0} END {print line}' <<<"${output}")"
    max_abs="$(sed -n 's/.*max_abs=\([^,]*\).*/\1/p' <<<"${diff_line}")"
    over="$(sed -n 's/.*elements_over_0.01=\([^/]*\).*/\1/p' <<<"${diff_line}")"
    correctness="not-applicable"
    if [[ "${mode}" == "k10" ]]; then
      correctness="$(grep -q 'Correctness vs strict FP32 cuBLAS: PASS' <<<"${output}" && printf 'PASS' || printf 'FAIL')"
    elif [[ "${mode}" == "legacy" && "${max_abs:-}" == "0" ]]; then
      correctness="bit-equal-to-strict-for-this-input"
    elif [[ "${mode}" == "strict" ]]; then
      correctness="reference"
    else
      correctness="different-precision"
    fi
    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,reset-each-repeat,%s\n' \
      "${round}" "$((position + 1))" "${mode}" "${kernel}" "${BETA}" \
      "${gflops}" "${max_abs}" "${over}" "${correctness}" "${SEED}" >>"${CSV}"
  done
done

printf 'CSV: %s\nLog: %s\n' "${CSV}" "${LOG}"
