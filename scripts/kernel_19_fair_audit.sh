#!/usr/bin/env bash

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/benchmark_results}"
ROUNDS="${ROUNDS:-3}"
REPEAT="${REPEAT:-50}"
WARMUP="${WARMUP:-10}"
SEED="${SEED:-20261011}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d-%H%M%S)}"
CSV="${OUTPUT_DIR}/kernel_19_fair_${RUN_TAG}.csv"
LOG="${OUTPUT_DIR}/kernel_19_fair_${RUN_TAG}.log"
MODES=(strict k10 v1 v2 v3)
mkdir -p "${OUTPUT_DIR}"
[[ ! -e "${CSV}" && ! -e "${LOG}" ]] || { printf 'Refusing to overwrite output\n' >&2; exit 1; }
printf '%s\n' 'round,position,mode,kernel,variant,gflops,kernel_time_ms,correctness,temp_before_c,temp_after_c,sm_clock_before_mhz,sm_clock_after_mhz,mem_clock_before_mhz,mem_clock_after_mhz,beta,c_policy,seed' >"${CSV}"
: >"${LOG}"

snapshot() {
  nvidia-smi --query-gpu=temperature.gpu,clocks.current.sm,clocks.current.memory \
    --format=csv,noheader,nounits | head -n 1 | tr -d ' '
}

for ((round=1; round<=ROUNDS; round++)); do
  start=$(((round - 1) % ${#MODES[@]}))
  for ((position=0; position<${#MODES[@]}; position++)); do
    mode="${MODES[$(((start + position) % ${#MODES[@]}))]}"
    variant=na
    if [[ "${mode}" == strict ]]; then kernel=13; binary="${ROOT_DIR}/build-prefetch-v1/sgemm"
    elif [[ "${mode}" == k10 ]]; then kernel=10; binary="${ROOT_DIR}/build-prefetch-v1/sgemm"
    else kernel=19; variant="${mode#v}"; binary="${ROOT_DIR}/build-prefetch-v${variant}/sgemm"
    fi
    before="$(snapshot)"
    printf '\n[round=%d position=%d mode=%s]\n' "${round}" "$((position+1))" "${mode}" | tee -a "${LOG}"
    output="$(env -u NVIDIA_TF32_OVERRIDE DEVICE="${DEVICE:-0}" \
      SGEMM_BENCH_SIZE=4096 SGEMM_WARMUP="${WARMUP}" SGEMM_REPEAT="${REPEAT}" \
      SGEMM_SEED="${SEED}" SGEMM_RESET_C_EACH_REPEAT=1 SGEMM_BETA=3 \
      "${binary}" "${kernel}" 2>&1)"
    after="$(snapshot)"
    printf '%s\n' "${output}" | tee -a "${LOG}"
    seconds="$(awk -F'[()]' '/Average elapsed time:/ {v=$2} END {gsub(/[^0-9.]/,"",v); print v}' <<<"${output}")"
    gflops="$(awk -F'[()]' '/performance:/ {v=$4} END {gsub(/[^0-9.]/,"",v); print v}' <<<"${output}")"
    milliseconds="$(awk -v s="${seconds}" 'BEGIN {printf "%.4f",s*1000}')"
    correctness=reference
    [[ "${kernel}" == 13 ]] || correctness="$(grep -q 'Correctness vs strict FP32 cuBLAS: PASS' <<<"${output}" && printf PASS || printf FAIL)"
    IFS=',' read -r tb sb mb <<<"${before}"; IFS=',' read -r ta sa ma <<<"${after}"
    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,3,reset-each-repeat,%s\n' \
      "${round}" "$((position+1))" "${mode}" "${kernel}" "${variant}" "${gflops}" \
      "${milliseconds}" "${correctness}" "${tb}" "${ta}" "${sb}" "${sa}" "${mb}" "${ma}" "${SEED}" >>"${CSV}"
  done
done
printf 'CSV: %s\nLog: %s\n' "${CSV}" "${LOG}"
