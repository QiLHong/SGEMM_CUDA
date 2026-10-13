#!/usr/bin/env bash
# Reproducible K23 correctness, fair timing, memcheck and Nsight runs.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/benchmark_results}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d-%H%M%S)}"
SEED="${SEED:-20261013}"
mkdir -p "${OUTPUT_DIR}"
mode="${1:-}"
case "${mode}" in
  build)
    for variant in 1 2 3; do
      build="${ROOT_DIR}/build-k23-v${variant}"
      cmake -S "${ROOT_DIR}" -B "${build}" -DCMAKE_BUILD_TYPE=Release \
        -DSGEMM_K23_VARIANT="${variant}" -DCMAKE_CUDA_FLAGS='--ptxas-options=-v'
      cmake --build "${build}" --target sgemm --parallel "${JOBS:-2}"
    done
    ;;
  correctness)
    csv="${OUTPUT_DIR}/kernel_23_correctness_${RUN_TAG}.csv"
    [[ ! -e "${csv}" ]] || { printf 'Refusing to overwrite %s\n' "${csv}" >&2; exit 1; }
    printf 'variant,beta,size,result\n' >"${csv}"
    for variant in 2 3; do
      for beta in 0 1 3; do
        output="$(env -u NVIDIA_TF32_OVERRIDE DEVICE="${DEVICE:-0}" \
          SGEMM_WARMUP=1 SGEMM_REPEAT=1 SGEMM_SEED="${SEED}" \
          SGEMM_RESET_C_EACH_REPEAT=1 SGEMM_BETA="${beta}" \
          "${ROOT_DIR}/build-k23-v${variant}/sgemm" 23)"
        awk -v v="${variant}" -v b="${beta}" '
          /dimensions\(m=n=k\)/ {gsub(/,/, "", $2); size=$2}
          /Correctness vs strict FP32 cuBLAS:/ {print v "," b "," size "," $NF}
        ' <<<"${output}" >>"${csv}"
        printf 'V%s beta=%s: %s\n' "${variant}" "${beta}" \
          "$(grep -c 'Correctness vs strict FP32 cuBLAS: PASS' <<<"${output}")"
      done
    done
    [[ "$(awk -F, 'NR>1 && $4=="PASS" {n++} END {print n+0}' "${csv}")" == 36 ]]
    printf 'Correctness 36/36 PASS: %s\n' "${csv}"
    ;;
  memcheck)
    csv="${OUTPUT_DIR}/kernel_23_memcheck_${RUN_TAG}.csv"
    [[ ! -e "${csv}" ]] || { printf 'Refusing to overwrite %s\n' "${csv}" >&2; exit 1; }
    printf 'variant,size,result,log\n' >"${csv}"
    for variant in 2 3; do
      for size in 128 256; do
        log="${OUTPUT_DIR}/kernel_23_memcheck_v${variant}_${size}_${RUN_TAG}.log"
        if env -u NVIDIA_TF32_OVERRIDE DEVICE="${DEVICE:-0}" \
          SGEMM_BENCH_SIZE="${size}" SGEMM_REPEAT=1 SGEMM_WARMUP=0 \
          SGEMM_SEED="${SEED}" SGEMM_RESET_C_EACH_REPEAT=1 SGEMM_BETA=3 \
          compute-sanitizer --tool memcheck --error-exitcode=99 \
          "${ROOT_DIR}/build-k23-v${variant}/sgemm" 23 >"${log}" 2>&1; then
          result=PASS
        else
          result=FAIL
        fi
        printf '%s,%s,%s,%s\n' "${variant}" "${size}" "${result}" "${log#${ROOT_DIR}/}" >>"${csv}"
      done
    done
    [[ "$(awk -F, 'NR>1 && $3!="PASS" {n++} END {print n+0}' "${csv}")" == 0 ]]
    printf 'Memcheck 4/4 PASS: %s\n' "${csv}"
    ;;
  fair)
    csv="${OUTPUT_DIR}/kernel_23_fair_${RUN_TAG}.csv"
    [[ ! -e "${csv}" ]] || { printf 'Refusing to overwrite %s\n' "${csv}" >&2; exit 1; }
    printf '%s\n' 'round,position,mode,kernel,variant,gflops,kernel_time_ms,correctness,temp_before_c,temp_after_c,sm_clock_before_mhz,sm_clock_after_mhz,mem_clock_before_mhz,mem_clock_after_mhz,beta,c_policy,seed' >"${csv}"
    modes=(strict k10 v1 v2 v3)
    snapshot() {
      nvidia-smi --query-gpu=temperature.gpu,clocks.current.sm,clocks.current.memory \
        --format=csv,noheader,nounits | head -n 1 | tr -d ' '
    }
    for ((round=1;round<=${ROUNDS:-3};round++)); do
      start=$(((round - 1) % ${#modes[@]}))
      for ((position=0;position<${#modes[@]};position++)); do
        name="${modes[$(((start + position) % ${#modes[@]}))]}"
        variant=na
        if [[ "${name}" == strict ]]; then kernel=13; binary="${ROOT_DIR}/build-k23-v1/sgemm"
        elif [[ "${name}" == k10 ]]; then kernel=10; binary="${ROOT_DIR}/build-k23-v1/sgemm"
        else kernel=23; variant="${name#v}"; binary="${ROOT_DIR}/build-k23-v${variant}/sgemm"
        fi
        before="$(snapshot)"
        output="$(env -u NVIDIA_TF32_OVERRIDE DEVICE="${DEVICE:-0}" \
          SGEMM_BENCH_SIZE=4096 SGEMM_WARMUP="${WARMUP:-10}" \
          SGEMM_REPEAT="${REPEAT:-50}" SGEMM_SEED="${SEED}" \
          SGEMM_RESET_C_EACH_REPEAT=1 SGEMM_BETA=3 "${binary}" "${kernel}")"
        after="$(snapshot)"
        seconds="$(awk -F'[()]' '/Average elapsed time:/ {v=$2} END {gsub(/[^0-9.]/,"",v); print v}' <<<"${output}")"
        gflops="$(awk -F'[()]' '/performance:/ {v=$4} END {gsub(/[^0-9.]/,"",v); print v}' <<<"${output}")"
        milliseconds="$(awk -v s="${seconds}" 'BEGIN {printf "%.4f",s*1000}')"
        correctness=reference
        if [[ "${kernel}" != 13 ]]; then
          grep -q 'Correctness vs strict FP32 cuBLAS: PASS' <<<"${output}"
          correctness=PASS
        fi
        IFS=',' read -r tb sb mb <<<"${before}"; IFS=',' read -r ta sa ma <<<"${after}"
        printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,3,reset-each-repeat,%s\n' \
          "${round}" "$((position+1))" "${name}" "${kernel}" "${variant}" \
          "${gflops}" "${milliseconds}" "${correctness}" "${tb}" "${ta}" \
          "${sb}" "${sa}" "${mb}" "${ma}" "${SEED}" >>"${csv}"
        printf 'Round %s %s: %s GFLOPS, %s ms\n' "${round}" "${name}" "${gflops}" "${milliseconds}"
      done
    done
    printf 'Fair CSV: %s\n' "${csv}"
    ;;
  *) printf 'Usage: %s {build|correctness|memcheck|fair}\n' "$0" >&2; exit 2;;
esac
