#!/usr/bin/env bash

set -euo pipefail

# RTX 3080 / SM86 focused Kernel 10 search. Unlike the upstream tuner, this
# script passes compile-time definitions through CMake and never edits sources.
# Search runs use one 4096^3 shape and fewer repeats; the winner is rebuilt and
# rechecked with a longer run. Every run still performs the stock cuBLAS
# correctness comparison before timing.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build-tune}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/benchmark_results}"
DEVICE="${DEVICE:-0}"
SEARCH_REPEAT="${SEARCH_REPEAT:-15}"
FINAL_REPEAT="${FINAL_REPEAT:-50}"
WARMUP="${WARMUP:-10}"
AS_PAD="${AS_PAD:-4}"
MIN_BLOCKS="${MIN_BLOCKS:-2}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d-%H%M%S)}"
CSV="${OUTPUT_DIR}/kernel_10_rtx3080_${RUN_TAG}.csv"
LOG="${OUTPUT_DIR}/kernel_10_rtx3080_${RUN_TAG}.log"

mkdir -p "${BUILD_DIR}" "${OUTPUT_DIR}"

configure_candidate() {
  local bm="$1" bn="$2" bk="$3" wm="$4" wn="$5"
  local wniter="$6" tm="$7" tn="$8" threads="$9"
  cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" -G "Unix Makefiles" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_FLAGS_RELEASE="-O3 -DNDEBUG" \
    -DSGEMM_K10_BM="${bm}" -DSGEMM_K10_BN="${bn}" \
    -DSGEMM_K10_BK="${bk}" -DSGEMM_K10_WM="${wm}" \
    -DSGEMM_K10_WN="${wn}" -DSGEMM_K10_WNITER="${wniter}" \
    -DSGEMM_K10_TM="${tm}" -DSGEMM_K10_TN="${tn}" \
    -DSGEMM_K10_NUM_THREADS="${threads}" -DSGEMM_K10_AS_PAD="${AS_PAD}" \
    -DSGEMM_K10_MIN_BLOCKS="${MIN_BLOCKS}"
  cmake --build "${BUILD_DIR}" --target sgemm --parallel 2
}

measure_kernel() {
  local kernel="$1" repeats="$2"
  DEVICE="${DEVICE}" SGEMM_BENCH_SIZE=4096 SGEMM_WARMUP="${WARMUP}" \
    SGEMM_REPEAT="${repeats}" "${BUILD_DIR}/sgemm" "${kernel}"
}

extract_gflops() {
  awk -F'[()]' '/performance:/ {value=$4} END {gsub(/[^0-9.]/, "", value); print value}'
}

# id BM BN BK WM WN WNITER TM TN threads
readarray -t CANDIDATES <<'EOF'
baseline 128 128 16 64 64 4 8 4 128
bk8 128 128 8 64 64 4 8 4 128
bk32 128 128 32 64 64 4 8 4 128
sub_8x4_n2 128 128 16 64 64 2 8 4 128
sub_4x8_n2 128 128 16 64 64 2 4 8 128
sub_8x8_n1 128 128 16 64 64 1 8 8 128
sub_4x4_n4 128 128 16 64 64 4 4 4 128
sub_4x4_n2 128 128 16 64 64 2 4 4 128
warp_128x32 128 128 16 128 32 2 8 4 128
warp_32x128 128 128 16 32 128 4 8 4 128
threads256_32x64_bk8 128 128 8 32 64 2 4 4 256
threads256_32x64_bk16 128 128 16 32 64 2 4 4 256
threads256_32x64_bk32 128 128 32 32 64 2 4 4 256
threads256_64x32 128 128 16 64 32 2 4 4 256
threads256_4x8 128 128 16 32 64 1 4 8 256
threads256_8x4 128 128 16 32 64 2 8 4 256
block_128x64_bk8 128 64 8 64 32 2 8 4 128
block_128x64_bk16 128 64 16 64 32 2 8 4 128
block_64x128_bk8 64 128 8 32 64 2 4 8 128
block_64x128_bk16 64 128 16 32 64 2 4 8 128
block_128x256_bk8 128 256 8 64 64 4 8 4 256
block_128x256_bk16 128 256 16 64 64 4 8 4 256
block_256x128_bk8 256 128 8 64 64 4 8 4 256
block_256x128_bk16 256 128 16 64 64 4 8 4 256
EOF

printf 'id,gflops,bm,bn,bk,wm,wn,wniter,tm,tn,threads,as_pad,min_blocks,status\n' >"${CSV}"
: >"${LOG}"

configure_candidate 128 128 16 64 64 4 8 4 128 >>"${LOG}" 2>&1
BASELINE_OUTPUT="$(measure_kernel 0 "${FINAL_REPEAT}" 2>&1)"
printf '%s\n' "${BASELINE_OUTPUT}" | tee -a "${LOG}"
CUBLAS_GFLOPS="$(printf '%s\n' "${BASELINE_OUTPUT}" | extract_gflops)"
printf 'cuBLAS warmed baseline: %s GFLOPS\n' "${CUBLAS_GFLOPS}" | tee -a "${LOG}"

for candidate in "${CANDIDATES[@]}"; do
  read -r id bm bn bk wm wn wniter tm tn threads <<<"${candidate}"
  printf '\n[%s] BM=%s BN=%s BK=%s WM=%s WN=%s WNITER=%s TM=%s TN=%s threads=%s AS_PAD=%s MIN_BLOCKS=%s\n' \
    "${id}" "${bm}" "${bn}" "${bk}" "${wm}" "${wn}" "${wniter}" \
    "${tm}" "${tn}" "${threads}" "${AS_PAD}" "${MIN_BLOCKS}" | tee -a "${LOG}"

  if ! configure_candidate "${bm}" "${bn}" "${bk}" "${wm}" "${wn}" \
      "${wniter}" "${tm}" "${tn}" "${threads}" >>"${LOG}" 2>&1; then
    printf '%s,,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,build_failed\n' \
      "${id}" "${bm}" "${bn}" "${bk}" "${wm}" "${wn}" "${wniter}" \
      "${tm}" "${tn}" "${threads}" "${AS_PAD}" "${MIN_BLOCKS}" >>"${CSV}"
    continue
  fi

  if ! RUN_OUTPUT="$(measure_kernel 10 "${SEARCH_REPEAT}" 2>&1)"; then
    printf '%s\n' "${RUN_OUTPUT}" | tee -a "${LOG}"
    printf '%s,,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,run_failed\n' \
      "${id}" "${bm}" "${bn}" "${bk}" "${wm}" "${wn}" "${wniter}" \
      "${tm}" "${tn}" "${threads}" "${AS_PAD}" "${MIN_BLOCKS}" >>"${CSV}"
    continue
  fi

  printf '%s\n' "${RUN_OUTPUT}" | tee -a "${LOG}"
  GFLOPS="$(printf '%s\n' "${RUN_OUTPUT}" | extract_gflops)"
  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,ok\n' \
    "${id}" "${GFLOPS}" "${bm}" "${bn}" "${bk}" "${wm}" "${wn}" \
    "${wniter}" "${tm}" "${tn}" "${threads}" "${AS_PAD}" "${MIN_BLOCKS}" >>"${CSV}"
done

BEST="$(awk -F, 'NR > 1 && $14 == "ok" {print}' "${CSV}" | sort -t, -k2,2nr | head -n1)"
IFS=, read -r BEST_ID _ BEST_BM BEST_BN BEST_BK BEST_WM BEST_WN \
  BEST_WNITER BEST_TM BEST_TN BEST_THREADS _ _ _ <<<"${BEST}"

configure_candidate "${BEST_BM}" "${BEST_BN}" "${BEST_BK}" "${BEST_WM}" \
  "${BEST_WN}" "${BEST_WNITER}" "${BEST_TM}" "${BEST_TN}" \
  "${BEST_THREADS}" >>"${LOG}" 2>&1
FINAL_OUTPUT="$(measure_kernel 10 "${FINAL_REPEAT}" 2>&1)"
printf '\nWinner: %s\n%s\n' "${BEST_ID}" "${FINAL_OUTPUT}" | tee -a "${LOG}"
FINAL_GFLOPS="$(printf '%s\n' "${FINAL_OUTPUT}" | extract_gflops)"
RATIO="$(awk -v kernel="${FINAL_GFLOPS}" -v cublas="${CUBLAS_GFLOPS}" 'BEGIN {printf "%.2f", 100 * kernel / cublas}')"
printf 'Final: %s GFLOPS, %s%% of warmed cuBLAS (%s GFLOPS)\n' \
  "${FINAL_GFLOPS}" "${RATIO}" "${CUBLAS_GFLOPS}" | tee -a "${LOG}"
printf 'CSV: %s\nLog: %s\n' "${CSV}" "${LOG}"
