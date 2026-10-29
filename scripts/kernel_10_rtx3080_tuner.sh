#!/usr/bin/env bash
# 【文件导读】RTX 3080 的受限 K10 参数搜索
# 给初学者：这是保留实验历史的工具，不是推荐的一键运行入口；先看下面的改写/覆盖风险。
# 运行示例：MIN_BLOCKS=2 bash scripts/kernel_10_rtx3080_tuner.sh；需可用 GPU 与 CMake/nvcc。
# MIN_BLOCKS 必须显式给出；每种 tile 需独立检查寄存器/spill，launch bounds 不是运行时驻留保证。
# 候选来自固定 CANDIDATES 表；通过 CMake 宏重新构建 BUILD_DIR，不用 sed 改源码。
# SEARCH_REPEAT/FINAL_REPEAT 控制初筛和复核次数；AS_PAD、SEED、WARMUP 控制输入与布局。
# 输出带 RUN_TAG 的 CSV/LOG：每组保留参数、GFLOPS、build_failed/run_failed/ok，最后重建最快候选。
# 此历史脚本吞吐分母使用 kernel 0 的原 cuBLAS 配置；不能直接称为 kernel 13 strict 基线。
# 只有 4096³ 初筛及最后长测，不等于六尺寸正确性与多轮独立确认；最佳参数不自动替换源码默认值。
# 中文总索引：docs/zh/23-代码与脚本使用索引.md；脚本命令默认在仓库根目录执行。


# Bash 的安全开关：-e 遇失败尽早停，-u 防止误用未定义变量，pipefail 让管道中间失败也能被发现。
set -euo pipefail

# 针对 RTX 3080 / SM86 搜索 Kernel 10 参数；本脚本通过 CMake 传递
# 编译期宏定义，搜索过程中不修改源码。
# 搜索阶段使用 4096³ 这一种矩阵规模，并减少重复次数；筛出的最佳配置会重新编译，
# 再用较长的计时过程复查。每次运行仍先与原有 cuBLAS 参考结果
# 比较正确性，再开始计时。

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 根据脚本自身位置找仓库，不是把当前 shell 所在目录随意当根目录；后续路径都由这里展开。
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build-tune}"
# OUTPUT_DIR 是结果存放处，RUN_TAG（若有）是批次名字；两者不是 kernel 参数，换标签不会重新选 tile。
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/benchmark_results}"
DEVICE="${DEVICE:-0}"
SEARCH_REPEAT="${SEARCH_REPEAT:-15}"
FINAL_REPEAT="${FINAL_REPEAT:-50}"
WARMUP="${WARMUP:-10}"
AS_PAD="${AS_PAD:-4}"
MIN_BLOCKS="${MIN_BLOCKS:-}"
SEED="${SEED:-20261010}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d-%H%M%S)}"
CSV="${OUTPUT_DIR}/kernel_10_rtx3080_${RUN_TAG}.csv"
LOG="${OUTPUT_DIR}/kernel_10_rtx3080_${RUN_TAG}.log"

if [[ -z "${MIN_BLOCKS}" || ! "${MIN_BLOCKS}" =~ ^[0-9]+$ ]]; then
  printf 'Set MIN_BLOCKS explicitly (for example 0, 1, or 2). Launch bounds are tile-specific.\n' >&2
  exit 2
fi
if [[ -e "${CSV}" || -e "${LOG}" ]]; then
  printf 'Refusing to overwrite an existing benchmark: %s or %s\n' "${CSV}" "${LOG}" >&2
  exit 2
fi

mkdir -p "${BUILD_DIR}" "${OUTPUT_DIR}"

# 把一组候选参数交给 CMake 宏并重新编译；只改 shell 变量、不重编译不会改变模板内核。
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

# 在统一输入/预热/重复参数下启动可执行程序，把原始输出留到日志，解析放在后面。
measure_kernel() {
  local kernel="$1" repeats="$2"
  DEVICE="${DEVICE}" SGEMM_BENCH_SIZE=4096 SGEMM_WARMUP="${WARMUP}" \
    SGEMM_REPEAT="${repeats}" SGEMM_SEED="${SEED}" \
    SGEMM_RESET_C_EACH_REPEAT=1 SGEMM_BETA=3 \
    "${BUILD_DIR}/sgemm" "${kernel}"
}

# 从 benchmark 的固定文本行取吞吐；解析到一个数字不代表 correctness 一定 PASS。
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
