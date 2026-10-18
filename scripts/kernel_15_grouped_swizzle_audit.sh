#!/usr/bin/env bash
# 【文件导读】K15 grouped-grid 五组参数构建与交错计时
# 给初学者：benchmark 是重复计时实验；warmup 让 GPU/库先热身，repeat 求轮内平均，rounds 检查轮间波动。
# 运行：bash scripts/kernel_15_grouped_swizzle_audit.sh；会编译并使用 GPU。
# GROUP_M=1/2/4/8/16 分别构建到 build-swizzle-g*/；相同 K10 core 只改变 block 坐标遍历。
# MODES 包含 strict、k10 与五组 K15；ROUNDS 循环移位顺序，REPEAT/WARMUP 控制每组测量。
# 4096³、同 seed/beta、每次恢复 C；CSV 保存 group、PASS、吞吐及温度/时钟/功耗快照。
# OUTPUT_DIR/RUN_TAG 指定新结果；额外六尺寸检查用 kernel_15_correctness_matrix.sh。
# 缓存指标改善是否有用要结合最终吞吐；本脚本不采 Nsight，也不自动宣布某组为最优。
# 中文总索引：docs/zh/23-代码与脚本使用索引.md；脚本命令默认在仓库根目录执行。


# Bash 的安全开关：-e 遇失败尽早停，-u 防止误用未定义变量，pipefail 让管道中间失败也能被发现。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 根据脚本自身位置找仓库，不是把当前 shell 所在目录随意当根目录；后续路径都由这里展开。
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
# OUTPUT_DIR 是结果存放处，RUN_TAG（若有）是批次名字；两者不是 kernel 参数，换标签不会重新选 tile。
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
# mode 是本次要比较的实现名单；strict 是精度基线，v1/v2 等编号只在当前实验内有意义。
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

# 测量前后记录设备状态，帮助发现温度/频率漂移；不是整轮的连续监控。
gpu_snapshot() {
  nvidia-smi --query-gpu=temperature.gpu,clocks.current.sm,clocks.current.memory,power.draw \
    --format=csv,noheader,nounits | head -n 1 | tr -d ' '
}

# 把便于阅读的 mode 名称翻译成 sgemm 主程序的数字编号；名称本身不是 CUDA 参数。
kernel_for_mode() {
  case "$1" in
  strict) printf '13\n' ;;
  k10) printf '10\n' ;;
  g*) printf '15\n' ;;
  *) return 1 ;;
  esac
}

# 从模式标签取 GROUP_M 便于写入 CSV；真正布局参数早已在对应二进制编译时固定。
group_for_mode() {
  case "$1" in
  strict | k10) printf '1\n' ;;
  g*) printf '%s\n' "${1#g}" ;;
  *) return 1 ;;
  esac
}

# 每轮循环移位模式顺序，而非总让同一实现第一个跑；每行记录该轮均值，最后再看多轮波动。
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
