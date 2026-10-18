#!/usr/bin/env bash
# 【文件导读】核对 cuBLAS 精度口径的四模式实验
# 给初学者：benchmark 是重复计时实验；warmup 让 GPU/库先热身，repeat 求轮内平均，rounds 检查轮间波动。
# 运行：BUILD_DIR=<已编译目录> bash scripts/rtx3080_fp32_audit.sh；默认 build-audit。
# legacy/strict/k10/tf32 分别对应主 dispatcher 的 0/13/10/14；TF32 是独立精度对照。
# 4096³、固定 seed/beta，warmup/repeat 可配置；每次恢复 C，D2D 恢复在 event 区间外。
# 按轮移位执行顺序；解析 Difference vs strict FP32 与 K10 数值 PASS，而非只记录速度。
# 输出带 RUN_TAG 的 CSV/LOG；max_abs 与 elements_over_0_01 用于展示精度差异。
# 本次输入 legacy 恰好与 strict 位相等，不构成所有形状/版本的精度保证；TF32 不能混入严格 FP32 排名。
# 中文总索引：docs/zh/23-代码与脚本使用索引.md；脚本命令默认在仓库根目录执行。


# Bash 的安全开关：-e 遇失败尽早停，-u 防止误用未定义变量，pipefail 让管道中间失败也能被发现。
set -euo pipefail

# 公平比较 legacy cuBLAS、pedantic FP32、显式 TF32 与 CUDA-core K10。
# 每次 GEMM 前恢复同一份 C；D2D 恢复不计入 CUDA event 时间。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 根据脚本自身位置找仓库，不是把当前 shell 所在目录随意当根目录；后续路径都由这里展开。
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build-audit}"
# OUTPUT_DIR 是结果存放处，RUN_TAG（若有）是批次名字；两者不是 kernel 参数，换标签不会重新选 tile。
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

# mode 是本次要比较的实现名单；strict 是精度基线，v1/v2 等编号只在当前实验内有意义。
MODES=(legacy strict k10 tf32)

# 把便于阅读的 mode 名称翻译成 sgemm 主程序的数字编号；名称本身不是 CUDA 参数。
kernel_for_mode() {
  case "$1" in
  legacy) printf '0\n' ;;
  strict) printf '13\n' ;;
  k10) printf '10\n' ;;
  tf32) printf '14\n' ;;
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
