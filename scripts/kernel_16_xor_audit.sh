#!/usr/bin/env bash
# 【文件导读】K16 A shared XOR 九组参数构建与计时
# 给初学者：benchmark 是重复计时实验；warmup 让 GPU/库先热身，repeat 求轮内平均，rounds 检查轮间波动。
# 运行：bash scripts/kernel_16_xor_audit.sh；需 CMake/nvcc 与空闲 GPU。
# mask=1/3/7、shift=1/2/3 共九组，分别构建 build-xor-m*-s*/sgemm。
# 每轮交错 strict、k10、九种 XOR；固定 shape、seed、beta，C 每次复位。
# ROUNDS/REPEAT/WARMUP 控制统计，OUTPUT_DIR/RUN_TAG 控制 CSV 与完整日志。
# CSV 保存 mask/shift、correctness、GFLOPS 和设备状态快照；不能只凭映射双射就判断数值或性能。
# CPU 布局边界检查用 verify_a_smem_xor_swizzle.py；GPU 六尺寸检查用 kernel_16_correctness_matrix.sh。
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
SEED="${SEED:-20261011}"
BETA="${BETA:-3}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d-%H%M%S)}"
CSV="${OUTPUT_DIR}/kernel_16_xor_swizzle_${RUN_TAG}.csv"
LOG="${OUTPUT_DIR}/kernel_16_xor_swizzle_${RUN_TAG}.log"
CONFIGS=(m1s1 m1s2 m1s3 m3s1 m3s2 m3s3 m7s1 m7s2 m7s3)
# mode 是本次要比较的实现名单；strict 是精度基线，v1/v2 等编号只在当前实验内有意义。
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

# 测量前后记录设备状态，帮助发现温度/频率漂移；不是整轮的连续监控。
gpu_snapshot() {
  nvidia-smi --query-gpu=temperature.gpu,clocks.current.sm,clocks.current.memory \
    --format=csv,noheader,nounits | head -n 1 | tr -d ' '
}

# 每轮循环移位模式顺序，而非总让同一实现第一个跑；每行记录该轮均值，最后再看多轮波动。
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
