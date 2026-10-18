#!/usr/bin/env bash
# 【文件导读】交错测量 K19 与严格 FP32 基线
# 给初学者：benchmark 是重复计时实验；warmup 让 GPU/库先热身，repeat 求轮内平均，rounds 检查轮间波动。
# 运行：bash scripts/kernel_19_fair_audit.sh；需提前构建下文 MODES 对应的所有二进制。
# 输入：4096³、固定 SEED、beta 默认或固定为 3；每次 GEMM 前恢复同一份 C0。
# 参数：ROUNDS 是模式轮次，REPEAT 是每进程计时次数，WARMUP 是预热次数；三者不可混淆。
# 默认 ROUNDS=3、REPEAT=50、WARMUP=10；OUTPUT_DIR/RUN_TAG 决定新 CSV 与日志路径。
# 流程：每轮将 MODES 循环移位 → 采集温度/频率快照 → 调用 sgemm → 解析结果。
# CSV 每行是一种模式的一轮均值，position 记录顺序；严格 cuBLAS 的 correctness 标记为 reference。
# 解析器依赖 sgemm.cu 的输出格式，秒会换算为 ms；正确性标记和吞吐要一起看。
# 这不是调参器：不会选择或替换默认 kernel；不要把单轮最大值当作稳定收益。
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
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d-%H%M%S)}"
CSV="${OUTPUT_DIR}/kernel_19_fair_${RUN_TAG}.csv"
LOG="${OUTPUT_DIR}/kernel_19_fair_${RUN_TAG}.log"
# mode 是本次要比较的实现名单；strict 是精度基线，v1/v2 等编号只在当前实验内有意义。
MODES=(strict k10 v1 v2 v3)
mkdir -p "${OUTPUT_DIR}"
[[ ! -e "${CSV}" && ! -e "${LOG}" ]] || { printf 'Refusing to overwrite output\n' >&2; exit 1; }
printf '%s\n' 'round,position,mode,kernel,variant,gflops,kernel_time_ms,correctness,temp_before_c,temp_after_c,sm_clock_before_mhz,sm_clock_after_mhz,mem_clock_before_mhz,mem_clock_after_mhz,beta,c_policy,seed' >"${CSV}"
: >"${LOG}"

# 测量前后记录设备状态，帮助发现温度/频率漂移；不是整轮的连续监控。
snapshot() {
  nvidia-smi --query-gpu=temperature.gpu,clocks.current.sm,clocks.current.memory \
    --format=csv,noheader,nounits | head -n 1 | tr -d ' '
}

# 每轮循环移位模式顺序，而非总让同一实现第一个跑；每行记录该轮均值，最后再看多轮波动。
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
