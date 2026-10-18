#!/usr/bin/env bash
# 【文件导读】K20 的六尺寸、三种 beta 数值检查
# 给初学者：正确性检查是在回答“算对了吗”；它与“跑得快吗”是两件事，PASS 也只覆盖实际测试过的情况。
# 运行：bash scripts/kernel_20_correctness_matrix.sh；前提是 build-simt256-v<variant>/sgemm 已编译。
# 用途：遍历脚本列出的各 variant 在 beta=0/1/3 下与 strict FP32 cuBLAS 比较。
# 默认六尺寸由 sgemm.cu 提供：128、256、512、1024、2048、4096；绝对误差门槛仍为 0.01。
# 环境中若残留 SGEMM_BENCH_SIZE，会缩小测试范围；完整矩阵运行前应取消该变量。
# 参数：DEVICE、SEED、OUTPUT_DIR、RUN_TAG；日志标签不改变编译期参数。
# 输出：CSV 保存 variant/beta/size/result，并保留完整日志；同名主输出已存在会退出。
# 预热/重复各一次只是为了数值检查，不是可信的性能样本；需结合退出码与 PASS 行核对覆盖。
# 下一步：kernel_20_fair_audit.sh；工具检查还需要独立运行 Compute Sanitizer。
# 中文总索引：docs/zh/23-代码与脚本使用索引.md；脚本命令默认在仓库根目录执行。


# Bash 的安全开关：-e 遇失败尽早停，-u 防止误用未定义变量，pipefail 让管道中间失败也能被发现。
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 根据脚本自身位置找仓库，不是把当前 shell 所在目录随意当根目录；后续路径都由这里展开。
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
# OUTPUT_DIR 是结果存放处，RUN_TAG（若有）是批次名字；两者不是 kernel 参数，换标签不会重新选 tile。
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/benchmark_results}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d-%H%M%S)}"
CSV="${OUTPUT_DIR}/kernel_20_correctness_${RUN_TAG}.csv"
LOG="${OUTPUT_DIR}/kernel_20_correctness_${RUN_TAG}.log"
mkdir -p "${OUTPUT_DIR}"
[[ ! -e "${CSV}" && ! -e "${LOG}" ]] || { printf 'Refusing to overwrite output\n' >&2; exit 1; }
printf 'variant,beta,size,result\n' >"${CSV}"
: >"${LOG}"

# 外层枚举变体/beta，尺寸由被调用的 sgemm 默认列表提供；应确认没有残留单尺寸环境变量。
for variant in 1 2 3; do
  binary="${ROOT_DIR}/build-simt256-v${variant}/sgemm"
  for beta in 0 1 3; do
    output="$(env -u NVIDIA_TF32_OVERRIDE DEVICE="${DEVICE:-0}" \
      SGEMM_WARMUP=1 SGEMM_REPEAT=1 SGEMM_SEED="${SEED:-20261012}" \
      SGEMM_RESET_C_EACH_REPEAT=1 SGEMM_BETA="${beta}" "${binary}" 20 2>&1)"
    printf '[variant=%s beta=%s]\n%s\n' "${variant}" "${beta}" "${output}" | tee -a "${LOG}"
    awk -v variant="${variant}" -v beta="${beta}" '
      /dimensions\(m=n=k\)/ {gsub(/,/, "", $2); size=$2}
      /Correctness vs strict FP32 cuBLAS:/ {print variant "," beta "," size "," $NF}
    ' <<<"${output}" >>"${CSV}"
  done
done
rows="$(($(wc -l <"${CSV}") - 1))"
passes="$(awk -F, 'NR>1 && $4=="PASS" {n++} END {print n+0}' "${CSV}")"
[[ "${rows}" -eq 54 && "${passes}" -eq 54 ]] || {
  printf 'Expected 54/54 PASS, got %s/%s\n' "${passes}" "${rows}" >&2
  exit 1
}
printf 'Correctness: %s/%s PASS\nCSV: %s\nLog: %s\n' "${passes}" "${rows}" "${CSV}" "${LOG}"
