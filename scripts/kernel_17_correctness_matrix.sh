#!/usr/bin/env bash
# 【文件导读】K17 的六尺寸、三种 beta 数值检查
# 给初学者：正确性检查是在回答“算对了吗”；它与“跑得快吗”是两件事，PASS 也只覆盖实际测试过的情况。
# 运行：bash scripts/kernel_17_correctness_matrix.sh；前提是 build-k17/sgemm 已编译。
# 用途：检查一个编译配置在 beta=0/1/3 下与 strict FP32 cuBLAS 比较。
# 默认六尺寸由 sgemm.cu 提供：128、256、512、1024、2048、4096；绝对误差门槛仍为 0.01。
# 环境中若残留 SGEMM_BENCH_SIZE，会缩小测试范围；完整矩阵运行前应取消该变量。
# 参数：BUILD_DIR 指向待测二进制；DEVICE、SEED、OUTPUT_DIR、RUN_TAG；日志标签不改变编译期参数。
# 输出：带批次名的完整日志；同名主输出已存在会退出。
# 预热/重复各一次只是为了数值检查，不是可信的性能样本；需结合退出码与 PASS 行核对覆盖。
# 下一步：对应 fair/audit 脚本；工具检查还需要独立运行 Compute Sanitizer。
# 中文总索引：docs/zh/23-代码与脚本使用索引.md；脚本命令默认在仓库根目录执行。


# Bash 的安全开关：-e 遇失败尽早停，-u 防止误用未定义变量，pipefail 让管道中间失败也能被发现。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 根据脚本自身位置找仓库，不是把当前 shell 所在目录随意当根目录；后续路径都由这里展开。
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build-k17}"
# OUTPUT_DIR 是结果存放处，RUN_TAG（若有）是批次名字；两者不是 kernel 参数，换标签不会重新选 tile。
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/benchmark_results}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d-%H%M%S)}"
LOG="${OUTPUT_DIR}/kernel_17_correctness_${RUN_TAG}.log"

mkdir -p "${OUTPUT_DIR}"
if [[ -e "${LOG}" ]]; then
  printf 'Refusing to overwrite existing result: %s\n' "${LOG}" >&2
  exit 1
fi
: >"${LOG}"

# 外层枚举变体/beta，尺寸由被调用的 sgemm 默认列表提供；应确认没有残留单尺寸环境变量。
for beta in 0 1 3; do
  printf '\n[kernel=17 beta=%s]\n' "${beta}" | tee -a "${LOG}"
  env -u NVIDIA_TF32_OVERRIDE DEVICE="${DEVICE:-0}" \
    SGEMM_WARMUP=1 SGEMM_REPEAT=1 SGEMM_SEED="${SEED:-20261011}" \
    SGEMM_RESET_C_EACH_REPEAT=1 SGEMM_BETA="${beta}" \
    "${BUILD_DIR}/sgemm" 17 | tee -a "${LOG}"
done

printf 'Log: %s\n' "${LOG}"
