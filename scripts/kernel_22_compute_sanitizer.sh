#!/usr/bin/env bash
# 【文件导读】K22 的小尺寸访存检查
# 给初学者：正确性检查是在回答“算对了吗”；它与“跑得快吗”是两件事，PASS 也只覆盖实际测试过的情况。
# 运行：bash scripts/kernel_22_compute_sanitizer.sh；先构建 build-k22-v<variant>/sgemm。
# 用途：用 compute-sanitizer --tool memcheck 检查越界、非法访问等问题。
# 范围：三个变体 × 尺寸 128/256，beta=3、repeat=1、warmup=0。
# 参数：OUTPUT_DIR、RUN_TAG、DEVICE、SEED；每个变体/尺寸有单独日志，另生成汇总 CSV。
# 工具错误通过 --error-exitcode=99 传递；脚本同时按自己的结果规则记录 PASS/FAIL。
# 这里只运行 memcheck；不是 racecheck/synccheck，也不代表六尺寸全部通过工具验证。
# 日志重定向可能写入同名单项文件，应使用新 RUN_TAG；检查范围见循环本身。
# 中文总索引：docs/zh/23-代码与脚本使用索引.md；脚本命令默认在仓库根目录执行。


# Bash 的安全开关：-e 遇失败尽早停，-u 防止误用未定义变量，pipefail 让管道中间失败也能被发现。
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 根据脚本自身位置找仓库，不是把当前 shell 所在目录随意当根目录；后续路径都由这里展开。
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
# OUTPUT_DIR 是结果存放处，RUN_TAG（若有）是批次名字；两者不是 kernel 参数，换标签不会重新选 tile。
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/benchmark_results}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d-%H%M%S)}"
CSV="${OUTPUT_DIR}/kernel_22_memcheck_${RUN_TAG}.csv"
mkdir -p "${OUTPUT_DIR}"
[[ ! -e "${CSV}" ]] || { printf 'Refusing to overwrite %s\n' "${CSV}" >&2; exit 1; }
printf 'variant,size,result,log\n' >"${CSV}"

for variant in 1 2 3; do
  for size in 128 256; do
    log="${OUTPUT_DIR}/kernel_22_memcheck_v${variant}_${size}_${RUN_TAG}.log"
    if env -u NVIDIA_TF32_OVERRIDE DEVICE="${DEVICE:-0}" SGEMM_BENCH_SIZE="${size}" \
      SGEMM_WARMUP=0 SGEMM_REPEAT=1 SGEMM_SEED="${SEED:-20261012}" \
      SGEMM_RESET_C_EACH_REPEAT=1 SGEMM_BETA=3 \
      compute-sanitizer --tool memcheck --error-exitcode=99 \
      "${ROOT_DIR}/build-k22-v${variant}/sgemm" 22 >"${log}" 2>&1; then
      result=PASS
    else
      result=FAIL
    fi
    printf '%s,%s,%s,%s\n' "${variant}" "${size}" "${result}" "${log#${ROOT_DIR}/}" >>"${CSV}"
  done
done
fails="$(awk -F, 'NR>1 && $3!="PASS" {n++} END {print n+0}' "${CSV}")"
[[ "${fails}" == 0 ]] || { printf 'Compute Sanitizer failures: %s\n' "${fails}" >&2; exit 1; }
printf 'Compute Sanitizer: 6/6 PASS\nCSV: %s\n' "${CSV}"
