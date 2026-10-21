#!/usr/bin/env bash
# 【文件导读】构建 K30 独立审计程序
# 给初学者：构建就是把源代码翻译成可执行文件；编译通过只是第一步，还没有证明计算正确或速度快。
# 运行：bash scripts/build_k30_audit.sh benchmark_results/k30-new-session；目标目录必须尚不存在。
# 输入：scripts/k30_audit.cu 与其包含的 kernel；不经过主 sgemm 编号分发。
# 编译：nvcc -O3 -DNDEBUG -std=c++17 -arch=sm_${CUDA_ARCH} -lineinfo，固定 K10 MIN_BLOCKS=2。
# CUDA_ARCH 默认 86；例如 CUDA_ARCH=89 bash scripts/build_k30_audit.sh <新输出目录>。
# 链接 cuBLAS 与 NVTX；-Xptxas=-v 把寄存器、stack、spill 统计写入 build.log。
# 输出：audit、build.log、cuda-architecture.txt、nvcc-version.txt、sha256.txt；哈希对应本次源码/可执行文件。
# 后续：run_k30_audit.py；静态调度另用 analyze_k30_schedule.py。
# 本脚本只编译；没有新 correctness 或 benchmark 结果。失败时先看 build.log。
# 中文总索引：docs/zh/23-代码与脚本使用索引.md；脚本命令默认在仓库根目录执行。

# Bash 的安全开关：-e 遇失败尽早停，-u 防止误用未定义变量，pipefail 让管道中间失败也能被发现。
set -euo pipefail
CUDA_ARCH="${CUDA_ARCH:-86}"
# 根据脚本自身位置找仓库，不是把当前 shell 所在目录随意当根目录；后续路径都由这里展开。
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# 输出目录属于这一批构建；选择一个没用过的名字，不要复用保存过报告的目录。
OUTPUT_DIR="${1:?Pass a fresh output directory}"
[[ ! -e "${OUTPUT_DIR}" ]] || { echo 'Refusing to overwrite output directory' >&2; exit 1; }
mkdir -p "${OUTPUT_DIR}"
nvcc -O3 -DNDEBUG -std=c++17 -arch="sm_${CUDA_ARCH}" -lineinfo \
  -DSGEMM_K10_MIN_BLOCKS=2 -Xptxas=-v \
  "${ROOT_DIR}/scripts/k30_audit.cu" -lcublas -lnvToolsExt \
  -o "${OUTPUT_DIR}/audit" >"${OUTPUT_DIR}/build.log" 2>&1
printf '%s\n' "sm_${CUDA_ARCH}" >"${OUTPUT_DIR}/cuda-architecture.txt"
nvcc --version >"${OUTPUT_DIR}/nvcc-version.txt"
sha256sum "${OUTPUT_DIR}/audit" "${ROOT_DIR}/scripts/k30_audit.cu" \
  "${ROOT_DIR}/src/kernels/30_kernel_mini_simt.cuh" >"${OUTPUT_DIR}/sha256.txt"
echo "Built ${OUTPUT_DIR}/audit"
