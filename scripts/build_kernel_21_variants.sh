#!/usr/bin/env bash
# 【文件导读】构建 K21 的独立变体
# 给初学者：构建就是把源代码翻译成可执行文件；编译通过只是第一步，还没有证明计算正确或速度快。
# 用途：为“iterator 与 256 线程流水骨架”实验准备可执行文件；这一步只编译，不进行 GPU 计时。
# 运行：bash scripts/build_kernel_21_variants.sh（从仓库根目录；需 CMake、nvcc 和 cuBLAS）。
# 输入：CMakeLists.txt、src/runner.cu 及 kernel；SGEMM_K21_VARIANT 决定模板实例。
# 输出：build-k21-v<variant>/sgemm；OUTPUT_DIR 下保存带 RUN_TAG 的编译日志。
# 变体范围：1、2；并行构建数由 JOBS 控制（默认 2）。
# 阅读顺序：定位仓库 → 枚举 variant → CMake 配置 → 编译；同名 build 目录会增量更新。
# 后续：先运行 kernel_21_correctness_matrix.sh，再做 fair/profile；编译成功不等于数值通过。
# 中文总索引：docs/zh/23-代码与脚本使用索引.md；脚本命令默认在仓库根目录执行。


# Bash 的安全开关：-e 遇失败尽早停，-u 防止误用未定义变量，pipefail 让管道中间失败也能被发现。
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 根据脚本自身位置找仓库，不是把当前 shell 所在目录随意当根目录；后续路径都由这里展开。
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
# 输出目录属于这一批构建；选择一个没用过的名字，不要复用保存过报告的目录。
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/benchmark_results}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "${OUTPUT_DIR}"

# 每个 variant 用独立构建目录，让不同模板实例共存，后面才能交错测量而不反复覆盖同一程序。
for variant in 1 2; do
  build="${ROOT_DIR}/build-k21-v${variant}"
  log="${OUTPUT_DIR}/kernel_21_build_v${variant}_${RUN_TAG}.log"
  [[ ! -e "${log}" ]] || { printf 'Refusing to overwrite %s\n' "${log}" >&2; exit 1; }
  cmake -S "${ROOT_DIR}" -B "${build}" -DCMAKE_BUILD_TYPE=Release \
    -DSGEMM_K21_VARIANT="${variant}" -DCMAKE_CUDA_FLAGS='--ptxas-options=-v' 2>&1 \
    | tee "${log}"
  cmake --build "${build}" --parallel "${JOBS:-2}" 2>&1 | tee -a "${log}"
done
