#!/usr/bin/env bash
# 【文件导读】构建固定版本的公开 CUTLASS SIMT 对照
# 给初学者：构建就是把源代码翻译成可执行文件；编译通过只是第一步，还没有证明计算正确或速度快。
# 运行：bash scripts/build_cutlass_reference.sh <CUTLASS检出目录> <尚不存在的输出目录>。
# 必须使用源码中 EXPECTED 对应的干净 CUTLASS v3.5.1 检出；脚本不下载也不修改第三方库。
# 以 -I 引入 CUTLASS headers，nvcc 按 CUDA_ARCH（默认 86）编译独立 reference，链接 cuBLAS/NVTX。
# 指定其他架构：CUDA_ARCH=89 bash scripts/build_cutlass_reference.sh <CUTLASS检出目录> <新输出目录>。
# 输出 reference、build.log、cuda-architecture.txt、cutlass-commit.txt、nvcc-version.txt、sha256.txt；全部属于本次构建。
# 之后用 run_cutlass_reference.py 做数值/公平计时/采样；公开 CUTLASS 是可读对照，不等于闭源 cuBLAS 实现。
# 中文总索引：docs/zh/23-代码与脚本使用索引.md；脚本命令默认在仓库根目录执行。

# Bash 的安全开关：-e 遇失败尽早停，-u 防止误用未定义变量，pipefail 让管道中间失败也能被发现。
set -euo pipefail
CUDA_ARCH="${CUDA_ARCH:-86}"
# 根据脚本自身位置找仓库，不是把当前 shell 所在目录随意当根目录；后续路径都由这里展开。
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CUTLASS_DIR="${1:?Pass CUTLASS v3.5.1 checkout directory}"
# OUTPUT_DIR 是结果存放处，RUN_TAG（若有）是批次名字；两者不是 kernel 参数，换标签不会重新选 tile。
OUTPUT_DIR="${2:?Pass a fresh output directory}"
EXPECTED=f7b19de32c5d1f3cedfc735c2849f12b537522ee
[[ "$(git -C "${CUTLASS_DIR}" rev-parse HEAD)" == "${EXPECTED}" ]]
[[ -z "$(git -C "${CUTLASS_DIR}" status --porcelain)" ]]
[[ ! -e "${OUTPUT_DIR}" ]] || { echo 'Refusing to overwrite output directory' >&2; exit 1; }
mkdir -p "${OUTPUT_DIR}"
nvcc -O3 -DNDEBUG -std=c++17 -arch="sm_${CUDA_ARCH}" -lineinfo \
  -DSGEMM_K10_MIN_BLOCKS=2 -Xptxas=-v -I"${CUTLASS_DIR}/include" \
  "${ROOT_DIR}/scripts/cutlass_simt_reference.cu" -lcublas -lnvToolsExt \
  -o "${OUTPUT_DIR}/reference" >"${OUTPUT_DIR}/build.log" 2>&1
printf '%s\n' "sm_${CUDA_ARCH}" >"${OUTPUT_DIR}/cuda-architecture.txt"
git -C "${CUTLASS_DIR}" rev-parse HEAD >"${OUTPUT_DIR}/cutlass-commit.txt"
nvcc --version >"${OUTPUT_DIR}/nvcc-version.txt"
sha256sum "${OUTPUT_DIR}/reference" "${ROOT_DIR}/scripts/cutlass_simt_reference.cu" \
  >"${OUTPUT_DIR}/sha256.txt"
echo "Built ${OUTPUT_DIR}/reference"
