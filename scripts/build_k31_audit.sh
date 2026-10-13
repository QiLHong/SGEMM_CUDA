#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="${1:?Pass a fresh output directory}"
[[ ! -e "${OUTPUT_DIR}" ]] || { echo 'Refusing to overwrite output directory' >&2; exit 1; }
mkdir -p "${OUTPUT_DIR}"
nvcc -O3 -DNDEBUG -std=c++17 -arch=sm_86 -lineinfo \
  -DSGEMM_K10_MIN_BLOCKS=2 -Xptxas=-v \
  "${ROOT_DIR}/scripts/k31_audit.cu" -lcublas -lnvToolsExt \
  -o "${OUTPUT_DIR}/audit" >"${OUTPUT_DIR}/build.log" 2>&1
nvcc --version >"${OUTPUT_DIR}/nvcc-version.txt"
sha256sum "${OUTPUT_DIR}/audit" "${ROOT_DIR}/scripts/k31_audit.cu" \
  "${ROOT_DIR}/src/kernels/30_kernel_mini_simt.cuh" \
  "${ROOT_DIR}/src/kernels/31_kernel_mini_simt_refine.cuh" >"${OUTPUT_DIR}/sha256.txt"
echo "Built ${OUTPUT_DIR}/audit"
