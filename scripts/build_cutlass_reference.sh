#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CUTLASS_DIR="${1:?Pass CUTLASS v3.5.1 checkout directory}"
OUTPUT_DIR="${2:?Pass a fresh output directory}"
EXPECTED=f7b19de32c5d1f3cedfc735c2849f12b537522ee
[[ "$(git -C "${CUTLASS_DIR}" rev-parse HEAD)" == "${EXPECTED}" ]]
[[ -z "$(git -C "${CUTLASS_DIR}" status --porcelain)" ]]
[[ ! -e "${OUTPUT_DIR}" ]] || { echo 'Refusing to overwrite output directory' >&2; exit 1; }
mkdir -p "${OUTPUT_DIR}"
nvcc -O3 -DNDEBUG -std=c++17 -arch=sm_86 -lineinfo \
  -DSGEMM_K10_MIN_BLOCKS=2 -Xptxas=-v -I"${CUTLASS_DIR}/include" \
  "${ROOT_DIR}/scripts/cutlass_simt_reference.cu" -lcublas -lnvToolsExt \
  -o "${OUTPUT_DIR}/reference" >"${OUTPUT_DIR}/build.log" 2>&1
git -C "${CUTLASS_DIR}" rev-parse HEAD >"${OUTPUT_DIR}/cutlass-commit.txt"
nvcc --version >"${OUTPUT_DIR}/nvcc-version.txt"
sha256sum "${OUTPUT_DIR}/reference" "${ROOT_DIR}/scripts/cutlass_simt_reference.cu" \
  >"${OUTPUT_DIR}/sha256.txt"
echo "Built ${OUTPUT_DIR}/reference"
