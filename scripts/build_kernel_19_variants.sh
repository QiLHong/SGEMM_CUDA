#!/usr/bin/env bash

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

for variant in 1 2 3; do
  build_dir="${ROOT_DIR}/build-prefetch-v${variant}"
  cmake -S "${ROOT_DIR}" -B "${build_dir}" -DCMAKE_BUILD_TYPE=Release \
    -DSGEMM_K19_VARIANT="${variant}" \
    '-DCMAKE_CUDA_FLAGS_RELEASE=-O3 --ptxas-options=-v'
  cmake --build "${build_dir}" -j"${JOBS:-2}"
done
