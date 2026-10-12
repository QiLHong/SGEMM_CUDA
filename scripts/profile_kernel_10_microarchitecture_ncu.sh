#!/usr/bin/env bash

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build-prefetch-v1}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/benchmark_results}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d-%H%M%S)}"
METRICS="gpu__time_duration.sum,launch__block_size,launch__grid_size,launch__registers_per_thread,launch__shared_mem_per_block,launch__waves_per_multiprocessor,sm__throughput.avg.pct_of_peak_sustained_elapsed,sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active,sm__warps_active.avg.pct_of_peak_sustained_active,smsp__issue_active.avg.pct_of_peak_sustained_active,smsp__issue_inst0.avg.pct_of_peak_sustained_active,smsp__warps_eligible.avg.per_cycle_active,smsp__warps_issue_stalled_barrier.avg.per_cycle_active,smsp__warps_issue_stalled_dispatch_stall.avg.per_cycle_active,smsp__warps_issue_stalled_long_scoreboard.avg.per_cycle_active,smsp__warps_issue_stalled_short_scoreboard.avg.per_cycle_active,smsp__warps_issue_stalled_math_pipe_throttle.avg.per_cycle_active,smsp__warps_issue_stalled_mio_throttle.avg.per_cycle_active,smsp__warps_issue_stalled_wait.avg.per_cycle_active,smsp__warps_issue_stalled_no_instruction.avg.per_cycle_active,smsp__warps_issue_stalled_not_selected.avg.per_cycle_active,smsp__inst_executed.sum,smsp__inst_executed_pipe_fma.sum,smsp__inst_executed_pipe_lsu.sum,smsp__inst_executed_pipe_adu.sum,smsp__inst_executed_pipe_alu.sum,smsp__inst_executed_pipe_ipa.sum,smsp__inst_executed_pipe_uniform.sum,smsp__inst_executed_pipe_xu.sum,smsp__sass_thread_inst_executed_op_ffma_pred_on.sum,l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum"
mkdir -p "${OUTPUT_DIR}"

for mode in k10 strict; do
  report="${OUTPUT_DIR}/ncu_k10_micro_${mode}_${RUN_TAG}"
  [[ ! -e "${report}.ncu-rep" ]] || {
    printf 'Refusing to overwrite %s.ncu-rep\n' "${report}" >&2
    exit 1
  }
  common=(--target-processes all --launch-count 1 --metrics "${METRICS}" -o "${report}")
  if [[ "${mode}" == k10 ]]; then
    common+=(--kernel-name 'regex:sgemmWarptiling')
    kernel=10
  else
    kernel=13
  fi
  env -u NVIDIA_TF32_OVERRIDE DEVICE="${DEVICE:-0}" SGEMM_BENCH_SIZE=4096 \
    SGEMM_WARMUP=0 SGEMM_REPEAT=1 SGEMM_SEED="${SEED:-20261012}" \
    SGEMM_RESET_C_EACH_REPEAT=1 SGEMM_BETA=3 \
    ncu "${common[@]}" "${BUILD_DIR}/sgemm" "${kernel}"
done
