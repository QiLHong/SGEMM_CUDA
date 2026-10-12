#!/usr/bin/env bash

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/benchmark_results}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d-%H%M%S)}"
METRICS="gpu__time_duration.sum,launch__registers_per_thread,launch__shared_mem_per_block,sm__throughput.avg.pct_of_peak_sustained_elapsed,sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active,sm__warps_active.avg.pct_of_peak_sustained_active,smsp__issue_active.avg.pct_of_peak_sustained_active,smsp__issue_inst0.avg.pct_of_peak_sustained_active,smsp__warps_eligible.avg.per_cycle_active,smsp__warps_issue_stalled_barrier.avg.per_cycle_active,smsp__warps_issue_stalled_dispatch_stall.avg.per_cycle_active,smsp__warps_issue_stalled_long_scoreboard.avg.per_cycle_active,smsp__warps_issue_stalled_short_scoreboard.avg.per_cycle_active,dram__bytes_read.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,lts__t_sector_hit_rate.pct,lts__throughput.avg.pct_of_peak_sustained_elapsed,smsp__inst_executed.sum,smsp__inst_executed_pipe_fma.sum,smsp__inst_executed_pipe_lsu.sum,smsp__inst_executed_pipe_adu.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum,l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum,l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum"
mkdir -p "${OUTPUT_DIR}"

for mode in ${MODES:-strict k10 v1 v2 v3}; do
  if [[ "${mode}" == strict ]]; then variant=1; kernel=13; regex='cutlass_80_simt_sgemm'
  elif [[ "${mode}" == k10 ]]; then variant=1; kernel=10; regex='sgemmWarptiling'
  else variant="${mode#v}"; kernel=20; regex='sgemmSimt256x128'
  fi
  report="${OUTPUT_DIR}/ncu_k20_${mode}_${RUN_TAG}"
  [[ ! -e "${report}.ncu-rep" ]] || { printf 'Refusing to overwrite %s.ncu-rep\n' "${report}" >&2; exit 1; }
  env -u NVIDIA_TF32_OVERRIDE DEVICE="${DEVICE:-0}" SGEMM_BENCH_SIZE=4096 \
    SGEMM_WARMUP=0 SGEMM_REPEAT=1 SGEMM_SEED="${SEED:-20261012}" \
    SGEMM_RESET_C_EACH_REPEAT=1 SGEMM_BETA=3 \
    ncu --target-processes all --kernel-name "regex:${regex}" --launch-count 1 \
      --metrics "${METRICS}" -o "${report}" \
      "${ROOT_DIR}/build-simt256-v${variant}/sgemm" "${kernel}"
done
