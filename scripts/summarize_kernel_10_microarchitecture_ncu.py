#!/usr/bin/env python3
"""Export the K10/strict-cuBLAS microarchitecture NCU reports as CSV."""

import argparse
import csv
import io
import subprocess
import sys
from pathlib import Path

FIELDS = {
    "gpu__time_duration.sum": "profile_duration_ms",
    "launch__block_size": "threads_per_block",
    "launch__grid_size": "blocks_in_grid",
    "launch__registers_per_thread": "registers_per_thread",
    "launch__shared_mem_per_block": "shared_memory_kb",
    "launch__waves_per_multiprocessor": "waves_per_sm",
    "sm__warps_active.avg.pct_of_peak_sustained_active": "occupancy_pct",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed": "sm_throughput_pct",
    "sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active": "fp32_pipe_active_pct",
    "smsp__issue_active.avg.pct_of_peak_sustained_active": "issue_slots_busy_pct",
    "smsp__issue_inst0.avg.pct_of_peak_sustained_active": "no_eligible_pct",
    "smsp__warps_eligible.avg.per_cycle_active": "eligible_warps_per_scheduler",
    "smsp__warps_issue_stalled_dispatch_stall.avg.per_cycle_active": "dispatch_stall",
    "smsp__warps_issue_stalled_barrier.avg.per_cycle_active": "barrier_stall",
    "smsp__warps_issue_stalled_long_scoreboard.avg.per_cycle_active": "long_scoreboard",
    "smsp__warps_issue_stalled_short_scoreboard.avg.per_cycle_active": "short_scoreboard",
    "smsp__warps_issue_stalled_math_pipe_throttle.avg.per_cycle_active": "math_pipe_throttle",
    "smsp__warps_issue_stalled_mio_throttle.avg.per_cycle_active": "mio_throttle",
    "smsp__warps_issue_stalled_wait.avg.per_cycle_active": "wait_stall",
    "smsp__warps_issue_stalled_no_instruction.avg.per_cycle_active": "no_instruction",
    "smsp__warps_issue_stalled_not_selected.avg.per_cycle_active": "not_selected",
    "smsp__inst_executed.sum": "instructions_executed",
    "smsp__inst_executed_pipe_fma.sum": "fma_pipe_instructions",
    "smsp__inst_executed_pipe_lsu.sum": "lsu_pipe_instructions",
    "smsp__inst_executed_pipe_adu.sum": "adu_pipe_instructions",
    "smsp__inst_executed_pipe_alu.sum": "alu_pipe_instructions",
    "smsp__inst_executed_pipe_ipa.sum": "ipa_pipe_instructions",
    "smsp__inst_executed_pipe_uniform.sum": "uniform_pipe_instructions",
    "smsp__inst_executed_pipe_xu.sum": "xu_pipe_instructions",
    "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum": "thread_ffma_instructions",
    "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum": "shared_load_wavefronts",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum": "shared_load_conflicts",
    "l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum": "local_load_sectors",
    "l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum": "local_store_sectors",
}


def row_from_report(path):
    result = subprocess.run(["ncu", "--import", str(path), "--csv", "--page", "raw"],
                            check=True, text=True, capture_output=True)
    rows = list(csv.DictReader(io.StringIO(result.stdout)))
    data = next(row for row in reversed(rows) if row.get("Kernel Name"))
    return {"kernel_name": data["Kernel Name"],
            **{label: data.get(metric, "") for metric, label in FIELDS.items()}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("benchmark_results"))
    parser.add_argument("--tag", default="20261012-final")
    args = parser.parse_args()
    writer = csv.DictWriter(sys.stdout,
                            fieldnames=["mode", "kernel_name", *FIELDS.values()],
                            lineterminator="\n")
    writer.writeheader()
    for mode in ["k10", "strict"]:
        report = args.results / f"ncu_k10_micro_{mode}_{args.tag}.ncu-rep"
        writer.writerow({"mode": mode, **row_from_report(report)})


if __name__ == "__main__":
    main()
