#!/usr/bin/env python3
"""Export selected K21 Nsight Compute reports as one compact CSV."""

import argparse
import csv
import io
import subprocess
from pathlib import Path

FIELDS = {
    "gpu__time_duration.sum": "profile_duration_ms",
    "launch__registers_per_thread": "registers_per_thread",
    "launch__shared_mem_per_block": "shared_memory_kb",
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
    "dram__bytes_read.sum": "dram_read_bytes",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed": "dram_throughput_pct",
    "lts__t_sector_hit_rate.pct": "l2_hit_rate_pct",
    "lts__throughput.avg.pct_of_peak_sustained_elapsed": "l2_throughput_pct",
    "smsp__inst_executed.sum": "instructions_executed",
    "smsp__inst_executed_pipe_fma.sum": "fma_pipe_instructions",
    "smsp__inst_executed_pipe_lsu.sum": "lsu_pipe_instructions",
    "smsp__inst_executed_pipe_adu.sum": "adu_pipe_instructions",
    "smsp__inst_executed_op_ldgsts.sum": "ldgsts_instructions",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum": "shared_load_conflicts",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum": "shared_store_conflicts",
    "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum": "shared_load_wavefronts",
    "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum": "shared_store_wavefronts",
    "l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum": "local_load_sectors",
    "l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum": "local_store_sectors",
}


def report_row(report: Path):
    result = subprocess.run(
        ["ncu", "--import", str(report), "--csv", "--page", "raw"],
        check=True, text=True, capture_output=True,
    )
    records = list(csv.DictReader(io.StringIO(result.stdout)))
    data = next(row for row in reversed(records) if row.get("Kernel Name"))
    return {label: data.get(metric, "") for metric, label in FIELDS.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("benchmark_results"))
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()
    writer = csv.DictWriter(
        __import__("sys").stdout,
        fieldnames=["mode", *FIELDS.values()], lineterminator="\n",
    )
    writer.writeheader()
    for name in ["strict", "k10", "v1", "v2"]:
        writer.writerow({"mode": name, **report_row(
            args.results / f"ncu_k21_{name}_{args.tag}.ncu-rep")})


if __name__ == "__main__":
    main()
