#!/usr/bin/env python3
# 【文件导读】把 K20 的 Nsight 报告整理为 CSV
# 给初学者：summarize 只整理已有结果，不会把失败变成功，也不会重新测出 GFLOPS；先确认输入文件齐全。
# 运行：python3 scripts/summarize_kernel_20_ncu.py --results benchmark_results --tag <本次RUN_TAG>。
# 输入：profile_kernel_20_ncu.sh 生成的 ncu_k20_<mode>_<tag>.ncu-rep。
# 输出：CSV 写标准输出；脚本用 ncu --import 读取报告，不重新运行 GPU。
# FIELDS 把完整 metric 名缩写成列名；最后一个含 Kernel Name 的记录作为数据行。
# 旧列 no_eligible_pct 实际来自 smsp__issue_inst0：应解释为 no-issue，不能等同于无 eligible warp。
# 部分旧列名带 ms/kb，但直接保留 ncu 显示值；换工具/量级时必须核对原报告单位。
# 本文件保留历史输出格式；不要把 profiler duration 混进 fair 性能统计。
# 先确认报告 mode/tag 对应本次构建，再把资源、等待和吞吐联系起来解释。
# 中文总索引：docs/zh/23-代码与脚本使用索引.md；脚本命令默认在仓库根目录执行。

"""Export selected K20 Nsight Compute reports as one compact CSV."""

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
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum": "shared_load_conflicts",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum": "shared_store_conflicts",
    "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum": "shared_load_wavefronts",
    "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum": "shared_store_wavefronts",
    "l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum": "local_load_sectors",
    "l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum": "local_store_sectors",
}


# 把一个已有 Nsight 报告转换成一行指标；先认列名/单位，再转数字，不能仅凭显示列名猜含义。
# 本脚本保留历史格式；报告缺失或工具版本不兼容时应先查原始 CSV，而不是填 0 冒充测得值。
def report_row(report: Path):
    result = subprocess.run(
        ["ncu", "--import", str(report), "--csv", "--page", "raw"],
        check=True,
        text=True,
        capture_output=True,
    )
    records = list(csv.DictReader(io.StringIO(result.stdout)))
    data = next(row for row in reversed(records) if row.get("Kernel Name"))
    return {label: data.get(metric, "") for metric, label in FIELDS.items()}


# 解析报告路径/标签，逐模式汇总并输出表；这一步不运行新的性能实验。
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("benchmark_results"))
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()
    writer = csv.DictWriter(
        __import__("sys").stdout,
        fieldnames=["mode", *FIELDS.values()],
        lineterminator="\n",
    )
    writer.writeheader()
    for name in ["strict", "k10", "v1", "v2", "v3"]:
        report = args.results / f"ncu_k20_{name}_{args.tag}.ncu-rep"
        writer.writerow({"mode": name, **report_row(report)})


if __name__ == "__main__":
    main()
