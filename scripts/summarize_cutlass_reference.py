#!/usr/bin/env python3
# 【文件导读】公开 CUTLASS 对照的结果归档与统计
# 给初学者：summarize 只整理已有结果，不会把失败变成功，也不会重新测出 GFLOPS；先确认输入文件齐全。
# 运行：python3 scripts/summarize_cutlass_reference.py <实验目录> --output <新的报告目录>。
# 输入：timing/fair.csv、tests/correctness.csv、profiles 及编译/工具日志；不是 benchmark 启动入口。
# 统计：按 mode 汇总各轮 GFLOPS 均值、最小/最大值和样本标准差；这些不是单 launch 的分布。
# Nsight 按 cold/steady 分组；shared/DRAM 字节量按单位行换算，静态 opcode 与动态执行计数分开。
# 输出：performance/fair/correctness/nsight/resources 等 CSV、哈希和环境信息；相同结果可核对，不同内容拒绝覆盖。
# 这是早期汇总入口，完整性规则不同于 K31/K32；不要把成功导出误当成所有验证齐全。
# 同一份报告应引用同批模式，不要跨日期挑最好的基线；独立确认结果另外保存。
# 环境记录可能调用 nvcc/ncu/nvidia-smi 查询版本/设备；不重新测吞吐，但并非完全不依赖 CUDA 工具。
# 历史源码哈希属于原实验时点；给源码加注释后哈希会变化，不能改写旧归档让它假装对应新文件。
# 中文总索引：docs/zh/23-代码与脚本使用索引.md；脚本命令默认在仓库根目录执行。

"""汇总参考实验的原始日志，明确保留单位以及静态/动态计数的区别。"""
import argparse
import collections
import csv
from pathlib import Path
import re
import statistics
import hashlib
import subprocess
import io
import math

p = argparse.ArgumentParser()
p.add_argument("directory", type=Path)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--cutlass-dir",type=Path,default=Path("/tmp/sgemm-cutlass-v3.5.1"))
args = p.parse_args()
args.output.mkdir(parents=True, exist_ok=True)

# 把派生结果写入新文件或按本函数规则核对旧内容，保存历史证据；不要手工改 CSV 数字来凑结论。
def write(name, rows):
    buffer = io.StringIO(newline="")
    w = csv.DictWriter(buffer, fieldnames=list(rows[0]), lineterminator="\n")
    w.writeheader(); w.writerows(rows)
    dest = args.output / name
    if dest.exists():
        old=list(csv.reader(io.StringIO(dest.read_text())))
        new=list(csv.reader(io.StringIO(buffer.getvalue())))
        def equivalent(a,b):
            if a==b: return True
            try: return math.isclose(float(a),float(b),rel_tol=1e-12,abs_tol=1e-8)
            except ValueError: return False
        if len(old)!=len(new) or any(len(a)!=len(b) or
            not all(equivalent(x,y) for x,y in zip(a,b)) for a,b in zip(old,new)):
            raise FileExistsError(dest)
        return
    with dest.open("x") as f: f.write(buffer.getvalue())

data = list(csv.DictReader((args.directory / "timing/fair.csv").open()))
summary = []
for mode in ["strict", "k10", "k21", "cutlass"]:
    samples = [r for r in data if r["mode"] == mode]
    perf = [float(r["gflops"]) for r in samples]
    summary.append(dict(mode=mode, rounds=len(samples), mean_gflops=statistics.mean(perf),
                        min_gflops=min(perf),max_gflops=max(perf),
                        stdev_gflops=statistics.stdev(perf),
                        mean_ms=statistics.mean(float(r["ms"]) for r in samples)))
write("performance.csv",summary)
write("fair.csv",data)
write("correctness.csv",list(csv.DictReader((args.directory/"tests/correctness.csv").open())))

metrics = {
    "launch__registers_per_thread":"registers",
    "launch__shared_mem_per_block":"shared_bytes",
    "sm__warps_active.avg.pct_of_peak_sustained_active":"occupancy_pct",
    "sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active":"fp32_pct",
    "smsp__issue_active.avg.pct_of_peak_sustained_active":"issue_pct",
    "smsp__issue_inst0.avg.pct_of_peak_sustained_active":"no_issue_pct",
    "smsp__warps_eligible.avg.per_cycle_active":"eligible_warps",
    "smsp__warps_issue_stalled_dispatch_stall.avg.per_cycle_active":"dispatch",
    "smsp__warps_issue_stalled_barrier.avg.per_cycle_active":"barrier",
    "smsp__warps_issue_stalled_long_scoreboard.avg.per_cycle_active":"long_scoreboard",
    "smsp__warps_issue_stalled_short_scoreboard.avg.per_cycle_active":"short_scoreboard",
    "dram__bytes_read.sum":"dram_bytes",
    "lts__t_sector_hit_rate.pct":"l2_hit_pct",
    "smsp__inst_executed.sum":"warp_instructions",
    "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum":"thread_ffma",
    "smsp__inst_executed_op_ldgsts.sum":"warp_ldgsts",
    "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum":"shared_load_wavefronts",
    "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum":"shared_store_wavefronts",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum":"shared_load_conflicts",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum":"shared_store_conflicts",
    "l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum":"local_load_sectors",
    "l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum":"local_store_sectors",
}
profiles, sass = [], []
for policy in ["cold","steady"]:
    for mode in ["strict","k10","k21","cutlass"]:
        records = list(csv.DictReader((args.directory/f"profiles/ncu_{policy}_{mode}.csv").open()))
        units, values = records[0], records[-1]
        assert values.get("Kernel Name"), records
        row = dict(policy=policy,mode=mode,kernel=values["Kernel Name"])
        for metric,name in metrics.items():
            v = float(values[metric].replace(",",""))
            unit = units[metric]
            if name in ["shared_bytes","dram_bytes"]:
                v *= {"byte":1,"Kbyte":1e3,"Mbyte":1e6,"Gbyte":1e9}[unit.split("/")[0]]
            row[name]=v
        # Wavefront 是硬件处理的工作单元数，不能当作字节数或 warp 指令数。
        row["load_wavefronts_per_1024_thread_ffma"] = row["shared_load_wavefronts"]*1024/row["thread_ffma"]
        row["warp_instructions_per_32_thread_ffma"] = row["warp_instructions"]*32/row["thread_ffma"]
        profiles.append(row)
        if policy != "cold": continue
        counts = collections.Counter()
        lines = list(csv.reader((args.directory/f"profiles/sass_{policy}_{mode}.csv").open()))
        for line in lines:
            if len(line) != 2 or not line[0].startswith("0x"): continue
            original = line[1].strip()
            ins = re.sub(r"^@!?U?P(?:\d+|T)\s+", "", original)
            match = re.match(r"([A-Z][A-Z0-9_.]*)",ins)
            if match:
                op = match[1].split(".")[0]
                if original.startswith("@!PT "): op += "_PRED_OFF"
                counts[op] += 1
        sass.append(dict(mode=mode,static_total=sum(counts.values()),
                         **{op:counts[op] for op in ["FFMA","LDS","LDS_PRED_OFF","LDGSTS","LDG","STS","DEPBAR","BAR","IMAD","IADD3","LEA","MOV","HMMA","MMA"]}))
write("nsight.csv",profiles)
write("static-sass.csv",sass)
# 编译资源统计只对应当前这个二进制文件，不代表更早的 CMake 构建产物。
resources = []
mode = None
for line in (args.directory / "build.log").read_text().splitlines():
    if "Compiling entry function" in line:
        mode = "cutlass" if "cutlass" in line else ("k21" if "K21" in line else "k10")
    if "bytes spill stores" in line:
        store, loads = re.search(r"(\d+) bytes spill stores, (\d+) bytes spill loads",line).groups()
    if (match := re.search(r"Used (\d+) registers",line)):
        resources.append(dict(mode=mode, registers=int(match[1]),
                              spill_store_bytes=int(store),spill_load_bytes=int(loads),
                              shared_bytes=49152 if mode=="cutlass" else 16640))
write("resources.csv",resources)
checks=[]
for n in [128,256,4096]:
    logfile=args.directory/f"sanitizer/memcheck_{n}.log"
    if logfile.exists():
        log=logfile.read_text()
        checks.append(dict(mode="cutlass",size=n,beta=3,
                           result="PASS" if "ERROR SUMMARY: 0 errors" in log and ",PASS" in log else "FAIL",
                           log=str(logfile)))
if checks: write("memcheck.csv",checks)
root = Path(__file__).resolve().parents[1]
files = [args.directory / "reference", root/"scripts/cutlass_simt_reference.cu",
         root/"src/kernels/10_kernel_warptiling.cuh",root/"src/kernels/21_kernel_cutlass_style_skeleton.cuh"]
write("sha256.csv",[dict(file=str(f.relative_to(root)) if f.is_relative_to(root) else str(f),
                         sha256=hashlib.sha256(f.read_bytes()).hexdigest()) for f in files])
out = io.StringIO()
if not (args.output/"environment.txt").exists():
    for cmd in [["nvcc","--version"],["ncu","--version"],
                ["git","-C",str(args.cutlass_dir),"rev-parse","HEAD"],
                ["nvidia-smi","--query-gpu=name,uuid,driver_version","--format=csv"]]:
        result = subprocess.run(cmd, text=True, capture_output=True)
        out.write(" ".join(cmd)+"\n"+result.stdout+result.stderr+"\n")
    out.write("nvcc -O3 -DNDEBUG -std=c++17 -arch=sm_86 -lineinfo -DSGEMM_K10_MIN_BLOCKS=2 -Xptxas=-v\n")
    out.write("seed=20261013 alpha=0.5 beta=3 reset-C-before-each-call; timed repeat=50 warmup=10\n")
    with (args.output/"environment.txt").open("x") as f: f.write(out.getvalue())
for row in summary: print(row)
for row in profiles:
    print({k:v for k,v in row.items() if k != "kernel"})
