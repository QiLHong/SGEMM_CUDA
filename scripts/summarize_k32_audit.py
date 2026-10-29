#!/usr/bin/env python3
# 【文件导读】K32的结果归档与统计
# 给初学者：summarize 只整理已有结果，不会把失败变成功，也不会重新测出 GFLOPS；先确认输入文件齐全。
# 运行：python3 scripts/summarize_k32_audit.py <实验目录> --output <新的报告目录>。
# 输入：timing/fair.csv、tests/correctness.csv、profiles 及编译/工具日志；不是 benchmark 启动入口。
# 统计：按 mode 汇总各轮 GFLOPS 均值、最小/最大值和样本标准差；这些不是单 launch 的分布。
# Nsight 按 cold/steady 分组；shared/DRAM 字节量按单位行换算，静态 opcode 与动态执行计数分开。
# 输出：performance/fair/correctness/nsight/resources 等 CSV、哈希和环境信息；相同结果可核对，不同内容拒绝覆盖。
# 发布前核对资源通过的模式、轮数、六尺寸×三 beta、工具日志与必要 profile；缺一项即失败。
# --rounds 与实际 fair 轮数一致；--confirm-rounds 对独立确认生效；--steady-modes 指定必须完整的 steady 模式。
# 环境记录可能调用 nvcc/ncu/nvidia-smi 查询版本/设备；不重新测吞吐，但并非完全不依赖 CUDA 工具。
# 历史源码哈希属于原实验时点；给源码加注释后哈希会变化，不能改写旧归档让它假装对应新文件。
# 中文总索引：docs/zh/23-代码与脚本使用索引.md；脚本命令默认在仓库根目录执行。

"""将 K32 实验证据保存为新的 CSV 文件，保留全部历史测量结果。"""
import argparse
import collections
import csv
import hashlib
import io
import math
from pathlib import Path
import re
import statistics
import subprocess
from audit_log_checks import sanitizer_passed

p=argparse.ArgumentParser()
p.add_argument("directory",type=Path)
p.add_argument("--output",type=Path,required=True)
p.add_argument("--rounds",type=int,default=6)
p.add_argument("--confirm-rounds",type=int,default=5)
p.add_argument("--steady-modes",nargs="*",default=[],
               help="Require these steady profiles to be complete before publishing")
args=p.parse_args()
args.output.mkdir(parents=True,exist_ok=True)

# CSV 按列名读取；返回的是记录集合，后续必须再验证是否齐全，不能把成功打开文件等同完成实验。
def read(path):
    with path.open() as f: return list(csv.DictReader(f))

# 把派生结果写入新文件或按本函数规则核对旧内容，保存历史证据；不要手工改 CSV 数字来凑结论。
def save(name,rows):
    if not rows: return
    stream=io.StringIO(newline="")
    w=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator="\n")
    w.writeheader();w.writerows(rows)
    content=stream.getvalue(); dest=args.output/name
    if dest.exists():
        a=list(csv.reader(io.StringIO(dest.read_text())))
        b=list(csv.reader(io.StringIO(content)))
        def equal(x,y):
            if x==y: return True
            try: return math.isclose(float(x),float(y),rel_tol=1e-12,abs_tol=1e-8)
            except ValueError: return False
        assert len(a)==len(b) and all(len(x)==len(y) and all(equal(u,v) for u,v in zip(x,y))
                                    for x,y in zip(a,b)),f"Refusing to overwrite {dest}"
    else:
        with dest.open("x") as f: f.write(content)

# 文本证据也要遵守不可覆盖不同历史内容的约定；需要重跑时换新的实验目录。
def save_text(name,content):
    dest=args.output/name
    if dest.exists():
        assert dest.read_text()==content,f"Refusing to overwrite {dest}"
    else:
        with dest.open("x") as f: f.write(content)

# 对各轮做均值、标准差与最小/最大统计；先检查轮次和模式完整，避免只挑最好的一轮。
# 同一轮内的配对提升与“两个总体均值之比”不是同一个统计量，报告字段应分别解释。
def performance(data):
    modes=list(dict.fromkeys(r["mode"] for r in data))
    means={m:statistics.mean(float(r["gflops"]) for r in data if r["mode"]==m) for m in modes}
    paired={r["round"]:float(r["gflops"]) for r in data if r["mode"]=="k30"}
    result=[]
    for mode in modes:
        rows=[r for r in data if r["mode"]==mode]
        values=[float(r["gflops"]) for r in rows]
        gains=[100*(float(r["gflops"])/paired[r["round"]]-1) for r in rows]
        result.append(dict(mode=mode,rounds=len(rows),mean_gflops=means[mode],
                           min_gflops=min(values),max_gflops=max(values),
                           stdev_gflops=statistics.stdev(values),
                           mean_ms=statistics.mean(float(r["ms"]) for r in rows),
                           pct_strict=100*means[mode]/means["strict"],
                           gain_vs_k30_pct=100*(means[mode]/means["k30"]-1),
                           paired_gain_min_pct=min(gains),paired_gain_max_pct=max(gains)))
    return result

data=read(args.directory/"timing/fair.csv")
# 先验证整个实验的日志完整性，再创建任何不允许覆盖的 CSV 结果文件。
compiled=read(args.directory/"resources.csv")
accepted={r["mode"] for r in compiled if r["resource_gate"]=="PASS"}
expected_modes=accepted|{"strict"}
if {r["mode"] for r in data}!=expected_modes:
    raise RuntimeError("Incomplete benchmark mode set")
for mode in expected_modes:
    rows=[r for r in data if r["mode"]==mode]
    if len(rows)!=args.rounds or {int(r["round"]) for r in rows}!=set(range(1,args.rounds+1)):
        raise RuntimeError(f"Incomplete benchmark rounds: {mode}")
    if any(r["correctness"]!="PASS" or not math.isfinite(float(r["gflops"])) for r in rows):
        raise RuntimeError(f"Invalid benchmark: {mode}")
    for prefix in ["ncu_cold_","sass_cold_"]:
        if not (args.directory/f"profiles/{prefix}{mode}.csv").exists():
            raise RuntimeError(f"Missing profile: {prefix}{mode}")
correctness=read(args.directory/"tests/correctness.csv")
expected_cases={(m,str(n),str(b)) for m in accepted for n in [128,256,512,1024,2048,4096] for b in [0,1,3]}
if len(correctness)!=len(expected_cases) or {(r["mode"],r["size"],r["beta"]) for r in correctness}!=expected_cases:
    raise RuntimeError("Incomplete correctness matrix")
if any(r["correctness"]!="PASS" or not math.isfinite(float(r["max_abs_error"])) or float(r["max_abs_error"])>0.01 for r in correctness):
    raise RuntimeError("Correctness failed")
for mode in accepted-{"k10"}:
    for tool,sizes in [("memcheck",[128,256,4096]),("racecheck",[128,256]),("synccheck",[128,256])]:
        for n in sizes:
            path=args.directory/f"sanitizer/{tool}_{mode}_{n}.log"
            content=path.read_text()
            if not sanitizer_passed(content, tool):
                raise RuntimeError(f"Sanitizer check failed: {path}")
for mode in args.steady_modes:
    if mode not in expected_modes:
        raise RuntimeError(f"Unknown required steady mode: {mode}")
    for prefix in ["ncu_steady_","sass_steady_"]:
        path=args.directory/f"profiles/{prefix}{mode}.csv"
        if not path.is_file() or not path.stat().st_size:
            raise RuntimeError(f"Missing required steady profile: {path}")
    records=read(args.directory/f"profiles/ncu_steady_{mode}.csv")
    if len(records)<2 or not records[-1].get("Kernel Name"):
        raise RuntimeError(f"Incomplete steady profile: {mode}")
confirmation=[]
if (args.directory/"confirm/fair.csv").exists():
    confirmation=read(args.directory/"confirm/fair.csv")
    confirm_modes={r["mode"] for r in confirmation}
    if not {"strict","k30"} <= confirm_modes <= expected_modes:
        raise RuntimeError("Invalid confirmation mode set")
    for mode in confirm_modes:
        rows=[r for r in confirmation if r["mode"]==mode]
        if len(rows)!=args.confirm_rounds or {int(r["round"]) for r in rows}!=set(range(1,args.confirm_rounds+1)):
            raise RuntimeError(f"Incomplete confirmation rounds: {mode}")
        if any(r["correctness"]!="PASS" or not math.isfinite(float(r["gflops"])) for r in rows):
            raise RuntimeError(f"Invalid confirmation: {mode}")
save("fair.csv",data); summary=performance(data); save("performance.csv",summary)
if confirmation:
    save("confirm-fair.csv",confirmation); save("confirm-performance.csv",performance(confirmation))
for name,path in [("correctness.csv","tests/correctness.csv"),("resources.csv","resources.csv"),
                  ("runtime-resources.csv","runtime-resources.csv"),("static-sass.csv","static-sass.csv"),
                  ("schedule.csv","schedule.csv")]:
    save(name,read(args.directory/path))

metrics={
    "launch__registers_per_thread":"registers",
    "launch__shared_mem_per_block":"shared_bytes",
    "sm__warps_active.avg.pct_of_peak_sustained_active":"occupancy_pct",
    "sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active":"fp32_pct",
    "smsp__issue_active.avg.pct_of_peak_sustained_active":"issue_pct",
    "smsp__issue_inst0.avg.pct_of_peak_sustained_active":"no_issue_pct",
    "smsp__warps_eligible.avg.per_cycle_active":"eligible_warps",
    "smsp__warps_issue_stalled_dispatch_stall.avg.per_cycle_active":"dispatch",
    "smsp__warps_issue_stalled_barrier.avg.per_cycle_active":"barrier",
    "smsp__warps_issue_stalled_wait.avg.per_cycle_active":"wait",
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
profiles=[]
for policy in ["cold","steady"]:
    for mode in [r["mode"] for r in summary]:
        path=args.directory/f"profiles/ncu_{policy}_{mode}.csv"
        if policy=="steady" and not path.exists(): continue
        records=read(path); units,values=records[0],records[-1]
        row=dict(policy=policy,mode=mode,kernel=values["Kernel Name"])
        for metric,name in metrics.items():
            value=float(values[metric].replace(",",""))
            if name in ["shared_bytes","dram_bytes"]:
                value*={"byte":1,"Kbyte":1e3,"Mbyte":1e6,"Gbyte":1e9}[units[metric].split("/")[0]]
            row[name]=value
        profiles.append(row)
save("nsight.csv",profiles)
# 库内核的 SASS 来自 NCU 报告，不是对当前可执行文件运行 cuobjdump 得到的内容。
strict_counts=collections.Counter()
for row in csv.reader((args.directory/"profiles/sass_cold_strict.csv").open()):
    if len(row)!=2 or not row[0].startswith("0x"): continue
    original=row[1].strip()
    ins=re.sub(r"^@!?U?P(?:\d+|T)\s+","",original)
    match=re.match(r"([A-Z][A-Z0-9_.]*)",ins)
    if match:
        op=match[1].split(".")[0]
        if original.startswith("@!PT "): op+="_PRED_OFF"
        strict_counts[op]+=1
save("cublas-static-sass.csv",[dict(mode="strict",static_total=sum(strict_counts.values()),
      **{op:strict_counts[op] for op in ["FFMA","LDS","LDS_PRED_OFF","LDGSTS","DEPBAR","BAR","LDG","STS","IMAD","IADD3","MOV","HMMA","MMA"]})])
checks=[]
for path in sorted((args.directory/"sanitizer").glob("*.log")):
    text=path.read_text()
    tool=path.stem.split("_")[0]
    checks.append(dict(test=path.stem,result="PASS" if sanitizer_passed(text,tool) else "FAIL",
                       summary="; ".join(line for line in text.splitlines() if "SUMMARY:" in line),
                       log_sha256=hashlib.sha256(text.encode()).hexdigest()))
save("sanitizer.csv",checks)
for dst,src in [("ptxas.txt","build.log"),("nvcc-version.txt","nvcc-version.txt"),("sha256.txt","sha256.txt")]:
    save_text(dst,(args.directory/src).read_text())
root=Path(__file__).resolve().parents[1]
files=[args.directory/"audit",root/"scripts/k32_audit.cu",root/"src/kernels/10_kernel_warptiling.cuh",
       root/"src/kernels/30_kernel_mini_simt.cuh",root/"src/kernels/32_kernel_warp_specialized.cuh"]
save("sha256.csv",[dict(file=str(f.relative_to(root)) if f.is_relative_to(root) else str(f),
                       sha256=hashlib.sha256(f.read_bytes()).hexdigest()) for f in files])
if not (args.output/"environment.txt").exists():
    out=io.StringIO()
    for command in [["nvcc","--version"],["ncu","--version"],["git","rev-parse","HEAD"],
                    ["nvidia-smi","--query-gpu=name,uuid,driver_version","--format=csv"]]:
        r=subprocess.run(command,text=True,capture_output=True)
        if r.returncode: raise RuntimeError(r.stderr)
        out.write(" ".join(command)+"\n"+r.stdout+"\n")
    out.write("nvcc -O3 -DNDEBUG -std=c++17 -arch=sm_86 -lineinfo -DSGEMM_K10_MIN_BLOCKS=2 -Xptxas=-v\n")
    out.write("seed=20261013 alpha=0.5 beta=3 reset-C-before-every-call warmup=10 repeat=50\n")
    out.write("cold: NVTX measure/ kernel-replay cache=all clock=base; steady: application-replay cache=none clock=none\n")
    save_text("environment.txt",out.getvalue())
for row in summary: print(row)
print(f"Saved {len(profiles)} Nsight rows and {len(checks)} sanitizer outcomes")
