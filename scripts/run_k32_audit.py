#!/usr/bin/env python3
# 【文件导读】K32 实验的统一入口与资源筛选
# 给初学者：audit 在这里指一套实验检查，不是另一个算法；构建、验算、计时、性能采样是不同步骤。
# 运行示例：python3 scripts/run_k32_audit.py correctness --directory benchmark_results/k32-new-session。
# 前置：先 build_k32_audit.sh，再 analyze_k32_binary.py；目录内必须有 audit 与 resources.csv。
# action：resources 查询函数属性；correctness/fair/profile/source/memcheck/sync 执行对应阶段。
# resources 将运行时 registers 与编译统计交叉核对，写 runtime-resources.csv；理论 blocks/SM 是上限。
# 其他 action 从 ORDER 选择模式，按 resource_gate 跳过有 spill 或超预算的版本。
# 实际尺寸、beta、预热、计时、NVTX 和 Sanitizer 流程复用 run_k30_audit.py，避免另写一套口径。
# 参数：--modes 可选子集，--rounds 控制轮数，--policies cold/steady 控制采样策略，--output 可指定新目录。
# 结果默认放 tests/timing/profiles/source/sanitizer；子集实验不自动满足完整 summarize 的发布要求。
# 运行会占用 GPU；它不是只读报告脚本，重复动作应使用新的输出位置。
# 中文总索引：docs/zh/23-代码与脚本使用索引.md；脚本命令默认在仓库根目录执行。

"""选择 K32 实验并检查资源约束，沿用 K30 的公平计时框架。"""
import argparse
import csv
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
ORDER=["strict","k10","k30","serial","overlap","single"]
p=argparse.ArgumentParser()
p.add_argument("action",choices=["resources","correctness","fair","profile","source","memcheck","sync"])
p.add_argument("--directory",type=Path,required=True)
p.add_argument("--output",type=Path)
p.add_argument("--modes",nargs="+")
p.add_argument("--rounds",type=int,default=6)
p.add_argument("--policies",nargs="+",default=["cold"],choices=["cold","steady"])
args=p.parse_args()
binary=args.directory/"audit"
# 先读 analyze_k32_binary.py 生成的资源表；因此即使运行 resources 动作，也要先完成静态分析。
# resource_gate 是本实验预设的筛选门槛，不是设备通用上限，更不是性能已经合格。
resources={r["mode"]:r for r in csv.DictReader((args.directory/"resources.csv").open())}
env=dict(os.environ); env.pop("NVIDIA_TF32_OVERRIDE",None)
# 向 CUDA 查询已编译核的资源并与编译日志交叉检查；这是查询动作，不测 GFLOPS。
if args.action=="resources":
    out=subprocess.check_output([str(binary),"resources"],text=True,env=env)
    rows=[]
    for line in out.splitlines():
        fields=line.split(",")
        if fields[0]!="RESOURCE": continue
        mode=fields[1]
        row=dict(mode=mode,threads=int(fields[2]),registers=int(fields[3]),
                 dynamic_shared_bytes=int(fields[4]),local_bytes=int(fields[5]),
                 theoretical_blocks_per_sm=int(fields[6]))
        assert row["registers"]==int(resources[mode]["registers"])
        rows.append(row)
    assert len(rows)==len(resources)
    with (args.directory/"runtime-resources.csv").open("x") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator="\n")
        w.writeheader();w.writerows(rows)
    print(out,end="")
else:
    selected=args.modes or [m for m in ORDER if m=="strict" or m in resources]
    # 只把通过资源门槛的候选交给后续实验；被跳过的是失败/不合格候选，不应在报告中当作未测试的赢家。
    modes=[]
    for mode in selected:
        if mode not in ORDER: p.error(f"Unknown mode: {mode}")
        if mode!="strict" and resources[mode]["resource_gate"]!="PASS":
            print(f"Skipping {mode}: compiler resource gate failed",flush=True)
            continue
        modes.append(mode)
    if not modes: p.error("No variants passed the resource gate")
    folders={"correctness":"tests","fair":"timing","profile":"profiles",
             "memcheck":"sanitizer","sync":"sanitizer","source":"source"}
    output=args.output or args.directory/folders[args.action]
    # 复用 K30 的统一执行器，保持输入、C 复位、event 和 warmup/repeat 规则一致；这里只选择模式与目录。
    command=[sys.executable,str(ROOT/"scripts/run_k30_audit.py"),args.action,
             "--binary",str(binary),"--output",str(output),"--modes",*modes,
             "--rounds",str(args.rounds),"--policies",*args.policies,
             "--extra-metrics","smsp__warps_issue_stalled_wait.avg.per_cycle_active"]
    raise SystemExit(subprocess.call(command,env=env))
