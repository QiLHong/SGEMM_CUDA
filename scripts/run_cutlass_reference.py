#!/usr/bin/env python3
# 【文件导读】运行公开 CUTLASS SIMT 对照
# 给初学者：audit 在这里指一套实验检查，不是另一个算法；构建、验算、计时、性能采样是不同步骤。
# 运行：python3 scripts/run_cutlass_reference.py correctness --binary <已构建的reference> --output <全新结果子目录>。
# 模式：strict/k10/k21/cutlass。
# correctness 枚举六尺寸 × beta=0/1/3；fair 使用 4096³、beta=3、warmup=10、repeat=50。
# fair 先热身，再按轮数循环移位模式顺序；不是随机抽样，每条 CSV 是轮内平均结果。
# run 以排他方式创建日志，调用失败则中止；RESULT 字段来自 CUDA harness 的预先数值检查和 event 计时。
# source 采集逐 PC 指标/SASS；profile 采集整核指标，通过 NVTX measure/ 选定计时 launch。
# cold 清缓存并控制时钟、kernel replay；steady 保留缓存/时钟策略并用 application replay，不能混表。
# memcheck 只检查 cutlass 的 128/256/4096，beta=3；没有提供 sync action。
# 需要 CUDA GPU、nvidia-smi；采样/工具动作还需 ncu 或 compute-sanitizer。日志失败不能只看 CSV 中已有行。
# 中文总索引：docs/zh/23-代码与脚本使用索引.md；脚本命令默认在仓库根目录执行。

"""运行独立的 SIMT 参考实验；输出文件只允许新建，避免覆盖过去的运行结果。"""
import argparse
import csv
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument("action", choices=["correctness", "fair", "memcheck", "profile", "source"])
p.add_argument("--output", type=Path, required=True)
p.add_argument("--binary", type=Path, required=True)
p.add_argument("--rounds", type=int, default=7)
args = p.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
env = dict(os.environ)
env.pop("NVIDIA_TF32_OVERRIDE", None)
MODES = ["strict", "k10", "k21", "cutlass"]

# 统一执行外部命令：完整 stdout/stderr 留到日志，open('x') 拒绝同名覆盖。
# 子进程失败立即中止；已有部分 CSV 行不代表整组实验完成。
def run(command, name):
    with (args.output / name).open("x") as f:
        result = subprocess.run([str(x) for x in command], env=env,
                                stdout=f, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"Failed ({result.returncode}): {name}")
    return (args.output / name).read_text()

# 这里只组装参数列表，真正的 GPU 计算在编译好的 .cu 程序中；返回列表并不会开始测量。
def bench(mode, n, beta, repeats, warmup):
    return [args.binary, mode, n, beta, repeats, warmup]

# 记录温度与时钟来排查设备状态漂移；只是前后快照，不是持续采样，也不保证整轮频率恒定。
def snapshot():
    return subprocess.check_output([
        "nvidia-smi", "--query-gpu=temperature.gpu,clocks.current.sm,clocks.current.memory",
        "--format=csv,noheader,nounits"], text=True).strip().replace(" ", "")

# correctness 回答算得对不对，fair 回答同条件下多快；两者使用不同尺寸/次数表，不能混用。
if args.action in ("correctness", "fair"):
    with (args.output / f"{args.action}.csv").open("x") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["round", "position", "mode", "size", "beta", "ms", "gflops",
                    "max_abs_error", "correctness", "gpu_before", "gpu_after"])
        if args.action == "correctness":
            cases = [(0,0,m,n,b,1,0) for m in MODES[1:]
                     for n in [128,256,512,1024,2048,4096] for b in [0,1,3]]
        else:
            # 第一轮随机顺序计时前先预热 GPU，减少从空闲状态恢复带来的影响。
            run(bench("k10",4096,3,100,20), "thermal_warmup.log")
            # 循环移位顺序而非随机：如 A/B/C，下一轮 B/C/A。减少总是第一个跑的偏差，但仍需多轮看波动。
            cases = [(r+1,i+1,MODES[(i+r)%4],4096,3,50,10)
                     for r in range(args.rounds) for i in range(4)]
        for r, pos, mode, n, beta, repeat, warm in cases:
            before = snapshot()
            out = run(bench(mode,n,beta,repeat,warm),
                      f"{args.action}_{r}_{mode}_{n}_{beta}.log")
            fields = next(line.split(",")[1:] for line in out.splitlines()
                          if line.startswith("RESULT,"))
            w.writerow([r,pos,*fields,before,snapshot()]); f.flush()
            print(",".join(fields), flush=True)
elif args.action == "memcheck":
    for n in [128,256,4096]:
        run(["compute-sanitizer","--tool","memcheck","--error-exitcode","99",
             *bench("cutlass",n,3,1,0)], f"memcheck_{n}.log")
        print(f"memcheck {n} PASS", flush=True)
# source 导出逐条 SASS 的统计，适合找哪一条 LDS/FFMA 在等待；不要与整核指标的总数混算。
elif args.action == "source":
    for mode in ["strict", "k10", "cutlass"]:
        dest = args.output / f"source_{mode}"
        if dest.with_suffix(".ncu-rep").exists(): raise FileExistsError(dest)
        run(["ncu","--nvtx","--nvtx-include","measure/","--launch-count","1",
             "--replay-mode","kernel","--cache-control","all","--clock-control","base",
             "--section","SourceCounters","--section","InstructionStats","-o",dest,
             *bench(mode,4096,3,1,10)], f"source_{mode}.log")
        run(["ncu","--import",dest.with_suffix(".ncu-rep"),"--page","source",
             "--print-source","sass","--csv"], f"source_{mode}.csv")
        print(f"source {mode} done", flush=True)
else:
    # 指标表不是越大越好：FP32 活跃、issue、等待、shared、local 分别回答不同问题。
    # issue_inst0 是没有发射指令的比例，不能无条件改叫没有 eligible warp；local 访问还需结合编译 spill。
    metrics = [
        "gpu__time_duration.sum", "launch__registers_per_thread", "launch__shared_mem_per_block",
        "sm__warps_active.avg.pct_of_peak_sustained_active",
        "sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active",
        "smsp__issue_active.avg.pct_of_peak_sustained_active",
        "smsp__issue_inst0.avg.pct_of_peak_sustained_active",
        "smsp__warps_eligible.avg.per_cycle_active",
        "smsp__warps_issue_stalled_dispatch_stall.avg.per_cycle_active",
        "smsp__warps_issue_stalled_barrier.avg.per_cycle_active",
        "smsp__warps_issue_stalled_long_scoreboard.avg.per_cycle_active",
        "smsp__warps_issue_stalled_short_scoreboard.avg.per_cycle_active",
        "dram__bytes_read.sum", "lts__t_sector_hit_rate.pct", "smsp__inst_executed.sum",
        "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum",
        "smsp__inst_executed_op_ldgsts.sum",
        "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum",
        "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum",
        "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum",
        "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum",
        "l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum",
        "l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum"]
    for policy in ["cold", "steady"]:
        for mode in MODES:
            dest = args.output / f"ncu_{policy}_{mode}"
            if dest.with_suffix(".ncu-rep").exists():
                raise FileExistsError(dest)
            replay = (["--replay-mode","kernel","--cache-control","all","--clock-control","base"]
                      if policy == "cold" else
                      ["--replay-mode","application","--cache-control","none","--clock-control","none"])
            run(["ncu","--nvtx","--nvtx-include","measure/","--launch-count","1",
                 *replay,"--metrics",",".join(metrics),"-o",dest,
                 *bench(mode,4096,3,1,10)], f"ncu_{policy}_{mode}.log")
            run(["ncu","--import",dest.with_suffix(".ncu-rep"),"--page","raw","--csv"],
                f"ncu_{policy}_{mode}.csv")
            run(["ncu","--import",dest.with_suffix(".ncu-rep"),"--page","source",
                 "--print-source","sass","--csv"], f"sass_{policy}_{mode}.csv")
            print(f"profile {policy} {mode} done", flush=True)
