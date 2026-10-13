#!/usr/bin/env python3
"""Run isolated SIMT reference; exclusive output creation preserves past runs."""
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

def run(command, name):
    with (args.output / name).open("x") as f:
        result = subprocess.run([str(x) for x in command], env=env,
                                stdout=f, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"Failed ({result.returncode}): {name}")
    return (args.output / name).read_text()

def bench(mode, n, beta, repeats, warmup):
    return [args.binary, mode, n, beta, repeats, warmup]

def snapshot():
    return subprocess.check_output([
        "nvidia-smi", "--query-gpu=temperature.gpu,clocks.current.sm,clocks.current.memory",
        "--format=csv,noheader,nounits"], text=True).strip().replace(" ", "")

if args.action in ("correctness", "fair"):
    with (args.output / f"{args.action}.csv").open("x") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["round", "position", "mode", "size", "beta", "ms", "gflops",
                    "max_abs_error", "correctness", "gpu_before", "gpu_after"])
        if args.action == "correctness":
            cases = [(0,0,m,n,b,1,0) for m in MODES[1:]
                     for n in [128,256,512,1024,2048,4096] for b in [0,1,3]]
        else:
            # Bring GPU out of idle before measuring the first randomized-order round.
            run(bench("k10",4096,3,100,20), "thermal_warmup.log")
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
