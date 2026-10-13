#!/usr/bin/env python3
"""K32 experiment selection, resource gate, and the unchanged K30 fair harness."""
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
resources={r["mode"]:r for r in csv.DictReader((args.directory/"resources.csv").open())}
env=dict(os.environ); env.pop("NVIDIA_TF32_OVERRIDE",None)
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
    command=[sys.executable,str(ROOT/"scripts/run_k30_audit.py"),args.action,
             "--binary",str(binary),"--output",str(output),"--modes",*modes,
             "--rounds",str(args.rounds),"--policies",*args.policies,
             "--extra-metrics","smsp__warps_issue_stalled_wait.avg.per_cycle_active"]
    raise SystemExit(subprocess.call(command,env=env))
