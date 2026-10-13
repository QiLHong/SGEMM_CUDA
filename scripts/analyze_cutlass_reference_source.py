#!/usr/bin/env python3
"""Aggregate NCU per-PC executed instructions; sampled stalls are not cycle counts."""
import argparse
import collections
import csv
from pathlib import Path
import re

p = argparse.ArgumentParser()
p.add_argument("source",type=Path)
p.add_argument("--output",type=Path,required=True)
p.add_argument("--modes",nargs="+",default=["strict","k10","cutlass"])
p.add_argument("--skip-mapping",action="store_true",
               help="For new kernels, use their own independently checked ownership mapping")
args = p.parse_args()
args.output.mkdir(parents=True,exist_ok=True)
stats = ["Instructions Executed", "Predicated-On Thread Instructions Executed",
         "L1 Wavefronts Shared", "L1 Wavefronts Shared Ideal",
         "L1 Wavefronts Shared Excessive", "L2 Theoretical Sectors Global",
         "L2 Theoretical Sectors Global Ideal", "Warp Stall Sampling (Not-issued Samples)"]
stall_names = ["stall_dispatch", "stall_long_sb", "stall_short_sb", "stall_barrier", "stall_wait", "stall_mio"]

def number(value):
    if value in ("", "-", "N/A", None): return 0
    return float(value.replace(",",""))

def write(name,rows):
    with (args.output/name).open("x") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator="\n");w.writeheader();w.writerows(rows)

totals, pcs, widths, hot_lds = [], [], [], []
for mode in args.modes:
    with (args.source/f"source_{mode}.csv").open() as f:
        next(f)
        rows=list(csv.DictReader(f))
    rows=[r for r in rows if r["Address"].startswith("0x")]
    base=int(rows[0]["Address"],16)
    counters=collections.defaultdict(collections.Counter)
    loads=collections.defaultdict(collections.Counter)
    for r in rows:
        original = r["Source"].strip()
        never_active = original.startswith("@!PT ")
        ins=re.sub(r"^@!?U?P(?:\d+|T)\s+","",original)
        full=ins.split()[0]; op=full.split(".")[0]
        # Always-false predicated LDS still consumes issue slots, but loads no data.
        if never_active: op += "_PRED_OFF"
        for name in stats: counters[op][name]+=number(r.get(name))
        counters[op]["static_PCs"]+=1
        if number(r.get("Instructions Executed"))>0: counters[op]["executed_PCs"]+=1
        if op in ["LDS","LDS_PRED_OFF","LDGSTS","LDG","STS"]:
            if never_active: full = "@!PT " + full
            for name in stats: loads[(full,r.get("Access Size",""))][name]+=number(r.get(name))
        executed = number(r.get("Instructions Executed"))
        if full == "LDS.128" and executed > 100000:
            hot_lds.append(dict(mode=mode,offset=hex(int(r["Address"],16)-base),
                                instruction=original,executed=executed,
                                wavefronts=number(r.get("L1 Wavefronts Shared")),
                                wavefronts_per_instruction=number(r.get("L1 Wavefronts Shared"))/executed))
        samples=number(r.get("Warp Stall Sampling (Not-issued Samples)"))
        if samples:
            pcs.append(dict(mode=mode,offset=hex(int(r["Address"],16)-base),instruction=r["Source"].strip(),
                            not_issued_samples=samples,executed=number(r.get("Instructions Executed")),
                            **{s:number(r.get(s+" (Not Issued)")) for s in stall_names}))
    for op,c in sorted(counters.items()):
        totals.append(dict(mode=mode,opcode=op,static_PCs=c["static_PCs"],executed_PCs=c["executed_PCs"],
                           **{s:c[s] for s in stats}))
    for (op,size),c in sorted(loads.items()):
        widths.append(dict(mode=mode,instruction=op,access_size=size,**{s:c[s] for s in stats}))
    print(mode)
    for op in ["FFMA","LDS","LDS_PRED_OFF","LDGSTS","IMAD","IADD3","UIADD3","LEA","MOV","BAR","DEPBAR"]:
        print(op,dict(counters[op]))
write("dynamic-opcodes.csv",totals)
write("memory-instructions.csv",widths)
pcs.sort(key=lambda r:(r["mode"],-r["not_issued_samples"]))
write("pc-stalls.csv",pcs)
write("mainloop-lds.csv",hot_lds)

# Logical warp-relative coordinates from the public MmaSimtPolicy and iterators.
if args.skip_mapping:
    raise SystemExit(0)
mapping=[]
for mode in ["k10","cutlass"]:
    ownership=set()
    for lane in range(32):
        if mode=="k10":
            a=[8*(lane//4)+i for i in range(8)]
            b=[16*s+4*(lane%4)+j for s in range(4) for j in range(4)]
        else:
            lm=2*(lane//16)+lane%2; ln=(lane%16)//2
            a=[4*lm+16*s+i for s in range(4) for i in range(4)]
            b=[4*ln+32*s+j for s in range(2) for j in range(4)]
        for m in a:
            for n in b:
                assert (m,n) not in ownership
                ownership.add((m,n))
        mapping.append(dict(mode=mode,lane=lane,A_rows=" ".join(map(str,a)),
                            B_columns=" ".join(map(str,b)),accumulators=len(a)*len(b)))
    assert ownership=={(m,n) for m in range(64) for n in range(64)}
write("lane-mapping.csv",mapping)
