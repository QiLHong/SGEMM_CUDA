#!/usr/bin/env python3
"""Record exact compiler resources and SASS; no GPU or performance inference."""
import argparse
import collections
import csv
import hashlib
from pathlib import Path
import re
import subprocess

NAMES={0:"control",1:"late",2:"vectors",3:"stage3",4:"stage4",5:"cta256"}
def mode_for(name):
    if "sgemmWarptiling" in name: return "k10"
    if "sgemmK30" in name: return "k30"
    if "sgemmK31Budget" in name:
        return "r"+re.search(r"BudgetILi(\d+)EE",name)[1]
    return NAMES[int(re.search(r"sgemmK31ILi(\d+)EE",name)[1])]

def shape(mode):
    threads=256 if mode=="cta256" else 128
    stages=3 if mode=="stage3" else 4 if mode=="stage4" else 1 if mode=="k10" else 2
    bm=256 if threads==256 else 128
    return dict(threads=threads,bm=bm,bn=128,bk=16,stages=stages,
                shared_bytes=(bm+4+128)*16*4*stages)

def write(path,rows):
    with path.open("x") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator="\n")
        w.writeheader(); w.writerows(rows)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("directory",type=Path)
    args=p.parse_args()
    resources=[]
    for line in (args.directory/"build.log").read_text().splitlines():
        if "Compiling entry function" in line:
            mode=mode_for(line)
        if "bytes spill stores" in line:
            stack,stores,loads=map(int,re.search(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads",line).groups())
        m=re.search(r"Used (\d+) registers",line)
        if m:
            resources.append(dict(mode=mode,registers=int(m[1]),stack_bytes=stack,
                                  spill_store_bytes=stores,spill_load_bytes=loads,
                                  resource_gate="PASS" if stores==loads==0 and int(m[1])<=240 else "FAIL",
                                  **shape(mode)))
    assert len(resources)==13, len(resources)
    write(args.directory/"resources.csv",resources)
    text=subprocess.check_output(["cuobjdump","--dump-sass",str(args.directory/"audit")],text=True)
    counts,schedule=[],[]
    for block in text.split("Function : ")[1:]:
        name=block.splitlines()[0]; mode=mode_for(name)
        ins=[(int(pc,16),s.strip()) for pc,s in re.findall(r"/\*([0-9a-f]+)\*/\s+(.*?)\s*;",block)]
        ops=collections.Counter(re.sub(r"^@!?U?P(?:\d+|T)\s+","",s).split()[0].split(".")[0] for _,s in ins)
        raw="\n".join(l.strip() for l in block.splitlines() if "/*" in l)
        with (args.directory/f"sass_{mode}.txt").open("x") as f: f.write(raw+"\n")
        counts.append(dict(mode=mode,static_total=len(ins),
                           **{op:ops[op] for op in ["FFMA","LDS","LDG","STS","LDGSTS","DEPBAR","BAR","IMAD","IADD3","LEA","MOV","HMMA","MMA","LDL","STL"]},
                           sass_sha256=hashlib.sha256(raw.encode()).hexdigest()))
        loops=[]
        for pc,s in ins:
            branch=re.search(r"\bBRA (0x[0-9a-f]+)",s)
            if branch and int(branch[1],16)<pc:
                start=int(branch[1],16)
                body=[(a,t) for a,t in ins if start<=a<=pc]
                if sum("FFMA" in t for _,t in body)==2048: loops.append((start,pc,body))
        assert len(loops)==1,(mode,len(loops))
        start,end,body=loops[0]
        for pos,(pc,s) in enumerate(body):
            if not re.search(r"\b(LDG|STS|BAR)(\.|\s)",s): continue
            schedule.append(dict(mode=mode,loop_start=hex(start),loop_end=hex(end),pc=hex(pc),instruction=s,
                                 ffma_before=sum("FFMA" in t for _,t in body[:pos]),
                                 ffma_after=sum("FFMA" in t for _,t in body[pos+1:])))
    write(args.directory/"static-sass.csv",counts)
    write(args.directory/"schedule.csv",schedule)
    for r in resources: print(r)
    for r in counts: print({k:r[k] for k in ["mode","static_total","FFMA","LDS","sass_sha256"]})

if __name__=="__main__": main()
