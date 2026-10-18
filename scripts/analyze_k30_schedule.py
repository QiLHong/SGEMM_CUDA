#!/usr/bin/env python3
# 【文件导读】从 K30 汇编识别加载与计算的先后位置
# 给初学者：这类脚本像读图纸，不负责测 GPU 跑多快；先准备它要求的二进制或报告，再读输出表。
# 运行：python3 scripts/analyze_k30_schedule.py <audit二进制> --output <新的调度csv文件>。
# 需要 CUDA 的 cuobjdump，但无需在 GPU 上执行 benchmark。
# 识别含 2048 条 FFMA 的主循环并记录 LDG/STS/BAR 在循环中的位置；二进制结构改变时识别假设须重新检查。
# 指令距离是文本顺序上的间隔，不是时钟周期；load 出现在 FFMA 前面不等于硬件真的隐藏了等待。
# 初学者先看生成表中 lane=0/1/16 的行列，再回到对应 kernel 的索引公式。
# 中文总索引：docs/zh/23-代码与脚本使用索引.md；脚本命令默认在仓库根目录执行。

"""Static mainloop LDG placement: instruction positions, not latency/cycle claims."""
import argparse
import csv
import re
import subprocess
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument("binary",type=Path)
p.add_argument("--output",type=Path,required=True)
args=p.parse_args()
sass=subprocess.check_output(["cuobjdump","--dump-sass",str(args.binary)],text=True)
rows=[]
for block in sass.split("Function : ")[1:]:
    name=block.splitlines()[0]
    mode=("k10" if "sgemmWarptiling" in name else "v4" if "ShapeILi256" in name
          else "v1" if "ELi1EE" in name else "v2" if "ELi2EE" in name else "v3")
    ins=[(int(pc,16),text.strip()) for pc,text in
         re.findall(r"/\*([0-9a-f]+)\*/\s+(.*?)\s*;",block)]
    loops=[]
    for pc,text in ins:
        branch=re.search(r"\bBRA (0x[0-9a-f]+)",text)
        if branch and int(branch[1],16)<pc:
            start=int(branch[1],16)
            body=[(a,t) for a,t in ins if start<=a<=pc]
            if sum("FFMA" in t for _,t in body)==2048:
                loops.append((start,pc,body))
    assert len(loops)==1,(name,len(loops))
    start,end,body=loops[0]
    for pos,(pc,text) in enumerate(body):
        if not re.search(r"\b(LDG|STS|BAR)(\.|\s)",text): continue
        rows.append(dict(mode=mode,loop_start=hex(start),loop_end=hex(end),pc=hex(pc),
                         instruction=text,ffma_before=sum("FFMA" in t for _,t in body[:pos]),
                         ffma_after=sum("FFMA" in t for _,t in body[pos+1:])))
    ldg=[r for r in rows if r["mode"]==mode and "LDG." in r["instruction"]]
    print(mode,"mainloop LDG FFMA-after range",min(r["ffma_after"] for r in ldg),
          max(r["ffma_after"] for r in ldg))
args.output.parent.mkdir(parents=True,exist_ok=True)
with args.output.open("x") as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator="\n")
    w.writeheader();w.writerows(rows)
