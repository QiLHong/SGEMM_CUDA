#!/usr/bin/env python3
# 【文件导读】K31 的资源门槛与静态机器码审计
# 给初学者：这类脚本像读图纸，不负责测 GPU 跑多快；先准备它要求的二进制或报告，再读输出表。
# 运行：python3 scripts/analyze_k31_binary.py <build_k31_audit.sh 的输出目录>。
# 输入：build.log 和 audit；需 cuobjdump，不启动 GPU workload。
# 流程：识别模板符号对应的 mode → 读取 ptxas registers/stack/spill → 反汇编每个函数。
# resource_gate 要求 registers<=240 且 spill load/store 都为零；门槛是本阶段规则，不是普适最优值。
# 输出：resources.csv、static-sass.csv、schedule.csv、sass_<mode>.txt；原始 SASS 留给进一步依赖分析。
# shape 中 shared_bytes 是源码布局计算的用户容量；运行时/NCU 口径另外核对。
# 主循环识别依赖 2048 次 FFMA 的展开形态；编译器改变结构会触发断言，不能直接删掉断言继续解释。
# 静态 FFMA/LDS 数和代码哈希能说明生成了什么；不能独立证明哪个版本更快。
# 中文总索引：docs/zh/23-代码与脚本使用索引.md；脚本命令默认在仓库根目录执行。

"""Record exact compiler resources and SASS; no GPU or performance inference."""
import argparse
import collections
import csv
import hashlib
from pathlib import Path
import re
import subprocess

NAMES={0:"control",1:"late",2:"vectors",3:"stage3",4:"stage4",5:"cta256"}
# 把编译器的长模板符号翻译成人能读的 mode；名字匹配错误会把不同实验混为一谈。
def mode_for(name):
    if "sgemmWarptiling" in name: return "k10"
    if "sgemmK30" in name: return "k30"
    if "sgemmK31Budget" in name:
        return "r"+re.search(r"BudgetILi(\d+)EE",name)[1]
    return NAMES[int(re.search(r"sgemmK31ILi(\d+)EE",name)[1])]

# 从已知 mode 还原线程/分块/stage 配置；shared_bytes 是源码布局推导的用户字节数，不是 profiler 值。
def shape(mode):
    threads=256 if mode=="cta256" else 128
    stages=3 if mode=="stage3" else 4 if mode=="stage4" else 1 if mode=="k10" else 2
    bm=256 if threads==256 else 128
    return dict(threads=threads,bm=bm,bn=128,bk=16,stages=stages,
                shared_bytes=(bm+4+128)*16*4*stages)

# 用 open('x') 新建 CSV，已有文件就报错，避免悄悄覆盖先前证据。
def write(path,rows):
    with path.open("x") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator="\n")
        w.writeheader(); w.writerows(rows)

# 先读 ptxas 构建日志做资源筛选，再读 SASS 统计指令和主循环搬运位置；全程不跑 GPU。
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
