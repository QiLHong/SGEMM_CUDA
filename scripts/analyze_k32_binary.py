#!/usr/bin/env python3
# 【文件导读】K32 的资源门槛与静态机器码审计
# 给初学者：这类脚本像读图纸，不负责测 GPU 跑多快；先准备它要求的二进制或报告，再读输出表。
# 运行：python3 scripts/analyze_k32_binary.py <build_k32_audit.sh 的输出目录>。
# 输入：build.log 和 audit；需 cuobjdump，不启动 GPU workload。
# 流程：识别模板符号对应的 mode → 读取 ptxas registers/stack/spill → 反汇编每个函数。
# resource_gate 要求 registers<=240 且 spill load/store 都为零；门槛是本阶段规则，不是普适最优值。
# 输出：resources.csv、static-sass.csv、schedule.csv、sass_<mode>.txt；原始 SASS 留给进一步依赖分析。
# shape 中 shared_bytes 是源码布局计算的用户容量；运行时/NCU 口径另外核对。
# K32 分辨 producer/consumer 两条循环；linear_ffma_before/after 只是线性窗口，互斥路径不代表同时执行。
# 静态 FFMA/LDS 数和代码哈希能说明生成了什么；不能独立证明哪个版本更快。
# 中文总索引：docs/zh/23-代码与脚本使用索引.md；脚本命令默认在仓库根目录执行。

"""记录当前二进制的编译资源和 SASS 汇编；不执行 GPU，也不推断性能。"""
import argparse
import collections
import csv
import hashlib
from pathlib import Path
import re
import subprocess
import io


# 区分 serial、overlap、single 的模板布尔参数；保留旧符号格式兼容，未知函数直接拒绝。
def mode_for(name):
    if "sgemmWarptiling" in name: return "k10"
    if "sgemmK30" in name: return "k30"
    if "sgemmK32ILb1ELb1EE" in name: return "single"
    if "sgemmK32ILb1ELb0EE" in name or "sgemmK32ILb1EE" in name: return "overlap"
    if "sgemmK32ILb0ELb0EE" in name or "sgemmK32ILb0EE" in name: return "serial"
    raise ValueError(name)

# 从模式推导静态形状；K32 有 160 个线程，但只有后 128 个沿用 K30 的输出分工。
def shape(mode):
    threads=160 if mode in ("serial","overlap","single") else 128
    stages=1 if mode=="k10" else 2
    bm=128
    return dict(threads=threads,bm=bm,bn=128,bk=16,stages=stages,
                shared_bytes=(bm+4+128)*16*4*stages)

# 先在内存生成完整 CSV，再交给 save_text 检查是否与旧文件一致；不覆盖不同内容。
def write(path,rows):
    f=io.StringIO()
    w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator="\n")
    w.writeheader(); w.writerows(rows)
    save_text(path,f.getvalue())

# 结果已存在时只允许逐字相同；有差异时停下，而不是把历史失败结果改写成新结果。
def save_text(path,text):
    if path.exists():
        assert path.read_text()==text, f"Refusing to overwrite {path}"
    else:
        with path.open("x") as f: f.write(text)

# build.log 告诉我们寄存器/spill；SASS 告诉我们编译器排了哪些指令；它们都不能替代实际计时。
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
    assert len(resources) in (4,5), len(resources)
    write(args.directory/"resources.csv",resources)
    text=subprocess.check_output(["cuobjdump","--dump-sass",str(args.directory/"audit")],text=True)
    counts,schedule=[],[]
    for block in text.split("Function : ")[1:]:
        name=block.splitlines()[0]; mode=mode_for(name)
        ins=[(int(pc,16),s.strip()) for pc,s in re.findall(r"/\*([0-9a-f]+)\*/\s+(.*?)\s*;",block)]
        ops=collections.Counter(re.sub(r"^@!?U?P(?:\d+|T)\s+","",s).split()[0].split(".")[0] for _,s in ins)
        raw="\n".join(l.strip() for l in block.splitlines() if "/*" in l)
        save_text(args.directory/f"sass_{mode}.txt",raw+"\n")
        counts.append(dict(mode=mode,static_total=len(ins),
                           **{op:ops[op] for op in ["FFMA","LDS","LDG","STS","LDGSTS","DEPBAR","BAR","IMAD","IADD3","LEA","MOV","HMMA","MMA","LDL","STL"]},
                           sass_sha256=hashlib.sha256(raw.encode()).hexdigest()))
        loops=[]
        for pc,s in ins:
            branch=re.search(r"\bBRA (0x[0-9a-f]+)",s)
            if branch and int(branch[1],16)<pc:
                start=int(branch[1],16)
                body=[(a,t) for a,t in ins if start<=a<=pc]
                fmas=sum("FFMA" in t for _,t in body)
                if fmas>=2048 or (mode in ("serial","overlap","single") and any("LDG." in t for _,t in body)):
                    loops.append((start,pc,body,fmas))
        assert len(loops)==(2 if mode in ("serial","overlap","single") else 1),(mode,len(loops))
        for start,end,body,fmas in loops:
            # 线性地址区间不能当作动态执行时的指令间隔：
            # 编译器为 serial 生成了两条互斥的计算路径，一次执行不会同时经过两条。
            role="producer" if fmas==0 else "consumer" if mode in ("serial","overlap","single") else "combined"
            for pos,(pc,s) in enumerate(body):
                if not re.search(r"\b(LDG|STS|BAR|CALL)(\.|\s)",s): continue
                schedule.append(dict(mode=mode,role=role,loop_start=hex(start),loop_end=hex(end),pc=hex(pc),instruction=s,
                                     linear_ffma_before=sum("FFMA" in t for _,t in body[:pos]),
                                     linear_ffma_after=sum("FFMA" in t for _,t in body[pos+1:])))
    write(args.directory/"static-sass.csv",counts)
    write(args.directory/"schedule.csv",schedule)
    for r in resources: print(r)
    for r in counts: print({k:r[k] for k in ["mode","static_total","FFMA","LDS","sass_sha256"]})

if __name__=="__main__": main()
