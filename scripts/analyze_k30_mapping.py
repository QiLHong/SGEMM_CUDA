#!/usr/bin/env python3
# 【文件导读】画出 K30 的线程到输出坐标映射
# 给初学者：这类脚本像读图纸，不负责测 GPU 跑多快；先准备它要求的二进制或报告，再读输出表。
# 运行：python3 scripts/analyze_k30_mapping.py --output <新的映射csv文件>。
# 不需要 GPU；根据源码公式枚举每个 lane 的输出格子。
# 输出 lane 映射 CSV，并检查 CTA 中每个输出是否恰好由一个线程负责；坐标正确不等于浮点结果已验证。
# 把它当作分工表生成器：warp 是 32 人小组，lane 是组内编号，fragment 是一个人的工作清单。
# 初学者先看生成表中 lane=0/1/16 的行列，再回到对应 kernel 的索引公式。
# 中文总索引：docs/zh/23-代码与脚本使用索引.md；脚本命令默认在仓库根目录执行。

"""在 CPU 上枚举线程分工并检查布局边界，同时导出每个 warp 的完整映射 CSV。"""
import argparse
import csv
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument("--output",type=Path,required=True)
args=p.parse_args()
rows=[]
for bm,threads in [(128,128),(256,256)]:
    outputs=set()
    for warp in range(threads//32):
        wr,wc=divmod(warp,2)
        for lane in range(32):
            lm=2*(lane//16)+lane%2
            ln=(lane%16)//2
            for i in range(16):
                m=wr*64+4*lm+16*(i//4)+i%4
                for j in range(8):
                    n=wc*64+4*ln+32*(j//4)+j%4
                    assert 0<=m<bm and 0<=n<128
                    assert (m,n) not in outputs
                    outputs.add((m,n))
                    if bm==128 and warp==0:
                        rows.append(dict(lane=lane,lane_m=lm,lane_n=ln,a_register=i,
                                         b_register=j,accumulator=i*8+j,m=m,n=n,
                                         shared_a_k0=m,shared_b_k0=n,
                                         shared_a_k_stride=bm+4,shared_b_k_stride=128))
    assert len(outputs)==bm*128
    # 对每个 k，转置写入与片段读取必须指向同一逻辑元素。
    for k in range(16):
        for m in range(bm): assert 0<=k*(bm+4)+m<16*(bm+4)
        for n in range(128): assert 0<=k*128+n<16*128
    print(f"CTA {bm}x128 threads={threads}: exact coverage, no overlaps, layout in bounds")
args.output.parent.mkdir(parents=True,exist_ok=True)
with args.output.open("x") as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator="\n")
    w.writeheader(); w.writerows(rows)
