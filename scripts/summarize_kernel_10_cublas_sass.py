#!/usr/bin/env python3
# 【文件导读】历史 K10/cuBLAS 的 SASS 指令数量对照
# 给初学者：summarize 只整理已有结果，不会把失败变成功，也不会重新测出 GFLOPS；先确认输入文件齐全。
# 运行前：python3 scripts/summarize_kernel_10_cublas_sass.py --help；用 --results 与 --tag 指定历史报告目录和命名标签。
# 输入匹配标签的 Nsight 报告，调用 ncu 导出 SASS；需要已安装 ncu，本身不重新跑 GPU benchmark。
# 输出 CSV 到终端，按 FFMA/LDS/整数等 opcode 统计；默认 tag 保留当时实验日期，换报告要显式指定。
# SASS 是 GPU 最终机器指令的文本形式；本脚本数的是静态出现次数，不含循环执行次数权重。
# 两个库核有不同尾部/分支时，总静态条数不能直接推出吞吐差异。
# 中文总索引：docs/zh/23-代码与脚本使用索引.md；脚本命令默认在仓库根目录执行。

"""统计从 K10/cuBLAS 的 NCU 报告导出的静态 SASS 操作码。

cuBLAS 内核包含多个谓词执行路径或分支。对整个函数计数只能说明存在
哪些指令族，不能代表一次实际运行中执行了多少条指令。"""

import argparse
import csv
import io
import re
import subprocess
import sys
from pathlib import Path

OPCODES = ["FFMA", "LDS", "LDGSTS", "LDGDEPBAR", "DEPBAR", "BAR", "IMAD",
           "IADD3", "UIADD3", "LEA", "SHF", "USHF", "MOV", "UMOV", "STS",
           "LDG", "NOP"]


# 按 opcode 汇总 SASS 的静态数量；循环里的指令实际执行多少次仍需动态计数。
def summarize(path):
    output = subprocess.run(
        ["ncu", "--import", str(path), "--page", "source", "--print-source", "sass", "--csv"],
        check=True, text=True, capture_output=True
    ).stdout
    rows = list(csv.reader(io.StringIO(output)))
    kernel_name = rows[0][1]
    counts = {}
    total = 0
    for row in rows:
        if len(row) != 2 or not row[0].startswith("0x"):
            continue
        source = row[1].strip()
        source = re.sub(r"^@!?P\d+\s+", "", source)
        match = re.match(r"([A-Z][A-Z0-9_.]*)", source)
        if not match:
            continue
        base = match.group(1).split(".", 1)[0]
        counts[base] = counts.get(base, 0) + 1
        total += 1
    return {"kernel_name": kernel_name, "static_instructions": total,
            **{opcode: counts.get(opcode, 0) for opcode in OPCODES}}


# 解析报告路径/标签，逐模式汇总并输出表；这一步不运行新的性能实验。
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("benchmark_results"))
    parser.add_argument("--tag", default="20261012-final2")
    args = parser.parse_args()
    writer = csv.DictWriter(sys.stdout,
                            fieldnames=["mode", "kernel_name", "static_instructions", *OPCODES],
                            lineterminator="\n")
    writer.writeheader()
    for mode in ["k10", "strict"]:
        path = args.results / f"ncu_k10_micro_{mode}_{args.tag}.ncu-rep"
        writer.writerow({"mode": mode, **summarize(path)})


if __name__ == "__main__":
    main()
