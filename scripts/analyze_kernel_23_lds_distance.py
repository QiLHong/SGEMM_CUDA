#!/usr/bin/env python3
# 【文件导读】比较 K23 共享布局变化后的 load-use 静态距离
# 给初学者：这类脚本像读图纸，不负责测 GPU 跑多快；先准备它要求的二进制或报告，再读输出表。
# 运行：python3 scripts/analyze_kernel_23_lds_distance.py。
# 读取固定 build-k23-v1/v2/v3/v4/sgemm；V1 的符号来自 K21 iterator，需先构建变体。
# 终端输出每种模式的 LDS.128 到 FFMA 首次使用间隔；需要 cuobjdump，不运行计时。
# 距离少不一定更好，距离多也可能占用更多寄存器；它用于解释 SASS，不用于单独选最快实现。
# 初学者先看生成表中 lane=0/1/16 的行列，再回到对应 kernel 的索引公式。
# 中文总索引：docs/zh/23-代码与脚本使用索引.md；脚本命令默认在仓库根目录执行。

"""Static LDS.128 to first FFMA use distance (instructions, not cycles)."""
import csv
import re
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INST = re.compile(r"/\*([0-9a-f]+)\*/\s+(?:@[!P0-9]+\s+)?([A-Z0-9_.]+)\s+(.*?)\s*;")
REG = re.compile(r"\bR(\d+)\b")
MODES = {
    "k10": (1, "_Z15sgemmWarptilingI"),
    "v1": (1, "_Z16sgemmK21Iterator"),
    "v2": (2, "_Z30sgemmK23SharedFragmentIteratorILi2EE"),
    "v3": (3, "_Z30sgemmK23SharedFragmentIteratorILi3EE"),
    "v4": (4, "_Z30sgemmK23SharedFragmentIteratorILi4EE"),
}


# 只分析 mode 对应的函数；V1 实际复用 K21，不能只搜索 K23 字符串。
def instructions(binary, marker):
    raw = subprocess.run(["cuobjdump", "--dump-sass", str(binary)], check=True,
                         text=True, capture_output=True).stdout
    on = False
    result = []
    for line in raw.splitlines():
        if "Function : " in line:
            if on:
                break
            on = marker in line
            continue
        if on and (match := INST.search(line)):
            result.append((match.group(2), match.group(3)))
    return result


writer = csv.writer(sys.stdout, lineterminator="\n")
writer.writerow(["mode", "lds128", "consumed_by_ffma", "min_distance",
                 "median_distance", "mean_distance", "max_distance"])
for mode, (variant, marker) in MODES.items():
    insts = instructions(ROOT / f"build-k23-v{variant}/sgemm", marker)
    values = []
    for index, (op, operands) in enumerate(insts):
        if op != "LDS.128":
            continue
        match = REG.search(operands.split(",", 1)[0])
        if not match:
            continue
        first = int(match.group(1))
        written = set(range(first, first + 4))
        for use_index in range(index + 1, len(insts)):
            use_op, use_operands = insts[use_index]
            if use_op != "FFMA":
                continue
            parts = use_operands.split(",")
            sources = {int(r) for part in parts[1:] for r in REG.findall(part)}
            if written & sources:
                values.append(use_index - index - 1)
                break
    writer.writerow([mode, sum(op == "LDS.128" for op, _ in insts), len(values),
                     min(values), statistics.median(values),
                     f"{statistics.mean(values):.2f}", max(values)])
