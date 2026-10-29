#!/usr/bin/env python3
# 【文件导读】比较 K19 的 shared load 到第一次乘加使用的间隔
# 给初学者：这类脚本像读图纸，不负责测 GPU 跑多快；先准备它要求的二进制或报告，再读输出表。
# 运行：python3 scripts/analyze_kernel_19_sass_dependencies.py。
# 读取固定 build-prefetch-v1/v2/v3/sgemm；先构建这些变体，需要 cuobjdump，不需要运行 GPU。
# 终端输出 K10 与 K19 变体的 LDS.128→FFMA 距离分布；一条 LDS.128 按四个连续物理寄存器处理。
# 只是沿汇编文本寻找后续使用的近似，不是完整控制流/重命名分析；配合 Nsight 动态指标看，不能当作真实等待周期。
# 初学者先按 mode 比较 min_distance/median_distance/max_distance，再回到汇编追踪 LDS.128 结果首次参与 FFMA 的位置。
# 中文总索引：docs/zh/23-代码与脚本使用索引.md；脚本命令默认在仓库根目录执行。

"""统计每条 LDS.128 到其结果首次被 FFMA 使用之间的静态指令间隔。

这是 SASS 顺序检查，不是以周期为单位的延迟测量。LDS.128 写入四个连续寄存器；
对每次读取，脚本寻找后面第一条把其中任意寄存器作为源操作数的 FFMA。"""

import csv
import re
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODES = {
    "k10": (ROOT / "build-prefetch-v1/sgemm", "_Z15sgemmWarptilingI"),
    "v1": (ROOT / "build-prefetch-v1/sgemm", "_Z31sgemmWarptilingFragmentPrefetchI"),
    "v2": (ROOT / "build-prefetch-v2/sgemm", "_Z31sgemmWarptilingFragmentPrefetchI"),
    "v3": (ROOT / "build-prefetch-v3/sgemm", "_Z31sgemmWarptilingFragmentPrefetchI"),
}
INST = re.compile(r"/\*([0-9a-f]+)\*/\s+(?:@[!P0-9]+\s+)?([A-Z0-9_.]+)\s+(.*?)\s*;")
REG = re.compile(r"\bR(\d+)\b")

# 提取指定 kernel 的汇编文本；同一 sgemm 二进制还含其他核，必须按 marker 区分。
def instructions(binary: Path, marker: str):
    text = subprocess.run(["cuobjdump", "--dump-sass", str(binary)], check=True,
                          text=True, capture_output=True).stdout
    on = False
    result = []
    for line in text.splitlines():
        if "Function : " in line:
            if on:
                break
            on = marker in line
            continue
        if on and (match := INST.search(line)):
            result.append((int(match.group(1), 16), match.group(2), match.group(3)))
    return result

# LDS.128 写四个连续寄存器，再找其后 FFMA 第一次读它的位置；简单近似没有完整控制流证明。
def distances(insts):
    found = []
    for index, (_, opcode, operands) in enumerate(insts):
        if opcode != "LDS.128":
            continue
        destination = int(REG.search(operands.split(",", 1)[0]).group(1))
        written = set(range(destination, destination + 4))
        for use_index in range(index + 1, len(insts)):
            _, use_opcode, use_operands = insts[use_index]
            if use_opcode != "FFMA":
                continue
            parts = use_operands.split(",")
            sources = {int(value) for part in parts[1:] for value in REG.findall(part)}
            if written & sources:
                found.append(use_index - index - 1)
                break
    return found

writer = csv.writer(sys.stdout, lineterminator="\n")
writer.writerow(["mode", "lds128", "consumed_by_ffma", "min_distance", "median_distance",
                 "mean_distance", "max_distance", "distance_le_8", "distance_ge_32"])
for mode, (binary, marker) in MODES.items():
    insts = instructions(binary, marker)
    values = distances(insts)
    lds_count = sum(opcode == "LDS.128" for _, opcode, _ in insts)
    writer.writerow([mode, lds_count, len(values), min(values), f"{statistics.median(values):.1f}",
                     f"{statistics.mean(values):.2f}", max(values),
                     sum(value <= 8 for value in values), sum(value >= 32 for value in values)])
