#!/usr/bin/env python3
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
}


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
