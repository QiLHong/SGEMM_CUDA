#!/usr/bin/env python3
"""Measure static instruction distance from each LDS.128 to its first FFMA use.

This is a SASS ordering audit, not a latency-in-cycles measurement.  LDS.128 writes
four consecutive registers.  For each load, the script finds the first later FFMA
whose source operands reference any of those registers.
"""

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
