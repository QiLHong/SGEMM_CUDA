#!/usr/bin/env python3
"""Count static SASS opcodes exported from K10/cuBLAS NCU reports.

The cuBLAS kernel contains multiple predicated/branched paths.  These whole-function
counts prove which instruction families exist, but are not dynamic execution counts.
"""

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
