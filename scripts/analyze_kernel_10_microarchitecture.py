#!/usr/bin/env python3
"""Static SM86 SASS and register-liveness audit for Kernel 10.

The accumulator and LDS distances are counts of intervening static SASS
instructions, not latency cycles.  Nsight Compute counters are required to decide
whether a static dependency is a runtime bottleneck.
"""

import argparse
import csv
import re
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

KERNEL_MARKER = "_Z15sgemmWarptilingI"
INST = re.compile(r"/\*([0-9a-f]+)\*/\s+(?:@[!P0-9]+\s+)?([A-Z0-9_.]+)\s+(.*?)\s*;")
REG = re.compile(r"\bR(\d+)\b")
LIVE = re.compile(r"// \|\s*(\d+)\s+\|")


def command(*args, cwd=None):
    return subprocess.run(args, cwd=cwd, check=True, text=True,
                          capture_output=True).stdout


def extract_function(text):
    on = False
    lines = []
    for line in text.splitlines():
        if "Function : " in line:
            if on:
                break
            on = KERNEL_MARKER in line
            continue
        if on:
            lines.append(line)
    if not lines:
        raise RuntimeError("Kernel 10 SASS was not found")
    return lines


def parse_instructions(lines):
    result = []
    for line in lines:
        match = INST.search(line)
        if match:
            result.append({"address": int(match.group(1), 16),
                           "opcode": match.group(2),
                           "operands": match.group(3)})
    return result


def register_number(text):
    match = REG.search(text)
    return int(match.group(1)) if match else None


def accumulator_chains(instructions):
    """Follow FFMA accumulator values even when ptxas renames the GPR.

    The two BAR instructions delimit the fully unrolled BK=16 compute body.
    Each FFMA's fourth operand is the old accumulator value and its destination
    is the new value.  Ownership therefore follows source GPR -> destination GPR.
    """
    barriers = [index for index, instruction in enumerate(instructions)
                if instruction["opcode"].startswith("BAR.")]
    if len(barriers) != 2:
        raise RuntimeError(f"expected two block barriers, found {len(barriers)}")
    ffmas = [(index, instruction) for index, instruction in enumerate(instructions)
             if barriers[0] < index < barriers[1] and instruction["opcode"] == "FFMA"]
    ownership = {}
    chains = []
    physical_renames = 0
    for ordinal, (instruction_index, instruction) in enumerate(ffmas):
        parts = instruction["operands"].split(",")
        destination = register_number(parts[0])
        source = register_number(parts[3])
        chain = ownership.get(source)
        if chain is None:
            chain = len(chains)
            chains.append([])
        old_destination_owner = ownership.get(destination)
        if old_destination_owner is not None and old_destination_owner != chain:
            del ownership[destination]
        if destination != source:
            physical_renames += 1
            ownership.pop(source, None)
        ownership[destination] = chain
        chains[chain].append((instruction_index, ordinal))
    return ffmas, chains, physical_renames


def lds_to_ffma_distances(instructions):
    distances = []
    for index, instruction in enumerate(instructions):
        if instruction["opcode"] != "LDS.128":
            continue
        destination = register_number(instruction["operands"].split(",", 1)[0])
        written = set(range(destination, destination + 4))
        for use_index in range(index + 1, len(instructions)):
            use = instructions[use_index]
            if use["opcode"] != "FFMA":
                continue
            parts = use["operands"].split(",")
            sources = {int(value) for part in parts[1:] for value in REG.findall(part)}
            if written & sources:
                distances.append(use_index - index - 1)
                break
    return distances


def nvdisasm_liveness(binary):
    with tempfile.TemporaryDirectory(prefix="k10-nvdisasm-") as directory:
        listing = command("cuobjdump", "-lelf", str(binary))
        names = re.findall(r"ELF file\s+\d+:\s+(\S+)", listing)
        command("cuobjdump", "-xelf", "all", str(binary), cwd=directory)
        best = None
        for name in names:
            path = Path(directory) / name
            text = command("nvdisasm", "-c", "-plr", "-lrm", "count", str(path))
            on = False
            values = []
            max_line = ""
            maximum = -1
            allocated = None
            for line in text.splitlines():
                if line.startswith("//--------------------- .text."):
                    if on:
                        break
                    on = KERNEL_MARKER in line
                    continue
                if not on:
                    continue
                if "SHI_REGISTERS=" in line:
                    allocated = int(re.search(r"SHI_REGISTERS=(\d+)", line).group(1))
                match = LIVE.search(line)
                if match:
                    value = int(match.group(1))
                    values.append(value)
                    if value > maximum:
                        maximum = value
                        max_line = line.strip()
            if values and (best is None or maximum > best["max_live_gpr"]):
                address = re.search(r"/\*([0-9a-f]+)\*/", max_line)
                best = {"max_live_gpr": maximum,
                        "median_live_gpr": statistics.median(values),
                        "max_live_address": address.group(1) if address else "unknown",
                        "max_live_line": max_line,
                        "cubin": name,
                        "allocated_gpr": allocated}
        if best is None:
            raise RuntimeError("nvdisasm did not find Kernel 10")
        return best


def summarize(binary):
    sass = command("cuobjdump", "--dump-sass", str(binary))
    function_lines = extract_function(sass)
    instructions = parse_instructions(function_lines)
    main_ffmas, chains, physical_renames = accumulator_chains(instructions)
    instruction_gaps = []
    ffma_gaps = []
    for chain in chains:
        instruction_gaps.extend(current[0] - previous[0] - 1
                                for previous, current in zip(chain, chain[1:]))
        ffma_gaps.extend(current[1] - previous[1] - 1
                        for previous, current in zip(chain, chain[1:]))
    lds_distances = lds_to_ffma_distances(instructions)
    counts = {}
    for instruction in instructions:
        base = instruction["opcode"].split(".", 1)[0]
        counts[base] = counts.get(base, 0) + 1
    liveness = nvdisasm_liveness(binary)
    updates = [len(value) for value in chains]
    if len(main_ffmas) != 2048 or len(chains) != 128 or set(updates) != {16}:
        raise RuntimeError(
            f"unexpected K10 accumulator graph: ffma={len(main_ffmas)}, "
            f"chains={len(chains)}, updates={sorted(set(updates))}"
        )
    static_instruction_lines = sum(
        bool(re.search(r"/\*[0-9a-f]+\*/", line)) for line in function_lines
    )
    metrics = [
        ("static_instructions", static_instruction_lines, "instruction"),
        ("static_ffma", counts.get("FFMA", 0), "instruction"),
        ("mainloop_accumulating_ffma", len(main_ffmas), "instruction"),
        ("logical_accumulator_chains", len(chains), "chain"),
        ("physical_accumulator_gpr_renames", physical_renames, "FFMA"),
        ("updates_per_accumulator_per_bk_tile", min(updates), "FFMA"),
        ("runtime_dependency_chain_k4096", 4096, "FFMA/accumulator"),
        ("accumulator_reuse_min_static_distance", min(instruction_gaps), "instruction"),
        ("accumulator_reuse_median_static_distance", statistics.median(instruction_gaps), "instruction"),
        ("accumulator_reuse_mean_static_distance", f"{statistics.mean(instruction_gaps):.2f}", "instruction"),
        ("accumulator_reuse_max_static_distance", max(instruction_gaps), "instruction"),
        ("independent_ffma_between_same_acc_min", min(ffma_gaps), "FFMA"),
        ("independent_ffma_between_same_acc_median", statistics.median(ffma_gaps), "FFMA"),
        ("independent_ffma_between_same_acc_mean", f"{statistics.mean(ffma_gaps):.2f}", "FFMA"),
        ("independent_ffma_between_same_acc_max", max(ffma_gaps), "FFMA"),
        ("lds128", sum(i["opcode"] == "LDS.128" for i in instructions), "instruction"),
        ("lds_to_ffma_min_static_distance", min(lds_distances), "instruction"),
        ("lds_to_ffma_median_static_distance", statistics.median(lds_distances), "instruction"),
        ("lds_to_ffma_mean_static_distance", f"{statistics.mean(lds_distances):.2f}", "instruction"),
        ("lds_to_ffma_max_static_distance", max(lds_distances), "instruction"),
        ("imad", counts.get("IMAD", 0), "instruction"),
        ("iadd3", counts.get("IADD3", 0), "instruction"),
        ("lea", counts.get("LEA", 0), "instruction"),
        ("shf", counts.get("SHF", 0), "instruction"),
        ("shl", counts.get("SHL", 0), "instruction"),
        ("mov", counts.get("MOV", 0), "instruction"),
        ("allocated_gpr_per_thread", liveness["allocated_gpr"], "register"),
        ("nvdisasm_peak_live_gpr", liveness["max_live_gpr"], "register"),
        ("nvdisasm_peak_live_address", liveness["max_live_address"], "SASS address"),
        ("nvdisasm_median_live_gpr", liveness["median_live_gpr"], "register"),
        ("source_accumulator_values", 128, "float"),
        ("source_a_fragment_values", 8, "float"),
        ("source_b_fragment_values", 16, "float"),
        ("source_accumulator_plus_fragments", 152, "float"),
        ("allocated_gpr_above_source_floor", 32, "register"),
    ]
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=Path("build-prefetch-v1/sgemm"))
    args = parser.parse_args()
    writer = csv.writer(sys.stdout, lineterminator="\n")
    writer.writerow(["metric", "value", "unit"])
    writer.writerows(summarize(args.binary.resolve()))


if __name__ == "__main__":
    main()
