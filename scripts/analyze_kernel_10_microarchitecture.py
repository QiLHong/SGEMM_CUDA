#!/usr/bin/env python3
# 【文件导读】把 K10 汇编变成微内核指标
# 给初学者：这类脚本像读图纸，不负责测 GPU 跑多快；先准备它要求的二进制或报告，再读输出表。
# 运行：python3 scripts/analyze_kernel_10_microarchitecture.py --binary <sgemm二进制>。
# 需要 cuobjdump/nvdisasm；默认二进制路径为历史 build-prefetch-v1/sgemm。
# 终端输出 CSV：主循环 FFMA 链、累加器复用距离、LDS 首次使用距离及 nvdisasm 寄存器信息。
# 这里既有源码推导常量，也有汇编统计；静态距离不能直接换算延迟，更不能单凭它断言 scoreboard 瓶颈。
# 初学者先对照 metric/value/unit 三列，区分指令间隔与寄存器数量，再追查指标的推导或汇编统计代码。
# 中文总索引：docs/zh/23-代码与脚本使用索引.md；脚本命令默认在仓库根目录执行。

"""静态分析 Kernel 10 的 SM86 SASS 汇编和寄存器存活情况。

累加器与 LDS 的距离统计的是中间隔了多少条静态 SASS 指令，而不是延迟周期。
要判断某个静态依赖是否成为运行时瓶颈，还需要结合 Nsight Compute 的动态计数器。"""

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


# 调用外部工具并收集文本；返回码非零表示工具失败，不应继续生成看似完整的统计。
def command(*args, cwd=None):
    return subprocess.run(args, cwd=cwd, check=True, text=True,
                          capture_output=True).stdout


# 从包含多个函数的 cuobjdump 输出中挑出 K10；不能把整份二进制所有函数都计入一个核。
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


# 去掉地址/编码展示，只留下便于分析的指令记录；格式假设依赖工具版本。
def parse_instructions(lines):
    result = []
    for line in lines:
        match = INST.search(line)
        if match:
            result.append({"address": int(match.group(1), 16),
                           "opcode": match.group(2),
                           "operands": match.group(3)})
    return result


# 从 R123 这样的物理寄存器名称取编号；编号不是变量名，也不直接等于存活寄存器总数。
def register_number(text):
    match = REG.search(text)
    return int(match.group(1)) if match else None


# 沿 FFMA 的源/目的寄存器追踪同一个部分和，不能只盯住固定寄存器号，因为编译器可能重命名。
def accumulator_chains(instructions):
    """即使 ptxas 重命名了通用寄存器（GPR），也沿值的流向跟踪 FFMA 累加器。

    两条 BAR 指令界定了完全展开的 BK=16 计算主体。
    每条 FFMA 的第四个操作数提供旧累加值，目标寄存器保存新累加值，
    因此要沿“源 GPR → 目标 GPR”追踪同一条累加链，而不能只看固定寄存器名。"""
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


# 估算 shared load 到第一次后续 FFMA 使用之间隔了多少条指令；这是静态文本距离，不是周期。
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


# 提取 cubin 并调用 nvdisasm 的存活分析；临时文件用临时目录隔离，不改原二进制。
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


# 把循环结构、依赖间隔和寄存器指标装进报告；部分源码规模常量只适用于当前 K10。
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


# 命令行只选择二进制，输出 metric/value/unit 表到终端；重定向保存时要另选新文件名。
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=Path("build-prefetch-v1/sgemm"))
    args = parser.parse_args()
    writer = csv.writer(sys.stdout, lineterminator="\n")
    writer.writerow(["metric", "value", "unit"])
    writer.writerows(summarize(args.binary.resolve()))


if __name__ == "__main__":
    main()
