#!/usr/bin/env python3
# 【文件导读】估算 K32 中数据寄存器的存活区间
# 给初学者：这类脚本像读图纸，不负责测 GPU 跑多快；先准备它要求的二进制或报告，再读输出表。
# 运行：python3 scripts/analyze_k32_payload_liveness.py <含sass_overlap/sass_single的目录> --output <全新目录>。
# 输入是 analyze_k32_binary.py 导出的汇编文本，不启动 GPU。
# 输出 payload-liveness.csv 与 high-register-values.csv；核对 2048 FFMA、96 LDS.128 的主体结构后统计。
# lifetime 指一个值从产生到最后使用这段区间；这里只跟踪部分数据值，不包括全部地址/控制寄存器，不能冒充总资源需求。
# 初学者先比较 payload_live_peak/fragment_live_peak，再用逐寄存器表中的 defining_pc/last_use_pc 追踪值的存活区间。
# 中文总索引：docs/zh/23-代码与脚本使用索引.md；脚本命令默认在仓库根目录执行。

"""跟踪 K32 直线代码段中的 LDS/FFMA 数据值，不统计整个内核的寄存器分配。

输入为 analyze_k32_binary.py 保存的 cuobjdump 文本，不需要 GPU。
预期结构为 128 个累加器各更新 16 次；结构改变时立即报错，避免输出误导性统计。
这里的距离是指令索引之差，不是实际测得的时钟周期。"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
from pathlib import Path
import re
import statistics


# 跟踪数据值从写入到最后一次使用，估计同时活着的 payload 数；先检查汇编符合识别假设。
# 这里没有完整重建地址/谓词/控制寄存器，不等价于 ptxas 的总寄存器分配。
def analyze(path, mode):
    raw = path.read_text()
    instructions = [(int(pc, 16), text.strip()) for pc, text in
                    re.findall(r"/\*([0-9a-f]+)\*/\s+(.*?)\s*;", raw)]
    start = next(pc for pc, text in instructions if text.startswith("LDS."))
    end = next(pc for pc, text in instructions if pc > start and "BRA.CONV" in text)
    body = [(pc, text) for pc, text in instructions if start <= pc < end]
    assert Counter(text.split()[0] for _, text in body) == {"FFMA": 2048, "LDS.128": 96}
    registers, values, uses, outputs, chains = {}, [], defaultdict(list), {}, defaultdict(list)
    rename_count, depths = 0, Counter()

    def define(reg, kind, index, pc, origin=None, depth=None):
        value = len(values)
        values.append((reg, kind, index, pc, origin, depth))
        registers[reg] = value
        return value

    for index, (pc, text) in enumerate(body):
        regs = [int(x) for x in re.findall(r"\bR(\d+)", text)]
        if text.startswith("LDS.128"):
            for reg in range(regs[0], regs[0] + 4):
                define(reg, "fragment", index, pc)
        else:
            dest, a, b, acc = regs
            if acc not in registers:
                define(acc, "accumulator", -1, 0, acc, 0)
            assert values[registers[a]][1] == values[registers[b]][1] == "fragment"
            old = values[registers[acc]]
            assert old[1] == "accumulator"
            origin, depth = old[4:]
            for reg in (a, b, acc):
                uses[registers[reg]].append(index)
            outputs[origin] = define(dest, "accumulator", index, pc, origin, depth + 1)
            chains[origin].append(index)
            depths[depth] += 1
            rename_count += dest != acc
    assert len(outputs) == 128 and depths == {k: 128 for k in range(16)}
    for value in outputs.values():
        assert values[value][5] == 16
        uses[value].append(len(body))

    # 在指令边界处，最后一次读取刚结束的值不再计入存活集合。
    # 累加器在入口就视为存活，128 个最终累加值也都必须保留到出口。
    # 这里不统计地址/控制寄存器，也不统计因未完成指令而仍需保留的操作数。
    changes, fragments, loads, high = Counter(), Counter(), defaultdict(list), []
    for value, (reg, kind, begin, pc, origin, depth) in enumerate(values):
        assert uses[value], (mode, reg, hex(pc))
        finish = max(uses[value])
        changes[begin] += 1
        changes[finish] -= 1
        if kind == "fragment":
            fragments[begin] += 1
            fragments[finish] -= 1
            loads[pc].append(min(uses[value]) - begin)
        if reg >= 167:
            high.append(dict(mode=mode,register=f"R{reg}",kind=kind,
                             defining_pc=hex(pc),updates_completed=depth,
                             last_use_pc=hex(body[finish][0]) if finish < len(body) else "live-out",
                             instruction_span=finish-begin))
    alive = fragment_alive = peak = fragment_peak = 0
    for index in range(-1, len(body) + 1):
        alive += changes[index]
        fragment_alive += fragments[index]
        peak = max(peak, alive)
        fragment_peak = max(fragment_peak, fragment_alive)
    distances = [min(x) for x in loads.values()]
    reuse = [b-a for chain in chains.values() for a,b in zip(chain,chain[1:])]
    summary = dict(mode=mode,input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                   mainloop_ffma=2048,mainloop_lds128=96,accumulators=128,
                   payload_live_peak=peak,fragment_live_peak=fragment_peak,
                   ffma_non_inplace=rename_count,
                   lds_first_use_min=min(distances),lds_first_use_median=statistics.median(distances),
                   lds_first_use_mean=statistics.mean(distances),lds_first_use_max=max(distances),
                   accumulator_reuse_min=min(reuse),accumulator_reuse_median=statistics.median(reuse),
                   accumulator_reuse_mean=statistics.mean(reuse),accumulator_reuse_max=max(reuse))
    return summary, high


# 保存派生表时保留既有结果约束；CSV 是汇编分析证据，不是新的硬件测量。
def save(path, rows):
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    content = stream.getvalue()
    if path.exists():
        assert path.read_text() == content, f"Refusing to overwrite {path}"
    else:
        with path.open("x") as output:
            output.write(content)


# 读取 overlap/single 的 SASS，生成对比表与输入哈希；哈希用来确认分析的是哪份汇编。
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summaries, high = [], []
    for mode in ("overlap", "single"):
        summary, rows = analyze(args.directory / f"sass_{mode}.txt", mode)
        summaries.append(summary)
        high.extend(rows)
    args.output.mkdir(parents=True, exist_ok=True)
    save(args.output / "payload-liveness.csv", summaries)
    if high:
        save(args.output / "high-register-values.csv", high)
    for summary in summaries:
        print(summary)


if __name__ == "__main__":
    main()
