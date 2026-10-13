#!/usr/bin/env python3
"""Trace K32's straight-line LDS/FFMA payload, not whole-kernel allocation.

Input: cuobjdump text saved by analyze_k32_binary.py. No GPU is required.
Fails closed if the expected 128 accumulator x 16 updates pattern changes.
Distances are instruction-index differences, never measured clock cycles.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
from pathlib import Path
import re
import statistics


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

    # At an instruction boundary, values whose last read just completed die.
    # Accumulators are seeded live-in and all 128 final values are live-out.
    # Address/control registers and scoreboard-held operands are NOT counted.
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
