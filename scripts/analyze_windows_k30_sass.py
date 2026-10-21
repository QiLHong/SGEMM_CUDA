#!/usr/bin/env python3
"""Read cuobjdump SASS without executing a GPU kernel.

Adapted from this repository's scripts/analyze_k30_schedule.py. The original
author/source attribution remains in the repository; this parser adds explicit
loop discovery, diagnostics and immutable Windows experiment output.

Example:
  python scripts/analyze_windows_k30_sass.py sass.txt --output new-static-dir \
      --binary build-windows-k30-sync-sm89/windows_benchmark.exe

Instruction positions/counts are static, not execution counts, cycles, measured
latencies, or proof that a load overlaps arithmetic. cuBLAS device code is in its
library and is not inferred from the executable's SASS.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path
import re


INSTRUCTION = re.compile(r"^\s*/\*([0-9a-f]+)\*/\s+(.*?)\s*;", re.MULTILINE | re.IGNORECASE)
OPCODE = re.compile(r"^(?:@(!?[A-Z]+[0-9]*)\s+)?([A-Z][A-Z0-9_.]*)\b")
COMMON_OPS = (
    "FFMA", "FADD", "FMUL", "LDG", "LDS", "STS", "STG", "LDL", "STL",
    "LDGSTS", "BAR", "DEPBAR", "WARPSYNC", "BRA", "IMAD", "IADD3", "MOV",
    "HMMA", "MMA",
)


def classify(name: str) -> str:
    # These suffixes encode template parameters in nvcc's mangled symbol.
    if "sgemmK30" in name:
        stage = re.search(r"ELi([124])EEv", name)
        if stage and stage[1] in ("1", "2"):
            return f"k30_v{stage[1]}"
        # STAGES=4 is not a variant number: historical V3/V4 also differ by Shape.
        return f"k30_stages{stage[1]}" if stage else "k30_unrecognized"
    if "sgemmWarptiling" in name:
        padding = re.search(r"ELi([0-9]+)EEv", name)
        if padding:
            return "k10" if padding[1] == "4" else f"k10_pad{padding[1]}"
        return "k10_unrecognized"
    return "other"


def instructions(block: str) -> list[dict]:
    result = []
    for pc, text in INSTRUCTION.findall(block):
        match = OPCODE.match(text)
        if not match:
            raise ValueError(f"Unrecognized instruction at {pc}: {text}")
        predicate, full_opcode = match.groups()
        result.append({
            "pc": int(pc, 16), "instruction": text.strip(),
            "predicate": predicate or "", "opcode": full_opcode.split(".")[0],
            "full_opcode": full_opcode, "statically_pred_off": predicate == "!PT",
        })
    return result


def function_blocks(sass: str) -> list[dict]:
    starts = list(re.finditer(r"Function\s*:\s*([^\r\n]+)", sass))
    result = []
    for index, start in enumerate(starts):
        end = starts[index + 1].start() if index + 1 < len(starts) else len(sass)
        name = start[1].strip()
        arch = re.findall(r"\barch\s*=\s*(sm_[0-9a-z]+)", sass[:start.start()])
        body = instructions(sass[start.end():end])
        if not body:
            raise ValueError(f"No SASS instructions found for {name}")
        result.append({
            "function_id": index, "mode": classify(name), "arch": arch[-1] if arch else "unknown",
            "function": name, "instructions": body,
        })
    if not result:
        raise ValueError("No 'Function :' blocks found; supply cuobjdump --dump-sass output.")
    return result


def ffma(instruction: dict) -> bool:
    return instruction["opcode"] == "FFMA" and not instruction["statically_pred_off"]


def loops_for(function: dict) -> list[dict]:
    """Find backward BRA intervals, exposing every candidate before selection."""
    ins = function["instructions"]
    pc_set = {i["pc"] for i in ins}
    candidates = []
    for branch in ins:
        if branch["opcode"] != "BRA" or branch["statically_pred_off"]:
            continue
        target = re.search(r"\b(0x[0-9a-f]+)\b", branch["instruction"], re.IGNORECASE)
        if not target:
            continue
        begin = int(target[1], 16)
        if begin >= branch["pc"] or begin not in pc_set:
            continue
        body = [i for i in ins if begin <= i["pc"] <= branch["pc"]]
        candidates.append({
            "start": begin, "end": branch["pc"], "body": body,
            "static_ffma": sum(ffma(i) for i in body),
        })
    return candidates


def analyze(sass: str) -> dict:
    functions = function_blocks(sass)
    function_rows, opcode_rows, candidate_rows, order_rows, summary_rows = [], [], [], [], []
    warnings = []
    for function in functions:
        identity = {k: function[k] for k in ("function_id", "mode", "arch", "function")}
        ins = function["instructions"]
        counts = collections.Counter(i["opcode"] for i in ins)
        pred_off = collections.Counter(i["opcode"] for i in ins if i["statically_pred_off"])
        function_rows.append({
            **identity, "static_instructions": len(ins),
            "statically_pred_off_instructions": sum(pred_off.values()),
            **{op: counts[op] for op in COMMON_OPS},
        })
        for opcode in sorted(counts):
            opcode_rows.append({
                **identity, "opcode": opcode, "static_count": counts[opcode],
                "statically_pred_off_count": pred_off[opcode],
            })
        if function["mode"] == "other":
            continue
        candidates = loops_for(function)
        arithmetic = [c for c in candidates if c["static_ffma"] > 0]
        selected = None
        selection = "no_backward_BRA_loop_with_FFMA"
        if arithmetic:
            best_ffma = max(c["static_ffma"] for c in arithmetic)
            largest = [c for c in arithmetic if c["static_ffma"] == best_ffma]
            selected = min(largest, key=lambda c: (len(c["body"]), c["start"], c["end"]))
            selection = "unique_maximum_static_FFMA" if len(largest) == 1 else "ambiguous_maximum_FFMA_shortest_interval_selected"
            if len(largest) > 1:
                warnings.append(f"{function['mode']}: multiple maximal-FFMA loops; inspect loop-candidates.csv.")
        else:
            warnings.append(f"{function['mode']}: no arithmetic loop recognized; manual SASS review required.")
        for number, candidate in enumerate(candidates):
            candidate_rows.append({
                **identity, "candidate": number,
                "loop_start": hex(candidate["start"]), "loop_end": hex(candidate["end"]),
                "static_instructions": len(candidate["body"]), "static_ffma": candidate["static_ffma"],
                "selected": candidate is selected, "selection_rule": selection,
            })
        if selected is None:
            continue
        body = selected["body"]
        total_ffma = selected["static_ffma"]
        before = 0
        memory_rows = []
        for position, instruction in enumerate(body):
            after = total_ffma - before - int(ffma(instruction))
            # 保留全部主循环指令，便于直接查看 LDG、LDS、FFMA、STS 和 BAR 的完整顺序。
            order = {
                **identity, "loop_start": hex(selected["start"]), "loop_end": hex(selected["end"]),
                "instruction_index": position, "pc": hex(instruction["pc"]),
                "opcode": instruction["opcode"], "predicate": instruction["predicate"],
                "statically_pred_off": instruction["statically_pred_off"],
                "ffma_before": before, "ffma_after": after,
                "instruction": instruction["instruction"],
            }
            order_rows.append(order)
            if instruction["opcode"] == "LDG" and not instruction["statically_pred_off"]:
                memory_rows.append(order)
            before += int(ffma(instruction))
        loop_counts = collections.Counter(i["opcode"] for i in body if not i["statically_pred_off"])
        summary_rows.append({
            **identity, "selection_rule": selection,
            "loop_start": hex(selected["start"]), "loop_end": hex(selected["end"]),
            "static_instructions": len(body), "static_ffma": total_ffma,
            "static_ldg": loop_counts["LDG"], "static_lds": loop_counts["LDS"],
            "static_sts": loop_counts["STS"], "static_bar": loop_counts["BAR"],
            "static_depbar": loop_counts["DEPBAR"], "static_ldl": loop_counts["LDL"],
            "static_stl": loop_counts["STL"], "static_ldgsts": loop_counts["LDGSTS"],
            "ldg_first_instruction_index": memory_rows[0]["instruction_index"] if memory_rows else "",
            "ldg_last_instruction_index": memory_rows[-1]["instruction_index"] if memory_rows else "",
            "ldg_min_ffma_after": min(r["ffma_after"] for r in memory_rows) if memory_rows else "",
            "ldg_max_ffma_after": max(r["ffma_after"] for r in memory_rows) if memory_rows else "",
            "bar_instruction_indices": ";".join(str(p) for p, i in enumerate(body) if i["opcode"] == "BAR"),
            "bar_ffma_before": ";".join(str(r["ffma_before"]) for r in order_rows
                                      if r["function_id"] == function["function_id"] and r["opcode"] == "BAR"),
        })
    return {
        "functions": function_rows, "opcodes": opcode_rows, "candidates": candidate_rows,
        "order": order_rows, "summary": summary_rows, "warnings": warnings,
    }


def write_csv(path: Path, rows: list[dict], empty_header: tuple[str, ...]) -> None:
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else empty_header)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sass", type=Path, help="Saved cuobjdump --dump-sass text; does not execute GPU code")
    parser.add_argument("--output", required=True, type=Path, help="New directory; existing output is refused")
    parser.add_argument("--binary", type=Path, help="Optional exact executable, recorded by SHA256")
    parser.add_argument("--require-modes", nargs="+", default=["k30_v1", "k30_v2"],
                        help="Required recognized symbols (default: k30_v1 k30_v2)")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    raw = args.sass.read_bytes()
    result = analyze(raw.decode("utf-8-sig"))
    present = {row["mode"] for row in result["functions"]}
    missing = set(args.require_modes) - present
    if missing:
        raise ValueError(f"Required symbols missing: {sorted(missing)}; present={sorted(present)}")
    metadata = {
        "source": str(args.sass.resolve()), "sass_sha256": hashlib.sha256(raw).hexdigest(),
        "binary": str(args.binary.resolve()) if args.binary else None,
        "binary_sha256": hashlib.sha256(args.binary.read_bytes()).hexdigest() if args.binary else None,
        "parser_source": "Adapted from scripts/analyze_k30_schedule.py in the same repository",
        "selection": "Backward BRA intervals; choose maximum static FFMA count, then shortest interval; flag ties.",
        "counting": "Function counts include all parsed static instructions. Loop summary excludes @!PT; other predicates are not evaluated.",
        "limitations": [
            "Static instruction order/counts are not dynamic execution counts or cycles.",
            "A load appearing before arithmetic does not prove overlap or hidden latency.",
            "A backward BRA interval is a structural heuristic, not a full control-flow graph.",
            "Executable SASS does not expose dynamically linked cuBLAS kernels.",
            "Only input evidence is analyzed; no historical result values are assumed.",
        ],
        "warnings": result["warnings"],
    }
    args.output.mkdir(parents=True, exist_ok=False)
    for filename, key in (
        ("functions.csv", "functions"), ("opcode-counts.csv", "opcodes"),
        ("loop-candidates.csv", "candidates"), ("mainloop-order.csv", "order"),
        ("mainloop-summary.csv", "summary"),
    ):
        write_csv(args.output / filename, result[key], ("mode", "function", "status"))
    with (args.output / "analysis.json").open("x", encoding="utf-8") as stream:
        json.dump(metadata, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    for row in result["summary"]:
        print(f"{row['mode']}: FFMA={row['static_ffma']} LDG={row['static_ldg']} "
              f"BAR={row['static_bar']} LDG FFMA-after="
              f"{row['ldg_min_ffma_after']}..{row['ldg_max_ffma_after']} "
              f"({row['selection_rule']}; static positions only)")
    for warning in result["warnings"]:
        print(f"WARNING: {warning}")


if __name__ == "__main__":
    main()
