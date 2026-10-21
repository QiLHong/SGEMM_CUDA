#!/usr/bin/env python3
"""CPU-only SASS/PTX/NCU evidence analysis for the vitamin SIMT progression.

Example (run separately for A and B; output directories must be new):
  python scripts/analyze_vitamin_evidence.py --sass A.sass --ptx A.ptx \
      --label A --binary build-windows-vitamin-sm89-A/windows_vitamin_benchmark.exe \
      --output-dir new-analysis-A

Optional --profile accepts ncu-summary.csv from Profile-Vitamin.ps1. The SASS
reader and backward-branch discovery reuse analyze_windows_k30_sass.py (itself
adapted from analyze_k30_schedule.py); no original kernels or parsers are edited.
Counts and order are static evidence, not measured latency or proof of overlap.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re
import statistics

import analyze_windows_k30_sass as sass_reader


VITAMIN_MODES = (
    "sgemm_tiling", "sgemm_at_tiling", "sgemm_at_bcf_swizzling",
    "sgemm_at_bcf_swizzling_rw", "sgemm_at_bcf_swizzling_dbf_rw",
)
PROFILE_IDENTITIES = {"size", "mode", "profile_repeat", "kernel", "source_csv"}
PTX_INSTRUCTION = re.compile(r"^\s*(?:@!?[%\w]+\s+)?([a-z][a-z0-9_.]*)\s+[^;]*;")


def classify(name: str) -> str:
    for mode in sorted(VITAMIN_MODES, key=len, reverse=True):
        if mode + "_kernel" in name:
            return mode
    return sass_reader.classify(name)


def identity(function: dict, label: str) -> dict:
    return {"build": label, **{key: function[key] for key in
            ("function_id", "mode", "arch", "function")}}


def choose_loop(candidates: list[dict], arithmetic_key: str) -> tuple[dict | None, str]:
    arithmetic = [candidate for candidate in candidates if candidate[arithmetic_key] > 0]
    if not arithmetic:
        return None, "no_backward_arithmetic_branch"
    maximum = max(candidate[arithmetic_key] for candidate in arithmetic)
    best = [candidate for candidate in arithmetic if candidate[arithmetic_key] == maximum]
    selected = min(best, key=lambda candidate:
                   (len(candidate["body"]), candidate["start"], candidate["end"]))
    return selected, ("unique_maximum_static_fma" if len(best) == 1 else
                      "ambiguous_maximum_static_fma_shortest_interval_selected")


def analyze_sass(text: str, label: str) -> tuple[dict, list[str]]:
    tables = {key: [] for key in ("functions", "opcodes", "loops", "order", "summary")}
    warnings = []
    for function in sass_reader.function_blocks(text):
        function["mode"] = classify(function["function"])
        who = identity(function, label)
        instructions = function["instructions"]
        counts = collections.Counter(i["opcode"] for i in instructions)
        full_counts = collections.Counter(i["full_opcode"] for i in instructions)
        tables["functions"].append({
            **who, "static_instructions": len(instructions),
            **{op: counts[op] for op in sass_reader.COMMON_OPS},
            "tensor_instructions": sum(count for op, count in counts.items()
                                       if op in ("HMMA", "IMMA", "DMMA", "MMA", "HGMMA")),
            "ftz_instructions": sum(".FTZ" in i["full_opcode"] for i in instructions),
        })
        for op, count in sorted(full_counts.items()):
            tables["opcodes"].append({**who, "opcode": op, "static_count": count,
                "statically_pred_off_count": sum(i["full_opcode"] == op and
                i["statically_pred_off"] for i in instructions)})
        if function["mode"] == "other":
            continue
        candidates = sass_reader.loops_for(function)
        selected, rule = choose_loop(candidates, "static_ffma")
        if selected is None or rule.startswith("ambiguous"):
            warnings.append(f"SASS {function['mode']}: {rule}; inspect candidates manually.")
        for candidate in candidates:
            tables["loops"].append({
                **who, "start": hex(candidate["start"]), "end": hex(candidate["end"]),
                "static_instructions": len(candidate["body"]),
                "static_fma": candidate["static_ffma"], "selected": candidate is selected,
                "selection_rule": rule,
            })
        if selected is None:
            continue
        before = 0
        loads = []
        body_counts = collections.Counter()
        for position, instruction in enumerate(selected["body"]):
            live = not instruction["statically_pred_off"]
            body_counts[instruction["opcode"]] += int(live)
            current_fma = int(sass_reader.ffma(instruction))
            row = {
                **who, "loop_start": hex(selected["start"]), "loop_end": hex(selected["end"]),
                "position": position, "pc": hex(instruction["pc"]),
                "opcode": instruction["full_opcode"],
                "ffma_before": before, "ffma_after": selected["static_ffma"] - before - current_fma,
                "statically_pred_off": instruction["statically_pred_off"],
                "instruction": instruction["instruction"],
            }
            tables["order"].append(row)
            if live and instruction["opcode"] == "LDG":
                loads.append(row)
            before += current_fma
        tables["summary"].append({
            **who, "selection_rule": rule, "start": hex(selected["start"]),
            "end": hex(selected["end"]), "static_fma": selected["static_ffma"],
            **{op: body_counts[op] for op in sass_reader.COMMON_OPS},
            "ldg_min_ffma_after": min(row["ffma_after"] for row in loads) if loads else "",
            "ldg_max_ffma_after": max(row["ffma_after"] for row in loads) if loads else "",
        })
    return tables, warnings


def is_ptx_fma(instruction: dict) -> bool:
    return instruction["opcode"].startswith("fma.") and instruction["opcode"].endswith(".f32")


def analyze_ptx(text: str, label: str) -> tuple[dict, list[str]]:
    tables = {key: [] for key in ("functions", "opcodes", "loops", "order", "summary")}
    warnings = []
    starts = list(re.finditer(r"\.visible\s+\.entry\s+([^\s(]+)", text))
    if not starts:
        raise ValueError("No visible PTX entry found.")
    for number, start in enumerate(starts):
        end = starts[number + 1].start() if number + 1 < len(starts) else len(text)
        block = text[start.start():end]
        name = start[1]
        who = {"build": label, "function_id": number, "mode": classify(name), "function": name}
        lines = block.splitlines()
        labels = {}
        instructions = []
        for line_number, line in enumerate(lines, start=1):
            source = line.split("//", 1)[0].strip()
            if match := re.match(r"([\w$]+):", source):
                labels[match[1]] = line_number
            if match := PTX_INSTRUCTION.match(source):
                instructions.append({"line": line_number, "opcode": match[1], "instruction": source})
        counts = collections.Counter(i["opcode"] for i in instructions)
        tables["functions"].append({
            **who, "static_instructions": len(instructions),
            "fma_f32": sum(is_ptx_fma(i) for i in instructions),
            "ftz_instructions": sum(".ftz." in i["opcode"] for i in instructions),
            "cp_async_instructions": sum(i["opcode"].startswith("cp.async") for i in instructions),
            "tensor_instructions": sum(i["opcode"].startswith(("mma.", "wmma.", "wgmma."))
                                       for i in instructions),
        })
        for op, count in sorted(counts.items()):
            tables["opcodes"].append({**who, "opcode": op, "static_count": count})
        if who["mode"] == "other":
            continue
        candidates = []
        for instruction in instructions:
            branch = re.search(r"\bbra(?:\.uni)?\s+([\w$]+);", instruction["instruction"])
            if not branch or branch[1] not in labels:
                continue
            begin = labels[branch[1]]
            if begin >= instruction["line"]:
                continue
            body = [i for i in instructions if begin <= i["line"] <= instruction["line"]]
            candidates.append({"start": begin, "end": instruction["line"], "body": body,
                               "static_fma": sum(is_ptx_fma(i) for i in body)})
        selected, rule = choose_loop(candidates, "static_fma")
        if selected is None or rule.startswith("ambiguous"):
            warnings.append(f"PTX {who['mode']}: {rule}; inspect candidates manually.")
        for candidate in candidates:
            tables["loops"].append({**who, "start_entry_line": candidate["start"],
                "end_entry_line": candidate["end"], "static_instructions": len(candidate["body"]),
                "static_fma": candidate["static_fma"], "selected": candidate is selected,
                "selection_rule": rule})
        if selected is None:
            continue
        before = 0
        loads = []
        for position, instruction in enumerate(selected["body"]):
            current_fma = int(is_ptx_fma(instruction))
            row = {**who, "loop_start_entry_line": selected["start"],
                   "loop_end_entry_line": selected["end"], "position": position,
                   "entry_line": instruction["line"], "opcode": instruction["opcode"],
                   "fma_before": before, "fma_after": selected["static_fma"] - before - current_fma,
                   "instruction": instruction["instruction"]}
            tables["order"].append(row)
            if instruction["opcode"].startswith("ld.global"):
                loads.append(row)
            before += current_fma
        tables["summary"].append({
            **who, "selection_rule": rule, "start_entry_line": selected["start"],
            "end_entry_line": selected["end"], "static_fma": selected["static_fma"],
            "ld_global": len(loads),
            "bar_sync": sum(i["opcode"].startswith("bar.sync") for i in selected["body"]),
            "ldg_min_fma_after": min(row["fma_after"] for row in loads) if loads else "",
            "ldg_max_fma_after": max(row["fma_after"] for row in loads) if loads else "",
        })
    return tables, warnings


def profile_ranges(text: str, label: str) -> list[dict]:
    reader = csv.DictReader(io.StringIO(text))
    fields = reader.fieldnames or []
    if not PROFILE_IDENTITIES <= set(fields):
        raise ValueError("Profile input must be ncu-summary.csv from Profile-Vitamin.ps1.")
    metric_names = [field for field in fields if field not in PROFILE_IDENTITIES]
    groups = collections.defaultdict(list)
    for row in reader:
        if int(row["size"]) <= 0 or int(row["profile_repeat"]) <= 0:
            raise ValueError("Profile size and repetition must be positive.")
        groups[(int(row["size"]), row["mode"])].append(row)
    if not groups or not metric_names:
        raise ValueError("No profile groups or metrics found.")
    output = []
    for (size, mode), rows in sorted(groups.items()):
        if len({row["profile_repeat"] for row in rows}) != len(rows):
            raise ValueError(f"Duplicate profile repetitions: {size}, {mode}")
        for metric in metric_names:
            values = [float(row[metric]) for row in rows]
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f"Nonfinite profile metric: {size}, {mode}, {metric}")
            output.append({"build": label, "size": size, "mode": mode, "metric": metric,
                           "samples": len(values), "minimum": min(values),
                           "median": statistics.median(values), "maximum": max(values)})
    return output


def save_csv(path: Path, rows: list[dict]) -> None:
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ["status"])
        writer.writeheader()
        writer.writerows(rows)


def compare_sass(first: Path, second: Path) -> dict:
    """Compare full parsed functions, not only the selected arithmetic loops."""
    def read(path):
        return {(f["arch"], f["function"]): f for f in
                sass_reader.function_blocks(path.read_text(encoding="utf-8-sig"))}

    def normalize(instruction):
        opcode = instruction["full_opcode"]
        return instruction["instruction"].replace(opcode, opcode.replace(".FTZ", ""), 1)

    left, right = read(first), read(second)
    rows = []
    for key in sorted(left.keys() | right.keys()):
        a, b = left.get(key), right.get(key)
        if a is None or b is None:
            rows.append({"arch": key[0], "function": key[1], "mode": classify(key[1]),
                         "present_first": a is not None, "present_second": b is not None,
                         "normalized_equal": False})
            continue
        ai, bi = a["instructions"], b["instructions"]
        raw_diffs, normalized_diffs = [], []
        for position in range(max(len(ai), len(bi))):
            aa = ai[position] if position < len(ai) else None
            bb = bi[position] if position < len(bi) else None
            ar = aa["instruction"] if aa else None
            br = bb["instruction"] if bb else None
            if ar != br:
                raw_diffs.append({"position": position, "first": ar, "second": br})
            if (normalize(aa) if aa else None) != (normalize(bb) if bb else None):
                normalized_diffs.append({"position": position, "first": ar, "second": br})
        rows.append({
            "arch": key[0], "function": key[1], "mode": classify(key[1]),
            "present_first": True, "present_second": True,
            "first_static_instructions": len(ai), "second_static_instructions": len(bi),
            "raw_different_positions": len(raw_diffs),
            "normalized_different_positions": len(normalized_diffs),
            "raw_equal": not raw_diffs, "normalized_equal": not normalized_diffs,
            "raw_first_8_differences": raw_diffs[:8],
            "normalized_first_8_differences": normalized_diffs[:8],
        })
    return {
        "first": {"path": str(first.resolve()), "sha256": hashlib.sha256(first.read_bytes()).hexdigest()},
        "second": {"path": str(second.resolve()), "sha256": hashlib.sha256(second.read_bytes()).hexdigest()},
        "scope": "All parsed SASS instructions in every matched architecture/function; not only main loops.",
        "normalization": "Remove .FTZ from the opcode token only; retain instruction order, predicates and operands.",
        "excluded": "cuobjdump headers, source-line comments, instruction PC labels and binary encodings.",
        "meaning": "Textual normalized equality does not make FTZ arithmetic semantically identical.",
        "all_functions_normalized_equal": all(row["normalized_equal"] for row in rows),
        "functions": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sass", type=Path, required=True)
    parser.add_argument("--ptx", type=Path, required=True)
    parser.add_argument("--label", required=True, help="Build condition label, e.g. A or B")
    parser.add_argument("--binary", type=Path, help="Exact executable for SHA256 provenance")
    parser.add_argument("--profile", type=Path, help="Optional existing NCU summary CSV")
    parser.add_argument("--compare-sass", type=Path, help="Optional other build's SASS; compare full functions ignoring only .FTZ")
    parser.add_argument("--require-modes", nargs="+", default=list(VITAMIN_MODES))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    inputs = {"sass": args.sass, "ptx": args.ptx,
              "script": Path(__file__), "sass_reader": Path(sass_reader.__file__)}
    if args.binary:
        inputs["binary"] = args.binary
    if args.profile:
        inputs["profile"] = args.profile
    if args.compare_sass:
        inputs["compare_sass"] = args.compare_sass
    evidence = {key: {"path": str(path.resolve()),
                     "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                for key, path in inputs.items()}
    sass, sass_warnings = analyze_sass(args.sass.read_text(encoding="utf-8-sig"), args.label)
    ptx, ptx_warnings = analyze_ptx(args.ptx.read_text(encoding="utf-8-sig"), args.label)
    for language, tables in (("SASS", sass), ("PTX", ptx)):
        missing = set(args.require_modes) - {row["mode"] for row in tables["functions"]}
        if missing:
            raise ValueError(f"{language} missing required modes: {sorted(missing)}")
    ranges = profile_ranges(args.profile.read_text(encoding="utf-8-sig"), args.label) if args.profile else []
    metadata = {
        "build": args.label, "inputs": evidence, "required_modes": args.require_modes,
        "loop_selection": "Backward branch interval with maximum static FMA count; shortest tie flagged.",
        "warnings": sass_warnings + ptx_warnings,
        "limitations": [
            "Static instruction counts are not dynamic execution counts, cycles, or latency.",
            "PTX is virtual ISA. Source/PTX prefetch order does not establish final SASS order.",
            "Load positions and instruction distances alone cannot prove overlap or latency hiding.",
            "Backward branch selection is a structural heuristic, not a complete control-flow analysis.",
            "SASS loop counts exclude @!PT; other predicates are not dynamically evaluated.",
            "Executable dumps do not establish dynamically linked cuBLAS instructions.",
            "NCU ranges describe observed samples, not confidence intervals or normal benchmark timing.",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for language, tables in (("sass", sass), ("ptx", ptx)):
        for name, rows in tables.items():
            save_csv(args.output_dir / f"{language}-{name}.csv", rows)
    if args.profile:
        save_csv(args.output_dir / "profile-ranges.csv", ranges)
    if args.compare_sass:
        with (args.output_dir / "config-comparison.json").open("x", encoding="utf-8") as stream:
            json.dump(compare_sass(args.sass, args.compare_sass), stream, indent=2, ensure_ascii=False)
            stream.write("\n")
    with (args.output_dir / "analysis.json").open("x", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(f"Saved CPU-only evidence to {args.output_dir}")
    for row in sass["summary"]:
        print(f"{row['mode']}: SASS loop FFMA={row['static_fma']} LDG={row['LDG']} "
              f"LDS={row['LDS']} STS={row['STS']} BAR={row['BAR']}; "
              f"LDG FFMA-after={row['ldg_min_ffma_after']}..{row['ldg_max_ffma_after']}")
    for warning in metadata["warnings"]:
        print(f"WARNING: {warning}")


if __name__ == "__main__":
    main()
