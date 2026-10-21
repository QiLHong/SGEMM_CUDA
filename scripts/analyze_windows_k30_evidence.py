#!/usr/bin/env python3
"""Reproduce K30 PTX placement and per-metric profiler ranges using only CPU.

Example:
  python scripts/analyze_windows_k30_evidence.py --ptx windows_benchmark.ptx \
      --profile profile01/ncu-summary.csv --output-dir new-evidence-dir

Reads existing evidence without invoking a compiler, profiler, or GPU kernel.
The profile input is produced by scripts/windows/Profile-K30Sync.ps1. PTX line
positions describe virtual instruction order, not final SASS or runtime latency.
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


PTX_FIELDS = (
    "mode", "loop_begin_entry_line", "loop_end_entry_line",
    "position_in_loop_lines", "ffma_before", "ffma_after", "instruction",
)
RANGE_FIELDS = ("size", "mode", "metric", "samples", "minimum", "median", "maximum")
PROFILE_ID_FIELDS = {"size", "mode", "profile_repeat", "kernel", "source_csv"}
MODE_ORDER = ("k10", "k30_v1", "k30_v2", "strict")


def ptx_placement(text: str) -> tuple[list[dict], list[dict]]:
    rows, diagnostics = [], []
    recognized = set()
    for block in re.split(r"(?=\.visible \.entry )", text):
        if not block.startswith(".visible .entry "):
            continue
        name = block.splitlines()[0]
        if "sgemmK30" not in name:
            continue
        stage = re.search(r"ELi([12])EEv", name)
        if not stage:
            continue
        mode = f"k30_v{stage[1]}"
        if mode in recognized:
            raise ValueError(f"More than one {mode} PTX function; provide a single-architecture dump.")
        recognized.add(mode)
        lines = block.splitlines()
        labels = {
            match[1]: index for index, line in enumerate(lines)
            if (match := re.match(r"\s*(\$[\w_]+):", line))
        }
        loops = []
        for index, line in enumerate(lines):
            branch = re.search(r"\bbra(?:\.uni)?\s+(\$[\w_]+);", line)
            if branch and branch[1] in labels and labels[branch[1]] < index:
                begin = labels[branch[1]]
                body = lines[begin:index + 1]
                count = sum("fma.rn.f32 " in instruction for instruction in body)
                if count:
                    loops.append((count, begin, index, body))
        if not loops:
            raise ValueError(f"No backward PTX branch enclosing fma.rn.f32 found for {mode}.")
        maximum = max(loop[0] for loop in loops)
        candidates = [loop for loop in loops if loop[0] == maximum]
        # 与 SASS 分析一样，不把历史 2048 写进识别规则。
        if len(candidates) != 1:
            raise ValueError(f"Ambiguous maximal-FMA PTX loop for {mode}; inspect the PTX manually.")
        count, begin, end, body = candidates[0]
        before = 0
        for position, line in enumerate(body):
            if "ld.global" in line or "bar.sync" in line:
                rows.append({
                    "mode": mode, "loop_begin_entry_line": begin + 1,
                    "loop_end_entry_line": end + 1, "position_in_loop_lines": position,
                    "ffma_before": before, "ffma_after": count - before,
                    "instruction": line.strip(),
                })
            before += "fma.rn.f32 " in line
        diagnostics.append({
            "mode": mode, "function": name, "loop_static_fma": count,
            "backward_branch_arithmetic_candidates": len(loops),
            "cp_async_occurrences": block.count("cp.async"),
        })
    if recognized != {"k30_v1", "k30_v2"}:
        raise ValueError(f"Expected both K30 V1/V2 symbols, found {sorted(recognized)}.")
    return rows, diagnostics


def profile_ranges(text: str) -> tuple[list[dict], list[dict]]:
    reader = csv.DictReader(io.StringIO(text))
    fields = reader.fieldnames or []
    if not PROFILE_ID_FIELDS <= set(fields):
        raise ValueError("Profile input must be ncu-summary.csv from Profile-K30Sync.ps1.")
    metric_names = [field for field in fields if field not in PROFILE_ID_FIELDS]
    if not metric_names:
        raise ValueError("No numeric profile metrics found.")
    data = list(reader)
    groups = collections.defaultdict(list)
    for row in data:
        size, repetition = int(row["size"]), int(row["profile_repeat"])
        if size <= 0 or repetition <= 0 or row["mode"] not in MODE_ORDER:
            raise ValueError(f"Unexpected profile identity: {row}")
        groups[(size, row["mode"])].append(row)
    if not groups:
        raise ValueError("Profile input contains no rows.")
    rows, diagnostics = [], []
    for size in sorted({size for size, _ in groups}):
        for mode in MODE_ORDER:
            samples = groups.get((size, mode), [])
            if not samples:
                raise ValueError(f"Missing profile group: size={size}, mode={mode}")
            repetitions = [int(row["profile_repeat"]) for row in samples]
            if len(set(repetitions)) != len(repetitions):
                raise ValueError(f"Duplicate profile repetitions: size={size}, mode={mode}")
            diagnostics.append({"size": size, "mode": mode, "repetitions": sorted(repetitions)})
            for metric in metric_names:
                values = [float(row[metric]) for row in samples]
                if not all(math.isfinite(value) for value in values):
                    raise ValueError(f"Nonfinite metric {metric}: size={size}, mode={mode}")
                rows.append({
                    "size": size, "mode": mode, "metric": metric, "samples": len(values),
                    "minimum": min(values), "median": statistics.median(values), "maximum": max(values),
                })
    return rows, diagnostics


def save_csv(path: Path, rows: list[dict], fields: tuple[str, ...]) -> None:
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ptx", type=Path, required=True, help="Saved cuobjdump --dump-ptx output")
    parser.add_argument("--profile", type=Path, required=True, help="Existing ncu-summary.csv")
    parser.add_argument("--output-dir", type=Path, required=True, help="New output directory; never overwritten")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    ptx_bytes = args.ptx.read_bytes()
    profile_bytes = args.profile.read_bytes()
    placement, ptx_diagnostics = ptx_placement(ptx_bytes.decode("utf-8-sig"))
    ranges, profile_diagnostics = profile_ranges(profile_bytes.decode("utf-8-sig"))
    metadata = {
        "ptx": str(args.ptx.resolve()), "ptx_sha256": hashlib.sha256(ptx_bytes).hexdigest(),
        "profile": str(args.profile.resolve()),
        "profile_sha256": hashlib.sha256(profile_bytes).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "ptx_loop_selection": "Unique backward-branch interval with maximal static fma.rn.f32 count",
        "ptx_diagnostics": ptx_diagnostics, "profile_groups": profile_diagnostics,
        "limitations": [
            "PTX is virtual ISA; inspect SASS before drawing machine-scheduling conclusions.",
            "Static line/instruction distances are not cycles or measured overlap.",
            "Profiler ranges describe available observations, not confidence intervals.",
            "NCU-instrumented CUDA-event timing is not unprofiled benchmark timing.",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    save_csv(args.output_dir / "ptx-placement.csv", placement, PTX_FIELDS)
    save_csv(args.output_dir / "profile-ranges.csv", ranges, RANGE_FIELDS)
    with (args.output_dir / "evidence-metadata.json").open("x", encoding="utf-8") as stream:
        json.dump(metadata, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(f"Saved {len(placement)} PTX positions and {len(ranges)} metric range rows to {args.output_dir}.")
    print("CPU-only evidence analysis; no GPU measurement executed.")


if __name__ == "__main__":
    main()
