#!/usr/bin/env python3
"""Enumerate five vitamin-cuda SIMT kernels without executing CUDA.

Formula source: vitamin-cuda, kernels/sgemm/sgemm.cu at
520f97093a2abc239977be83d244b327d0e2e2ef, copyright (c) 2025 wingedge,
MIT license. The upstream checkout and its license remain unchanged.

This verifies transcribed source formulas for the exact default shape. Bank
projections are address models, not measured SASS transactions or conflicts.
Every output directory must be new. Only the Python standard library is used.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import traceback


COMMIT = "520f97093a2abc239977be83d244b327d0e2e2ef"
MODES = (
    "sgemm_tiling",
    "sgemm_at_tiling",
    "sgemm_at_bcf_swizzling",
    "sgemm_at_bcf_swizzling_rw",
    "sgemm_at_bcf_swizzling_dbf_rw",
)
BM = BN = 128
BK = 16
THREADS = 256
REPO = Path(__file__).resolve().parent.parent


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, fields, records):
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


def coordinates(mode, thread):
    warp, lane = divmod(thread, 32)
    if mode.endswith("_rw"):
        rows = [warp * 16 + (lane // 16) * 8 + i for i in range(8)]
        cols = [(lane % 16) * 4 + (j // 4) * 64 + j % 4 for j in range(8)]
    else:
        rows = [(warp // 4) * 64 + (lane // 4) * 8 + i for i in range(8)]
        cols = [(warp % 4) * 32 + (lane % 4) * 8 + j for j in range(8)]
    return rows, cols


def a_shared(mode, m, k):
    if mode == "sgemm_tiling":
        return m * BK + k
    physical_m = m ^ ((k >> 2) << 3) if "swizzling" in mode else m
    return k * BM + physical_m


def coverage(records, coordinates_of, expected, context):
    counts = Counter(coordinates_of(row) for row in records)
    observed = set(counts)
    duplicates = sum(count - 1 for count in counts.values())
    missing, out_of_bounds = expected - observed, observed - expected
    require(not duplicates and not missing and not out_of_bounds, context)
    return {
        "records": len(records), "unique": len(observed), "duplicates": duplicates,
        "missing": len(missing), "out_of_bounds": len(out_of_bounds),
    }


def output_records(mode):
    records = []
    for thread in range(THREADS):
        rows, cols = coordinates(mode, thread)
        for i, m in enumerate(rows):
            for j, n in enumerate(cols):
                records.append({
                    "thread": thread, "warp": thread // 32, "lane": thread % 32,
                    "a_register": i, "b_register": j, "accumulator": i * 8 + j,
                    "m": m, "n": n, "c_flat": m * BN + n,
                })
    stats = coverage(records, lambda r: (r["m"], r["n"]),
                     {(m, n) for m in range(BM) for n in range(BN)}, mode + " C")
    for thread in range(THREADS):
        own = [r for r in records if r["thread"] == thread]
        require(len(own) == 64 and {r["accumulator"] for r in own} == set(range(64)),
                mode + " accumulator coverage")
    for warp in range(8):
        own = {(r["m"], r["n"]) for r in records if r["warp"] == warp}
        if mode.endswith("_rw"):
            expected = {(m, n) for m in range(warp * 16, warp * 16 + 16) for n in range(128)}
        else:
            expected = {(m, n) for m in range((warp // 4) * 64, (warp // 4 + 1) * 64)
                        for n in range((warp % 4) * 32, (warp % 4 + 1) * 32)}
        require(own == expected, mode + " warp rectangle")
    return records, stats


def producer_records(mode):
    records = []
    for thread in range(THREADS):
        for operand in ("A", "B"):
            for load_pass in range(2):
                for component in range(4):
                    if operand == "A":
                        row, col = thread // 4 + load_pass * 64, (thread % 4) * 4 + component
                        shared = a_shared(mode, row, col)
                        global_flat = row * BK + col
                    else:
                        row, col = thread // 32 + load_pass * 8, (thread % 32) * 4 + component
                        shared = global_flat = row * BN + col
                    records.append({
                        "operand": operand, "thread": thread, "warp": thread // 32,
                        "lane": thread % 32, "pass": load_pass, "component": component,
                        "logical_row": row, "logical_col": col,
                        "global_tile_flat": global_flat, "shared_slot0_float": shared,
                        "bank32": shared % 32,
                        "store_width_floats": 4 if operand == "B" or mode == "sgemm_tiling" else 1,
                    })
    stats, lookup = {}, {}
    for operand, rows, cols in (("A", BM, BK), ("B", BK, BN)):
        own = [r for r in records if r["operand"] == operand]
        stats[operand] = coverage(
            own, lambda r: (r["logical_row"], r["logical_col"]),
            {(r, c) for r in range(rows) for c in range(cols)}, mode + " producer " + operand)
        physical = [r["shared_slot0_float"] for r in own]
        require(set(physical) == set(range(2048)) and len(set(physical)) == len(physical),
                mode + " shared bijection " + operand)
        lookup[operand] = {
            r["shared_slot0_float"]: (r["logical_row"], r["logical_col"]) for r in own
        }
    if "dbf" in mode:
        for operand in ("A", "B"):
            both = [slot * 2048 + address for slot in range(2) for address in lookup[operand]]
            require(len(set(both)) == 4096 and min(both) == 0 and max(both) == 4095,
                    "double-buffer slots must be disjoint and in bounds")
    return records, stats, lookup


def consumer_records(mode, producers):
    """Enumerate every per-k register load; verify what its producer stored."""
    for thread in range(THREADS):
        rows, cols = coordinates(mode, thread)
        for k in range(BK):
            for operand, owned in (("A", rows), ("B", cols)):
                for register, coordinate in enumerate(owned):
                    logical = (coordinate, k) if operand == "A" else (k, coordinate)
                    shared = a_shared(mode, coordinate, k) if operand == "A" else k * BN + coordinate
                    require(producers[operand][shared] == logical, mode + " producer-consumer mismatch")
                    scalar = operand == "A" and mode == "sgemm_tiling"
                    vector_start = shared if scalar else (
                        a_shared(mode, rows[(register // 4) * 4], k) if operand == "A"
                        else k * BN + cols[(register // 4) * 4])
                    require(scalar or (vector_start % 4 == 0 and shared == vector_start + register % 4),
                            mode + " float4 alignment/order")
                    yield {
                        "operand": operand, "thread": thread, "warp": thread // 32,
                        "lane": thread % 32, "k": k, "register": register,
                        "logical_row": logical[0], "logical_col": logical[1],
                        "shared_slot0_float": shared, "bank32": shared % 32,
                        "instruction_group": register if scalar else register // 4,
                        "component": 0 if scalar else register % 4,
                        "load_width_floats": 1 if scalar else 4,
                    }


def summarize_banks(accesses):
    """Count unique addresses per bank; same-address read is a broadcast."""
    banks = defaultdict(set)
    address_counts = Counter(accesses)
    for address in accesses:
        banks[address % 32].add(address)
    return {
        "requests": len(accesses), "unique_addresses": len(address_counts),
        "distinct_banks": len(banks),
        "max_distinct_addresses_per_bank": max(map(len, banks.values())),
        "duplicate_address_requests": sum(count - 1 for count in address_counts.values()),
    }


def bank_projection_records(mode, producers, consumers):
    # Scalar instructions: all 32 lanes. float4 model: each contiguous group of
    # eight lanes requests at most 128 B; actual emitted width/transactions need SASS/NCU.
    grouped = defaultdict(list)
    for r in producers:
        width = r["store_width_floats"]
        instruction = r["component"] if width == 1 else 0
        group = 0 if width == 1 else r["lane"] // 8
        key = ("store", r["operand"], r["warp"], r["pass"], instruction, group, width)
        grouped[key].append(r["shared_slot0_float"])
    for r in consumers:
        width = r["load_width_floats"]
        group = 0 if width == 1 else r["lane"] // 8
        key = ("load", r["operand"], r["warp"], r["k"], r["instruction_group"], group, width)
        grouped[key].append(r["shared_slot0_float"])
    for key, addresses in sorted(grouped.items()):
        phase, operand, warp, step, instruction, lane_group, width = key
        yield {
            "phase": phase, "operand": operand, "warp": warp, "step": step,
            "instruction_group": instruction, "lane_group": lane_group,
            "width_floats": width,
            "model": "scalar_32_lanes" if width == 1 else "float4_8_lane_128B_projection",
            **summarize_banks(addresses),
        }


def c_sector_records(mode):
    # Coordinates here use N=128, aligned C. The same result holds for the
    # tested N multiples of 128; counts describe each source float4 instruction.
    for warp in range(8):
        for row_register in range(8):
            for vector_pass in range(2):
                addresses = []
                for lane in range(32):
                    rows, cols = coordinates(mode, warp * 32 + lane)
                    addresses.extend(rows[row_register] * 128 + cols[vector_pass * 4 + q] for q in range(4))
                require(len(set(addresses)) == 128, mode + " C write collision")
                sectors = {address // 8 for address in addresses}
                yield {
                    "warp": warp, "row_register": row_register, "vector_pass": vector_pass,
                    "requested_bytes": len(addresses) * 4, "sectors32B": len(sectors),
                    "requested_bytes_per_sector_bytes": len(addresses) * 4 / (32 * len(sectors)),
                }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=REPO.parent / "vitamin-cuda")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    source = args.source_root.resolve()
    kernel = source / "kernels/sgemm/sgemm.cu"
    require(kernel.is_file(), "Upstream kernels/sgemm/sgemm.cu is missing")
    head = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    require(head == COMMIT, "This analyzer is audited for the fixed upstream commit only")
    git_source = subprocess.check_output(["git", "-C", str(source), "show", f"{COMMIT}:kernels/sgemm/sgemm.cu"])
    require(kernel.read_bytes().replace(b"\r\n", b"\n") == git_source.replace(b"\r\n", b"\n"),
            "Upstream working source differs from the audited commit")
    require(not args.output_dir.exists(), "Refusing to overwrite an existing output directory")
    args.output_dir.mkdir(parents=True)
    summary = {
        "status": "PASS", "shape": {"BM": BM, "BN": BN, "BK": BK, "TM": 8, "TN": 8, "threads": THREADS},
        "source_commit": COMMIT, "source_root": str(source), "python": sys.version,
        "platform": platform.platform(), "source_sha256": sha256(kernel),
        "license_sha256": sha256(source / "LICENSE"), "analyzer_sha256": sha256(Path(__file__)),
        "scope": "Transcribed default-shape formulas, one K tile and slot0; dbf slot separation checked independently. No GPU execution.",
        "bank_model": "bank=(float_offset%32); scalar full warp; float4 contiguous eight-lane 128-byte projection. Same-address reads count once. Not a measured conflict counter.",
        "implementations": {},
    }
    lane_examples = []
    try:
        for mode in MODES:
            outputs, output_stats = output_records(mode)
            producers, producer_stats, producer_lookup = producer_records(mode)
            consumers = list(consumer_records(mode, producer_lookup))
            banks = list(bank_projection_records(mode, producers, consumers))
            sectors = list(c_sector_records(mode))
            for suffix, rows in (("outputs", outputs), ("producers", producers), ("consumers", consumers),
                                 ("bank-projections", banks), ("c-write-sectors", sectors)):
                write_csv(args.output_dir / f"{mode}-{suffix}.csv", tuple(rows[0]), rows)
            for lane in (0, 1, 2, 15, 16, 31):
                rows, cols = coordinates(mode, lane)
                lane_examples.append({
                    "implementation": mode, "warp": 0, "lane": lane,
                    "rows": " ".join(map(str, rows)), "cols": " ".join(map(str, cols)),
                })
            scalar_stores = [r["max_distinct_addresses_per_bank"] for r in banks
                             if r["phase"] == "store" and r["operand"] == "A" and r["width_floats"] == 1]
            a_loads = [r["max_distinct_addresses_per_bank"] for r in banks
                       if r["phase"] == "load" and r["operand"] == "A"]
            summary["implementations"][mode] = {
                "outputs": output_stats, "producers": producer_stats,
                "consumer_checks": len(consumers), "producer_consumer_mismatches": 0,
                "float4_alignment_or_order_failures": 0,
                "shared_slots": 2 if "dbf" in mode else 1,
                "user_shared_bytes": 32768 if "dbf" in mode else 16384,
                "bank_projection_rows": len(banks),
                "a_scalar_store_max_addresses_per_bank": sorted(set(scalar_stores)),
                "a_load_model_max_addresses_per_bank": sorted(set(a_loads)),
                "c_sectors32B_per_source_float4_instruction": sorted({r["sectors32B"] for r in sectors}),
            }
            print(f"PASS {mode}: C={len(outputs)}, producer={len(producers)}, consumer={len(consumers)}")
        write_csv(args.output_dir / "warp0-lane-examples.csv", tuple(lane_examples[0]), lane_examples)
        (args.output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    except Exception:
        (args.output_dir / "FAILED.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    print("All five full-CTA/source-address audits passed. No GPU code executed.")


if __name__ == "__main__":
    main()
