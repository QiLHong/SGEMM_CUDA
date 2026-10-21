#!/usr/bin/env python3
"""CPU enumeration of K10 and K30 V1/V2 full-CTA ownership; no GPU required.

Sources: Simon Boehm's K10 in this repository and zh-cn-learning's K30
FragmentIterator / analyze_k30_mapping.py. Existing files remain unchanged.
This checks transcribed source formulas, not generated device instructions or
floating-point results. Output directories must be new.
"""

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import traceback


BM = BN = 128
BK = 16
THREADS = 128
A_STRIDE = 132
REPO = Path(__file__).resolve().parent.parent
SELECTED_LANES = (0, 1, 2, 31)
FIELDS = (
    "implementation", "thread", "warp", "lane", "a_register", "b_register",
    "accumulator", "m", "n", "c_flat", "shared_a_k0", "shared_b_k0",
    "shared_a_k_stride", "shared_b_k_stride",
)


def require(condition, message):
    # Explicit check rather than assert: python -O must not disable verification.
    if not condition:
        raise ValueError(message)


def write_csv(path, fields, rows):
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def coordinates(implementation, thread):
    warp, lane = divmod(thread, 32)
    warp_row, warp_col = divmod(warp, 2)
    if implementation == "k10":
        # K10: WMITER=1, WNITER=4, TM=8, TN=4.
        rows = [warp_row * 64 + (lane // 4) * 8 + i for i in range(8)]
        cols = [
            warp_col * 64 + (j // 4) * 16 + (lane % 4) * 4 + j % 4
            for j in range(16)
        ]
    else:
        # K30 V1/V2 share this FragmentIterator; STAGES changes lifetime only.
        lane_m = 2 * (lane // 16) + lane % 2
        lane_n = (lane % 16) // 2
        rows = [
            warp_row * 64 + 4 * lane_m + 16 * (i // 4) + i % 4
            for i in range(16)
        ]
        cols = [
            warp_col * 64 + 4 * lane_n + 32 * (j // 4) + j % 4
            for j in range(8)
        ]
    return rows, cols


def enumerate_ownership(implementation):
    records = []
    for thread in range(THREADS):
        warp, lane = divmod(thread, 32)
        rows, cols = coordinates(implementation, thread)
        for i, m in enumerate(rows):
            for j, n in enumerate(cols):
                records.append(dict(
                    implementation=implementation, thread=thread, warp=warp,
                    lane=lane, a_register=i, b_register=j,
                    accumulator=i * len(cols) + j, m=m, n=n,
                    c_flat=m * BN + n, shared_a_k0=m, shared_b_k0=n,
                    shared_a_k_stride=A_STRIDE, shared_b_k_stride=BN,
                ))
    counts = Counter((row["m"], row["n"]) for row in records)
    expected = {(m, n) for m in range(BM) for n in range(BN)}
    actual = set(counts)
    duplicate_outputs = sum(count - 1 for count in counts.values() if count > 1)
    missing = len(expected - actual)
    out_of_bounds = sum(
        not (0 <= row["m"] < BM and 0 <= row["n"] < BN)
        for row in records
    )
    summary = dict(
        implementation=implementation, csv_rows=len(records),
        distinct_outputs=len(actual), expected_outputs=BM * BN,
        duplicate_outputs=duplicate_outputs, missing_outputs=missing,
        out_of_bounds=out_of_bounds,
    )
    require(len(records) == BM * BN, f"{implementation}: wrong record count")
    require(duplicate_outputs == missing == out_of_bounds == 0,
            f"{implementation}: invalid coverage {summary}")
    for thread in range(THREADS):
        owned = [row for row in records if row["thread"] == thread]
        require(len(owned) == 128, f"{implementation}: thread {thread} count")
        require({row["accumulator"] for row in owned} == set(range(128)),
                f"{implementation}: thread {thread} accumulator coverage")
    for warp in range(4):
        owned = {(row["m"], row["n"]) for row in records if row["warp"] == warp}
        wr, wc = divmod(warp, 2)
        expected_warp = {
            (m, n)
            for m in range(wr * 64, (wr + 1) * 64)
            for n in range(wc * 64, (wc + 1) * 64)
        }
        require(owned == expected_warp, f"{implementation}: warp {warp} coverage")
    summary.update(per_thread_outputs=128, per_warp_outputs=4096, status="PASS")
    return records, summary


def check_global_shared_flow():
    # Both K10 and Small K30 load the same logical tile with 128 threads.
    # A: (m,k)->k*132+m; B: (k,n)->k*128+n. Padding is not matrix data.
    a_writes, b_writes = {}, {}
    for thread in range(THREADS):
        ar, aq = divmod(thread, BK // 4)
        br, bq = divmod(thread, BN // 4)
        for t in range(4):
            for q in range(4):
                m, k = ar + t * 32, aq * 4 + q
                a_offset = k * A_STRIDE + m
                require(a_offset not in a_writes, "Duplicate shared A writer")
                a_writes[a_offset] = (m, k, thread)
                k, n = br + t * 4, bq * 4 + q
                b_offset = k * BN + n
                require(b_offset not in b_writes, "Duplicate shared B writer")
                b_writes[b_offset] = (k, n, thread)
    require(len(a_writes) == BM * BK, "Incomplete A tile")
    require(len(b_writes) == BK * BN, "Incomplete B tile")
    for k in range(BK):
        for m in range(BM):
            require(a_writes[k * A_STRIDE + m][:2] == (m, k),
                    "A producer/consumer layout mismatch")
        for n in range(BN):
            require(b_writes[k * BN + n][:2] == (k, n),
                    "B producer/consumer layout mismatch")
    require(max(a_writes) < BK * A_STRIDE, "Shared A out of bounds")
    require(max(b_writes) < BK * BN, "Shared B out of bounds")
    return dict(
        shared_a_logical_elements=len(a_writes),
        shared_b_logical_elements=len(b_writes),
        shared_a_allocated_floats=BK * A_STRIDE,
        shared_a_unused_padding_floats=BK * (A_STRIDE - BM),
        one_stage_bytes=(BK * A_STRIDE + BK * BN) * 4,
        k30_v2_two_stage_bytes=2 * (BK * A_STRIDE + BK * BN) * 4,
        duplicate_writers=0, missing_elements=0, out_of_bounds=0, status="PASS",
    )


def legacy_crosscheck(output_dir, k30_rows):
    old_script = REPO / "scripts/analyze_k30_mapping.py"
    old_output = output_dir / "legacy-k30-warp0.csv"
    command = [sys.executable, str(old_script), "--output", str(old_output)]
    completed = subprocess.run(command, capture_output=True, text=True,
                               encoding="utf-8", errors="replace", check=False)
    with (output_dir / "legacy-crosscheck.log").open("x", encoding="utf-8") as log:
        log.write("COMMAND_JSON: " + json.dumps(command, ensure_ascii=False) + "\n")
        log.write("EXIT_CODE: " + str(completed.returncode) + "\n")
        log.write(completed.stdout)
        log.write(completed.stderr)
    require(completed.returncode == 0, "Existing analyze_k30_mapping.py failed")
    with old_output.open(encoding="utf-8", newline="") as handle:
        legacy = list(csv.DictReader(handle))
    current = [row for row in k30_rows if row["warp"] == 0]
    require(len(legacy) == len(current) == 4096, "Legacy warp0 row count mismatch")
    compare_fields = (
        "lane", "a_register", "b_register", "accumulator", "m", "n",
        "shared_a_k0", "shared_b_k0", "shared_a_k_stride", "shared_b_k_stride",
    )
    for index, (old, new) in enumerate(zip(legacy, current)):
        for field in compare_fields:
            require(int(old[field]) == new[field],
                    f"Legacy mismatch row={index} field={field}")
    return dict(
        legacy_export_rows=4096, compared_fields=list(compare_fields),
        matching_rows=4096, legacy_script_exit_code=0,
        note="Legacy script additionally checks 256x128 CTA bounds internally.",
        status="PASS",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    require(not output_dir.exists(), f"Refusing existing output directory: {output_dir}")
    all_rows, coverage, examples = {}, [], []
    for implementation in ("k10", "k30_v1", "k30_v2"):
        rows, proof = enumerate_ownership(implementation)
        all_rows[implementation] = rows
        coverage.append(proof)
        for lane in SELECTED_LANES:
            m, n = coordinates(implementation, lane)
            examples.append(dict(
                implementation=implementation, warp=0, lane=lane,
                rows=" ".join(map(str, m)), cols=" ".join(map(str, n)),
                output_count=len(m) * len(n),
            ))
    compare_fields = [field for field in FIELDS if field != "implementation"]
    require(
        all(tuple(row[field] for field in compare_fields) ==
            tuple(other[field] for field in compare_fields)
            for row, other in zip(all_rows["k30_v1"], all_rows["k30_v2"])),
        "K30 V1/V2 ownership should be identical",
    )
    flow = check_global_shared_flow()
    output_dir.mkdir(parents=True, exist_ok=False)
    try:
        for implementation, rows in all_rows.items():
            write_csv(output_dir / f"{implementation}-full-cta.csv", FIELDS, rows)
        write_csv(output_dir / "warp0-lane-examples.csv",
                  tuple(examples[0]), examples)
        legacy = legacy_crosscheck(output_dir, all_rows["k30_v2"])
        sources = (
            "src/kernels/10_kernel_warptiling.cuh",
            "src/kernels/30_kernel_mini_simt.cuh",
            "scripts/analyze_k30_mapping.py",
            "scripts/analyze_windows_k30_mapping.py",
        )
        fingerprints = {
            path: hashlib.sha256((REPO / path).read_bytes()).hexdigest()
            for path in sources
        }
        summary = dict(
            status="PASS", scope="CPU enumeration of source formulas; no GPU run",
            python_executable=sys.executable, python_version=platform.python_version(),
            platform=platform.platform(), source_sha256=fingerprints,
            shape=dict(BM=BM, BN=BN, BK=BK, threads=THREADS, A_stride=A_STRIDE),
            ownership=coverage, k30_v1_v2_identical_ownership=True,
            global_shared_flow=flow, legacy_crosscheck=legacy,
            selected_lanes=examples,
        )
        with (output_dir / "summary.json").open("x", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        for proof in coverage:
            print(f"{proof['implementation']}: rows={proof['csv_rows']} "
                  f"distinct={proof['distinct_outputs']} duplicates=0 missing=0 "
                  "out_of_bounds=0 per_thread=128 per_warp=4096 PASS")
        print("Global/shared A+B: 2048+2048 elements, exact writer coverage PASS")
        print("Legacy K30 warp0: 4096/4096 rows agree; V1/V2 ownership identical PASS")
        print(f"Output: {output_dir}")
    except Exception:
        (output_dir / "FAILURE.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
