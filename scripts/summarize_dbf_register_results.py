"""CPU-only audit of paired DBF binaries; keep every unprofiled CUDA event."""
import argparse
import csv
from collections import defaultdict
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics as st


def read(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write(path, rows):
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    root = args.result_dir.resolve()
    out = args.output_dir or root / "analysis"
    out.mkdir(parents=True, exist_ok=False)
    metadata = json.loads((root/"environment.json").read_text(encoding="utf-8-sig"))
    invocations = read(root/"invocations.csv")
    sizes, pairs, repeats = metadata["sizes"], metadata["pairs"], metadata["repeats"]
    assert len(invocations) == 2 + len(sizes)*pairs*2
    seen, fingerprints, checks, rounds, raw_hashes = set(), defaultdict(set), [], [], []
    event_count = 0
    for invocation in invocations:
        n, pair, order = int(invocation["n"]), int(invocation["pair"]), int(invocation["order"])
        variant, phase = invocation["variant"], invocation["phase"]
        key = phase, n, pair, variant
        assert key not in seen
        seen.add(key)
        assert variant in ("default", "maxr128") and int(invocation["exit_code"]) == 0
        if phase == "timings":
            assert n in sizes and 1 <= pair <= pairs
            assert order == (1 if (pair % 2 == 1) == (variant == "default") else 2)
        else:
            assert phase == "correctness" and n == 128 and pair == order == 0
        path = root/invocation["event_file"]
        events, validation = read(path), read(path.with_name(path.name+".correctness.csv"))
        count = repeats if phase == "timings" else 1
        assert len(events) == count and [int(e["iteration"]) for e in events] == list(range(1, count+1))
        assert len(validation) == 3
        assert {(r["phase"], r["implementation"], int(r["round"])) for r in validation} == {
            ("reference", "independent_strict_reference", 0),
            ("preflight", "sgemm_at_bcf_swizzling_dbf_rw", 0),
            ("timed_round", "sgemm_at_bcf_swizzling_dbf_rw", 1)}
        for record in validation:
            assert record["configuration"] == f"dbf_{variant}"
            assert int(record["n"]) == n and float(record["alpha"]) == 1 and float(record["beta"]) == 0
            assert int(record["seed"]) == 20261020 and int(record["full_element_count"]) == n*n
            assert int(record["cpu_reference_count"]) == (16384 if n == 128 else 256)
            assert record["status"] == "PASS"
            for field in ("failure_count", "nonfinite_count", "reference_nonfinite_count",
                          "cpu_reference_failure_count", "cpu_reference_nonfinite_count",
                          "cpu_implementation_failure_count", "cpu_implementation_nonfinite_count"):
                assert int(record[field]) == 0, (path, field)
            for field in ("max_abs_error", "max_rel_error", "cpu_reference_max_abs_error",
                          "cpu_reference_max_rel_error", "cpu_implementation_max_abs_error",
                          "cpu_implementation_max_rel_error"):
                value = float(record[field])
                assert math.isfinite(value) and value >= 0
                if "abs" in field:
                    assert value <= 0.01
            assert (record["upstream_allclose_status"] == "PASS") == (int(record["upstream_allclose_failure_count"]) == 0)
            fingerprints[n].add(tuple(record[k] for k in ("a_hash", "b_hash", "c0_hash")))
            checks.append(dict(source=path.name, pair=pair, variant=variant, **record))
        last = next(r for r in validation if r["phase"] == "timed_round")
        first, timings = events[0], [float(e["event_ms"]) for e in events]
        mean = st.mean(timings)
        for event in events:
            assert int(event["round"]) == int(event["order"]) == 1
            assert event["implementation"] == "sgemm_at_bcf_swizzling_dbf_rw"
            assert math.isfinite(float(event["event_ms"])) and float(event["event_ms"]) > 0
            assert math.isclose(float(event["round_mean_ms"]), mean, rel_tol=1e-12)
            assert math.isclose(float(event["gflops"]), 2*n**3/(float(event["event_ms"])*1e6), rel_tol=1e-12)
            for field in set(first)-{"iteration", "event_ms", "gflops"}:
                assert event[field] == first[field]
            for field in set(event) & set(last):
                assert event[field] == last[field], (path, field)
            for field in ("max_abs_error", "max_rel_error", "failure_count", "nonfinite_count"):
                assert event["round_"+field] == last[field]
        if phase == "timings":
            event_count += len(events)
            med = st.median(timings)
            rounds.append(dict(n=n, pair=pair, variant=variant, order=order,
                               event_count=len(events), mean_ms=mean,
                               tflops=2*n**3/(mean*1e9), min_event_ms=min(timings),
                               median_event_ms=med, max_event_ms=max(timings),
                               max_event_over_median=max(timings)/med,
                               max_abs_error=float(last["max_abs_error"]),
                               cpu_max_abs_error=float(last["cpu_implementation_max_abs_error"]),
                               upstream_allclose_status=last["upstream_allclose_status"],
                               round_start_unix_ms=int(first["round_start_unix_ms"]),
                               round_end_unix_ms=int(first["round_end_unix_ms"])))
    assert all(len(v) == 1 for v in fingerprints.values())
    assert len(rounds) == len(sizes)*pairs*2
    summary, paired, pair_summary = [], [], []
    for n in sizes:
        for variant in ("default", "maxr128"):
            selected = [r for r in rounds if r["n"] == n and r["variant"] == variant]
            assert len(selected) == pairs and {r["pair"] for r in selected} == set(range(1, pairs+1))
            times = [r["mean_ms"] for r in selected]
            med = st.median(times)
            summary.append(dict(n=n, variant=variant, rounds=pairs, events=pairs*repeats,
                                median_round_ms=med, median_tflops=2*n**3/(med*1e9),
                                min_round_ms=min(times), max_round_ms=max(times),
                                range_over_median_pct=100*(max(times)-min(times))/med,
                                slowest_event_ms=max(r["max_event_ms"] for r in selected)))
        speedups = []
        for pair in range(1, pairs+1):
            baseline = next(r for r in rounds if (r["n"],r["pair"],r["variant"]) == (n,pair,"default"))
            limited = next(r for r in rounds if (r["n"],r["pair"],r["variant"]) == (n,pair,"maxr128"))
            change = 100*(baseline["mean_ms"]/limited["mean_ms"]-1)
            speedups.append(change)
            paired.append(dict(n=n, pair=pair, default_mean_ms=baseline["mean_ms"],
                               maxr128_mean_ms=limited["mean_ms"], maxr128_first=limited["order"] == 1,
                               speedup_pct=change))
        pair_summary.append(dict(n=n, pairs=pairs, median_speedup_pct=st.median(speedups),
                                 min_speedup_pct=min(speedups), max_speedup_pct=max(speedups),
                                 positive_pairs=sum(x>0 for x in speedups)))
    zone = timezone(timedelta(minutes=metadata["host_utc_offset_minutes"]))
    telemetry = []
    for raw in read(root/"telemetry.csv"):
        item = {k.strip(): v.strip() for k,v in raw.items() if k is not None and v is not None}
        if not item.get("timestamp") or item["timestamp"] == "timestamp":
            continue
        item["unix_ms"] = datetime.strptime(item["timestamp"], "%Y/%m/%d %H:%M:%S.%f").replace(tzinfo=zone).timestamp()*1000
        telemetry.append(item)
    windows = []
    for row in rounds:
        points = [p for p in telemetry if p.get("index") == "0" and
                  row["round_start_unix_ms"] <= p["unix_ms"] <= row["round_end_unix_ms"]]
        item = {k:row[k] for k in ("n", "pair", "variant", "round_start_unix_ms", "round_end_unix_ms")}
        item["telemetry_points"] = len(points)
        for name,field in {"temperature_c":"temperature.gpu", "sm_clock_mhz":"clocks.current.sm [MHz]",
                           "memory_clock_mhz":"clocks.current.memory [MHz]"}.items():
            # nvidia-smi --format=csv 默认保留 "2520 MHz" 等单位。
            values = [float(p[field].split()[0]) for p in points]
            item[name+"_min"] = min(values) if values else ""
            item[name+"_max"] = max(values) if values else ""
        windows.append(item)
    for path in sorted(root.glob("*.csv")):
        raw_hashes.append(dict(file=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    for filename, records in (("rounds",rounds), ("summary",summary), ("paired",paired),
                              ("paired-summary",pair_summary), ("correctness-all",checks),
                              ("round-telemetry",windows), ("raw-sha256",raw_hashes)):
        write(out/(filename+".csv"), records)
    report = dict(status="PASS", event_count=event_count, small_events=2, rounds=len(rounds),
                  correctness_rows=len(checks), max_abs_error=max(float(r["max_abs_error"]) for r in checks),
                  max_cpu_abs_error=max(float(r["cpu_implementation_max_abs_error"]) for r in checks),
                  allclose_failed_checks=sum(r["upstream_allclose_status"] != "PASS" for r in checks),
                  slow_samples_removed=0, input_fingerprints={str(n):list(next(iter(v))) for n,v in fingerprints.items()},
                  telemetry_raw_points=len(telemetry), covered_windows=sum(r["telemetry_points"]>0 for r in windows),
                  paired_summary=pair_summary,
                  limitations=["Separate process per paired member; clocks unlocked, one WDDM session.",
                               "Pair ratios descriptive; no cross-session confidence interval.",
                               "Only final timed output checked; CPU double full128, fixed256points elsewhere.",
                               "Per-round UTC windows include warmup and validation; missing telemetry remains missing.",
                               "Register cap may change scheduling; performance cannot be assigned solely to occupancy."])
    (out/"audit.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,indent=2))


if __name__ == "__main__":
    main()
