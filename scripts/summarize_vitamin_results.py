"""Audit every raw CUDA event, then summarize rounds and paired comparisons."""
import argparse
import csv
from collections import defaultdict
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics as st

MODES = ["sgemm_tiling", "sgemm_at_tiling", "sgemm_at_bcf_swizzling",
         "sgemm_at_bcf_swizzling_rw", "sgemm_at_bcf_swizzling_dbf_rw",
         "k10", "k30_v2", "cublas"]
CONFIGS = {"A": "original_config", "B": "strict_fp32"}


def read(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write(path, rows):
    assert rows, path
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--result-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path)
    args = p.parse_args()
    root = args.result_dir.resolve()
    out = args.output_dir or root / "analysis"
    out.mkdir(parents=True, exist_ok=False)
    metadata = json.loads((root / "environment.json").read_text(encoding="utf-8-sig"))
    nr, ne = int(metadata["rounds"]), int(metadata["repeats"])
    allrounds, summaries, hashes, raw_hashes, correctness = [], [], defaultdict(set), [], []
    raw_count = 0
    for config, label in CONFIGS.items():
        for n in (1024, 2048, 4096):
            path = root / config / f"timings-{n}.csv"
            data = read(path)
            raw_count += len(data)
            assert len(data) == 8 * nr * ne, (path, len(data))
            grouped = defaultdict(list)
            for row in data:
                assert row["configuration"] == label and int(row["n"]) == n
                assert float(row["alpha"]) == 1 and float(row["beta"]) == 0
                assert int(row["seed"]) == 20261020
                assert row["status"] == "PASS"
                assert row["implementation"] in MODES
                assert 1 <= int(row["round"]) <= nr and 1 <= int(row["order"]) <= 8
                ms = float(row["event_ms"])
                assert math.isfinite(ms) and ms > 0
                assert math.isclose(float(row["gflops"]), 2*n**3/(ms*1e6), rel_tol=1e-12)
                for key in ("round_failure_count", "round_nonfinite_count",
                            "reference_nonfinite_count", "cpu_reference_failure_count",
                            "cpu_reference_nonfinite_count", "cpu_implementation_failure_count",
                            "cpu_implementation_nonfinite_count"):
                    assert int(row[key]) == 0, (path, key, row[key])
                hashes[n].add((row["a_hash"], row["b_hash"], row["c0_hash"]))
                grouped[(row["implementation"], int(row["round"]))].append(row)
            assert len(grouped) == 8 * nr
            for (impl, round_id), events in grouped.items():
                assert len(events) == ne
                iterations = [int(e["iteration"]) for e in events]
                assert sorted(iterations) in [list(range(ne)), list(range(1, ne+1))]
                vals = [float(e["event_ms"]) for e in events]
                mean = st.mean(vals)
                assert all(math.isclose(float(e["round_mean_ms"]), mean, rel_tol=1e-12)
                           for e in events)
                first = events[0]
                fixed = set(first) - {"iteration", "event_ms", "gflops"}
                assert all(all(e[k] == first[k] for k in fixed) for e in events)
                assert int(first["round_start_unix_ms"]) <= int(first["round_end_unix_ms"])
                allrounds.append(dict(
                    configuration=config, n=n, implementation=impl, round=round_id,
                    order=int(first["order"]), event_count=len(vals), mean_ms=mean,
                    tflops=2*n**3/(mean*1e9), event_min_ms=min(vals), event_max_ms=max(vals),
                    event_median_ms=st.median(vals), event_cv_pct=100*st.pstdev(vals)/mean,
                    max_abs_error=max(float(e["round_max_abs_error"]) for e in events),
                    upstream_allclose_status=first["upstream_allclose_status"],
                    round_start_unix_ms=int(first["round_start_unix_ms"]),
                    round_end_unix_ms=int(first["round_end_unix_ms"])))
            for impl in MODES:
                rounds = [r for r in allrounds if r["configuration"] == config
                          and r["n"] == n and r["implementation"] == impl]
                assert len(rounds) == nr and len({r["round"] for r in rounds}) == nr
                vals = [r["mean_ms"] for r in rounds]
                med = st.median(vals)
                summaries.append(dict(
                    configuration=config, n=n, implementation=impl, rounds=nr,
                    events=nr*ne, median_round_mean_ms=med, median_tflops=2*n**3/(med*1e9),
                    min_round_mean_ms=min(vals), max_round_mean_ms=max(vals),
                    round_range_over_median_pct=100*(max(vals)-min(vals))/med,
                    round_cv_pct=100*st.pstdev(vals)/st.mean(vals),
                    slowest_event_ms=max(r["event_max_ms"] for r in rounds),
                    allclose_failed_rounds=sum(r["upstream_allclose_status"] != "PASS" for r in rounds)))
        expected_checks = [(n, root/config/f"correctness-{n}.csv.correctness.csv", 1) for n in (128,256,512)]
        expected_checks += [(n, root/config/f"timings-{n}.csv.correctness.csv", nr) for n in (1024,2048,4096)]
        for expected_n, path, expected_rounds in expected_checks:
            records = read(path)
            assert len(records) == 1+8+8*expected_rounds, (path, len(records))
            expected_keys = {("reference","independent_strict_reference",0)}
            expected_keys |= {("preflight",m,0) for m in MODES}
            expected_keys |= {("timed_round",m,r) for m in MODES for r in range(1,expected_rounds+1)}
            assert {(r["phase"],r["implementation"],int(r["round"])) for r in records} == expected_keys
            for row in records:
                row = dict(source=str(path.relative_to(root)), **row)
                correctness.append(row)
                assert row["configuration"] == label and row["status"] == "PASS"
                assert int(row["n"]) == expected_n and int(row["full_element_count"]) == expected_n**2
                assert float(row["alpha"]) == 1 and float(row["beta"]) == 0 and int(row["seed"]) == 20261020
                for key in ("max_abs_error","max_rel_error","cpu_reference_max_abs_error",
                            "cpu_reference_max_rel_error","cpu_implementation_max_abs_error",
                            "cpu_implementation_max_rel_error"):
                    value = float(row[key])
                    assert math.isfinite(value) and value >= 0
                    if "abs" in key: assert value <= 0.01
                assert (row["upstream_allclose_status"] == "PASS") == (int(row["upstream_allclose_failure_count"]) == 0)
                for key in ("failure_count", "nonfinite_count", "reference_nonfinite_count",
                            "cpu_reference_failure_count", "cpu_reference_nonfinite_count",
                            "cpu_implementation_failure_count", "cpu_implementation_nonfinite_count"):
                    assert int(row[key]) == 0, (path, key)
                n = int(row["n"])
                assert int(row["cpu_reference_count"]) == (n*n if n == 128 else 256)
                hashes[n].add((row["a_hash"], row["b_hash"], row["c0_hash"]))
        rejects = read(root / config / "rejected-shapes.csv")
        assert {int(r["n"]) for r in rejects} == {0,16,127,129,255,4100}
        assert len(rejects) == 6 and all(int(r["exit_code"]) != 0 and
                                       r["created_output"].lower() == "false" for r in rejects)
    assert all(len(v) == 1 for v in hashes.values()), "Input bytes differed across implementations/configs"
    # All eight orders exactly once per round; RW/DBF are adjacent and swap which goes first.
    for config in CONFIGS:
        for n in (1024, 2048, 4096):
            for r in sorted({x["round"] for x in allrounds}):
                group = [x for x in allrounds if (x["configuration"],x["n"],x["round"]) == (config,n,r)]
                assert len({x["order"] for x in group}) == 8
                rw = next(x for x in group if x["implementation"] == MODES[3])
                db = next(x for x in group if x["implementation"] == MODES[4])
                assert abs(rw["order"] - db["order"]) == 1
            signs = {next(x["order"] for x in allrounds if (x["configuration"],x["n"],x["round"],x["implementation"]) == (config,n,r,MODES[3])) <
                     next(x["order"] for x in allrounds if (x["configuration"],x["n"],x["round"],x["implementation"]) == (config,n,r,MODES[4]))
                     for r in sorted({x["round"] for x in allrounds})}
            assert len(signs) == 2
    pair_specs = list(zip(MODES[:4], MODES[1:5])) + [("k30_v2", MODES[4])] + [("cublas", m) for m in MODES[:-1]]
    pairs, pair_summary = [], []
    index = {(r["configuration"],r["n"],r["implementation"],r["round"]):r for r in allrounds}
    for config in CONFIGS:
        for n in (1024, 2048, 4096):
            for baseline, candidate in pair_specs:
                group = []
                for round_id in sorted({r["round"] for r in allrounds}):
                    b, c = index[(config,n,baseline,round_id)], index[(config,n,candidate,round_id)]
                    group.append(dict(configuration=config,n=n,baseline=baseline,candidate=candidate,
                                      round=round_id,baseline_mean_ms=b["mean_ms"],candidate_mean_ms=c["mean_ms"],
                                      speedup_pct=100*(b["mean_ms"]/c["mean_ms"]-1),
                                      adjacent=abs(b["order"]-c["order"])==1,
                                      candidate_first=c["order"]<b["order"]))
                pairs.extend(group)
                vals = [r["speedup_pct"] for r in group]
                pair_summary.append(dict(configuration=config,n=n,baseline=baseline,candidate=candidate,
                                         median_speedup_pct=st.median(vals),min_speedup_pct=min(vals),
                                         max_speedup_pct=max(vals),positive_rounds=sum(v>0 for v in vals),
                                         rounds=len(vals),adjacent_rounds=sum(r["adjacent"] for r in group)))
    # Samples belong to host warmup+events+validation windows. No interpolation.
    tz = timezone(timedelta(minutes=metadata["host_utc_offset_minutes"]))
    telemetry = []
    for raw in read(root / "telemetry.csv"):
        r = {k.strip():v.strip() for k,v in raw.items() if k is not None and v is not None}
        if not r.get("timestamp") or r["timestamp"] == "timestamp":
            continue
        r["unix_ms"] = datetime.strptime(r["timestamp"], "%Y/%m/%d %H:%M:%S.%f").replace(tzinfo=tz).timestamp()*1000
        telemetry.append(r)
    rt = []
    fields = {"temperature_c":"temperature.gpu", "sm_clock_mhz":"clocks.current.sm [MHz]",
              "memory_clock_mhz":"clocks.current.memory [MHz]", "power_w":"power.draw [W]"}
    for r in allrounds:
        points = [t for t in telemetry if t.get("index") == "0" and
                  r["round_start_unix_ms"] <= t["unix_ms"] <= r["round_end_unix_ms"]]
        item = {k:r[k] for k in ("configuration","n","implementation","round","round_start_unix_ms","round_end_unix_ms")}
        item["telemetry_points"] = len(points)
        for name,key in fields.items():
            vals = []
            for t in points:
                try: vals.append(float(t.get(key, "").split()[0]))
                except (ValueError,IndexError): pass
            item[name+"_min"] = min(vals) if vals else ""
            item[name+"_max"] = max(vals) if vals else ""
        rt.append(item)
    for path in sorted(root.rglob("*.csv")):
        if out not in path.parents:
            raw_hashes.append(dict(path=str(path.relative_to(root)),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    for name,rows in [("rounds",allrounds),("summary",summaries),("paired-rounds",pairs),
                      ("paired-summary",pair_summary),("correctness-all",correctness),
                      ("round-telemetry",rt),("raw-sha256",raw_hashes)]:
        write(out / (name+".csv"),rows)
    report = dict(status="PASS",raw_events=raw_count,rounds=len(allrounds),correctness_rows=len(correctness),
                  allclose_failed_checks=sum(r["upstream_allclose_status"] != "PASS" for r in correctness),
                  absolute_or_nonfinite_failed_checks=0,
                  max_abs_error=max(float(r["max_abs_error"]) for r in correctness),
                  max_rel_error=max(float(r["max_rel_error"]) for r in correctness),
                  max_cpu_implementation_abs_error=max(float(r["cpu_implementation_max_abs_error"]) for r in correctness),
                  input_fingerprints={n:list(next(iter(v))) for n,v in hashes.items()},
                  all_inputs_identical_across_configs=True,negative_shapes_rejected=12,
                  telemetry_raw_points=len(telemetry),round_windows_with_telemetry=sum(r["telemetry_points"]>0 for r in rt),
                  slow_samples_removed=0,scope="Audit success; allclose decisions remain independent. Seven rounds from one WDDM session are not independent sessions.")
    (out / "audit.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,indent=2))


if __name__ == "__main__":
    main()
