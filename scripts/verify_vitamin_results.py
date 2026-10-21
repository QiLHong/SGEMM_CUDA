"""Independent CPU-only check of raw vitamin events and saved summary products.

Uses only Python's standard library; does not import the production summarizer,
start CUDA, change existing results, or discard slow samples. A new JSON output
path is required. The check verifies recorded observations; it cannot re-execute
the matrix computation from CSV files, which contain errors rather than C.
"""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys


MODES = (
    "sgemm_tiling", "sgemm_at_tiling", "sgemm_at_bcf_swizzling",
    "sgemm_at_bcf_swizzling_rw", "sgemm_at_bcf_swizzling_dbf_rw",
    "k10", "k30_v2", "cublas",
)
CONFIGS = {"A": "original_config", "B": "strict_fp32"}
TIMED_SIZES = (1024, 2048, 4096)
SMALL_SIZES = (128, 256, 512)
FAIL_COLUMNS = (
    "failure_count", "nonfinite_count", "reference_nonfinite_count",
    "cpu_reference_failure_count", "cpu_reference_nonfinite_count",
    "cpu_implementation_failure_count", "cpu_implementation_nonfinite_count",
)
ERROR_COLUMNS = (
    "max_abs_error", "max_rel_error", "cpu_reference_max_abs_error",
    "cpu_reference_max_rel_error", "cpu_implementation_max_abs_error",
    "cpu_implementation_max_rel_error",
)


class Audit:
    def __init__(self):
        self.issues = []
        self.files = {}
        self.checks = 0

    def require(self, ok, description):
        self.checks += 1
        if not ok:
            self.issues.append(description)

    def equal_float(self, actual, expected, description):
        a, b = float(actual), float(expected)
        self.require(math.isfinite(a) and math.isfinite(b) and
                     math.isclose(a, b, rel_tol=2e-12, abs_tol=1e-12),
                     f"{description}: saved={a}, independently_computed={b}")

    def hash_file(self, path):
        path = path.resolve()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        self.files[str(path)] = digest
        return digest

    def csv(self, path):
        self.hash_file(path)
        with path.open(encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))

    def json(self, path):
        self.hash_file(path)
        return json.loads(path.read_text(encoding="utf-8-sig"))


def expected_order(round_id):
    pair = [MODES[3], MODES[4]] if round_id % 2 else [MODES[4], MODES[3]]
    groups = [[MODES[0]], [MODES[1]], [MODES[2]], pair, [MODES[5]],
              [MODES[6]], [MODES[7]]]
    shift = (round_id - 1) % 7
    return [mode for group in groups[shift:] + groups[:shift] for mode in group]


def stats(values):
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values), "max": max(values),
        "cv_pct": 100 * statistics.pstdev(values) / statistics.fmean(values),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--history-manifest", type=Path)
    args = parser.parse_args()
    root, output = args.result_dir.resolve(), args.output_path.resolve()
    if output.exists():
        raise SystemExit(f"Refusing to overwrite {output}")
    audit = Audit()
    meta = audit.json(root / "environment.json")
    rounds, repeats = int(meta["rounds"]), int(meta["repeats"])
    audit.require(rounds >= 7 and repeats >= 30 and int(meta["warmup"]) >= 10,
                  "measurement counts meet protocol")
    audit.require(float(meta["alpha"]) == 1 and float(meta["beta"]) == 0 and
                  int(meta["seed"]) == 20261020, "metadata C=AB / fixed seed")
    audit.require(meta["NVIDIA_TF32_OVERRIDE"] == "0", "TF32 process override disabled")
    audit.require((root / "RUN_COMPLETE.txt").is_file(), "run completion marker")
    hashes = defaultdict(set)
    all_checks, main_rounds, small_rounds = [], {}, {}
    all_events, small_events = [], []
    rejects_count = 0

    for config, label in CONFIGS.items():
        manifest = audit.json(root / config / "verified-build.json")
        command = manifest["compile_command"]
        audit.require(manifest["configuration"] == config and
                      manifest["architecture"] == "sm_89", f"{config} build configuration")
        audit.require("arch=compute_89,code=[compute_89,sm_89]" in command,
                      f"{config} actual architecture flags")
        audit.require(("--use_fast_math" in command) == (config == "A") and
                      manifest["use_fast_math"] == (config == "A"),
                      f"{config} fast-math flag")
        audit.require(f"VITAMIN_ORIGINAL_CONFIG={int(config == 'A')}" in command,
                      f"{config} original configuration macro")
        rejected = audit.csv(root / config / "rejected-shapes.csv")
        audit.require(len(rejected) == 6 and
                      {int(r["n"]) for r in rejected} == {0, 16, 127, 129, 255, 4100},
                      f"{config} rejected sizes")
        for row in rejected:
            n = int(row["n"])
            audit.require(int(row["exit_code"]) != 0 and
                          row["created_output"].lower() == "false" and
                          not (root / config / f"rejected-{n}.csv").exists() and
                          not (root / config / f"rejected-{n}.csv.correctness.csv").exists(),
                          f"{config} reject before creating outputs: {n}")
        rejects_count += len(rejected)

        for n in SMALL_SIZES + TIMED_SIZES:
            timed = n in TIMED_SIZES
            nr, ne = (rounds, repeats) if timed else (1, 1)
            stem = ("timings" if timed else "correctness") + f"-{n}.csv"
            path = root / config / stem
            events = audit.csv(path)
            checks = audit.csv(path.with_name(path.name + ".correctness.csv"))
            audit.require(len(events) == 8 * nr * ne, f"{config}/{n} event count")
            audit.require(len(checks) == 9 + 8 * nr, f"{config}/{n} check count")
            check_index = {(r["phase"], r["implementation"], int(r["round"])): r
                           for r in checks}
            expected_keys = {("reference", "independent_strict_reference", 0)}
            expected_keys.update(("preflight", impl, 0) for impl in MODES)
            expected_keys.update(("timed_round", impl, rd)
                                 for impl in MODES for rd in range(1, nr + 1))
            audit.require(len(check_index) == len(checks) and
                          set(check_index) == expected_keys, f"{config}/{n} unique check keys")
            for row in checks:
                key = f"{config}/{n}/{row['phase']}/{row['implementation']}/{row['round']}"
                all_checks.append(dict(file=str(path.relative_to(root)), **row))
                audit.require(row["configuration"] == label and int(row["n"]) == n and
                              float(row["alpha"]) == 1 and float(row["beta"]) == 0 and
                              int(row["seed"]) == 20261020, f"{key} semantics")
                audit.require(int(row["full_element_count"]) == n * n and
                              int(row["cpu_reference_count"]) == (n * n if n == 128 else 256),
                              f"{key} coverage")
                audit.require(row["status"] == "PASS", f"{key} absolute/CPU status")
                for field in FAIL_COLUMNS:
                    audit.require(int(row[field]) == 0, f"{key} {field}")
                for field in ERROR_COLUMNS:
                    value = float(row[field])
                    audit.require(math.isfinite(value) and value >= 0 and
                                  ("abs" not in field or value <= 0.01),
                                  f"{key} {field} finite and unrelaxed")
                audit.require((row["upstream_allclose_status"] == "PASS") ==
                              (int(row["upstream_allclose_failure_count"]) == 0),
                              f"{key} independent allclose status")
                hashes[n].add(tuple(row[k] for k in ("a_hash", "b_hash", "c0_hash")))
            grouped = defaultdict(list)
            for row in events:
                key = f"{config}/{n}/{row['implementation']}/{row['round']}/{row['iteration']}"
                audit.require(row["configuration"] == label and int(row["n"]) == n and
                              float(row["alpha"]) == 1 and float(row["beta"]) == 0 and
                              int(row["seed"]) == 20261020, f"{key} event semantics")
                audit.require(row["implementation"] in MODES, f"{key} implementation")
                ms = float(row["event_ms"])
                audit.require(math.isfinite(ms) and ms > 0, f"{key} event duration")
                audit.equal_float(row["gflops"], 2 * n**3 / (ms * 1e6), f"{key} GFLOPS")
                hashes[n].add(tuple(row[k] for k in ("a_hash", "b_hash", "c0_hash")))
                grouped[(row["implementation"], int(row["round"]))].append(row)
                (all_events if timed else small_events).append(
                    dict(configuration=config, n=n, implementation=row["implementation"],
                         round=int(row["round"]), iteration=int(row["iteration"]), event_ms=ms))
            audit.require(set(grouped) == {(impl, rd) for impl in MODES
                                          for rd in range(1, nr + 1)},
                          f"{config}/{n} complete round groups")
            for (impl, rd), rows in grouped.items():
                key = (config, n, impl, rd)
                audit.require(len(rows) == ne and
                              [int(r["iteration"]) for r in rows] == list(range(1, ne + 1)),
                              f"{key} every event in original order")
                first = rows[0]
                fixed = set(first) - {"iteration", "event_ms", "gflops"}
                audit.require(all(all(r[f] == first[f] for f in fixed) for r in rows),
                              f"{key} invariant round metadata")
                values = [float(r["event_ms"]) for r in rows]
                calculated = stats(values)
                audit.equal_float(first["round_mean_ms"], calculated["mean"], f"{key} mean")
                audit.require(int(first["order"]) == expected_order(rd).index(impl) + 1,
                              f"{key} exact rotation and alternating pair")
                start, end = int(first["round_start_unix_ms"]), int(first["round_end_unix_ms"])
                audit.require(start <= end, f"{key} host window order")
                check = check_index[("timed_round", impl, rd)]
                mapping = {
                    "round_max_abs_error": "max_abs_error",
                    "round_max_rel_error": "max_rel_error",
                    "round_failure_count": "failure_count",
                    "round_nonfinite_count": "nonfinite_count",
                }
                for field in set(first) & set(check):
                    audit.require(first[field] == check[field], f"{key} sidecar {field}")
                for field, check_field in mapping.items():
                    audit.require(first[field] == check[check_field], f"{key} sidecar {field}")
                item = dict(configuration=config, n=n, implementation=impl, round=rd,
                            order=int(first["order"]), event_count=len(values),
                            mean_ms=calculated["mean"], values=values,
                            start=start, end=end,
                            allclose=first["upstream_allclose_status"])
                (main_rounds if timed else small_rounds)[key] = item

    audit.require(all(len(values) == 1 for values in hashes.values()),
                  "same input fingerprints in every implementation and A/B configuration")
    saved_rounds = audit.csv(root / "analysis" / "rounds.csv")
    audit.require(len(saved_rounds) == len(main_rounds), "saved rounds no deletion")
    for row in saved_rounds:
        key = (row["configuration"], int(row["n"]), row["implementation"], int(row["round"]))
        raw = main_rounds[key]
        calculated = stats(raw["values"])
        for field, value in {
            "mean_ms": raw["mean_ms"], "event_count": raw["event_count"],
            "order": raw["order"], "event_min_ms": calculated["min"],
            "event_max_ms": calculated["max"], "event_median_ms": calculated["median"],
            "event_cv_pct": calculated["cv_pct"],
            "tflops": 2 * key[1]**3 / (raw["mean_ms"] * 1e9),
        }.items():
            audit.equal_float(row[field], value, f"saved round {key} {field}")

    summary = audit.csv(root / "analysis" / "summary.csv")
    recomputed_summary = []
    audit.require(len(summary) == 2 * 3 * 8, "48 saved summaries")
    for row in summary:
        config, n, impl = row["configuration"], int(row["n"]), row["implementation"]
        items = [main_rounds[(config, n, impl, rd)] for rd in range(1, rounds + 1)]
        means = [r["mean_ms"] for r in items]
        events = [v for r in items for v in r["values"]]
        med = statistics.median(means)
        expected = dict(rounds=rounds, events=rounds * repeats,
                        median_round_mean_ms=med, median_tflops=2 * n**3 / (med * 1e9),
                        min_round_mean_ms=min(means), max_round_mean_ms=max(means),
                        round_range_over_median_pct=100 * (max(means) - min(means)) / med,
                        round_cv_pct=100 * statistics.pstdev(means) / statistics.fmean(means),
                        slowest_event_ms=max(events),
                        allclose_failed_rounds=sum(r["allclose"] != "PASS" for r in items))
        for field, value in expected.items():
            audit.equal_float(row[field], value, f"summary {config}/{n}/{impl} {field}")
        event_median = statistics.median(events)
        recomputed_summary.append(dict(
            configuration=config, n=n, implementation=impl, **expected,
            raw_event_median_ms=event_median, max_event_over_median=max(events)/event_median,
            events_gt_2x_impl_median=sum(v > 2*event_median for v in events),
            events_gt_5x_impl_median=sum(v > 5*event_median for v in events),
            events_gt_10x_impl_median=sum(v > 10*event_median for v in events)))

    pair_specs = list(zip(MODES[:4], MODES[1:5]))
    pair_specs += [("k30_v2", MODES[4])] + [("cublas", mode) for mode in MODES[:-1]]
    paired = audit.csv(root / "analysis" / "paired-rounds.csv")
    expected_pair_keys = {(c, n, b, m, rd) for c in CONFIGS for n in TIMED_SIZES
                          for b, m in pair_specs for rd in range(1, rounds + 1)}
    actual_pair_keys = {(r["configuration"], int(r["n"]), r["baseline"],
                         r["candidate"], int(r["round"])) for r in paired}
    audit.require(len(paired) == len(expected_pair_keys) and
                  actual_pair_keys == expected_pair_keys, "complete unique paired rounds")
    pair_groups = defaultdict(list)
    for row in paired:
        config, n, rd = row["configuration"], int(row["n"]), int(row["round"])
        baseline, candidate = row["baseline"], row["candidate"]
        b, c = main_rounds[(config, n, baseline, rd)], main_rounds[(config, n, candidate, rd)]
        speedup = 100 * (b["mean_ms"]/c["mean_ms"] - 1)
        for field, value in {
            "baseline_mean_ms": b["mean_ms"], "candidate_mean_ms": c["mean_ms"],
            "speedup_pct": speedup,
        }.items():
            audit.equal_float(row[field], value, f"pair {config}/{n}/{baseline}/{candidate}/{rd} {field}")
        adjacent, first = abs(b["order"] - c["order"]) == 1, c["order"] < b["order"]
        audit.require(row["adjacent"].lower() == str(adjacent).lower() and
                      row["candidate_first"].lower() == str(first).lower(),
                      f"pair {config}/{n}/{baseline}/{candidate}/{rd} order metadata")
        pair_groups[(config, n, baseline, candidate)].append((speedup, adjacent))
    pair_summary = audit.csv(root / "analysis" / "paired-summary.csv")
    audit.require(len(pair_summary) == len(pair_groups) == 72, "72 paired summaries")
    recomputed_pairs = []
    for row in pair_summary:
        key = (row["configuration"], int(row["n"]), row["baseline"], row["candidate"])
        values = [v for v, _ in pair_groups[key]]
        expected = dict(median_speedup_pct=statistics.median(values),
                        min_speedup_pct=min(values), max_speedup_pct=max(values),
                        positive_rounds=sum(v > 0 for v in values), rounds=len(values),
                        adjacent_rounds=sum(a for _, a in pair_groups[key]))
        for field, value in expected.items():
            audit.equal_float(row[field], value, f"pair summary {key} {field}")
        recomputed_pairs.append(dict(configuration=key[0], n=key[1],
                                     baseline=key[2], candidate=key[3], **expected))

    saved_checks = audit.csv(root / "analysis" / "correctness-all.csv")
    audit.require(len(saved_checks) == len(all_checks), "combined correctness count")
    raw_check_set = Counter(tuple(sorted((k, v) for k, v in r.items() if k != "file"))
                            for r in all_checks)
    saved_check_set = Counter(tuple(sorted((k, v) for k, v in r.items() if k != "source"))
                              for r in saved_checks)
    audit.require(raw_check_set == saved_check_set, "all saved correctness rows preserved")
    raw_manifest = audit.csv(root / "analysis" / "raw-sha256.csv")
    for row in raw_manifest:
        audit.require(audit.hash_file(root / row["path"]) == row["sha256"].lower(),
                      f"raw archive digest {row['path']}")

    zone = timezone(timedelta(minutes=float(meta["host_utc_offset_minutes"])))
    telemetry = []
    for raw in audit.csv(root / "telemetry.csv"):
        row = {k.strip(): v.strip() for k, v in raw.items() if k and v is not None}
        if not row.get("timestamp") or row["timestamp"] == "timestamp":
            continue
        row["unix_ms"] = datetime.strptime(row["timestamp"], "%Y/%m/%d %H:%M:%S.%f").replace(
            tzinfo=zone).timestamp() * 1000
        telemetry.append(row)
    saved_telemetry = audit.csv(root / "analysis" / "round-telemetry.csv")
    audit.require(len(saved_telemetry) == len(main_rounds), "telemetry rows preserve empty windows")
    coverage, covered, empty = Counter(), 0, 0
    for row in saved_telemetry:
        key = (row["configuration"], int(row["n"]), row["implementation"], int(row["round"]))
        window = main_rounds[key]
        points = [p for p in telemetry if p.get("index") == "0" and
                  window["start"] <= p["unix_ms"] <= window["end"]]
        audit.require(int(row["telemetry_points"]) == len(points), f"telemetry count {key}")
        covered += bool(points)
        empty += not points
        coverage[(key[0], key[1], "covered" if points else "empty")] += 1
        for field, raw_field in {
            "temperature_c": "temperature.gpu", "sm_clock_mhz": "clocks.current.sm [MHz]",
            "memory_clock_mhz": "clocks.current.memory [MHz]", "power_w": "power.draw [W]",
        }.items():
            values = []
            for point in points:
                try:
                    values.append(float(point.get(raw_field, "").split()[0]))
                except (ValueError, IndexError):
                    continue
            if values:
                audit.equal_float(row[field+"_min"], min(values), f"telemetry {key} {field} min")
                audit.equal_float(row[field+"_max"], max(values), f"telemetry {key} {field} max")
            else:
                audit.require(row[field+"_min"] == row[field+"_max"] == "",
                              f"telemetry {key} no invented {field}")

    history = {"checked": 0, "unchanged": 0}
    if args.history_manifest:
        for row in audit.csv(args.history_manifest.resolve()):
            path = Path(row["Path"])
            history["checked"] += 1
            good = path.is_file() and audit.hash_file(path) == row["Hash"].lower()
            history["unchanged"] += good
            audit.require(good, f"prior history unchanged: {path}")

    repo = Path(__file__).resolve().parents[1]
    source = repo / "scripts" / "windows_vitamin_benchmark.cu"
    audit.hash_file(source)
    audit.hash_file(repo / "src" / "kernels" / "vendor" / "vitamin_sgemm.cuh")
    audit.hash_file(Path(__file__))
    source_text = source.read_text(encoding="utf-8-sig")
    # These guard the manually reviewed protocol; they are not a C++ proof/parser.
    source_guards = {
        "alpha1_beta0": "constexpr float Alpha = 1.0f, Beta = 0.0f;" in source_text,
        "independent_reference_handle": "referenceBlas(true)" in source_text,
        "reset_before_start_event": "reset();\n          CUDA_OK(cudaEventRecord(events.start" in source_text,
        "stop_after_launch": "run(impl);\n          CUDA_OK(cudaEventRecord(events.stop" in source_text,
        "validation_after_events": "timings.push_back(ms);\n        }\n        const Validation result = validate();" in source_text,
        "per_implementation_cpu_check": "result.cpu = compareCpu(actual,indices,cpuExpected);" in source_text,
        "upstream_candidate_relative_term": "UpstreamAtol+UpstreamRtol*std::abs(candidate)" in source_text,
    }
    for key, good in source_guards.items():
        audit.require(good, f"source protocol guard: {key}")
    phase_counts = dict(Counter(row["phase"] for row in all_checks))
    slowest = sorted(all_events, key=lambda r: r["event_ms"], reverse=True)[:20]
    saved_audit = audit.json(root / "analysis" / "audit.json")
    for field, value in {
        "raw_events": len(all_events), "rounds": len(main_rounds),
        "correctness_rows": len(all_checks),
        "allclose_failed_checks": sum(int(r["upstream_allclose_failure_count"]) > 0 for r in all_checks),
        "max_abs_error": max(float(r["max_abs_error"]) for r in all_checks),
        "max_rel_error": max(float(r["max_rel_error"]) for r in all_checks),
        "max_cpu_implementation_abs_error": max(float(r["cpu_implementation_max_abs_error"]) for r in all_checks),
        "negative_shapes_rejected": rejects_count, "telemetry_raw_points": len(telemetry),
        "round_windows_with_telemetry": covered, "slow_samples_removed": 0,
    }.items():
        audit.equal_float(saved_audit[field], value, f"saved audit {field}")
    report = {
        "status": "PASS" if not audit.issues else "FAIL",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "result_dir": str(root),
        "checks_evaluated": audit.checks, "issues": audit.issues,
        "main_event_count": len(all_events), "small_validation_event_count": len(small_events),
        "main_round_count": len(main_rounds), "small_validation_round_count": len(small_rounds),
        "correctness_rows": len(all_checks), "correctness_phase_counts": phase_counts,
        "negative_shapes_rejected": rejects_count,
        "max_abs_error": max(float(r["max_abs_error"]) for r in all_checks),
        "max_rel_error": max(float(r["max_rel_error"]) for r in all_checks),
        "max_cpu_implementation_abs_error": max(float(r["cpu_implementation_max_abs_error"]) for r in all_checks),
        "allclose_failed_checks": sum(int(r["upstream_allclose_failure_count"]) > 0 for r in all_checks),
        "input_fingerprints": {str(n): [list(v) for v in sorted(values)] for n, values in sorted(hashes.items())},
        "input_fingerprints_identical": all(len(v) == 1 for v in hashes.values()),
        "rw_dbf_adjacent_alternating_pairs": 2 * 3 * rounds,
        "slow_samples_removed": 0, "slowest_events": slowest,
        "independently_recomputed_summary": recomputed_summary,
        "independently_recomputed_paired_summary": recomputed_pairs,
        "telemetry": {"raw_points": len(telemetry), "covered_windows": covered,
                      "empty_windows": empty,
                      "coverage_by_config_n": [
                          dict(configuration=c, n=n, covered=coverage[(c, n, "covered")],
                               empty=coverage[(c, n, "empty")])
                          for c in CONFIGS for n in TIMED_SIZES]},
        "historical_files": history, "source_protocol_guards": source_guards,
        "scope": [
            "CPU-only independent recalculation from unfiltered CSV; no CUDA launch.",
            "Full stored errors vs independent strict cuBLAS; CPU double checks are full at128 and256 fixed points otherwise.",
            "Every round validates only its final output; CSV cannot prove every timed invocation's output.",
            "Upstream allclose mathematical rule uses candidate for relative term and independent pedantic reference, not PyTorch.",
            "A/B use separate processes; one WDDM session with unlocked clocks is not a repeated-session causal comparison.",
            "Paired round ratios are descriptive; only rw/dbf is always adjacent and alternates order.",
            "Slow events are retained. Periodic telemetry brackets warmup/events/validation and is not kernel-level telemetry.",
            "Source guards accompany a manual source review, not formal proof or binary/source equivalence verification.",
        ],
        "input_sha256": audit.files,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({k: report[k] for k in (
        "status", "checks_evaluated", "main_event_count", "small_validation_event_count",
        "main_round_count", "correctness_rows", "max_abs_error",
        "allclose_failed_checks", "historical_files", "issues")}, indent=2))
    print(f"Saved {output}")
    return int(bool(audit.issues))


if __name__ == "__main__":
    sys.exit(main())
