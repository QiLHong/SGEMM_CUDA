"""Summarize the prespecified BN64 robustness experiment without removing events.

Every performance estimate is a median of seven round means (30 events each).
The independent verifier deliberately does not import this module.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics as stat

NAMES = ("vitamin_tf32_dbf", "tf32_bn64_bk32", "cublas_tf32")
SIZES = (1024, 2048, 3072, 4096)
SEEDS = (20261020, 42, 12345)
REFERENCES = ("original_fp32", "represented_input_strict_reference",
              "same_precision_cublas", "represented_input_cpu_double", "vitamin_tf32_baseline")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def save(path, rows):
    if rows:
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def identity(row):
    return tuple(int(row[k]) for k in ("batch", "seed", "n"))


def round_key(row):
    return (*identity(row), row["implementation"], int(row["round"]))


def plan_key(row):
    return (*round_key(row), int(row["order"]), int(row["case_index"]))


def meta_read(path):
    return dict(line.split("=", 1) for line in path.read_text(encoding="utf-8-sig").splitlines() if "=" in line)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run, out = args.run.resolve(), args.output.resolve()
    require(not out.exists(), f"Refusing to overwrite {out}")
    require((run / "RUN_COMPLETE.txt").exists(), "Full run is not marked complete; preserve partial data")
    plan = read(run / "plan.csv")
    require(len(plan) == 756, "Expected 3 * 12 * 7 * 3 plan rows")
    require(len({plan_key(x) for x in plan}) == len(plan), "Duplicate plan rows")
    expected_plan = {plan_key(x) for x in plan}
    by_case_plan = defaultdict(list)
    for row in plan:
        by_case_plan[identity(row)].append(row)
    expected_cases = {(b, seed, n) for b in (1, 2, 3) for seed in SEEDS for n in SIZES}
    require(set(by_case_plan) == expected_cases, "Plan size/seed/batch set differs")
    seventh = defaultdict(list)
    case_orders = {}
    for case, rows in by_case_plan.items():
        perms = []
        for r in range(7):
            group = sorted((x for x in rows if int(x["round"]) == r), key=lambda x: int(x["order"]))
            require([int(x["order"]) for x in group] == [0, 1, 2], f"Bad plan order {case} {r}")
            require(set(x["implementation"] for x in group) == set(NAMES), "Bad implementation set")
            perms.append(tuple(x["implementation"] for x in group))
        require(len(set(perms[:6])) == 6, f"First six permutations are not complete: {case}")
        seventh[case[1:]].append(perms[6])
        indices = {int(x["case_index"]) for x in rows}
        require(len(indices) == 1, "Case has conflicting indices")
        case_orders[case] = indices.pop()
    require(all(len(set(p)) == 3 for p in seventh.values()), "Seventh permutation does not rotate over batches")
    order_lists = [tuple(sorted((idx, seed, n) for (batch, seed, n), idx in case_orders.items() if batch == b)) for b in (1, 2, 3)]
    require(len(set(order_lists)) == 3, "Case order was not rotated across batches")
    require(all(sorted(x[0] for x in rows) == list(range(12)) for rows in order_lists), "Bad case indices")

    events, checks, bitwise, inputs, windows, source_files, metas, processes = [], [], [], [], [], [], {}, []
    for batch in (1, 2, 3):
        batch_dir = run / f"batch-{batch}"
        require((batch_dir / "BATCH_COMPLETE.txt").exists(), f"Incomplete batch {batch}")
        process_rows = read(batch_dir / "process.csv")
        require(len(process_rows) == 1, f"Expected one process record for batch {batch}")
        process = process_rows[0]
        require(int(process["batch"]) == batch and int(process["cases"]) == 12 and
                int(process["formal_events"]) == 7560 and process["status"] == "COMPLETE", "Invalid process completion record")
        processes.append(process)
        source_files.append(batch_dir / "process.csv")
        paths = sorted(batch_dir.glob("timings-*.csv"))
        paths = [p for p in paths if p.name.count(".csv") == 1 and p.stem.count("-") == 2]
        require(len(paths) == 12, f"Expected 12 event files in {batch_dir}")
        for path in paths:
            rows = read(path)
            require(len(rows) == 630, f"Expected 630 events: {path}")
            require(len({identity(x) for x in rows}) == 1, f"Mixed cases: {path}")
            case = identity(rows[0])
            require(case[0] == batch and case in expected_cases, f"Unexpected case {path}")
            require(case not in metas, f"Duplicate case {case}")
            events += rows
            for suffix, collection in ((".checks.csv", checks), (".bitwise.csv", bitwise),
                                       (".inputs.csv", inputs), (".windows.csv", windows)):
                p = Path(str(path) + suffix)
                data = read(p)
                for row in data:
                    row["n"] = str(case[2]) if "n" not in row else row["n"]
                    require(identity(row) == case, f"Identity mismatch {p}")
                collection += data
                source_files.append(p)
            mp = Path(str(path) + ".meta.txt")
            metas[case] = meta_read(mp)
            require(metas[case].get("shared_tf32_input_pointers", "").lower() == "true", "Missing shared-input contract")
            require(metas[case].get("pid") == process["pid"], "Case process differs from batch record")
            source_files += [path, mp, Path(str(path) + ".cpu-indices.csv")]
    require(set(metas) == expected_cases, "Missing formal case")
    require(len({x["pid"] for x in processes}) == 3, "Expected three distinct process IDs")
    require(all(int(x["end_utc_ms"]) >= int(x["start_utc_ms"]) for x in processes), "Reversed process timestamps")
    require(all(int(processes[i]["end_utc_ms"]) <= int(processes[i+1]["start_utc_ms"]) for i in range(2)), "Formal processes overlap")

    groups = defaultdict(list)
    for row in events:
        require(int(row["m"]) == int(row["n"]) == int(row["k"]), "Formal table contains nonsquare shape")
        require(plan_key(row) in expected_plan, "Actual execution order differs from saved plan")
        require(int(row["cuda_launch_status"]) == int(row["cuda_event_sync_status"]) == 0, "CUDA error retained in formal data")
        ms = float(row["milliseconds"])
        require(ms > 0 and math.isfinite(ms), "Invalid CUDA event")
        require(math.isclose(float(row["tflops"]), 2 * int(row["n"]) ** 3 / (ms * 1e9), rel_tol=1e-12), "Raw TFLOPS disagrees with event")
        groups[round_key(row)].append(row)
    require(len(events) == 22680 and len(groups) == 756, "Incorrect event or round count")
    rounds, means = [], {}
    for key, rows in sorted(groups.items()):
        batch, seed, n, name, r = key
        require(sorted(int(x["iteration"]) for x in rows) == list(range(30)), f"Missing or duplicate iteration {key}")
        require(len({(x["order"], x["case_index"]) for x in rows}) == 1, "Mixed order within round")
        ms = [float(x["milliseconds"]) for x in rows]
        means[key] = stat.mean(ms)
        rounds.append(dict(batch=batch, seed=seed, n=n, implementation=name, round=r,
                           case_index=rows[0]["case_index"], order=rows[0]["order"], events=len(ms),
                           mean_ms=means[key], event_min_ms=min(ms), event_max_ms=max(ms),
                           event_median_ms=stat.median(ms), tflops=2*n**3 / (means[key]*1e9)))
    summary, summary_lookup, pairs, paired, cases = [], {}, [], [], []
    for case in sorted(expected_cases):
        batch, seed, n = case
        for name in NAMES:
            ms = [means[(*case, name, r)] for r in range(7)]
            row = dict(batch=batch, seed=seed, n=n, implementation=name,
                       median_round_mean_ms=stat.median(ms), tflops=2*n**3/(stat.median(ms)*1e9),
                       round_min_ms=min(ms), round_max_ms=max(ms), round_mean_ms=stat.mean(ms),
                       round_sd_ms=stat.stdev(ms), round_cv_pct=100*stat.stdev(ms)/stat.mean(ms))
            summary.append(row)
            summary_lookup[(*case, name)] = row
        case_row = dict(batch=batch, seed=seed, n=n, case_index=case_orders[case])
        case_row.update({name+"_tflops": summary_lookup[(*case, name)]["tflops"] for name in NAMES})
        for ref in (NAMES[0], NAMES[2]):
            gains = []
            for r in range(7):
                ref_ms, candidate_ms = means[(*case, ref, r)], means[(*case, NAMES[1], r)]
                gain = 100*(ref_ms/candidate_ms-1)
                gains.append(gain)
                pairs.append(dict(batch=batch, seed=seed, n=n, candidate=NAMES[1], reference=ref, round=r,
                                  reference_ms=ref_ms, candidate_ms=candidate_ms, gain_pct=gain))
            ratio_gain = 100*(summary_lookup[(*case, NAMES[1])]["tflops"]/summary_lookup[(*case, ref)]["tflops"]-1)
            row = dict(batch=batch, seed=seed, n=n, candidate=NAMES[1], reference=ref,
                       median_paired_gain_pct=stat.median(gains), min_gain_pct=min(gains), max_gain_pct=max(gains),
                       positive_rounds=sum(x>0 for x in gains), negative_rounds=sum(x<0 for x in gains),
                       tied_rounds=sum(x==0 for x in gains), rounds=7,
                       ratio_of_median_throughputs_gain_pct=ratio_gain)
            paired.append(row)
            tag = "author" if ref == NAMES[0] else "cublas"
            for field in ("median_paired_gain_pct", "min_gain_pct", "max_gain_pct", "positive_rounds", "negative_rounds", "ratio_of_median_throughputs_gain_pct"):
                case_row[f"vs_{tag}_{field}"] = row[field]
        cases.append(case_row)
    size_summary = []
    for n in SIZES:
        for ref in (NAMES[0], NAMES[2]):
            rows = [x for x in paired if x["n"] == n and x["reference"] == ref]
            medians = [x["median_paired_gain_pct"] for x in rows]
            size_summary.append(dict(n=n, reference=ref, independent_processes=3, seeds_per_process=3, case_combinations=9,
                                     median_of_case_paired_medians_pct=stat.median(medians),
                                     min_case_paired_median_pct=min(medians), max_case_paired_median_pct=max(medians),
                                     positive_case_medians=sum(x>0 for x in medians), negative_case_medians=sum(x<0 for x in medians),
                                     positive_rounds=sum(x["positive_rounds"] for x in rows),
                                     negative_rounds=sum(x["negative_rounds"] for x in rows), total_rounds=63,
                                     scope="descriptive case summary; 63 rounds are nested in 3 processes; no IID-event inference"))

    # Check records are complete; strict-FP32 precision failures are preserved, not vetoed.
    check_keys = Counter((*identity(x), x["implementation"], int(x["round"]), x["reference"]) for x in checks)
    expected_checks = {(*case, name, r, ref) for case in expected_cases for name in NAMES for r in range(-1, 7) for ref in REFERENCES}
    require(set(check_keys) == expected_checks and all(x==1 for x in check_keys.values()), "Missing/duplicate output checks")
    for row in checks:
        require(int(row["count"]) == (256 if row["reference"] == "represented_input_cpu_double" else int(row["n"])**2), "Wrong output-check scope")
        require(float(row["native_eps"]) == .1, "Native allclose threshold changed")
        require(int(row["nonfinite"]) == 0, "Nonfinite output/reference reported")
        if row["reference"] != "original_fp32":
            require(int(row["native_allclose_fail"]) == 0, "Native TF32 gate failed")
    bits = Counter((*identity(x), x["implementation"], int(x["round"])) for x in bitwise)
    expected_bits = {(*case, name, r) for case in expected_cases for name in NAMES[:2] for r in range(-1, 7)}
    require(set(bits) == expected_bits and all(v==1 for v in bits.values()), "Missing/duplicate baseline bitwise checks")
    require(all(int(x["count"]) == int(x["n"])**2 and int(x["bitwise_mismatches"]) == 0 for x in bitwise), "Author bitwise comparison failed")
    error_groups = defaultdict(list)
    for row in checks:
        error_groups[(*identity(row), row["implementation"], row["reference"])].append(row)
    errors = []
    for key, rows in sorted(error_groups.items()):
        row = dict(zip(("batch", "seed", "n", "implementation", "reference"), key))
        row["check_records"] = len(rows)
        row["points_per_check"] = int(rows[0]["count"])
        for field in ("max_abs", "max_rel", "relative_l2", "nonfinite", "abs_001_fail", "simt_allclose_fail", "native_allclose_fail"):
            row["worst_"+field] = max(float(x[field]) for x in rows)
        for field in ("abs_001_fail", "simt_allclose_fail", "native_allclose_fail"):
            row["failed_checks_"+field] = sum(int(x[field])>0 for x in rows)
        errors.append(row)

    fingerprints = defaultdict(dict)
    for row in inputs:
        case = identity(row)
        key = (row["representation"], row["matrix"])
        stage = row["stage"]
        require(stage in ("before", "after") and key[0] in ("original_fp32", "tf32") and key[1] in ("A", "B"), "Unexpected input row")
        require(int(row["element_count"]) == case[2]**2 and int(row["bytes"]) == 4*case[2]**2, "Incorrect input byte count")
        require((key, stage) not in fingerprints[case], "Duplicate input fingerprint")
        fingerprints[case][key, stage] = row["fnv1a64"]
    require(set(fingerprints) == expected_cases, "Missing case fingerprint")
    for case, values in fingerprints.items():
        require(len(values) == 8, "Expected four before and four after fingerprints")
        for key in ((p,m) for p in ("original_fp32", "tf32") for m in ("A", "B")):
            require(values[key,"before"] == values[key,"after"], f"Input changed during case {case}")
            require(len({fingerprints[b,case[1],case[2]][key,"before"] for b in (1,2,3)}) == 1, "Same seed/shape differs across batches")
    for n in SIZES:
        for key in ((p,m) for p in ("original_fp32", "tf32") for m in ("A", "B")):
            require(len({fingerprints[1,seed,n][key,"before"] for seed in SEEDS}) == 3, "Different seeds did not change inputs")
    window_keys = Counter(round_key(x) for x in windows)
    require(set(window_keys) == set(groups) and all(v==1 for v in window_keys.values()), "Missing round time window")
    require(all(int(x["end_utc_ms"]) >= int(x["start_utc_ms"]) for x in windows), "Reversed time window")

    # Telemetry is a sparse observation, never proof of constant event-level clocks.
    telemetry_summary = []
    for path in sorted(run.rglob("telemetry*.csv")):
        raw = [{k.strip():(v or "").strip() for k,v in row.items() if k is not None} for row in read(path)]
        row = dict(path=str(path.relative_to(run)), samples=len(raw),
                   clock_scope="sparse samples; clocks not locked; includes CPU validation gaps, not event-only intervals")
        for term in ("temperature.gpu", "clocks.current.sm", "clocks.current.memory", "clocks.sm", "clocks.mem", "power.draw", "utilization.gpu"):
            matching = [k for k in (raw[0] if raw else []) if k.split(" ")[0] == term]
            if not matching:
                continue
            values = []
            for item in raw:
                try:
                    v = float(item[matching[0]].split()[0])
                    if math.isfinite(v): values.append(v)
                except (ValueError, IndexError):
                    pass
            row[term+"_available_samples"] = len(values)
            row[term+"_missing_samples"] = len(raw)-len(values)
            row[term+"_min"] = min(values) if values else "missing"
            row[term+"_max"] = max(values) if values else "missing"
        telemetry_summary.append(row)

    out.mkdir(parents=True)
    for name, data in (("rounds.csv",rounds),("summary.csv",summary),("paired-rounds.csv",pairs),
                       ("paired.csv",paired),("cases.csv",cases),("size-summary.csv",size_summary),
                       ("errors.csv",errors),("input-fingerprints.csv",inputs),("processes.csv",processes)):
        save(out/name,data)
    if telemetry_summary:
        fields = list(dict.fromkeys(k for row in telemetry_summary for k in row))
        save(out/"telemetry-summary.csv",[{k:row.get(k,"missing") for k in fields} for row in telemetry_summary])
    source_files += list(run.rglob("telemetry*.csv"))
    manifest = [dict(path=str(p.relative_to(run)), bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(set(source_files+[run/"plan.csv"]))]
    (out/"raw-files-read-sha256.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    audit = dict(formal_events=len(events),round_groups=len(groups),case_combinations=len(metas),
                 process_batches=3,check_records=len(checks),bitwise_records=len(bitwise),
                 input_fingerprint_records=len(inputs),round_time_windows=len(windows),removed_events=0,
                 telemetry_files=len(telemetry_summary),telemetry_missing_marker=(run/"TELEMETRY_MISSING.txt").exists(),
                 original_fp32_abs_001_failed_checks=sum(int(x["abs_001_fail"])>0 for x in checks if x["reference"]=="original_fp32"),
                 represented_fp32_abs_001_failed_checks=sum(int(x["abs_001_fail"])>0 for x in checks if x["reference"]=="represented_input_strict_reference"),
                 allclose_expression="abs(got-ref) <= atol + rtol*abs(got)",
                 relative_error_denominator="max(abs(ref), 1e-12)",
                 confidence_intervals="not computed; rounds and cases are nested within three same-session processes",
                 correctness_scope="full GPU outputs, 256 CPU-double positions per large square; bitwise equality does not prove race freedom",
                 timing_scope="one GEMM per event; conversion and same-stream C0 restore excluded; restore can affect cache")
    (out/"audit.json").write_text(json.dumps(audit,indent=2)+"\n",encoding="utf-8")
    lines = ["# BN64 稳健性原始数据统计", "", "本目录只汇总本轮正式 event；不混入冒烟、Sanitizer、profiler 或历史成绩。", "",
             "- 36 个 size × seed × batch 组合；三个独立进程均在本次会话内运行。",
             "- 每实现七轮，每轮 30 次 event 的全部数据求均值，再取七轮均值的中位数。",
             "- 配对提升是每轮 `reference_ms / BN64_ms - 1`，与中位数吞吐之比单独列出。",
             "- 所有慢样本保留；没有计算置信区间，没有把 22,680 events 当作独立实验样本。",
             "- `cases.csv` 列出全部 36 组；`summary.csv` 为 108 个实现结果；`paired-rounds.csv` 保留 504 个配对轮。",
             "- `errors.csv` 保留严格精度失败。GPU 参考检查全量输出，CPU double 对每个大尺寸固定抽样 256 点。",
             "- `telemetry-summary.csv` 是整批稀疏采样范围，包含 CPU 正确性检查间隙，不能证明逐 event 频率恒定。", "",
             "| N | 比较参考 | 九个组合的配对提升中位数范围 | 正收益组合 | 正收益轮 |", "|---:|---|---:|---:|---:|"]
    for row in size_summary:
        lines.append(f'| {row["n"]} | {row["reference"]} | {row["min_case_paired_median_pct"]:+.3f}% .. {row["max_case_paired_median_pct"]:+.3f}% | {row["positive_case_medians"]}/9 | {row["positive_rounds"]}/63 |')
    lines += ["", "跨组合表是描述性汇总；三个 seed 和七轮嵌套在同一批进程中，不能据此宣称跨天、重启或其它机器稳定。", ""]
    (out/"README.md").write_text("\n".join(lines),encoding="utf-8")
    print(json.dumps(audit,ensure_ascii=False))
    for row in size_summary:
        print(f'N={row["n"]} versus {row["reference"]}: case paired medians {row["min_case_paired_median_pct"]:+.3f}..{row["max_case_paired_median_pct"]:+.3f}%, {row["positive_case_medians"]}/9 positive cases, {row["positive_rounds"]}/63 positive rounds')


if __name__ == "__main__":
    main()
