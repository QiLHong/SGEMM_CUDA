"""Audit and summarize the frozen Windows RTX 4090 D CUTLASS comparison.

Usage: python scripts/summarize_cutlass_4090d.py --run <run> --output <new-dir>
Only formal events listed in plan-tf32.csv / plan-fp32.csv enter performance
tables. Each estimate uses every event: mean of 30 events, then median of seven
round means. TF32 and strict FP32 remain separate throughout.
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


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def save(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)


def ck(row):
    return (row["group"], *(int(row[k]) for k in ("batch", "seed", "n")))


def rk(row):
    return (*ck(row), row["implementation"], int(row["round"]))


def meta_read(path):
    return dict(line.split("=", 1) for line in path.read_text(encoding="utf-8-sig").splitlines() if "=" in line)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    run, out = args.run.resolve(), args.output.resolve()
    require(not out.exists(), f"Refusing overwrite: {out}")
    require((run / "RUN_COMPLETE.txt").exists(), "Formal run is incomplete")
    environment = json.loads((run / "environment.json").read_text(encoding="utf-8-sig"))
    groups = environment["groups"]
    require(set(groups) == {"tf32", "fp32"}, "Expected two separate precision groups")
    batches = range(1, int(environment["batches"]) + 1)
    seeds, sizes = environment["seeds"], environment["sizes"]
    rounds, repeats, warmup = (int(environment[k]) for k in ("rounds", "repeats", "warmup"))
    require((rounds, repeats, warmup) == (7, 30, 10), "Prespecified timing counts changed")
    require(environment["alpha"] == 1 and environment["beta"] == 0, "C=AB contract changed")
    expected_cases = {(g, b, s, n) for g in groups for b in batches for s in seeds for n in sizes}
    plan, source_files = [], [run / "environment.json", run / "RUN_COMPLETE.txt"]
    for group in groups:
        path = run / f"plan-{group}.csv"
        require(hashlib.sha256(path.read_bytes()).hexdigest().lower() == environment["plans"][group]["sha256"].lower(), "Frozen plan hash changed")
        rows = read(path)
        for r in rows:
            r["group"] = group
            require(r["implementation"] in groups[group], "Implementation outside precision group")
            require(int(r["m"]) == int(r["n"]) == int(r["k"]), "Formal case is not square")
        plan += rows
        source_files.append(path)
    plan_map = {rk(r): (int(r["case_index"]), int(r["order"])) for r in plan}
    expected_rounds = {(*c, name, r) for c in expected_cases for name in groups[c[0]] for r in range(rounds)}
    require(len(plan) == len(plan_map) and set(plan_map) == expected_rounds, "Plan incomplete/duplicate")
    order_rows = []
    for case in sorted(expected_cases):
        group, batch, seed, n = case
        indices = {plan_map[(*case, name, r)][0] for name in groups[group] for r in range(rounds)}
        require(len(indices) == 1, "Case index changed inside case")
        for r in range(rounds):
            actual = sorted((plan_map[(*case, name, r)][1], name) for name in groups[group])
            require([x[0] for x in actual] == list(range(len(groups[group]))), "Invalid order permutation")
            order_rows.append(dict(group=group, batch=batch, seed=seed, n=n, round=r,
                                   case_index=next(iter(indices)), order=" > ".join(x[1] for x in actual)))
        require(len({x["order"] for x in order_rows if (x["group"],x["batch"],x["seed"],x["n"]) == case}) == rounds,
                "Seven distinct round permutations required")

    events, checks, inputs, windows, bits, processes, metas = [], [], [], [], [], [], {}
    cpu_positions_checked = 0
    for group, names in groups.items():
        for batch in batches:
            folder = run / f"{group}-batch-{batch}"
            require((folder / "BATCH_COMPLETE.txt").exists(), f"Incomplete {folder}")
            ps = read(folder / "process.csv")
            require(len(ps) == 1, "Expected one process record per batch/group")
            require(ps[0]["group"] == group, "Mislabeled process precision group")
            process = dict(ps[0], group=group)
            require(int(process["batch"]) == batch and process["status"] == "COMPLETE", "Invalid process record")
            require(int(process["cases"]) == len(seeds) * len(sizes), "Wrong process case count")
            require(int(process["formal_events"]) == len(seeds) * len(sizes) * len(names) * rounds * repeats, "Wrong process event count")
            processes.append(process)
            source_files += [folder / "process.csv", folder / "BATCH_COMPLETE.txt"]
            for seed in seeds:
                for n in sizes:
                    path = folder / f"timings-{n}-{seed}.csv"
                    rows = read(path)
                    case = (group, batch, seed, n)
                    require(len(rows) == len(names) * rounds * repeats, f"Wrong event count: {path}")
                    previous = (-1, -1, -1)
                    for row in rows:
                        require(row["group"] == group, "Mislabeled event precision group")
                        require(ck(row) == case, "Mislabeled event case")
                        key = rk(row)
                        require(key in plan_map and plan_map[key] == (int(row["case_index"]), int(row["order"])), "Event order differs from frozen plan")
                        position = tuple(int(row[k]) for k in ("round", "order", "iteration"))
                        require(position > previous, "Event file is not in execution order")
                        previous = position
                        require(int(row["m"]) == int(row["k"]) == n, "Nonsquare event")
                        require(row["cuda_launch_status"] == row["cuda_event_sync_status"] == "0", "Recorded CUDA failure")
                        ms = float(row["milliseconds"])
                        require(math.isfinite(ms) and ms > 0, "Invalid CUDA event")
                        require(math.isclose(float(row["tflops"]), 2*n**3/(ms*1e9), rel_tol=1e-12), "Raw TFLOPS inconsistent")
                    events += rows
                    source_files.append(path)
                    for suffix, collection in (("checks", checks), ("inputs", inputs), ("windows", windows)):
                        side = Path(str(path) + f".{suffix}.csv")
                        for row in read(side):
                            require(row["group"] == group, "Mislabeled sidecar precision group")
                            row.setdefault("n", str(n))
                            require(ck(row) == case, "Sidecar case differs")
                            collection.append(row)
                        source_files.append(side)
                    bitfile = Path(str(path) + ".bitwise.csv")
                    if bitfile.exists():
                        for row in read(bitfile):
                            require(row["group"] == group, "Mislabeled bitwise precision group")
                            row["n"] = str(n)
                            require(ck(row) == case, "Bitwise case differs")
                            bits.append(row)
                        source_files.append(bitfile)
                    metafile = Path(str(path) + ".meta.txt")
                    meta = meta_read(metafile)
                    metas[case] = meta
                    require(meta["pid"] == process["pid"] and meta["CASE_COMPLETE"] == "true", "Incomplete case or mismatched PID")
                    require(meta["alpha"] == "1" and meta["beta"] == "0", "Case C=AB differs")
                    require(meta["group"] == group and meta["shared_input_pointers_within_group"] == "true", "Missing shared-input contract")
                    require(tuple(int(meta[x]) for x in ("rounds","repeats","warmup")) == (rounds,repeats,warmup), "Case timing counts changed")
                    source_files.append(metafile)
                    cpufile = Path(str(path) + ".cpu-indices.csv")
                    positions = read(cpufile)
                    require(len(positions) == 256, "Formal CPU check must contain 256 positions")
                    expected = [0, n-1, (n-1)*n, n*n-1]
                    i = 0
                    while len(expected) < 256:
                        index = (20261020 + i*104729) % (n*n)
                        if index not in expected:
                            expected.append(index)
                        i += 1
                    require([int(x["sample_index"]) for x in positions] == list(range(256)), "Bad CPU sample indices")
                    require([int(x["linear_index"]) for x in positions] == expected, "CPU sample rule changed")
                    require(all(int(x["row"])*n + int(x["col"]) == int(x["linear_index"]) for x in positions), "CPU position mismatch")
                    cpu_positions_checked += 1
                    source_files.append(cpufile)
    require(set(metas) == expected_cases, "Missing formal cases")
    require(len(events) == int(environment["expected_formal_events"]) == len(plan)*repeats, "Wrong formal event total")
    require(len({x["pid"] for x in processes}) == len(processes), "Processes are not independent PIDs")
    ordered_processes = sorted(processes, key=lambda x: int(x["start_utc_ms"]))
    require(all(int(x["end_utc_ms"]) >= int(x["start_utc_ms"]) for x in processes), "Reversed process interval")
    require(all(int(a["end_utc_ms"]) <= int(b["start_utc_ms"]) for a,b in zip(ordered_processes, ordered_processes[1:])), "Formal processes overlap")

    event_groups = defaultdict(list)
    for row in events:
        event_groups[rk(row)].append(row)
    require(set(event_groups) == expected_rounds, "Missing/extra event group")
    round_rows, means = [], {}
    for key, rows in sorted(event_groups.items()):
        require([int(x["iteration"]) for x in rows] == list(range(repeats)), "Missing/duplicate event iteration")
        times = [float(x["milliseconds"]) for x in rows]
        means[key] = stat.mean(times)
        group, batch, seed, n, impl, r = key
        round_rows.append(dict(group=group, batch=batch, seed=seed, n=n, implementation=impl, round=r,
                               case_index=rows[0]["case_index"], order=rows[0]["order"], events=len(times),
                               mean_ms=means[key], event_min_ms=min(times), event_max_ms=max(times),
                               event_median_ms=stat.median(times), tflops=2*n**3/(means[key]*1e9)))
    summaries, summary_map, cases = [], {}, []
    pairs, paired = [], []
    for case in sorted(expected_cases):
        group, batch, seed, n = case
        crow = dict(group=group, batch=batch, seed=seed, n=n)
        for impl in groups[group]:
            times = [means[(*case, impl, r)] for r in range(rounds)]
            med = stat.median(times)
            row = dict(group=group, batch=batch, seed=seed, n=n, implementation=impl,
                       median_round_mean_ms=med, tflops=2*n**3/(med*1e9), round_min_ms=min(times),
                       round_max_ms=max(times), round_mean_ms=stat.mean(times), round_sd_ms=stat.stdev(times),
                       round_cv_pct=100*stat.stdev(times)/stat.mean(times))
            summaries.append(row)
            summary_map[(*case, impl)] = row
            crow[impl + "_tflops"] = row["tflops"]
        cases.append(crow)
        # Every ordered pair is retained within the same precision group only.
        for candidate in groups[group]:
            for reference in groups[group]:
                if reference == candidate:
                    continue
                gains = []
                for r in range(rounds):
                    a, b = means[(*case, reference, r)], means[(*case, candidate, r)]
                    gain = 100*(a/b-1)
                    gains.append(gain)
                    pairs.append(dict(group=group, batch=batch, seed=seed, n=n, candidate=candidate,
                                      reference=reference, round=r, reference_ms=a, candidate_ms=b, gain_pct=gain))
                paired.append(dict(group=group, batch=batch, seed=seed, n=n, candidate=candidate, reference=reference,
                                   median_paired_gain_pct=stat.median(gains), min_gain_pct=min(gains), max_gain_pct=max(gains),
                                   positive_rounds=sum(x>0 for x in gains), negative_rounds=sum(x<0 for x in gains),
                                   tied_rounds=sum(x==0 for x in gains), rounds=rounds,
                                   ratio_of_median_throughputs_gain_pct=100*(summary_map[(*case,candidate)]["tflops"]/summary_map[(*case,reference)]["tflops"]-1)))
    size_summary = []
    for group, names in groups.items():
        for n in sizes:
            for candidate in names:
                for reference in names:
                    rows = [x for x in paired if x["group"] == group and x["n"] == n and x["candidate"] == candidate and x["reference"] == reference]
                    if not rows:
                        continue
                    values = [x["median_paired_gain_pct"] for x in rows]
                    size_summary.append(dict(group=group,n=n,candidate=candidate,reference=reference,
                        process_batches=len(batches), seeds_per_process=len(seeds), case_combinations=len(rows),
                        median_of_case_paired_medians_pct=stat.median(values), min_case_paired_median_pct=min(values),
                        max_case_paired_median_pct=max(values), positive_case_medians=sum(x>0 for x in values),
                        negative_case_medians=sum(x<0 for x in values), positive_rounds=sum(x["positive_rounds"] for x in rows),
                        negative_rounds=sum(x["negative_rounds"] for x in rows), total_rounds=len(rows)*rounds))
    throughput_ranges = []
    for group,names in groups.items():
        for n in sizes:
            for name in names:
                rows = [x for x in summaries if x["group"] == group and x["n"] == n and x["implementation"] == name]
                values = [x["tflops"] for x in rows]
                throughput_ranges.append(dict(group=group,n=n,implementation=name,cases=len(rows),
                    median_case_tflops=stat.median(values),min_case_tflops=min(values),max_case_tflops=max(values),
                    min_case_round_cv_pct=min(x["round_cv_pct"] for x in rows),max_case_round_cv_pct=max(x["round_cv_pct"] for x in rows)))

    check_keys = Counter((*rk(x),x["reference"]) for x in checks)
    references = {g: set(x["reference"] for x in checks if x["group"] == g) for g in groups}
    for group, refs in references.items():
        required_refs = {"original_fp32","represented_input_strict_reference","same_precision_cublas","represented_input_cpu_double",
                         "vitamin_tf32_baseline" if group == "tf32" else "k30_fp32_baseline"}
        require(refs == required_refs, "Required error references missing/changed")
    expected_checks = {(*c,name,r,ref) for c in expected_cases for name in groups[c[0]] for r in range(-1,rounds) for ref in references[c[0]]}
    require(set(check_keys) == expected_checks and all(v==1 for v in check_keys.values()), "Missing/duplicate correctness checks")
    for x in checks:
        require(int(x["count"]) == (256 if "cpu_double" in x["reference"] else int(x["n"])**2), "Incorrect full/sample scope")
        require(float(x["native_eps"]) == (.1 if x["group"] == "tf32" else .001), "Native error threshold changed")
        require(int(x["nonfinite"]) == 0, "Nonfinite output/reference")
        if x["group"] == "fp32":
            require(int(x["abs_001_fail"]) == 0, "Strict FP32 absolute-error gate failed")
        elif x["reference"] != "original_fp32":
            require(int(x["native_allclose_fail"]) == 0, "Native TF32 numerical gate failed")
    errors = []
    error_groups = defaultdict(list)
    for x in checks:
        error_groups[(*ck(x),x["implementation"],x["reference"])].append(x)
    for key, rows in sorted(error_groups.items()):
        row = dict(zip(("group","batch","seed","n","implementation","reference"),key))
        row.update(check_records=len(rows),points_per_check=int(rows[0]["count"]))
        for field in ("max_abs","max_rel","relative_l2","nonfinite","abs_001_fail","simt_allclose_fail","native_allclose_fail"):
            row["worst_"+field] = max(float(x[field]) for x in rows)
        for field in ("abs_001_fail","simt_allclose_fail","native_allclose_fail"):
            row["failed_checks_"+field] = sum(int(x[field])>0 for x in rows)
        errors.append(row)

    # Accuracy criteria remain separate judgments. In particular, the original
    # abs<=0.01 acceptance gate does not imply the stricter SIMT allclose passes.
    # For TF32, original FP32 is a precision-loss reference, not a same-input
    # correctness oracle. Its failures remain in the per-reference table.
    def quality_row(group, n, impl, reference, rows):
        row = dict(group=group,n=n,implementation=impl,reference=reference,check_records=len(rows),
                   max_abs=max(float(x["max_abs"]) for x in rows),max_rel=max(float(x["max_rel"]) for x in rows))
        for source,label in (("abs_001_fail","abs_001"),("simt_allclose_fail","simt_allclose"),
                             ("native_allclose_fail","native_allclose"),("nonfinite","nonfinite")):
            failed = sum(int(x[source])>0 for x in rows)
            row[label+"_status"] = "FAIL" if failed else "PASS"
            row[label+"_failed_checks"] = failed
            # Repeated checks inspect the same coordinates again; this sum is
            # observations, not the number of distinct erroneous elements.
            row[label+"_failure_observations"] = sum(int(x[source]) for x in rows)
        return row
    quality, reference_quality = [], []
    for group,names in groups.items():
        for n in sizes:
            for impl in names:
                selected = [x for x in checks if x["group"]==group and int(x["n"])==n and x["implementation"]==impl]
                applicable = [x for x in selected if group=="fp32" or x["reference"]!="original_fp32"]
                quality.append(quality_row(group,n,impl,"applicable full GPU references plus CPU-double samples",applicable))
                for ref in sorted(references[group]):
                    reference_quality.append(quality_row(group,n,impl,ref,[x for x in selected if x["reference"]==ref]))

    bit_keys = Counter(rk(x) for x in bits)
    expected_bits = {(*c,name,r) for c in expected_cases for name in groups[c[0]] for r in range(-1,rounds)}
    require(set(bit_keys) == expected_bits and all(v==1 for v in bit_keys.values()), "Missing/duplicate bitwise records")
    for row in bits:
        required = row["group"] == "tf32" and row["implementation"] in ("vitamin_tf32_dbf","tf32_bn64_bk32")
        require(int(row["required"]) == int(required), "Bitwise gate changed")
        require(int(row["count"]) == int(row["n"])**2, "Bitwise check is not full output")
        if required:
            require(int(row["bitwise_mismatches"]) == 0, "Author/BN64 bitwise comparison failed")

    fingerprint_map = {}
    for row in inputs:
        case = ck(row)
        key = (*case,row["representation"],row["matrix"],row["stage"])
        require(key not in fingerprint_map, "Duplicate fingerprint")
        require(row["stage"] in ("before","after") and row["matrix"] in ("A","B"), "Invalid fingerprint label")
        require(row["representation"] in ("original_fp32","tf32"), "Unknown input representation")
        require(int(row["element_count"]) == case[3]**2 and int(row["bytes"]) == 4*case[3]**2, "Incorrect input byte count")
        fingerprint_map[key] = row["fnv1a64"]
    require(len(fingerprint_map) == len(expected_cases)*8, "Expected 8 fingerprints per case")
    for group,batch,seed,n in expected_cases:
        for representation in ("original_fp32","tf32"):
            for matrix in ("A","B"):
                values = {fingerprint_map[(g,b,seed,n,representation,matrix,stage)] for g in groups for b in batches for stage in ("before","after")}
                require(len(values)==1, "Inputs changed between precision groups/batches or before/after")
    for n in sizes:
        for representation in ("original_fp32","tf32"):
            for matrix in ("A","B"):
                require(len({fingerprint_map[("tf32",1,s,n,representation,matrix,"before")] for s in seeds}) == len(seeds), "Different seeds produced duplicate inputs")
    window_keys = Counter(rk(x) for x in windows)
    require(set(window_keys) == expected_rounds and all(v==1 for v in window_keys.values()), "Missing/duplicate execution windows")
    for process in processes:
        selected = sorted((x for x in windows if x["group"]==process["group"] and x["batch"]==process["batch"]),
                          key=lambda x:(int(x["case_index"]),int(x["round"]),plan_map[rk(x)][1]))
        previous = int(process["start_utc_ms"])
        for x in selected:
            require(previous <= int(x["start_utc_ms"]) <= int(x["end_utc_ms"]), "Window order conflicts with plan")
            previous = int(x["end_utc_ms"])
        require(previous<=int(process["end_utc_ms"]), "Window exceeds process")

    telemetry = []
    for path in sorted(run.glob("telemetry-*.csv")):
        data = [{k.strip():(v or "").strip() for k,v in x.items() if k is not None} for x in read(path)]
        row = dict(path=str(path.relative_to(run)),samples=len(data),scope="sparse whole-process samples; includes CPU checks; not event-level constant clocks")
        for term in ("temperature.gpu","clocks.current.sm","clocks.current.memory","clocks.sm","clocks.mem","power.draw","utilization.gpu"):
            keys = [k for k in (data[0] if data else []) if k.split(" ")[0] == term]
            if not keys:
                continue
            values = []
            for item in data:
                try:
                    v = float(item[keys[0]].split()[0])
                    if math.isfinite(v): values.append(v)
                except (ValueError,IndexError):
                    pass
            row.update({term+"_available_samples":len(values),term+"_missing_samples":len(data)-len(values),
                        term+"_min":min(values) if values else "missing",term+"_max":max(values) if values else "missing"})
        telemetry.append(row)
        source_files.append(path)
    audit = dict(status="DATA_AUDIT_PASS",formal_events=len(events),round_groups=len(event_groups),
                 cases=len(expected_cases),processes=len(processes),implementation_summaries=len(summaries),
                 paired_rounds=len(pairs),paired_summaries=len(paired),check_records=len(checks),
                 bitwise_records=len(bits),input_fingerprint_records=len(inputs),cpu_sample_position_files=cpu_positions_checked,
                 required_bitwise_checks=sum(int(x["required"]) for x in bits),
                 required_bitwise_mismatches=sum(int(x["bitwise_mismatches"]) for x in bits if x["required"] == "1"),
                 allclose_expression="abs(got-ref)<=eps+eps*abs(got)",relative_error_denominator="max(abs(ref),1e-12)",
                 references=references and {g:sorted(v) for g,v in references.items()},
                 tf32_original_fp32_abs001_failed_checks=sum(x["group"]=="tf32" and x["reference"]=="original_fp32" and int(x["abs_001_fail"])>0 for x in checks),
                 fp32_simt_allclose_failed_checks=sum(x["group"]=="fp32" and int(x["simt_allclose_fail"])>0 for x in checks),
                 accuracy_policy="No NaN/Inf; FP32 abs<=0.01 acceptance; SIMT eps0.001 separately judged and failures retained; TF32 represented-input native eps0.1 acceptance",
                 numerical_status={g:{criterion:("FAIL" if any(x[criterion+"_status"]=="FAIL" for x in quality if x["group"]==g) else "PASS")
                                      for criterion in ("abs_001","simt_allclose","native_allclose","nonfinite")} for g in groups},
                 removed_events=0,telemetry_files=len(telemetry),telemetry_missing_marker=(run/"TELEMETRY_MISSING.txt").exists(),
                 scope="within-session descriptive comparison; cases/rounds nested in three processes per precision; no IID event inference",
                 timing="public launch API CUDA events; conversion, initialization and same-stream C0 D2D restoration excluded")
    out.mkdir(parents=True)
    for filename,data in (("rounds.csv",round_rows),("summary.csv",summaries),("paired-rounds.csv",pairs),("paired.csv",paired),
                          ("cases.csv",cases),("size-summary.csv",size_summary),("throughput-ranges.csv",throughput_ranges),
                          ("errors.csv",errors),("input-fingerprints.csv",inputs),("processes.csv",processes),
                          ("correctness-overview.csv",quality),("correctness-by-reference.csv",reference_quality),
                          ("bitwise.csv",bits),("execution-order.csv",order_rows),("telemetry-summary.csv",telemetry)):
        save(out/filename,data)
    (out/"audit.json").write_text(json.dumps(audit,indent=2)+"\n",encoding="utf-8")
    manifest = [dict(path=str(path.relative_to(run)),bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for path in sorted(set(source_files))]
    (out/"raw-files-read-sha256.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    lines = ["# RTX 4090 D CUTLASS 对照统计", "",f"正式数据：{len(events):,} 个 events，两个精度组，{len(processes)} 个独立进程。", "",
             "每组合先计算每轮全部 30 events 的平均延迟，再取七轮均值的中位数。未删除慢样本；不混入 profiler、smoke 或旧成绩。", "",
             "下表为三个 batch × 三个 seed 的九个组合：TFLOPS 中位数（最小～最大）。这只是描述性汇总，组合嵌套在同一进程中，不是九次完全独立试验。", ""]
    for group,names in groups.items():
        lines += ["## "+("TF32 输入 / FP32 累加" if group=="tf32" else "Strict FP32 SIMT"),"",
                  "| N | "+" | ".join(names)+" |","|---:|"+"---:|"*len(names)]
        for n in sizes:
            data = {x["implementation"]:x for x in throughput_ranges if x["group"]==group and x["n"]==n}
            cells = [f'{data[name]["median_case_tflops"]:.3f} ({data[name]["min_case_tflops"]:.3f}～{data[name]["max_case_tflops"]:.3f})' for name in names]
            lines.append(f"| {n} | "+" | ".join(cells)+" |")
        lines += ["", "| N | CUTLASS 配置 | 参考 | 九组合配对提升中位数范围 | 正收益组合 | 正收益轮 |", "|---:|---|---|---:|---:|---:|"]
        for x in size_summary:
            if x["group"]==group and x["candidate"].startswith("cutlass") and not x["reference"].startswith("cutlass"):
                lines.append(f'| {x["n"]} | {x["candidate"]} | {x["reference"]} | {x["min_case_paired_median_pct"]:+.3f}%～{x["max_case_paired_median_pct"]:+.3f}% | {x["positive_case_medians"]}/{x["case_combinations"]} | {x["positive_rounds"]}/{x["total_rounds"]} |')
        lines += ["", "各误差标准独立判定（任一 seed/batch/check 失败即该格为 FAIL）：TF32 这里采用同一量化输入的 GPU 参考及 CPU double；strict FP32 采用原始输入。CPU 仅抽样 256 点，GPU 参考为全量输出。", "",
                  "| N | 实现 | 最大绝对误差 | abs≤0.01 | SIMT allclose | 本精度 native allclose | NaN/Inf 检查 |",
                  "|---:|---|---:|---|---|---|---|"]
        for x in quality:
            if x["group"]==group:
                lines.append(f'| {x["n"]} | {x["implementation"]} | {x["max_abs"]:.7g} | {x["abs_001_status"]} | {x["simt_allclose_status"]} | {x["native_allclose_status"]} | {x["nonfinite_status"]} |')
        if group=="tf32":
            lines += ["", "TF32 对原始 FP32 的精度损失另表，不与上面的同输入正确性混淆：", "",
                      "| N | 实现 | 最大绝对误差 | abs≤0.01 | SIMT allclose |", "|---:|---|---:|---|---|"]
            for x in reference_quality:
                if x["group"]==group and x["reference"]=="original_fp32":
                    lines.append(f'| {x["n"]} | {x["implementation"]} | {x["max_abs"]:.7g} | {x["abs_001_status"]} | {x["simt_allclose_status"]} |')
        lines += [""]
    lines += ["配对提升按每轮 `reference_ms / candidate_ms - 1` 计算，和两个中位数之比不同；两种数值均保留在 CSV。", "",
              "TF32 相对原始 FP32 的误差失败保留在 errors.csv，不能解释为严格 FP32 成绩。CUTLASS 仅测预先选定模板，未搜索整个库。", "",
              "DATA_AUDIT_PASS 表示原始记录、计划和统计完整性通过，不表示所有数值阈值都通过。SIMT allclose 的 FAIL 保持原阈值和失败数量，未删除失败检查，也未将其改成 PASS。", ""]
    (out/"README.md").write_text("\n".join(lines),encoding="utf-8")
    print(json.dumps(audit,ensure_ascii=False))


if __name__ == "__main__":
    main()
