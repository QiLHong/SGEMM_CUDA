"""Independent CPU statistics; never remove events or merge profiler timing."""
import argparse
import collections
import csv
import json
import math
import re
import statistics as st
from pathlib import Path


def read(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write(path, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, rows[0].keys())
        w.writeheader()
        w.writerows(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--build", type=Path, required=True)
    p.add_argument("--profile", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    assert (args.run / "RUN_COMPLETE.txt").exists(), "Incomplete run"
    env = json.loads((args.run / "environment.json").read_text(encoding="utf-8-sig"))
    rounds, repeats = env["rounds"], env["repeats"]
    sizes = (4096,) if env.get("pairs_only") else (1024, 2048, 4096)
    implementation_count = 4 if env.get("pairs_only") else 8
    groups = collections.defaultdict(list)
    all_events = []
    for n in sizes:
        events = read(args.run / f"timings-{n}.csv")
        assert len(events) == implementation_count*rounds*repeats
        keys = set()
        for row in events:
            assert int(row["m"]) == int(row["n"]) == int(row["k"]) == n
            key = (n, row["implementation"], int(row["round"]), int(row["iteration"]))
            assert key not in keys
            keys.add(key)
            ms = float(row["milliseconds"])
            assert ms > 0 and math.isfinite(ms)
            assert math.isclose(float(row["tflops"]), 2*n**3/(ms*1e9), rel_tol=1e-12)
            groups[key[:3]].append(ms)
        all_events.extend(events)
    means, round_rows = {}, []
    for (n, impl, r), values in sorted(groups.items()):
        assert len(values) == repeats
        means[n, impl, r] = st.mean(values)
        round_rows.append(dict(n=n, implementation=impl, round=r, events=len(values),
                               mean_ms=st.mean(values), median_event_ms=st.median(values),
                               min_event_ms=min(values), max_event_ms=max(values),
                               tflops_from_mean=2*n**3/(st.mean(values)*1e9)))
    write(args.output / "rounds.csv", round_rows)
    summary = []
    names = sorted({key[1] for key in means})
    for n in sizes:
        for name in names:
            values = [means[n, name, r] for r in range(rounds)]
            median = st.median(values)
            summary.append(dict(n=n, implementation=name, rounds=rounds, events=rounds*repeats,
                                median_round_mean_ms=median, tflops=2*n**3/(median*1e9),
                                min_round_mean_ms=min(values), max_round_mean_ms=max(values),
                                round_cv_pct=100*st.stdev(values)/st.mean(values),
                                range_over_median_pct=100*(max(values)-min(values))/median))
    write(args.output / "summary.csv", summary)
    paired, pair_summary = [], []
    pairs = [("vitamin_tf32_dbf", "vitamin_tf32_single"),
             ("vitamin_tf32_dbf", "cublas_tf32"),
             ("vitamin_fp16_dbf_rw", "vitamin_fp16_single"),
             ("vitamin_fp16_dbf_rw", "cublas_fp16"),
             ("vitamin_tf32_dbf", "k30_fp32"),
             ("vitamin_fp16_dbf_rw", "k30_fp32")]
    pairs = [(a, b) for a, b in pairs if a in names and b in names]
    for n in sizes:
        for candidate, baseline in pairs:
            changes = []
            for r in range(rounds):
                change = 100*(means[n, baseline, r]/means[n, candidate, r]-1)
                changes.append(change)
                paired.append(dict(n=n, candidate=candidate, baseline=baseline, round=r,
                                   throughput_change_pct=change))
            pair_summary.append(dict(n=n, candidate=candidate, baseline=baseline,
                                     paired_median_pct=st.median(changes), min_pct=min(changes),
                                     max_pct=max(changes), faster_rounds=sum(x>0 for x in changes), rounds=rounds))
    write(args.output / "pairs.csv", paired)
    write(args.output / "paired-summary.csv", pair_summary)
    checks = []
    for file in [args.run / f"small-{k}.csv.checks.csv" for k in (32, 64)] + [
            args.run / f"timings-{n}.csv.checks.csv" for n in sizes]:
        checks.extend(read(file))
    assert len(checks) == 2*8*2*4+len(sizes)*implementation_count*(rounds+1)*4
    assert all(int(x["nonfinite"]) == 0 for x in checks)
    checkgroups = collections.defaultdict(list)
    for x in checks:
        checkgroups[int(x["m"]), int(x["n"]), int(x["k"]), x["implementation"], x["reference"]].append(x)
    checksummary = []
    for (m, n, k, impl, ref), group in sorted(checkgroups.items()):
        checksummary.append(dict(m=m, n=n, k=k, implementation=impl, reference=ref,
                                validations=len(group), max_abs=max(float(x["max_abs"]) for x in group),
                                max_rel=max(float(x["max_rel"]) for x in group),
                                relative_l2=max(float(x["relative_l2"]) for x in group),
                                nonfinite=max(int(x["nonfinite"]) for x in group),
                                abs001_failed_checks=sum(int(x["abs_001_fail"])>0 for x in group),
                                simt_failed_checks=sum(int(x["simt_allclose_fail"])>0 for x in group),
                                native_failed_checks=sum(int(x["native_allclose_fail"])>0 for x in group),
                                max_abs001_failed_elements=max(int(x["abs_001_fail"]) for x in group)))
    write(args.output / "correctness-summary.csv", checksummary)
    # 机器码是被测二进制的导出，按函数计数。静态数量不是动态执行量或重叠收益。
    sass = (args.build / "binary.sass").read_text(encoding="utf-8-sig")
    sections = re.split(r"Function\s*:\s*(\S+)", sass)
    static = []
    for i in range(1, len(sections), 2):
        body = sections[i+1]
        row = dict(function=sections[i])
        for opcode in ("HMMA", "FFMA", "LDGSTS", "LDSM", "BAR", "DEPBAR", "LDL", "STL"):
            row[opcode] = len(re.findall(r"\b"+opcode+r"(?:\.|\s)", body))
        row["hmma_variants"] = ";".join(sorted(set(re.findall(r"\bHMMA\S*", body))))
        static.append(row)
    write(args.output / "sass-summary.csv", static)
    assert len([x for x in static if x["HMMA"]]) == 4
    if args.profile:
        profiles = []
        for f in sorted(args.profile.glob("*.csv")):
            lines = f.read_text(encoding="utf-8-sig").splitlines()
            headers = [i for i, line in enumerate(lines) if line.startswith('"ID",')]
            if not headers:
                continue
            data = list(csv.DictReader(lines[headers[0]:]))
            units = data[0]
            rows = [x for x in data[1:] if x.get("ID") == "0"]
            assert len(rows) == 1, f
            raw = rows[0]
            for metric in raw:
                if metric.startswith(("sm__", "smsp__", "launch__", "gpu__", "l1tex__")):
                    try:
                        value = float(raw[metric].replace(",", ""))
                    except (ValueError, AttributeError):
                        continue  # NCU includes string metadata such as CachePreferNone.
                    assert math.isfinite(value)
                    profiles.append(dict(file=f.name, kernel=raw["Kernel Name"], metric=metric,
                                         unit=units[metric], value=value))
        if profiles:
            write(args.output / "ncu-metrics.csv", profiles)
    table = ["| implementation | "+" | ".join(f"{n} TFLOPS" for n in sizes)+" |",
             "|---|"+"---:|"*len(sizes)]
    for name in names:
        values = [next(x["tflops"] for x in summary if x["n"]==n and x["implementation"]==name)
                  for n in sizes]
        table.append("| "+name+" | "+" | ".join(f"{v:.3f}" for v in values)+" |")
    (args.output / "table.md").write_text("\n".join(table)+"\n", encoding="utf-8")
    audit = dict(events=len(all_events), round_groups=len(means), correctness_rows=len(checks),
                 nonfinite=0, no_samples_removed=True,
                 scope="Numerical FP32 failures preserved; same-precision comparison is a separate contract")
    (args.output / "audit.json").write_text(json.dumps(audit, indent=2)+"\n")
    print("\n".join(table))
    print(json.dumps(audit))


if __name__ == "__main__":
    main()
