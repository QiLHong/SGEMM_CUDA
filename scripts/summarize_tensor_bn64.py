"""Stdlib-only event analysis: no event removal and no profiler timing mixing."""
import argparse
import collections
import csv
import io
import json
import math
import re
import statistics as st
from pathlib import Path


def read(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def save(path, rows):
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, rows[0].keys())
        w.writeheader()
        w.writerows(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--profile", type=Path)
    args = p.parse_args()
    run, out = args.run, args.output
    assert (run / "RUN_COMPLETE.txt").exists(), "Incomplete run"
    meta = json.loads((run / "environment.json").read_text(encoding="utf-8-sig"))
    out.mkdir(parents=True, exist_ok=False)
    rounds, summary, paired, pairs = [], [], [], []
    total = 0
    for path in sorted(run.glob("timings-*.csv")):
        if not re.fullmatch(r"timings-\d+\.csv", path.name):
            continue
        data = read(path)
        n = int(data[0]["n"])
        groups = collections.defaultdict(list)
        for row in data:
            ms = float(row["milliseconds"])
            assert ms > 0 and math.isfinite(ms)
            assert int(row["m"]) == int(row["k"]) == n
            groups[row["implementation"], int(row["round"])].append(row)
        means = {}
        names = {x[0] for x in groups}
        expected = 4 if meta["selection"] == "all" else 3
        assert len(names) == expected and len(data) == expected * meta["rounds"] * meta["repeats"]
        total += len(data)
        for (name, r), values in sorted(groups.items()):
            assert sorted(int(x["iteration"]) for x in values) == list(range(meta["repeats"]))
            ms = [float(x["milliseconds"]) for x in values]
            means[name, r] = st.mean(ms)
            rounds.append(dict(n=n, implementation=name, round=r, order=values[0]["order"],
                               mean_ms=st.mean(ms), event_min_ms=min(ms), event_max_ms=max(ms),
                               tflops=2*n**3 / (st.mean(ms)*1e9)))
        for name in sorted(names):
            ms = [means[name, r] for r in range(meta["rounds"])]
            summary.append(dict(n=n, implementation=name, median_round_mean_ms=st.median(ms),
                                tflops=2*n**3 / (st.median(ms)*1e9),
                                round_min_ms=min(ms), round_max_ms=max(ms),
                                round_cv_pct=st.stdev(ms)/st.mean(ms)*100))
        for kind, baseline in (("tf32", "vitamin_tf32_dbf"),):
            for name in sorted(names):
                if name == "cublas_tf32":
                    continue
                for ref in (baseline, "tf32_dynamic_bk32", "cublas_tf32"):
                    if name == ref or ref not in names:
                        continue
                    gain = [(means[ref, r]/means[name, r]-1)*100 for r in range(meta["rounds"])]
                    for r, g in enumerate(gain):
                        pairs.append(dict(n=n, candidate=name, reference=ref, round=r, gain_pct=g,
                                          reference_ms=means[ref, r], candidate_ms=means[name, r]))
                    paired.append(dict(n=n, candidate=name, reference=ref,
                                       median_paired_gain_pct=st.median(gain), min_gain_pct=min(gain),
                                       max_gain_pct=max(gain), wins=sum(x > 0 for x in gain),
                                       rounds=len(gain)))
    checks = [row for f in sorted(run.glob("*.checks.csv")) for row in read(f)]
    bitwise = [row for f in sorted(run.glob("*.bitwise.csv")) for row in read(f)]
    assert bitwise and all(int(x["bitwise_mismatches"]) == 0 for x in bitwise)
    assert all(int(x["nonfinite"]) == 0 for x in checks)
    assert all(int(x["native_allclose_fail"]) == 0 for x in checks if x["reference"] != "original_fp32")
    for name, rows in [("rounds.csv", rounds), ("summary.csv", summary),
                       ("paired.csv", paired), ("paired-rounds.csv", pairs)]:
        save(out / name, rows)
    error_groups = collections.defaultdict(list)
    for row in checks:
        error_groups[row["m"], row["n"], row["k"], row["implementation"], row["reference"]].append(row)
    errors = []
    for key, values in sorted(error_groups.items()):
        result = dict(zip(("m", "n", "k", "implementation", "reference"), key))
        for field in ("max_abs", "max_rel", "relative_l2", "nonfinite", "abs_001_fail",
                      "simt_allclose_fail", "native_allclose_fail"):
            result["worst_"+field] = max(float(x[field]) for x in values)
        errors.append(result)
    save(out / "errors.csv", errors)
    audit = dict(events=total, round_groups=len(rounds), check_rows=len(checks),
                 bitwise_checks=len(bitwise), bitwise_mismatches=0, nonfinite=0,
                 original_fp32_failed_rows=sum(int(x["abs_001_fail"]) > 0 for x in checks
                                               if x["reference"] == "original_fp32"),
                 removed_events=0)
    if args.profile:
        metrics = []
        for f in sorted(args.profile.glob("*.csv")):
            lines = f.read_text(encoding="utf-8-sig").splitlines()
            start = next((i for i, x in enumerate(lines) if x.startswith('"ID",')), None)
            if start is None:
                continue
            rows = list(csv.DictReader(io.StringIO("\n".join(lines[start:]))))
            assert len(rows) == 2, f
            for key, value in rows[1].items():
                if key.startswith(("launch__", "sm__", "smsp__", "l1tex__", "lts__", "dram__")):
                    metrics.append(dict(file=f.name, metric=key, value=value, unit=rows[0][key]))
        save(out / "ncu-metrics.csv", metrics)
        audit["profile_files"] = len({x["file"] for x in metrics})
    (out / "audit.json").write_text(json.dumps(audit, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(audit))
    for row in summary:
        print(f'{row["n"]:4} {row["implementation"]:24} {row["tflops"]:8.3f} TFLOPS CV {row["round_cv_pct"]:.2f}%')
    for row in paired:
        if "bn64" in row["candidate"] and row["reference"].startswith("vitamin"):
            print(f'{row["n"]} {row["candidate"]} vs original BK16: {row["median_paired_gain_pct"]:+.2f}% ({row["wins"]}/{row["rounds"]})')


if __name__ == "__main__":
    main()
