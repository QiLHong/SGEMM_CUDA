"""Audit preserved files, device SASS, timestamps and Tensor Core counters."""
import argparse
import collections
import csv
import datetime as dt
import hashlib
import json
import re
from pathlib import Path


def read(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def functions(path):
    parts = re.split(r"Function\s*:\s*(\S+)", path.read_text(encoding="utf-8-sig"))
    return {parts[i]: re.findall(r"/\*([0-9a-f]+)\*/\s+(.+?)\s*;", parts[i+1])
            for i in range(1, len(parts), 2)}


def write(path, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, rows[0].keys())
        w.writeheader()
        w.writerows(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--telemetry-offset-minutes", type=int, default=480)
    args = p.parse_args()
    root, out = args.root, args.output
    out.mkdir(parents=True, exist_ok=False)
    repo = Path(__file__).resolve().parents[1]
    sources = read(root / "preflight/source-before.csv")
    history = read(root / "preflight/historical-results-before.csv")
    for row in sources:
        assert sha(repo / row["path"]) == row["sha256"].upper(), row["path"]
    for row in history:
        assert sha(Path(row["path"])) == row["sha256"].upper(), row["path"]
    before = functions(root / "build03/binary.sass")
    after = functions(root / "build04/binary.sass")
    assert before == after, "Host pair-order change must not change device instructions"
    loops = []
    for name, ins in before.items():
        if "vitamin_" not in name:
            continue
        instructions = [(int(addr, 16), text) for addr, text in ins]
        for address, text in instructions:
            match = re.search(r"\bBRA\s+(0x[0-9a-f]+)", text)
            if not match or int(match[1], 16) >= address:
                continue
            start = int(match[1], 16)
            body = [(a, t) for a, t in instructions if start <= a <= address]
            hmma = [(a, t) for a, t in body if t.startswith("HMMA")]
            copies = [(a, t) for a, t in body if t.startswith("LDGSTS")]
            waits = [(a, t) for a, t in body if t.startswith("DEPBAR.")]
            if not hmma or not copies:
                continue
            loops.append(dict(function=name, loop_begin=hex(start), loop_end=hex(address),
                              hmma=len(hmma), async_copy_instructions=len(copies),
                              first_hmma=hex(hmma[0][0]), last_async_copy=hex(copies[-1][0]),
                              wait=hex(waits[0][0]) if waits else "",
                              hmma_after_last_copy_before_wait=sum(copies[-1][0]<a<waits[0][0] for a,t in hmma) if waits else -1))
            filtered = [(a, t) for a, t in body if re.search(r"HMMA|LDGSTS|DEPBAR|BAR\.|BRA", t)]
            (out / (name+".loop.txt")).write_text("\n".join(f"{a:04x} {t};" for a,t in filtered)+"\n")
    assert len(loops) == 4
    write(out / "pipeline.csv", loops)
    telemetry = []
    for run in ("run02", "pair-confirm01"):
        folder = root / run
        assert (folder / "RUN_COMPLETE.txt").exists()
        points = []
        zone = dt.timezone(dt.timedelta(minutes=args.telemetry_offset_minutes))
        for row in read(folder / "telemetry.csv"):
            normalized = {k.strip(): v.strip() for k,v in row.items()}
            stamp = dt.datetime.strptime(normalized["timestamp"], "%Y/%m/%d %H:%M:%S.%f").replace(tzinfo=zone)
            points.append(dict(utc_ms=stamp.timestamp()*1000,
                               temperature=float(normalized["temperature.gpu"]),
                               sm_mhz=float(normalized["clocks.current.sm [MHz]"].split()[0]),
                               memory_mhz=float(normalized["clocks.current.memory [MHz]"].split()[0])))
        windows = []
        for f in folder.glob("timings-*.csv.windows.csv"):
            windows.extend(read(f))
        matched = 0
        for w in windows:
            samples = [x for x in points if int(w["start_utc_ms"]) <= x["utc_ms"] <= int(w["end_utc_ms"])]
            matched += bool(samples)
        telemetry.append(dict(run=run, samples=len(points), windows=len(windows), windows_with_samples=matched,
                              temperature_min=min(x["temperature"] for x in points),
                              temperature_max=max(x["temperature"] for x in points),
                              sm_mhz_min=min(x["sm_mhz"] for x in points), sm_mhz_max=max(x["sm_mhz"] for x in points),
                              memory_mhz_min=min(x["memory_mhz"] for x in points),
                              memory_mhz_max=max(x["memory_mhz"] for x in points),
                              timestamp_utc_offset_minutes=args.telemetry_offset_minutes,
                              caveat="Windows include warmup/validation; sampled clocks do not prove every event's clock"))
    write(out / "telemetry-summary.csv", telemetry)
    metric_names = [
        "launch__registers_per_thread", "launch__shared_mem_per_block",
        "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active",
        "sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active",
        "sm__warps_active.avg.pct_of_peak_sustained_active",
        "smsp__warps_eligible.avg.per_cycle_active",
        "smsp__warps_issue_stalled_barrier.avg.per_cycle_active",
        "smsp__warps_issue_stalled_short_scoreboard.avg.per_cycle_active",
        "smsp__warps_issue_stalled_long_scoreboard.avg.per_cycle_active",
        "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum",
        "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum",
        "l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum",
        "l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum",
    ]
    ncu = []
    for f in sorted((root / "profile01").glob("*.csv")):
        lines = f.read_text(encoding="utf-8-sig").splitlines()
        h = next(i for i,x in enumerate(lines) if x.startswith('"ID",'))
        data = list(csv.DictReader(lines[h:]))
        raw = next(x for x in data if x["ID"] == "0")
        row = dict(file=f.name, kernel=raw["Kernel Name"])
        for metric in metric_names:
            row[metric] = float(raw[metric].replace(",", ""))
        tensor = row["sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active"]
        if "fp32" in f.stem:
            assert tensor == 0
        else:
            assert tensor > 0
        assert row["l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum"] == 0
        assert row["l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum"] == 0
        ncu.append(row)
    assert len(ncu) == 12
    write(out / "ncu-selected.csv", ncu)
    # 两次大尺寸实验的实际输入指纹逐项相同，而不只依赖seed相同。
    def fingerprints(path):
        return sorted(x for x in path.read_text().splitlines() if x.startswith("input_"))
    assert fingerprints(root / "run02/timings-4096.csv.meta.txt") == fingerprints(root / "pair-confirm01/timings-4096.csv.meta.txt")
    result = dict(preserved_source_files=len(sources), preserved_historical_files=len(history),
                  device_instruction_lists_identical=True, pipeline_loops=len(loops),
                  profiles=len(ncu), all_tensor_profiles_have_tensor_activity=True,
                  all_local_memory_sectors_zero=True, confirmation_input_fingerprints_identical=True,
                  sanitizer="unverified")
    (out / "audit.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
