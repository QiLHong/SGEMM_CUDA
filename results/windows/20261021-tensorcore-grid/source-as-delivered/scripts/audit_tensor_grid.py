"""Preservation, static SASS, paired ordering and telemetry evidence."""
import argparse
import collections
import csv
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, rows[0].keys())
        w.writeheader()
        w.writerows(rows)


def functions(path):
    source = path.read_text(encoding="utf-8-sig")
    result = {}
    for block in re.split(r"Function\s*:\s*", source)[1:]:
        name = block.splitlines()[0].strip()
        instructions = re.findall(r"/\*([0-9a-f]+)\*/\s+(.+?)\s*;\s*/\*", block)
        result[name] = [(int(a, 16), i.strip()) for a, i in instructions]
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    archive, out = args.archive, args.output
    out.mkdir(parents=True, exist_ok=False)
    audit = {}
    for stem in ("source-before", "results-before"):
        rows = load(archive / f"preflight/{stem}.csv")
        bad = []
        for row in rows:
            path = ROOT / row["path"]
            if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest().lower() != row["sha256"].lower():
                bad.append(row["path"])
        audit[stem] = dict(count=len(rows), changed=bad)
        assert not bad, bad
    current = functions(archive / "build01/binary.sass")
    old = functions(ROOT / "results/windows/20261021-tensorcore/build04/binary.sass")
    selected = {n: v for n, v in current.items() if "dbf" in n}
    static, loops = [], []
    for name, instructions in selected.items():
        texts = [t for _, t in instructions]
        opcodes = [re.sub(r"^@\S+\s+", "", t).split()[0].split(".")[0] for t in texts]
        counts = collections.Counter(opcodes)
        row = dict(function=name, instructions=len(texts),
                   non_nop_instructions=sum(op != "NOP" for op in opcodes),
                   **{op: counts[op] for op in ("HMMA", "LDGSTS", "LDSM", "BAR", "DEPBAR", "LDL", "STL")})
        static.append(row)
        if name.startswith("_ZN12vitamin"):
            same = texts == [t for _, t in old[name]]
            audit[name+"_old_sass_equal"] = same
            assert same
        # Backward branch enclosing async copies is the K-tile main loop.
        for addr, text in instructions:
            target = re.search(r"\bBRA\s+(0x[0-9a-f]+)", text)
            if target and int(target[1], 16) < addr:
                body = [(a, t) for a, t in instructions if int(target[1], 16) <= a <= addr]
                if not any("LDGSTS" in t for _, t in body):
                    continue
                last_copy = max(a for a, t in body if "LDGSTS" in t)
                # LDGDEPBAR commits a group; DEPBAR.LE waits for its completion.
                # A substring match would wrongly treat the commit as a wait.
                waits = [a for a, t in body if t.startswith("DEPBAR.") and a > last_copy]
                assert waits
                wait = waits[0]
                overlap = sum("HMMA" in t and last_copy < a < wait for a, t in body)
                loops.append(dict(function=name, start=hex(int(target[1], 16)), end=hex(addr),
                                  instructions=len(body), last_copy=hex(last_copy), wait=hex(wait),
                                  hmma_between_copy_and_wait=overlap,
                                  hmma=sum("HMMA" in t for _, t in body)))
                width = re.search(r"kernelILi([124])E", name)
                label = ("tf32" if "tf32" in name else "fp16")+"-"+(width[1] if width else "8")
                (out / (label+"-main-loop.sass.txt")).write_text(
                    "\n".join(f"{a:04x}: {t}" for a, t in body)+"\n")
    write(out / "sass-counts.csv", static)
    write(out / "sass-loops.csv", loops)
    telemetry = []
    for folder in ("scan01", "confirm-w1-01"):
        data = load(archive / folder / "telemetry.csv")
        cleaned = [{k.strip(): v.strip() for k, v in row.items()} for row in data]
        row = dict(run=folder, samples=len(data))
        for col, label in (("temperature.gpu", "temperature_c"), ("clocks.current.sm [MHz]", "sm_mhz"),
                           ("power.draw [W]", "power_w"), ("utilization.gpu [%]", "utilization_pct")):
            if col not in cleaned[0]:
                continue
            values = [float(r[col].split()[0]) for r in cleaned]
            row[label+"_min"], row[label+"_max"] = min(values), max(values)
        telemetry.append(row)
    write(out / "telemetry-ranges.csv", telemetry)
    # Every confirmation round keeps baseline/candidate adjacent, swaps order.
    round_rows = load(archive / "confirm-analysis01/rounds.csv")
    for n in (1024, 2048, 4096):
        for r in range(7):
            for kind, baseline in (("tf32", "vitamin_tf32_dbf"), ("fp16", "vitamin_fp16_dbf_rw")):
                order = {x["implementation"]: int(x["order"]) for x in round_rows
                         if int(x["n"]) == n and int(x["round"]) == r}
                diff = order[kind+"_w1"] - order[baseline]
                assert diff == (1 if r % 2 == 0 else -1)
    audit["confirmation_adjacent_and_swapping"] = True
    for n in (1024, 2048, 4096):
        fingerprints = []
        for folder in ("scan01", "confirm-w1-01"):
            text = (archive / folder / f"timings-{n}.csv.meta.txt").read_text()
            fingerprints.append(sorted(x for x in text.splitlines() if "_fnv1a64=" in x))
        assert len(fingerprints[0]) == 6 and fingerprints[0] == fingerprints[1]
    audit["scan_confirmation_fingerprints_equal"] = True
    metrics_path = archive / "confirm-analysis02/ncu-metrics.csv"
    if metrics_path.exists():
        selected_keys = [
            "launch__registers_per_thread", "launch__shared_mem_per_block",
            "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active",
            "sm__warps_active.avg.pct_of_peak_sustained_active",
            "smsp__warps_eligible.avg.per_cycle_active",
            "smsp__issue_active.avg.pct_of_peak_sustained_active",
            "smsp__warps_issue_stalled_barrier.avg.per_cycle_active",
            "smsp__warps_issue_stalled_long_scoreboard.avg.per_cycle_active",
            "smsp__warps_issue_stalled_short_scoreboard.avg.per_cycle_active",
            "lts__t_sector_hit_rate.pct", "dram__bytes_read.sum",
            "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum",
            "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum",
            "l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum",
            "l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum",
        ]
        selected_metrics = {}
        for row in load(metrics_path):
            if row["metric"] in selected_keys:
                selected_metrics.setdefault(row["file"], dict(file=row["file"]))[row["metric"]] = \
                    float(row["value"].replace(",", ""))
        assert len(selected_metrics) == 12
        assert all(len(row) == len(selected_keys)+1 for row in selected_metrics.values())
        write(out / "ncu-selected.csv",
              [{key: row[key] for key in ["file"]+selected_keys} for row in selected_metrics.values()])
    audit["static_functions"] = len(static)
    audit["loops"] = len(loops)
    (out / "audit.json").write_text(json.dumps(audit, indent=2)+"\n")
    print(json.dumps(audit))


if __name__ == "__main__":
    main()
