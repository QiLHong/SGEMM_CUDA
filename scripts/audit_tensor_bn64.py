"""BN64 audit; adapted from audit_tensor_ktile.py with preserved helper functions."""
import argparse
import collections
import hashlib
import json
import re
import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
helpers = runpy.run_path(str(ROOT / "scripts/audit_tensor_grid.py"))
load, write, functions = (helpers[x] for x in ("load", "write", "functions"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    archive, out = args.archive, args.output
    out.mkdir(parents=True, exist_ok=False)
    audit = {}
    for stem in ("source-before", "history-before"):
        rows = load(archive / f"preflight/{stem}.csv")
        bad = [r["path"] for r in rows if not (ROOT / r["path"]).exists() or
               hashlib.sha256((ROOT / r["path"]).read_bytes()).hexdigest().lower() != r["sha256"].lower()]
        audit[stem] = dict(count=len(rows), changed=bad)
        assert not bad, bad
    current = functions(archive / "build02/binary.sass")
    old = functions(ROOT / "results/windows/20261021-tensorcore/build04/binary.sass")
    selected = {n: v for n, v in current.items() if "tf32" in n and "dbf" in n}
    previous = functions(ROOT / "results/windows/20261021-tensorcore-ktile/build01/binary.sass")
    first_build = functions(archive / "build01/binary.sass")
    assert all(v == first_build[n] for n, v in selected.items())
    assert all(v == previous[n] for n, v in selected.items() if n in previous)
    audit["host_order_fix_device_sass_equal"] = True
    audit["all_old_device_sass_equal"] = True
    static, loops = [], []
    assert len(selected) == 4
    for name, instructions in selected.items():
        texts = [t for _, t in instructions]
        ops = [re.sub(r"^@\S+\s+", "", t).split()[0].split(".")[0] for t in texts]
        counts = collections.Counter(ops)
        label = "bn64" if "bn64_tf32" in name else "original16" if name.startswith("_ZN12vitamin") else ("dynamic32" if "Li32E" in name else "dynamic16")
        static.append(dict(implementation=label, function=name, instructions=len(texts),
                           non_nop_instructions=sum(op != "NOP" for op in ops),
                           **{op: counts[op] for op in ("HMMA", "LDGSTS", "LDSM", "BAR", "DEPBAR", "LDL", "STL")}))
        if label == "original16":
            audit["original_sass_unchanged"] = texts == [t for _, t in old[name]]
            assert audit["original_sass_unchanged"]
        for addr, text in instructions:
            target = re.search(r"\bBRA\s+(0x[0-9a-f]+)", text)
            if not target or int(target[1], 16) >= addr:
                continue
            body = [(a, t) for a, t in instructions if int(target[1], 16) <= a <= addr]
            if not any("LDGSTS" in t for _, t in body):
                continue
            last_copy = max(a for a, t in body if "LDGSTS" in t)
            waits = [a for a, t in body if t.startswith("DEPBAR.") and a > last_copy]
            assert waits  # LDGDEPBAR is commit; it is not the completion wait.
            wait = waits[0]
            loops.append(dict(implementation=label, start=hex(int(target[1], 16)), end=hex(addr),
                              instructions=len(body), last_copy=hex(last_copy), wait=hex(wait),
                              copies=sum("LDGSTS" in t for _, t in body),
                              hmma=sum("HMMA" in t for _, t in body),
                              hmma_between_last_copy_and_wait=sum("HMMA" in t and last_copy < a < wait for a, t in body)))
            (out / f"{label}-main-loop.sass.txt").write_text(
                "\n".join(f"{a:04x}: {t}" for a, t in body)+"\n", encoding="utf-8")
    write(out / "sass-counts.csv", static)
    write(out / "sass-loops.csv", loops)
    telemetry = []
    for folder, analysis, baseline in (("run01", "analysis01", "tf32_dynamic_bk32"),
                                       ("confirm01", "confirm-analysis01", "vitamin_tf32_dbf")):
        data = load(archive / folder / "telemetry.csv")
        cleaned = [{k.strip(): v.strip() for k, v in row.items()} for row in data]
        row = dict(run=folder, samples=len(data))
        for col, label in (("temperature.gpu", "temperature_c"), ("clocks.current.sm [MHz]", "sm_mhz"),
                           ("clocks.current.memory [MHz]", "memory_mhz"),
                           ("power.draw [W]", "power_w"), ("utilization.gpu [%]", "utilization_pct")):
            values = [float(r[col].split()[0]) for r in cleaned]
            row[label+"_min"], row[label+"_max"] = min(values), max(values)
        telemetry.append(row)
        round_rows = load(archive / analysis / "rounds.csv")
        for n in (1024, 2048, 4096):
            for r in range(7):
                order = {x["implementation"]: int(x["order"]) for x in round_rows
                         if int(x["n"]) == n and int(x["round"]) == r}
                assert order["tf32_bn64_bk32"] - order[baseline] == (1 if r % 2 == 0 else -1)
        audit[folder+"_adjacent_and_swapping"] = True
    write(out / "telemetry-ranges.csv", telemetry)
    for n in (1024, 2048, 4096):
        fingerprints = []
        for folder in ("run01", "confirm01"):
            text = (archive / folder / f"timings-{n}.csv.meta.txt").read_text(encoding="utf-8-sig")
            fingerprints.append(sorted(x for x in text.splitlines() if "_fnv1a64=" in x))
        assert len(fingerprints[0]) == 6 and fingerprints[0] == fingerprints[1]
    audit["input_fingerprints_equal"] = True
    metrics_path = archive / "confirm-analysis02/ncu-metrics.csv"
    if metrics_path.exists():
        selected_metrics = {}
        for row in load(metrics_path):
            value = row["value"]
            try:
                value = float(value.replace(",", ""))
            except ValueError:
                pass  # NCU also emits string-valued launch/cache configuration.
            selected_metrics.setdefault(row["file"], dict(file=row["file"]))[row["metric"]] = value
        assert len(selected_metrics) == 12
        write(out / "ncu-selected.csv", list(selected_metrics.values()))
        audit["ncu_launches"] = 12
    audit["static_functions"] = len(static)
    audit["k_loops"] = len(loops)
    elapsed_path = archive / "elapsed-analysis01/elapsed-selected.csv"
    if elapsed_path.exists():
        elapsed = load(elapsed_path)
        assert len(elapsed) == 6
        assert all(0 < float(row[key]) <= 100 for row in elapsed for key in
                   ("sm_active_elapsed_pct", "tensor_active_pct", "tensor_elapsed_pct"))
        assert (archive / "profile-elapsed01/PROFILE_COMPLETE.txt").exists()
        write(out / "elapsed-selected.csv", elapsed)
        audit["supplemental_ncu_launches"] = 6
        audit["total_ncu_launches"] = 18
    (out / "audit.json").write_text(json.dumps(audit, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(audit))


if __name__ == "__main__":
    main()
