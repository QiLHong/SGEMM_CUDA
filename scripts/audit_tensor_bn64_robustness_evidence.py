"""Compare actual device sources, binary bytes and normalized instructions separately."""
import argparse
import collections
import csv
import difflib
import hashlib
import io
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def functions(path):
    blocks = re.split(r"Function\s*:\s*", path.read_text(encoding="utf-8-sig"))[1:]
    result = {}
    for b in blocks:
        name = b.splitlines()[0].strip()
        ins = re.findall(r"/\*([0-9a-f]+)\*/\s+(.+?)\s*;\s*/\*", b)
        # Actual encoded 128-bit instructions include scheduling/control words.
        words = re.findall(r"/\* (0x[0-9a-f]{16}) \*/", b)
        result[name] = dict(instructions=[(int(a, 16), i.strip()) for a, i in ins], words=words)
    return result


def save(path, rows):
    if rows:
        with path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, list(rows[0]))
            w.writeheader(); w.writerows(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--build", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--profile", type=Path)
    a = p.parse_args(); a.output.mkdir(parents=True, exist_ok=False)
    old = ROOT / "results/windows/20261021-tensorcore-bn64/build02"
    before = json.loads((a.archive / "preflight/sgemm-hashes.json").read_text(encoding="utf-8"))
    sources = []
    for row in before:
        if row["path"].startswith("experiments/"):
            f = ROOT / row["path"]
            sources.append(dict(path=row["path"], before=row["sha256"], after=sha(f), unchanged=sha(f)==row["sha256"]))
    assert all(r["unchanged"] for r in sources), "Existing experiments changed"
    save(a.output / "existing-experiment-hashes.csv", sources)
    current = functions(a.build / "binary.sass"); previous = functions(old / "binary.sass")
    selected = {k: v for k, v in current.items() if ("vitamin_tf32" in k or "bn64_tf32" in k) and "dbf_kernel" in k}
    assert len(selected) == 2, list(selected)
    comparisons, loops = [], []
    for name, values in selected.items():
        now = values["instructions"]; prev = previous[name]
        label = "vitamin_tf32_dbf" if "vitamin_tf32" in name else "tf32_bn64_bk32"
        texts = [t for _, t in now]
        counts = collections.Counter(re.sub(r"^@\S+\s+", "", t).split()[0].split(".")[0] for t in texts)
        comparisons.append(dict(implementation=label, function=name, instruction_count=len(now),
            normalized_instructions_equal=now == prev["instructions"],
            encoded_words_equal=values["words"] == prev["words"], encoded_word_count=len(values["words"]),
            **{op: counts[op] for op in ("HMMA", "LDGSTS", "LDSM", "BAR", "DEPBAR", "LDL", "STL")}))
        if now != prev["instructions"]:
            (a.output / f"{label}.diff").write_text("\n".join(difflib.unified_diff(
                [str(x) for x in prev["instructions"]], [str(x) for x in now])), encoding="utf-8")
        for addr, text in now:
            match = re.search(r"\bBRA\s+(0x[0-9a-f]+)", text)
            if not match or int(match[1],16) >= addr: continue
            body = [(x,t) for x,t in now if int(match[1],16) <= x <= addr]
            if not any("LDGSTS" in t for _,t in body): continue
            last = max(x for x,t in body if "LDGSTS" in t)
            waits = [x for x,t in body if t.startswith("DEPBAR.") and x>last]
            assert waits
            loops.append(dict(implementation=label,start=hex(int(match[1],16)),end=hex(addr),
                instructions=len(body),last_copy=hex(last),wait=hex(waits[0]),
                hmma=sum("HMMA" in t for _,t in body),copies=sum("LDGSTS" in t for _,t in body),
                hmma_between_copy_and_wait=sum("HMMA" in t and last<x<waits[0] for x,t in body)))
            (a.output/f"{label}-main-loop.sass.txt").write_text(
                "\n".join(f"{x:04x}: {t}" for x,t in body)+"\n",encoding="utf-8")
    save(a.output/"sass-comparison.csv", comparisons); save(a.output/"sass-loops.csv",loops)
    result = dict(existing_experiment_files=len(sources), existing_experiments_unchanged=True,
        old_binary_sha256=sha(old/"windows_tensor_bn64_benchmark.exe"),
        new_binary_sha256=sha(a.build/"windows_tensor_bn64_robustness_benchmark.exe"),
        raw_sass_file_equal=sha(old/"binary.sass")==sha(a.build/"binary.sass"),
        all_selected_normalized_equal=all(x["normalized_instructions_equal"] for x in comparisons),
        all_selected_encoded_equal=all(x["encoded_words_equal"] for x in comparisons))
    if a.profile:
        metrics, selected_rows = [], []
        for f in sorted(a.profile.glob("*.csv")):
            lines=f.read_text(encoding="utf-8-sig").splitlines()
            pos=next((i for i,s in enumerate(lines) if s.startswith('"ID",')),None)
            if pos is None: continue
            rows=list(csv.DictReader(io.StringIO("\n".join(lines[pos:]))))
            assert len(rows)==2, f
            row=dict(file=f.name)
            for key,value in rows[1].items():
                if key.startswith(("launch__","sm__","smsp__","dram__","lts__","l1tex__","gpu__")):
                    metrics.append(dict(file=f.name,metric=key,value=value,unit=rows[0][key]))
                    try: row[key]=float(value.replace(",",""))
                    except ValueError: row[key]=value
            selected_rows.append(row)
        save(a.output/"ncu-metrics.csv",metrics);save(a.output/"ncu-selected.csv",selected_rows)
        result['ncu_launches']=len(selected_rows)
    (a.output/"audit.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2))


if __name__ == "__main__": main()
