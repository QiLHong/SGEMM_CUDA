"""Freeze this dated experiment without modifying old sources or results."""
import ast
import csv
import hashlib
import json
import re
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/windows/20261021-tensorcore-ktile"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write_csv(path, rows):
    with path.open("x", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    assert not (OUT / "archive-sha256.csv").exists(), "Already frozen; do not overwrite"
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip()
    assert head == "b718c7bfe3577822e3335b215e386d2c57c420eb"
    assert subprocess.check_output(["git", "diff", "--binary"], cwd=ROOT) == (OUT / "preflight/working-before.patch").read_bytes()
    assert not subprocess.check_output(["git", "diff", "--cached"], cwd=ROOT)
    upstream = ROOT.parent / "vitamin-cuda"
    assert not subprocess.check_output(["git", "status", "--porcelain"], cwd=upstream)
    assert subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=upstream).decode().strip() == "520f97093a2abc239977be83d244b327d0e2e2ef"
    for stem in ("source-before", "history-before"):
        with (OUT / f"preflight/{stem}.csv").open(encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                assert sha((ROOT / row["path"]).read_bytes()) == row["sha256"].lower(), row["path"]
    scripts = [ROOT / "scripts" / name for name in
               ("prepare_tensor_ktile.py", "analyze_tensor_ktile_mapping.py", "summarize_tensor_ktile.py",
                "audit_tensor_ktile.py", "finalize_tensor_ktile.py")]
    scripts += list((ROOT / "experiments/tensorcore-ktile").glob("derive_*.py"))
    for path in scripts:
        ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    links = 0
    docs = [ROOT / "experiments/tensorcore-ktile/README.md", OUT / "README.md"]
    docs += list((ROOT / "docs/zh").glob("30-TensorCore-*.md"))
    assert len(docs) == 3
    for path in docs:
        source = path.read_text(encoding="utf-8")
        assert source.count("```") % 2 == 0, path
        for target in re.findall(r"\]\(([^)]+)\)", source):
            if "://" in target or target.startswith("#"):
                continue
            assert (path.parent / target.split("#")[0]).exists(), (path, target)
            links += 1
    manifest = json.loads((OUT / "build01/verified-build.json").read_text(encoding="utf-8-sig"))
    assert sha((OUT / "build01/windows_tensor_ktile_benchmark.exe").read_bytes()) == manifest["binary_sha256"].lower()
    assert sha((ROOT / "experiments/tensorcore-ktile/benchmark.cu").read_bytes()) == manifest["source_sha256"].lower()
    analyses = [json.loads((OUT / sub / "audit.json").read_text(encoding="utf-8"))
                for sub in ("analysis01", "confirm-analysis02")]
    assert sum(x["events"] for x in analyses) == 4410
    assert sum(x["bitwise_checks"] for x in analyses) == 156
    assert all(x["bitwise_mismatches"] == x["nonfinite"] == x["removed_events"] == 0 for x in analyses)
    names = subprocess.check_output(["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"], cwd=ROOT)
    paths = sorted({p.decode("utf-8") for p in names.split(b"\0") if p})
    paths = [p for p in paths if not p.startswith(("results/", "build")) and (ROOT / p).is_file()]
    rows = []
    with zipfile.ZipFile(OUT / "source-as-delivered.zip", "x", zipfile.ZIP_DEFLATED) as z:
        for name in paths:
            data = (ROOT / name).read_bytes()
            z.writestr(name, data)
            rows.append(dict(path=name, sha256=sha(data)))
    write_csv(OUT / "source-as-delivered-sha256.csv", rows)
    with zipfile.ZipFile(OUT / "source-as-delivered.zip") as z:
        assert set(z.namelist()) == set(paths)
        assert all(sha(z.read(row["path"])) == row["sha256"] for row in rows)
    subprocess.run(["git", "diff", "--check"], cwd=ROOT, check=True)
    (OUT / "git-status-final.txt").write_bytes(subprocess.check_output(["git", "status", "--short"], cwd=ROOT))
    result = dict(head=head, old_tracked_diff_equal=True, upstream_unchanged=True,
                  python_syntax_files=len(scripts), checked_doc_links=links,
                  sources_in_zip=len(rows), formal_events=4410, bitwise_checks=156,
                  binary_hash_verified=True, historical_hashes_verified=True,
                  no_commit_or_push=True)
    (OUT / "delivery-check.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    files = sorted(p for p in OUT.rglob("*") if p.is_file())
    hashes = [dict(path=p.relative_to(OUT).as_posix(), sha256=sha(p.read_bytes())) for p in files]
    write_csv(OUT / "archive-sha256.csv", hashes)
    assert all(sha((OUT / row["path"]).read_bytes()) == row["sha256"] for row in hashes)
    print(json.dumps(dict(**result, archive_entries=len(hashes), archive_manifest="UTF-8")))


if __name__ == "__main__":
    main()
