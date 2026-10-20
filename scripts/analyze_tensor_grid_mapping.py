"""Enumerate block permutation; combine with unchanged complete-CTA mapping."""
import argparse
import collections
import csv
import json
import subprocess
import sys
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--output", type=Path, required=True)
args = p.parse_args()
out = args.output
out.mkdir(parents=True, exist_ok=False)
subprocess.run([sys.executable, str(Path(__file__).with_name("analyze_tensor_mapping.py")),
                "--output", str(out / "unchanged-cta")], check=True)
rows, checks = [], []
for m, n in ((128, 1024), (1024, 1024), (2048, 2048), (4096, 4096)):
    gx, gy = n//128, m//128
    for width in (1, 2, 4, 8):
        count = collections.Counter()
        for linear in range(gx*gy):
            bx = linear % width + linear//(width*gy)*width
            by = linear//width % gy
            count[bx, by] += 1
            rows.append(dict(m=m, n=n, width=width, linear=linear,
                             physical_block_x=linear % gx, physical_block_y=linear//gx,
                             output_block_x=bx, output_block_y=by))
        assert set(count) == {(x, y) for x in range(gx) for y in range(gy)}
        assert set(count.values()) == {1}
        checks.append(dict(m=m, n=n, width=width, ctas=gx*gy,
                           duplicate=0, missing=0, out_of_bounds=0))
with (out / "grid-coordinates.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, rows[0].keys())
    w.writeheader()
    w.writerows(rows)
(out / "grid-summary.json").write_text(json.dumps(checks, indent=2)+"\n")
print(f"Verified {len(checks)} grid cases, {len(rows)} CTA coordinates")
