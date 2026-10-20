"""CPU enumeration of the exact four selected upstream MMA mappings."""
import argparse
import collections
import csv
import json
from pathlib import Path


def save(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, rows[0].keys())
        w.writeheader()
        w.writerows(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    out = p.parse_args().output
    out.mkdir(parents=True, exist_ok=False)
    rows = []
    for tid in range(256):
        warp, lane = divmod(tid, 32)
        for mi in range(4):
            for ni in range(4):
                for reg in range(4):
                    row = (warp // 4) * 64 + mi * 16 + lane // 4 + (reg // 2) * 8
                    col = (warp % 4) * 32 + ni * 8 + (lane % 4) * 2 + reg % 2
                    rows.append(dict(tid=tid, warp=warp, lane=lane, m_fragment=mi, n_fragment=ni,
                                     accumulator_register=reg, row=row, col=col))
    counts = collections.Counter((r["row"], r["col"]) for r in rows)
    assert len(counts) == 128*128 and set(counts.values()) == {1}
    assert all(0 <= r < 128 and 0 <= c < 128 for r, c in counts)
    save(out / "output-fragments.csv", rows)
    loads = []
    for kind, bk, vec, browdiv, browoff in (("tf32", 16, 4, 32, 8), ("fp16", 32, 8, 16, 16)):
        for tid in range(256):
            for matrix in ("A", "B"):
                for copy in range(2):
                    for element in range(vec):
                        if matrix == "A":
                            row = tid//4 + copy*64
                            col = tid % 4 * vec + element
                            swcol = col ^ (((row >> 1) & 3) * vec)
                        else:
                            row = tid//browdiv + copy*browoff
                            col = tid % browdiv * vec + element
                            swcol = col ^ ((row & 7) << 3)
                        loads.append(dict(precision=kind, matrix=matrix, tid=tid, copy=copy,
                                          element=element, row=row, col=col, shared_col=swcol))
        for matrix in ("A", "B"):
            subset = [x for x in loads if x["precision"] == kind and x["matrix"] == matrix]
            nr, nc = (128, bk) if matrix == "A" else (bk, 128)
            for column in ("col", "shared_col"):
                counts = collections.Counter((x["row"], x[column]) for x in subset)
                assert len(counts) == nr*nc and set(counts.values()) == {1}
                assert all(0 <= r < nr and 0 <= c < nc for r, c in counts)
    save(out / "global-to-shared.csv", loads)
    # HGEMM RW最后由不同线程搬运连续8个half；ownership与accumulator线程不同。
    stores = []
    for tid in range(256):
        for step in range(8):
            for e in range(8):
                row, col = divmod((step*256+tid)*8+e, 128)
                stores.append(dict(tid=tid, step=step, element=e, row=row, col=col,
                                   shared_col=col ^ ((row & 7) << 3)))
    counts = collections.Counter((x["row"], x["col"]) for x in stores)
    assert len(counts) == 16384 and set(counts.values()) == {1}
    save(out / "fp16-rw-shared-to-global.csv", stores)
    grids = []
    for m, n in ((128, 1024), (1024, 1024), (2048, 2048), (4096, 4096)):
        gx, gy = n//128, m//128
        mapped = [(l % 8 + l//(8*gy)*8, l//8 % gy) for l in range(gx*gy)]
        assert set(mapped) == {(x, y) for x in range(gx) for y in range(gy)}
        assert len(set(mapped)) == len(mapped)
        grids.append(dict(m=m, n=n, ctas=len(mapped), status="PASS"))
    result = dict(outputs=16384, duplicates=0, omissions=0, out_of_bounds=0,
                  applies_to=["tf32_single", "tf32_dbf", "fp16_single", "fp16_dbf_rw"],
                  grid_checks=grids, caveat="CPU formulas audit coverage; numerical GPU checks remain necessary")
    (out / "mapping-summary.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
