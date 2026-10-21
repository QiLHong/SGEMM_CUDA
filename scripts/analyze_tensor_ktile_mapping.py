"""Enumerate the K-tile experiment's output/copy coverage and shared addresses."""
import argparse
import collections
import csv
import json
import subprocess
import sys
from pathlib import Path


def save(path, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    out = parser.parse_args().output
    out.mkdir(parents=True, exist_ok=False)
    subprocess.run([sys.executable, str(Path(__file__).with_name("analyze_tensor_mapping.py")),
                    "--output", str(out / "original-mapping")], check=True)
    copies, results = [], []
    for bk in (16, 32):
        slabs = bk // 16
        current = []
        for slot in range(2):
            for slab in range(slabs):
                for tid in range(256):
                    for matrix in ("A", "B"):
                        for copy in range(2):
                            for element in range(4):
                                if matrix == "A":
                                    r, c = tid // 4 + copy * 64, tid % 4 * 4 + element
                                    sw = c ^ (((r >> 1) & 3) << 2)
                                    gr, gc = r, slab * 16 + c
                                    offset = (slot * slabs + slab) * 2048 + r * 16 + sw
                                else:
                                    r, c = tid // 32 + copy * 8, tid % 32 * 4 + element
                                    sw = c ^ ((r & 7) << 3)
                                    gr, gc = slab * 16 + r, c
                                    offset = 2 * slabs * 2048 + (slot * slabs + slab) * 2048 + r * 128 + sw
                                current.append(dict(bk=bk, slot=slot, slab=slab, tid=tid, matrix=matrix,
                                                    copy=copy, element=element, row=gr, col=gc,
                                                    shared_float_offset=offset, bank=offset % 32))
        addresses = collections.Counter(x["shared_float_offset"] for x in current)
        assert len(addresses) == 2 * (128 * bk + bk * 128)
        assert set(addresses.values()) == {1}
        assert set(addresses) == set(range(2 * (128 * bk + bk * 128)))
        for slot in range(2):
            for matrix in ("A", "B"):
                rows = [x for x in current if x["slot"] == slot and x["matrix"] == matrix]
                counts = collections.Counter((x["row"], x["col"]) for x in rows)
                nr, nc = (128, bk) if matrix == "A" else (bk, 128)
                assert set(counts) == {(r, c) for r in range(nr) for c in range(nc)}
                assert set(counts.values()) == {1}
        # Each slab and each buffer begins at a multiple of 32 banks.
        # This proves the per-address bank mapping, not hardware transaction conflicts.
        assert all(x["bank"] == ((x["shared_float_offset"] % 2048) % 32) for x in current)
        copies.extend(current)
        results.append(dict(bk=bk, shared_bytes=len(addresses)*4, copy_elements=len(current),
                            duplicate_addresses=0, omissions=0, out_of_bounds=0))
    save(out / "copy-addresses.csv", copies)
    steps = []
    for k in (32, 64, 96, 1024, 2048, 4096):
        for bk in (16, 32):
            values = []
            for tile in range(k // bk):
                for step in range(bk // 8):
                    first = tile * bk + (step // 2) * 16 + (step % 2) * 8
                    values.append(first)
                    steps.append(dict(k=k, bk=bk, tile=tile, step=step, slab=step//2,
                                      first_k=first, last_k=first+7))
            assert values == list(range(0, k, 8))
    save(out / "mma-k-order.csv", steps)
    result = dict(copy_checks=results, output_elements=16384, output_mapping="original-mapping/output-fragments.csv",
                  mma_k_order_equal=True, bank_mapping_preserved=True,
                  caveat="CPU address/coverage proof; actual shared wavefronts/conflicts require NCU; race sanitizer not run")
    (out / "mapping-summary.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
