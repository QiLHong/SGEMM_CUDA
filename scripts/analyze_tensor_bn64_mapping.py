"""Complete CPU enumeration for the BN64 kernel; no GPU correctness inference."""
import argparse
import collections
import csv
import json
from pathlib import Path


def write(path, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, rows[0].keys())
        w.writeheader()
        w.writerows(rows)


def exact(rows, nr, nc):
    counts = collections.Counter((x["row"], x["col"]) for x in rows)
    assert set(counts) == {(r, c) for r in range(nr) for c in range(nc)}
    assert set(counts.values()) == {1}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    out = p.parse_args().output
    out.mkdir(parents=True, exist_ok=False)
    rows = []
    for tid in range(256):
        warp, lane = divmod(tid, 32)
        for mi in range(4):
            for ni in range(2):
                for reg in range(4):
                    rows.append(dict(tid=tid, warp=warp, lane=lane, mi=mi, ni=ni, reg=reg,
                                     row=(warp//4)*64+mi*16+lane//4+(reg//2)*8,
                                     col=(warp%4)*16+ni*8+(lane%4)*2+reg%2))
    exact(rows, 128, 64)
    write(out / "output-fragments.csv", rows)
    copies = []
    for slot in range(2):
        for slab in range(2):
            for tid in range(256):
                for matrix in ("A", "B"):
                    for copy in range(2 if matrix == "A" else 1):
                        for e in range(4):
                            if matrix == "A":
                                row, col = tid//4+copy*64, tid%4*4+e
                                swcol = col ^ (((row>>1)&3)<<2)
                                gr, gc = row, slab*16+col
                                off = (slot*2+slab)*2048+row*16+swcol
                            else:
                                row, col = tid//16, tid%16*4+e
                                swcol = col ^ ((row&7)<<3)
                                gr, gc = slab*16+row, col
                                off = 8192+(slot*2+slab)*1024+row*64+swcol
                            assert off % 4 == e
                            copies.append(dict(slot=slot, slab=slab, tid=tid, matrix=matrix,
                                               copy=copy, element=e, row=gr, col=gc,
                                               shared_float_offset=off, bank=off%32))
    addresses = collections.Counter(x["shared_float_offset"] for x in copies)
    assert set(addresses) == set(range(12288)) and set(addresses.values()) == {1}
    for slot in range(2):
        for matrix, nr, nc in (("A", 128, 32), ("B", 32, 64)):
            exact([x for x in copies if x["slot"] == slot and x["matrix"] == matrix], nr, nc)
    write(out / "global-to-shared.csv", copies)
    # Operand packets within each warp and K8 step. A ldmatrix lane provides four
    # consecutive floats (8 halfwords); B has two scalar registers per fragment.
    operands = []
    for warp in range(8):
        for step in range(4):
            for lane in range(32):
                for mi in range(4):
                    for e in range(4):
                        row = (warp//4)*64+mi*16+lane%16
                        col = step*8+(lane//16)*4+e
                        operands.append(dict(warp=warp, lane=lane, step=step, matrix="A", fragment=mi,
                                             element=e, row=row, col=col))
                for ni in range(2):
                    for e in range(2):
                        row = step*8+lane%4+e*4
                        col = (warp%4)*16+ni*8+lane//4
                        operands.append(dict(warp=warp, lane=lane, step=step, matrix="B", fragment=ni,
                                             element=e, row=row, col=col))
            for matrix, fragments, nr, nc in (("A", 4, 16, 8), ("B", 2, 8, 8)):
                for frag in range(fragments):
                    data = [x.copy() for x in operands if x["warp"] == warp and x["step"] == step and
                            x["matrix"] == matrix and x["fragment"] == frag]
                    for x in data:
                        if matrix == "A":
                            x["row"] -= (warp//4)*64+frag*16
                            x["col"] -= step*8
                        else:
                            x["row"] -= step*8
                            x["col"] -= (warp%4)*16+frag*8
                    exact(data, nr, nc)
    write(out / "shared-to-register-packets.csv", operands)
    grids = []
    for m, n in ((128, 1024), (1024, 1024), (2048, 2048), (4096, 4096)):
        gx, gy = n//64, m//128
        mapped = [(i%8+i//(8*gy)*8, i//8%gy) for i in range(gx*gy)]
        assert len(set(mapped)) == gx*gy and set(mapped) == {(x,y) for x in range(gx) for y in range(gy)}
        grids.append(dict(m=m, n=n, ctas=gx*gy, status="PASS"))
    result = dict(outputs=8192, output_elements_per_thread=32, shared_bytes=49152,
                  copy_elements=12288, duplicates=0, omissions=0, out_of_bounds=0,
                  copy_16byte_alignment=True, operand_packet_coverage=True, grids=grids,
                  caveat="CPU formula audit; hardware race sanitizer remains unverified")
    (out / "mapping-summary.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
