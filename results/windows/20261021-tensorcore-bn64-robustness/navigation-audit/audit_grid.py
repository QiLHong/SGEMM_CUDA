"""CPU-only audit of preserved W8 device indexing; never launches a kernel.

Grid bijection plus one CTA's exact fragment coverage proves coordinate coverage.
Copy alignment is checked by additive address components, not by executing memory
accesses. Bounds are checked on local vector ends and every CTA / K-tile origin.
This does not prove GPU synchronization or race freedom.
"""
from pathlib import Path
import collections
import csv
import hashlib
import json

OUT = Path(__file__).resolve().parent
REPO = OUT.parents[3]


def save(name, rows):
    with (OUT / name).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    variants = [
        ("vitamin_tf32_dbf", 128, 16, "experiments/tensorcore/vendor/vitamin_tf32.cuh"),
        ("tf32_bn64_bk32", 64, 32, "experiments/tensorcore-bn64/tf32_bn64.cuh"),
    ]
    grids, fragments, copies, summaries = [], [], [], []
    for name, bn, bk, source in variants:
        text = (REPO / source).read_text(encoding="utf-8-sig")
        for expression in (
            "int linear_id = blockIdx.y * gridDim.x + blockIdx.x;",
            "const int SWIZZLE_W = 8;",
            "int bx = (linear_id % SWIZZLE_W) + (linear_id / (SWIZZLE_W * gridDim.y)) * SWIZZLE_W;",
            "int by = (linear_id / SWIZZLE_W) % gridDim.y;",
        ):
            assert expression in text, (source, expression)
        coords = []
        for tid in range(256):
            warp, lane = divmod(tid, 32)
            for mi in range(4):
                for ni in range(bn // 32):
                    for reg in range(4):
                        row = (warp // 4) * 64 + mi * 16 + lane // 4 + (reg // 2) * 8
                        col = (warp % 4) * (bn // 4) + ni * 8 + (lane % 4) * 2 + reg % 2
                        coords.append((row, col))
                        fragments.append(dict(implementation=name, tid=tid, warp=warp,
                                              lane=lane, mi=mi, ni=ni, reg=reg, row=row, col=col))
        counts = collections.Counter(coords)
        assert len(counts) == 128 * bn and set(counts.values()) == {1}
        assert set(counts) == {(r, c) for r in range(128) for c in range(bn)}
        local = []
        for slab in range(bk // 16):
            for tid in range(256):
                for part in range(2):
                    local.append(("A", tid // 4 + part * 64, slab * 16 + tid % 4 * 4))
                for part in range(2 if bn == 128 else 1):
                    local.append(("B", slab * 16 + tid // (bn // 4) + part * 8, tid % (bn // 4) * 4))
        for matrix, rows, cols in (("A", 128, bk), ("B", bk, bn)):
            vectors = [(r, c) for mat, r, c in local if mat == matrix]
            elements = [(r, c + e) for r, c in vectors for e in range(4)]
            assert set(elements) == {(r, c) for r in range(rows) for c in range(cols)}
            assert len(elements) == rows * cols
            assert all(c % 4 == 0 and 0 <= r < rows and c + 3 < cols for r, c in vectors)
        for n in (1024, 2048, 3072, 4096):
            assert n % 128 == n % 1024 == n % 32 == 0
            gx, gy = n // bn, n // 128
            mapped = []
            for linear in range(gx * gy):
                bx = linear % 8 + linear // (8 * gy) * 8
                by = linear // 8 % gy
                mapped.append((bx, by))
                assert bx * bn % 4 == by * 128 * n % 4 == 0
                assert (bx + 1) * bn <= n and (by + 1) * 128 <= n
                grids.append(dict(implementation=name, n=n, block_x=linear % gx,
                                  block_y=linear // gx, linear_id=linear,
                                  mapped_bx=bx, mapped_by=by,
                                  output_row_begin=by * 128, output_col_begin=bx * bn,
                                  output_row_end=by * 128 + 127, output_col_end=bx * bn + bn - 1))
            assert set(mapped) == {(x, y) for y in range(gy) for x in range(gx)}
            assert len(mapped) == len(set(mapped))
            for kk in range(0, n, bk):
                assert kk % 4 == kk * n % 4 == 0 and kk + bk <= n
            # Global address = aligned CTA origin + aligned K tile origin +
            # local row * aligned leading dimension + aligned vector column.
            assert n % 4 == 0
            for mat, r, c in local:
                assert (r * n + c) % 4 == 0
            copies.append(dict(implementation=name, n=n, lda=n, ldb=n,
                               local_vectors_per_cta_ktile=len(local), k_tiles=n // bk,
                               all_cta_origins_aligned_16bytes=True,
                               all_ktile_origins_aligned_16bytes=True,
                               all_local_vector_offsets_aligned_16bytes=True,
                               local_vector_bounds=True, full_tile_bounds=True))
            summaries.append(dict(implementation=name, n=n, grid_x=gx, grid_y=gy,
                                  ctas=gx * gy, k_tiles=n // bk, swizzle_width=8,
                                  outputs_per_cta=128 * bn, duplicates=0,
                                  omissions=0, out_of_bounds=0,
                                  source=source,
                                  source_sha256=hashlib.sha256((REPO / source).read_bytes()).hexdigest()))
    save("grid-ctas.csv", grids)
    save("output-fragments.csv", fragments)
    save("copy-alignment-components.csv", copies)
    (OUT / "grid-summary.json").write_text(json.dumps(dict(
        status="PASS", grids=summaries, grid_rows=len(grids), fragment_rows=len(fragments),
        copy_method="All CTA and K-tile origins, full local copy vectors, additive modulo-4 proof; cudaMalloc base alignment is required",
        scope="CPU formula / bounds audit; no claim of hardware race freedom or GPU execution"
    ), indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dict(status="PASS", grid_rows=len(grids), fragment_rows=len(fragments))))


if __name__ == "__main__":
    main()
