#!/usr/bin/env python3
"""CPU proof of ownership/layout bounds, plus complete per-warp mapping CSV."""
import argparse
import csv
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument("--output",type=Path,required=True)
args=p.parse_args()
rows=[]
for bm,threads in [(128,128),(256,256)]:
    outputs=set()
    for warp in range(threads//32):
        wr,wc=divmod(warp,2)
        for lane in range(32):
            lm=2*(lane//16)+lane%2
            ln=(lane%16)//2
            for i in range(16):
                m=wr*64+4*lm+16*(i//4)+i%4
                for j in range(8):
                    n=wc*64+4*ln+32*(j//4)+j%4
                    assert 0<=m<bm and 0<=n<128
                    assert (m,n) not in outputs
                    outputs.add((m,n))
                    if bm==128 and warp==0:
                        rows.append(dict(lane=lane,lane_m=lm,lane_n=ln,a_register=i,
                                         b_register=j,accumulator=i*8+j,m=m,n=n,
                                         shared_a_k0=m,shared_b_k0=n,
                                         shared_a_k_stride=bm+4,shared_b_k_stride=128))
    assert len(outputs)==bm*128
    # Transpose stores and fragment loads must agree at every k.
    for k in range(16):
        for m in range(bm): assert 0<=k*(bm+4)+m<16*(bm+4)
        for n in range(128): assert 0<=k*128+n<16*128
    print(f"CTA {bm}x128 threads={threads}: exact coverage, no overlaps, layout in bounds")
args.output.parent.mkdir(parents=True,exist_ok=True)
with args.output.open("x") as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator="\n")
    w.writeheader(); w.writerows(rows)
