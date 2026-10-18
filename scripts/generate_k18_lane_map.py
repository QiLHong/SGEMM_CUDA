#!/usr/bin/env python3
# 【文件导读】生成 K18 六个变体的 lane 分工表
# 给初学者：这类脚本像读图纸，不负责测 GPU 跑多快；先准备它要求的二进制或报告，再读输出表。
# 运行：python3 scripts/generate_k18_lane_map.py --output <新的csv文件>。
# Python 标准库即可；不带 --output 时打印 CSV 到终端，已有输出文件会拒绝覆盖。
# 枚举 lane=0..31 的行/列分组，检查每 lane 128 个输出、整个 warp 恰好覆盖 64×64 且无重叠。
# 只证明有限配置的几何覆盖，不验证 GPU 地址对齐、shared 冲突、FP32 误差或性能。
# 初学者先看生成表中 lane=0/1/16 的行列，再回到对应 kernel 的索引公式。
# 中文总索引：docs/zh/23-代码与脚本使用索引.md；脚本命令默认在仓库根目录执行。

"""Generate and validate the six K18 lane-to-output-fragment maps."""

import argparse
import csv
from pathlib import Path


VARIANTS = {
    0: ("baseline-8x4x4", 4, 8, 4),
    1: ("transposed-lane-lattice", 4, 8, 4),
    2: ("tm4-tn8", 4, 4, 8),
    3: ("xor-lane-columns", 4, 8, 4),
    4: ("two-8-wide-n-fragments", 2, 8, 8),
    5: ("one-16-wide-n-fragment", 1, 8, 16),
}


# 输入变体号与 lane，返回这一线程的行组/列组；注意完整输出是行集合与列集合的笛卡尔积。
def lane_fragment(variant: int, lane: int):
    name, wniter, tm, tn = VARIANTS[variant]
    wsubn = 64 // wniter
    row_groups = 64 // tm
    col_groups = wsubn // tn
    assert row_groups * col_groups == 32
    if variant == 1:
        lane_m, lane_n = lane % row_groups, lane // row_groups
    else:
        lane_m, lane_n = lane // col_groups, lane % col_groups
        if variant == 3:
            lane_n ^= lane_m & (col_groups - 1)
    rows = tuple(range(lane_m * tm, (lane_m + 1) * tm))
    col_ranges = tuple(
        (sub * wsubn + lane_n * tn, sub * wsubn + lane_n * tn + tn - 1)
        for sub in range(wniter)
    )
    coords = {(row, col) for row in rows for begin, end in col_ranges
              for col in range(begin, end + 1)}
    return name, tm, tn, wniter, lane_m, lane_n, rows, col_ranges, coords


# 枚举 32 个 lane，用集合断言无漏格/重叠，再写 CSV；覆盖检查和 GPU 正确性是两步。
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = []
    for variant in VARIANTS:
        covered = set()
        for lane in range(32):
            name, tm, tn, wniter, lane_m, lane_n, m_rows, n_ranges, coords = (
                lane_fragment(variant, lane)
            )
            if covered & coords:
                raise RuntimeError(f"variant {variant}: overlapping lane {lane}")
            covered |= coords
            rows.append({
                "variant": variant,
                "name": name,
                "lane": lane,
                "lane_m": lane_m,
                "lane_n": lane_n,
                "TM": tm,
                "TN": tn,
                "WNITER": wniter,
                "warp_rows": f"{m_rows[0]}-{m_rows[-1]}",
                "warp_column_ranges": ";".join(f"{a}-{b}" for a, b in n_ranges),
                "elements": len(coords),
            })
        expected = {(m, n) for m in range(64) for n in range(64)}
        if covered != expected:
            raise RuntimeError(f"variant {variant}: does not cover 64x64 exactly")

    fields = list(rows[0])
    if args.output:
        if args.output.exists():
            raise SystemExit(f"refusing to overwrite {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        print(f"validated 6 layouts; wrote {args.output}")
    else:
        writer = csv.DictWriter(__import__("sys").stdout, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
