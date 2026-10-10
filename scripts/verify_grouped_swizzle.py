#!/usr/bin/env python3
"""验证 K15 grouped-grid 映射对每个 tile 恰好覆盖一次。"""

GROUPS = (1, 2, 4, 8, 16)
GRIDS = (
    (1, 1),
    (2, 2),
    (3, 5),
    (7, 3),
    (17, 7),
    (32, 32),
    (33, 32),
)


def grouped_tile(pid: int, grid_m: int, grid_n: int, group_m: int):
    group_size = group_m * grid_n
    group_id = pid // group_size
    first_m = group_id * group_m
    actual_group_m = min(grid_m - first_m, group_m)
    local_pid = pid % group_size
    tile_m = first_m + local_pid % actual_group_m
    tile_n = local_pid // actual_group_m
    return tile_m, tile_n


def main() -> None:
    for grid_m, grid_n in GRIDS:
        expected = {(m, n) for m in range(grid_m) for n in range(grid_n)}
        for group_m in GROUPS:
            mapped = [
                grouped_tile(pid, grid_m, grid_n, group_m)
                for pid in range(grid_m * grid_n)
            ]
            assert len(mapped) == len(set(mapped)), (
                f"duplicate tile: grid={grid_m}x{grid_n}, group={group_m}"
            )
            assert set(mapped) == expected, (
                f"missing/out-of-range tile: grid={grid_m}x{grid_n}, "
                f"group={group_m}"
            )
    print(f"PASS: {len(GRIDS)} grids x {len(GROUPS)} GROUP_M values")


if __name__ == "__main__":
    main()
