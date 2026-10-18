#!/usr/bin/env python3
# 【文件导读】验证 grouped-grid 遍历不重不漏
# 给初学者：这些小检查可以帮助理解索引或测试规则；不需要因此先跑一次大型 GPU 性能实验。
# 运行：python3 scripts/verify_grouped_swizzle.py；Python 标准库即可，不启动 GPU。
# 对多种矩形 grid、GROUP_M 和不满尾组枚举逻辑 block 映射；集合大小/边界断言检查双射。
# 输入是文件内固定的教学数据/有限枚举范围；失败以断言或 unittest 非零退出体现。
# 结果打印终端，不生成 benchmark CSV；模型通过不等于设备 FP32、异步调度或吞吐验证。
# 中文总索引：docs/zh/23-代码与脚本使用索引.md；脚本命令默认在仓库根目录执行。

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


# 把一维 block 编号转成输出块坐标；最后不足一组时用实际组高，避免重复或越界。
def grouped_tile(pid: int, grid_m: int, grid_n: int, group_m: int):
    group_size = group_m * grid_n
    group_id = pid // group_size
    first_m = group_id * group_m
    actual_group_m = min(grid_m - first_m, group_m)
    local_pid = pid % group_size
    tile_m = first_m + local_pid % actual_group_m
    tile_n = local_pid // actual_group_m
    return tile_m, tile_n


# 枚举多种网格与组高，要求映射完整且一一对应；它验证任务分派，不测试 GEMM 数值。
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
