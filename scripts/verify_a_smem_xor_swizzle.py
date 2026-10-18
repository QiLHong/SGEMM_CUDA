#!/usr/bin/env python3
# 【文件导读】验证 A shared XOR 布局的双射与边界
# 给初学者：这些小检查可以帮助理解索引或测试规则；不需要因此先跑一次大型 GPU 性能实验。
# 运行：python3 scripts/verify_a_smem_xor_swizzle.py；Python 标准库即可，不启动 GPU。
# 固定 BM=128、BK=16，枚举 mask=1/3/7 × shift=1/2/3；验证 m 映射不越界、无重复且 XOR 自逆，不检查 float4 对齐。
# 输入是文件内固定的教学数据/有限枚举范围；失败以断言或 unittest 非零退出体现。
# 结果打印终端，不生成 benchmark CSV；模型通过不等于设备 FP32、异步调度或吞吐验证。
# 中文总索引：docs/zh/23-代码与脚本使用索引.md；脚本命令默认在仓库根目录执行。

"""验证 K16 的 A shared-memory XOR 映射保持每行双射且可逆。"""

BM = 128
BK = 16
MASKS = (1, 3, 7)
SHIFTS = (1, 2, 3)


# Python 写出与设备端一致的 XOR 映射；同一 k 的置换自逆，但还需检查输出范围。
def swizzle(k: int, m: int, mask: int, shift: int) -> int:
    return m ^ ((k & mask) << shift)


# 穷举九组 mask/shift 的合法范围，验证双射/自逆；不会检测实际编译对齐或 bank 吞吐。
def main() -> None:
    for mask in MASKS:
        for shift in SHIFTS:
            for k in range(BK):
                mapped = [swizzle(k, m, mask, shift) for m in range(BM)]
                assert min(mapped) >= 0 and max(mapped) < BM
                assert len(set(mapped)) == BM
                for m, physical_m in enumerate(mapped):
                    assert swizzle(k, physical_m, mask, shift) == m
    print(f"PASS: {len(MASKS) * len(SHIFTS)} configs x {BK} rows")


if __name__ == "__main__":
    main()
