#!/usr/bin/env python3
"""验证 K16 的 A shared-memory XOR 映射保持每行双射且可逆。"""

BM = 128
BK = 16
MASKS = (1, 3, 7)
SHIFTS = (1, 2, 3)


def swizzle(k: int, m: int, mask: int, shift: int) -> int:
    return m ^ ((k & mask) << shift)


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
