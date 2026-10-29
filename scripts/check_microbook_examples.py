#!/usr/bin/env python3
# 【文件导读】微教材的 CPU 手算与坐标练习
# 给初学者：这些小检查可以帮助理解索引或测试规则；不需要因此先跑一次大型 GPU 性能实验。
# 运行：python3 scripts/check_microbook_examples.py；Python 标准库即可，不启动 GPU。
# 验证小矩阵、行主序地址、K1/K2/K3/K5 线程追踪、K10/K30 输出覆盖及双槽顺序模型。
# 输入是文件内固定的教学数据/有限枚举范围；失败以断言或 unittest 非零退出体现。
# 结果打印终端，不生成 benchmark CSV；模型通过不等于设备 FP32、异步调度或吞吐验证。
# 中文总索引：docs/zh/23-代码与脚本使用索引.md；脚本命令默认在仓库根目录执行。

"""仅在 CPU 上运行的小册子配套练习，不执行 CUDA，也不测量性能。

Python 算术用于核对教学示例，不模拟设备上的 FP32 舍入行为。
所有数据都在内存中，运行不会改写历史实验结果。"""
import unittest


# 在 CPU 用小整数矩阵验证点积或分块循环是否算同一件事；不是 GPU 性能模型。
def matmul(a, b, bk=None):
    """以行主序列表模拟矩阵；分块遍历 K 时仍保持原来的累加顺序。"""
    m, k, n = len(a), len(a[0]), len(b[0])
    assert len(b) == k
    assert all(len(row) == k for row in a)
    assert all(len(row) == n for row in b)
    bk = k if bk is None else bk
    assert bk > 0
    out = [[0 for _ in range(n)] for _ in range(m)]
    for start in range(0, k, bk):
        for row in range(m):
            for col in range(n):
                for dot in range(start, min(start + bk, k)):
                    out[row][col] += a[row][dot] * b[dot][col]
    return out


# 用 Python 重现 K10/K30 的线程到输出坐标公式；便于检查每格是否恰好分给一个线程。
def output_coordinates(mode, tid):
    """只覆盖小册子使用的 K10/K30 特定 128 线程块映射。"""
    warp, lane = divmod(tid, 32)
    if mode == "k10":
        row = 64 * (warp // 2) + 8 * (lane // 4)
        col = 64 * (warp % 2) + 4 * (lane % 4)
        return [(row + i, col + 16 * (j // 4) + j % 4)
                for i in range(8) for j in range(16)]
    assert mode == "k30"
    row = 64 * (warp // 2) + 4 * (2 * (lane // 16) + lane % 2)
    col = 64 * (warp % 2) + 4 * ((lane % 16) // 2)
    return [(row + 16 * (i // 4) + i % 4, col + 32 * (j // 4) + j % 4)
            for i in range(16) for j in range(8)]


class MicrobookExamples(unittest.TestCase):
    def test_small_gemm(self):
        a = [[1, 2, 3], [4, 5, 6]]
        b = [[7, 8], [9, 10], [11, 12]]
        product = matmul(a, b)
        self.assertEqual(product, [[58, 64], [139, 154]])
        self.assertEqual([[0.5 * x + 3 * 1 for x in row] for row in product],
                         [[32, 35], [72.5, 80]])

    def test_row_major(self):
        a = [1, 2, 3, 4, 5, 6]
        b = [7, 8, 9, 10, 11, 12]
        self.assertEqual(a[1 * 3 + 2], 6)
        self.assertEqual([1 * 3 + k for k in range(3)], [3, 4, 5])
        self.assertEqual([k * 2 + 0 for k in range(3)], [0, 2, 4])
        self.assertEqual(sum(a[3 + k] * b[k * 2] for k in range(3)), 139)
        self.assertEqual((2 * 5 + 4) * 4, 56)

    def test_thread_mapping(self):
        self.assertEqual(3 * 128 + 70, 454)
        self.assertEqual(divmod(70, 32), (2, 6))
        self.assertEqual(divmod(3 + 16 * 1, 32), (0, 19))
        self.assertEqual([divmod(tid, 2) for tid in range(4)],
                         [(0, 0), (0, 1), (1, 0), (1, 1)])

    def test_k_tiling(self):
        a = [[1, 2, 3], [4, 5, 6]]
        b = [[7, 8], [9, 10], [11, 12]]
        self.assertEqual(matmul([r[:2] for r in a], b[:2]), [[25, 28], [73, 82]])
        self.assertEqual(matmul([r[2:] for r in a], b[2:]), [[33, 36], [66, 72]])
        # 这里的矩形和尾块用例只验证 CPU 教学模型，不表示 K30 支持这些形状。
        for m, k, n in [(2, 3, 2), (3, 5, 7), (4, 4, 4)]:
            a = [[(r * k + c) % 7 - 3 for c in range(k)] for r in range(m)]
            b = [[(r * n + c) % 5 - 2 for c in range(n)] for r in range(k)]
            for bk in [1, 2, 3, 8]:
                with self.subTest(m=m, k=k, n=n, bk=bk):
                    self.assertEqual(matmul(a, b, bk), matmul(a, b))

    def test_outer_product(self):
        a, b = [2, 3], [4, 5, 6]
        self.assertEqual([[x * y for y in b] for x in a], [[8, 10, 12], [12, 15, 18]])
        self.assertEqual(len(a) + len(b), 5)
        self.assertEqual(len(a) * len(b), 6)
        self.assertEqual(4 + 8, 12)
        self.assertEqual(2 * 4 * 8, 64)

    def test_output_coverage(self):
        expected = {(m, n) for m in range(128) for n in range(128)}
        for mode in ["k10", "k30"]:
            points = []
            for tid in range(128):
                lane_points = output_coordinates(mode, tid)
                self.assertEqual(len(set(lane_points)), 128)
                points.extend(lane_points)
            self.assertEqual(len(points), len(expected))
            self.assertEqual(set(points), expected)
        points = output_coordinates("k30", 31)
        self.assertEqual(max(m for m, _ in points), 63)
        self.assertEqual(max(n for _, n in points), 63)
        self.assertEqual(set(output_coordinates("k30", 32)),
                         {(m, n + 64) for m, n in output_coordinates("k30", 0)})
        rows = list(range(4)) + list(range(16, 20)) + list(range(32, 36)) + list(range(48, 52))
        self.assertEqual(set(output_coordinates("k30", 0)),
                         {(m, n) for m in rows for n in list(range(4)) + list(range(32, 36))})

    def test_scalar_bank_model(self):
        offsets32 = [lane * 32 for lane in range(32)]
        offsets33 = [lane * 33 for lane in range(32)]
        self.assertEqual({x % 32 for x in offsets32}, {0})
        self.assertEqual({x % 32 for x in offsets33}, set(range(32)))
        self.assertEqual(len(set(offsets32)), 32)  # 不同的存储字落在同一个 bank，会形成共享内存 bank 冲突。
        self.assertEqual(len(set([0] * 32)), 1)   # 多个线程读取同一个存储字：这是读取广播的情况。
        self.assertNotEqual((1 * 4) % 16, 0)
        self.assertEqual((4 * 4) % 16, 0)

    def test_k1_k2_block_trace(self):
        # 第 2 章：取线程块 (1,2)、BLOCKSIZE=32，对照两种 lane 映射。
        k1 = [(32 + tid % 32, 64 + tid // 32) for tid in (0, 1, 31, 32)]
        k2 = [(32 + tid // 32, 64 + tid % 32) for tid in (0, 1, 31, 32)]
        self.assertEqual(k1, [(32, 64), (33, 64), (63, 64), (32, 65)])
        self.assertEqual(k2, [(32, 64), (32, 65), (32, 95), (33, 64)])
        self.assertEqual((4096 // 128)**2 * 128 * 128, 4096**2)

    def test_k3_cooperative_load_trace(self):
        # 第 3 章：取第一个 K 分块、tid=33、线程块 (1,2)，矩阵维度 M=N=K=128。
        row, col = divmod(33, 32)
        self.assertEqual((row, col), (1, 1))
        self.assertEqual(32 * 128 + row * 128 + col, 4225)  # A[33,1]
        self.assertEqual(64 + row * 128 + col, 193)       # B[1,65]
        self.assertEqual(32 * 128 + 64 + row * 128 + col, 4289)
        a_producer, b_producer = row * 32 + 3, 3 * 32 + col
        self.assertEqual((a_producer, b_producer), (35, 97))
        self.assertEqual(divmod(a_producer, 32), (1, 3))
        self.assertEqual(divmod(b_producer, 32), (3, 1))

    def test_k5_loader_vs_output_trace(self):
        # 第 4 章：线程块分块为 128×128×8；每线程计算 8×8 个输出，共 256 个线程。
        tid = 17
        tr, tc = divmod(tid, 128 // 8)
        self.assertEqual((tr, tc), (1, 1))
        a_rows = [tid // 8 + offset for offset in range(0, 128, 256 // 8)]
        b_rows = [tid // 128 + offset for offset in range(0, 8, 256 // 128)]
        self.assertEqual(a_rows, [2, 34, 66, 98])
        self.assertEqual(b_rows, [0, 2, 4, 6])
        self.assertEqual((tid % 8, tid % 128), (1, 17))
        i, j = 2, 3
        row, col = tr * 8 + i, 128 + tc * 8 + j
        self.assertEqual((i * 8 + j, row, col, row * 256 + col), (19, 10, 139, 2699))
        self.assertEqual(8 + 5, 13)  # 查看第二个 K 分块中 dotIdx=5 对应的读取位置。

    def test_k30_global_to_fragment_trace(self):
        # 第 5 章：枚举一个 128×128×16 分块中全部逻辑写入坐标。
        # 这里只检查坐标公式，不检查 CUDA 地址对齐或设备执行结果。
        a_stores, b_stores = {}, {}
        for tid in range(128):
            ar, ak = tid // 4, (tid % 4) * 4
            bk, bn = tid // 32, (tid % 32) * 4
            for t in range(4):
                for component in range(4):
                    ac = (ar + t * 32, ak + component)
                    bc = (bk + t * 4, bn + component)
                    self.assertNotIn(ac, a_stores)
                    self.assertNotIn(bc, b_stores)
                    a_stores[ac] = (tid, t, component)
                    b_stores[bc] = (tid, t, component)
        self.assertEqual(set(a_stores), {(m, k) for m in range(128) for k in range(16)})
        self.assertEqual(set(b_stores), {(k, n) for k in range(16) for n in range(128)})
        self.assertEqual(a_stores[17, 2], (68, 0, 2))
        self.assertEqual(b_stores[2, 38], (73, 0, 2))
        self.assertEqual(output_coordinates("k30", 2)[5 * 8 + 6], (17, 38))
        self.assertEqual(output_coordinates("k30", 3)[5 * 8 + 6], (21, 38))
        self.assertEqual((2 * 132 + 17, 2 * 128 + 38), (281, 294))
        self.assertEqual(output_coordinates("k30", 2)[44:48],
                         [(17, 36), (17, 37), (17, 38), (17, 39)])

    def test_k30_two_slot_sequential_trace(self):
        # 第 6 章：只检查顺序执行时各槽位的内容，不证明并发屏障协议正确。
        for tiles in (1, 2, 3, 256):
            slots = [0, None]  # 预备阶段先填好第 0 个分块，供主循环第一次计算使用。
            consumed = []
            for tile in range(tiles):
                self.assertEqual(slots[tile % 2], tile)
                consumed.append(slots[tile % 2])
                future = tile + 1
                if future < tiles:
                    self.assertNotEqual(future % 2, tile % 2)
                    slots[future % 2] = future
            self.assertEqual(consumed, list(range(tiles)))
        self.assertEqual((2 * 256, 1 + 256), (512, 257))

    def test_resources_and_units(self):
        self.assertEqual(2 * (16 * 132 + 16 * 128) * 4, 33280)
        self.assertEqual(2 * 132 + 3, 267)
        self.assertEqual(4096 // 16, 256)
        self.assertEqual(4096 // 8, 512)
        self.assertEqual(128 * 128 / (2 * (128 + 128)), 32)
        self.assertAlmostEqual(2 * 1024**3 / 1e6, 2147.483648)


if __name__ == "__main__":
    print("CPU teaching models only; no GPU correctness or throughput claim.", flush=True)
    unittest.main(verbosity=2)
