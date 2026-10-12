# RTX 3080 K10：SASS 微结构指标报告

日期：2026-10-12

本页是 [K10 微内核与 cuBLAS 差距分析](10-K10-微内核与cuBLAS差距分析.md) 的指标
账本。只记录工具输出、统计定义和可支持的判断，不把静态指令距离伪装成 cycle latency。

## 1. 工具与对象

```text
GPU: RTX 3080 / SM86
CUDA: 12.4
Nsight Compute: 2024.1
K10 binary: build-prefetch-v1/sgemm
K10 symbol: sgemmWarptiling<128,128,16,64,64,4,8,4,128,4>
strict cuBLAS symbol: cutlass_80_simt_sgemm_256x128_8x4_nn_align1
```

K10 使用 `cuobjdump --dump-sass` 与 `nvdisasm -plr -lrm count`。动态 cuBLAS 内核的
SASS 从同一次 NCU report 的 source page 导出。

## 2. K10 静态指令

| 指标 | 数值 |
|---|---:|
| Static instructions | 2,768 |
| FFMA | 2,176 |
| Mainloop accumulator FFMA | 2,048 |
| LDS.128 | 96 |
| LDG | 40 |
| STS | 20 |
| IMAD | 20 |
| IADD3 | 26 |
| LEA | 22 |
| SHF | 9 |
| MOV | 29 |
| BAR | 2 |

2,048 条主循环 FFMA 等于 `BK16 × 128 accumulators`；另外 128 条来自 epilogue。

## 3. Accumulator 依赖

统计方法：只取两条 block `BAR` 之间的 FFMA，沿第四操作数 GPR 到 destination GPR
建立 value chain；因此即使 ptxas 改变 accumulator 的物理寄存器也不会拆错逻辑链。

| 指标 | 数值 |
|---|---:|
| 逻辑链 | 128 |
| 每条链每个 BK tile 更新 | 16 |
| 物理 GPR rename | 83 |
| 静态指令复用距离 min / median / mean / max | 16 / 134 / 133.58 / 211 |
| 中间独立 FFMA min / median / mean / max | 16 / 127 / 126.57 / 201 |

判断：逻辑链的总长度是 K=4096，但 128 条链高度交错；没有证据表明单 accumulator
的直接 FFMA latency 是主要限制。

## 4. LDS→FFMA

统计方法：`LDS.128` 写入连续 4 个 GPR；向后寻找第一条消费其中任意 GPR 的 FFMA，
记录两者之间的静态指令数。

| LDS.128 | 找到消费者 | min | median | mean | max |
|---:|---:|---:|---:|---:|---:|
| 96 | 96 | 4 | 66 | 59.72 | 114 |

K10 NCU：

| 指标 | 数值 |
|---|---:|
| Short scoreboard | 0.0294 warp/cycle |
| Long scoreboard | 0.0825 warp/cycle |
| Eligible warps/scheduler | 1.351 |
| Issue slots busy | 79.01% |
| No eligible | 20.99% |

判断：少数距离为 4 的 load-use pair 存在，但 short-scoreboard 不高；主要现象是整体
ready-warp 不足，不能仅凭最小距离把问题归为 LDS latency。

## 5. Register lifetime

| 指标 | 数值 |
|---|---:|
| ptxas allocated GPR/thread | 184 |
| nvdisasm peak live GPR | 176 |
| Peak address | `0x0fb0` (`LDS.128`) |
| Median live GPR | 162 |
| Accumulator / A fragment / B fragment | 128 / 8 / 16 |
| 上述源码必要值合计 | 152 |

判断：accumulator 与 fragment 已占 152 个值；峰值距分配量仅 8 个 GPR。没有“大量
明显死寄存器”证据，也没有 spill。

## 6. K10 与 strict cuBLAS 动态指标

| 指标 | K10 | strict cuBLAS |
|---|---:|---:|
| Threads/block | 128 | 256 |
| Registers/thread | 184 | 202 |
| Shared/block | 17.664 KB | 50.176 KB |
| Achieved occupancy | 16.59% | 16.67% |
| SM throughput | 74.53% | 82.11% |
| FP32 pipe active | 73.64% | 78.60% |
| Issue slots busy | 79.01% | 87.34% |
| No eligible | 20.99% | 12.66% |
| Dispatch stall | 0.2510 | 0.1432 |
| Long scoreboard | 0.0825 | 0.0235 |
| Wait stall | 0.0890 | 0.0122 |
| Thread FFMA | 68.736 B | 68.736 B |
| Shared-load wavefronts | 335.546 M | 201.900 M |
| Shared-load conflicts | 0 | 0 |
| Local load/store sectors | 0/0 | 0/0 |
| Dynamic ALU instructions | 36.97 M | 89.94 M |

判断：cuBLAS 用更多资源和更多 ALU 指令，却有更少的 shared wavefront 和更多 ready
warp。地址指令数量不是 K10 的主导差距。

## 7. cuBLAS SASS 边界

whole-function 静态扫描在 strict cuBLAS 中找到：

```text
48 × LDGSTS
6  × LDGDEPBAR
4  × DEPBAR
```

这只证明这些指令族存在于被选择的内核。该函数包含分支、对齐和 beta 路径，不能把
4,490 条 whole-function 静态指令全部当作单次动态执行路径。动态 NCU counter 才用于
性能归因。

## 8. 复现

```bash
./scripts/analyze_kernel_10_microarchitecture.py
RUN_TAG=my-run ./scripts/profile_kernel_10_microarchitecture_ncu.sh
./scripts/summarize_kernel_10_microarchitecture_ncu.py --tag my-run
./scripts/summarize_kernel_10_cublas_sass.py --tag my-run
```

- [静态依赖与 lifetime CSV](../results/rtx3080-k10-microarchitecture-static-20261012.csv)
- [Nsight 动态指标 CSV](../results/rtx3080-k10-cublas-microarchitecture-nsight-20261012.csv)
- [SASS 指令族 CSV](../results/rtx3080-k10-cublas-sass-summary-20261012.csv)
