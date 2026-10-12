# RTX 3080 Kernel 10：Fragment Register Prefetch 受控实验

日期：2026-10-11

环境：RTX 3080 10 GB（SM86）、CUDA 12.4、Nsight Compute 2024.1

问题：K10 距离同口径 strict FP32 cuBLAS 约 9% 的差距，是否主要来自 B fragment
在 shared→register→FFMA 路径上的 load-use 延迟？

## 1. 结论先行

K19 的三个版本全部正确、无 spill，但**显式 B fragment register prefetch 无效且稳定
回退**：

- K10 三轮平均 19,688.4 GFLOPS；V1/V2/V3 分别为 18,265.9、18,447.6、
  18,239.2 GFLOPS，相对回退 7.22%、6.30%、7.36%。
- V1/V3 使用 196 registers/thread，V2 使用 187，均低于 200；local load/store 为 0。
- 每个版本仍是 96 条静态 `LDS.128`、2,176 条静态 FFMA，动态 shared-load
  wavefront 也不变。预取没有减少 Shared→Register 工作。
- K10 的 issue slots busy 为 79.27%，V1/V2/V3 降到 69.54%/71.51%/69.67%；
  FP32 pipe active 从 73.71% 降到 64.78%/66.25%/64.67%。
- no-eligible 从 20.73% 升到 30.46%/28.49%/30.33%。这与正常计时的回退方向一致。
- V1 与显式两步展开的 V3 生成完全相同的规范化 SASS。ptxas 已把两种源码表达
  归一成同一调度，手写展开没有创造新的硬件流水。

所以本轮否定的是“**K10 只是漏写了 B(k+1) 寄存器预取**”这一具体假设，而不是声称
shared/register 调度完全没有优化余量。K10 源码看似先 load 再 compute，但 ptxas 的
SASS 已经把未来 LDS 插入当前 FFMA 序列；额外的 B 寄存器 bank 延长 live range，反而
降低了可发射性。

## 2. 冻结条件与三个版本

K19 用编译期断言固定：

```text
BM=128, BN=128, BK=16
WM=64, WN=64, WNITER=4
TM=8, TN=4, NUM_THREADS=128
AS_PAD=4, launch_bounds(128,2)
```

它复用 K10 的 global→shared loader、`As[BK][BM+4]`、`Bs[BK][BN]`、lane mapping、
128 个 accumulator、两次 block barrier 和 epilogue。没有第二份 shared tile、没有
`cp.async`，也没有改变 FP32 乘加顺序。

| 版本 | 唯一意图 |
|---|---|
| V1 | 两组 `regN[16]` ping-pong，计算 B(k) 前加载 B(k+1) |
| V2 | V1 加四个递增 shared 指针，提前形成 B fragment 地址 |
| V3 | 把 K loop 显式展开两步，再安排 B(k+1)/B(k+2) 的加载 |

## 3. K10 的每个 K iteration 实际做什么

对一个 `dotIdx`，每线程读取 8 个 A 和 16 个 B 值，然后更新 128 个 accumulator：

```text
2 × LDS.128  -> regM[8]
4 × LDS.128  -> regN[16]
128 × FFMA   -> threadResults[128]
```

`BK=16` 完全展开后，主计算部分是 96 条 `LDS.128` 和 2,048 条 FFMA；完整 kernel
静态 SASS 中还有 128 条 epilogue FFMA，所以审计总数是 2,176。

源码的“load fragment；compute fragment”不是最终机器指令顺序。K10 SASS 在一组
FFMA 尚未结束时就插入后续 `LDS.128`。按“LDS 与第一条读取其四个目标寄存器的 FFMA
之间隔了多少条静态指令”统计：

| 模式 | LDS.128 | 全部找到 FFMA 使用 | 最小 | 中位 | 平均 | 最大 |
|---|---:|---:|---:|---:|---:|---:|
| K10 | 96 | 96 | 4 | 66 | 59.72 | 114 |
| V1 | 96 | 96 | 3 | 44 | 55.99 | 144 |
| V2 | 96 | 96 | 2 | 54 | 55.61 | 136 |
| V3 | 96 | 96 | 3 | 44 | 55.99 | 144 |

这是静态指令距离，不是 cycle latency，也不能处理运行时 scoreboard；它只用于确认
编译器是否已经跨 FFMA 组调度 LDS。显式预取没有整体拉长 load-use 距离，V1/V3 的
中位距离反而从 66 缩短到 44。更重要的是，最终判据 issue/FMA 和 GFLOPS 都下降。

## 4. 三轮交错性能

固定 `4096³, alpha=0.5, beta=3, seed=20261011`，每个 measured call 前恢复 C，
warmup=10、repeat=50，五种模式轮换起始位置：

| 模式 | 平均时间 | 平均 GFLOPS | 三轮范围 | 相对 K10 |
|---|---:|---:|---:|---:|
| strict FP32 cuBLAS | 6.3333 ms | 21,700.1 | 21,602.1–21,797.6 | +10.22% |
| **K10** | **6.9807 ms** | **19,688.4** | 19,608.0–19,789.2 | 基线 |
| K19 V1 | 7.5243 ms | 18,265.9 | 18,188.4–18,329.4 | -7.22% |
| K19 V2 | 7.4507 ms | 18,447.6 | 18,397.5–18,529.0 | -6.30% |
| K19 V3 | 7.5353 ms | 18,239.2 | 18,198.7–18,288.3 | -7.36% |

K10 是同轮 strict FP32 cuBLAS 的 90.73%。Profiler 自己显示的 benchmark 时间受
11-pass replay 和时钟状态影响，不用于替代这张正常运行表。

## 5. Nsight：预取为何变慢

| 指标 | K10 | V1 | V2 | V3 |
|---|---:|---:|---:|---:|
| Registers/thread | 184 | 196 | 187 | 196 |
| Occupancy | 16.59% | 16.17% | 16.46% | 16.15% |
| SM throughput | 74.69% | 66.61% | 67.51% | 66.48% |
| FP32 pipe active | 73.71% | 64.78% | 66.25% | 64.67% |
| Issue slots busy | 79.27% | 69.54% | 71.51% | 69.67% |
| No eligible | 20.73% | 30.46% | 28.49% | 30.33% |
| Eligible warps/scheduler | 1.361 | 1.187 | 1.234 | 1.185 |
| Long scoreboard | 0.080 | 0.185 | 0.212 | 0.191 |
| Short scoreboard | 0.030 | 0.100 | 0.119 | 0.100 |
| Shared-load wavefronts | 335.55 M | 335.55 M | 335.55 M | 335.55 M |
| Shared-load wavefront/s | 41.98 G | 37.88 G | 38.41 G | 37.81 G |
| Shared-load conflicts | 0 | 0 | 0 | 0 |
| Local load/store sectors | 0/0 | 0/0 | 0/0 | 0/0 |

显式预取没有降低 LDS 或 shared wavefront，却提高了寄存器生命周期和 scoreboard
等待；活跃 warp 略降、eligible warp 更少，最终让 FFMA pipe 吃不满。V2 虽只比 K10
多 3 个寄存器，也把静态 IMAD 从 20 降到 9，但吞吐仍回退 6.30%，说明“减少几个
地址 IMAD”本身不是剩余差距的答案。

## 6. SASS 指令对照

| 模式 | 静态指令 | FFMA | LDS | IMAD | IADD3 | MOV | SASS 关系 |
|---|---:|---:|---:|---:|---:|---:|---|
| K10 | 2,768 | 2,176 | 96 | 20 | 26 | 29 | baseline |
| V1 | 2,768 | 2,176 | 96 | 12 | 26 | 40 | 与 V3 完全相同 |
| V2 | 2,760 | 2,176 | 96 | 9 | 26 | 28 | 独立序列 |
| V3 | 2,768 | 2,176 | 96 | 12 | 26 | 40 | 与 V1 完全相同 |

V1/V3 的 SHA-256 都是
`be4ab146c7de8866c01a4c14e7c5c83a50b6e248f8acc9d8989a017500f813a8`。
V2 总指令更少也没有变快，再次说明关键是依赖与发射时机，不是静态条数最小化。

## 7. 正确性与资源门禁

- 3 variants × 6 sizes × 3 beta = **54/54 PASS**；
- 尺寸为 128、256、512、1024、2048、4096，beta 为 0、1、3；
- strict FP32 cuBLAS 参考与 0.01 absolute error 阈值保持不变；
- 三个版本在 128³ 上做 Compute Sanitizer memcheck，均为 0 errors；
- ptxas 报告 K10/V1/V2/V3 为 184/196/187/196 registers，全部 0 stack、0 spill；
- static shared 都是 16,640 B，Nsight 的分配口径都是 17.664 KB。

## 8. 对剩余约 9% 差距的新判断

1. **是否来自 Shared→Register 调度？** K10 的低 occupancy 与约 20.7% no-eligible
   说明片上供给/依赖调度仍是可测限制；但本实验不支持“缺少 B(k+1) prefetch 是主要
   原因”。编译器已经做跨组 LDS 调度，简单显式化反而破坏 issue efficiency。
2. **Fragment prefetch 是否有效？** 对这台 RTX 3080、这个 K10 shape 无效。三个版本
   均稳定下降 6.30%–7.36%，不应合入默认 K10。
3. **当时的下一步方向：** 优先 A——对照 control code 和 SASS 调度，研究 accumulator
   依赖链/指令顺序，但不要直接内联大量手写 SASS；B 只做针对性的地址 lifetime
   实验，因为 V2 已证明少几个 IMAD 不足以提速；C 应升级为 CUTLASS 风格
   iterator/layout/pipeline 的整体对照研究，而不是在冻结的 K10 上继续叠一个局部技巧。

不能把 cuBLAS 的约 9% 优势精确分摊给某几条指令。当前证据支持的说法是：差距来自
更成熟微内核对寄存器生命周期、LDS/FFMA 交织、独立 accumulator 链和整体 pipeline
的联合安排，而不是 Grid、L2、bank conflict、简单 global latency、lane permutation
或一层 B fragment ping-pong 中的某一个单项开关。

## 9. 复现与数据

```bash
./scripts/build_kernel_19_variants.sh
RUN_TAG=my-correctness DEVICE=0 ./scripts/kernel_19_correctness_matrix.sh
RUN_TAG=my-fair DEVICE=0 ./scripts/kernel_19_fair_audit.sh
RUN_TAG=my-ncu DEVICE=0 ./scripts/profile_kernel_19_ncu.sh
./scripts/summarize_kernel_19_ncu.py --tag my-ncu
./scripts/audit_kernel_19_sass.sh
./scripts/analyze_kernel_19_sass_dependencies.py
```

- [三轮交错 benchmark](../results/rtx3080-k19-fragment-prefetch-fair-20261011.csv)
- [54 项正确性矩阵](../results/rtx3080-k19-fragment-prefetch-correctness-20261011.csv)
- [Nsight 指标](../results/rtx3080-k19-fragment-prefetch-nsight-20261011.csv)
- [SASS 指令计数与哈希](../results/rtx3080-k19-fragment-prefetch-sass-20261011.csv)
- [LDS→FFMA 静态距离](../results/rtx3080-k19-fragment-prefetch-lds-distance-20261011.csv)
- [ptxas 资源门禁](../results/rtx3080-k19-fragment-prefetch-resources-20261011.csv)
- [Compute Sanitizer 汇总](../results/rtx3080-k19-fragment-prefetch-memcheck-20261011.csv)

所有脚本通过 `RUN_TAG` 拒绝覆盖同名结果；K10 继续作为默认最佳实现，K19 只保留为
可复现的失败实验。

后续的 [K10 微内核与 cuBLAS 差距分析](10-K10-微内核与cuBLAS差距分析.md) 已进一步
证明：K10 有 128 条逻辑 accumulator 链，中位间隔 127 条独立 FFMA；strict cuBLAS
则通过更大的 CTA 资源包络和集成 `LDGSTS/DEPBAR` 流水获得更高 ready-warp/issue。
