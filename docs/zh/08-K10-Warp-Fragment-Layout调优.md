# RTX 3080 Kernel 10：Warp Fragment Layout 受控实验

日期：2026-10-11

环境：RTX 3080 10 GB（SM86）、CUDA 12.4、Nsight Compute 2024.1

问题：K10 距离同口径 strict FP32 cuBLAS 约 9% 的差距，能否只靠改变 warp 内
lane 对输出 fragment 的所有权、`TM/TN` 和累加器组织来缩小？

## 1. 结论先行

这轮实现了 K18 的 6 个版本（含完全不变的 V0 基线），但**没有找到可接受的优化**：

- K10 三轮均值为 19,648.9 GFLOPS，V0 为 19,671.5 GFLOPS；二者 SASS 完全相同，
  0.115% 的差异就是测量噪声。
- 最接近的 V3 XOR lane permutation 为 19,651.3 GFLOPS，只比 K10 高 0.012%，不构成收益。
- V1、V2、V4、V5 分别比 K10 慢 0.19%、5.78%、1.43%、9.12%。
- 所有版本的动态 FFMA 总数相同、local load/store 都为 0；失败不是少算、改变精度或 spill。
- V1/V5 破坏 shared-load bank 行为；V2 增加 10 registers、50.3 M LSU 指令和 48 条静态
  LDS；V4/V5 则增加地址整数指令并降低 issue efficiency。

因此 K10 的当前 `8×4` lane lattice 和 `TM=8,TN=4,WNITER=4` 已经是一个很强的局部
最优。粗粒度地交换 M/N 所有权或把 N fragment 做宽，没有填补 issue 空洞。

## 2. 实验边界

K18 用编译期断言冻结以下所有条件：

```text
BM=128, BN=128, BK=16
WM=64, WN=64, NUM_THREADS=128
AS_PAD=4, launch_bounds(128,2)
128 accumulators/thread
```

同时复用 K10 的 global→shared helper、`As[BK][BM+4]`、`Bs[BK][BN]`、两次
`__syncthreads()`、K tile 循环和 float4 epilogue。没有更改 global load、shared 布局、
pipeline、block/warp tile 或 FP32 计算公式。V0 直接调用原 K10 core；其规范化 SASS
哈希与 K10 都是：

```text
d4ea858a478d5588347edaa6d95b92ed79faa135fe408dd2df31ffa42dee30f7
```

## 3. 当前 K10 如何把 64×64 warp tile 分给 32 个 lane

一个 block 有 4 个 warp。`warpIdx` 的映射为：

```cpp
warpCol = warpIdx % 2;
warpRow = warpIdx / 2;
```

所以 4 个 warp 分别覆盖 128×128 block tile 内的左上、右上、左下、右下 64×64 区域。
K10 内部 `WMITER=1, WSUBM=64, WSUBN=16`，lane 映射是：

```cpp
lane = threadIdx.x % 32;
threadColInWarp = lane % 4;
threadRowInWarp = lane / 4;
```

每个 lane 负责 8 行，并在 N 方向负责四段 4 列，共 `8×(4×4)=128` 个结果。以下坐标
都相对于该 warp 的 64×64 输出 tile：

| Lane | M 行 | N 列区间 |
|---:|---|---|
| 0 | 0–7 | 0–3; 16–19; 32–35; 48–51 |
| 1 | 0–7 | 4–7; 20–23; 36–39; 52–55 |
| 2 | 0–7 | 8–11; 24–27; 40–43; 56–59 |
| 3 | 0–7 | 12–15; 28–31; 44–47; 60–63 |
| 4 | 8–15 | 0–3; 16–19; 32–35; 48–51 |
| 5 | 8–15 | 4–7; 20–23; 36–39; 52–55 |
| 6 | 8–15 | 8–11; 24–27; 40–43; 56–59 |
| 7 | 8–15 | 12–15; 28–31; 44–47; 60–63 |
| 8 | 16–23 | 0–3; 16–19; 32–35; 48–51 |
| 9 | 16–23 | 4–7; 20–23; 36–39; 52–55 |
| 10 | 16–23 | 8–11; 24–27; 40–43; 56–59 |
| 11 | 16–23 | 12–15; 28–31; 44–47; 60–63 |
| 12 | 24–31 | 0–3; 16–19; 32–35; 48–51 |
| 13 | 24–31 | 4–7; 20–23; 36–39; 52–55 |
| 14 | 24–31 | 8–11; 24–27; 40–43; 56–59 |
| 15 | 24–31 | 12–15; 28–31; 44–47; 60–63 |
| 16 | 32–39 | 0–3; 16–19; 32–35; 48–51 |
| 17 | 32–39 | 4–7; 20–23; 36–39; 52–55 |
| 18 | 32–39 | 8–11; 24–27; 40–43; 56–59 |
| 19 | 32–39 | 12–15; 28–31; 44–47; 60–63 |
| 20 | 40–47 | 0–3; 16–19; 32–35; 48–51 |
| 21 | 40–47 | 4–7; 20–23; 36–39; 52–55 |
| 22 | 40–47 | 8–11; 24–27; 40–43; 56–59 |
| 23 | 40–47 | 12–15; 28–31; 44–47; 60–63 |
| 24 | 48–55 | 0–3; 16–19; 32–35; 48–51 |
| 25 | 48–55 | 4–7; 20–23; 36–39; 52–55 |
| 26 | 48–55 | 8–11; 24–27; 40–43; 56–59 |
| 27 | 48–55 | 12–15; 28–31; 44–47; 60–63 |
| 28 | 56–63 | 0–3; 16–19; 32–35; 48–51 |
| 29 | 56–63 | 4–7; 20–23; 36–39; 52–55 |
| 30 | 56–63 | 8–11; 24–27; 40–43; 56–59 |
| 31 | 56–63 | 12–15; 28–31; 44–47; 60–63 |

这个布局有三个重要性质：

1. 连续 4 个 lane 共享同一组 8 个 A 值；shared load 可使用 warp broadcast。
2. 同一组 4 个 lane 在每个 16-column 子块中读取相邻的 4-column B fragment。
3. `threadResults[row*16+col]` 是规则的 8×16 寄存器数组；物理 C 列虽然分成四段，
   fragment 内部和最终 store 始终是 float4 对齐。

因此它不是随意的二维拆分：A broadcast、B 的 N 向量和 float4 store 三者已经协调。

## 4. 六个手工布局

| 版本 | lane / fragment 变化 | 每线程 fragment | 假设 |
|---|---|---|---|
| V0 | 原 K10 | `TM8×TN4×WNITER4` | 完全等价控制组 |
| V1 | 交换 8×4 lane lattice 的行列展开顺序 | 同 V0 | 连续 lane 沿 M，增强 B broadcast |
| V2 | 16×2 lane lattice | `TM4×TN8×WNITER4` | 减小 M、加宽 N fragment |
| V3 | `lane_n ^= lane_m & 3` | 同 V0 | 仅改变物理 lane 对 N group 的所有权 |
| V4 | 保持 8×4 lattice | `TM8×TN8×WNITER2` | 两段更宽的 N fragment |
| V5 | 保持 8×4 lattice | `TM8×TN16×WNITER1` | 单段连续 16-column fragment |

`generate_k18_lane_map.py` 会枚举全部 32 lane，并断言每个版本恰好覆盖 64×64 的 4096
个输出一次，没有遗漏和重叠。这不是几十上百组参数搜索。

## 5. 三轮交错性能

固定 `4096³, alpha=0.5, beta=3, seed=20261011`，每次计时前恢复 C，warmup=10、
repeat=50。8 个模式按轮次旋转起始位置：

| 模式 | 平均时间 | 平均 GFLOPS | 三轮范围 | 相对 K10 |
|---|---:|---:|---:|---:|
| strict FP32 cuBLAS | 6.3623 ms | 21,603.8 | 21,488.8–21,806.0 | +9.95% |
| **K10** | **6.9950 ms** | **19,648.9** | 19,573.1–19,756.7 | 基线 |
| V0 | 6.9870 ms | 19,671.5 | 19,617.6–19,756.4 | +0.115%（噪声） |
| V1 | 7.0080 ms | 19,611.1 | 19,524.1–19,714.8 | -0.192% |
| V2 | 7.4240 ms | 18,513.4 | 18,423.3–18,591.5 | -5.779% |
| V3 | 6.9940 ms | 19,651.3 | 19,602.2–19,696.7 | +0.012%（噪声） |
| V4 | 7.0963 ms | 19,367.7 | 19,311.7–19,427.6 | -1.431% |
| V5 | 7.6967 ms | 17,856.7 | 17,806.5–17,942.8 | -9.121% |

没有一个变体满足“稳定 GFLOPS 提升”的首要条件。K10 是同轮 strict cuBLAS 的
90.95%；不能拿 V0 的最高值与 cuBLAS 的最低值重新组合一个更好看的比例。

## 6. Nsight：为什么这些布局没有变快

Profiler duration 受 replay 影响，只用于同组计数器比较，不替代上面的正常 benchmark。

| 指标 | K10 | V1 | V2 | V3 | V4 | V5 |
|---|---:|---:|---:|---:|---:|---:|
| Registers/thread | 184 | 184 | 194 | 184 | 186 | 190 |
| Occupancy | 16.59% | 16.59% | 16.63% | 16.58% | 16.49% | 16.61% |
| FP32 pipe active | 73.62% | 73.34% | 68.65% | 73.72% | 72.02% | 65.62% |
| Issue slots busy | 79.13% | 78.92% | 75.26% | 79.46% | 77.17% | 70.59% |
| No eligible | 20.87% | 21.08% | 24.74% | 20.54% | 22.83% | 29.41% |
| Eligible warps/scheduler | 1.356 | 1.375 | 1.299 | 1.401 | 1.354 | 1.247 |
| Dispatch stall | 0.252 | 0.253 | 0.254 | 0.258 | 0.284 | 0.250 |
| Long scoreboard | 0.081 | 0.079 | 0.155 | 0.057 | 0.053 | 0.189 |
| Shared-load conflicts | 0 | 134.22 M | 0 | 0 | 0 | 268.44 M |
| Local load/store sectors | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 |

解释：

- V1 让连续 lane 沿 M 走，B broadcast 更直接，却破坏原布局中 A 的四-lane broadcast，
  产生 134.22 M shared-load conflicts，净结果回退。
- V2 的 `regN` 从 16 增到 32 个值，`regM` 从 8 降到 4；总 accumulator 不变，但
  数据供给成本增大。寄存器到 194，LSU 指令从 130.29 M 增到 180.63 M，long scoreboard
  几乎翻倍，issue 和 FMA 同时下降。
- V3 的 profiler 中 issue/no-eligible/scoreboard 略好，但正常三轮只提高 0.012%。这正是
  “指标变好但性能不变仍判失败”的例子。
- V4 没有 shared conflict 或 spill，但更宽 fragment 没有改善 FFMA 供给，dispatch stall
  增加、issue 下降。
- V5 同时产生 268.44 M shared-load conflicts和更多整数地址指令，no-eligible 达
  29.41%，是最慢的版本。

## 7. SASS 对照

| 模式 | 静态指令 | FFMA | IMAD | IADD3 | LDS | MOV | SHFL |
|---|---:|---:|---:|---:|---:|---:|---:|
| K10 / V0 | 2,768 | 2,176 | 20 | 26 | 96 | 29 | 0 |
| V1 | 2,776 | 2,176 | 20 | 26 | 96 | 29 | 0 |
| V2 | 2,832 | 2,176 | 21 | 25 | 144 | 46 | 0 |
| V3 | 2,776 | 2,176 | 20 | 26 | 96 | 30 | 0 |
| V4 | 2,808 | 2,176 | 21 | 46 | 96 | 33 | 0 |
| V5 | 2,880 | 2,176 | 20 | 92 | 96 | 33 | 0 |

所有版本保持 2,176 条静态 FFMA 和相同的动态 thread-level FFMA 总量
68,736,253,952。V2 的明显代价是更多 LDS/MOV；V4/V5 的明显代价是更多 IADD3。
没有版本生成 SHFL，也没有靠 warp shuffle 暗中改变数据路径。

注意：静态 FFMA 包含完全展开的主循环和 epilogue；它不是 runtime 总指令数。

## 8. 正确性与资源门禁

- 6 variants × 6 sizes × 3 beta = **108/108 PASS**；
- 尺寸为 128、256、512、1024、2048、4096，beta 为 0、1、3；
- 参考始终是 strict FP32 cuBLAS，绝对误差阈值保持 0.01；
- 6 个版本各做一次 128³ Compute Sanitizer memcheck，均为 0 errors；
- ptxas registers 为 184/184/194/184/186/190，全部 0 stack、0 spill；
- static shared 都是 16,640 B，实际 Nsight 分配口径为 17.664 KB（含 driver 保留部分）。

## 9. 与 CUTLASS 风格的区别

现代 GEMM 微内核中的 warp tile、thread fragment 和 iterator 是一套共同设计，而不是
“tile 越大越快”：

- layout 决定 lane 之间是连续访问、broadcast 还是 bank conflict；
- iterator 把逻辑 fragment 映射为适合硬件 load/store/MMA 指令的地址序列；
- fragment shape 决定 accumulator 数量、独立依赖链和寄存器生命周期；
- pipeline stage 必须与 tile、occupancy 和 copy 粒度共同预算。

K18 只允许改变其中的 lane/fragment 部分，结果表明单独把 N fragment 加宽会让 LDS、
地址计算或 bank 行为变差。CUTLASS 风格的收益来自整体协同，不能只抄一个 thread tile
形状而保持其他 iterator 与调度不变。

## 10. 对剩余差距的新判断

本轮可以明确排除的是粗粒度假设 A：**当前差距不是换一个简单 warp fragment 所有权
就能解决。** K10 的布局已经协调 A broadcast、B fragment 与 float4 epilogue。

对四类来源的判断是：

| 来源 | 当前判断 |
|---|---|
| A. Warp fragment layout | 本轮 5 个受控变体无收益；简单重排不是主因 |
| B. Register scheduling | 仍然重要；K10 只有 79.13% issue，低 occupancy 下依赖距离敏感 |
| C. Address calculation | 是可测成本，但不是单独答案；V4/V5 的 IADD3 增加与回退相关 |
| D. 其他 | cuBLAS 的微内核、编译器调度、软件流水、epilogue 和更成熟的整体协同 |

不能把 cuBLAS 的 9.05% 优势精确伪拆成 B/C/D 各几个百分点。证据只支持：K10 不是
DRAM bound、不是 shared-load conflict bound，也不能通过 coarse lane remap 提升；最
可见的余量仍是 **shared→register→FFMA 的指令调度和低 occupancy 下的 latency hiding**。

下一步若继续，最值得投入的是直接审计 K10 SASS 的 LDS→FFMA 依赖距离，做很小的
load-scheduling / accumulator-order 实验，且不改变 ownership。若这也没有至少约 1%
的稳定收益，应停止在这个教学微内核上继续堆复杂度，转向系统比较 cuBLAS/CUTLASS
微内核的 instruction mix 与 pipeline，而不是再搜索 lane 公式。

后续 K19 已完成这项验证：三种 B fragment register prefetch 均回退 6.30%–7.36%。
SASS 证明 K10 的 LDS 本来就被 ptxas 穿插进 FFMA 序列，额外寄存器组没有减少 LDS，
反而降低 issue/FMA 利用率。见 [K19 Fragment Register Prefetch](09-K10-Fragment寄存器预取调优.md)。

## 11. 复现命令与数据

```bash
./scripts/generate_k18_lane_map.py
./scripts/build_kernel_18_variants.sh
RUN_TAG=my-correctness DEVICE=0 ./scripts/kernel_18_correctness_matrix.sh
RUN_TAG=my-fair DEVICE=0 ./scripts/kernel_18_fair_audit.sh
RUN_TAG=my-ncu DEVICE=0 ./scripts/profile_kernel_18_ncu.sh
./scripts/summarize_kernel_18_ncu.py --tag my-ncu
./scripts/audit_kernel_18_sass.sh
```

仓库内原始/汇总数据：

- [三轮交错 benchmark](../results/rtx3080-k18-warp-layout-fair-20261011.csv)
- [Nsight 指标](../results/rtx3080-k18-warp-layout-nsight-20261011.csv)
- [SASS 指令计数与哈希](../results/rtx3080-k18-warp-layout-sass-20261011.csv)
- [正确性与 memcheck 汇总](../results/rtx3080-k18-warp-layout-correctness-20261011.csv)

每个 benchmark / NCU 脚本都使用 `RUN_TAG`，拒绝覆盖同名历史结果。
