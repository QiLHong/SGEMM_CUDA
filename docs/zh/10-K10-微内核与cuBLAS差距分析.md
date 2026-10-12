# RTX 3080 K10：微内核结构与 cuBLAS 剩余差距

日期：2026-10-12

环境：RTX 3080 10 GB（SM86）、CUDA 12.4、Nsight Compute 2024.1

问题：K15–K19 已排除若干局部假设后，为什么 K10 仍只有同口径 strict FP32
cuBLAS 的约 90.7%，剩余约 9% 来自哪里？

## 1. 结论先行

K10 已经是一个很强的 FP32 SIMT 微内核。它有 128 条独立 accumulator 链、无
shared-load bank conflict、无 spill，ptxas 还主动交织 LDS/FFMA 并进行 accumulator
物理寄存器重命名。剩余差距不是一个漏掉的局部开关。

本轮直接 profile 到的 strict FP32 cuBLAS 内核是：

```text
cutlass_80_simt_sgemm_256x128_8x4_nn_align1
```

它同样是 SIMT SGEMM，不是 Tensor Core 路径。它与 K10 的关键差别是：

- cuBLAS 使用 256 threads/block、512 blocks；K10 使用 128 threads/block、1024 blocks；
- cuBLAS 使用 202 registers/thread 和 50.176 KB shared；K10 是 184 和 17.664 KB；
- 两者 achieved occupancy 都约 16.6%，也都有约 7.53 waves/SM；
- cuBLAS issue slots busy 为 87.34%，K10 为 79.01%；
- cuBLAS no-eligible 为 12.66%，K10 为 20.99%；
- cuBLAS shared-load wavefront 比 K10 少 39.83%，且两者 conflict 都为 0；
- cuBLAS SASS 明确出现 `LDGSTS`、`LDGDEPBAR`、`DEPBAR`，说明它把异步搬运、
  CTA 形状、shared working set 和 fragment 调度作为整体设计。

因此，剩余差距的最高可信解释是：**cuBLAS 用更大的 CTA 资源包络和成熟的整体
pipeline/iterator 调度，让同样数量级的活跃 warp 更经常处于 ready 状态。** 它不是
靠更高 occupancy，也不是靠更少的整数指令。

K17 只能证明“把 B-only `cp.async` 单独嫁接到 K10”失败，不能推出成熟的
global→shared pipeline 没有价值。K19 同理：它证明简单 B fragment ping-pong 失败，
不等于 shared→register 调度已经不存在任何余量。

## 2. K10 的完整 Block → Warp → Lane → Register 映射

### 2.1 Block 与 warp

一个 block 计算 `128×128` 的 C tile，共 128 threads、4 warps：

| warpIdx | warpRow | warpCol | block tile 内 M | block tile 内 N |
|---:|---:|---:|---:|---:|
| 0 | 0 | 0 | 0–63 | 0–63 |
| 1 | 0 | 1 | 0–63 | 64–127 |
| 2 | 1 | 0 | 64–127 | 0–63 |
| 3 | 1 | 1 | 64–127 | 64–127 |

每个 warp 的输出 tile 是 `64×64`。对 K10 固定参数：

```text
WMITER = 1
WNITER = 4
WSUBM = 64
WSUBN = 16
TM = 8
TN = 4
```

### 2.2 32 个 lane

对 `lane∈[0,31]`：

```text
lane_m = lane / 4       // 0..7
lane_n = lane % 4       // 0..3
```

令 `lane = 4*r+c`，则该 lane 在所属 `64×64` warp tile 内负责：

```text
M = {8*r + i | i=0..7}
N = {16*s + 4*c + j | s=0..3, j=0..3}
```

这条公式完整覆盖全部 32 lane：`r` 的 8 组覆盖 64 行，`c` 的 4 组与 `s` 的
4 个子块覆盖 64 列。每个 lane 恰好拥有 `8×16=128` 个输出，没有重叠或遗漏。

### 2.3 Register fragment

一个线程的寄存器逻辑组织是：

```text
regM[8]                 // 当前 k 的 8 个 A 值
regN[4 * 4]             // 4 个 N 子块，每块 4 个 B 值
threadResults[8 * 16]   // 128 个 accumulator
```

逻辑 accumulator `(i,s,j)` 的线性下标为：

```text
acc = i * 16 + s * 4 + j
```

其中 `i=0..7`、`s=0..3`、`j=0..3`。这是从 lane 到物理 C 元素、再到源码
accumulator 的完整映射。

## 3. 一个线程一次 K iteration 的数据流

Global→Shared 阶段由整个 block 合作完成：

```text
A[128×16] --float4 load--> As[16][132]   // 转置并加 4-column padding
B[16×128] --float4 load--> Bs[16][128]
                         __syncthreads()
```

对一个 `dotIdx`，每线程的数据流是：

```text
As[k][8 rows]  -- 2 × LDS.128 --> regM[8]
Bs[k][16 cols] -- 4 × LDS.128 --> regN[16]
regM × regN    -- 128 × FFMA  --> threadResults[128]
```

`BK=16` 完全展开后，主计算体为：

```text
96 × LDS.128
2048 × accumulator FFMA
```

完整函数还有 128 条 epilogue FFMA，所以静态总数为 2,176 FFMA。最后每线程以
32 次 float4 load/store 完成 128 个结果的：

```text
C = alpha * accumulator + beta * C
```

每个 K step 只需 8 个 A 和 16 个 B fragment 值，就更新 128 个结果。连续 4 个 lane
共享同一组 A 地址，B 则在 N 方向形成规则的 4-float fragment。这种 reuse、广播和
float4 epilogue 是 K10 已经很快的根本原因。

## 4. FFMA dependency chain

从数学上看，4096³ 时每个 C 元素有一条长度为 4096 的累加依赖链。但硬件看到的不是
“先把 acc0 连算 4096 次”，而是 128 条独立链交错执行。

对完全展开的一个 `BK=16` 主体，沿 FFMA 第四操作数到目的寄存器跟踪逻辑 accumulator：

| 指标 | K10 |
|---|---:|
| 逻辑 accumulator 链 | 128 |
| 每条链每个 BK tile 的更新次数 | 16 |
| 同一链两次更新之间的静态指令：min / median / mean / max | 16 / 134 / 133.58 / 211 |
| 中间独立 FFMA：min / median / mean / max | 16 / 127 / 126.57 / 201 |
| ptxas 物理 accumulator GPR 重命名 | 83 次 |

“物理重命名”指某条 FFMA 的 accumulator source 与 destination GPR 不同；ptxas 把
累加结果迁移到已经释放的 fragment 寄存器，而不是永远固定 `accN` 的物理编号。

最短路径之间仍有 16 条其他 FFMA，中位则为 127 条。因此，单个 accumulator 的
固定延迟不是当前最可信的首要瓶颈；K10 已有足够多的独立链。真正的问题是这些链
是否持续拿到 A/B operand、以及 scheduler 当时是否有 ready warp。

## 5. LDS → FFMA 与 scoreboard

对每条 `LDS.128`，统计它写入的 4 个 GPR 到第一条消费这些 GPR 的 FFMA 之间的
静态指令数：

| LDS.128 | min | median | mean | max |
|---:|---:|---:|---:|---:|
| 96 | 4 | 66 | 59.72 | 114 |

最小值 4 说明少数 fragment 仍有较紧的 load-use 窗口；中位 66 则证明 ptxas 已把
大量 LDS 提前穿插到 FFMA 序列。静态距离不能等价为 latency cycle，必须结合 Nsight：

| 指标 | K10 | strict FP32 cuBLAS |
|---|---:|---:|
| Issue slots busy | 79.01% | 87.34% |
| No eligible | 20.99% | 12.66% |
| Eligible warps/scheduler | 1.351 | 1.580 |
| Long scoreboard | 0.0825 | 0.0235 |
| Short scoreboard | 0.0294 | 0.0834 |
| Dispatch stall | 0.2510 | 0.1432 |
| Wait stall | 0.0890 | 0.0122 |
| Math-pipe throttle | 0.0466 | 0.0191 |

K10 的 short-scoreboard 绝对值并不高，所以不能把差距简单命名为“LDS 延迟”。更明显
的是 cuBLAS 同时减少 long-scoreboard、dispatch 和 fixed-latency wait，并把 eligible
warps 提高约 17%。cuBLAS 的 short-scoreboard 更高，却仍然更快；这再次说明单个
stall 指标不能脱离整体 issue readiness 解读。

## 6. 地址计算成本

K10 全函数静态 SASS：

| 指令 | 数量 |
|---|---:|
| IMAD | 20 |
| IADD3 | 26 |
| LEA | 22 |
| SHF（Ampere 上承担多种 shift/address 组合） | 9 |
| SHL | 0 |
| MOV | 29 |

不含 MOV 的上述整数/地址指令只有 77 条，占 2,768 条静态指令的 2.78%。动态计数中，
K10 ALU pipe 约 36.97 M instructions，而 cuBLAS 约 89.94 M；cuBLAS 总动态指令也比
K10 更多，却仍有更高 issue 和吞吐。

所以 K10 不是因为“整数地址指令特别多”而落后。减少几个 IMAD/IADD3 可能缩短个别
依赖，但不应列为从 90% 到 95% 的第一投资方向；K19 V2 已经给出了相同反例。

## 7. 为什么需要 184 registers/thread

源码层面的不可回避下限是：

| 用途 | float/GPR 数量 |
|---|---:|
| Accumulator | 128 |
| A fragment | 8 |
| B fragment | 16 |
| 合计 | 152 |

128 个 accumulator 已占分配寄存器的 69.6%；加上 A/B fragment 后占 82.6%。剩余
寄存器用于 64-bit global pointers、shared addresses、lane/warp index、global load
float4 临时值、循环状态、epilogue C fragment 和 predicate 相关中间量。

`nvdisasm --print-life-ranges` 的数据流分析给出：

```text
allocated GPR/thread = 184
peak live GPR        = 176（SASS 0x0fb0，一条 LDS.128）
median live GPR      = 162
```

分配量与峰值只差 8，说明 184 不是大量死寄存器造成的虚高。ptxas 已进行 83 次
accumulator GPR 重命名，也说明它在主动缩短/复用 lifetime。

更重要的是，少量减少寄存器不会改变 occupancy 档位。128-thread block 若想从 2 blocks/SM
提高到 3 blocks/SM，需要大致压到 170 registers/thread 以下；已有 168-register 版本虽
把 occupancy 提到约 24%，性能反而下降。因此不能为了一个 occupancy 数字破坏 FFMA
调度或制造 spill。

## 8. 与 strict FP32 cuBLAS 的直接微结构对照

正常三轮 benchmark 仍采用 K19 报告的公平口径：K10 19,688.4 GFLOPS，strict cuBLAS
21,700.1 GFLOPS，K10 为 90.73%。下面的 profiler duration 受 replay 影响，不用于
重新计算 GFLOPS。

| 指标 | K10 | strict FP32 cuBLAS |
|---|---:|---:|
| Kernel symbol | `sgemmWarptiling<128,...>` | `cutlass_80_simt_sgemm_256x128_8x4_nn_align1` |
| Threads/block | 128 | 256 |
| Blocks/grid | 1024 | 512 |
| Registers/thread | 184 | 202 |
| Shared memory/block | 17.664 KB | 50.176 KB |
| Waves/SM | 7.53 | 7.53 |
| Achieved occupancy | 16.59% | 16.67% |
| SM throughput | 74.53% | 82.11% |
| FP32 pipe active | 73.64% | 78.60% |
| Issue slots busy | 79.01% | 87.34% |
| No eligible | 20.99% | 12.66% |
| Thread FFMA | 68.736 B | 68.736 B |
| Shared-load wavefronts | 335.546 M | 201.900 M |
| Shared-load conflicts | 0 | 0 |
| Local load/store sectors | 0/0 | 0/0 |

cuBLAS 没有靠更少数学工作、Tensor Core、spill 或更高 occupancy 取胜。它使用更大的
CTA、更大的 shared working set 和更多寄存器，在相同约 8 个活跃 warp 的资源档位上，
减少 shared wavefront 并提高 ready-warp 供给。

从 NCU 导出的 cuBLAS SASS 还明确包含：

```text
LDGSTS.E.LTC128B
LDGDEPBAR
DEPBAR.LE
```

这证明 global→shared async staging 是其整体微内核的一部分。50.176 KB shared 与
202 registers/thread 也说明它愿意用更多片上资源换取调度余量。仅凭这些指标不能
严谨断言具体 stage 数；但可以严谨地说，它不是 K17 那种只给 B 加一层异步 copy 的
局部修改。

cuBLAS 整个函数的静态 SASS 有多个分支和对齐/beta 路径，因此其 4,490 条静态指令
不能直接与 K10 的 2,768 条当作 runtime 指令比较。动态计数才是主判据。

## 9. K10 与 CUTLASS 风格的思想差异

| 层次 | K10 | CUTLASS/cuBLAS 风格 | 性能意义 |
|---|---|---|---|
| Global iterator | 在线程函数中直接形成 float4 地址 | 独立 iterator 管理 shape、delta、advance、alignment | 把地址推进与 stage 调度共同设计 |
| Shared iterator | `As/Bs` 直接索引 | layout-aware iterator | 控制每条 LDS 的 lane transaction/broadcast |
| Warp iterator | 固定 lane 公式与嵌套循环 | warp tile iterator | 把 CTA 资源布局映射到 warp fragment |
| Fragment iterator | `regM/regN/threadResults` 平面数组 | 明确的 fragment layout 与访问次序 | 控制 GPR lifetime、operand reuse、FFMA order |
| MMA operation | C++ `+=`，交给 ptxas 展开 | architecture-tagged SIMT/MMA operator | 生成与流水级、fragment shape 匹配的指令序列 |
| Pipeline policy | 单 shared tile、两次 barrier | iterator + stage + wait policy 整体生成 | 在资源预算内维持 ready warp |

K10 并非“完全缺少这些层”：`loadFromGmem`、`processFromSmem` 和 lane 公式分别承担了
相似功能。它缺少的是**把这些层作为同一个资源/调度问题联合生成**的机制。增加 C++
类名不会自动提速；CUTLASS iterator 的价值是让 tile、layout、stage、fragment 与目标
架构共同决定最终 SASS。

## 10. 剩余约 9% 的原因排序

无法把 9% 精确拆成互不重叠的百分点。按现有测量证据排序：

| 排名 | 原因 | 置信度 | 证据 |
|---:|---|---|---|
| 1 | CTA 资源组织与整体 staging/pipeline 协同 | 高 | 256-thread/50 KB/202-reg cuBLAS，SASS 有 LDGSTS/DEPBAR；同 occupancy 下 issue 更高 |
| 2 | Shared→register fragment 供给效率与 ready-warp 数 | 高 | cuBLAS shared wavefront 少 39.83%，eligible warp 高约 17%，no-eligible 低 8.33 pp |
| 3 | control code、GPR lifetime 与 LDS/FFMA/整数指令的整体排程 | 中高 | cuBLAS 总指令和 ALU 更多但 dispatch/wait 更低、issue 更高 |
| 4 | CTA 数量、边界/epilogue 与固定开销 | 中低 | cuBLAS block 数减半；但 4096³ 主体计算占绝对多数 |
| 5 | 地址计算指令过多 | 低 | K10 静态地址整数指令占比低；cuBLAS 动态 ALU 更多仍更快 |
| 6 | 单 accumulator FFMA 串行依赖 | 低 | 128 条链；中位隔 127 条独立 FFMA，最短也隔 16 条 |

这个排序也修正了早期过强的表述：K17 失败只排除“简单 B-only async graft”，并没有
排除与 CTA/iterator/fragment 共同设计的成熟 pipeline。

## 11. 最多两个下一步验证实验

这里只提出，不在本轮实现。

### 实验 A：单点验证 256×128 CTA aggregation

目的：验证 cuBLAS 的更大 CTA 资源组织是否是 issue/readiness 收益的必要条件。

只改变：

```text
BM: 128 -> 256
NUM_THREADS: 128 -> 256
warp grid: 2×2 -> 4×2
```

保持 `BN=128, BK=16, warp tile=64×64, TM=8, TN=4, WNITER=4`、单 shared tile、
同步 global load、数学顺序和 shared layout。明确禁止同时加入 `cp.async` 或 double buffer。

这个配置仍是每 SM 约 8 个活跃 warp：当前是两个 4-warp CTA，候选是一个 8-warp CTA。
它不是 tile 暴力搜索，而是只验证“CTA aggregation”一个因素。重点看 shared wavefront、
long scoreboard、eligible warp、issue 和 FP32 pipe。若 issue/GFLOPS 不提升，则说明更大
CTA 本身不够，不能继续盲目追随 cuBLAS 的外形。

### 实验 B：单一 accumulator issue-order 变体

保持所有 tile、lane ownership、global/shared load 和每个输出的 K 累加顺序不变，只把
128 个独立 accumulator 的访问顺序分成 4 个交错 band，目标是提高最短复用距离并
观察 ptxas control code。只实现一个候选，不搜索排列。

门禁：registers/thread 不超过 200、无 spill、静态 FFMA/LDS 数不增加。由于当前最短
间隔已有 16 条 FFMA，这个实验优先级低于 A；若 issue/no-eligible 没有同步改善，立即
判失败，不继续手工排列。

不再提出 pointer-only 实验：K19 V2 已减少 IMAD 但下降，且 cuBLAS 动态 ALU 明显更多。

## 12. 从 90% 提升到 95%，最值得投入什么？

最值得投入的是：**以 256-thread、256×128 CTA 为单点入口，重做“CTA resource
envelope + shared/warp iterator + fragment schedule”的协同微内核，而不是继续修补
当前 K10。**

第一步只做实验 A，验证较大 CTA 是否能在不加 pipeline 的情况下提高 ready-warp 和
shared transaction 效率；验证成立后，才值得研究与该 CTA 匹配的集成 staging。这里
的“集成”不等于重新复制 K17：它必须和 A/B iterator、shared working set、wait policy
及寄存器预算一起设计。

目标 95% 约为 20.62 TFLOPS，相对当前 K10 需要约 4.7% 的稳定提升。cuBLAS 相对 K10
已有 8.33 个百分点的 issue-slots 优势，硬件上存在足够余量；但这部分余量来自整体
微内核，不太可能由 accumulator 重排或少几个地址指令单独兑现。

## 13. 复现命令与数据

```bash
./scripts/analyze_kernel_10_microarchitecture.py
RUN_TAG=my-run DEVICE=0 ./scripts/profile_kernel_10_microarchitecture_ncu.sh
./scripts/summarize_kernel_10_microarchitecture_ncu.py --tag my-run
./scripts/summarize_kernel_10_cublas_sass.py --tag my-run

# nvdisasm register lifetime 的底层入口：
cuobjdump -lelf build-prefetch-v1/sgemm
# 在临时目录执行 cuobjdump -xelf all 后：
nvdisasm -c -plr -lrm count <包含K10的cubin>
```

数据：

- [K10 静态依赖与 register lifetime](../results/rtx3080-k10-microarchitecture-static-20261012.csv)
- [K10/cuBLAS Nsight 对照](../results/rtx3080-k10-cublas-microarchitecture-nsight-20261012.csv)
- [K10/cuBLAS SASS 指令族汇总](../results/rtx3080-k10-cublas-sass-summary-20261012.csv)

正常性能仍引用不受 profiler replay 干扰的
[K19 三轮交错 benchmark](../results/rtx3080-k19-fragment-prefetch-fair-20261011.csv)。
