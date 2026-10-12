# RTX 3080 Kernel 10：Shared XOR 与 B-only `cp.async` 实验

日期：2026-10-11

环境：RTX 3080 10 GB（SM86）、CUDA 12.4、Nsight Compute 2024.1

问题：K10 距离同口径 strict FP32 cuBLAS 约 10% 的差距，是否主要来自 shared-memory
布局，或者可以用一个轻量 B-only 异步流水补上？

## 1. 结论先行

两个假设都经过了实际代码、正确性、三轮 benchmark、SASS 与 Nsight 验证，但都没有
成为新的默认 kernel：

- **K16 A-tile XOR swizzle 失败。** K10 的 `AS_PAD=4` 已使 shared-load conflicts 为 0；
  9 个 XOR 配置全部反而产生 134.22 M load conflicts，最好配置也比 K10 慢 18.35%。
- **K17 B-only `cp.async` 正确实现了异步搬运，但仍失败。** 强制恢复 BK 循环展开后，
  它从约 17.46 提高到 18.22 TFLOPS，但仍比 K10 慢 7.23%。
- K17 没有违反资源门槛：190 registers/thread，只比 K10 多 6；24,832 B static
  shared/block；没有 local-memory spill。回退来自更差的片上数据供给和缓存行为，而不
  是简单的寄存器爆炸。
- 默认最快实现仍然是 **K10，19.64 TFLOPS**；同轮 strict FP32 cuBLAS 为
  **21.67 TFLOPS，K10 达到 90.66%**。

因此，本轮没有用一次偶然峰值把 K16/K17 宣布为优化成功，也没有把它们替换成默认
K10。实验代码作为独立 kernel 保留，用于复现失败原因。

> 2026-10-12 边界修正：这里排除的是 **B-only cp.async graft**，不是所有
> global→shared pipeline。后续直接 profile 到的 strict cuBLAS SIMT 内核使用
> `LDGSTS/LDGDEPBAR/DEPBAR`，但同时采用更大的 CTA、50.176 KB shared 和 202
> registers/thread；其收益来自整体协同，不能与 K17 的单因素嫁接等同。

## 2. 实验边界与公平口径

所有正式性能比较固定为：

```text
M=N=K=4096, alpha=0.5, beta=3
Release: -O3 -DNDEBUG, sm_86
seed=20261011
每次预热和计时调用前恢复相同 C，恢复 copy 不计时
warmup=10, repeat=50
strict FP32 cuBLAS / K10 / K16 / K17 三轮轮换顺序
```

K16 只改变 A tile 的 shared-memory 物理地址。K17 只改变 B 的 global→shared 搬运与
buffer 生命周期。两者都保留 K10 的 BM/BN/BK、warp tile、thread mapping、寄存器
结果 tile、`alpha/beta` 和 FFMA 数学顺序。

K10 在重构前的已验证二进制和最终构建中的 SASS 规范化哈希均为：

```text
f1d18f1ff7890e7c983c63d6ce58902694aea71d8e4aee098ce5e66a0dc2c459
```

这证明新增实验入口没有暗中改变 K10 的机器码。

## 3. K16：为什么直觉合理，结果却相反

K16 使用以下物理地址：

```cpp
swizzled_m = m ^ ((k & mask) << shift);
index = k * (BM + AS_PAD) + swizzled_m;
```

写 A 和读 A 都调用同一个映射。固定 `k` 时 XOR 自反，因此逻辑矩阵和浮点计算顺序
不变。搜索严格限制为 `mask={1,3,7}`、`shift={1,2,3}` 九组。

### 3.1 三轮性能与资源

| 配置 | GFLOPS | 相对 K10 | Registers | Occupancy | Issue | FMA active |
|---|---:|---:|---:|---:|---:|---:|
| K10 | 19,631.4 | 基线 | 184 | 16.56% | 79.20% | 73.51% |
| m1s1 | 15,780.5 | -19.62% | 247 | 16.21% | 63.18% | 55.95% |
| m1s2 | 15,852.7 | -19.25% | 249 | 16.21% | 63.34% | 56.09% |
| **m1s3** | **16,029.5** | **-18.35%** | 243 | 16.21% | 64.08% | 56.72% |
| m3s1 | 15,971.6 | -18.64% | 227 | 16.20% | 64.49% | 56.61% |
| m3s2 | 15,899.2 | -19.01% | 216 | 16.18% | 64.69% | 56.64% |
| m3s3 | 15,918.6 | -18.91% | 216 | 16.18% | 64.64% | 56.63% |
| m7s1 | 15,720.7 | -19.92% | 248 | 16.22% | 65.44% | 56.31% |
| m7s2 | 15,471.8 | -21.19% | 248 | 16.22% | 64.15% | 55.16% |
| m7s3 | 15,747.3 | -19.78% | 248 | 16.23% | 65.44% | 56.34% |

所有配置的 local load/store sectors 都是 0，不能把回退归因于 spill。真正的问题是
额外 XOR/地址计算和被破坏的 bank 映射。

### 3.2 Bank conflict 证据

| 指标 | K10 | K16 m1s3 | 变化 |
|---|---:|---:|---:|
| Shared-load conflicts | **0** | 134,217,728 | 从无到有 |
| Shared-store conflicts | 16,777,216 | 16,777,216 | 无改善 |
| Shared-load wavefronts | 335,546,417 | 536,875,336 | +60.0% |
| Shared-store wavefronts | 50,331,648 | 50,331,648 | 无改善 |
| LSU instructions | 130,293,760 | 230,957,056 | +77.3% |

`m7s2` 更差：store conflicts 增到 50,331,648，store wavefronts 增到 83,886,080。
这回答了第一个核心问题：**K10 剩余差距不来自 A shared-load 布局；现有 padding 已经
把这部分读冲突清零。** 仍存在的 16.78 M store conflicts 是次要成本，不能用这个
XOR 公式消除。

## 4. K17：只异步搬 B，而不是复制完整 matmul4

K17 的 shared memory 为：

```text
As[BK][BM+4]       // 单缓冲，同步 float4 load
Bs[2][BK][BN]      // 双缓冲，16-byte cp.async.ca
```

流水顺序为：

1. 同步加载 A0，异步加载 B0，`commit_group`、`wait_group 0`、block barrier；
2. 发出 B(next) 的 `cp.async`；
3. 计算当前 A/B tile；
4. 所有线程读完 A 后 barrier，再同步加载 A(next)；
5. `wait_group 0` 后 barrier，交换 B buffer。

A 没有双 shared buffer，也没有 next-A register fragment；因此没有复制 K11/K12 或
matmul4 的完整 producer/consumer 结构。16-byte global 与 shared 地址都由当前
`N % 4 == 0`、`BN=128` 和 CUDA allocation 对齐条件保证。

### 4.1 一个重要的编译器陷阱

最初 K17 使用运行时 `currentBuffer` 后，ptxas 不再完全展开 `BK=16` 的计算循环：

| K17 版本 | Static SASS FFMA | Registers | 三轮/筛选 GFLOPS |
|---|---:|---:|---:|
| 未强制展开 | 256 | 176 | 17,459.3（三轮） |
| **显式 `#pragma unroll`** | **2,176** | **190** | **18,221.9（三轮）** |
| K10 | 2,176 | 184 | 19,642.7（三轮） |

显式展开让 K17 提升约 4.4%，同时保持 190 registers、0 spill，证明关键依赖和编译器
调度确实重要。这个 pragma 只放在 K17 的独立计算函数；K10 SASS 哈希保持不变。

`.cg` 和 `.ca` 缓存策略都做过受控检查。`.cg` 的一次筛选约 17.69 TFLOPS；`.ca`
保留后仍未解决 L2/DRAM 回退。缓存后缀不是隐藏的性能开关。

### 4.2 最终三轮公平性能

| 实现 | 平均时间 | 平均 GFLOPS | 相对 K10 | 相对 strict cuBLAS |
|---|---:|---:|---:|---:|
| strict FP32 cuBLAS | 6.3433 ms | 21,667.1 | 110.31% | 100.00% |
| **K10** | **6.9970 ms** | **19,642.7** | **100.00%** | **90.66%** |
| K16 m1s3 | 8.5323 ms | 16,108.2 | 82.01% | 74.34% |
| K17 B-only async | 7.5423 ms | 18,221.9 | 92.77% | 84.10% |

K17 的每轮范围是 18,193.2–18,268.4 GFLOPS；稳定低于 K10 的
19,609.1–19,664.7 GFLOPS，不是噪声内结果。

## 5. 最终 Nsight 对照

Nsight 会重放 kernel，Profiler duration 不能替代上表的正常 benchmark。它用于比较
同口径计数器：

| 指标 | K10 | K16 m1s3 | K17 final |
|---|---:|---:|---:|
| Registers/thread | 184 | 243 | 190 |
| Static shared/block | 16,640 B | 16,640 B | 24,832 B |
| Local load/store | 0 / 0 | 0 / 0 | 0 / 0 |
| Achieved occupancy | 16.60% | 16.21% | 16.54% |
| SM throughput | 74.58% | 60.62% | 66.07% |
| FMA pipe active | 73.73% | 56.67% | 66.12% |
| Issue active | 79.15% | 64.03% | 71.16% |
| Eligible warps/scheduler | 1.356 | 1.093 | 1.266 |
| Long scoreboard | 0.086 warp/cycle | 0.160 | 0.189 |
| Short scoreboard | 0.029 warp/cycle | 0.038 | 0.135 |
| L2 hit rate | 77.11% | 72.74% | 53.80% |
| DRAM read | 941.8 MB | 1,202.7 MB | 1,647.7 MB |
| DRAM throughput | 17.03% | 16.56% | 25.80% |
| `LDGSTS` executed | 0 | 0 | 4,194,304 |

K17 确实执行了 `LDGSTS`，所以不能说异步复制没有生效。它还把普通 shared-store
wavefronts 从 50.33 M 降到 33.55 M。但收益没有传到 FFMA：issue active 低 7.99 个
百分点，FMA active 低 7.61 个百分点，eligible warps 下降，long/short scoreboard
显著上升。与此同时 L2 hit 下降、DRAM read 增加。

Barrier stall 并没有上升，反而略低。因此 K17 的失败不能简化成“多了一次 barrier”；
主要是 B-only 流水没有覆盖仍然同步的 A load，又引入 buffer/wait 生命周期和更差的
缓存时序。K10 本来只用约 17% DRAM throughput，隐藏一半 global→shared 搬运并不是
当前最值钱的优化。

## 6. 正确性与安全性

- K16：9 配置 × 6 尺寸 × 3 beta，共 162/162 PASS；
- K17 final：6 尺寸 × 3 beta，共 18/18 PASS；
- 误差门槛保持原来的 absolute error 0.01；
- 没有改变 FP32 累加顺序或放宽阈值；
- 9 个 K16 配置和最终 K17 的 Compute Sanitizer memcheck 均为 0 errors；
- K16/K17 的 ptxas/cubin 资源统计均为 local 0、stack 0。

## 7. 为什么 matmul4 式双缓冲可能在别处有效，在这里却不一定

双缓冲适合“可隐藏的内存等待足够大、额外资源和调度成本足够小”的 kernel。不同 GPU、
tile、warp 数、寄存器预算和缓存状态会改变这个平衡。本机 K10 已经具有：

- 128 个线程和 128×128×16 tile；
- 184 registers/thread、约 16.6% occupancy；
- 很高的数据复用，DRAM throughput 只有约 17%；
- 完全展开的密集 FFMA 主体。

完整 matmul4 风格通常还会双缓冲 A/B shared tile和寄存器 fragment。这里只双缓冲 B，
已经增加 8 KB shared，并出现 scoreboard/L2 回退；若再加入 A buffer 和完整 register
prefetch，很容易超过 200 registers 或增加更多同步生命周期。因此“同样叫双缓冲”
不能跨 GPU 和微内核直接推断收益。

## 8. 剩余 10% 到底来自哪里

本轮能测量并排除的是：

1. **不是 DRAM bandwidth bound。** K10 DRAM throughput 约 17%，K15 已证明大幅减少
   DRAM read 也不提高吞吐。
2. **不是 K10 的 shared-load bank conflict。** 它实测为 0；K16 让它变坏。
3. **不是简单增加异步复制就能补上。** K17 真的执行 `cp.async`，却降低 issue/FMA。
4. **184 registers 是有目的的。** K10 无 spill；此前压寄存器换 occupancy 会增加
   指令，K17 190 registers 也没有因资源门槛失败。
5. **主要可见差距是 instruction issue 与片上 latency hiding。** K10 仍只有约
   79.15% issue active、73.73% FMA pipe active，并且低 occupancy 下可用于遮蔽依赖的
   eligible warps 有限。

不能仅凭 K10 计数器把 cuBLAS 的 9.34% 优势精确拆成若干百分比；那需要对 cuBLAS
实际选择的 SIMT kernel 做同样的 SASS/source-counter 依赖分析。当前证据支持的表述是：

> 剩余差距主要来自 cuBLAS 更成熟的微内核指令调度、寄存器/shared 数据供给和延迟
> 隐藏，而不是一个尚未打开的带宽或 bank-conflict 开关。

## 9. 后续 K18/K19 与微内核审计

当时优先级最高的是 **shared→register 的细粒度调度**，而不是第三套 global→shared
双缓冲：

1. 从 K10 SASS 中量化每组 LDS 到消费 FFMA 的依赖距离；
2. 只预取下一 `dotIdx` 的一小部分 B fragment，严格限制最多 +16 registers；
3. 保持单份 As/Bs，不增加 block barrier；
4. 每次检查静态 FFMA 是否仍为 2,176、动态 FFMA 总量是否不变；
5. 只有 issue、eligible warps 和 FMA active 同时改善，且三轮至少稳定提升约 1.5%，
   才保留实现。

这比完整 register ping-pong 更审慎：当前 `regM[8] + regN[16]` 若全部复制，理论上先
增加 24 个 float registers，容易越过 200-register 门槛。可先尝试 B-only fragment
的 16-register 上限，或把 WN 子块拆成更小的预取批次。

后续 K18 已先验证“不改 pipeline、只改 warp fragment ownership”这一更小假设。
5 个手工变体均未稳定快于 K10，简单 lane 重排或加宽 N fragment 不是答案。当时进一步
收敛为保持现有 ownership，直接分析 LDS→FFMA 依赖距离与 load/accumulator
调度；见 [K18 Warp Fragment Layout 实验](08-K10-Warp-Fragment-Layout调优.md)。

K19 随后完成了上述 B fragment 预取验证，三个版本均回退 6%–7%。2026-10-12 的
微内核审计又直接对照 strict cuBLAS：K10 已有 128 条独立 accumulator 链和 ptxas
自动 LDS/FFMA 调度；cuBLAS 的优势主要来自更大的 CTA 资源包络、较少 shared-load
wavefront 和集成流水。完整结论见
[K10 微内核与 cuBLAS 差距分析](10-K10-微内核与cuBLAS差距分析.md)。

## 10. 复现命令与数据

```bash
# K16 九组布局检查与三轮性能
./scripts/verify_a_smem_xor_swizzle.py
RUN_TAG=my-k16 DEVICE=0 ./scripts/kernel_16_xor_audit.sh

# 单个 K16 配置的六尺寸 × 三 beta
BUILD_DIR="$PWD/build-xor-m1-s3" CONFIG=m1s3 RUN_TAG=my-correctness \
  DEVICE=0 ./scripts/kernel_16_correctness_matrix.sh

# K17 构建与正确性
cmake -S . -B build-k17 -G "Unix Makefiles" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_FLAGS_RELEASE="-O3 -DNDEBUG"
cmake --build build-k17 --target sgemm --parallel 2
RUN_TAG=my-k17 DEVICE=0 ./scripts/kernel_17_correctness_matrix.sh

# strict cuBLAS / K10 / best K16 / K17 三轮轮换
RUN_TAG=my-fair DEVICE=0 ./scripts/kernel_16_17_fair_audit.sh

# K10、九组 K16、K17 的同口径 Nsight 报告（耗时较长）
RUN_TAG=my-ncu DEVICE=0 ./scripts/profile_kernel_16_17_ncu.sh

# 最终 K17 memcheck
SGEMM_BENCH_SIZE=128 SGEMM_WARMUP=0 SGEMM_REPEAT=1 \
SGEMM_SEED=20261011 SGEMM_RESET_C_EACH_REPEAT=1 SGEMM_BETA=3 \
compute-sanitizer --tool memcheck --error-exitcode 99 ./build-k17/sgemm 17
```

仓库内结果：

- [K16 九组性能与 Nsight 汇总](../results/rtx3080-k16-xor-summary-20261011.csv)
- [最终三轮公平 benchmark](../results/rtx3080-k16-k17-fair-20261011.csv)
- [K10/K16/K17 Nsight 对照](../results/rtx3080-k10-k16-k17-nsight-20261011.csv)
- [正确性与 memcheck 汇总](../results/rtx3080-k16-k17-correctness-20261011.csv)

每次脚本运行都使用新 `RUN_TAG`，拒绝覆盖同名历史结果。
