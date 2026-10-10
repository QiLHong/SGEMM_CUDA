# Kernel 10：Grouped Grid Swizzle 调优实验

日期：2026-10-10
环境：NVIDIA GeForce RTX 3080 10 GB（SM86）、驱动 550.127.05、CUDA 12.4、
Nsight Compute 2024.1、Release `-O3 -DNDEBUG`。

## 1. 结论先行

本实验没有找到可保留为性能优化的 Grouped Grid Swizzle 参数。

- K10 三轮平均为 **19688.9 GFLOPS**。
- K15 的 `GROUP_M=1/2/4/8/16` 相对 K10 分别为
  **-0.12% / -0.28% / -0.42% / -0.84% / -1.19%**。
- 最接近的 G1 没有真正形成 M 方向分组，且差异处于测量波动内；不能声称改善。
- 最强分组 G16 把 L2 hit rate 从 **76.69%** 提高到 **86.62%**，并把
  DRAM read 从 **962.5 MB** 降到 **563.1 MB**，但正式吞吐仍下降 **1.19%**。
- 所有五种配置在 `beta=0/1/3` 和六个尺寸上均通过原有绝对误差 0.01
  的严格 FP32 cuBLAS 对照；Compute Sanitizer 的小尺寸检查均为 0 error。

因此，Grouped Grid Swizzle 改善了强分组下的跨 block 缓存局部性，却没有改善 K10
当前决定吞吐的瓶颈。K15 作为独立实验入口保留，不替换 K10，也不改默认最优配置。

## 2. 只改变了什么

K10 的配置保持不变：

```text
BM=128, BN=128, BK=16
WM=64, WN=64, WNITER=4
TM=8, TN=4, NUM_THREADS=128
AS_PAD=4, MIN_BLOCKS=2
```

实现把 K10 的数学主体提取为 `wt::runWarptilingCore()`，K10 仍用二维 grid：

```cpp
cRow = blockIdx.y;
cCol = blockIdx.x;
```

K15 使用一维 `pid`，只在进入同一数学主体前改变 tile 映射：

```cpp
gridM = ceil_div(M, BM);
gridN = ceil_div(N, BN);
groupSize = GROUP_M * gridN;
groupId = pid / groupSize;
firstM = groupId * GROUP_M;
actualGroupM = min(gridM - firstM, GROUP_M);
localPid = pid % groupSize;
cRow = firstM + localPid % actualGroupM;
cCol = localPid / actualGroupM;
```

尾部不足 `GROUP_M` 时使用 `actualGroupM`，避免重复、遗漏或越界 tile。独立 CPU
映射检查覆盖 7 种 grid（包括 `17x7`、`33x32` 尾组）和五个组宽，共 35 种组合，
每个输出 tile 恰好出现一次。

重构前后的 K10 cubin SASS SHA-256 完全相同：

```text
adf39f2da01e3d59cd5ef74f14283874e6b4c6b984b697a58dc4b00635d20b53
```

这比只对照源码更强：K10 的实际设备指令没有被公共主体重构改变。

## 3. 为什么这种映射理论上可能有效

在 4096³、128×128 输出 tile 下，grid 是 32×32。K10 的二维 grid 从语义上把
`x` 映射到 N、`y` 映射到 M；实际 block 的驻留顺序仍由硬件调度器决定，不能把
逻辑编号误写成严格执行顺序。

Grouped Grid Swizzle 让一段连续 pid 在有限的 M 行组内优先遍历 M：

- 相同 `cCol` 的相邻 tile 可以复用 B 的 K×BN 数据；
- 代价是改变原有沿 N 方向时对 A 的 BM×K 数据复用机会；
- 是否获益取决于 block 实际调度、L2 容量/分区、A/B 工作集和映射计算开销。

G1 是必要的控制组：其 tile 集合和逻辑次序等价于不分组，只多了一维映射计算。
G2/4/8/16 才逐步增强 M 方向分组。

## 4. 正确性与内存安全

每个 `GROUP_M=1/2/4/8/16` 都用固定随机 seed 20261010 测试：

- `M=N=K=128/256/512/1024/2048/4096`；
- `alpha=0.5`，`beta=0/1/3`；
- 每次调用前恢复同一份初始 C；
- 对照模式 13，即 pedantic strict FP32 cuBLAS；
- 沿用原阈值，未放宽绝对误差 0.01。

共 5×3×6=90 个组合，结果全部 PASS。另对五个组宽分别运行
Compute Sanitizer memcheck（128³、`beta=3`），全部为 `ERROR SUMMARY: 0 errors`。

版本化摘要见
[正确性 CSV](../results/rtx3080-k15-grouped-swizzle-correctness-20261010.csv)。完整终端日志保存在
`benchmark_results/kernel_15_group*_correctness_20261010-audit.log`；该目录按仓库规则
不提交，避免把体积较大的原始日志混入源码历史。

## 5. 公平性能实验

正式 benchmark 使用：

- 4096³、`alpha=0.5`、`beta=3`；
- strict FP32 cuBLAS、K10、K15 G1/G2/G4/G8/G16；
- warmup 10 次、计时 50 次、三轮循环移位交错执行；
- seed 固定为 20261010；
- 每个预热和计时调用前恢复 C，恢复 copy 不计时；
- `NVIDIA_TF32_OVERRIDE` unset；
- 每个方案使用相同 Release 选项，K15 只改变编译期 `GROUP_M`。

GPU 在正式样本中的温度为 36–45 °C，显存时钟为 9501 MHz，主要 SM 时钟为
1935–1965 MHz。CSV 中功耗是调用前后的瞬时采样，不是时间积分平均功耗，因此不用于
判断能效。

| 方案 | 三轮 GFLOPS | 平均 | 相对 K10 | 相对 strict FP32 |
|---|---|---:|---:|---:|
| strict cuBLAS | 21792.9 / 21544.6 / 21513.6 | 21617.0 | +9.79% | 100.00% |
| K10 | 19742.4 / 19651.8 / 19672.4 | 19688.9 | 基线 | 91.08% |
| K15 G1 | 19748.5 / 19632.2 / 19617.9 | 19666.2 | -0.12% | 90.98% |
| K15 G2 | 19712.0 / 19614.3 / 19577.4 | 19634.6 | -0.28% | 90.83% |
| K15 G4 | 19657.3 / 19582.2 / 19580.3 | 19606.6 | -0.42% | 90.70% |
| K15 G8 | 19548.1 / 19531.0 / 19489.1 | 19522.7 | -0.84% | 90.31% |
| K15 G16 | 19517.3 / 19457.5 / 19390.0 | 19454.9 | -1.19% | 90.00% |

原始逐轮数据见
[性能 CSV](../results/rtx3080-k15-grouped-swizzle-20261010.csv)，聚合结果见
[摘要 CSV](../results/rtx3080-k15-grouped-swizzle-summary-20261010.csv)。判断标准是跨轮
持续改善且最好达到约 1.5%；本次所有参数都是负收益，因此没有“挑最好的一次”报告
正结果。

## 6. Nsight Compute 证据

使用相同输入、seed、C 复位和 `beta=3`，对 K10、控制组 G1、最小真实分组 G2、
强分组 G16 运行 `ncu --set full`。Profiler 会重放 kernel，表中的 duration 只用于
同一采集设置内辅助比较，不替代上面的三轮 benchmark。

| 指标 | K10 | G1 | G2 | G16 |
|---|---:|---:|---:|---:|
| Compute (SM) throughput | 74.73% | 74.79% | 74.53% | 74.02% |
| FMA pipe active | 69.53% | 69.52% | 69.34% | 68.87% |
| Issue slots busy | 79.20% | 79.26% | 78.80% | 77.62% |
| No eligible | 20.76% | 20.88% | 21.11% | 22.42% |
| Eligible warps/scheduler | 1.36 | 1.36 | 1.35 | 1.31 |
| Achieved occupancy | 16.60% | 16.54% | 16.50% | 16.24% |
| L2 hit rate | 76.69% | 76.57% | 84.65% | 86.62% |
| DRAM read | 962.5 MB | 955.7 MB | 991.4 MB | 563.1 MB |
| DRAM throughput | 17.40% | 17.32% | 17.92% | 10.57% |
| Shared-store bank conflicts | 16.78 M | 16.78 M | 16.78 M | 16.78 M |
| Local load/store sectors | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| Registers/thread | 184 | 184 | 186 | 186 |
| Static shared/block | 16640 B | 16640 B | 16640 B | 16640 B |

完整摘要见
[Nsight CSV](../results/rtx3080-k15-grouped-swizzle-nsight-20261010.csv)。原始 `.ncu-rep`
保存在本机 `benchmark_results/k15-audit-{k10,g1,g2,g16}-full.ncu-rep`。

### 6.1 能确认的事实

1. K10 不是 DRAM bandwidth bound。Compute throughput 为 74.73%，DRAM 仅 17.40%；
   Nsight 也把 compute 标为较重的一侧。
2. 强分组确实改变了缓存行为。G16 的 L2 hit 提高 9.92 个百分点，DRAM read 减少
   约 41.5%，所以不能说 swizzle 完全没有作用。
3. 缓存改善没有转成 FFMA 吞吐。G16 的 FMA、issue、eligible warps 和 achieved
   occupancy 均略降，而 No Eligible 上升 1.66 个百分点。
4. shared-memory 冲突不受 grid 映射影响，四个方案都是 16.78 M；K15 也没有 spill。
5. G2 虽提高 L2 hit，却增加 L2 read sectors/requests，DRAM read 也略增。只看命中率
   会漏掉分母和总请求量变化，因此不能用单个 cache 百分比宣布成功。

### 6.2 不能从本次实验断言的内容

Nsight 的一次多 pass replay 不能证明每个小幅指标变化都由同一条指令单独造成；
逻辑 pid 顺序也不等于硬件严格按 pid 执行。可以确认的是：在当前 GPU、尺寸和完整
工作负载下，节省的 DRAM 流量不足以抵消 issue/调度侧损失，正式三轮吞吐没有收益。

## 7. 寄存器与 SASS

`cuobjdump --dump-resource-usage` 和 `--dump-sass` 给出：

| 方案 | Registers/thread | Shared | Stack/Local | 静态 SASS 指令行 |
|---|---:|---:|---:|---:|
| K10 | 184 | 16640 B | 0 / 0 | 2768 |
| G1 | 184 | 16640 B | 0 / 0 | 2792 |
| G2/G4/G8/G16 | 186 | 16640 B | 0 / 0 | 2816 |

G1 比 K10 多 24 条静态指令，真实分组多 48 条，并多用 2 个寄存器。原因是 K10 可
直接读取二维 `blockIdx.x/y`，K15 要执行运行时 grid 大小、除法/取模、尾组宽度和
二维坐标恢复。新增工作只在 block 入口执行，其动态指令占总工作量很小，不能把全部
1.19% 回退简单归因于“多了 48 条指令”；它同时改变了 tile 驻留/缓存顺序，并使
G16 的 eligible warps 和 issue 指标变差。

184→186 registers 没有改变理论上每 SM 两个 block、16.67% occupancy 的上限，也
没有 spill，但实测 achieved occupancy 从 K10 的 16.60% 降到 G16 的 16.24%。

## 8. 每个方案的判定

- **G1：不采用。** 性能 -0.12%，与 K10 资源相同但多映射指令；它证明 1D 控制组
  没有免费收益。
- **G2：不采用。** L2 hit 明显提高，但 DRAM read 未降低，性能 -0.28%。
- **G4：不采用。** 三轮都未超过同轮趋势，平均 -0.42%。
- **G8：不采用。** 平均 -0.84%，开始明显偏离 K10 波动区间。
- **G16：不采用。** 缓存改善最强，但平均 -1.19%，issue/no-eligible 指标更差。

K15 仍保留作可复现实验入口；默认 `GROUP_M=1` 只是最安全的控制值，不代表它优于
K10。没有继续无边界搜索 32/64 或混入 tile、launch-bounds、数学顺序等第二因素。

## 9. 三个核心问题

### 9.1 K10 的主要差距是否真的来自跨 Block 的 L2 复用不足？

**不是主要差距。** G16 证明跨 block L2 复用仍有改善空间，但在 DRAM read 减少
约 41.5% 后，吞吐没有提高。K10 的 90% 左右 strict-cuBLAS 水平更符合
compute / instruction-issue / data-supply 效率不足：FMA pipe 约 69.5%，Issue slots
约 79.2%，约 20.8% 周期没有 eligible warp，同时仍有 16.78 M shared-store bank
conflicts。L2 不足是次要可测因素，不是剩余约 9% 差距的单一解释。

### 9.2 Grouped Grid Swizzle 能否稳定提高 RTX 3080 FP32 SGEMM？

**不能。** 五个参数的三轮平均都低于 K10；实际形成分组后，组越大总体越慢。
最接近的 G1 也只是 -0.12% 的控制组结果，不满足持续正收益和约 1.5% 的采用门槛。

### 9.3 下一步优先 Shared Memory 布局还是 SASS 指令调度？

**先做受控的 SASS 指令调度/依赖链实验，Shared Memory 布局排第二。** 理由是本次
把外存流量显著压低后，性能仍随 issue slots、eligible warps 和 FMA utilization 一起
下降；这直接指向 FFMA 与 shared/global load 的发射间隔、地址计算和依赖链。下一步
应从 K10 保持数学顺序不变的小改动开始，观察 issue、No Eligible、FMA 和 SASS。

Shared-store bank conflicts 的 16.78 M 仍是明确问题，后续可单独尝试 lane/XOR
layout，但必须证明它降低冲突且不增加寄存器、地址指令或破坏向量化。不要同时重写
完整双缓冲，否则无法归因。

## 10. 复现命令

```bash
# 映射双射检查。
./scripts/verify_grouped_swizzle.py

# 自动构建 G1/G2/G4/G8/G16，并做三轮交错 benchmark；新时间戳拒绝覆盖旧数据。
DEVICE=0 ./scripts/kernel_15_grouped_swizzle_audit.sh

# 六尺寸 × beta=0/1/3；每个 build 分别运行。
for group in 1 2 4 8 16; do
  BUILD_DIR="$PWD/build-swizzle-g${group}" GROUP_M="${group}" \
    RUN_TAG="$(date +%Y%m%d-%H%M%S)" \
    ./scripts/kernel_15_correctness_matrix.sh
done

# 同口径 Nsight 示例。分别换 build 目录和 GROUP_M 即可复核 G2/G16。
SGEMM_BENCH_SIZE=4096 SGEMM_WARMUP=0 SGEMM_REPEAT=1 \
SGEMM_SEED=20261010 SGEMM_RESET_C_EACH_REPEAT=1 SGEMM_BETA=3 \
ncu --set full --kernel-name regex:sgemmWarptilingGrouped --launch-count 1 \
  -o "benchmark_results/k15-g1-$(date +%Y%m%d-%H%M%S)" \
  ./build-swizzle-g1/sgemm 15

# 资源和 SASS。
cuobjdump --dump-resource-usage build-swizzle-g1/sgemm
cuobjdump --dump-sass build-swizzle-g1/sgemm
```

原始运行中还保留了一次脚本失败日志：最初误用了 Bash 特殊数组名 `GROUPS`，被系统
组号覆盖。脚本已改为 `GROUP_M_VALUES`，失败运行没有生成性能 CSV，也没有覆盖任何
历史数据。保留这条记录是为了不把实验工具故障伪装成 GPU 结果。
