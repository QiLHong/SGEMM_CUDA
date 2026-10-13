# 公开 CUTLASS SIMT 对照与主循环成本审计

日期：2026-10-13。实际执行于同一台 RTX 3080 10 GB、SM86、CUDA 12.4、驱动 550.127.05。本轮不是 K24，也没有修改 K10 或替换默认 kernel。

## 结论

公开 CUTLASS v3.5.1 的 256×128、BK=8、4-stage SIMT 对照平均 19.956 TFLOPS，比 K10 高 1.85%，但只有 strict cuBLAS 的 92.65%。它复现了 cuBLAS 较少的 Shared-load wavefront，却没有复现其全部吞吐优势。不能把“完整 CUTLASS 参数”当作 cuBLAS 实现的替代品。

最重要的新证据不是更高的单次 GFLOPS，而是逐指令成本：

- K10 主循环实际 LDS 总量与参考实现接近，但每条 LDS 的硬件服务成本不同。
- K10 的 96 个主循环 `LDS.128` 位置，32 个每次产生 2 个 wavefront，64 个每次产生 4 个；公开 CUTLASS 和 cuBLAS 的 48 个主循环位置全部每次 2 个。96/48 是 BK16/BK8 的静态循环长度差异，不是动态读取量减半。
- 三者有效线程 FFMA 同为 68,736,253,952，Shared-load bank conflict 都为零；所以差异属于访问组织的基础成本，不是传统 bank conflict。
- 公开 CUTLASS 比 cuBLAS 多 102,416,384 条动态 warp 指令（+4.26%）。其中 44,040,192 条是 `@!PT LDS RZ, [RZ]`，predicated-on thread count 为 0，不读数据，但出现在执行指令计数中。不能当成有效 LDS，也不能未经验证直接删除。

这支持继续研究具体的 lane/fragment 访问组织；不支持笼统宣称 layout、pipeline 或 global latency 已被完全排除。

## 实验与复现边界

新增独立 [reference 程序](../../scripts/cutlass_simt_reference.cu)，直接包含未修改的 K10/K21 头文件，不走生产 dispatcher。四个模式共用同一输入、同一 CUDA event 计时与 C 复位流程。

- CUTLASS tag：v3.5.1，commit `f7b19de32c5d1f3cedfc735c2849f12b537522ee`；源码在 `/tmp/sgemm-cutlass-v3.5.1`，未将第三方源码复制进仓库。
- `OpClassSimt`、`Sm80`，实际编译 `sm_86`；CTA 256×128×8，warp 64×64×8，256 threads，4 stages，alignment A/B=1，scalar epilogue，`GemmIdentityThreadblockSwizzle<1>`，split-K=1。
- A/B/C 均为 row-major；公开类型没有模拟 cuBLAS 私有 grid swizzle、iterator 或编译配置。核函数名称相近不证明内部映射完全相同。
- `nvcc -O3 -DNDEBUG -std=c++17 -arch=sm_86 -lineinfo -DSGEMM_K10_MIN_BLOCKS=2 -Xptxas=-v`，不使用 fast-math。
- seed=20261013，生成方法与原程序相同；alpha=0.5，性能测试 beta=3；每次调用前从初始 C 复位，D2D 不计入 GEMM 时间。
- strict cuBLAS：PEDANTIC math、COMPUTE_32F_PEDANTIC、GEMM_DEFAULT；取消 `NVIDIA_TF32_OVERRIDE`。SASS 中没有 HMMA/MMA。
- 4096³，10 次预热、每组 50 次计时、7 轮轮换顺序；额外先运行 K10 使 GPU 离开 idle。GPU boost 未锁定，温度/频率前后快照保存在 fair.csv；结果只与本轮基线比较。

公开 CUTLASS 的列主序输出封装会交换操作数；因此不能简单根据 cuBLAS API 的 N/M 交换推断内部 CTA 一定是 128×256。本轮明确采用 row-major 256×128 参考，未证明私有 cuBLAS 的 CTA 方向。参见 [官方 Gemm 封装](https://github.com/NVIDIA/cutlass/blob/v3.5.1/include/cutlass/gemm/device/gemm.h)。

## 正确性与资源

K10、K21 V1、公开 CUTLASS 各通过 128/256/512/1024/2048/4096 × beta=0/1/3，共 **54/54 PASS**；原始 absolute error=0.01，没有放宽。六尺寸最大误差约 0.000610352；4096 下公开 CUTLASS 与 strict 对照相同，K10/K21 最大误差约 0.000007629。这里只指当前固定输入，不声称对任意输入 bitwise 相同。

公开 CUTLASS 在 128、256、4096，beta=3 的 Compute Sanitizer memcheck **3/3 PASS、0 errors**。没有执行 racecheck/synccheck，不能把 memcheck 描述成所有同步性质的证明。

| 模式 | Registers/thread | 编译/显式 Shared | NCU Shared/block | spill/local sectors |
|---|---:|---:|---:|---:|
| K10 | 184 | 16,640 B | 17,664 B | 0 |
| K21 V1 | 190 | 16,640 B | 17,664 B | 0 |
| 公开 CUTLASS | 254 | 49,152 B | 50,176 B | 0 |
| strict cuBLAS | 202 | NCU dynamic=49,152 B | 50,176 B | local=0；无 ptxas 日志 |

NCU 总 Shared 包含本次报告的 1,024 B driver 分配。公开参考超过此前候选的 220 registers/thread 限制，**仅作为诊断参考，不接受为优化候选**。没有盲目施加 maxrregcount；254 对 202 的差异不等于已经证明寄存器压力导致全部性能差距。

## 七轮交错性能

| 模式 | 平均 GFLOPS | 七轮范围 | 样本标准差 | 平均 ms | strict 比例 |
|---|---:|---:|---:|---:|---:|
| strict cuBLAS | 21,540.2 | 21,467.2–21,704.1 | 88.4 | 6.3807 | 100% |
| K10 | 19,594.2 | 19,508.4–19,675.3 | 56.3 | 7.0143 | 90.97% |
| K21 V1 | 19,715.1 | 19,589.1–19,824.5 | 74.9 | 6.9714 | 91.53% |
| 公开 CUTLASS | 19,956.2 | 19,908.5–20,004.6 | 41.2 | 6.8870 | 92.65% |

公开 CUTLASS 每轮均快于 K10，但没有达到 95%。95% 对应约 20,463.2 GFLOPS，比 K10 还需 +4.43%，比当前公开参考还需 +2.54%。小幅收益不能跨旧实验日期/时钟状态直接比较。

## 冷缓存与稳态 Nsight

通过 NVTX `measure/` 只选中预热后的测量 launch，排除 correctness/reference 和 warmup。旧实验文件不覆盖。

- cold：kernel replay、cache-control=all、clock-control=base。
- steady：application replay、cache-control=none、clock-control=none；每次 replay 都重新执行相同输入和预热。此组不是固定时钟实验，counter 是单组采样，不作为七轮性能置信区间。
- NCU 插桩时 reference 程序打印的 event 时间包含 profiler 影响，**不用于性能表**。

| 策略/模式 | Issue % | FP32 active % | Eligible warps | Occupancy % | Long scoreboard | Short scoreboard |
|---|---:|---:|---:|---:|---:|---:|
| cold K10 | 78.97 | 73.41 | 1.352 | 16.53 | 0.0818 | 0.0294 |
| cold K21 | 78.57 | 73.24 | 1.362 | 16.56 | 0.1077 | 0.0386 |
| cold CUTLASS | 85.99 | 74.50 | 1.549 | 16.66 | 0.0077 | 0.0651 |
| cold strict | 87.31 | 78.56 | 1.579 | 16.66 | 0.0237 | 0.0832 |
| steady K10 | 78.76 | 73.26 | 1.338 | 16.58 | 0.1093 | 0.0299 |
| steady K21 | 78.30 | 72.98 | 1.352 | 16.60 | 0.1275 | 0.0385 |
| steady CUTLASS | 85.83 | 74.37 | 1.547 | 16.66 | 0.0091 | 0.0649 |
| steady strict | 87.20 | 78.49 | 1.578 | 16.66 | 0.0239 | 0.0803 |

scoreboard 单位是 warp/cycle，不是百分比。`issue_inst0` 在 CSV 命名为 no_issue，不把无发射周期一律解释成没有 eligible warp；也不把它与 issue 的互补变化当成两份独立证据。

cold/steady 的关键动态工作量一致：K10 load wavefront 335.55 M，CUTLASS/strict 201.90 M；三者有效线程 FFMA 68.736 B。公开参考虽然 issue 大幅高于 K10，FP32 active 却只略高，说明更多 issue 并非全部用于有效计算。

steady DRAM read：K10 1.024 GB、K21 1.270 GB、公开 CUTLASS 1.653 GB、strict 0.575 GB；L2 hit 分别为 75.74%、68.08%、84.25%、86.06%。L2 hit 是整体 sector 指标，不可直接用它与 DRAM 比值推算某一操作数的流量；两套 replay 的缓存/频率条件也不同。公开参考与 cuBLAS 的 tile scheduling 并未匹配，当前无法唯一解释其 DRAM 差异。

## 主循环成本：不仅看整函数静态数量

SourceCounters + InstructionStats 另外采集 cold 策略，逐 PC 求和与主报告的总动态指令数完全一致：

| 动态 warp 指令，M 次 | K10 | 公开 CUTLASS | strict cuBLAS |
|---|---:|---:|---:|
| 总计 | 2,336.014 | 2,504.417 | 2,402.001 |
| FFMA | 2,148.008 | 2,148.008 | 2,148.008 |
| 有效 LDS 类指令位置执行 | 100.663 | 101.212 | 101.212 |
| 恒假谓词 LDS（无有效线程） | 0 | 44.040 | 0 |
| LDGSTS | 0 | 25.313 | 25.313 |
| IMAD | 15.749 | 11.563 | 7.004 |
| IADD3 | 3.240 | 49.582 | 40.837 |
| LEA | 3.224 | 12.657 | 0.045 |
| BAR | 2.097 | 2.236 | 2.241 |

公开 CUTLASS/strict 额外 LDS 包括 epilogue；主循环有效 `LDS.128` 每 K step 都是每 warp 6 条。K10 每 BK16 有 96 条，参考每 BK8 有 48 条。逐 PC 的波前结果解释了为何指令数几乎相同而 wavefront 不同：K10 平均每条主循环 LDS 为 20/6 个 wavefront，参考为 12/6。不是通过改变数学计算量或降低精度省下来的。

静态 SASS 表另存。解析器显式区分 `@!PT LDS`，不把它误计为有效 load，也不漏掉其指令位置。公开参考的 21 个恒假 LDS 位置各执行 2,097,152 次，共 44,040,192 次；cuBLAS 虽有 4 个相似静态位置，但本次执行路径没有经过。产生原因是否属于调度填充/编译器约束仍需 PTX/编译器对照，不依据外观删指令。

### 异步写入不能用普通 STS counter 代替

SourceCounters 将公开参考 LDGSTS 的 Shared wavefront 归因到约 192.94 M，strict 约 109.05 M；对应 excessive 约 167.77 M 与 83.89 M。普通 `pipe_lsu_mem_shared_op_st` counter 并不完整覆盖 LDGSTS 路径，因此“普通 Shared-store conflict 为零”不证明异步目的端没有额外服务成本。这些是逐指令归因/模型指标，不能与独立硬件 replay 的不同 pipe 计数无条件相加，也不能直接换算损失时间。

逐 PC stall CSV 已保存。公开 CUTLASS 有主循环 LDS 位置出现 MIO stall 样本；K10 有 STS 位置出现 long-scoreboard 样本。样本数不是周期数，受采样、插桩及执行路径影响；不能将单个最高样本位置直接解释为总性能瓶颈。

## 公共参考的 fragment 映射究竟不同在哪里

官方 `MmaSimtPolicy` 使用 4×8 lane lattice 和 `RowMajorInterleaved<2>`；每线程是 **16 个 A、8 个 B、128 个 accumulator**，不是从核函数名字推出来的 8×4 accumulator。

令 lane 为 `l`，则 warp 内：

```text
lane_m = 2*(l/16) + l%2
lane_n = (l%16)/2
A rows = 4*lane_m + 16*s + i    s=0..3, i=0..3
B cols = 4*lane_n + 32*t + j    t=0..1, j=0..3
C fragment = A rows × B cols
```

这把一个线程的 M fragment 分成四组，而 K10 是连续 8 行、N 方向四组。32 lane 的完整映射保存在 lane-mapping.csv，CPU 枚举验证两种 ownership 都无重叠且完整覆盖 64×64 warp tile。它只描述公开 CUTLASS，不声称重建了私有 cuBLAS 的 ownership。

K18 所有变体强制 `WMITER=1`；本参考对应 M 方向四组，因此不是已做过的 K18 lane permutation。K23 又冻结了 ownership，本来就无法测试这个维度。前面的失败不排除此映射。

源码依据：[MmaSimt iterator](https://github.com/NVIDIA/cutlass/blob/v3.5.1/include/cutlass/gemm/warp/mma_simt_tile_iterator.h)、[SIMT policy](https://github.com/NVIDIA/cutlass/blob/v3.5.1/include/cutlass/gemm/threadblock/default_mma_core_simt.h)、[细粒度 multi-stage 主循环](https://github.com/NVIDIA/cutlass/blob/v3.5.1/include/cutlass/gemm/threadblock/mma_multistage.h)。

## 下一步：只提出一个受控移植实验

建议先保留 K10 的 BM/BN/BK=128/128/16、128 threads、global loader、shared layout 和单阶段同步，只移植上述完整的 fragment ownership + Shared iterator + 匹配 epilogue：逻辑 TM=TN=4、WMITER=4、WNITER=2、lane interleave=2，仍为 128 accumulator/thread，并保持每个 C 的 K 累加顺序。

这不是继续随机 lane 搜索：唯一假设是，已在公开参考中测到的 LDS 每指令 wavefront=2，能否在 K10 外形下复现。先看 per-PC LDS 工作量是否真的降低，再看 7 轮 GFLOPS、FP32/issue、寄存器与 spill。超过 220 寄存器、出现 spill 或没有稳定收益就停止，不同时加四阶段/大 CTA。

如果该移植失败，不能直接将差距归因于 warp specialization；应回到已测出的 LDGSTS 目的端成本、恒假指令、地址/谓词控制与寄存器生命周期，逐项设计对照。当前结果能证明结构差异，尚不能给剩余约 9% 精确分摊百分比。

## 数据与命令

所有版本化 CSV 位于 [本轮结果目录](../results/cutlass-reference-20261013/)：performance/fair、correctness、memcheck、resources、nsight、static-sass、dynamic-opcodes、memory-instructions、mainloop-lds、pc-stalls、lane-mapping、sha256 与 environment。

原始 ptxas 日志、逐 PC CSV、SASS、`.ncu-rep`、sanitizer 日志保留于 `benchmark_results/cutlass-reference-20261013/`，该目录被 git 忽略；没有覆盖 K15–K23 的历史数据。K23 之前尚未提交的改动也保留。

```bash
git clone --depth 1 --branch v3.5.1 https://github.com/NVIDIA/cutlass.git /tmp/cutlass-v351-repro
bash scripts/build_cutlass_reference.sh /tmp/cutlass-v351-repro benchmark_results/cutlass-repro-new
# 后续 --binary 使用该新目录下 reference 的绝对路径。
python3 scripts/run_cutlass_reference.py correctness --binary "$PWD/benchmark_results/cutlass-repro-new/reference" --output benchmark_results/cutlass-repro-new/tests
python3 scripts/run_cutlass_reference.py fair --binary "$PWD/benchmark_results/cutlass-repro-new/reference" --output benchmark_results/cutlass-repro-new/timing --rounds 7
python3 scripts/run_cutlass_reference.py profile --binary "$PWD/benchmark_results/cutlass-repro-new/reference" --output benchmark_results/cutlass-repro-new/profiles
python3 scripts/run_cutlass_reference.py source --binary "$PWD/benchmark_results/cutlass-repro-new/reference" --output benchmark_results/cutlass-repro-new/source
python3 scripts/run_cutlass_reference.py memcheck --binary "$PWD/benchmark_results/cutlass-repro-new/reference" --output benchmark_results/cutlass-repro-new/sanitizer
python3 scripts/summarize_cutlass_reference.py benchmark_results/cutlass-repro-new --output docs/results/cutlass-repro-new --cutlass-dir /tmp/cutlass-v351-repro
python3 scripts/analyze_cutlass_reference_source.py benchmark_results/cutlass-repro-new/source --output docs/results/cutlass-repro-new
```

不要同时运行计时、NCU 和 sanitizer。采样策略解释参见 [NVIDIA Nsight Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#cache-control)。
