# K21：从 CUDA kernel 调参进入 GEMM micro-kernel 设计

日期：2026-10-12

环境：RTX 3080 10 GB（SM86）、CUDA 12.4、驱动 550.127.05、Nsight Compute
2024.1。全部性能数据来自同一 GPU；`NVIDIA_TF32_OVERRIDE` 未设置，参考为
`CUBLAS_COMPUTE_32F_PEDANTIC`。

## 1. 实验边界

K21 不修改 K10，而是建立两个独立版本：

| 版本 | CTA / threads | BK / shared stages | 唯一主要问题 |
|---|---|---|---|
| V1 iterator | 128×128 / 128 | 16 / 1 | 只把 K10 shared fragment 索引改成 `load()/advance()` iterator |
| V2 pipeline skeleton | 256×128 / 256 | 16 / 2 | 48 KiB shared 双阶段，A/B 都用 `cp.async`，下一 tile 与当前 FFMA 重叠 |

V1 保持 K10 的 warp 64×64、thread fragment、global load、shared layout、barrier 和
浮点累加顺序不变。V2 使用 4×2 warp 网格，每个 warp 仍负责 64×64，每线程仍有
128 个 accumulator；它验证整体 staging 资源包络，不冒充 cuBLAS 私有实现。

源码见 [21_kernel_cutlass_style_skeleton.cuh](../../src/kernels/21_kernel_cutlass_style_skeleton.cuh)。

## 2. 重新解释 cuBLAS kernel 名称

strict FP32 cuBLAS 选择：

```text
cutlass_80_simt_sgemm_256x128_8x4_nn_align1
```

这里最重要的修正是：`8x4` 就是 **threadblock K=8、pipeline stages=4**，不能解释成
“每线程 8×4 fragment”。NVIDIA CUTLASS 的 `TileDescription.procedural_name()` 对 SM90
之前的格式明确使用 `threadblock_shape[0] x threadblock_shape[1] _
threadblock_shape[2] x stages`；shared 使用量也按每 stage 的 CTA M/N/K 数据量乘 stages
计算（[官方定义](https://github.com/NVIDIA/cutlass/blob/main/python/cutlass_library/library.py#L1411-L1433)、
[shared 计算](https://github.com/NVIDIA/cutlass/blob/main/python/cutlass_library/library.py#L1523-L1547)）。
资源算术与实测完全吻合：

```text
每 stage shared = (256×8 + 8×128) × 4 B = 12,288 B
四 stage          = 49,152 B
NCU launch shared = 50,176 B = 49,152 B + 1,024 B launch overhead
```

同样的 1,024 B 差值也出现在 K10：ptxas 报告 16,640 B，NCU 报告 17,664 B。
因此该 cuBLAS kernel 是 **BK=8 的四阶段 pipeline**。K21 V2 虽然也占 49,152 B，实际是
BK=16×2 stages；容量相同不代表时间深度和调度相同。

`align1` 表示输入只有一个 float 的对齐保证。cuBLAS 动态执行约 25.31 M 条
`LDGSTS`，K21 V2 使用 16-byte copy，仅约 6.29 M 条，正好相差约四倍；这与 align1
路径使用更细粒度 async copy 相符，但仅凭该计数不把每条指令的字节数写成既定事实。

现有证据不能证明 cuBLAS 使用“专门 producer warp / consumer warp”意义上的 warp
specialization。可以确定的是多阶段异步搬运和成熟调度，不能只凭 kernel 名称虚构
warp 角色。

## 3. 正确性与资源

两个版本在 128/256/512/1024/2048/4096、`beta=0/1/3` 下共 36/36 PASS；
Compute Sanitizer 对 128 edge path 和 256 fast path 共 4/4 PASS。

| 模式 | threads | Registers/thread | ptxas shared | NCU shared | spill |
|---|---:|---:|---:|---:|---:|
| K10 | 128 | 184 | 16,640 B | 17.664 KB | 0 |
| K21 V1 | 128 | 190 | 16,640 B | 17.664 KB | 0 |
| K21 V2 fast | 256 | 255 | 49,152 B | 50.176 KB | 0 |
| strict cuBLAS | 256 | 202 | 不可由 ptxas 取得 | 50.176 KB | 0 |

V1 说明 C++ iterator 不是免费抽象：即使数学与访存集合不变，寄存器仍增加 6。
V2 则暴露更大的缺口：shared 容量已经与 cuBLAS 相同，寄存器生命周期却增长到硬件
上限附近的 255，而 cuBLAS 只需 202。

## 4. 七轮交错性能

口径：4096³、`alpha=0.5`、`beta=3`、固定 seed、每次复位 C、10 次预热、50 次
计时，七轮轮换执行顺序。

| 模式 | 平均 GFLOPS | 范围 GFLOPS | 平均时间 | 相对 K10 | strict 比例 |
|---|---:|---:|---:|---:|---:|
| strict FP32 cuBLAS | 21,541.6 | 21,483.8–21,645.3 | 6.3801 ms | +9.87% | 100.00% |
| K10 | 19,607.1 | 19,508.2–19,660.1 | 7.0096 ms | 基线 | 91.02% |
| K21 V1 | 19,738.0 | 19,696.4–19,780.6 | 6.9631 ms | **+0.667%** | **91.63%** |
| K21 V2 | 16,425.3 | 16,278.1–16,584.8 | 8.3679 ms | -16.23% | 76.25% |

V1 的最低值仍高于本组 K10 的最高值，所以小幅提升可重复；但它还没有达到 95%。
V2 明显失败，不能因为它使用 `cp.async` 和 50 KB shared 就称为 CUTLASS 等价实现。

## 5. SASS：iterator 改变了什么

| 模式 | 静态指令 | FFMA | LDS | IMAD | IADD3 | LEA | MOV | LDGSTS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| K10 | 2,768 | 2,176 | 96 | 20 | 26 | 22 | 29 | 0 |
| V1 | 2,760 | 2,176 | 96 | 9 | 28 | 41 | 28 | 0 |
| V2 | 2,968 | 2,176 | 105 | 28 | 44 | 49 | 165 | 14 |

V1 没有减少 FFMA 或 LDS，也没有改变数学工作。IMAD 减少 11，但 LEA 增加 19；总静态
指令只减少 8。NCU 的动态总指令从 K10 的 2.336 B 降到 2.309 B，约 -1.16%，与
七轮约 +0.67% 的时间收益方向一致。因此可以把 V1 描述为很小的地址/control-code
生成收益，不能描述为 pipeline 或 issue 提升。

V2 的 14 条静态 `LDGSTS`、2 条 `LDGDEPBAR` 和 2 条 `DEPBAR` 证明异步流水确实生成；
失败不是“编译器没有使用 cp.async”。问题在于 fragment layout、寄存器生命周期和
shared transaction 组织。

## 6. Nsight Compute

| 指标 | strict | K10 | V1 | V2 |
|---|---:|---:|---:|---:|
| Registers/thread | 202 | 184 | 190 | 255 |
| Achieved occupancy | 16.67% | 16.56% | 16.49% | 16.65% |
| Issue slots busy | 87.30% | 79.19% | 78.63% | 62.00% |
| FP32 pipe active | 78.55% | 73.49% | 73.06% | 56.69% |
| No eligible | 12.70% | 20.81% | 21.37% | 38.00% |
| Eligible warps/scheduler | 1.579 | 1.358 | 1.365 | 1.082 |
| Long scoreboard | 0.024 | 0.083 | 0.107 | 0.078 |
| Short scoreboard | 0.083 | 0.029 | 0.039 | 0.264 |
| Shared-load conflicts | 0 | 0 | 0 | **201.33 M** |
| Shared-load wavefronts | 201.90 M | 335.55 M | 335.55 M | **536.87 M** |
| Local load/store sectors | 0/0 | 0/0 | 0/0 | 0/0 |

V1 没有满足“issue、FP32、no-eligible 同时改善”的 scheduler 成功标准。它的性能收益
来自略少的工作，而不是单位周期发射率更高；因此保留为独立实验，不替换 K10。

V2 的结果更有架构价值：occupancy 与 cuBLAS 一样约 16.6%，shared 容量也一样，性能
仍然低 23.8%。row-major A 便于直接 `cp.async`，却使 shared→register 产生 201 M bank
conflicts，wavefront 增至 536.9 M，short scoreboard 和 no-eligible 同时恶化。成熟
micro-kernel 必须让 global copy layout、shared layout 和 warp fragment iterator共同设计；
不能先按 global 形状写 shared，再让每个线程跨 16-float stride 标量读取 A。

## 7. 四个核心问题

### 7.1 为什么 K20 简单复制 256-thread CTA 会失败？

K20 只有一个同步 stage。8 个 warp 容易一起进入 load/barrier/compute 相位；同样约
16.6% occupancy 下，K10 的两个 4-warp CTA 反而提供两个独立调度相位。K20 没有
cuBLAS 的 BK=8×4-stage staging、layout-aware iterator 和受控寄存器生命周期，所以
线程数加倍只扩大同步域，没有增加 ready warp。

### 7.2 cuBLAS `256x128_8x4` 的关键在哪里？

关键不是“256 threads”，也不是“每线程 8×4”。官方命名和资源实测共同说明它采用
256×128 CTA、BK=8、4-stage pipeline、align1 global iterator，再配合能避免 bank conflict 的 shared
layout 和 fragment iterator。202 registers 说明它在 128 accumulator、pipeline state、
地址状态之间做了比 K21 V2 更紧凑的 lifetime 管理。

### 7.3 K10 与 CUTLASS-style SIMT 最大结构差异是什么？

K10 是扁平的 `global load -> barrier -> shared load -> FFMA -> barrier` 循环，编译器负责
在一个 BK 内调度 LDS/FFMA。CUTLASS-style kernel 把 global iterator、stage ring、shared
layout、warp iterator、fragment layout 和 MMA/FFMA schedule 当作一个共同的静态数据流。
差异不是多几个类，而是每层 layout 和时间流水拥有同一份设计约束。

### 7.4 从约 90%/91% 推到 95%，最值得投入什么？

下一步最值得投入的是一个受控的 **BK=8、4-stage、layout-aware A/B shared iterator**：

1. shared 总量仍固定 49,152 B，避免把容量变化混入实验；
2. 先设计能由 align1-compatible fine-grained async copy 写入、又能让 warp fragment 无 conflict 读取的
   permuted layout；
3. 把 fast-path registers/thread 从 255 压到接近 202，且保持无 spill；
4. 用 `wait_group` 建立真实四阶段 ring，再检查 issue/no-eligible，而不是继续叠加局部
   prefetch；
5. 不预设专用 loader warp，除非 SASS/warp-state 测量证明它有价值。

这是一项新的 micro-kernel 设计，不应继续以“给 K10 加一个技巧”的方式实现。

## 8. 证据与复现

- [correctness CSV](../results/rtx3080-k21-correctness-20261012.csv)
- [Compute Sanitizer CSV](../results/rtx3080-k21-memcheck-20261012.csv)
- [七轮交错性能 CSV](../results/rtx3080-k21-fair-20261012.csv)
- [Nsight Compute CSV](../results/rtx3080-k21-nsight-20261012.csv)
- [SASS 统计 CSV](../results/rtx3080-k21-sass-20261012.csv)
- [资源统计 CSV](../results/rtx3080-k21-resources-20261012.csv)

```bash
./scripts/build_kernel_21_variants.sh
./scripts/kernel_21_correctness_matrix.sh
./scripts/kernel_21_compute_sanitizer.sh
ROUNDS=7 ./scripts/kernel_21_fair_audit.sh
./scripts/profile_kernel_21_ncu.sh
./scripts/audit_kernel_21_sass.sh
python3 scripts/summarize_kernel_21_ncu.py --tag <profile-tag>
```
