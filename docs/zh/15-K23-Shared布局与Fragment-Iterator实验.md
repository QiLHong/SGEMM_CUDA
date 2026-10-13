# K23：Shared Layout 与 Fragment Iterator 的受控实验

日期：2026-10-13。环境：RTX 3080 10 GB、SM86、CUDA 12.4、驱动 550.127.05、Nsight Compute 2024.1。

## 结论

冻结 K10 的 128×128 CTA、BK=16、4-warp ownership 和单阶段流程后，按 B 的 16 列 fragment 分组或按 A 的 64 行 warp tile 分组，均未提高性能。两种变体保持 Shared 读取零 bank conflict，却分别增加了 Shared 写入冲突；issue 与 FP32 活跃度下降。后续 V4 将两种置换组合，性能进一步下降。K23 没有缩小与同口径 strict FP32 cuBLAS 的差距。

V1 直接调用原有 K21 V1，SASS SHA-256 与 [K21 的记录](../results/rtx3080-k21-sass-20261012.csv)一致。它是本轮同时计时的 iterator 对照，不是新实现。

## 冻结条件与物理布局

四种自写模式共用 BM=BN=128、BK=16、128 threads、4 warps、warp tile=64×64、每线程 128 accumulator、`__launch_bounds__(128,2)` 和两次 CTA 同步。V1/V2/V3 复用 K21 的 `computeTile`/`storeTile`；K10 使用原有等价数学循环。Global A/B 的读取元素、每线程读取归属与 float4 粒度，以及 C 的所有权均保持不变。所有模式都分配 A 2,112 float + B 2,048 float Shared，即 ptxas 报告 16,640 B。

| 模式 | A 逻辑 `(k,m)` 的物理 offset | B 逻辑 `(k,n)` 的物理 offset | 假设 |
|---|---|---|---|
| K10/V1 | `k*132+m` | `k*128+n` | 原有转置 A、连续 B |
| V2 | `k*132+m` | `(n/16)*256+k*16+n%16` | B 的 16 列 fragment 集中于同一 Shared slab |
| V3 | `(m/64)*1024+k*64+m%64` | `k*128+n` | A 的 64 行 warp tile 集中于同一 Shared slab |

V2 只改变 B；V3 只改变 A。它们是分组转置，不是 K16 的 K 相关 XOR。V2 的 B offset 可逆：设 `g=n/16`、`q=n%16`，则地址 `p=g*256+k*16+q`，其中 `0≤p≤2047`；逆式为 `g=p/256`、`k=(p%256)/16`、`q=p%16`、`n=16g+q`。V3 设 `w=m/64`、`r=m%64`，则 `p=w*1024+k*64+r`，其中 `0≤p≤2047`；逆式为 `w=p/1024`、`k=(p%1024)/64`、`r=p%64`、`m=64w+r`。V3 的 A 数组末尾 64 float 未使用。B 的 float4 store 在 V2 仍对齐：`n`、16、256 都是 4 的倍数。入口只接受 `M,N` 为 128 的倍数、`K` 为 16 的倍数，避免未屏蔽的边缘 tile。

## CTA → Warp → Lane → Shared → Register Fragment

`warp=threadIdx.x/32`；`warpRow=warp/2`、`warpCol=warp%2`。Warp 0、1、2、3 分别拥有 C 的 `(M,N)` 区域 `(0:63,0:63)`、`(0:63,64:127)`、`(64:127,0:63)`、`(64:127,64:127)`。

令 `r=lane/4`、`c=lane%4`。每个 K step，一个 lane 的 A fragment 是 `A[k,warpRow*64+8r+i]`，`i=0..7`；B fragment 是 `B[k,warpCol*64+16s+4c+j]`，`s=0..3`、`j=0..3`。8 个 A 与 16 个 B 值更新 128 个 accumulator，索引 `i*16+4s+j`。V1/V2/V3 把这些逻辑坐标代入上表的物理 offset，iterator 的 `advance()` 再进入下一个 K。FFMA 源码主体、warp/lane 和 accumulator ownership 都未改变。

下表覆盖 32 个 lane。`B0` 是 `s=0` 的四列；其他子块再加 `16s`，最终列还要加 `warpCol*64`。A 的行还要加 `warpRow*64`。

| r | c=0：lane，B0 | c=1：lane，B0 | c=2：lane，B0 | c=3：lane，B0 | A 的 8 行 |
|---:|---|---|---|---|---|
| 0 | 0，0–3 | 1，4–7 | 2，8–11 | 3，12–15 | 0–7 |
| 1 | 4，0–3 | 5，4–7 | 6，8–11 | 7，12–15 | 8–15 |
| 2 | 8，0–3 | 9，4–7 | 10，8–11 | 11，12–15 | 16–23 |
| 3 | 12，0–3 | 13，4–7 | 14，8–11 | 15，12–15 | 24–31 |
| 4 | 16，0–3 | 17，4–7 | 18，8–11 | 19，12–15 | 32–39 |
| 5 | 20，0–3 | 21，4–7 | 22，8–11 | 23，12–15 | 40–47 |
| 6 | 24，0–3 | 25，4–7 | 26，8–11 | 27，12–15 | 48–55 |
| 7 | 28，0–3 | 29，4–7 | 30，8–11 | 31，12–15 | 56–63 |

源码见 [K23 内核](../../src/kernels/23_kernel_shared_fragment_iterator.cuh)。相对 V1 的核心 diff：V2 的 B store 使用 `bFragmentOffset(k,n)` 且 B iterator 的 K 步长从 128 变为 16；V3 的 A store 使用 `aWarpOffset(k,m)` 且 A iterator 步长从 132 变为 64。Global source 地址不变。

## 正确性与资源

V2/V3 在 128、256、512、1024、2048、4096 与 `beta=0/1/3` 下共 36/36 PASS；原始 absolute error 阈值仍为 0.01。Compute Sanitizer 对两个版本的 128、256 尺寸共 4/4 PASS。V1 复用先前已完成六尺寸和三 beta 验证的 K21 V1，且本次 4096³ 计时入口再次通过参考校验。

| 模式 | Registers/thread | ptxas Shared | Achieved occupancy | spill/local sectors |
|---|---:|---:|---:|---:|
| K10 | 184 | 16,640 B | 16.57% | 0 |
| V1 | 190 | 16,640 B | 16.53% | 0 |
| V2 | 190 | 16,640 B | 16.23% | 0 |
| V3 | 190 | 16,640 B | 16.40% | 0 |

相同 register 数量不代表 fragment 精确 live range 相同；本次只报告总分配、spill 与静态 LDS→FFMA 距离。

## 三轮交错性能

同一 GPU、Release、seed 20261013、4096³、`alpha=0.5`、`beta=3`、每次复位 C、10 次预热、50 次计时。三轮轮换顺序。strict cuBLAS 使用 `CUBLAS_COMPUTE_32F_PEDANTIC`，未设置 `NVIDIA_TF32_OVERRIDE`。

| 模式 | 平均 GFLOPS | 范围 GFLOPS | 平均 kernel time | 相对 K10 | strict 比例 |
|---|---:|---:|---:|---:|---:|
| strict cuBLAS | 21,620.2 | 21,463.6–21,804.4 | 6.357 ms | +10.21% | 100.00% |
| K10 | 19,617.0 | 19,569.1–19,652.5 | 7.006 ms | 基线 | 90.74% |
| V1 = K21 V1 | 19,736.4 | 19,706.6–19,776.3 | 6.964 ms | +0.61% | 91.29% |
| V2，B fragment 分组 | 18,840.2 | 18,807.3–18,875.1 | 7.295 ms | -3.96% | 87.14% |
| V3，A warp 分组 | 19,205.0 | 19,176.1–19,262.3 | 7.156 ms | -2.10% | 88.83% |

V1 的小收益与 K21 七轮实验方向一致；V2/V3 的最大值仍低于 K10 最小值。没有以单次峰值宣布成功。

## Nsight 与 SASS

| 指标 | strict | K10 | V1 | V2 | V3 |
|---|---:|---:|---:|---:|---:|
| Issue slots busy | 87.34% | 79.32% | 78.47% | 73.88% | 75.74% |
| FP32 pipe active | 78.59% | 73.57% | 73.08% | 68.88% | 71.00% |
| No eligible | 12.66% | 20.68% | 21.53% | 26.12% | 24.26% |
| Eligible warps/scheduler | 1.580 | 1.361 | 1.358 | 1.291 | 1.282 |
| Long scoreboard | 0.023 | 0.082 | 0.108 | 0.174 | 0.171 |
| Shared-load conflicts | 0 | 0 | 0 | 0 | 0 |
| Shared-load wavefronts | 201.90 M | 335.55 M | 335.55 M | 335.55 M | 335.55 M |
| Shared-store conflicts | 0 | 16.78 M | 16.78 M | **33.55 M** | **50.33 M** |
| Shared-store wavefronts | 0.52 M | 50.33 M | 50.33 M | **67.11 M** | **83.89 M** |
| L2 hit rate | 87.83% | 78.07% | 71.51% | 55.96% | 58.76% |
| DRAM read，NCU 自适应单位 | 515.9 MB | 912.5 MB | 1.153 GB | 1.829 GB | 1.722 GB |

三种 iterator 变体都执行 96 条静态 `LDS.128`，读取零 conflict，读取 wavefront 基本相同。V2/V3 未节约 Shared→Register transaction；写入 conflict 和 wavefront 增加，同时 issue、eligible warp、FP32 active 下降。Global source 地址与静态 `LDG` 数相同，L2/DRAM 行为却恶化；可能与整体排程和缓存时序有关，单凭本次指标不能断定唯一因果。Nsight 多次 replay 的 profile duration 不等同正常计时。

| 静态 SASS | K10 | V1 | V2 | V3 |
|---|---:|---:|---:|---:|
| FFMA | 2,176 | 2,176 | 2,176 | 2,176 |
| LDS | 96 | 96 | 96 | 96 |
| LDG | 40 | 40 | 40 | 40 |
| IMAD | 20 | 9 | 9 | 13 |
| IADD3 | 26 | 28 | 28 | 31 |
| LEA | 22 | 41 | 43 | 31 |
| MOV | 29 | 28 | 28 | 29 |
| 总静态指令 | 2,768 | 2,760 | 2,768 | 2,784 |

沿 `LDS.128` 到首次使用其 GPR 的 FFMA 统计静态距离：K10 中位 66 条，V1/V2/V3 均为 60 条。该指标是 SASS 指令数，不是延迟周期；物理 GPR 的完整 live range 尚未独立重建。FFMA/LDS 的静态数量一致，动态 FMA 计数略有编译差异，不应据此声称调度完全相同。

## 研究判断

K10 的 Shared 读取已零冲突，且每 K step 用 8 个 A 与 16 个 B fragment 值完成 128 次 FFMA，复用充分。cuBLAS 的 Shared-load wavefront 更少、issue 更高，但 CTA、BK 和 pipeline 也都不同；K23 只能排除本次两个具体地址重排，无法单独归因 cuBLAS 的全部优势。

Fragment Iterator 的作用是显式描述 warp/lane 对 Shared 坐标的访问、前进规则和 register fragment 形状，使写入布局与读取布局可以一同审计。K21 V1 已测到少量地址组织收益；本次 V2/V3 没有改善 LDS transaction 或吞吐。因此不能把 Iterator 抽象自身当作 CUTLASS 性能的证明。

在 K10 的外形下停止继续 Shared 地址置换。若继续追求 95%，下一实验应把 256-thread CTA、BK=8、四阶段，以及 Global copy、Shared 写入/读取和 Fragment Iterator 作为一个共同设计的数据流；当前结果还不能证明需要专门的 loader warp。

## 补充实验：V4 A+B coordinated layout

用户后续提供的 K23 说明要求第三个实验同时协调 A+B。原 V3 已作为 A-only 历史结果提交，不能用新实现覆盖旧 CSV；因此将相同问题实现为 V4：

```text
A: (k,m) -> (m/64)*1024 + k*64 + m%64
B: (k,n) -> (n/16)*256 + k*16 + n%16
```

V4 组合原 V2 的 B layout 和 V3 的 A layout；Global loader 的源元素与 float4 粒度、lane/output ownership、128 accumulator、FFMA 循环、BM/BN/BK、同步方式全部保持不变。重新编译 V1/V2/V3 后，SASS SHA-256 与原记录逐项相同，说明补充实现没有改写先前三个版本。

### 正确性和资源

V4 在六尺寸 × beta=0/1/3 下 **18/18 PASS**；阈值仍为 absolute error=0.01。Compute Sanitizer 对 128、256 共 **2/2 PASS**。ptxas/NCU：190 registers/thread、16,640/17,664 B Shared、local load/store=0、无 spill。

### 七轮交错性能

本轮只重新交错 strict、K10、V1、V4；4096³、alpha=0.5、beta=3、seed=20261013、每次复位 C、10 次预热、50 次计时。V2/V3 的旧数据保留，不把跨轮日期的数据混成同一统计样本。

| 模式 | 平均 GFLOPS | 范围 | 样本标准差 | 平均 ms | 相对 K10 | strict 比例 |
|---|---:|---:|---:|---:|---:|---:|
| strict | 21,506.8 | 21,418.7–21,805.4 | 136.3 | 6.3907 | +9.87% | 100.00% |
| K10 | 19,574.9 | 19,461.8–19,650.4 | 67.8 | 7.0213 | 基线 | 91.02% |
| V1 | 19,697.4 | 19,646.7–19,808.5 | 55.9 | 6.9761 | +0.63% | 91.59% |
| V4 A+B | 18,553.7 | 18,452.9–18,678.3 | 95.5 | 7.4079 | **-5.22%** | 86.27% |

V4 最大值仍低于 K10 最小值，失败幅度明显大于七轮波动。

### 指标解释

| 指标 | K10，原同口径采样 | V2 B-only | V3 A-only | V4 A+B |
|---|---:|---:|---:|---:|
| Issue slots busy | 79.32% | 73.88% | 75.74% | **72.50%** |
| FP32 pipe active | 73.57% | 68.88% | 71.00% | **67.82%** |
| No issue (`issue_inst0`) | 20.68% | 26.12% | 24.26% | **27.50%** |
| Eligible warps/scheduler | 1.361 | 1.291 | 1.282 | **1.254** |
| Long scoreboard | 0.082 | 0.174 | 0.171 | **0.201** |
| Shared-load conflict | 0 | 0 | 0 | 0 |
| Shared-load wavefront | 335.55 M | 335.55 M | 335.55 M | **335.55 M** |
| Shared-store conflict | 16.78 M | 33.55 M | 50.33 M | **67.11 M** |
| Shared-store wavefront | 50.33 M | 67.11 M | 83.89 M | **100.66 M** |

原 K10/V2/V3 与 V4 分属独立 NCU 进程，但使用相同脚本、环境、输入和第一匹配 launch；表用于结构归因，不用 NCU duration 计算 GFLOPS。`No issue` 是 counter 名称的简写，不把所有无发射周期都等同于“没有 eligible warp”。

V4 的 96 条静态 LDS、2,176 条 FFMA 和 335.55 M Shared-load wavefront 与其他 K23 版本基本相同。它没有改变 fragment 消费模式，只把两套物理地址置换叠加，因此没有得到公开 CUTLASS 已测到的每条 LDS 较低 wavefront；相反，写入 conflict/wavefront 接近 V2 与 V3 的增量叠加，issue、eligible warp 和 FP32 active 同时恶化。

这回答了补充问题：**单纯把 A/B Shared 置换“协调”在一起，不等于 CUTLASS 的 Shared Layout + Fragment Iterator 协同设计。** 真正有研究价值的变量是 lane fragment ownership 和每条 LDS 的访问宽度/广播结构，而不是继续组合地址公式。该方向在冻结 K10 ownership 的约束下结束。

补充数据：

- [七轮原始性能](../results/rtx3080-k23-v4-fair-20261013.csv)
- [性能摘要](../results/rtx3080-k23-v4-summary-20261013.csv)
- [正确性](../results/rtx3080-k23-v4-correctness-20261013.csv)
- [Compute Sanitizer](../results/rtx3080-k23-v4-memcheck-20261013.csv)
- [Nsight](../results/rtx3080-k23-v4-nsight-20261013.csv)
- [SASS](../results/rtx3080-k23-v4-sass-20261013.csv)
- [LDS→FFMA 静态距离](../results/rtx3080-k23-v4-lds-distance-20261013.csv)

## 数据与复现

- [三轮性能](../results/rtx3080-k23-fair-20261013.csv)
- [正确性](../results/rtx3080-k23-correctness-20261013.csv)
- [Compute Sanitizer](../results/rtx3080-k23-memcheck-20261013.csv)
- [Nsight Compute](../results/rtx3080-k23-nsight-20261013.csv)
- [SASS](../results/rtx3080-k23-sass-20261013.csv)
- [LDS→FFMA 静态距离](../results/rtx3080-k23-lds-distance-20261013.csv)
- [资源](../results/rtx3080-k23-resources-20261013.csv)

```bash
./scripts/kernel_23_audit.sh build
./scripts/kernel_23_audit.sh correctness
./scripts/kernel_23_audit.sh memcheck
ROUNDS=3 REPEAT=50 WARMUP=10 ./scripts/kernel_23_audit.sh fair
./scripts/profile_kernel_23_ncu.sh
./scripts/audit_kernel_23_sass.sh
python3 scripts/analyze_kernel_23_lds_distance.py
python3 scripts/summarize_kernel_23_ncu.py --tag <profile-tag>
```

原始日志和 `.ncu-rep` 保留在被忽略的 `benchmark_results/`；版本化 CSV 在 `docs/results/`。
