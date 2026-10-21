# 33. BN64/BK32 稳健性与尺寸适用范围：RTX 4090 D 阶段收尾

> 当前发布状态：本章的稳健性证据已随 `a2b23da` 上传；后续官方 CUTLASS 对照随 [9225bae](https://github.com/QiLHong/SGEMM_CUDA/commit/9225bae1d8b6ea1038f8be257067d841528e99b9) 上传，见[第 34 章](34-CUTLASS与BN64的4090D公平对照.md)。下面的实验日志、旧发布补记与未执行事项保留时间语境。

> 发布补记（2026-10-22）：用户在本轮实验交付后授权全部上传；本章、host benchmark、复跑与审计脚本及完整原始归档已纳入本次 RTX 4090 D 发布。正文和冻结归档中的“未提交/未推送”、未跟踪文件数量记录的是实验交付时的状态。归档里的源码快照、哈希清单和实验数据保持原字节，本次发布没有重新测量。

本轮重新运行四个方阵、三个随机输入、三个独立进程批次。**2048 的优势延续：BN64 对作者版的九个组合配对提升中位数为 +19.485%～+19.612%，对同输入 TF32 cuBLAS 为 +11.370%～+11.739%，两项均 63/63 轮领先。3072 对作者版也稳定领先，对 cuBLAS 优势较小；4096 仍未稳定优于作者版。**

这验证了已测尺寸上的适用差异，没有找到一个连续的“最佳尺寸区间”，也没有证明所有矩形都应选 BN64。作者基线和候选都保留，未新增自动分派、tile/grid 搜索或 Tensor Core 算法。

阅读衔接：[第 31 章原实验](31-TensorCore-BN64优化与尺寸效应.md) → [第 32 章代码教程](32-RTX4090D-TensorCore优化学习指南.md) → 本章。全部新数据位于[独立归档](../../results/windows/20261021-tensorcore-bn64-robustness/README.md)。历史 57.998 TFLOPS、+19.56%、+11.57% 已从旧原始 CSV 重新核实，单列在 `historical-statistics.json`，没有填入本轮表。

## 33.1 改动范围与实际环境

起点仍是 `a54435d9abfdb673b2ef8bca147350a7820046cf`，分支 `results/rtx4090d-k30-stage2-20261021`；vitamin-cuda 为 `520f97093a2abc239977be83d244b327d0e2e2ef`，工作区干净。开始时的 Git 状态、暂存/未暂存差异、实际字节 SHA256 保存在 `preflight/`。原有未提交源码与历史结果保留；本轮没有提交、推送或合并。

新增的 [host benchmark](../../experiments/tensorcore-bn64-robustness/benchmark.cu) 直接包含既有作者头文件及 `tf32_bn64.cuh`，改的是 seed、计划、校验和记录入口。三实现仍叫 `vitamin_tf32_dbf`、`tf32_bn64_bk32`、`cublas_tf32`。来源作者 QiLHong、上游路径 `kernels/sgemm/sgemm_tf32.cu`、原函数名与[许可证](../../experiments/tensorcore/vendor/LICENSE)均保留。

| 本次实际检查 | 值 |
|---|---|
| GPU / 驱动 | RTX 4090 D，24564 MiB，114 SM；610.60，WDDM |
| Windows CUDA Toolkit | `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4`；nvcc 12.4.131 |
| MSVC / SDK | x64 19.37.32826.1，工具集目录 14.37.32822；SDK 10.0.26100.0 |
| 构建 | VS 内置 CMake 3.31.6、Ninja 1.12.1；Release、`-O3`、`sm_89` |
| 实际 cuBLAS / NCU | 120405 / 2024.1.1 |

构建日志含 `--generate-code=arch=compute_89,code=[compute_89,sm_89]`，无 `--use_fast_math` 或忽略编译器版本检查参数。没有升级工具链或改变系统配置。

完整 exe 哈希由旧 `faeb5091…` 变为新 `9440ffa1…`，完整 SASS 文本也不同；这与 host 入口和文件信息改变有关。按函数解析后，**两个被测 device kernel 的地址/规范化指令以及全部 128-bit 编码字均与旧 build02 相同**。见 `evidence/sass-comparison.csv`；这比仅比较 exe 哈希更能回答算法是否改变。35 个已有 experiment 文件的实际字节也核验未变。

## 33.2 运行前固定的方案

[protocol.md](../../results/windows/20261021-tensorcore-bn64-robustness/protocol.md) 与 [plan.csv](../../results/windows/20261021-tensorcore-bn64-robustness/run/plan.csv) 在本轮 GPU 检查前保存；计划 SHA256 为 `6ef11585…bc7c0a4c`，执行时再次校验。

- 尺寸 1024、2048、3072、4096；seed 20261020、42、12345。
- 三个独立 benchmark 进程，PID 分别为 39360、41760、39264；各自建立上下文，每批 12 案例。三个批次连续进行，不能称为跨天或跨重启验证。
- 每案例每实现七轮，每轮先预热 10 次，再计时 30 次。总计 `4×3×3×3×7×30 = 22680` 个正式 event。
- 每个案例前六轮覆盖三个实现的六种排列；第七排列随批次轮换。尺寸与 seed 顺序也跨批次轮换。cuBLAS 没有固定在末尾。
- 全部 event 保留。每个 event 只包围一次 GEMM，启动和 event 同步返回值成功后才记录；没有从慢样本中选择“最好七轮”。

三个进程实际起止时间及命令见 `run/execution.json`、各 `batch-*/process.csv`；独立审计确认窗口不重叠。正式表不包含 9 个小矩形辅助 event、数值参考、转换或六次 profiler 采集。

### 输入与 cuBLAS 的精度契约

保留旧 xorshift 生成器和有界分布：符号由随机位决定，数值为 `±(i+0.01f*j)`，`i,j=0..4`，seed 只改变输入。GPU 使用既有 `__float_to_tf32` 提前生成实际 TF32 表示，三实现复用同一份 `ta/tb` 指针和字节；FP32 累加、FP32 输出，`C=AB, alpha=1, beta=0`。

每个案例在测量前后回读原始 A/B 和实际 TF32 A/B，记录 FNV1a64 并与预期字节 `memcmp`。这是非加密指纹；输入相同的依据还包括共享指针与字节检查，不能只看 seed。跨三批指纹一致，不同 seed 指纹不同；20261020 的三个旧尺寸共 12 个指纹与旧归档匹配。

cuBLAS 沿用 `CUBLAS_COMPUTE_32F_FAST_TF32`、`CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION`、`CUBLAS_ATOMICS_NOT_ALLOWED`、`CUBLAS_GEMM_DEFAULT`。运行时读取实际 math/atomics/stream/version 并核对，详见每例 `.meta.txt`。这里的 TF32 主表允许 TF32 运算；PEDANTIC FP32 仅作数值参考，不混入计时表。

输入转换不在 GEMM event 内，保留的准备函数还会生成未参与本轮测量的 FP16 副产物；这不是端到端耗时。每次预热和正式调用前都在同一 default stream 执行全零 C0 的 D2D 恢复，复制在 start event 前。三实现都复制每输出 4 字节；该操作及组间校验会影响缓存，因此不称为完全冷缓存测量。

## 33.3 为什么 3072 合法

共同 host 入口要求 `M=128..4096, M%128=0`，`N=1024..4096, N%1024=0`，`K=32..4096, K%32=0`。六种不合法形状已实际拒绝，未进入 kernel。`ceil(grid)` 不是边界保护。

两版沿用宽 8 网格公式，令 `i=blockIdx.y*gridDim.x+blockIdx.x`：

```text
bx = i % 8 + (i / (8 * gridDim.y)) * 8
by = (i / 8) % gridDim.y
作者：grid=(N/128,M/128)；BN64：grid=(N/64,M/128)
```

3072 时作者 grid 为 24×24、576 CTA，BN64 为 48×24、1152 CTA；两者 x 维均为 8 的整数倍。末 CTA 分别映射到 `(23,23)` 与 `(47,23)`。BN64 线程输出公式仍是：

```text
row = by*128 + (warp/4)*64 + mi*16 + lane/4 + (r/2)*8
col = bx*64  + (warp%4)*16 + ni*8  + (lane%4)*2 + r%2
warp=tid/32; lane=tid%32; mi=0..3; ni=0..1; r=0..3
```

代入末 CTA、tid255、mi3、ni1、r3，得到 `(3071,3071)`，刚好落在矩阵右下角。CPU 枚举检查了四尺寸两核共 5760 个 CTA，以及 24576 个 CTA 内输出 fragment；没有重复、遗漏、越界。另验证完整 K tile 和 16B 地址分量对齐。原 BN64 映射脚本也重新检查了 8192 输出、12288 个 shared float 地址。证据在 `mapping/` 与 `navigation-audit/grid-*`，它们不能代替 GPU 数据竞争检查。

数据流与同步沿用原实现：`global --cp.async--> shared双buffer → registers → mma.sync FP32累加 → global C`。wait 保证本线程异步复制完成，CTA barrier 保证跨线程消费者看到数据并协调 buffer 复用。没有加入 TMA，没有改 MMA 的 K 求和次序。

## 33.4 正确性结果与精度失败分别保留

先运行 `M128/N1024/K32、64、96`，CPU double 每例全量检查 131072 个输出。正式大方阵每实现预检一次、每轮结束检查一次完整输出；CPU double 仅固定抽样 256 个唯一位置（含四角，位置不随输入 seed 改变）。

本轮正式数据有 **4320 条参考检查、576 次完整逐位比较，NaN/Inf 与逐位差异均为 0**。其中 BN64 对作者版逐位比较为 288 次，另 288 次是重复运行作者版对保存作者输出的比较。全部同精度 cuBLAS 比较的 max abs/max rel 为 0；这项记录是数值比较，不额外宣称 cuBLAS 做过逐位验证。

| 参考 | 回答的问题 | 本轮结果 |
|---|---|---|
| 保存的 vitamin TF32 输出 | 候选是否改变输出 | 全量数值一致；BN64 全量逐位一致 |
| 同输入 TF32 cuBLAS | 是否与指定库配置一致 | 全量 max abs/max rel 为 0 |
| 实际 TF32 输入的 PEDANTIC FP32 | 同一输入表示下的运算差异 | TF32 allclose 通过；更严标准仍有失败 |
| 原始未量化输入的 PEDANTIC FP32 | 量化路线是否满足原始精度 | 全部 864 条检查的 abs0.01、SIMT、TF32 native 标准均失败 |
| 实际 TF32 输入的 CPU double | 独立数值参考 | 小尺寸全量、大尺寸256点；TF32 allclose 通过，部分严标准失败 |

沿用实际表达式 `abs(got-ref) <= eps + eps*abs(got)`，SIMT `eps=.001`、TF32 `eps=.1`；相对项使用 **got**。最大相对误差另用 `abs(got-ref)/max(abs(ref),1e-12)`，不是同一个分母。所有阈值、失败元素数均未放宽或删除。

| N | 对量化输入 PEDANTIC 的最大绝对误差（跨输入最坏） | 对量化输入 CPU double 的最大绝对误差（256点） |
|---:|---:|---:|
| 1024 | 0.003845215 | 0.002028 |
| 2048 | 0.012939453 | 0.005869 |
| 3072 | 0.021362305 | 0.010065 |
| 4096 | 0.030761719 | 0.013802 |

原始 FP32 对照最大绝对误差 0.773101807，最大相对误差约 8238.20；后者受近零参考值影响，不能单独用于衡量所有输出。量化输入 PEDANTIC 最大相对误差 304.25，CPU 抽样约 0.005271；详细逐项最大误差/失败元素数见 [errors.csv](../../results/windows/20261021-tensorcore-bn64-robustness/analysis/errors.csv) 和未汇总的 `.checks.csv`。TF32 allclose 通过不能写成严格 FP32 通过。

Compute Sanitizer 2024.1.1 对单个 128×1024×32 案例尝试 memcheck：20 秒附加超时后仍等待目标，到总时限 60 秒终止该项，状态 **TIMEOUT_UNVERIFIED**。工具日志为 `No attachable process found`；终止命令另返回凭据错误，事后进程查询未发现残留，不能把终止命令当成功证据。没有重试绕过权限，没有运行 racecheck/synccheck，不能宣称任何 Sanitizer 模式通过。日志在 `sanitizer01/`；当前复跑脚本已补充终止返回码的记录，实际运行的旧脚本快照保留。

## 33.5 本轮性能：先看每个组合，再归纳尺寸

每个组合先取 30 events 的轮平均延迟，再取七个轮均值的中位数 `t_med`，吞吐为 `2N³/(t_med×10⁹)` TFLOPS（延迟单位 ms）。配对提升另计算七个 `t_reference,r/t_BN64,r - 1` 的中位数；它与两项中位吞吐之比不相同。两种值都保存在 [cases.csv](../../results/windows/20261021-tensorcore-bn64-robustness/analysis/cases.csv)。

以下是本轮全部 36 个尺寸 × seed × 批次组合，不用一个整体均值掩盖回退。它们嵌套于三个进程，不是36个完全独立实验。吞吐列为 TFLOPS，百分比为逐轮配对提升中位数。

| N | seed | 批次 | 作者 | BN64 | cuBLAS | 对作者 | 对cuBLAS |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 42 | 1 | 31.362 | 32.683 | 35.266 | +4.151% | -8.257% |
| 1024 | 42 | 2 | 33.714 | 35.223 | 38.180 | +4.362% | -7.986% |
| 1024 | 42 | 3 | 31.473 | 32.682 | 35.705 | +3.573% | -8.426% |
| 1024 | 12345 | 1 | 31.426 | 32.682 | 35.686 | +3.945% | -8.273% |
| 1024 | 12345 | 2 | 33.718 | 35.089 | 38.151 | +3.965% | -8.072% |
| 1024 | 12345 | 3 | 33.717 | 35.109 | 38.240 | +4.137% | -8.188% |
| 1024 | 20261020 | 1 | 31.394 | 32.666 | 34.721 | +4.098% | -5.946% |
| 1024 | 20261020 | 2 | 31.379 | 32.716 | 35.703 | +4.107% | -8.405% |
| 1024 | 20261020 | 3 | 33.694 | 35.183 | 38.156 | +4.546% | -7.789% |
| 2048 | 42 | 1 | 48.489 | 57.979 | 51.925 | +19.582% | +11.608% |
| 2048 | 42 | 2 | 48.493 | 57.965 | 52.028 | +19.577% | +11.522% |
| 2048 | 42 | 3 | 48.475 | 57.945 | 51.889 | +19.513% | +11.672% |
| 2048 | 12345 | 1 | 48.465 | 57.945 | 51.979 | +19.561% | +11.375% |
| 2048 | 12345 | 2 | 48.461 | 57.985 | 51.985 | +19.590% | +11.541% |
| 2048 | 12345 | 3 | 48.489 | 57.952 | 51.883 | +19.485% | +11.739% |
| 2048 | 20261020 | 1 | 48.475 | 57.952 | 52.033 | +19.583% | +11.370% |
| 2048 | 20261020 | 2 | 48.475 | 57.952 | 51.910 | +19.553% | +11.600% |
| 2048 | 20261020 | 3 | 48.471 | 57.959 | 51.990 | +19.612% | +11.464% |
| 3072 | 42 | 1 | 54.482 | 59.136 | 58.160 | +8.564% | +1.920% |
| 3072 | 42 | 2 | 54.297 | 58.905 | 58.294 | +7.897% | +0.458% |
| 3072 | 42 | 3 | 54.389 | 58.835 | 58.381 | +7.881% | +0.724% |
| 3072 | 12345 | 1 | 54.905 | 59.066 | 57.990 | +7.598% | +2.139% |
| 3072 | 12345 | 2 | 54.275 | 59.443 | 58.433 | +9.298% | +1.557% |
| 3072 | 12345 | 3 | 54.016 | 59.112 | 58.713 | +9.330% | +1.100% |
| 3072 | 20261020 | 1 | 54.288 | 58.829 | 57.587 | +7.834% | +1.538% |
| 3072 | 20261020 | 2 | 54.445 | 58.882 | 58.479 | +8.145% | +0.662% |
| 3072 | 20261020 | 3 | 55.129 | 59.463 | 58.156 | +7.455% | +2.247% |
| 4096 | 42 | 1 | 69.222 | 69.223 | 66.639 | -0.050% | +3.745% |
| 4096 | 42 | 2 | 69.550 | 69.455 | 66.871 | -0.022% | +4.165% |
| 4096 | 42 | 3 | 69.315 | 69.118 | 66.759 | -1.071% | +3.533% |
| 4096 | 12345 | 1 | 69.315 | 68.699 | 66.982 | -1.007% | +3.000% |
| 4096 | 12345 | 2 | 69.746 | 68.550 | 67.137 | -1.716% | +2.207% |
| 4096 | 12345 | 3 | 69.524 | 69.381 | 67.040 | -0.777% | +3.492% |
| 4096 | 20261020 | 1 | 69.194 | 69.061 | 66.991 | -0.716% | +3.131% |
| 4096 | 20261020 | 2 | 69.780 | 69.256 | 66.802 | -1.130% | +3.756% |
| 4096 | 20261020 | 3 | 69.176 | 68.964 | 67.428 | +0.214% | +2.706% |

九组合的描述性汇总如下；“正收益轮”只是观察计数，不把嵌套的轮次/event 当独立随机实验样本，没有计算置信区间。

| N | 对作者：九组合配对中位数范围 | 正组合 / 正轮 | 对 cuBLAS：九组合范围 | 正组合 / 正轮 |
|---:|---:|---:|---:|---:|
| 1024 | +3.573%～+4.546% | 9/9；58/63 | −8.426%～−5.946% | 0/9；1/63 |
| 2048 | +19.485%～+19.612% | 9/9；63/63 | +11.370%～+11.739% | 9/9；63/63 |
| 3072 | +7.455%～+9.330% | 9/9；63/63 | +0.458%～+2.247% | 9/9；51/63 |
| 4096 | −1.716%～+0.214% | 1/9；19/63 | +2.207%～+4.165% | 9/9；59/63 |

2048 的 BN64 吞吐为 57.945～57.985 TFLOPS。3072 为 58.829～59.463 TFLOPS，对作者版的提升在全部63轮中为正，但对 cuBLAS 的 12 个负收益轮不可略去。4096 有 8/9 组合的对作者配对中位数为负，继续保留作者版；1024 继续保留本轮更快的 cuBLAS。此处没有替换通用默认实现。

每组七轮延迟范围、标准差与 CV 在 [summary.csv](../../results/windows/20261021-tensorcore-bn64-robustness/analysis/summary.csv)，每轮均值、event min/max 在 [rounds.csv](../../results/windows/20261021-tensorcore-bn64-robustness/analysis/rounds.csv)，全部配对正负值在 `paired-rounds.csv`。正式 event 范围约 0.055296～3.267584 ms，慢样本完整保留。

九组合中各自七轮均值延迟CV的范围如下（不是22680个event的总体CV）：

| N | 作者CV | BN64 CV | cuBLAS CV |
|---:|---:|---:|---:|
| 1024 | 0.097%～11.957% | 0.124%～5.867% | 0.802%～6.530% |
| 2048 | 0.023%～3.529% | 0.044%～3.475% | 0.134%～3.443% |
| 3072 | 0.572%～1.182% | 0.743%～1.870% | 0.548%～1.476% |
| 4096 | 0.545%～3.190% | 0.309%～3.616% | 0.376%～3.386% |

3072 的 BN64 CV 并不低于作者版；稳定领先表示收益方向一致，不表示延迟方差更小。

三批遥测分别 205/217/215 点，共 637 点；温度分别 43～54、46～56、48～59°C，SM 频率均观测到 2520～2715 MHz，显存前两批为10251～10501 MHz，第三批仅观测到10251 MHz。功耗分别 62.57～162.19、67.32～168.41、67.08～169.09 W。已记录字段没有 N/A，但采样间隔约 200 ms，包含 CPU 检查间隙，不能证明每个 event 恒频。未锁频、未改功率上限，WDDM/桌面负载仍是限制。

## 33.6 NCU 支持哪些解释

正常测量结束后，固定 seed20261020 对 2048、3072、4096 的作者版/BN64 共采六个 launch。NCU kernel replay，`clock-control none`、`cache-control none`；使用安装目录 sections，原部署警告保留，没有改 HOME。六份 `.ncu-rep`/CSV 成功产出，profile duration 没有进入上一节的 TFLOPS。

### 资源与任务分布

两版 registers/thread 仍为 127→102，spill/stack 均 0，NCU local load/store sectors 均 0。用户 shared 为 32768→49152 B，加 driver 后为 33792→50176 B；两版理论驻留均是 **2 CTA/SM**，没有声称 BN64 把作者版的理论驻留翻倍。BN64 48 KiB 恰好在该数值，不写成“超过 48 KiB”。

表内箭头均为作者→BN64：

| N | CTA / 名义 waves | 平均 SM active/elapsed | Tensor active | Tensor elapsed | achieved occupancy |
|---:|---|---|---|---|---|
| 2048 | 256→512 / 1.12→2.25 | 75.23→89.78% | 44.26→44.52% | 33.30→39.97% | 27.31→29.88% |
| 3072 | 576→1152 / 2.53→5.05 | 84.46→91.90% | 45.35→45.28% | 38.30→41.61% | 30.47→31.78% |
| 4096 | 1024→2048 / 4.49→8.98 | 95.41→99.29% | 45.87→45.82% | 43.77→45.49% | 31.71→32.51% |

本次 NCU 的 SM throughput active/elapsed 数值恰与表内 Tensor 指标相同，原指标名与单位保留在 `evidence/ncu-metrics.csv`。active 是活跃周期分母，elapsed 是经过周期分母，不能拿 active 百分比充当整段运行的全卡利用率。

2048、3072 的 active Tensor 基本相同，而 elapsed Tensor 提升，支持“较小任务块改善活跃时间与尾部任务分布”的解释。**4096 本次单个 profile 也显示 elapsed 改善，但公平测量仍未稳定赢作者版；旧 profile 两版平均 SM 活跃时间都约99%。** 这提醒我们单次、多 replay、未锁频的计数器存在状态差异，不能由这一表精确分解加速百分比，更不能将当前4096一条采样包装成稳定的负载分配收益。

| N | eligible warps | issue active | barrier | short scoreboard | long scoreboard |
|---:|---|---|---|---|---|
| 2048 | .17→.19 | 9.78→10.37% | .37→.37 | .24→.15 | .03→.03 |
| 3072 | .18→.20 | 9.94→10.43% | .45→.42 | .24→.16 | .03→.02 |
| 4096 | .19→.20 | 10.03→10.50% | .48→.44 | .25→.16 | .03→.02 |

eligible/stall 的单位是每调度器每活跃周期平均 warp 数，不能直接相加得到延迟占比。尤其 2048 barrier 指标没有下降，不能把约19.5%收益归因于“少等 barrier”。每 CTA 的 K tile/barrier 次数减半，但 CTA 数翻倍，整个 GEMM 的 CTA barrier 调用总数并未减半。

### 缓存与额外访存代价

BN 减半后，相邻两个 CTA 各自读取相同 A tile；按请求 payload 推导，A 总读取翻倍、B 不变，A+B 请求约为原版 1.5 倍。此为请求量，不是 DRAM 字节保证。

| N | L2 read sectors（作者→BN64） | DRAM读 MiB | DRAM读+写 MiB | shared-load wavefront |
|---:|---|---|---|---|
| 2048 | 16,783,425→25,171,558 | 32.014→32.023 | 32.108→32.023 | 12,589,937→20,989,186 |
| 3072 | 56,638,370→84,946,201 | 85.266→73.443 | 104.988→94.439 | 42,495,744→70,851,926 |
| 4096 | 134,237,686→203,012,254 | 346.292→492.465 | 406.336→596.207 | 100,770,156→167,904,623 |

L2 请求增加约50%，与推导一致；shared-load wavefront也增加。2048 DRAM流量近似相同，3072本次较低，4096读+写增加约46.7%，支持“大尺寸额外搬运抵消任务粒度收益”的解释。缓存命中率在三尺寸均提高（4096 94.74→95.50%），但4096总DRAM字节仍增加，不能以命中率代替流量。

本机 L2 为72 MiB，三个尺寸的 A+B 分别为32、72、128 MiB；3072恰等于容量并不保证全缓存命中，因为 C/C0 与其他访问也竞争缓存。单次 kernel 期间的 DRAM 写计数可以为0（本次2048 BN64如此），它不表示没有写 C，也不包含之后所有脏缓存写回的端到端代价。

shared-load bank-conflict计数原版→BN64分别为6752→15553、27223→67053、104237→121692，相对wavefront均很少；store conflict为0。异步拷贝不属于这里普通LSU shared-store指标，不能由其为0断言shared没有写入。计数器不支持“修复bank conflict是主要收益来源”。

### SASS 没有出现新的优化

本轮两目标的完整编码指令与旧构建相同。原版 K 主循环 111 条指令、4 LDGSTS、32 HMMA；BN64 为116条、6 LDGSTS、32 HMMA，最后copy与完成等待之间分别有28/27条HMMA，片段见 `evidence/*-main-loop.sass.txt`。仍有异步copy与MMA交错，但不能由交错推出访存完全隐藏。每个主循环计算的CTA区域与BK不同，不能只比较循环指令数推断速度。

## 33.7 复算、导航与结论边界

汇总与验证脚本独立读取原始 CSV；验证器用 `math.fsum` 和独立中位数实现，未调用汇总函数。已核对 22680 events、756轮、36案例、执行顺序、所有输入指纹、4320检查、576逐位记录、三个PID和非重叠时段、主要统计及原始文件哈希。结果为 PASS，没有修改原始数值以匹配报告。

本轮支持：2048在三个输入、三批同日进程内稳健领先；3072扩大了已验证的获益尺寸集合；1024应保留cuBLAS，4096应保留作者基线。不同seed只覆盖同一有界分布，不覆盖任意动态范围、稀疏度、矩形、非对齐尺寸、其他GPU、跨天/重启状态。CPU大尺寸仍为抽样，Sanitizer未完成，逐位一致不是无竞争证明；没有SM逐时跟踪或控制实验来给各因素分配因果百分比。

本地导航初查的221条引用目标实际均存在，没有虚构“补回缺失结果”。问题是其中99条引用的目标当时未纳入Git，包括26/27章和部分SIMT归档，因此GitHub上可能不可达。已在根README、阅读路线、Windows结果索引及26/27章标明“本地已有、未随a54435d发布”，加入本章与新归档入口，并在scripts/README补本轮脚本表。保持实际路径，不创建空壳文件。五份导航文件最终229条本地链接、2个锚点均通过；107条引用的目标仍未跟踪（含本轮新文档），29条外部URL没有联网验证，见 `navigation-audit/summary.json`。远端发布仍待另行授权，本轮未推送。

## 33.8 准确复跑命令

在 **PowerShell 7** 中执行；`$out` 与 build 路径均为新目录，不覆盖历史。初始化脚本只改变当前进程环境。

```powershell
Set-Location 'D:\CUDA-Lab\cuda-learning\SGEMM_CUDA'
$ps7 = 'C:\Users\Administrator\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\powershell\pwsh.exe'
$py = 'D:\Anaconda\Anaconda\python.exe'
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$out = "results\windows\tensorcore-bn64-robustness-$stamp"
$build = "build-windows-tensor-bn64-robustness-$stamp"
& $ps7 -NoProfile -File scripts\windows\Build-TensorBN64Robustness.ps1 -BuildDir $build -EvidenceDir "$out\build01"
if ($LASTEXITCODE) { throw 'build failed' }
& $ps7 -NoProfile -File scripts\windows\Run-TensorBN64Robustness.ps1 -BuildDir $build -OutputDir "$out\run" -PrepareOnly
if ($LASTEXITCODE) { throw 'plan failed' }
& scripts\windows\Initialize-Toolchain.ps1
& ".\$build\windows_tensor_bn64_robustness_benchmark.exe" --smoke "$out\smoke"
if ($LASTEXITCODE) { throw 'smoke failed' }
& $ps7 -NoProfile -File scripts\windows\Test-TensorBN64RobustnessSanitizer.ps1 -BuildDir $build -OutputDir "$out\sanitizer01"
# 阅读 result.json；附加超时只保留未验证，不无限等待或绕过权限。
& $ps7 -NoProfile -File scripts\windows\Run-TensorBN64Robustness.ps1 -BuildDir $build -OutputDir "$out\run" -ExecutePrepared -SmokeEvidenceDir "$out\smoke"
if ($LASTEXITCODE) { throw 'formal run failed; raw files retained' }
& $py scripts\summarize_tensor_bn64_robustness.py --run "$out\run" --output "$out\analysis"
if ($LASTEXITCODE) { throw 'summary failed' }
& $py scripts\verify_tensor_bn64_robustness.py --run "$out\run" --analysis "$out\analysis" --output "$out\verification"
if ($LASTEXITCODE) { throw 'independent verification failed' }
& $ps7 -NoProfile -File scripts\windows\Profile-TensorBN64Robustness.ps1 -BuildDir $build -OutputDir "$out\profile01"
if ($LASTEXITCODE) { throw 'profile failed; evidence retained' }
```

只复算本次已归档数据，无需运行GPU：

```powershell
$archive = 'results\windows\20261021-tensorcore-bn64-robustness'
$check = "results\windows\bn64-stats-check-$(Get-Date -Format 'yyyyMMdd-HHmmss')"
& $py scripts\summarize_tensor_bn64_robustness.py --run "$archive\run" --output "$check\analysis"
& $py scripts\verify_tensor_bn64_robustness.py --run "$archive\run" --analysis "$check\analysis" --output "$check\verification"
& $py scripts\audit_tensor_bn64_robustness_evidence.py --archive $archive --build "$archive\build01" --profile "$archive\profile01" --output "$check\evidence"
```

阶段到此收尾。适合自检的问题：为什么3072合法但不能外推任意尺寸？为什么2048收益不能解释为驻留CTA翻倍？为什么命中率升高仍会多读DRAM？为什么两种中位数比值可能不同？为什么576次逐位一致仍不能宣称racecheck通过？
