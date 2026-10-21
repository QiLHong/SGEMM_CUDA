# 29. 从作者 Tensor Core 原核出发：CTA 分组优化与反例

> 后续进展（2026-10-22）：本章保留当时实验的参数、数据与局限。多 seed 稳健性验证见[第 33 章](33-TensorCore-BN64稳健性与尺寸适用范围.md)，官方 CUTLASS 对照及其精度失败见[第 34 章](34-CUTLASS与BN64的4090D公平对照.md)。后续批次不填入本章旧表；这里的“本轮”“下一步”按本章实验日期理解。

> 维护说明：本章是 W 分组实验的历史报告。其后 BK32、BN64 已分别完成，见[第 30 章](30-TensorCore-K分块实验与资源代价.md)、[第 31 章](31-TensorCore-BN64优化与尺寸效应.md)。完整 Tensor Core 原理见[微教材第 9 章](微教材/09-TensorCore从SIMT到矩阵指令.md)，不要把这里的 W1 收益叠加到后续默认 W8 结果。

日期：2026-10-21。仅 RTX 4090 D / Windows 原生实测；没有 RTX 3080 数据。

本轮结论：**已经做了代码优化实验，但没有获得通用的大幅提速。**
将网格分组宽度从 8 改为 1，TF32 的 1024 阶在两批独立测量中均有约 1% 收益；
2048 收益很小，4096 与 FP16 没有稳定收益。因此保留原版默认实现，实验版单独存放。

## 1. 上一轮是不是直接跑作者源码

[第28章](28-TensorCore初体验与4090D实测.md) 的四个 **device kernel** 从作者源码原样提取，
tile、线程归属、shared 布局、MMA 与同步算法没有改。
外层使用独立 Windows CUDA benchmark，显式准备 TF32 / FP16 输入，替换了 PyTorch 绑定与计时。
所以它属于“原 device kernel 的原生移植复现”，不是整个 Python 入口原封执行。

本轮是在原核基础上做实验修改，不能再把候选称为作者未修改的原核。

- 作者：QiLHong / vitamin-cuda；参考提交 `520f97093a2abc239977be83d244b327d0e2e2ef`。
- [TF32 原源码](https://github.com/QiLHong/vitamin-cuda/blob/520f97093a2abc239977be83d244b327d0e2e2ef/kernels/sgemm/sgemm_tf32.cu)：
  `sgemm_tf32_swizzle_bcf_dbf_kernel`。
- [FP16 原源码](https://github.com/QiLHong/vitamin-cuda/blob/520f97093a2abc239977be83d244b327d0e2e2ef/kernels/hgemm/hgemm.cu)：
  `hgemm_bcf_dbf_rw_kernel`。
- 原核、原函数名、[许可证](../../experiments/tensorcore/vendor/LICENSE)均保留。
- SGEMM_CUDA 实验开始时提交 `545fb01a576ae38d84d00df6359e084eb14c94c2`，
  工作分支 `results/rtx4090d-k30-stage2-20261021`；该时点本轮成果尚未提交；随后发布为 `b718c7b`。
- 两仓库均检查了状态和适用 AGENTS.md；未找到 AGENTS.md；vitamin 工作区保持干净。

## 2. 优化假设、改动边界与代码

**预先提出的假设**：缩小 CTA 列分组，可能让较近发射的 CTA 复用同一列 B tile，
减少 L2/DRAM 流量；代价是较少连续复用同一行 A tile。这个效果可能依赖尺寸。

只改一个源码因素：grid swizzle 的分组宽度 W。扫描 W=1、2、4，与原 W=8 对照。
没有改 tile、warp、MMA、shared swizzle、输出类型、精度或双缓冲协议。
网格映射也会影响索引指令，**单个源码因素不等于只影响一种硬件效应**。

先看这四个文件：

- [TF32 实验核](../../experiments/tensorcore-grid/tf32_grid.cuh)
- [FP16 实验核](../../experiments/tensorcore-grid/fp16_grid.cuh)
- [TF32 全部 device 差异](../../experiments/tensorcore-grid/tf32-device.diff)
- [FP16 全部 device 差异](../../experiments/tensorcore-grid/fp16-device.diff)

两个差异文件都只有：

```cpp
template <const int GRID_W, /* 原有 BM、BN、BK 等参数 */>
// ...
const int SWIZZLE_W = GRID_W;  // 原来是 8
```

新增 namespace 防止与原核重名；函数名保留。
[生成/校验脚本](../../scripts/prepare_tensor_grid.py)会反向恢复这两处并检查与原函数逐字符一致。
[provenance.json](../../experiments/tensorcore-grid/provenance.json)记录来源与哈希。
基线仍直接调用原 vendor 核；没有用一个重新写的 W8 版本冒充基线。

## 3. Grid、CTA 和线程究竟算哪里

设 `gx=N/128`、`gy=M/128`，物理 grid 的线性序号：

```cpp
l  = blockIdx.y * gx + blockIdx.x;
bx = l % W + l / (W * gy) * W;
by = l / W % gy;
```

该 CTA 写 `C[by*128 : by*128+128, bx*128 : bx*128+128]`。
这里的 grid swizzle 改的是 CTA 对应哪个输出块；shared swizzle 改的是一个块内的共享内存地址，
是两个不同层次。

以 1024 阶、`gx=gy=8` 为例，前几个逻辑输出块坐标 `(by,bx)`：

| W | l=0 | l=1 | l=2 | l=3 | l=4 | l=8 |
|---|---|---|---|---|---|---|
| 8 | (0,0) | (0,1) | (0,2) | (0,3) | (0,4) | (1,0) |
| 4 | (0,0) | (0,1) | (0,2) | (0,3) | (1,0) | (2,0) |
| 1 | (0,0) | (1,0) | (2,0) | (3,0) | (4,0) | (0,1) |

W=1 退化为 `bx=l/gy, by=l%gy`。列上的 CTA 可复用 B，行上的 CTA 可复用 A。
CUDA 不保证所有 CTA 严格按 l 的顺序执行；这是空间归属调整，不能当作固定调度时序。

W 整除 gx 时，反向关系
`l=(bx/W)*(W*gy)+by*W+bx%W` 唯一，映射构成一个排列。
主机仍保守要求 M 为128倍数、N为1024倍数、K为32倍数，并限制 M/N/K≤4096。
它会拒绝不支持形状，不能根据 grid 向上取整推断有尾块保护。

CTA 内完全不变：256线程、8个warp；warp排列2×4，每warp算64×32，每线程64个输出。
对 `tid=32*warp+lane`、`mi,ni=0..3`、`r=0..3`：

```text
row = by*128 + (warp/4)*64 + mi*16 + lane/4 + (r/2)*8
col = bx*128 + (warp%4)*32 + ni*8 + (lane%4)*2 + r%2
```

例如 warp0 的 lane0 对每个 fragment 的相对坐标是
`(0,0),(0,1),(8,0),(8,1)`；lane1为列2、3，lane2为列4、5；
lane31为行7、15与列6、7的四种组合。再加上 `mi*16,ni*8` 即得到全部64个输出。
FP16 RW 的最终全局写回由另一个协作映射完成，不能把 accumulator 所有者等同于 store 线程。

实际 CPU 枚举了一个完整 CTA 的16384个输出与搬运地址，
另枚举4种形状×4种宽度，共5408个 CTA 映射：
无重复、无遗漏、无越界。见 [mapping](../../results/windows/20261021-tensorcore-grid/mapping)。

## 4. 数据流与同步没有被这次优化改变

TF32：128×128×16；FP16：128×128×32。
每线程用 `cp.async` 搬运16字节，global → shared 两套 buffer；
随后 `ldmatrix` / shared load → operand registers → MMA → FP32 accumulator。
TF32直接写FP32 C；FP16转成half，经shared重排后协作写回half C。

```text
初始：copy tile0→buffer0 → commit → wait → CTA barrier
循环：copy tile(t+1)→另一buffer → commit
      读取当前buffer → registers → MMA(tile t)
      wait → CTA barrier → 交换读写buffer
尾部：计算最后一个tile → 写C
```

`commit` 不能代替 `wait`；`wait` 也不能代替跨线程的 CTA barrier。
本轮没有移除任何同步点。FP16写回复用shared时的额外barrier也保留。

实际SASS表明两类原核与所有候选都保留异步拷贝与MMA交错：
主循环4条LDGSTS、32条HMMA；
最后一条copy与 `DEPBAR.LE` 之间，TF32有28条HMMA、FP16有17条。
这是存在重叠机会的证据，不等于这些指令的全部延迟都被隐藏。

注意 `LDGDEPBAR` 是提交相关指令，不能用包含字符串 `DEPBAR` 就把它识别成等待。
中间分析audit01曾犯此错，已保留并纠正；最终引用
[evidence-final](../../results/windows/20261021-tensorcore-grid/evidence-final)，不引用audit01的wait列。

## 5. 真正执行的构建和验证

- GPU：NVIDIA GeForce RTX 4090 D，24564 MiB，WDDM；驱动610.60。
- CUDA：Windows nvcc 12.4.131；路径 `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin\nvcc.exe`。
- MSVC：19.37.32826.1；工具集 `D:\VS\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.37.32822`。
- Windows SDK：10.0.26100.0；Ninja / Release；cuBLAS版本120405。
- 独立构建目录 `build-windows-tensor-grid-sm89`。
- 实际 nvcc 命令包含 `arch=compute_89,code=[compute_89,sm_89]`；
  使用 `-O3 -lineinfo -Xptxas=-v`，不使用 `--use_fast_math` 或忽略编译器兼容检查的参数。
- 两个原版核的SASS指令序列与第28章build04逐条一致。候选都真实生成HMMA与LDGSTS。
- 350项已记录源码、1081项历史结果文件哈希未变；未覆盖历史归档。

数值检查：

- 先跑128×1024×32、128×1024×64，CPU double全量检查131072个输出。
- 1024、2048、4096方阵：全量GPU参考检查；CPU double每种精度各检查256个固定位置。
- 两批共352次原版/候选全量逐位比较，差异为0；无NaN/Inf。
- 同精度cuBLAS输出与本轮自定义核一致；保留原始FP32、量化输入PEDANTIC参考、
  CPU double及所有旧判据，未放宽阈值。
- 原始FP32参考下仍不满足严格SGEMM精度：1024/2048/4096最大绝对误差，
  TF32分别为0.3443909 / 0.5039673 / 0.7473450，
  FP16分别为0.4996338 / 0.7386475 / 1.1427002。
  原 `abs<=0.01` 与 SIMT `atol=rtol=0.001` 均保留FAIL。
- 作者Tensor精度判据单独记录：TF32为0.1，FP16为0.016。
  通过该契约不等于通过严格FP32，也不能用它覆盖量化输入参考的0.01失败。
- Sanitizer未成功完成，保持“未验证”；本轮未重跑原Python入口。

详细见各run中的 `.checks.csv`、`.bitwise.csv` 和
[errors.csv](../../results/windows/20261021-tensorcore-grid/confirm-analysis02/errors.csv)。

## 6. 公平计时与扫描结果

所有实现 `C=AB`，alpha=1、beta=0，seed=20261020；同精度组共享相同输入字节。
TF32输入显式转换，FP16输入显式转half，预处理不计入GEMM events。
输入指纹保存于 `.meta.txt`，扫描与确认两批的6项输入指纹逐尺寸一致。

每轮10次预热、30次独立CUDA event，7轮。每一次预热/测量前，
同stream D2D恢复全零C0，在event之外。恢复仍影响缓存；
FP16复制字节数为TF32的一半，因此不能把跨精度结果当作单因素算法比较。
cuBLAS使用对应Tensor精度；PEDANTIC FP32仅用作误差参考，本轮未重新测量严格SGEMM性能。

扫描轮换、反转每种精度内5个实现的顺序；扫描用于筛选候选。
下表为**轮均值延迟的中位数**换算的TFLOPS：

| 实现 | 1024 | 2048 | 4096 |
|---|---:|---:|---:|
| TF32 原W8 | 31.410 | 48.368 | 71.316 |
| TF32 W1 | 31.742 | 48.620 | 71.362 |
| TF32 W2 | 31.316 | 48.267 | 71.693 |
| TF32 W4 | 31.411 | 48.289 | 71.569 |
| TF32 cuBLAS | 35.029 | 50.473 | 66.973 |
| FP16 原W8 | 60.958 | 97.656 | 139.786 |
| FP16 W1 | 61.023 | 97.561 | 141.049 |
| FP16 W2 | 60.794 | 98.999 | 139.718 |
| FP16 W4 | 61.019 | 96.254 | 140.704 |
| FP16 cuBLAS | 79.038 | 110.380 | 138.374 |

例如扫描中FP16原W8的2048轮均值CV高达27.26%，不能看到候选较高数字就宣布优化成功。
全部6300个正式events保留，见 [scan01](../../results/windows/20261021-tensorcore-grid/scan01)、
[扫描统计](../../results/windows/20261021-tensorcore-grid/scan-analysis02)。

## 7. 独立确认：W1是否真的值得采用

根据扫描选W1后，重新采一批数据。原版/候选相邻、奇偶轮交换先后；
cuBLAS在配对之外轮换。全部3780个正式events保留。
加上两批小尺寸测试，共额外40个小尺寸events；没有删除慢样本。

| 精度/尺寸 | 原W8 TFLOPS | W1 TFLOPS | cuBLAS TFLOPS | W1对W8逐轮提升中位数 | 提升范围 | 胜出轮数 |
|---|---:|---:|---:|---:|---:|---:|
| TF32 1024 | 31.332 | 31.742 | 35.108 | +1.108% | +0.811%～+1.572% | 7/7 |
| TF32 2048 | 48.470 | 48.602 | 51.819 | +0.279% | +0.213%～+0.675% | 7/7 |
| TF32 4096 | 66.907 | 71.040 | 68.985 | +0.041% | −2.309%～+8.337% | 4/7 |
| FP16 1024 | 60.787 | 61.253 | 69.070 | +0.485% | −0.959%～+10.127% | 4/7 |
| FP16 2048 | 99.903 | 100.121 | 127.395 | +0.161% | −7.137%～+1.055% | 4/7 |
| FP16 4096 | 148.465 | 150.165 | 144.633 | +0.288% | −6.983%～+7.821% | 5/7 |

提升定义：对每轮算 `原版轮均值延迟 / 候选轮均值延迟 - 1`，再取7个比例的中位数。
**分别取中位数再相除不是同一个统计量。**
4096 TF32若只用71.040/66.907，会读出约6.18%；
但同轮配对中位数仅0.041%，不足以证明有6%的优化。
应结合 [逐轮原始配对](../../results/windows/20261021-tensorcore-grid/confirm-analysis02/paired-rounds.csv)看。

轮均值CV：TF32 W8/W1在1024为0.25%/0.37%，2048为0.25%/0.10%，
4096为3.46%/3.57%；FP16在4096为4.75%/4.88%。
cuBLAS FP16的1024确认批CV为17.58%，也如实保留。
扫描遥测43～48°C，确认44～52°C；两批SM时钟2520～2715 MHz，未锁频，
这是Windows/WDDM下的一组条件，不能等同Linux或其他GPU。

W1对同精度cuBLAS：1024、2048的配对中位数均较慢；
4096 TF32为+3.112%（4/7胜），FP16为+1.683%（5/7胜）。
这些结果不足以声称稳定超过cuBLAS，更不是对所有cuBLASLt算法的穷举比较。

## 8. Nsight和SASS支持什么、不支持什么

当前已装NCU 2024.1.1成功采样：3尺寸×2精度×原W8/W1=12次。
采用kernel replay、clock-control none、cache-control none；
每种情况只有一个profile样本、5次预热，不与公平benchmark时间混用。
NCU使用自带sections；未修改HOME、权限、驱动或安装任何依赖。

| 精度/尺寸 | W8→W1 L2 hit% | W8→W1 DRAM读(MB，十进制) | Tensor pipe active% | eligible warps |
|---|---:|---:|---:|---:|
| TF32 1024 | 99.93→99.99 | 0.047→0.010 | 40.86→41.28 | 0.13→0.13 |
| TF32 2048 | 95.42→95.42 | 33.740→33.788 | 44.28→44.41 | 0.17→0.18 |
| TF32 4096 | 94.71→93.79 | 364.539→461.960 | 45.87→45.90 | 0.19→0.19 |
| FP16 1024 | 99.93→99.93 | 0.027→0.026 | 42.23→42.29 | 0.14→0.14 |
| FP16 2048 | 95.42→95.45 | 16.903→16.790 | 46.57→46.69 | 0.19→0.19 |
| FP16 4096 | 97.52→97.67 | 71.701→67.640 | 48.39→48.53 | 0.20→0.20 |

4096 TF32的DRAM读增加约26.7%，反驳了“W越小缓存一定越好”的笼统说法。
1024基本已命中L2，不能从小幅hit变化推导1%的全部收益。
FP16的4096虽然DRAM读减少，配对速度仍不稳定；
单次profile、重放与缓存状态也限制了因果解释。

资源证据：

- TF32原版/各候选都为127 registers/thread，FP16都为128；
  静态shared均32768字节，NCU含driver开销报告33792字节。
- ptxas均无spill；NCU local load/store sectors均为0。
- TF32的4096 barrier指标W8/W1均0.48，FP16为0.80→0.75；
  没有源码同步减少，不能称作“删barrier提速”。
- shared wavefront、bank conflict、occupancy、issue、long/short scoreboard均保存于
  [完整NCU指标](../../results/windows/20261021-tensorcore-grid/confirm-analysis02/ncu-metrics.csv)，
  不把动态计数相近等同于没有bank conflict。

静态指令证据：

| 核 | W8非NOP静态指令 | W1非NOP静态指令 | K主循环指令数 | 每个主循环HMMA |
|---|---:|---:|---:|---:|
| TF32 DBF | 494 | 471 | 111→111 | 32→32 |
| FP16 DBF RW | 650 | 627 | 99→99 | 32→32 |

主循环前的代码缩短，W1的索引公式也更简单。
这支持“启动/索引开销下降可能帮助小矩阵”的解释，
但未做“相同映射只改索引算术”的第二个对照，所以仍属于解释而非已分离的因果结论。
以上是静态条数，不是运行一次GEMM执行的指令总数。

## 9. 决策与下一步

已验证：算法按Tensor精度复现；候选与原版逐位一致；
TF32 1024有两批约1%的小收益，其他情况不能证明通用改进。
保留原基线，W1作为小矩阵实验选项，未改项目默认调度。

下一轮更值得独立验证的假设：**TF32把K tile从16加到32，能否摊薄每段等待与地址计算成本？**
这要求同步审计搬运覆盖与尾部，64 KiB双buffer可能使CTA驻留数下降，
还需用动态shared opt-in。因此它可能更慢，不能只改一个模板数值就直接运行。
本轮当时没有实现或测量这个假设；后续[第30章](30-TensorCore-K分块实验与资源代价.md)已完成：BK32在1024小幅受益，在2048/4096回退，不能把减少同步次数直接等同于提速。

## 10. 实际可复跑命令

在普通PowerShell中运行以下命令；脚本自行在子进程加载兼容MSVC环境。
不需要管理员，不修改系统环境变量。使用新归档路径，脚本拒绝覆盖已有输出。

```powershell
Set-Location 'D:\CUDA-Lab\cuda-learning\SGEMM_CUDA'
$ps7 = 'C:\Users\Administrator\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\powershell\pwsh.exe'
$py = 'D:\Anaconda\Anaconda\python.exe'
$out = "results\windows\tensor-grid-$(Get-Date -Format yyyyMMdd-HHmmss)"

& $ps7 -NoProfile -File .\scripts\windows\Build-TensorGrid.ps1 -EvidenceDir "$out\build"
& $ps7 -NoProfile -File .\scripts\windows\Run-TensorGrid.ps1 -OutputDir "$out\scan"
& $ps7 -NoProfile -File .\scripts\windows\Run-TensorGrid.ps1 -Selection grid_w1 -OutputDir "$out\confirm"
& $py .\scripts\summarize_tensor_grid.py --run "$out\scan" --output "$out\scan-analysis"
& $py .\scripts\summarize_tensor_grid.py --run "$out\confirm" --output "$out\confirm-analysis"
& $py .\scripts\analyze_tensor_grid_mapping.py --output "$out\mapping"

# 可选：与上面的benchmark依次执行，不并发占用GPU
& $ps7 -NoProfile -File .\scripts\windows\Profile-TensorGrid.ps1 -OutputDir "$out\profile"
& $py .\scripts\summarize_tensor_grid.py --run "$out\confirm" --profile "$out\profile" --output "$out\profile-analysis"
```

校验提取和派生源码：

```powershell
& $py .\scripts\extract_vitamin_tensor.py --verify
& $py .\scripts\prepare_tensor_grid.py --verify
& $py .\experiments\tensorcore-grid\derive_benchmark.py --verify
```

归档入口：[20261021-tensorcore-grid](../../results/windows/20261021-tensorcore-grid)。
包含原始events、所有误差、遥测、编译命令、可执行文件、源码快照、NCU报告、
PTX/SASS、统计和最终文件清单。中间的空scan-analysis01记录首次统计文件匹配错误；
audit01的等待识别错误已纠正，最终证据以evidence-final为准。
