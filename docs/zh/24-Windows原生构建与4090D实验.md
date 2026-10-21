# Windows 原生构建与 RTX 4090 D：从线程坐标到受控实验

本章延续 `zh-cn-learning` 的 K10 教学入口和 K30 V2 独立研究入口，在 Windows 原生 PowerShell、CUDA Toolkit、MSVC 上复现。Windows 结果另行归档；README 中的 A6000、RTX 3080 记录保留原来的机器和测量含义。

阅读方式借鉴 vitamin-cuda 中文入门分支：先追踪一个输出元素，再推导线程分工、数据搬运和同步，最后用同条件实验检验优化假设。这里使用本项目的 CUDA 实现；不需要安装 vitamin-cuda 的 PyTorch 或其他 Python 依赖。

## 1. 先核对真正参与编译的工具

本机检查到以下 Windows 工具。`nvidia-smi` 的 CUDA UMD 版本是驱动能力，不能用来替代 `nvcc --version`。

| 项目 | 本次选择或检查结果 |
|---|---|
| GPU | NVIDIA GeForce RTX 4090 D，24564 MiB，WDDM |
| 驱动 | 610.60 |
| CUDA Toolkit | 12.4 Update 1，`nvcc V12.4.131` |
| nvcc | `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin\nvcc.exe` |
| Visual Studio | Community 2022，17.14.36327.8 |
| MSVC 工具集 | 14.37.32822，编译器 `19.37.32826.1`，x64 |
| Windows SDK | 10.0.26100.0，位于 `D:\Windows Kits\10` |
| CMake | VS 附带的 3.31.6-msvc6 |
| Ninja | VS 附带的 1.12.1 |
| PowerShell | 已有 PowerShell 7.6.5；脚本要求 7.0 或以上 |
| CUDA 架构 | `89`，实际设备代码目标 `sm_89` |

VS 的 IDE 版本与 C++ 工具集版本是两个概念。本机同时安装 14.37 和 14.44；本次明确选择 14.37。NVIDIA 的 [CUDA 12.4.1 Windows 编译器支持表](https://docs.nvidia.com/cuda/archive/12.4.1/cuda-installation-guide-microsoft-windows/index.html#system-requirements)列出 MSVC 193x，因此不因 PATH 中先找到 14.44 就直接使用它，也不添加忽略编译器版本检查的参数。

PowerShell 中能找到 `cl.exe` 还不够。编译和链接同时依赖 MSVC、UCRT、Windows SDK 的头文件与库目录；构建脚本应通过 VS 的 `vcvarsall.bat x64 -vcvars_ver=14.37` 初始化本次进程环境。没有找到 PATH 中的 CMake，也不等于机器没有 CMake：这里复用了 VS 附带版本。

构建和复跑脚本、实际日志索引在本章第 7 节。所有命令使用 Windows 原生工具，不依赖 WSL。Release 用于计时；Debug 的设备调试选项会改变性能，不能混入性能表。

## 2. 先把一个输出元素写清楚

三个矩阵采用行主序：

```text
A: M 行 K 列，A[m,k] 的地址下标是 m*K+k
B: K 行 N 列，B[k,n] 的地址下标是 k*N+n
C: M 行 N 列，C[m,n] 的地址下标是 m*N+n

C[m,n] = alpha * Σ(k=0..K-1) A[m,k]*B[k,n] + beta*C0[m,n]
```

`C0` 是每次调用前相同的初始矩阵。若 `beta=3` 时把上次输出接着作为下次输入，第二次计算就变成另一道题；重复运行还会放大数值。每次复位 C 是正确性与公平比较的一部分。

cuBLAS 默认按列主序解释矩阵。在严格 FP32 调用中把 B 指针放在 A 参数位置、A 指针放在 B 参数位置，利用的是 `Cᵀ=BᵀAᵀ`，最终物理内存仍表示我们需要的行主序 C。这没有把要求的数学运算改成 B×A。对照真实调用见 [`scripts/k30_audit.cu`](../../scripts/k30_audit.cu) 和新增的 [`scripts/windows_benchmark.cu`](../../scripts/windows_benchmark.cu)。

## 3. K10：一个 block、一个 warp、一个 thread 分别算什么

先打开 [`src/kernels/10_kernel_warptiling.cuh`](../../src/kernels/10_kernel_warptiling.cuh)，依次看 `sgemmWarptiling`、`wt::runWarptilingCore`、`wt::loadFromGmem`、`wt::processFromSmem`。默认参数的来源在 [`src/runner.cu`](../../src/runner.cu) 的 `runSgemmWarptiling`。

| 参数 | 数值 | 在代码中的含义 |
|---|---:|---|
| BM、BN | 128、128 | 一个 block 的输出矩形 |
| BK | 16 | 一轮共享内存缓存覆盖的 K 长度 |
| NUM_THREADS | 128 | 一个 block 有 4 个 warp |
| WM、WN | 64、64 | 一个 warp 的输出矩形 |
| TM、TN | 8、4 | 线程在一个子块中负责的小矩形 |
| WNITER | 4 | warp 沿 N 方向划为 4 个子块 |
| WMITER | 1 | 由 `WM*WN/(32*TM*TN*WNITER)` 推导 |
| WSUBM、WSUBN | 64、16 | 每个 warp 子块的高、宽 |
| AS_PAD | 4 | A 的共享内存每个 k 行额外保留 4 个 float |

block `(bx,by)` 负责 C 的 `[128*by,128*by+127] × [128*bx,128*bx+127]`。grid 为 `(N/128,M/128)`；x 对应列，y 对应行。

令 `t=threadIdx.x`、`w=t/32`、`l=t%32`：

```text
warpRow = w/2        warpCol = w%2
threadRowInWarp = l/4
threadColInWarp = l%4

m = 128*by + 64*warpRow + 8*(l/4) + i
n = 128*bx + 64*warpCol + 16*s + 4*(l%4) + j
i=0..7，s=0..3，j=0..3

threadResults[i*16 + s*4 + j] 对应 C[m,n]
```

每线程负责 8×16=128 个输出，但这些列分散在四组连续的 4 列中。线程有 128 个部分和，不表示它会启动 128 个硬件线程。

以 block `(0,0)` 的 warp 0 为例：

| lane | 行 | 列 |
|---:|---|---|
| 0 | 0～7 | 0～3、16～19、32～35、48～51 |
| 1 | 0～7 | 4～7、20～23、36～39、52～55 |
| 4 | 8～15 | 与 lane 0 相同的四组列 |
| 31 | 56～63 | 12～15、28～31、44～47、60～63 |

warp 1 把列整体加 64，warp 2 把行整体加 64，warp 3 同时把行、列加 64。检验覆盖关系：`4 warps × 32 lanes × 128 outputs = 128×128`；还要结合上述下标确认这些输出没有重复，不能只看元素总数。

## 4. A、B、C 在哪里流动，为什么需要同步

K10 的一轮工作可以沿代码写成：

```text
全局 A/B → 每线程搬运用的 float4 → block 的 shared As/Bs
        → 每线程 regM[8]/regN[16] → threadResults[128]
所有 K tile 完成后：threadResults 与全局 C0 → 最终全局 C
```

### 4.1 从全局内存搬入共享内存

设当前 K tile 起点为 `bk`。128 个线程共同搬运 A 的 128×16 与 B 的 16×128，共 4096 个 float。默认参数下：

```text
innerRowA=t/4，innerColA=t%4，rowStrideA=32
offsetA=0,32,64,96

global A[(128*by + innerRowA + offsetA)*K + bk + 4*innerColA + q]
    → As[(4*innerColA+q)*AS_STRIDE + innerRowA + offsetA]
q=0..3，AS_STRIDE=128+AS_PAD

innerRowB=t/32，innerColB=t%32，rowStrideB=4
offsetB=0,4,8,12

global B[(bk + innerRowB + offsetB)*N + 128*bx + 4*innerColB + q]
    → Bs[(innerRowB+offsetB)*128 + 4*innerColB + q]
```

A 在共享内存中按 `[k,m]` 存储，B 仍按 `[k,n]` 存储。全局矩阵没有被整体转置，也没有因此改变数学公式。`float4` 表示一次访问四个相邻 FP32 值，所需地址要满足 16 字节对齐；它不会把 FP32 变成低精度计算。

线程 0 搬运 A 的局部行 0、32、64、96，每行 k=0～3；同时搬运 B 的局部 k 行 0、4、8、12，每行列 0～3。**搬运者与使用者不是一一对应**：这些数据会被整个 block 的其他线程读取并复用。

### 4.2 从共享内存取 fragment，做寄存器外积

`processFromSmem` 对 `dotIdx=0..15` 循环。每步给线程填 8 个 A 值和 16 个 B 值，再做 8×16 次乘加，更新该线程的 128 个输出部分和。这就是寄存器分块：一次读入的 A 与多个 B 配对，一次读入的 B 与多个 A 配对。

默认共享内存的程序声明容量为：

```text
As: 16*(128+4)*4 = 8448 bytes
Bs: 16*128*4     = 8192 bytes
合计             = 16640 bytes
```

源码数组长度描述逻辑需求，不能直接当成编译器报告的物理寄存器数量。地址、临时值与编译器调度都会影响资源占用；应读本次 `ptxas` 报告，不能套用 RTX 3080 历史报告。

### 4.3 两次屏障解决两种数据依赖

第一处 `__syncthreads()` 在写完 As/Bs 后，保证所有线程都完成这一轮写入，其他线程才开始读取。第二处在本轮计算后，保证所有线程读完旧 tile，下一轮才能覆盖相同的 shared 数组。只靠 warp 内同步不足以保护跨 warp 的生产与消费。

全部 K tile 结束后，才读取 C0，计算 `alpha*threadResults+beta*C0`，按 `float4` 写回。C 不在每个 K tile 后写回全局内存；部分和始终跨 tile 保留。

默认 K10 要求 M、N 分别整除 128，K 整除 16，N/K 满足 `float4` 对齐要求；A、B、C 不重叠。它没有为所有尾部元素加边界屏蔽。测试 1024、2048、4096 都满足这些条件，不能据此宣称支持 1000×1000 或任意切片。

## 5. K30 V2 是哪个实现

K30 的入口是独立的 [`scripts/k30_audit.cu`](../../scripts/k30_audit.cu)，`v2` 对应 [`30_kernel_mini_simt.cuh`](../../src/kernels/30_kernel_mini_simt.cuh) 的 `Shape<128,128,16,64,64,4,4,128>` 和 `STAGES=2`。编号 30 不代表它一定比编号 10 快。

V2 仍然是 128×128 block、4 个 warp、每线程 128 个输出，变化在 fragment 的所有权和双槽同步协议。对 `l=lane`：

```text
lm = 2*(l/16) + l%2，ln=(l%16)/2
A fragment 行 = 64*warpRow + 4*lm + 16*(i/4) + i%4，i=0..15
B fragment 列 = 64*warpCol + 4*ln + 32*(j/4) + j%4，j=0..7
```

因此每次外积是 16×8，K10 是 8×16。K30 V2 不能作为“只改 shared stage 数”的 K10 对照；它在本章中是另外一个已有实现的基线。

双槽先装好 tile 0。计算 tile `t` 时，源码先把 tile `t+1` 读进临时寄存器，计算当前槽，再写另一槽，最后做全 block 屏障。这个屏障同时发布下一份数据并确认当前数据不再被使用。两个槽共声明 33280 字节 shared。这里没有使用 Tensor Core 或 `cp.async`，也不能仅因源码先写了加载，就断言编译器生成了有效的访存计算重叠。旧 RTX 3080 的 SASS 结论属于旧环境；本次性能结果须独立测量。

## 6. 本次受控实验：只改 A shared stride

在编写实验版本前提出的假设是：K10 把 A 转置写入 shared 时，不同 k 行的 bank 映射可能影响访存代价；Ada 上原来的 padding=4 未必是最合适的选择。实验把 **AS_PAD 从 4 改为 8**，即 `AS_STRIDE:132→136`，其他主要因素保持一致。

新入口直接实例化原来的 K10 模板，因此计算循环、输出分工、全局 `float4` 搬运、B 布局、两处屏障、精度和 launch bounds 都相同。没有复制一份 kernel 后悄悄修改多处代码，也没有修改默认 K10。

对 32 个 4 字节 bank 的简化地址模型，A 的 bank 为 `(k*AS_STRIDE+m)%32`。stride 132 时相邻 k 行的基址移动 4 个 bank，136 时移动 8 个 bank；改变的是物理布局，不是矩阵中的 m/k 坐标。同时 shared 用户容量从 16640 增至 16896 字节，多出 256 字节。

在测量前再把第 4 节的搬运坐标代入。若编译器把 A 的四个分量写入生成为分别执行的标量 shared store，对同一 `q`：

```text
bank = ((4*(lane%4)+q)*(128+pad) + lane/4) % 32
```

例如 `q=0`，lane 0、1、2、3 写入不同 k、相同 m：PAD4 的 bank 是 0、16、0、16；PAD8 则都是 0。枚举整个 warp，PAD4 覆盖 16 个 bank、每 bank 2 个不同地址，PAD8 覆盖 8 个 bank、每 bank 4 个不同地址。**这个模型预期 PAD8 可能恶化转置写入冲突**，而不是“padding 越多越好”。本实验也因此能检验一个常见误解。

这是对特定标量 store 的预测。warp 的实际请求如何拆分、指令宽度是否变化，以及最终耗时受哪些因素影响，仍需要 profiler 或 SASS 证据。模型不等于测量，也不能把 shared store 的推导直接套成 shared load 的冲突数。实验可以没有提速，这仍是有效结果。

验证步骤是：先比较全部输出元素，保持原项目绝对误差阈值 `0.01`，明确检查 NaN/Inf；再在相同 A/B/C0、alpha/beta、预热与计时方式下交替运行两个版本。最终比较多轮中位数与逐轮波动。

## 7. 构建、运行与实测记录

### 7.1 从 PowerShell 7 复跑

使用现有 PowerShell 7。若当前打开的是 Windows PowerShell 5.1，可先启动本机已存在的可执行文件，无需安装或修改客户端运行时：

```powershell
& 'C:\Users\Administrator\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\powershell\pwsh.exe' -NoProfile
```

随后在新的 PowerShell 7 提示符执行：

```powershell
Set-Location 'D:\CUDA-Lab\cuda-learning\SGEMM_CUDA'

# 初始化当前进程的 VS/CUDA 环境，使用独立 Ninja Release 目录，先运行最小 CUDA 验证。
.\scripts\windows\Build.ps1 -BuildDir build-windows-sm89 -Architecture 89 -WithExperiment

# 包含完整正确性矩阵，再做三种性能尺寸；每次创建新的结果目录。
.\scripts\windows\Run-Benchmarks.ps1 -BuildDir build-windows-sm89 -Rounds 7 -Repeats 30 -Warmup 10
```

[`Build.ps1`](../../scripts/windows/Build.ps1) 调用 [`Initialize-Toolchain.ps1`](../../scripts/windows/Initialize-Toolchain.ps1)，无需用户预先进入 Developer PowerShell。`-Architecture 89` 传入 `CMAKE_CUDA_ARCHITECTURES`，实际的 `--generate-code ... sm_89` 应在构建日志及 `build-windows-sm89/compile_commands.json` 中核实。换生成器或不兼容工具集时使用新的构建目录，不混用已有缓存。

[`Run-Benchmarks.ps1`](../../scripts/windows/Run-Benchmarks.ps1) 默认按北京时间创建 `results/windows/yyyyMMdd-HHmmss`，拒绝覆盖已有结果目录。它先运行最小测试、原 `sgemm 10` 入口、原 K30 V2 入口与新增统一入口的正确性矩阵，再测 1024、2048、4096，最后自动调用 [`Summarize-Results.ps1`](../../scripts/windows/Summarize-Results.ps1) 生成按轮及配对汇总。首次学习运行保留这些检查；`-SkipCorrectnessMatrix` 仅供已经完成同一构建验证后的追加计时。

若只想追踪一次调用，可在仓库根目录运行：

```powershell
.\scripts\windows\Initialize-Toolchain.ps1
.\build-windows-sm89\k30_audit.exe v2 1024 3 1 1

# 参数：N beta 轮数 每轮重复数 每轮预热数 CSV路径 实现集合。
# 临时单次检查不是性能报告；CSV 选择一个尚未存在的路径。
.\build-windows-sm89\windows_benchmark.exe 1024 3 1 1 1 .\single-check.csv all
```

### 7.2 Windows 适配为什么需要这些修改

| 改动 | 解决的问题 |
|---|---|
| CMake 的 CUDA 架构统一参数化 | 主程序、示例与独立 audit 都使用显式目标架构，避免主程序 sm_89、audit 仍硬编码 sm_86 |
| 使用现代 `CUDAToolkit` 导入库目标 | 复用本机 CUDA runtime/cuBLAS/NVTX 的实际头文件和链接路径 |
| NVTX 头文件及库按可用组件分支 | 本机提供 header-only NVTX3，缺少旧 NVTX 库；使用 `nvtx3/nvToolsExt.h` 与 `CUDA::nvtx3`，保留旧工具链回退 |
| Windows 下的主机计时/无固定 seed 分支使用 `std::chrono` | MSVC 没有 POSIX 的 `gettimeofday`；Linux 条件分支保留，GEMM 性能仍用 CUDA events |
| 显式补齐类型/标准头文件，`and` 改为 `&&` | Windows 不再依赖其他头文件间接提供 `uint`、`std::string`、`std::numeric_limits`；逻辑条件保持原义 |
| FLOP 计数先转为 `double` | Windows LLP64 的 `long` 只有 32 位，`2*1024³` 已溢出，结果正确也可能打印出错误 GFLOPS |
| PowerShell 显式加载 x64 工具链、引用带空格路径 | 防止找到不兼容的默认 cl，或在库/SDK 环境不完整时编译 |
| CMake 编译器探测也指定 Release | 本机默认 Debug 探测使用 `/Zi` 时遇到 C1902 PDB 错误；Release 探测成功，与本次性能构建保持一致 |

这些适配没有改变 K10/K30 的数学计算与算法编号。脚本明确要求 PowerShell 7，避免 Windows PowerShell 5.1 对原生命令 stderr 的处理差异把编译器正常输出当作脚本失败。

### 7.3 测量约定与结果索引

2026-10-20 北京时间 22:48 开始的本次结果保存在 [`results/windows/20261020-rtx4090d-run01`](../../results/windows/20261020-rtx4090d-run01/)。本地工作分支为 `windows-sm89-learning-20261020`，起点 commit 见来源节；本次新增代码尚未提交，因此同时保存源码快照、差异和哈希，不能仅凭基底 commit 标识这次构建。

| 要核实什么 | 对应文件 |
|---|---|
| GPU、驱动、Toolkit、MSVC、SDK、运行参数 | [`environment.json`](../../results/windows/20261020-rtx4090d-run01/environment.json)、[`nvidia-smi-before.txt`](../../results/windows/20261020-rtx4090d-run01/nvidia-smi-before.txt) |
| 每个目标实际用了什么编译参数 | [`compile_commands.json`](../../results/windows/20261020-rtx4090d-run01/compile_commands.json)、目录中的 `build-*.log` |
| 实际执行命令、原 K10 和 K30 入口检查 | [`run.log`](../../results/windows/20261020-rtx4090d-run01/run.log) |
| 统一入口的六尺寸、三种 beta 正确性 | 目录中的 `correctness-尺寸-beta值.csv` |
| 每次 event 原始毫秒及对应轮末误差 | [`timings-1024.csv`](../../results/windows/20261020-rtx4090d-run01/timings-1024.csv)、[`timings-2048.csv`](../../results/windows/20261020-rtx4090d-run01/timings-2048.csv)、[`timings-4096.csv`](../../results/windows/20261020-rtx4090d-run01/timings-4096.csv) |
| 每轮统计、总汇总、实验配对 | [`rounds.csv`](../../results/windows/20261020-rtx4090d-run01/rounds.csv)、[`summary.csv`](../../results/windows/20261020-rtx4090d-run01/summary.csv)、[`paired-summary.csv`](../../results/windows/20261020-rtx4090d-run01/paired-summary.csv) |
| 源码与可执行文件对应关系 | [`source-snapshot.zip`](../../results/windows/20261020-rtx4090d-run01/source-snapshot.zip)、[`tracked-changes.patch`](../../results/windows/20261020-rtx4090d-run01/tracked-changes.patch)、[`sha256.csv`](../../results/windows/20261020-rtx4090d-run01/sha256.csv) |

本次统一入口实际采用以下约定：

- `alpha=0.5`；性能测量固定 `beta=3`，另用 `beta=0/1/3` 验证语义。
- 固定 `seed=20261020`，使用显式 32 位整数随机过程生成输入。A、B、C0 对所有实现保持一致。
- 严格 FP32 cuBLAS 同时设置 `CUBLAS_PEDANTIC_MATH` 和 `CUBLAS_COMPUTE_32F_PEDANTIC`，不把 TF32 结果混入比较。
- 每次预热和测量调用前恢复 C；CUDA events 只包围 GEMM，不计分配、初始化、H2D/D2H 拷贝、D2D 复位与结果检查。
- 每项每轮先预热，再计时；顺序以四轮为周期轮换：`K10,PAD8,K30,strict` → `PAD8,K10,strict,K30` → `K30,strict,K10,PAD8` → `strict,K30,PAD8,K10`。K10/PAD8 总是相邻且互换先后。
- 每项每轮预热 10 次、计时 30 次，共 7 轮。保留每次 event 的原始耗时；先求每轮各实现的平均毫秒，再对 7 个轮均值求中位数。波动为轮均值的最小值、最大值及 `(最大-最小)/中位数`。这与把所有 event 样本直接取中位数是两种统计，不能混用。
- 计时前和每轮最后一次输出均做全矩阵比较；其余计时样本不逐次回读。CSV 的该轮误差来自轮末完整验算，不能解读为每一个 event 都验算了一次。
- strict 参考自身做全矩阵有限值检查；另用 CPU double 验证 N=128 的全部 16384 个输出，其他尺寸固定抽查 256 个互异位置，包含四角。抽查不等于大尺寸的完整 CPU 参考。
- 最大绝对误差按原阈值 `0.01` 判断；最大相对误差的分母为 `max(abs(reference),1e-12)`，只作诊断。参考接近零时相对误差可能很大，不能替代绝对误差标准。输入还保存 FNV 指纹以核对复跑的一致性。
- GFLOP/s 使用 `2*N³/(ms*10⁶)`，并不精确计入 epilogue 的少量标量运算；所有实现使用相同计数口径。
- 原始 `k30_audit` 的 alpha 也固定为 0.5，但随机数来自 C 库 `rand`；它用于证明原独立入口可运行，其日志与新增统一输入套件的性能表分开标注。

WDDM 桌面显示、动态频率、其他程序与缓存状态可能造成波动。D2D 复位虽然不在 event 内，仍可能影响缓存状态；这里报告的是这种明确复位方式下的 GEMM 时间，不能称为冷缓存或端到端时间。Profiler 可能重放 kernel，其耗时不填入正常基准表。

### 7.4 实际构建与正确性

实际完成了 `cuda_smoke`、`sgemm`、`cuBLAS_sgemm`、`simplest_kernel`、`k30_audit` 和 `windows_benchmark` 的 Release 构建。归档的实际编译命令包含：

```text
-ccbin="...\MSVC\14.37.32822\bin\Hostx64\x64\cl.exe"
--generate-code=arch=compute_89,code=[compute_89,sm_89]
-DNDEBUG
```

最小 CUDA 测试报告计算能力 8.9，257 个整数输出全部通过。它先证明本机编译、链接、设备启动与结果回读的最短路径可用，再进入 GEMM。可选 K31/K32 target 本次未构建、未测量；保留其架构参数化不等于宣称它们已在 Windows 验证。

| 检查 | 本次覆盖 | 结果 |
|---|---|---|
| 原 `sgemm 10` | 128、256、512、1024、2048、4096，beta=0/1/3，共 18 项 | 全部 PASS |
| 原 `k30_audit v2` | 相同六尺寸与三种 beta，共 18 项 | 全部 PASS |
| 新增统一入口 | K10、PAD8、K30 V2、strict × 六尺寸 × 三种 beta，共 72 组 | 全部 PASS，失败元素与 NaN/Inf 数均为 0 |
| 正式计时轮末输出 | 三尺寸 × 四实现 × 七轮，共 84 份完整输出 | 全部 PASS，保存 2520 个 event 样本 |

统一入口 72 组验证中，三种自写实现相对 strict cuBLAS 的最大绝对误差均为 `0.001708984375`，没有放宽 `0.01` 阈值。对应最大相对误差达到约 `3.57576`，来自参考接近零的元素，按既定标准作为诊断记录。strict 自身与保存参考的误差为零，不能把这种自比当作独立正确性证明：另做的 CPU double 检查中，N=128 全量最大误差 `2.27751002e-5`，全部六尺寸/三种 beta 的抽查最大误差为 `0.000916325856`，也全部通过。

正式 `beta=3` 性能输入的误差如下；这里只列本次实际输出，不把单一尺寸的零误差推广为普遍逐位一致。

| N | K10 / PAD8 对 strict 最大绝对误差 | K30 V2 对 strict 最大绝对误差 | strict 对 CPU double 抽查最大绝对误差 |
|---:|---:|---:|---:|
| 1024 | 0.000640869141 | 0.000640869141 | 0.000076410086 |
| 2048 | 0.001708984375 | 0.001708984375 | 0.000192861580 |
| 4096 | 0.000007629395 | 0 | 0.000887715627 |

另尝试了 Compute Sanitizer 的 `memcheck`（N=256，全部四实现）。工具报告没有发现可附加进程并发生附加超时，随后取消本次工具会话；日志保存在 [`sanitizer-memcheck.log`](../../results/windows/20261020-rtx4090d-run01/sanitizer-memcheck.log)。因此本章**不声称 memcheck、racecheck 或 synccheck 通过**；上表已完成的数值验证与 Nsight 采样各自成立，但不能替代内存/同步检查。

### 7.5 七轮性能与单因素实验结果

下表来自统一入口，全部是严格 FP32，单位 TFLOP/s=1000 GFLOP/s。耗时和区间都是**每轮 30 次 event 的均值**，中位数再跨七轮计算。

| N | 实现 | 中位数 ms | 轮均值最小～最大 ms | 波动范围/中位数 | TFLOP/s |
|---:|---|---:|---:|---:|---:|
| 1024 | K10 | 0.105817 | 0.098478～0.106189 | 7.29% | 20.294 |
| 1024 | K10 PAD8 | 0.107280 | 0.100117～0.107625 | 7.00% | 20.018 |
| 1024 | K30 V2 | 0.098814 | 0.093116～0.099533 | 6.49% | 21.733 |
| 1024 | strict cuBLAS | 0.073072 | 0.070685～0.073583 | 3.97% | 29.389 |
| 2048 | K10 | 0.565802 | 0.551940～0.577095 | 4.45% | 30.364 |
| 2048 | K10 PAD8 | 0.590988 | 0.582011～0.593963 | 2.02% | 29.070 |
| 2048 | K30 V2 | 0.543881 | 0.538903～0.562246 | 4.29% | 31.588 |
| 2048 | strict cuBLAS | 0.364034 | 0.361195～0.366017 | 1.32% | 47.193 |
| 4096 | K10 | 3.124565 | 3.032988～3.215935 | 5.86% | 43.987 |
| 4096 | K10 PAD8 | 3.198771 | 3.145968～3.258983 | 3.53% | 42.966 |
| 4096 | K30 V2 | 2.968748 | 2.951509～3.007762 | 1.89% | 46.295 |
| 4096 | strict cuBLAS | 2.838149 | 2.795827～2.863247 | 2.38% | 48.426 |

4096³ 下，K10 达同条件 strict cuBLAS 的 **90.83%**；K30 V2 达 **95.60%**，比 K10 的中位吞吐高 **5.25%**。这只描述本机、该尺寸与本章测量条件，不代表理论峰值比例，也不能直接拿来证明 Windows 与 Linux 谁更快。

PAD8 没有提速。为减少慢慢升温或频率变化对比较的影响，再按相同 round 把相邻运行的 K10/PAD8 配对，定义吞吐变化为 `(K10_ms/PAD8_ms-1)*100%`：

| N | 七轮配对吞吐变化中位数 | 配对最小～最大 | PAD8 更快的轮数 |
|---:|---:|---:|---:|
| 1024 | −1.462% | −1.970%～−0.925% | 0/7 |
| 2048 | −3.557% | −6.775%～−0.845% | 0/7 |
| 4096 | −2.320% | −4.623%～−0.684% | 0/7 |

三尺寸共 21 个配对都回退，因此本次保留默认 PAD4。2048 的“逐轮配对变化中位数”与“两个实现各自中位吞吐的比值”不同，这是统计顺序不同，不是数据冲突。这里不把 21 对样本写成统计显著性检验，也不声称完全排除了 WDDM 与动态频率影响。

`ptxas` 编译报告显示 K10/PAD8 均为 184 registers/thread、零 spill，shared 分别为 16640/16896 字节；K30 V2 为 235 registers/thread、零 spill，动态 shared 由启动参数设置为 33280 字节。资源结果说明 PAD8 没有通过增加寄存器数量或产生 spill 改变实验，但不能单凭这张表把回退全部归因于 bank conflict。

本次没有锁定频率或独占显示 GPU。三个尺寸开始前采集的 telemetry 显示 SM clock 为 2520/2715/2700 MHz，温度 44/50/49°C；这些只是时点采样，不是每个 event 的频率轨迹。1024 的 K10 轮间范围约 7.3%；2048 的部分原始 event 接近各自样本中位数的 1.9 倍，这些慢样本全部保留，没有删去后再提高分数。保留原始样本、配对表和波动范围比只报告一个“最佳成绩”更有解释力。

### 7.6 Profiler 补充

本机已有 Nsight Compute 2024.1.1，已完成 K10 与 PAD8 的针对性采样，无需安装 profiler。使用 4096³、alpha=0.5、beta=3，按 `regex:sgemmWarptiling` 筛选候选核，跳过前 11 次匹配启动（预检 1 次、预热 10 次），采集随后 1 次计时启动。`--clock-control none --cache-control none` 保持时钟和缓存不受工具控制，两次采样分别产生 CSV 与 `.ncu-rep`。

| 指标 | K10 PAD4 | K10 PAD8 |
|---|---:|---:|
| Shared store bank conflicts 总数 | 16,777,216 | 50,331,648 |
| Shared load bank conflicts 总数 | 0 | 0 |
| Registers/thread | 184 | 184 |
| 静态 shared 用户字节 | 16,640 | 16,896 |
| `smsp__issue_active.avg.pct_of_peak_sustained_active` | 77.06% | 72.62% |

原始指标见 [`ncu-k10.csv`](../../results/windows/20261020-rtx4090d-run01/ncu-k10.csv) 与 [`ncu-k10-pad8.csv`](../../results/windows/20261020-rtx4090d-run01/ncu-k10-pad8.csv)，原始报告为同目录的 [`ncu-k10.ncu-rep`](../../results/windows/20261020-rtx4090d-run01/ncu-k10.ncu-rep) 与 [`ncu-k10-pad8.ncu-rep`](../../results/windows/20261020-rtx4090d-run01/ncu-k10-pad8.ncu-rep)。Bank 指标完整名称分别为 `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum` 与 `..._op_ld.sum`。

PAD8 的 shared store 冲突计数增至 3 倍，shared load 仍为零；这支持第 6 节“改变 A 转置写入布局可能恶化冲突”的预测。寄存器数相同、没有 spill，issue 指标也下降。源码中的 2-way/4-way 是单条标量写入的地址推导，不能直接当成 profiler 计数的倍数定义。

**性能结论仍以第 7.5 节未插桩的七轮数据为准。** 这两次 profiler 采样的 GPU duration 分别为 3.412512 ms、3.159904 ms，排序甚至与七轮结果相反；工具运行下主程序 event 还包含很大的采集干扰。因此单次 profiler 时间不能证明 PAD8 更快，也不能用来重算性能表。未锁定时钟、缓存未控制及单次采样限制了对 issue 数值的因果解释；本次证据支持 store 冲突恶化，尚不足以量化它独自造成了多少百分比回退。

采样时工具无法向用户 Documents 目录部署默认 section，随后自动使用安装目录内的 section 并成功生成报告。没有修改 HOME、性能计数器权限或系统配置来消除提示。

下次可用 [`Profile-Experiment.ps1`](../../scripts/windows/Profile-Experiment.ps1) 复跑这一组采样。它查找已安装的 NCU、创建新的 `results/windows/profile-时间戳` 目录、记录参数与退出码，并汇总所选指标；不覆盖本次报告。

```powershell
Set-Location 'D:\CUDA-Lab\cuda-learning\SGEMM_CUDA'
.\scripts\windows\Profile-Experiment.ps1 -BuildDir build-windows-sm89
```

## 8. 顺着代码做三个下一步练习

1. **证明输出所有权。** 写一个不调用 GPU 的索引检查器，枚举一个 block 的所有 `(thread,i,s,j)`，验证 16384 个输出恰好覆盖一次；再与 K30 的 16×8 fragment 映射对照。先不改变 kernel。
2. **只改变一个合法寄存器分块参数组合。** 保持 block/warp tile 与 FP32 语义，推导新的 `WMITER/WSUBM/WSUBN`，先检查静态约束和全部输出，再观察寄存器、spill、耗时。这一步需要把“必要的伴随参数变化”写清楚。
3. **对照 K30 V1 与 V2 的同步协议。** 它们有相同的 fragment 所有权，适合单独研究单槽与双槽；画出每个槽的“写入→发布→读取→回收”，再测屏障和访存等待指标。不要把 K10→K30 的多处变化归因于双缓冲一个因素。

## 来源与贡献边界

- SGEMM 原作者 Simon Boehm，原项目 [siboehm/SGEMM_CUDA](https://github.com/siboehm/SGEMM_CUDA)，本仓库 [MIT LICENSE](../../LICENSE) 完整保留。本次起点为 QiLHong/SGEMM_CUDA `zh-cn-learning` 提交 `77567876cecc180a039d2855d748acd9d2ba427a`。
- 教学参考为 vitamin-cuda `zh-cn-beginner` 提交 `520f97093a2abc239977be83d244b327d0e2e2ef` 的[中文入门教程](https://github.com/QiLHong/vitamin-cuda/blob/520f97093a2abc239977be83d244b327d0e2e2ef/docs/zh-CN/入门教程.md)与 [SGEMM 优化文章](https://github.com/QiLHong/vitamin-cuda/blob/520f97093a2abc239977be83d244b327d0e2e2ef/posts/zh/sgemm_beating_cublas.md)。该项目许可证署名为 `Copyright (c) 2025 wingedge`。本次借鉴讲解顺序与 shared 布局实验思路，没有移植它的 PyTorch 包装或 kernel 实现。
- 本章为新增中文学习材料，紧贴现有代码与本次 Windows 实验；既有算法编号、原作者署名、历史数据和相关失败实验保持原含义。
