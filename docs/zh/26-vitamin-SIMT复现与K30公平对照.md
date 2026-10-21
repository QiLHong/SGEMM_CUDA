# vitamin SIMT 原生复现与 K30 公平对照

> 归档可用性补记（2026-10-21，核对起点 `a54435d`）：本章引用的本地文件和目录均存在；本章、`20261021-vitamin-simt-stage3` 及部分配套脚本仍是未纳入 Git 的工作区文件，未随该提交发布。下方保留真实本地来源，历史测量值未改写。本轮 Tensor Core 多 seed 验证另见[第 33 章](33-TensorCore-BN64稳健性与尺寸适用范围.md)。

> 发布补记（2026-10-22）：本章、上述完整归档与配套脚本已纳入本次 RTX 4090 D 发布。上一条说明保留 2026-10-21 的核对状态；测量时的源码快照与原始结果没有改写。

本章沿着 vitamin-cuda 的五个 SIMT 实现阅读：先确定一个线程负责哪些输出，再跟踪 A/B 的运输、shared 布局、计算寄存器与写回，最后用相同测量条件比较 K10、K30 V2 和 cuBLAS。五个版本均为普通 FP32 SIMT；不包含 naive、TF32 或 Tensor Core 实现。

本章主表来自**Windows 原生移植与数学配置复现**，不包含Python调用开销。没有安装新的PyTorch；另已复用本机现成pytorch环境，完成第8.8节的独立Python入口验证。原作者的历史结果、当前机器的原生结果与Python入口结果分别记录，不能把它们放进一张无条件的排行榜。

## 1. 固定来源，先说明“复现”的范围

源仓库为 vitamin-cuda `zh-cn-beginner`，固定提交 `520f97093a2abc239977be83d244b327d0e2e2ef`：

- [五个原始 kernel](https://github.com/QiLHong/vitamin-cuda/blob/520f97093a2abc239977be83d244b327d0e2e2ef/kernels/sgemm/sgemm.cu)。
- [原测试程序](https://github.com/QiLHong/vitamin-cuda/blob/520f97093a2abc239977be83d244b327d0e2e2ef/kernels/sgemm/test.py)、[原 cuBLAS 包装](https://github.com/QiLHong/vitamin-cuda/blob/520f97093a2abc239977be83d244b327d0e2e2ef/kernels/sgemm/cublas.cu)。
- [原中文文章](https://github.com/QiLHong/vitamin-cuda/blob/520f97093a2abc239977be83d244b327d0e2e2ef/posts/zh/sgemm_beating_cublas.md)。其中 RTX 5060 移动版的成绩属于原作者测试，不是 RTX 4090 D 实测。

本项目的原生头文件为 [`vitamin_sgemm.cuh`](../../src/kernels/vendor/vitamin_sgemm.cuh)，放在 `vitamin_cuda` 命名空间，保留五个原始名字，不占用 SGEMM_CUDA 的算法编号。移植删除 torch 绑定及无关 include，以原 `FLOAT4` 宏和 `WARP_SIZE=32` 替代整个通用 pack 头；算法主体未改。逐字比较范围、规范化 SHA256 与例外记录在 [`provenance.json`](../../src/kernels/vendor/provenance.json)。换行规范化后的五个 device body 与固定源提交一致。

原许可证为 MIT，`Copyright (c) 2025 wingedge`，[完整副本](../../src/kernels/vendor/vitamin-LICENSE)与代码保存在 vendor 目录。上游 checkout、署名、许可证和文章历史成绩均保留。

原 `test.py` 使用 `--use_fast_math`，cuBLAS 非 TF32 路径使用 `CUBLAS_COMPUTE_32F` 与 `CUBLAS_GEMM_DEFAULT`，原生核计算 `C=AB`。原 Python 计时用 `time.perf_counter()` 包围 200 次调用，前后 synchronize、预热 10 次；本轮采用 CUDA events，不能声称复刻了 Python 调度开销。原文件默认入口实际上是 `test_tf32_4096()`；SIMT 的 `test_4096()` 被注释，本轮显式选择五个 SIMT 核。

原测试的非 TF32 正确性标准是 `allclose(atol=1e-3, rtol=1e-3)`。它与本项目“最大绝对误差不超过 0.01”不是等价标准：大数值可能更宽，接近零可能更严，报告时应分别列出。

## 2. 五个版本与安全输入范围

| 文中简称 | 原始函数名 | 相比前一项的主要源码变化 |
|---|---|---|
| Tiling | `sgemm_tiling` | 128×128 CTA，A/B shared tile，每线程 8×8 |
| AT | `sgemm_at_tiling` | A 转置存入 shared，计算阶段 A 改为 float4 读取 |
| SW | `sgemm_at_bcf_swizzling` | 转置 A 的 shared 地址增加 XOR swizzle |
| RW | `sgemm_at_bcf_swizzling_rw` | 改变线程输出所有权，重排 shared 读取与 C 写回 |
| DBF | `sgemm_at_bcf_swizzling_dbf_rw` | RW 加两个 shared 槽、运输寄存器和流水尾部 |

五项固定参数均为：

```text
BM=128, BN=128, BK=16, TM=8, TN=8
blockDim=(256,1,1)，8 个 warp
每 CTA 128×128 个输出；每线程 64 个 accumulator
```

模板声明看起来可以自由改参数，但源码写死了 `64`、`8`、`128` 等映射常量，不能只换模板参数就期待正确。原包装用 ceil grid，并不代表有边界处理：这五个 kernel 都没有针对尾块的 load/store predicate。

原算法的安全域至少要求 M、N 为正的 128 倍数，K≥16 且为 16 倍数；A 为 M×K、B 为 K×N、C 为 M×N，均为连续行主序 FP32，float4 地址满足 16 字节对齐。C 不能与输入重叠。源代码使用 `int` 地址表达式，大矩阵还必须避免乘法/索引超出有符号 32 位范围；不能把“显存装得下”当作全部条件。本轮原生包装只接收其明确支持的对齐方阵。

其中 DBF 在循环之前无条件读取第一块 A/B，**K=0 也不安全**。前四项若 K=0 会跳过计算，但本轮统一不使用这个特殊情况。1024、2048、4096 均满足上述尺寸条件。

## 3. 从 thread 找到它负责的 64 个输出

令 `t=threadIdx.x`、`w=t//32`、`l=t%32`。`//` 是整数除法，下列 m/n 为 CTA 内坐标；全局行列为 `128*blockIdx.y+m` 与 `128*blockIdx.x+n`。

### 3.1 Tiling、AT、SW：warp 负责 64×32

```text
warp_row = w//4，warp_col = w%4
m(i) = 64*(w//4) + 8*(l//4) + i       i=0..7
n(j) = 32*(w%4) + 8*(l%4) + j        j=0..7
accumulator = 8*i+j
```

8 个 warp 按 2×4 排列。一个线程的 8 行、8 列都是连续的；一个 warp 里每四个 lane 横向覆盖 32 列，八组 lane 纵向覆盖 64 行。

### 3.2 RW、DBF：warp 负责 16×128

```text
m(i) = 16*w + 8*(l//16) + i                     i=0..7
n(j) = 4*(l%16) + 64*(j//4) + j%4              j=0..7
accumulator = 8*i+j
```

现在 8 个 warp 沿 M 方向排成 8×1。lane 0～15 覆盖上面的 8 行，lane 16～31 覆盖下面的 8 行；每线程的 8 列分成两组 4 列，中间相差 64。

| warp 0 的 lane | 前三项的行 / 列 | RW、DBF 的行 / 列 |
|---:|---|---|
| 0 | 0～7 / 0～7 | 0～7 / 0～3、64～67 |
| 1 | 0～7 / 8～15 | 0～7 / 4～7、68～71 |
| 2 | 0～7 / 16～23 | 0～7 / 8～11、72～75 |
| 15 | 24～31 / 24～31 | 0～7 / 60～63、124～127 |
| 16 | 32～39 / 0～7 | 8～15 / 0～3、64～67 |
| 31 | 56～63 / 24～31 | 8～15 / 60～63、124～127 |

例如 DBF 的 warp 0、lane 2，`sum[5][4]` 对应局部 `(5,72)`。若 CTA 为 `(x,y)=(1,2)`，全局 C 为 `(261,200)`。相同 thread/register 编号在 SW 对应另一位置，不能把两个版本的 register 数组直接按坐标相同来比较。

### 3.3 CPU 完整枚举已经执行

新增 [`analyze_vitamin_mapping.py`](../../scripts/analyze_vitamin_mapping.py) 只使用 Python 标准库，在已有 `D:\Anaconda\Anaconda\python.exe`（Python 3.12.4）运行。它核对固定提交与上游工作区内容，再枚举完整 CTA。

| 每个实现的检查 | 实际数量 | 结果 |
|---|---:|---|
| C 输出记录 / 不同输出 | 16384 / 16384 | 重复、遗漏、越界均为 0 |
| A 协作加载 / 不同逻辑元素 | 2048 / 2048 | 每元素恰好一个写入者 |
| B 协作加载 / 不同逻辑元素 | 2048 / 2048 | 每元素恰好一个写入者 |
| 逐 k、逐线程计算寄存器读取 | 65536 | 全部匹配对应 producer |
| 计算侧 float4 的起点与分量顺序 | 全部枚举 | 对齐或顺序失败为 0 |

此外检查每线程 64 个 accumulator、每 warp 的完整矩形、shared 物理地址双射。DBF 在每个 A/B 数组内的两槽相隔 2048 个 float，覆盖范围互不重叠。五项均 PASS，汇总在 [`mapping/summary.json`](../../results/windows/20261021-vitamin-simt-stage3/mapping/summary.json)。

本轮也重新运行已有K10/K30映射脚本，结果在 [`mapping-k30/summary.json`](../../results/windows/20261021-vitamin-simt-stage3/mapping-k30/summary.json)：K10、K30 V1/V2各16384输出，重复/遗漏/越界为0；A/B各2048元素，旧warp0的4096行逐字段相同。K30 V1只是复用脚本额外检查的映射，不在本轮性能比较中。

[`mapping`](../../results/windows/20261021-vitamin-simt-stage3/mapping/) 内每个实现各有 `*-outputs.csv`、`*-producers.csv`、`*-consumers.csv`、`*-bank-projections.csv`、`*-c-write-sectors.csv`，另有 `warp0-lane-examples.csv`。所有 shared 地址以一个 K tile 的 slot 0 为基准；第二槽通过独立的不重叠检查验证。

这些检查证明的是抄写索引公式的覆盖与布局一致性，不运行 GPU，不证明编译后指令、浮点数值或并发同步正确。

## 4. 协作加载与 shared 布局

五项的全局加载分工相同。对每个线程 t，两次 pass `p=0,1`，每个 float4 分量 `q=0..3`：

```text
A: m = t//4 + 64*p
   k = 4*(t%4) + q
   global[(blockM+m)*K + tileK+k]

B: k = t//32 + 8*p
   n = 4*(t%32) + q
   global[(tileK+k)*N + blockN+n]
```

一个线程搬 A8+B8，共四个 float4；256 个线程恰好覆盖 A 的 128×16 与 B 的 16×128。上游 `load_b_row` 的注释写“0～8”，实际 `t//32` 范围为 0～7，不额外产生第九行。

谁加载与谁计算是两套映射。例如线程 0 加载 A 的第 0/64 行、B 的第 0/8 行，但它只拥有前节规定的 64 个输出；其他线程会复用它搬来的数据。

### 4.1 Tiling：A 保持原行列顺序

```text
A[m,k] → As[m][k]，物理 float 偏移 m*16+k
B[k,n] → Bs[k][n]，物理 float 偏移 k*128+n
```

两个 A float4 直接存到 shared。计算时 `reg_a[i]=As[c_row+i][k]` 为八个标量读取；B 则读取两组 float4。每个 k 做 8×8 次乘加，结果保留在 sum[8][8] 中。

在 32 个 4-byte bank 的地址模型中，`bank=offset%32`。固定一个 k 和 i，warp 内 A 的八个不同行地址相隔 `8*16=128` 个 float，落在同一个 bank；每四个 lane 读相同地址可以广播，但八个不同行地址仍不同。因此这个源码标量读的模型为 8-way，不能错误地把 32 个 lane 都计成 32-way。

### 4.2 AT：A 转置，方便计算阶段向量化

```text
A[m,k] → As_T[k][m]，物理偏移 k*128+m
```

global 仍以 float4 读取，运输寄存器 tmp_a 的四个分量分散写入不同 k 行。计算时同一个 k 的八个 m 连续，可以用两次 float4 读取。这里同时改变 shared 布局与计算读取宽度，因此 Tiling→AT 不只是“交换两个下标”。

代价出现在转置写入。固定一个 scalar component q，warp 内：

```text
k = 4*(l%4)+q
m = 8*w + l//4 + 64*p
bank = (k*128+m)%32 = m%32
```

只有八个 bank，各有四个不同地址，地址模型为 4-way。转置解决了一处读取问题，却引入另一处写入冲突。

### 4.3 SW：XOR 把四组 k 旋转到不同 bank

原宏为 `SWIZZLE_A(x,y) = y ^ ((x>>2)<<3)`。这里 x 是 k、y 是 m：

```text
S(k,m) = m XOR (8*(k//4))
A[m,k] → As_T[k][S(k,m)]
physical = k*128 + S(k,m)
bank = S(k,m)%32
```

一个 warp 的 `k//4=l%4`。`l//4` 提供 bank 的低三位，`l%4` 经左移三位提供下一组位，原本重叠的四组 k 被分到不同 bank。固定 component 的 32 个线程现在对应 32 个 bank，每 bank 一个地址；完整枚举得到 4→1。

例如 warp 0、pass 0、component 0，lane 0/1/2/3 的逻辑 m 都是 0，k 分别为 0/4/8/12。AT 的 bank 都是 0；SW 变成 0/8/16/24。

XOR 是自身的逆：对每个固定 k，它只是把 0～127 的 m 重新排列，并没有填充或丢弃数据。写和读使用同一个 S，就能读回原始 A[m,k]。这个 mask 只改变 m 的第 3、4 位，保留最低三位，因此四个连续分量不会被打散，源 float4 的对齐与分量顺序得以保留。

**bank CSV 的边界：**标量操作按完整 32 lane 统计；float4 按连续八个 lane、最多 128 B 请求投影，同时对重复读取同一地址按广播处理。这个模型帮助核对地址，不是对真实 LDS/STS 指令宽度、硬件请求拆分或 NCU conflict counter 的测量。不能仅凭函数名含 `bcf` 就宣称运行时所有 shared 冲突为零。

## 5. RW 为什么不只是“最后一行 store 改了”

SW→RW 没有改变 A/B 的全局协作加载，也没有改变 A 的 XOR 公式，但改变了第 3 节的输出所有权。为了计算新坐标，A/B 的 shared 读取、每个 lane 的广播关系和 C 写回地址随之变化。

固定一个 `sum` 行 i 和一次 float4 写回：

```text
SW：每四个 lane 写同一行；各写4个float，下一lane起点跳8个float。
    单条指令覆盖半个32B sector；另一半在下一次float4指令中写。

RW：每十六个 lane 写同一行；各写4个float，起点连续前进4。
    两个半warp分别形成连续256B，合计512B。
```

对齐条件成立时，脚本按每条源码 float4 指令枚举 32 B sector：SW 请求 512 B、涉及 32 个 sector；RW 请求同样 512 B、涉及 16 个 sector。这是每条指令的地址分布，**不是 DRAM 流量已经减半**；两条指令之间的缓存/合并及实际 STG 宽度还要观察机器码与 profiler。

RW 的每个 warp 从 64×32 变为 16×128，A 的同地址广播更多，B 的地址覆盖更宽。若 RW 更快，可以说这次所有权重排的整体结果更好；没有独立消融就不能把所有收益精确分配给 C 写回。

## 6. DBF 的两槽生命周期与 K30 V2 差别

RW 每个 K tile 都是：

```text
global load → shared store → 发布屏障 → compute → 回收屏障
```

令 `T=K/16`，共有 2T 次 block 屏障。DBF 先将 tile 0 写 slot 0，再做一次屏障；主循环只有 T−1 轮，最后一个 tile 单独计算：

```text
初始化：load tile0 → store slot0 → Binit

t=0：load tile1到寄存器 → compute slot0(tile0) → store slot1(tile1) → B0 → 交换
t=1：load tile2到寄存器 → compute slot1(tile1) → store slot0(tile2) → B1 → 交换
...
尾部：compute 最后已发布的slot → write C
```

轮末屏障同时发布 future 与回收 current。写回旧槽前，上一轮末尾已经保证所有 warp 读完旧内容；本轮读写使用不同槽。最后一个 tile 没有后续复用，不需要再设置回收屏障。这些关系约束四类操作：

```text
global A/B → 运输寄存器 tmp_a/tmp_b → shared slot
           → reg_a[8]/reg_b[8] → sum[8][8] → global C
```

DBF 源代码总屏障数为 `1+(T−1)=T`。K=4096 时是 256 次，RW 是 512 次；第 25 章的 K30 V2 则为 `T+1=257`，因为它把最后一个 tile 放在仍有轮末屏障的统一循环中。这一处计数区别不证明可观的性能差距。

DBF 的 shared 从 16384 增至 32768 字节。源码宣称预取需要额外运输寄存器，注释还提到“保持少于 128 registers”；那是原作者的编译经验，不是本次 sm_89 的资源报告。编译器可以重排、下沉 global load，源码中的“预取后立即计算”不自动证明实际重叠。第 25 章已经展示了 K30 源码/PTX/SASS 的这种差异。

## 7. 与 K10/K30 对照时控制了什么

| 项目 | vitamin 五项 | K10 默认 | K30 V2 Small |
|---|---|---|---|
| CTA 输出 / BK | 128×128 / 16 | 相同 | 相同 |
| threads / warps | 256 / 8 | 128 / 4 | 128 / 4 |
| 每线程 accumulator | 64 | 128 | 128 |
| 线程输出 | 8×8，RW 后列拆两组 | 8×16，列拆四组 | 16×8，行列分组 |
| warp 输出 | 64×32 或 16×128 | 64×64 | 64×64 |
| A shared | 原序，或转置+XOR | 转置，stride132 | 转置，stride132 |
| 用户 shared 字节 | 16384，DBF 32768 | 16640 | 33280 |
| 原 epilogue | C=AB | alpha*AB+beta*C | alpha*AB+beta*C |

本轮统一 `alpha=1,beta=0`，以保留 vitamin 的原始 device body。K10/K30 在该语义下仍走自身代码路径。不能直接把第 25 章 alpha=0.5、beta=3 的吞吐拿来作本轮分母。

这些 kernel 之间同时存在线程数、寄存器分块、warp 所有权、shared 布局、同步与编译调度差异。**公平测量意味着输入、数学语义和测量方法一致；不意味着不同 kernel 之间只变了一个因素。** 若希望知道哪一个机制导致差距，还需在同一实现上逐项消融。

本轮将两种构建分别保存：

| 配置 | 自写 kernel 编译 | cuBLAS 对照 | 用途 |
|---|---|---|---|
| A：原数学配置 | `--use_fast_math` | `CUBLAS_COMPUTE_32F`、DEFAULT 算法，独立 handle | 接近原文章的数学/编译选择 |
| B：严格 FP32 | 不启用 `--use_fast_math` | PEDANTIC math + `CUBLAS_COMPUTE_32F_PEDANTIC` | 与 K10/K30 严格条件比较 |

`--use_fast_math` 包含 FTZ、精度较低的除法/平方根选项和 FMA 收缩相关设置；本 kernel 虽主要是乘加，也不能据此宣布两种编译数学行为无条件相同。B 不等于禁止 FMA，普通 FP32 FMA 仍允许。对应选项见 [CUDA 12.4.1 nvcc 文档](https://docs.nvidia.com/cuda/archive/12.4.1/cuda-compiler-driver-nvcc/index.html#use-fast-math-use-fast-math) 与 [cuBLAS compute type 文档](https://docs.nvidia.com/cuda/archive/12.4.1/cublas/index.html#cublascomputetype-t)。A 与 B 的结果分表，不将 A 的任何提升包装成严格 FP32 提速。

运行脚本对 A、B 都在本进程设置 `NVIDIA_TF32_OVERRIDE=0`，结束后恢复原值。A 保留普通 COMPUTE_32F API 选择，但没有复刻上游 PyTorch 当前 handle 的全部外部状态；本轮使用独立新建 handle，并打印实际 math/atomics 值。输入也采用本项目固定 xorshift，而非原 `torch.randn`。这些都是原生复现边界。

## 8. 本机实测与编译证据

### 8.1 环境、计时与校验范围

本轮在 2026-10-21 的 Windows 原生 RTX 4090 D（114 SM、WDDM、驱动610.60）完成，使用 CUDA 12.4.131、MSVC 19.37.32826.1、SDK 10.0.26100.0、CMake 3.31.6-msvc6 和 Ninja 1.12.1。A/B 分别使用 `build-windows-vitamin-sm89-A/B` 的独立 Release 可执行文件，实际编译命令均为 sm_89；每个配置保存编译命令、构建日志和二进制 SHA256。

[`run01/environment.json`](../../results/windows/20261021-vitamin-simt-stage3/run01/environment.json) 与 [`run01/run.log`](../../results/windows/20261021-vitamin-simt-stage3/run01/run.log) 保存真实命令与运行条件。统一 `alpha=1,beta=0,seed=20261020`；所有实现和两配置的 A/B/C0 字节指纹一致。每次 launch 前恢复 C0，D2D 复制与 start event 在同一流内排序，复制本身不计时。这个协议仍会改变缓存与内存流量，结果应连同该条件理解。

正式测试为三尺寸×两配置×八实现，每项先预热10次，每轮30次 CUDA event，共7轮。分配、H2D/D2H、C0复位、预热、校验均在 event 区间外。每轮只回读最后一次输出，不将30个 event 宣称为30份独立数值检查。轮次顺序按七个组轮转，其中 RW/DBF 固定相邻且交替先后；一个组含两项，所以八个位置在七轮中并非完全平衡。A/B 在1024、4096按 A→B 执行，在2048按 B→A 执行，**两配置仍是不同时段，不能作为同时配对的编译选项消融**。

最小 CUDA 测试通过。128、256、512 的小规模检查和1024、2048、4096的正式运行共保存492条正确性记录：12条独立 strict 参考、96条实现预检、384条轮末输出（其中正式测量336条，小检查48条）。正式测量保存 **10080个 event、336轮**。以下两套规则均全部通过，没有放宽阈值：

- 本项目绝对误差≤0.01；NaN/Inf与失败元素均为0。
- 原数学 allclose 规则 `|reference-candidate|≤0.001+0.001*|candidate|`，单独记录，全部通过。参考换成独立 strict cuBLAS、host double 评估，未声称与 PyTorch 位级一致。

全部检查最大绝对误差 `0.00341796875`，最大相对误差 `2.125`；近零参考导致的相对误差不作为替代通过阈值。每个实现还对照 CPU double：128时检查全部16384元素，其余尺寸固定256点，最大 CPU 绝对误差 `0.0018326517129025888`。大尺寸仍是CPU抽查，不是完整CPU参考。两配置各拒绝0、16、127、129、255、4100六种尺寸，确认在创建输出或启动CUDA前拒绝。

汇总审计见 [`analysis/audit.json`](../../results/windows/20261021-vitamin-simt-stage3/run01/analysis/audit.json)、[`correctness-all.csv`](../../results/windows/20261021-vitamin-simt-stage3/run01/analysis/correctness-all.csv)。本轮没有成功的 Compute Sanitizer 检查，不能把数值或 profiler 通过写成 memcheck/racecheck/synccheck 通过。

另一个不导入汇总脚本的CPU审计已独立复算全部10080个正式event、48个小检查event、48条汇总和72组配对，全部一致，见 [`verification/independent-event-audit.json`](../../results/windows/20261021-vitamin-simt-stage3/verification/independent-event-audit.json)。它验证CSV记录与统计一致，不重新执行矩阵计算。

### 8.2 正常计时：两种配置分开报告

先对每轮30次 event 求平均，再取7个轮均值的中位数，计算 `TFLOP/s=2*N³/(ms*10⁹)`。以下单元格为 **TFLOP/s（轮均值极差÷中位数）**，括号不是±误差、不是置信区间。全部慢样本保留；各轮、全部毫秒值与极值在 [`rounds.csv`](../../results/windows/20261021-vitamin-simt-stage3/run01/analysis/rounds.csv)、[`summary.csv`](../../results/windows/20261021-vitamin-simt-stage3/run01/analysis/summary.csv)。

配置A：fast_math 与普通 COMPUTE_32F cuBLAS。

| 实现 | 1024 | 2048 | 4096 |
|---|---:|---:|---:|
| Tiling | 14.941（7.12%） | 25.103（12.05%） | 33.618（3.83%） |
| AT | 18.735（23.68%） | 32.732（15.88%） | 40.295（5.62%） |
| SW | 21.083（28.32%） | 34.740（7.77%） | 44.497（10.13%） |
| RW | 21.867（7.19%） | 34.100（9.82%） | 42.548（4.06%） |
| DBF | 26.392（23.36%） | 38.661（13.26%） | 42.199（7.25%） |
| K10 | 19.992（26.31%） | 30.634（9.99%） | 43.235（4.64%） |
| K30 V2 | 22.093（7.13%） | 32.326（12.28%） | 44.103（4.05%） |
| cuBLAS | 29.789（7.58%） | 46.864（5.46%） | 46.849（5.68%） |

配置B：不启用 fast_math，严格 FP32 cuBLAS。

| 实现 | 1024 | 2048 | 4096 |
|---|---:|---:|---:|
| Tiling | 15.968（22.03%） | 24.699（9.78%） | 30.619（12.87%） |
| AT | 19.960（19.66%） | 33.065（8.61%） | 39.991（11.64%） |
| SW | 22.438（3.64%） | 35.874（8.66%） | 42.834（15.57%） |
| RW | 23.455（1.83%） | 36.377（11.69%） | 42.823（10.17%） |
| DBF | 28.379（25.62%） | 39.430（17.89%） | 42.156（7.59%） |
| K10 | 21.733（1.21%） | 30.698（8.79%） | 42.075（10.15%） |
| K30 V2 | 23.563（50.60%） | 32.728（10.56%） | 43.597（9.13%） |
| strict cuBLAS | 31.115（68.78%） | 47.131（2.38%） | 45.205（17.00%） |

本次两个配置、三个尺寸的所有自写实现，中位吞吐均低于对应 cuBLAS；没有复现原文章在另一款GPU上的“超过cuBLAS”。B的1024和2048中DBF领先K30 V2；到4096则K30 V2领先DBF。严格B、4096的RW/DBF/K30/strict中位轮时间依次为3.209455/3.260211/3.152478/3.040324 ms，不能把增加双缓冲描述成全面提速。

### 8.3 配对结果：DBF 在4096没有获胜

定义同一轮变化 `(baseline_ms/candidate_ms-1)*100%`。这与“两个中位时间之比”是不同统计量，保留各自名称。

| 配置、N | DBF相对RW：配对中位数 | 配对最小～最大 | DBF更快 | DBF相对K30 V2：配对中位数 |
|---|---:|---:|---:|---:|
| A、1024 | +21.095% | −3.176%～+21.466% | 6/7 | +19.920% |
| A、2048 | +8.120% | +4.215%～+17.457% | 7/7 | +18.414% |
| A、4096 | −0.123% | −3.143%～+4.529% | 3/7 | −4.801% |
| B、1024 | +21.054% | −3.555%～+22.699% | 6/7 | +20.260% |
| B、2048 | +6.535% | −1.484%～+19.246% | 6/7 | +18.233% |
| B、4096 | **−1.513%** | −4.002%～+5.436% | **1/7** | −3.716% |

RW/DBF每对均相邻；K30/DBF只同轮、不相邻，控制强度不同。完整配对见 [`paired-rounds.csv`](../../results/windows/20261021-vitamin-simt-stage3/run01/analysis/paired-rounds.csv)、[`paired-summary.csv`](../../results/windows/20261021-vitamin-simt-stage3/run01/analysis/paired-summary.csv)。

沿优化链看严格B：Tiling→AT的三尺寸配对中位提升为25.401%、33.649%、23.883%，各7/7轮更快；AT→SW为12.235%、8.340%、6.326%，也各7/7。SW→RW仅为4.278%、1.069%、1.414%，更快轮数7/7、4/7、5/7；A的4096中SW→RW反而为−4.688%，只有1/7轮更快。因此读写重排的整体效果依赖尺寸和运行条件，不能只用sector模型预言性能。

上述都是一次WDDM会话中的七轮，极值区间不是跨会话置信区间。B的1024甚至有个别配对显示DBF比cuBLAS快50%以上，但cuBLAS自身存在慢样本，整体中位吞吐仍领先；不能挑一个有利轮次宣称超越库实现。

### 8.4 波动与遥测覆盖

正式运行没有锁定频率，桌面WDDM调度和动态时钟可能干扰短kernel；目前没有证据将某个慢样本唯一归因于它们。B的1024，K30最大单次event为1.289152 ms、其中位event约0.091136 ms（约14.15倍）；cuBLAS最大0.704512 ms、其中位约0.066560 ms（约10.58倍）。这些慢值全部保留，解释了表中50.60%和68.78%的轮均值极差。

200 ms采样请求实际得到214点，01:15:13.381～01:16:00.826；观察到温度44～67°C、SM时钟2235～2715 MHz、显存时钟10251～10501 MHz。336个轮时间窗只有132个包含采样点，204个未知：

| 配置 | 1024窗口有采样 | 2048窗口有采样 | 4096窗口有采样 |
|---|---:|---:|---:|
| A | 2/56 | 10/56 | 52/56 |
| B | 2/56 | 11/56 | 55/56 |

窗口从预热前至完整校验后，不是kernel独占时间。没有插值或以“最近一次”代替缺失值，也不能据此说两配置或每个实现都在相同频率运行。原始 [`telemetry.csv`](../../results/windows/20261021-vitamin-simt-stage3/run01/telemetry.csv) 与 [`round-telemetry.csv`](../../results/windows/20261021-vitamin-simt-stage3/run01/analysis/round-telemetry.csv) 保留这些局限。

### 8.5 实际 sm_89 资源与加载位置

A/B二进制均已导出PTX、SASS和资源，分别在 [`static/A`](../../results/windows/20261021-vitamin-simt-stage3/static/A/) 与 [`static/B`](../../results/windows/20261021-vitamin-simt-stage3/static/B/)。以下为B；[全函数比较](../../results/windows/20261021-vitamin-simt-stage3/static/config-comparison.json)显示七个自写核忽略opcode的`.FTZ`修饰后，指令顺序、predicate和操作数逐条相同。A使用`FFMA.FTZ`，数学语义并非无条件相同；本轮输入不含非正规数，且A/B不是同时段实验，不能把表中差异宣称为fast_math收益或“严格编译更快”。

| 实现 | registers/thread | 用户shared字节 | 每主循环FFMA | LDG/LDS/STS | BAR |
|---|---:|---:|---:|---:|---:|
| Tiling | 126 | 16384 | 1024 | 4/64/4 | 2 |
| AT | 123 | 16384 | 1024 | 4/64/10 | 2 |
| SW | 127 | 16384 | 1024 | 4/64/10 | 2 |
| RW | 127 | 16384 | 1024 | 4/64/10 | 2 |
| DBF | **143** | 32768 | 1024 | 4/64/10 | 1 |
| K10 | 184 | 16640 | 2048 | 8/96/20 | 2 |
| K30 V2 | 235 | 33280（动态） | 2048 | 8/96/20 | 1 |

本次DBF实际143个寄存器，不能沿用原注释的“低于128”。全部自写核资源报告STACK/LOCAL为0；选中主循环无LDL/STL/LDGSTS/DEPBAR，也无Tensor指令。静态条数不是执行周期；例如vitamin每线程64个输出，K30为128个，不能直接用1024与2048的差说计算工作减半。

DBF源码把四个float4全局读取都放在计算前；本次PTX两条在前、两条在后；最终SASS四条LDG都位于计算末段，其后各剩192～210条FFMA。K30 V2的LDG之后只剩2条FFMA。它证明本次机器指令位置不同，**不能仅凭剩余指令条数证明隐藏了多少加载时延**。

对应 [`sass-summary.csv`](../../results/windows/20261021-vitamin-simt-stage3/static/B/analysis/sass-summary.csv)、[`ptx-summary.csv`](../../results/windows/20261021-vitamin-simt-stage3/static/B/analysis/ptx-summary.csv) 与完整order CSV可逐条回查。分析器按向后跳转区间寻找静态FMA最多的循环，属于启发式而非完整CFG；K10的PTX循环有歧义警告，本章不拿那个128-FMA子循环当作整个K tile，采用已明确识别的SASS主循环。

### 8.6 Profiler证据

已有NCU 2024.1.1完成配置B的2048/4096×五vitamin实现与K30 V2×两次，共24份采样，见 [`profile01/ncu-summary.csv`](../../results/windows/20261021-vitamin-simt-stage3/profile01/ncu-summary.csv) 和同目录原始CSV/`.ncu-rep`。另外独立采集strict cuBLAS两个尺寸各一次，共2份，见 [`profile-cublas-B/ncu-summary.csv`](../../results/windows/20261021-vitamin-simt-stage3/profile-cublas-B/ncu-summary.csv)，不将它们说成各两次。

采样使用kernel replay、clock-control none、cache-control none，选择预检与预热之后的一次调用。下面的两次数值范围是采样范围，不是置信区间；缓存、时钟与重放会影响指标，profiler duration及其内部event均不进入第8.2节的性能表。

**转置与swizzle的作用得到硬件计数支持。** 4096的Tiling shared-load conflict为402653184，AT降为0；但AT的转置shared-store conflict为50331648，SW/RW/DBF均为0。按每1024条thread FFMA归一化，shared-load wavefront从Tiling约12降为AT/SW/RW/DBF约6，K30 V2约3。这是动态服务量，不能直接等同于第4节某一条源码访问的bank模型倍数。

SW与RW的shared读写服务量在同一水平、上述冲突计数都是0。五个vitamin核都已经编成16条`STG.E.128`写回；因此RW不是本次才获得128位写回，也没有通过减少静态STG条数获得优势。本轮没有采global-store sectors/request，CPU算出的32→16 sector仍是地址模型，不能宣称NCU已经证明写回合并导致提速。

**DBF的部分stall指标降低，同时驻留资源减少。** 4096两次采样：

| 指标 | RW | DBF |
|---|---:|---:|
| registers/thread | 127 | 143 |
| register限制的CTA/SM | 2 | 1 |
| NCU记录的shared限制CTA/SM | 3 | 1 |
| 实际occupancy | 31.80% | 16.66% |
| eligible warp/SMSP active cycle | 2.27～2.28 | 1.27～1.28 |
| long-scoreboard warp/SMSP active cycle | 0.38 | 0.18 |
| short-scoreboard warp/SMSP active cycle | 0.15 | 0.18 |
| barrier warp/SMSP active cycle | 0.35 | 0.08～0.09 |
| issue active | 73.29%～73.44% | 71.53%～72.00% |
| FP32 FMA pipe active | 67.57%～67.58% | 65.48%～65.50% |

表中warp指标为每SMSP active cycle的平均warp数，不是等待时间百分比，不能相加作耗时分解。DBF的resident warp也减少，因而stall warp数下降本身不能证明每个warp等待改善或隐藏了相同比例的时延。[Ada的寄存器容量为每SM 64K个32位寄存器](https://docs.nvidia.com/cuda/ada-tuning-guide/index.html)；DBF每CTA有256线程，`143*256*2=73216>65536`，寄存器需求本身已排除两个CTA同时驻留。

但不能说“shared翻到32KB就一定只容纳一个CTA”：本轮未采实际shared carveout，Ada的100KB架构上限也不等于该次运行的实际划分。表中的shared-limit=1是NCU观测，不能据此独立认定它是首要原因。用户shared为16384/32768字节，NCU total还各含1024字节预留；两者定义要区分。

2048的权衡不同：RW的FMA pipe active为65.23%～66.05%，DBF为66.81%～67.66%；issue范围分别72.46%～72.47%与72.39%～76.57%。这与2048的正常计时DBF整体更快、4096略慢的方向相容，但不是单因素因果证明。4096的DBF还观察到约634.6MB DRAM读取、L2命中85.47%，RW约380.8～382.9MB、91.20%～91.25%；未控制缓存且使用replay，不能将这组数直接作为正常运行的流量定论。

**为什么K30仍值得比较。** K30 V2在4096并非无冲突：shared-load conflict为23549～24759，store conflict为17955383～18194835；但其归一化shared-load wavefront约为vitamin优化版本的一半，short-scoreboard为0.05、issue为78.03%～79.00%、FMA pipe active为73.05%～73.37%。它说明“零bank冲突”不足以保证总吞吐最好；每线程/warp复用方式和服务量同样需要看。

附加strict cuBLAS在两个尺寸选择同一 `cutlass_80_simt_sgemm_256x128_8x4_nn_align1`，202个寄存器、49152字节用户shared（total50176）、local流量为0；4096单次采样issue为84.27%、FMA pipe active为75.88%、long/short-scoreboard为0.04/0.03。单次库采样只提供对照，不用于构造置信区间或代替正常计时。

完整指标范围见 [`static/profile-analysis/profile-ranges.csv`](../../results/windows/20261021-vitamin-simt-stage3/static/profile-analysis/profile-ranges.csv) 与 [`static/cublas-profile-analysis/profile-ranges.csv`](../../results/windows/20261021-vitamin-simt-stage3/static/cublas-profile-analysis/profile-ranges.csv)。这些证据显示部分stall warp指标下降与可驻留warp减少同时发生，最终收益还取决于发射、访存和计算；不能给每项机制分配确切收益百分比。

### 8.7 下一轮只提出一个受控假设

下一候选实验：**DBF在4096是否主要受143寄存器带来的单CTA驻留限制？** 保留kernel源码、tile、映射、两个shared槽与数学配置，只对DBF的独立编译target/TU设置 `--maxrregcount=128`，与默认143寄存器版本配对比较，不把该选项施加到K10/K30或其他vitamin核。

必须测实际寄存器、spill/local流量、shared carveout、可驻留CTA和同协议多轮event。128只是编译上限目标，不保证实际变成两个CTA；降低上限也可能增加spill或破坏有利调度。如果未提高驻留，或spill抵消收益，就否定这个优化。本轮没有实施或测试此改动。

### 8.8 独立Python入口：复用原函数，保留Windows适配边界

完整环境盘点发现，`D:\Anaconda\Anaconda\python.exe` 的base环境没有torch，但 `D:\Anaconda\Anaconda\envs\pytorch\python.exe` 已有可用的Python3.12.7、PyTorch2.5.1/CUDA12.4和torchvision0.20.1；没有安装或升级这些包。原生 `run01/environment.json` 初查时的“torch absent”不能解释为整台电脑没有PyTorch。盘点与实际运行信息见 [`preflight/pytorch-env-runtime-audit.json`](../../results/windows/20261021-vitamin-simt-stage3/preflight/pytorch-env-runtime-audit.json)、[`python04/python-environment.json`](../../results/windows/20261021-vitamin-simt-stage3/python04/python-environment.json)。

[`vitamin_python_probe.py`](../../scripts/vitamin_python_probe.py) 以非`__main__`方式导入原`test.py`，复用**未修改的`benchmark`和`diff_check`函数体**，显式调用torch.matmul、原普通cuBLAS包装及五个SIMT实现。原默认TF32主入口没有执行，TF32 kernels没有运行。原`sgemm.cu/cublas.cu/sgemm_tf32.cu`三份源文件均保留在扩展构建中，所以上述结论是“未运行TF32”，不是“构建中没有TF32源码”。

原始`load`参数先实际尝试并记录失败，然后在调用包装层进行Windows适配：主机`-O3/-std=c++17`换成`/O2 /std:c++17`，显式链接已有`cublas.lib`，把Windows会当作一个未知选项的`"-Xptxas -v"`改为等价`-Xptxas=-v`。CUDA原优化参数与`--use_fast_math`保留，未改上游device函数或测试函数。UTF-8、sm_89、扩展缓存路径等只在本次进程设置并恢复；TF32关闭，matmul precision为highest。实际load参数及改动清单见 [`python04/build-attempts.json`](../../results/windows/20261021-vitamin-simt-stage3/python04/build-attempts.json)。

失败过程保留在 [`python01`](../../results/windows/20261021-vitamin-simt-stage3/python01/)（GBK读取UTF-8源码失败）、[`python02`](../../results/windows/20261021-vitamin-simt-stage3/python02/)（nvcc选项分词失败）、[`python03`](../../results/windows/20261021-vitamin-simt-stage3/python03/)（扩展编译成功，随后包装器解析原打印格式失败）。成功的 [`python04`](../../results/windows/20261021-vitamin-simt-stage3/python04/) 保存21条结果、实际加载`.pyd`、`build.ninja`与SHA256；首次完整成功的编译日志在python03。没有删除失败记录，也不把前三次称为完整通过。

成功测试为三尺寸×七项，共21行。每尺寸重设`torch.manual_seed(42)`，使用CPU `torch.randn`后传到GPU、C初始为0；每项10次预热、200次调用，**只有一个wallclock批次**。计时保留原`time.perf_counter`包围CPU派发与最终synchronize，未保存逐次CUDA events，也没有本章原生协议的逐次C0复制。输入、调用开销、计时边界、批次数均与原生主表不同，不能合并或计算跨表提升。

原打印的TFLOP/s如下，只有两位小数，**不是多轮中位数**：

| Python调用项 | 1024 | 2048 | 4096 |
|---|---:|---:|---:|
| torch.matmul参考 | 27.19 | 42.77 | 46.41 |
| 原普通cuBLAS包装 | 30.29 | 42.01 | 46.12 |
| Tiling | 13.53 | 25.09 | 33.42 |
| AT | 17.65 | 31.68 | 40.30 |
| SW | 21.07 | 34.11 | 41.85 |
| RW | 21.63 | 34.43 | 41.37 |
| DBF | 24.92 | 35.72 | 41.52 |

原始均值与所有校验字段在 [`python04/wallclock-results.csv`](../../results/windows/20261021-vitamin-simt-stage3/python04/wallclock-results.csv)。4096的RW/DBF wallclock均值为3.321946/3.310029 ms，这一个批次的微小正差不能推翻严格B原生多轮的反例，更不能证明跨会话稳定收益。

18个候选的原`diff_check(reference,candidate)`均通过，另行记录的abs≤0.01失败数与NaN/Inf均为0；最大绝对误差为0.00048828125。4096这批输入的候选与torch输出恰好相等，不表示数学上精确或任意输入都相等。该Python路线使用torch参考，没有另外执行CPU-double抽查；第8.1节CPU检查只属于原生实验。

## 9. 复跑入口

已执行的 CPU 映射复跑使用新目录，脚本拒绝覆盖现有产物：

```powershell
Set-Location 'D:\CUDA-Lab\cuda-learning\SGEMM_CUDA'
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
& 'D:\Anaconda\Anaconda\python.exe' .\scripts\analyze_vitamin_mapping.py `
  --output-dir ".\results\windows\vitamin-mapping-$stamp"
```

以下构建、运行和汇总入口已实际执行。使用已有PowerShell7，默认A/B均构建，运行目录和汇总目录均拒绝覆盖：

```powershell
Set-Location 'D:\CUDA-Lab\cuda-learning\SGEMM_CUDA'
.\scripts\windows\Build-Vitamin.ps1

$result = ".\results\windows\vitamin-$(Get-Date -Format 'yyyyMMdd-HHmmss')"
.\scripts\windows\Run-Vitamin.ps1 -OutputDir $result -Rounds 7 -Repeats 30 -Warmup 10
.\scripts\windows\Summarize-Vitamin.ps1 -ResultDir $result
```

[`Build-Vitamin.ps1`](../../scripts/windows/Build-Vitamin.ps1) 初始化当前构建进程工具链，验证移植代码与sm_89/fast_math编译参数；[`Run-Vitamin.ps1`](../../scripts/windows/Run-Vitamin.ps1) 验证binary哈希、执行数值检查/正常计时并保存遥测；[`Summarize-Vitamin.ps1`](../../scripts/windows/Summarize-Vitamin.ps1) 调用CPU统计审计，保留所有原始event。

可选profiler与正常计时分开执行；默认严格B、2048/4096、五vitamin实现与K30 V2，各两次：

```powershell
.\scripts\windows\Profile-Vitamin.ps1 `
  -OutputDir ".\results\windows\vitamin-profile-$(Get-Date -Format 'yyyyMMdd-HHmmss')"
```

补充strict cuBLAS各一次可追加独立调用 `-Modes cublas -ProfileRepeats 1`，同样选择新的输出目录。该脚本为 [`Profile-Vitamin.ps1`](../../scripts/windows/Profile-Vitamin.ps1)。只复算已保存数据可指定新的汇总输出，不必重跑GPU：

```powershell
.\scripts\windows\Summarize-Vitamin.ps1 `
  -ResultDir '.\results\windows\20261021-vitamin-simt-stage3\run01' `
  -OutputDir ".\results\windows\vitamin-analysis-$(Get-Date -Format 'yyyyMMdd-HHmmss')"

$stage3 = '.\results\windows\20261021-vitamin-simt-stage3'
& 'D:\Anaconda\Anaconda\python.exe' .\scripts\verify_vitamin_results.py `
  --result-dir "$stage3\run01" `
  --output-path ".\results\windows\vitamin-audit-$(Get-Date -Format 'yyyyMMdd-HHmmss').json"

& 'D:\Anaconda\Anaconda\python.exe' .\scripts\analyze_vitamin_evidence.py `
  --sass "$stage3\static\B\binary.sass" --ptx "$stage3\static\B\binary.ptx" `
  --label B --profile "$stage3\profile01\ncu-summary.csv" `
  --compare-sass "$stage3\static\A\binary.sass" `
  --output-dir ".\results\windows\vitamin-static-$(Get-Date -Format 'yyyyMMdd-HHmmss')"
```

两个Python命令分别使用 [`verify_vitamin_results.py`](../../scripts/verify_vitamin_results.py) 独立复算计时审计，以及 [`analyze_vitamin_evidence.py`](../../scripts/analyze_vitamin_evidence.py) 重算已有静态/profiler证据；不重新采集，也不暗示这些旧dump属于未来重建的二进制。

复跑第8.8节的可选Python wallclock路线，使用已有pytorch环境与新输出目录：

```powershell
.\scripts\windows\Run-VitaminPython.ps1 `
  -OutputDir ".\results\windows\vitamin-python-$(Get-Date -Format 'yyyyMMdd-HHmmss')"
```

[`Run-VitaminPython.ps1`](../../scripts/windows/Run-VitaminPython.ps1) 默认使用 `D:\Anaconda\Anaconda\envs\pytorch\python.exe`，初始化本次进程MSVC/CUDA并复用项目内扩展缓存；它不会调用默认TF32主入口，不安装依赖。若失败，日志与已有部分结果保留。

原生入口 [`windows_vitamin_benchmark.cu`](../../scripts/windows_vitamin_benchmark.cu) 的准确CLI为 `n rounds repeats warmup new.csv [all|原实现名|k10|k30_v2|cublas]`，固定alpha1/beta0；小尺寸可用1轮/1次检查，正式比较使用上述脚本的7轮/30次。它独立于前两阶段程序，前两阶段历史文件不作为可覆盖的输出目录。

## 10. 沿着代码做五道练习

**1. 为什么 256 个线程，每线程 8×8，能够覆盖 128×128？**

答案：`256*64=16384=128*128` 只证明数量相同；第 3 节的 m/n 映射和完整枚举另外证明了坐标不重叠、不遗漏、无越界。单靠乘法不能证明覆盖正确。

**2. SW 的 warp 0、lane 1 写 A 的第一个分量时，它写哪个 bank？**

答案：逻辑 m=0、k=4，`S=0 XOR 8=8`，物理偏移 `4*128+8`，bank8。AT 的对应偏移为 `4*128`，bank0；lane0 在两版本都是 bank0。

**3. RW 的 lane 2、`sum[5][4]` 是哪一个输出？**

答案：warp0 时 m=5，n=`4*2+64=72`。CTA `(1,2)` 给全局 `(261,200)`。列寄存器4的含义是第二组的第一个元素，而非 `c_col_base+4`。

**4. DBF 为什么可以在当前 compute 后写另一个槽，而不在写入前再加屏障？**

答案：上轮末尾已经回收那个槽，本轮所有线程读 current、写 future，两个物理槽不同；本轮末尾屏障再发布 future/回收 current。若改为同槽或去掉上一轮屏障，这个推理就失效。

**5. 如果 RW 的 sector 模型减半而时间没有变快，应如何解释？**

答案：模型只针对 C 的单条源码写指令，真实访存还有缓存、机器指令拆分、A/B 读取、计算、同步等成本；RW 同时改变共享读取与所有权。先保存未提速结果，再用实际 SASS/NCU 定位，不能改阈值或把模型数字当成吞吐结论。

## 来源与边界

本章的映射与流水线直接来自固定 vitamin 提交，CPU 分析脚本记录源文件及自身 SHA256；原作者版权与完整 MIT 许可证保留在移植目录。K10 的 Simon Boehm 来源与 K30 本分支历史见[第 24 章](24-Windows原生构建与4090D实验.md)、[第 25 章](25-K30同步机制与4090D性能分析.md)。

所有 bank、sector、同步时序图都注明各自层次：源代码的地址与先后关系、编译后的指令、硬件采样和未插桩计时各回答不同问题，不能互相替代。
