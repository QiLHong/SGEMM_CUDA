# CUDA 与 SGEMM 基础复习

如果希望连续学习而不是逐题查阅，先读[微教材 从矩阵乘法到 CUDA 微内核](微教材/README.md)。
它保留原八章 SIMT 推导与 CPU 练习，并新增第 9 章 Tensor Core 衔接；本篇保留为知识点速查。

这篇从基础概念出发，帮助你看懂代码、解释实验，再准备项目面试。建议先读本篇，
再读[优化故事](06-RTX3080-K10优化故事.md)和[项目面试 QA](21-SGEMM模拟面试QA.md)。
不要求一次记住全部指标：先回答“算什么、谁来算、数据在哪、什么时候能用”，
最后才回答“为什么快”。本文不新增 kernel 或性能实验。B1–B36 保留原 SIMT 基础，
新增 B37–B46 连接 RTX 4090 D Tensor Core 路线；早期暂停调优的表述只属于 RTX 3080 阶段。

## 一 先看懂矩阵乘法

### B1 GEMM 和 SGEMM 是什么

GEMM 是通用矩阵乘法运算，常见形式为 `C = alpha*A*B + beta*C`。
SGEMM 的 S 表示单精度浮点。这里 A 是 M×K，B 是 K×N，C 是 M×N；
M 决定输出行数，N 决定输出列数，K 是每个输出需要累加的长度。

例如 A 是 2×3、B 是 3×4，结果就是 2×4。计算 `C[1,2]`，需要 A 第 1 行
与 B 第 2 列做点积。所有索引从 0 开始。不是 A 与 B 对应位置逐元素相乘。

### B2 alpha 和 beta 有什么用

alpha 缩放新乘积，beta 缩放旧 C。若点积为 10、旧 C 为 2，alpha=0.5、beta=3，
结果是 `0.5*10+3*2=11`。beta=0 时数学结果不需要旧 C；实现可以专门省去其读取。

如果把 11 再作为下一次旧 C，结果就变为 38，不再是重复同一输入。
这正是本项目为什么在公平 benchmark 中每次恢复 C0。

### B3 行主序和 stride 是什么

行主序表示同一行相邻元素连续存放。一个紧密存储的 3×4 矩阵中，
`X[2,1]` 的元素偏移为 `2*4+1=9`。若元素是 float，字节偏移为 36。
指针 `X+9` 已经按 float 大小移动，不能再额外乘一次 4。

stride 是跨到下一行要移动多少元素；有 padding 时可能大于实际列数。
本项目紧密行主序下，A 用 `row*K+k`，B 用 `k*N+col`，C 用 `row*N+col`。
BM/BN 是 tile 大小，不能拿来替换完整矩阵 stride。

### B4 为什么矩阵乘法算作 2MNK 次浮点运算

每个输出大约做 K 次乘法、K 次累加，有 MN 个输出，常用计数便是 2MNK。
这是一致使用的性能约定，不额外计入 alpha/beta 和地址运算。
对于只求一次点积，严格写出的加法次数也可以是 K−1，不影响本项目采用的约定。

GFLOPS 是每秒十亿次浮点运算，不是每秒十亿条机器指令：

```text
GFLOPS = 2*M*N*K / (time_ms * 10^6)
TFLOPS = GFLOPS / 1000
```

### B5 FMA 和 FFMA 是什么

FMA 表示 fused multiply-add，计算 `a*b+c`，在融合运算的最终结果处舍入。
FFMA 是这里观察到的浮点融合乘加机器指令。一次 FMA 按乘与加计两个 FLOP，
不代表两条指令。它也不等于 Tensor Core 的矩阵乘指令。

融合与先乘再加可能产生不同数值，因为中间舍入不同。浮点运算顺序也会影响结果，
不能直接套用实数的结合律。参见 [NVIDIA 浮点说明](https://docs.nvidia.com/cuda/archive/12.4.1/floating-point/index.html)。

### B6 FP32 TF32 和误差门槛有什么区别

FP32 是存储/计算格式，TF32 是 Tensor Core 可使用的一种较低乘数精度路径；
float 输入、float 输出不自动证明中间乘法使用完整 FP32 精度。
本项目将严格 FP32 与 TF32 对照分开，不拿降低精度的成绩混入同一性能比例。

绝对误差是 `abs(actual-reference)`；相对误差还要参考结果尺度，接近零时需要
特别处理。本项目保留 0.01 绝对门槛并拒绝非有限值，这不是所有任务通用的精度标准。
同为 FP32 也不保证与 cuBLAS 位级一致。项目的实际精度配置见
[基线审计](05-RTX3080-FP32基线与瓶颈审计.md)。

## 二 再认识 CUDA 的执行层次

### B7 CPU 和 GPU 的思路有什么不同

可以先粗略理解为：CPU 擅长低延迟地处理复杂控制，GPU 擅长让大量相似工作并行推进。
GPU 有较多计算资源，但不是“把任意 CPU 程序搬过去就更快”。任务规模、数据传输、
分支和并行程度都影响收益。SGEMM 有大量独立输出及数据复用机会，因此适合 GPU。

### B8 Host Device kernel 分别是什么

Host 通常指 CPU 一侧，Device 指 GPU 一侧。kernel 是启动到 GPU 上执行的函数。
每个线程执行同一份函数代码，但拥有自己的索引和私有状态。

```cpp
__global__ void add(const float* a, const float* b, float* c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];
}
// 示例：设备指针已经分配和初始化，n>0；这里只展示启动形式。
// add<<<(n + 255) / 256, 256>>>(da, db, dc, n);
```

`__global__` 标记 kernel，`__device__` 标记设备函数，`__host__` 标记主机函数。
上面是索引示意，不是包含分配、拷贝、错误检查的完整可运行程序。

### B9 Grid Block Thread 如何对应

一次 kernel launch 产生一个 grid；grid 中有多个 block；block 中有多个 thread。
`<<<grid, block>>>` 指定块数和每块线程数，不是“核函数执行次数和 CPU 线程数”。

例如每块 128 threads，`blockIdx.x=2, threadIdx.x=7`，一维全局编号是 263。
二维 block 按 x 优先展开：`tid=x+blockDim.x*y`。变量名 x/y 不天然代表矩阵列/行，
要看程序如何映射。仓库 K1 的 x 就对应输出行，详见[索引教程](01-矩阵索引与线程映射.md)。

### B10 Warp 和 lane 是什么

在本项目的 NVIDIA 架构上，一个 warp 是 32 个线程的执行分组，lane 是组内编号。
128-thread block 有四个 warp。对一维线程块，`warp=tid/32, lane=tid%32`；
二维线程块应先得到线性 tid。

warp 不是 32 个 block，也不是 GPU 上的一个固定“核心”。理解一次访存时，需要
同时列出 32 个 lane 的地址，不能只看一个线程自己的循环是否连续。

### B11 SM 和 block 是什么关系

SM 是 GPU 的一个流式多处理器，里面有执行管线、调度器、寄存器等资源。
本项目普通 CUDA block 的线程分配到同一个 SM，以使用 block 级 shared 和同步；
一个 SM 可以同时驻留多个 block，前提是资源足够。block 不等于 SM。

block 的编号不是执行顺序，不能默认 block 0 先写好数据、block 1 后读取。
普通 kernel 应让各 block 独立推进，跨 block 同步另有约束。
执行模型可核对 [CUDA Programming Guide](https://docs.nvidia.com/cuda/archive/12.4.1/cuda-c-programming-guide/index.html#thread-hierarchy)。

### B12 SIMT 和分支发散是什么

SIMT 允许线程表达各自的程序状态，但 warp 内执行组织仍以共同指令及活跃线程为基础。
若同一 warp 的不同 lane 进入不同分支，可能需要分路径执行，降低有效 lane 利用。
并不是所有 `if` 都很慢：warp 内一致的分支、短分支的谓词化都应具体分析。

独立线程调度也不意味着线程之间自动完成同步。凡是交换数据，仍要用合法同步协议，
不能依赖“同一个 warp 大概总在同一步”。

## 三 数据放在哪里以及怎样搬运

### B13 GPU 内存层次先记哪几种

|概念|在本项目中的角色|最容易误解的地方|
|---|---|---|
|Global memory|存放完整 A/B/C|访问可能经过缓存，不是每次都到 DRAM|
|L2 cache|缓存跨 SM 的 global 等访问|命中提高不保证总耗时下降|
|L1 cache|SM 侧的缓存资源|不是源码显式管理的 shared 数组|
|Shared memory|block 合作缓存 A/B tile|需要明确谁写、谁读和何时回收|
|Registers|线程的 fragment 和 accumulator|源码局部变量不保证全部分配在寄存器|
|Local memory|线程私有的地址空间，可能承载 spill|local 指私有，不代表在片上很快|

不必背固定延迟数字；先分清作用域、谁管理数据和资源限制。L1/shared 的物理资源
关系因架构而异，不能据此把编程语义视为相同。

### B14 Spill 是什么

当寄存器分配无法容纳所需值时，编译器可能把部分值放入 local memory，增加读写。
但 local memory 也可能来自难以静态索引的数组等原因，所以“有 local”与
“所有 local 都是 spill”不是完全相同的判断。

本项目同时检查 ptxas spill 报告、机器码和运行时 local-memory 指标。
零 spill 只是一个重要条件，不证明吞吐最优；K32 单 barrier 版零 spill 仍然回退。

### B15 什么是合并访存

考察同一 warp 在同一条 global 访问上需要哪些地址。若 lane 0..31 读取
`A[0]..A[31]`，请求集中；若读取 `A[0],A[1024],A[2048]...`，则很分散。
后者每个线程下一轮可能读到自己的相邻元素，但不能据此说这一轮 warp 访问合并。

合并访问不保证只产生一个事务，还涉及访问宽度、地址对齐及参与 lane。
连续访问也不等于所有数据都来自 L2。K1/K2 的对照可手推地址验证，无需先背指标。

### B16 float4 为什么可能有帮助

float4 是四个 float，共 16 字节。它可以表达向量化读写，减少搬运所需指令，
但不是四个线程、不是 FP16，也不自动等于四倍性能。
使用对应的向量 load/store 时要满足地址对齐和四个元素全部有效的前提。

即使 cudaMalloc 的基地址对齐，`base+1` 个 float 也可能破坏 16 字节对齐。
能否生成期望的宽指令，要检查实际 SASS；不能仅凭 C++ 类型认定编译结果。

### B17 Shared bank conflict 到底是什么

对本项目的 32-bit shared 访问，可先用 `bank=(字节地址/4)%32` 手算。
在同一服务分组内，不同 lane 访问同一 bank 的不同字，可能导致串行服务；
多个 lane 读取完全相同的字可广播，不能把这种情况直接算成冲突。

例：标量读取 `tile[lane][0]`，行跨度 32 个 float 时，地址偏移为 `32*lane`，
bank 都是 0；若跨度为 33，则 bank 为 lane，分布到 32 个 bank。
这是说明 padding 原理的标量例子，不是对 K30 的 LDS.128 指令直接套用的结论。

更多基础访存示例见 [CUDA Best Practices 的内存章节](https://docs.nvidia.com/cuda/archive/12.4.1/cuda-c-best-practices-guide/index.html#memory-optimizations)。

### B18 Padding 对齐 Swizzle 是同一个东西吗

不是。padding 在行尾增加未用于数学输出的存储，改变 stride；对齐约束地址的
整除关系；swizzle 则改变逻辑坐标到物理地址的映射。
它们可能互相影响，但目标和约束不同。

布局改变后，写入和读取都必须使用一致的映射。XOR 不是“性能魔法”：K16 的
具体 XOR 破坏已有访问模式，实际更慢；在另一套映射里是否有效需要重新证明。

## 四 数据正确还依赖同步

### B19 __syncthreads 解决什么问题

它是 block 范围的 barrier，并为参与线程之间的相关内存访问提供同步可见性。
“我写完了”不等于“别人写完了”，也不等于“别人已读完了”。
因此单槽 shared tiling 常见两个同步点：

```text
合作写 tile → 同步确保写完 → 计算并读取 tile
            → 同步确保读完 → 下一轮覆盖 tile
```

不能让不同参与线程走不兼容的 barrier 路径。处理尾部时常见安全做法是对无效
数据补零或屏蔽访存，但仍按协议参与同步；不能机械地把越界线程提前 return。

### B20 __syncwarp atomic 和 barrier 有何不同

`__syncwarp(mask)` 用于约定 mask 中线程的 warp 范围同步，不能替代跨 warp 的
block barrier。原子操作保证指定位置的更新具有原子性，不代表所有线程都已
到达某处；例如 atomicAdd 不能自动承担“整个 tile 已装载完成”的协议。

`volatile` 也不是同步协议，不能拿它修复所有 race。先确定参与者和数据生命周期，
再选择同步工具，而不是看到 race 就随意加一个修饰符。

### B21 kernel 异步启动和 stream 是什么

kernel launch 通常不等待 GPU 完成就返回 CPU。同一 stream 内的操作按其顺序
约束执行；不同 stream 不能默认相互等待，也不能保证一定并行。
需要跨 stream 依赖时可用 event 等机制，CPU 要使用结果时必须确保计算完成。

所以 `launch(); CPU计时结束;` 通常只反映提交开销，不能作为 GPU kernel 耗时。
异步拷贝要形成实际重叠，还受内存类型、设备能力和操作依赖限制。

### B22 Double buffering 为什么不一定更快

双缓冲只是准备两个槽，让“当前读取”和“下一次写入”有机会不冲突。
它带来更多存储、状态与同步成本；如果编译器没有生成有效重叠，或资源增加
减少了可驻留 block，最终可能更慢。

理解时画出三个事件：槽位写入完成、读者全部退出、允许覆盖。K30 用一次轮末
barrier 同时完成发布 future 和回收 current，但这是特定协议，不是普遍删同步公式。

### B23 cp.async 是什么 为什么不能替代全部同步

在本项目目标架构上，相关指令可以异步发起 global→shared 的复制，减少传统
global→register→shared 路径中的搬运寄存器需求。但提交复制不等于完成复制。
仍需正确的等待、跨线程可见性和槽位回收协议；它不会替你计算 GEMM。

K17 的 B-only async 确实执行了相关指令，却没有净收益。这说明具体实现成本
没有被收益抵消，不是证明所有 async copy 都无效。

## 五 理解 tiling 与寄存器复用

### B24 为什么 SGEMM 要分块

同一个 A 元素会参与多列输出，同一个 B 元素会参与多行输出。若每个输出
都独立从 global 读取，逻辑上有大量重复。block tiling 让线程合作装载一个
A/B 子块，再多次消费它。真正节省了多少 DRAM 流量还受缓存影响，不能只按
源码 load 次数判断实际流量。

### B25 BM BN BK 分别控制什么

BM/BN 是 CTA 输出块的行数和列数，BK 是本次缓存的归约长度。例如
BM=BN=128、BK=16，装载 A 的 128×16 和 B 的 16×128，更新 128×128 输出。
完整 K=4096 时需要 256 个这样的 K tile。

BK 从 16 改为 8，不改变总数学工作量，但改变加载、同步、展开和资源占用的节奏。
BM/BN 越大也不必越快，它们会增加 accumulator/shared 需求并影响边界与并行粒度。

### B26 Thread tile 为什么能减少 shared 读取

每线程只算一个结果时，通常读一个 A 和一个 B 更新一个 accumulator。
如果算 TM×TN 输出，可以读取 TM 个 A、TN 个 B，做一个小外积：

```text
读 A: a0, a1
读 B: b0, b1, b2
更新 6 个结果: cij += ai * bj
```

这次读 5 个标量、做 6 次 FFMA，而不是为六个结果各读一对数据。
但同时要保存六个 accumulator。收益来自复用，成本来自寄存器与调度。

### B27 Warp tile Fragment Iterator 怎么用基础语言解释

Warp tile 是这个 warp 负责的输出区域；fragment 是某线程当前握在寄存器中的
一小组输入或结果；iterator 是把“逻辑上要哪几个元素”转成“实际从哪里读取”
的地址组织方式。它不意味着使用了 Tensor Core，也不必是动态迭代器对象。

K30 每个 warp 负责 64×64，每 lane 负责 16×8。一个 lane 的输出不一定是完整
连续矩形；逻辑所有权、shared 物理布局和寄存器编号是三个不同层次。

### B28 为什么 K30 的 128 个 accumulator 合理但不免费

`16*8=128`，每 lane 更新 128 个结果；`32*128=4096=64*64`，正好对应 warp tile。
乘积关系只是必要的数量检查，还要验证不同 lane 的坐标互不重叠并覆盖完整区域。

这些 accumulator 提供独立计算和复用，但占用寄存器。K30 实际分配 235 个，
不是恰好 128：还包括 fragment、地址和其他状态。具体分配由编译器决定。
进一步推导见[复现手册的数据流](20-项目复现与简历讲述.md)。

## 六 从基础概念走到性能分析

### B29 Occupancy 是什么 为什么不等于利用率

可先理解为 active warp 数相对该架构最大可驻留 warp 数的比例。
理论 occupancy 根据资源容量估算，实测 occupancy 反映运行过程；二者不是
“实际 FFMA 管线有多忙”。更多 active warp 可能帮助隐藏延迟，也可能只是更多
等待或搬运线程。

本项目 K10 与 K30 occupancy 接近，性能仍不同。K32 overlap 增加 producer warp，
总 occupancy 上升，但计算 warp 容量未增加，也没有稳定超过 K30。

### B30 ILP 与 latency hiding 是什么

Latency 是一项操作到结果可用的等待时间；throughput 是持续执行时单位时间
能处理多少工作。隐藏延迟的办法之一，是等待某项结果时执行其他独立工作。
这既可能来自其他 warp，也可能来自同一线程的独立指令，即 ILP。

`acc=acc+a*b` 反复更新同一个 acc 有依赖；多个独立 acc 可以交错更新。
但增加 acc 会占用寄存器，不能无限提高 ILP。本项目有 128 个 accumulator，
不应未经 SASS 分析就断言“只是累加链太长”。

### B31 Scoreboard Issue 和 FP32 active 是什么

Scoreboard 跟踪尚未完成的依赖，避免指令使用未就绪的数据。Nsight 的 stall
分类是定位线索，不能把名称当作单一根因，例如 long-scoreboard 不自动等于
“DRAM 带宽已经满了”。

Issue 关注指令是否发射；FP32 active 关注相关执行管线的活动；两者都不是
GFLOPS 本身。`nvidia-smi` 的 GPU 利用率也不是 FP32 算力利用率。计数器定义
和 replay 限制见 [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)。

### B32 Compute bound Memory bound 与算术强度怎么理解

Compute bound 指计算吞吐资源限制明显；memory bound 指某层内存服务成为主要
限制；也可能因依赖或发射效率受限，计算与 DRAM 指标都不满。
不能仅凭“矩阵很大”或“带宽不满”完成分类。

算术强度是 FLOP/byte，但必须说明针对哪一层和哪些流量。一个 BM×BN×BK 的
CTA 阶段，理想只装载一次 A/B、忽略 C 与 padding，global 输入层的逻辑强度是：

```text
2*BM*BN*BK / [4*(BM*BK + BK*BN)]
= BM*BN / [2*(BM+BN)]
```

BM=BN=128 时为 32 FLOP/byte。这不是实测 DRAM 强度，也没有计跨 CTA 缓存复用。
Roofline 用计算峰值与带宽×强度给出上界模型，不保证某个 kernel 能达到上界。

### B33 PTX SASS ptxas 是什么

CUDA 源码是你表达算法的层；PTX 是虚拟指令表示；ptxas 负责将其汇编/优化到
目标机器代码；SASS 是观察最终 GPU 指令的重要入口。实际构建也可能由驱动 JIT
将 PTX 编译为目标代码，具体要看交付的二进制内容。

写在源码前面的 load 不保证仍在最终机器码前面。寄存器数、spill 和指令调度
要看编译产物。K30 的“源码预取被下沉”和 K32 的“删 barrier 改变寄存器分配”
都是项目里的实际例子，不应只根据 C++ 行数解释性能。

### B34 Benchmark 最基础的公平条件是什么

相同输入、尺寸、alpha/beta、精度、编译参数和计时边界，预热后多轮交错测量，
报告均值与波动，保留失败结果。使用 CUDA event 计设备执行时间，并检查异步错误。
只调用 `cudaGetLastError` 不能代替等待完成后检查执行错误。

若 profiler 改变缓存或时钟条件，它的时间不要直接混入常规 benchmark。
约 +0.2% 的变化若与波动相当，应视为待确认信号，不是因为数值为正就宣布成功。

### B35 正确性检查和 Sanitizer 各解决什么

数值对照检查结果是否满足当前输入的误差要求；memcheck 检查指定范围内的非法
内存访问，racecheck 查 shared hazard，synccheck 查同步使用问题。
数值 PASS 可能掩盖偶发 race，工具 PASS 也不能替代数学正确性判断。

尤其要区分边界检查与同步：`if(valid)` 只保护某次访问，并不自动保证整个 block
后续 barrier 合法。本项目高性能 kernel 有尺寸/对齐契约，不是任意输入都安全。

### B36 学到什么程度可以进入项目面试题

先能手推一条 warp 的地址、算出 tile 的数据复用、解释两次 barrier 的不同用途，
再能区分占用率与吞吐、数学结果与数值误差。此时阅读 K15–K32 不再只是记住
“这个失败、那个成功”，而能问清它改变了什么成本、用什么证据验证。

正式复现使用[项目手册](20-项目复现与简历讲述.md)，深层追问进入
[35 道项目面试题](21-SGEMM模拟面试QA.md)。不要把本篇手算模型当作新的硬件测量。

## 七 八道手算与判断练习

先完成题目再看答案。除题目明确要求外，不需要 GPU。

1. A 为 3×5，B 为 5×7，C 形状是什么？紧密行主序下 A[2,4] 的元素与字节偏移？
2. 一维 block 有 128 threads，tid=70 是第几个 warp、哪个 lane？blockIdx.x=3
   时全局线程编号是多少？
3. warp 的 lane 0..31 分别读取 `p[lane]` 与 `p[lane*1024]`，哪个访问更集中？
   两种情况下线程下一轮都读自己的下一个元素，是否改变本轮结论？
4. 32 个 lane 标量读 shared 的 `tile[lane][0]`，stride 为 32 和 33 时 bank
   分别是什么？如果所有 lane 都读 `tile[0][0]` 又是什么情况？
5. TM=4、TN=8 的一次外积读取多少个输入标量、更新多少 accumulator、做多少 FLOP？
6. `if(threadIdx.x<16){ write_shared(); __syncthreads(); }` 适合作为全块
   发布协议吗？如何从参与者角度解释问题？
7. 1024³ SGEMM 测得 1 ms，GFLOPS 约多少？FMA 计一次指令和两个 FLOP 是否矛盾？
8. 新版 L2 hit 增加、无 spill，但均值与基线持平，能宣布性能优化成功吗？

### 参考答案与推导

1. C 为 3×7；A[2,4] 元素偏移 `2*5+4=14`，float 字节偏移 56。
2. `70/32=2`，lane=`70%32=6`；全局编号 `3*128+70=454`，均从 0 编号。
3. `p[lane]` 更集中。访存合并看同一条指令的跨 lane 地址，不能拿线程间隔一轮
   的访问来证明当前请求合并；仍需考虑对齐和请求宽度。
4. stride32：bank=`32*lane%32=0`；stride33：bank=`33*lane%32=lane`。
   全读同一个字是广播情形，不按不同字的 bank conflict 算。仅限题目的标量模型。
5. 读取 12 个标量，更新 32 个 accumulator，32 次 FMA 即 64 FLOP；还需为
   结果及输入提供寄存器，不能由此直接预测最终吞吐。
6. 不适合作为该全块协议。只有部分线程进入分支，不能按全块共同到达来同步。
   可将写入按条件屏蔽、把 block barrier 放在全部所需参与者一致执行的位置，
   但还必须确保每个会被读的 shared 元素均已正确初始化。
7. `2*1024^3/10^6=2147.483648 GFLOPS`，约 2.147 TFLOPS。不矛盾，指令数与
   浮点运算数是不同计量单位。
8. 不能宣布吞吐提升。可记录局部指标改善及其成本，结果判为无稳定净收益；
   K15/K32 都能帮助解释这种情况。

## 八 建议复习顺序

|一轮复习|内容|结束时应能独立完成|
|---|---|---|
|第一轮|B1–B12|手算索引、线程编号，区分 warp/block/SM|
|第二轮|B13–B23|画出 global/shared/register 路径与槽位生命周期|
|第三轮|B24–B33|推导复用、资源成本，区分 occupancy/issue/吞吐|
|第四轮|B34–B36 与练习|检查一次实验是否公平，解释一项失败结论|

建议先打开 [K1](../../src/kernels/1_naive.cuh)、
[K2](../../src/kernels/2_kernel_global_mem_coalesce.cuh)、
[K3](../../src/kernels/3_kernel_shared_mem_blocking.cuh)，把每个地址与同步画出来。
再看 [K5](../../src/kernels/5_kernel_2D_blocktiling.cuh) 的小外积，最后进入
[K10](../../src/kernels/10_kernel_warptiling.cuh) 与
[K30](../../src/kernels/30_kernel_mini_simt.cuh)。
现有 [01 索引教程](01-矩阵索引与线程映射.md)和
[02 分块与流水线](02-分块与流水线.md)可作为逐行源码配套，不必从最复杂的版本开始背。

## 九 从 SIMT 衔接 Tensor Core

以下以本仓库的 [BN64 TF32 kernel](../../experiments/tensorcore-bn64/tf32_bn64.cuh)为例。
先理解代码中的操作与数据，再查[第 32 章](32-RTX4090D-TensorCore优化学习指南.md)的实测；
若希望连续推导，转到[微教材第 9 章](微教材/09-TensorCore从SIMT到矩阵指令.md)。

### B37 Tensor Core 与普通 FP32 SIMT 的分工有何不同

K30 的主计算由每线程标量 accumulator 执行 FFMA，lane 分工由自己的循环与索引决定。
BN64 则把一组寄存器 operand 送入 warp 级 `mma.sync`，本例每次协作推进一个
16×8 输出的 8 宽 K 片段。两者仍需安排 thread、warp、shared 和寄存器，
Tensor Core 没有自动替程序解决数据搬运、输出覆盖或同步。

### B38 float 存储、TF32 乘数、FP32 累加怎样同时成立

“占几个字节”和“算术保留多少精度”不能混为一谈。本次 TF32 路线先显式量化原始
FP32 输入，仍用 float 存放量化后的位模式，随后执行 TF32 乘数、FP32 累加和输出。
`__float_to_tf32` 的输入输出类型均为 float，数值契约却改变；官方说明见
[CUDA 12.4 alternate floating point](https://docs.nvidia.com/cuda/archive/12.4.1/cuda-c-programming-guide/index.html#alternate-floating-point)。

| 本项目路线 | 实际输入 | 累加 | 输出 | 能否替代原始 FP32 精度结论 |
|---|---|---|---|---|
| strict FP32 / K30 | 原 FP32 | FP32 | FP32 | 仍需按原阈值检查 |
| TF32 Tensor | 显式 TF32 量化、float 存储 | FP32 | FP32 | 不能；本批原严格阈值失败 |
| 第 28 章 FP16 Tensor | half | FP32 | half | 不能；还有最终 half 舍入 |

BF16 没有本轮正式复现成绩，不能从 FP16 数据推断其精度或速度。FP32 累加也不会
恢复已经丢失的输入信息；低精度结果与同精度库一致，不等于与原始数学输入一致。

### B39 WMMA、内联 PTX MMA 和 SASS 分别在哪一层

WMMA 是 CUDA C++ 层的矩阵协作接口；本实验直接用内联 PTX 写出
`mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32`，再由编译器生成 sm_89 SASS。
当前二进制观察到 HMMA。不要把一个源码调用、一条 PTX、某个静态 SASS 数量和
一次完整 GEMM 混为一谈；指令数还受展开与循环动态次数影响。

WMMA 的 fragment 元素映射不由其 C++ 接口公开保证，不能把本 kernel 的内联 PTX
写回公式直接套到任意 `wmma::fragment::x` 上。语义界限见
[CUDA WMMA 描述](https://docs.nvidia.com/cuda/archive/12.4.1/cuda-c-programming-guide/index.html#wmma)。

### B40 一次 m16n8k8 为什么不是一个线程的 2048 FLOP

按本项目统一口径，一次 warp 级矩阵乘加对应 `2*16*8*8=2048 FLOP`，这是整个 warp
的总工作量；不能再乘 32。128 个输出分布在 32 个 lane 的 accumulator 中，每 lane
持有 4 个值。本 BN64 的每个 warp 组合 `4*2` 个输出 fragment，形成 64×16 输出。
MMA 的 warp 协作要求见 [PTX MMA](https://docs.nvidia.com/cuda/archive/12.4.1/parallel-thread-execution/index.html#warp-level-matrix-instructions-mma)。

### B41 fragment 是连续的小矩形吗

不是必须。本 kernel 的一个 `sum[mi][ni][0..3]`，在 warp0、lane0、mi=ni=0 时对应
(0,0)、(0,1)、(8,0)、(8,1)，在行方向分成两组。`sum` 的数组下标只是本线程
寄存器的编号；矩阵坐标必须经过写回公式求出。把 fragment 误画成四个连续列，
后续对 shared 布局或写回的推导都会错。

### B42 ldmatrix、cp.async 和 MMA 各自做什么

可以顺着代码分为搬到 shared、装入 operand、执行乘加三步：

```text
global A/B --cp.async--> shared 双 slot
shared A --ldmatrix--> reg_a
shared B --shared load--> reg_b
reg_a / reg_b --mma.sync--> FP32 sum --float2 store--> global C
```

本实现中 `.b16` 描述 ldmatrix 的加载组织，不能据此把随后的 TF32 数学称为 FP16。
`cp.async` 无需显式中间数据寄存器，但地址仍占寄存器；`float2` 写回一次两个结果，
也没有把两个 FP32 值变成一种新的计算精度。

### B43 wait、barrier 与双 slot 怎样合作

先给 slot0 发出 copy，commit、wait 后由 CTA barrier 发布；循环中向另一个 slot
预取下一 tile，同时从当前 slot 装载 operand、计算。末尾等待新 copy，再让全 CTA
会合，随后交换读写角色。wait 不是“所有线程都已读完旧 tile”的声明，barrier 也不是
异步 copy 自动完成的声明；必须按这份协议把两种条件都满足。末尾只计算最后一个已加载
tile，不再预取越界数据。[等待指令语义](https://docs.nvidia.com/cuda/archive/12.4.1/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-wait-group-cp-async-wait-all)解释了 wait 的作用范围。

### B44 为什么 BK、BN 不能只改模板数字

BK16→32 必须多搬一倍 K 数据并多做 K8 步；本实验使用两个 16 宽 slab 来保留原
swizzle 的行跨度。BN128→64 则同时改变 warp 的 N 跨度、B 的搬运分工、N fragment
循环上界和 C 写回。漏改一个环节，可能发生未加载读取、漏算或输出重叠。

检查顺序是先枚举映射与地址，再小尺寸数值验证，最后计时。合法 grid 不等于有
边界保护；本入口明确拒绝不满足尺寸限制的形状，不用向上取整掩盖尾块越界。

### B45 看 Tensor kernel 的 profiler 要换哪些问题

先用 ptxas 和 SASS 确认资源、spill 与实际 MMA 路径，再看 Tensor 管线和对应的
issue、eligible、barrier、scoreboard、shared wavefront、L2/DRAM 数据。K30 的
FP32 FFMA 活跃度不能直接当 Tensor kernel 主计算的利用率。

尤其检查分母：BN64 在 2048 上 Tensor active 44.25%→44.49% 变化不大，
Tensor elapsed 却从 33.26%→39.82%。更多 SM 在更多经过周期内有任务，可与更高吞吐
同时发生。单个 stall 降低或 L2 命中率升高都不等于总耗时下降；仍以公平 event 为准。

### B46 哪些证据组合起来才算复现成功

先说明复现的是作者原生 device 实现还是派生优化，再逐项核对：输入字节/指纹、
计算语义、精度、形状约束、误差标准、计时边界、编译架构和原始 event。
本轮相对作者基线逐位一致、无 NaN/Inf，支持指定测试集的实现一致性；不替代原始
FP32 精度要求。GPU 参考全量，CPU double 大尺寸只抽样 256 点，Sanitizer 未执行，
这三种范围必须分别写清。

## 十 Tensor Core 补充练习

1. 本实现的一个 warp 每个 K8 步做 `4*2` 次 m16n8k8，完成多少 FLOP？一个 BK32
   tile 含几步？8 warp 的 CTA 完成多少 FLOP？
2. BN64 的 warp0、lane1、mi=1、ni=1 对应哪四个 CTA 内输出？
3. `BM=128,BN=64,BK=32` 的双 buffer 用户 shared 有多少字节？能否仅凭此数保证
   每个 SM 运行期间始终有两个 CTA？
4. 相对原版每 CTA barrier 减半、CTA 数翻倍，能写“全 GEMM 同步调用减少一半”吗？
5. 同精度 cuBLAS 的 max abs 为 0，原始 FP32 max abs 为 0.503967285，应该如何报告？

### 补充练习答案

1. 每 warp 每步 `8*2048=16384 FLOP`；BK32 有 4 个 K8 步；CTA 共
   `8*4*16384=524288 FLOP`，等于 `2*128*64*32`。不要把 warp 工作再乘一次 32。
2. (16,10)、(16,11)、(24,10)、(24,11)。在 lane1 第一 fragment 坐标上加 `(16,8)`。
3. `2*(128*32+32*64)*4=49152` 字节。还要检查寄存器、线程、driver 开销和实际
   shared 配置；理论允许驻留也不是运行全过程的占用保证。
4. 不能。两项相乘后总调用数保持不变；等待时间可能变化，但需要单独观测，不能
   从调用计数直接推导耗时减半。
5. 实现与该同精度参考在已测输出上一致，但相对原始 FP32 仍有量化/累加误差，
   绝对 0.01 标准失败；不能把前一个 PASS 当成后一个 PASS。

练完后再进入 [Q36–Q47](21-SGEMM模拟面试QA.md#九-rtx-4090-d-tensor-core-补充面试题)，
用实际 CSV 而不是记忆中的最好成绩解释每个结论。

## 十一 读 CUTLASS 结果前再区分四件事

这里衔接[第34章](34-CUTLASS与BN64的4090D公平对照.md)，不改变前面历史练习的数字。

| 容易混淆的概念 | 本项目应怎样区分 |
|---|---|
| CUTLASS 与 Tensor Core | CUTLASS是模板库；本轮同时实例化TF32 TensorOp和严格FP32 SIMT，库名不决定数学精度 |
| FP32存储与严格FP32乘法 | `tfloat32_t`与float等宽不代表保留了原始FP32乘数精度；先显式量化的输入仍可能相对原始输入有明显误差 |
| 多阶段异步搬运与TMA | 本轮是普通`cp.async`/LDGSTS，加stage不自动变成TMA；PTX/SASS和目标架构必须一起检查 |
| 统计完整与所有数值标准通过 | `DATA_AUDIT_PASS`核验数据链；abs、SIMT allclose、native allclose和NaN/Inf仍需各自读结果 |

也要区分“库支持边界”和“统一入口支持边界”：CUTLASS的predicated iterator不能保护同一harness里另一个没有尾块检查的作者kernel，所以共同入口仍需明确拒绝非法形状。

本轮每个实现的结果先由30个event求轮均值，再由7个轮均值求中位数，最后跨3批×3seed呈现范围。逐轮配对收益用`reference_ms/candidate_ms-1`；两个汇总中位数的比值是另一种统计量。九组合的最小值与最大值不是置信区间，更不能删掉慢样本来缩小范围。

学习顺序建议：[微教材第9章](微教材/09-TensorCore从SIMT到矩阵指令.md)理解fragment与同步，再读[第32章32.11](32-RTX4090D-TensorCore优化学习指南.md#3211-用-cutlass-对照校正优化直觉)看性能反例，最后用[第34章练习](34-CUTLASS与BN64的4090D公平对照.md#3410-五道自检题与参考答案)自检。
