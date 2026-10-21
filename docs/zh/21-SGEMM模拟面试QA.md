# SGEMM 模拟面试 QA：RTX 3080 SIMT 与 RTX 4090 D Tensor Core

适用方向：CUDA 性能优化、GPU 算子开发、推理引擎与高性能计算岗位。
Q1–Q35 保留 2026-10-13 的 RTX 3080 阶段：K30 V2 为该阶段最佳研究基线，K31/K32 为未接受的后续实验。
这 35 题回答以 RTX 3080、SM86、CUDA 12.4、4096³ strict FP32 的已测范围为准。
这是练习用答案，不是个人经历证明；开源、AI 辅助和自己的实际贡献要分别说明。

还不习惯直接回答面试题时，可先顺序读[微教材的 SIMT 基础八章](微教材/README.md)，
把地址、映射和同步亲手推一遍，再回到本页练习表达。

若想先复习基础，读[CUDA 与 SGEMM 基础复习](22-CUDA与SGEMM基础复习.md)：
原 36 个基础问题、地址/线程/bank 小例子及 8 道手算题，从矩阵乘法讲到性能测量；
另增 B37–B46 的 Tensor Core 知识和 5 道练习。
本页保留原 35 道项目深挖题，另增 Q36–Q47 的 RTX 4090 D Tensor Core 部分；两篇不混用题号。

基础读完后，再读[优化故事](06-RTX3080-K10优化故事.md)，并按
[复现手册](20-项目复现与简历讲述.md)实际运行。练习时先不看答案，控制每题
在一分钟内；面试官追问时，再打开对应源码或 CSV，而不是背诵指标名称。

基础与项目的衔接：B1–B6 对应本页 Q5/Q9–Q13，B7–B18 对应 Q6/Q7/Q18，
B19–B28 对应 Q19–Q23，B29–B36 对应 Q14–Q17 和 Q24–Q30。

## 一 项目价值与贡献

### Q1 用一分钟介绍这个项目

**参考回答：** 基于开源 SGEMM 教学项目，我研究了 RTX 3080 上严格 FP32 的
SIMT 微内核。先统一 cuBLAS 精度与 benchmark，再通过 fragment ownership 和
双缓冲同步组织的协同重构，让 K30 V2 在 4096³ 达到约 20.8 TFLOPS、同口径
cuBLAS 的 96.3%，比已调优 K10 提升约 5.85%。我保留了正确性、Sanitizer、
交错测量和 Nsight/SASS 证据，也记录了未获得净收益的实验。

**追问准备：** 说清你亲自完成或能独立解释的部分；不要把上游教学 kernel 说成
自己的全部原创。项目的价值是实验与微内核推理能力，而不只是一个性能数字。

### Q2 为什么不直接使用 cuBLAS

实际业务若 cuBLAS 已满足需求，我会优先用库。这个项目的目的，是理解 GPU
执行与算子设计，并建立可验证的性能诊断流程。它不是用一个固定方阵 kernel
替代完整矩阵库；生产库还要处理大量 shape、布局、精度、架构和调度策略。
是否自写算子，应由特定工作负载、融合需求和测量收益决定。

### Q3 你的贡献与 CUTLASS 和原作者是什么关系

K1–K12 的教学框架和算法基础来自原项目，中文翻译和 RTX 3080 调优是本分支
后续工作。研究参考了公开 CUTLASS 的分层和访问映射思想，K30 没有链接 CUTLASS。
README 保留上游归属及 AI 辅助说明。回答时应具体说自己如何验证、理解和修改，
不能把“读过或辅助生成”都表述成“独立设计并手写”。

### Q4 为什么收益只有百分之五左右还值得做

比较对象是已经约 91% cuBLAS 的 K10，不是一个很慢的 naive baseline。
七轮中从 19.652 提高到 20.801 TFLOPS，独立复测仍约 +5.62%，说明是可重复的
微内核改进。更重要的是找到 shared 服务成本与同步组织的组合效应，而不是通过
换基线、降低精度或挑最高值获得一个漂亮百分比。

## 二 数学和线程映射

### Q5 SGEMM 在这里计算什么 GFLOPS 如何算

`C=alpha*A*B+beta*C`，A、B、C 均为 FP32。常用主乘加工作量为 `2*M*N*K`，
FMA 计两个 FLOP。若时间单位为 ms，GFLOPS=`2*M*N*K/(ms*10^6)`。
4096³ 约有 137.44×10^9 FLOP，6.607 ms 对应约 20.8 TFLOPS；这个口径忽略
epilogue 的附加操作，各模式一致。

### Q6 K30 的 CTA Warp Lane 分别负责什么

CTA 负责 128×128 输出，有 128 threads，即四个 warp。每个 warp 负责一个
64×64 输出块，每 lane 拥有 16×8 共 128 个输出。每个 k 读取 A16/B8，做
128 次 FFMA；BK=16，所以一次 shared tile 的计算为每线程 2,048 次 FFMA。
这样每个加载的 A/B 值被多个 accumulator 复用。

**白板检查：** `4 warps × 32 lanes × 128 outputs = 128×128`，还须证明坐标
无重叠且无遗漏，单纯数量相等并不充分。代码见
[FragmentIterator](../../src/kernels/30_kernel_mini_simt.cuh)。

### Q7 请具体画出 lane 0 和 lane 1

在 warp 0，lane 0 的 M 坐标是 `0..3,16..19,32..35,48..51`，N 坐标是
`0..3,32..35`；取笛卡尔积。lane 1 将 M 坐标加 4，N 不变。lane 2 则使用
lane 0 的 M，并把 N 加 4。其余 warp 再加 64 行或列偏移。
解释这个映射比说“用了 warp tiling”更能证明理解。

### Q8 为什么 accumulator 不能无限增加

更多 accumulator 能增加复用和独立计算，但占用寄存器；超出资源后可能减少
驻留 CTA，或产生 spill。更大 tile 也会改变 shared、边界处理和加载分工。
K30 的 235 registers/thread 无 spill，并不意味着再多一点更好，也不意味着
降低到 209 就更快；K31 的实际低寄存器版本反而回退。

## 三 精度和正确性

### Q9 float 输入为什么不一定等于严格 FP32 乘法

输入存储类型和内部乘法精度是两件事，必须检查 cuBLAS compute type、math mode
和实际执行指令。本项目公平参考使用 PEDANTIC math 与 32F PEDANTIC compute，
清除 TF32 override，并用 profiler/SASS 确认路径。不能只凭 kernel 名字或
`CUBLAS_GEMM_DEFAULT_TENSOR_OP` 这个选项判断是否实际执行了 Tensor Core。

证据见[FP32 基线审计](05-RTX3080-FP32基线与瓶颈审计.md)。
compute type 的定义可核对 [CUDA 12.4 cuBLAS 文档](https://docs.nvidia.com/cuda/archive/12.4.1/cublas/index.html#cublascomputetype-t)。

### Q10 严格 FP32 是否意味着与 cuBLAS 位级相同

不是。不同并行归约顺序、FMA 融合和实现细节可能产生不同舍入结果。
本项目没有要求 bitwise identical，而是保留原始 0.01 absolute error 并拒绝
非有限值。这个阈值对当前输入与尺寸通过，不等于证明任意数据都满足同一误差。
若扩展到高动态范围或强消去输入，应增加独立精度验证，而不是放宽阈值。

### Q11 beta 为什么要分别测 0 1 3

beta=0 检查不依赖旧 C 的路径，beta=1/3 检查 read-modify-write 的 epilogue。
公平计时每次恢复 C0，避免 beta=3 时反复递推放大 C。旧 benchmark 的递推行为
应作为另一种执行语义保留，不与固定输入的结果混合。
如果 beta 非零时才失败，要重点检查旧 C 的加载、输出索引和 alpha/beta 应用。

### Q12 你如何排除越界 race 和同步错误

数值检查覆盖 128/256/512/1024/2048/4096 和三种 beta。另用 memcheck 检查规定
尺寸的非法访问，racecheck 检查 shared hazard，synccheck 检查同步使用。
K32 还有 CPU 有限状态模型检查槽位发布/回收与无死锁。工具 PASS 是已测范围的
证据，不是对全部输入的形式化证明；Sanitizer 测量也不用于性能排名。

**精确范围：** K32 最终 90 项数值检查；四个研究模式共 28 项 Sanitizer。
不能把这 28 项描述为六个尺寸上所有工具都跑过。

### Q13 行主序数据如何调用列主序 cuBLAS

利用 `(AB)^T=B^T A^T`，将行主序数组视为转置后的列主序数组，并交换传入
A/B 的顺序及对应维度。当前方阵 harness 使用 `db`、`da` 的顺序调用。
非方阵必须重新核对 M/N/K 和 leading dimensions，不能把方阵调用照抄过去。

## 四 性能测量是否可信

### Q14 如何证明不是偶然峰值

先 GPU 预热，各模式同 seed、alpha/beta、C 复位策略和编译参数，多轮交错执行，
保留各轮均值、波动和同轮 cuBLAS 比例，再独立复测。K30 原七轮 +5.85%，追加
三轮仍 +5.62%；K32 重叠版则两批都约 −0.045%，按噪声内持平，不挑最高一轮。
没有进行的显著性检验不能声称已经通过。

### Q15 为什么使用 CUDA event 而不是 CPU 时间

CUDA launch 是异步的，简单测 CPU 调用耗时会漏掉设备执行。event 放在同一流
的 kernel 前后，等待 stop 完成后取时间。这里不含分配、输入拷贝和 C0 恢复，
所以是 kernel-only 指标。如果业务关注端到端时间，需要另设口径并保留旧结果。

### Q16 Nsight 上的 kernel duration 可以当最终成绩吗

不直接使用。Profiler replay、缓存策略和时钟控制会改变运行条件。本项目将
公平 event 计时与 cold/steady profiling 分开；前者用于排名，后者解释执行行为。
同一 counter 也要在相同 policy 下比较。完整策略和命令见
[复现手册](20-项目复现与简历讲述.md)。

### Q17 你说 96.3% cuBLAS 为什么不是 96.5%

96.33% 是 K30 首轮七轮研究的固定成果数字，独立确认是 96.25%；后续 K32 收尾
同轮复测 K30 得到 96.51%。这些都保留，但简历不应挑最高批次。推荐写约 96.3%，
并能指出是哪份 CSV。相对值有自己的参考基线，不能跨批次组合。

## 五 从失败实验到 K30 的突破

### Q18 零 bank conflict 为什么还能优化 shared

Bank conflict 描述的是一种额外服务成本，不等于一次 warp LDS 只需要一个
wavefront。有效线程的地址分布、宽度和广播/复用关系会影响基础服务量。
K30 保持相同动态 LDS 数量，却降低约 40% 主循环 waves。因此应同时看 LDS、
wavefront、excessive/conflict，而不是看到 conflict=0 就停止分析。

### Q19 K30 最重要的优化是什么

新的 fragment ownership 与双 slot 单 barrier 协议共同有效。硬件证据是
shared 服务量下降、动态 BAR 接近减半、issue 和 FP32 active 提升；性能由
约 19.65 提高到 20.80 TFLOPS。单阶段 V1 已有较低 shared 服务量但更慢，所以
我不能把 +5.85% 全归到 layout，或给每个因素分配未经测量的百分比。

指标定义可核对 [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)；
本项目具体数值来自版本化 CSV，不是文档中的架构示例。

### Q20 双缓冲为什么一次 barrier 就够

当前读 slot 和 future 写 slot 不同，生产者不会覆盖本轮读者。所有线程到达
轮末 barrier 后，future 的写入对下一轮可见，current 的读取也全部结束。
下一轮才能复用旧槽。这依赖具体协议和所有参与者到达，不能机械地从任何
双缓冲代码中删掉一次 barrier。初始化和最后一轮也必须遵守相同生命周期。

### Q21 K30 是靠 Global 与 Compute 完美重叠吗

不能这样说。虽然源码将预取写在前面，SASS 的主循环 LDG 被下沉到 2,046 条
FFMA 之后，只剩两条 FFMA 在后。证据支持同步和 fragment 服务成本的组合改善，
不支持“大范围 Global load/compute overlap 是主要原因”。这是源码意图与
机器码执行区别的一个重要反例。

### Q22 Grid Swizzle XOR 和 cp.async 为什么没成功

它们都改变了成本分布：K15 部分 L2 指标改善但 issue/吞吐下降；K16 的具体 XOR
与原访问模式不匹配；K17 B-only async 没有形成足够净收益。结论限定为这些
实现和条件未成功，不是这些技术普遍无效。不能把“没有加速”直接写成“缓存或
访存从来不是瓶颈”。原始证据见故事中各版本报告链接。

### Q23 为什么不照着 cuBLAS 的 kernel 名字写

名字不能完整证明 BK、stage 数、fragment、iterator 和 pipeline 的内部实现。
应结合资源、SASS 和公开参考代码分析，而且公开 CUTLASS 不等于私有 cuBLAS
二进制。K20/K21 的简单大 CTA 对照失败，说明外形相似不等于执行组织相同。
尤其不能因为名字里有 `8x4` 就直接断言“BK8 四阶段”。

## 六 寄存器与调度的深入追问

### Q24 Occupancy 越高越好吗

不是。驻留 warp 是 latency hiding 的条件之一，但提高 occupancy 可能牺牲
寄存器复用、引入 spill 或增加指令。K30 的改进发生在与 K10 相近 occupancy 下。
K32 overlap 总 occupancy 增加，却包含不做 FFMA 的 producer warp，计算 warp
容量没有增加，吞吐也未提高。要区分 active、eligible 和实际 issued warp。

### Q25 235 个寄存器是不是太多

不能只凭数值判断。K30 有 128 个 accumulator，加上 fragment、搬运及控制状态，
编译为 235 且无 spill，在本配置中仍能驻留两个 CTA。K31 降低寄存器的部分
版本更慢，说明寄存器减少同时改变了调度和访问组织。应该看资源台阶与吞吐，
而不是把低寄存器数本身当目标。

### Q26 launch bounds 的 minBlocks 是运行时保证吗

不是。它给编译器寄存器分配提供约束/优化信息，不保证每个 SM 始终有那么多
block。实际驻留还取决于 shared、线程、寄存器分配粒度、可用 grid 和运行状态。
应结合编译资源、CUDA occupancy 查询及 Nsight 实测，不能只看模板参数。
语义见 [CUDA 12.4 Launch Bounds](https://docs.nvidia.com/cuda/archive/12.4.1/cuda-c-programming-guide/index.html#launch-bounds)。

### Q27 K32 删除 barrier 为什么更慢

源码删一次 barrier 后，编译器将寄存器从 167 分配到 187；五-warp CTA 在本机
跨过寄存器子分区容量阈值，可驻留 CTA 从两个降到一个，计算 warp 容量减半，
issue/FP32 active 明显下降。静态追踪发现尾部 accumulator 重命名变化，计算
数据存活峰值仅 155→159，不能说算法多需要 20 个数据。分配器的完整决策还未解释。

证据见[K32 第 8 节](19-K32-Warp角色分工与同步代价.md)。不能由此推导“barrier
越多越快”，也不能把所有性能损失都精确归因于驻留，因为指令调度同时变化。

### Q28 如何区分 SASS 静态计数和动态工作量

静态计数是二进制里有多少条指令；循环、谓词和互斥路径会改变实际执行次数。
K32 serial 的静态 FFMA 更多，但有互斥计算路径，不能说它多算了一遍矩阵。
动态计数还要区分 warp 指令、有效 thread 指令及 predication，不能直接混用。
同样，LDS 到 FFMA 相隔几十条静态指令，不等于几十个时钟周期。

### Q29 Issue 没提升但 eligible 增加说明什么

Eligible 是具备发射条件的 warp 数量，不保证每个都能同时发射；还存在执行管线、
发射选择及指令组成差异。K32 增加 producer 也改变了 warp 组成。
`no-issue` 表示没有指令发射的周期，不能直接改名为 `no eligible warp`。
要把指标单位与最终吞吐一起看，不能仅凭一个数增加就宣布成功。

### Q30 剩余百分之三到四究竟来自哪里

尚未完成精确归因。已经观察到 K30 与 cuBLAS 的 issue、等待结构和访存行为
仍有差异，但这些 counter 不能相加成剩余耗时百分比。K32 降低 long-scoreboard
仍未更快，反证了只追一个 stall 的不足。可靠结论是当前数据组织与编译调度
仍有差别，不是已经证明 cuBLAS 靠 warp specialization 获胜。

## 七 工程判断与现场复现

### Q31 怎样把它变成生产可用的算子

先定义真实工作负载与接口，再覆盖矩形、tail、stride、batch、对齐与别名契约，
增加更广输入分布和独立数值参考；还要验证不同架构、流并发及端到端收益。
可能需要多套 kernel 和 dispatch，而不是让一个固定 tile 处理所有问题。
这些是后续工程方向，当前项目并未完成，不能写成已有成果。

### Q32 如果给你一天继续优化 你会做什么

不会继续枚举 tile 或 stage。我会选择一个具体假设，例如 K32 单 barrier
版本的尾部 accumulator 分配能否在无 spill 下保持两个 CTA 驻留，先做静态
与编译对照，再决定是否上 GPU。若没有可检验的假设，优先完善验证/工作负载
覆盖。目前项目已暂停，这只是将来恢复时的研究提案，不是新增收益。

### Q33 为什么决定停止而不是继续冲到百分之百

后续局部实验的收益低于噪声或明确回退，继续叠技巧会增加复杂度和解释成本。
保留稳定 K30、整理反证和复现流程，比把偶然峰值包装成突破更有价值。
工程目标不是无期限追一个分数，而是在收益、风险、可维护性和研究成本之间做判断。

### Q34 面试官要求当场证明一个结论 怎么办

先打开 [K30 七轮 CSV](../results/k30-20261013/performance.csv)，算出
`20801.161857/19652.211857-1` 和 `20801.161857/21592.726857`，再展示独立确认。
然后打开 K30 compute/同步源码，指出两槽与 barrier 生命周期。若现场有合适 GPU，
按复现手册跑新目录；没有 GPU 就明确只能展示已有证据，不能声称现场复现。

### Q35 如何回答 AI 辅助开发的追问

按事实说明使用范围，包括代码草拟、脚本、分析或文档辅助；不要把不会解释的
生成代码包装成个人熟练掌握。能现场推导映射、检查同步安全性、解释实验反例，
并独立跑通验证，才是可展示的能力。尚不理解的细节如 ptxas 内部分配策略，
可以明确说当前只定位到外部证据，没有假装完全掌握。

## 八 二十分钟模拟面试安排

|时间|面试任务|合格信号|
|---|---|---|
|0–2 分钟|项目介绍和贡献边界|说清硬件、shape、精度、基线、开源关系|
|2–6 分钟|白板画 lane 输出和两槽同步|数量、坐标和生命周期均能解释|
|6–10 分钟|解释 K30 成功及 V1 反例|不把组合相关性包装成单因素因果|
|10–14 分钟|讨论 K32 和寄存器阈值|区分逻辑存活、物理分配、实际驻留|
|14–18 分钟|检查 CSV 与复现命令|不混合历史口径，不拿 profiler 时间排名|
|18–20 分钟|讨论局限与下一步|承认未解决部分，提出窄而可证伪的假设|

自测不要只给自己打“回答过”的勾。至少完成三件事：不用文档画一次 mapping；
从原始 CSV 手算一次提升；解释一次失败实验为何不能永久排除整个方向。
如果只能背出 96.3%，还没有准备好把这个项目作为简历的主要技术亮点。

## 九 RTX 4090 D Tensor Core 补充面试题

以下回答限定为 Windows / sm_89 的原生 CUDA 复现与 BK32、BN64 实验。
先读[第 32 章](32-RTX4090D-TensorCore优化学习指南.md)和
[微教材第 9 章](微教材/09-TensorCore从SIMT到矩阵指令.md)。Q1–Q35 的旧测量和
Sanitizer 结论不能直接套到这些新 kernel 上。

### Q36 输入输出都是 float 为什么还叫混合精度

要分别描述存储、乘数精度、累加和输出。TF32 路线把量化后的输入放在 float 存储中，
`mma...f32.tf32.tf32.f32` 使用 TF32 乘数，FP32 累加及输出。FP32 累加不会补回输入
量化时丢失的信息。严格 FP32 结果另作参考；本次 Tensor 实现相对原始 FP32 的
绝对 0.01 和 SIMT allclose 均失败，不能把性能优势写成保持原精度的加速。

### Q37 Tensor Core 指令是每线程算一个 16×8 矩阵吗

不是。本实现的 `m16n8k8` 是整个 warp 协作的矩阵乘加；每 lane 持有其中的 operand
与 accumulator fragment。一个 16×8 输出有 128 个元素，32 lane 每 lane 持有 4 个。
这是一次 warp 级运算的输出分工，不是 32 份独立的 16×8 结果。warp 参与要求以
[PTX MMA 语义](https://docs.nvidia.com/cuda/archive/12.4.1/parallel-thread-execution/index.html#warp-level-matrix-instructions-mma)为准。

### Q38 请手算 BN64 中一个 lane 的输出

CTA 为 128×64，8 warp 按 2×4 排列。设 `w=tid/32,l=tid%32`，
`mi=0..3,ni=0..1,r=0..3`，CTA 内坐标为：

```text
row = (w/4)*64 + mi*16 + l/4 + (r/2)*8
col = (w%4)*16 + ni*8 + (l%4)*2 + r%2
```

warp0、lane1、mi=ni=0 的四个输出是 (0,2)、(0,3)、(8,2)、(8,3)。
全部 fragment 共 `4*2*4=32` 个结果；256 线程覆盖 8192 个输出。
数量相等不证明无重叠，还要用 CPU 枚举查重复、遗漏和越界。

### Q39 ldmatrix.b16 是不是说明 kernel 实际用了 FP16

不能这样判断。在此代码中，`ldmatrix.x4.b16` 按半字组织搬运 A 的位模式，
随后接的是 TF32 MMA；B 则从 shared 加载到位模式寄存器。输入转换、加载组织方式、
数学乘法类型是三个环节。判断精度应检查转换及 MMA 类型，判断真实计算路径还要
看最终 SASS；这里已保存 HMMA 与 LDGSTS 证据，不只看文件名含不含 Tensor。

### Q40 有 cp.async.wait_group 为什么还需要 __syncthreads

等待异步组完成和协调全 CTA 是不同责任。BN64 先为当前线程的 copy 等待完成，
再通过 CTA barrier 让跨线程消费者安全使用新 tile，并保证旧槽所有读者退出后
才复用。两个 buffer 只隔开当前读地址与下一次写地址，不能自动实现发布与回收。
等待指令的作用范围见 [PTX cp.async.wait_group](https://docs.nvidia.com/cuda/archive/12.4.1/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-wait-group-cp-async-wait-all)。

### Q41 BK32 减少同步为什么在 2048 和 4096 反而慢

原版每 CTA 的 barrier 为 K/16，BK32 为 K/32，但 shared 32→64 KiB、
寄存器 127→170，理论 CTA/SM 2→1。独立确认相对原版配对中位数分别为
−4.75%、−6.45%。计数器显示 eligible warp 和 Tensor 活跃指标下降，支持资源
代价抵消控制开销收益的解释；不能把几种同时变化的因素精确分解成耗时占比。

### Q42 BN64 快起来是不是只因为 occupancy 增大

不能说“只因为”。相对父 BK32，BN64 将 shared 降到 48 KiB、寄存器降到 102，
恢复理论 2 CTA/SM；相对作者 BK16，理论 CTA/SM 本来就是 2。BN64 同时减半
每 CTA 输出、翻倍 CTA 数、改 B 搬运及 fragment、增加 A 的重复请求。
2048 的同批确认是 57.998 TFLOPS，相对原版 +19.56%，是这一组合的实测结果。

### Q43 Tensor active 几乎不变为什么仍能明显加速

先问分母。2048 的原版→BN64，Tensor active 为 44.25%→44.49%，
但 SM 活跃周期占经过周期从 75.17%→89.49%，Tensor elapsed 从 33.26%→39.82%。
更细的 CTA 任务分配减少部分空闲，是与这些指标一致的解释；没有逐 SM 调度轨迹，
不能证明全部收益都来自尾波。4096 的活跃时间比例原本已约 99%，可改善空间不同。

### Q44 为什么 4096 的 L2 命中率更高却没有继续变快

命中率不是总流量。BN64 使逻辑 A 请求翻倍，B 请求不变；两者合计约 1.5 倍。
4096 L2 命中率 94.72%→95.65%，DRAM 读取仍从 364268160→493347456 字节。
同批确认对原版配对中位数 −0.43%，仅 2/7 轮胜出，没有稳定提高的证据。
这既不能证明 L2 毫无价值，也不能把更高命中率直接换算成加速。

### Q45 SASS 中 copy 与 HMMA 交错就证明完全隐藏访存了吗

没有。BN64 主循环最后一次 `LDGSTS` 与完成等待 `DEPBAR.LE` 之间有 27 条 HMMA，
说明编译后的指令确实交错；静态间隔不是实际延迟，也不表示等待必为零。
还要结合依赖、动态 stall 和最终耗时。`LDGDEPBAR` 是提交相关指令，不能因名字
含 DEPBAR 就当成完成等待；profile duration 也不能放进公平 event 成绩表。

### Q46 “超过 cuBLAS”必须附带哪些条件

本例只能说：在 Windows RTX 4090 D、2048 方阵、同一显式 TF32 输入、FP32 累加及
输出、C=AB 的七轮确认中，BN64 相对 **TF32 cuBLAS** 的配对提升中位数为 11.57%，
7/7 轮胜出。每轮10次预热、30次 event；输入转换与同 stream 的 C0 恢复均在 event 外。
1024 仍落后库，不能省掉尺寸；原始 FP32 阈值失败，不能省掉精度；不能用 profiler 时间替代。

### Q47 逐位一致、CPU 枚举、double 参考分别证明了什么

相对作者原版的 156 次全量逐位比较零差异，说明这些已测输出相同；CPU 枚举证明
映射公式的覆盖，未模拟所有硬件竞争。double 参考在小矩形全量、大方阵只检查固定
256 点。它们互相补充，但不能代替 Sanitizer；本轮 Sanitizer 未执行。
相对量化输入的严格参考，在 2048/4096 仍有绝对 0.01 和 SIMT 判据失败，不能省略。

### Tensor Core 方向的现场自检

用五分钟完成：手算 Q38 的一个 lane；从 [paired-rounds.csv](../../results/windows/20261021-tensorcore-bn64/confirm-analysis02/paired-rounds.csv)
重算 2048 同轮提升；对照 [elapsed-selected.csv](../../results/windows/20261021-tensorcore-bn64/evidence02/elapsed-selected.csv)
解释 Q43；最后说明 Q47 中未验证的部分。所有历史数字以各自同批 CSV 为准，不做跨 GPU 拼表。

## 十一 CUTLASS 对照后的追加追问（2026-10-22）

以下使用已发布[第34章](34-CUTLASS与BN64的4090D公平对照.md)的数据；Q36–Q47的历史BN64例子保持原批次含义。

### Q48 能说我的 Tensor Core kernel 超过 CUTLASS 和 cuBLAS 吗

不能概括成一个无条件排名。本次官方CUTLASS128×64×16/4-stage在四尺寸快于BN64，2048相对BN64配对提升8.095%～8.803%；相对同输入TF32 cuBLAS提升21.307%～21.762%。这说明当前BN64还有差距，不代表CUTLASS全部模板或所有矩阵都如此。严格FP32组的CUTLASS SIMT未超过K30或cuBLAS；两个精度组不能混算。

### Q49 为什么 occupancy 更低且 bank conflict 更多仍可能更快

性能还取决于需要发射的指令和总shared工作量。2048的CUTLASS小配置occupancy14.87%，BN64为29.87%；前者shared-load wavefront更少、barrier stall更低、Tensor elapsed更高。多个因素同时变化，所以它是相关证据，不是“低occupancy会提速”的证明。下一步应固定BN64配置，只测试operand寄存器复用能否减少实际shared读取，并检查寄存器代价。

### Q50 审计 PASS、误差 FAIL 与成功发布为什么能同时出现

数据审计验证的是记录完整、输入相同、统计正确和恢复链可信；数值检查则有多套独立标准。该批strict FP32的abs<=0.01通过，但3072的SIMT allclose失败；TF32 native标准通过，较大尺寸严格绝对门限失败。发布完整失败证据有助于复现，不等于把失败改成成功。也不能从六进程完整结束推断Sanitizer通过，本轮没有完成该验证。

继续手算见[34.10的五道练习](34-CUTLASS与BN64的4090D公平对照.md#3410-五道自检题与参考答案)，题目覆盖warp分区、异步等待、误差参考与配对统计。
