# K32：Warp 角色分工能否缩小最后的差距？

日期：2026-10-13。环境仍为同一 RTX 3080 / SM86 / CUDA 12.4；GPU UUID 与 K30/K31 一致。
K30 V2 是当前最佳研究基线，K10 仍为默认教学入口。本轮不修改二者，不改精度，
不搜索寄存器预算、TM/TN、stage 数量或 CTA 大小。

## 1. 为什么值得试，又为什么不能预先宣布答案

[K31](18-K31-微内核精修与因果审计.md)已定位到 K30 主循环末尾的具体 LDG→STS
等待：load 到消费它的 store 之间只有两条 FFMA，store 上有集中的 long-scoreboard
样本。这支持一个窄假设：让专职搬运 warp 与计算 warp 同时推进，能否避免计算 warp
自己等待下一 tile 的 global load？

它不证明 cuBLAS 采用专职 producer，也不证明 K30 的所有 shared、iterator、pipeline
问题均已解决。过去实验只能否定已经测试的实现。本轮目标是测试角色分工的净收益，
不是用“Warp Specialization”这个名字替代资源与同步分析。

## 2. 固定计算分工，只新增一个搬运 warp

```text
128×128 CTA，BK=16，两槽 shared ring，共 33,280 B
  ├─ warp 0 / threads 0..31：搬运 A/B，普通 LDG + STS
  └─ warps 1..4 / threads 32..159：复用 K30 的四个计算 warp
       └─ 每 warp 64×64；每 lane A16×B8；128 accumulators
            └─ K30 compute 与 epilogue 原样复用，K 顺序不变
```

总线程数 160，不是 K30 的 128。仅“消费者逻辑映射”不变：传入 FragmentIterator
的是 `threadIdx.x-32`。不能把增加 producer 说成线程数量或全局加载分工完全没变。

设计算 warp 索引 `w=(tid-32)/32`、lane `l=tid%32`：

```text
row = 64*(w/2) + 4*(2*(l/16)+l%2)
col = 64*(w%2) + 4*((l%16)/2)
m(i) = row + 16*(i/4) + i%4       i=0..15
n(j) = col + 32*(j/4) + j%4       j=0..7
acc[i*8+j] -> C[blockM+m(i), blockN+n(j)]
```

producer 的 lane `l` 搬运 A 时，`m=l/4+8*t`、`k=4*(l%4)`，`t=0..15`，
每次 float4 读取后按 K30 的 `As[k*132+m]` 转置写入。搬运 B 时，每个 k 一条
float4：`B[k*N+4*l] -> Bs[k*128+4*l]`。访问坐标、float4 对齐及 shared 物理布局
不变，但一个 producer warp 承担了原先四个 warp 的搬运工作，不能认为搬运成本免费。
没有 cp.async、TMA、Tensor Core、WMMA 或动态寄存器再分配。

CPU 测试 [`test_k32_protocol.py`](../../scripts/test_k32_protocol.py)枚举 producer
完整且无重复的 A/B 覆盖，以及四个 consumer 对 16,384 个 C 元素的完整独占覆盖。
它不替代 GPU 正确性或 Sanitizer。

## 3. 三个版本与同步安全性

独立源码：[`32_kernel_warp_specialized.cuh`](../../src/kernels/32_kernel_warp_specialized.cuh)。
前两版先比较，第三版只在确认 overlap 与 K30 持平后加入，研究冗余同步的实际代价。

|模式|模板参数|每个 K tile 的顺序|
|---|---|---|
|serial|`<false>`|producer 写 next → barrier 1 → consumers 算 current → barrier 2|
|overlap|`<true>`|producer 写 next 与 consumers 算 current 可并发 → barrier 1 → barrier 2|
|single|`<true,true>`|与 overlap 相同，但去掉末尾冗余 barrier 2|

serial/overlap 的 barrier 数量相同，避免把少一次同步冒充重叠本身的收益。
overlap 的第二次同步是实验性对照，不是声称最优算法需要连续两次 barrier。
single 是对这个人为成本的收尾检查，而不是另加一套复杂流水。

初始化时仅 producer 写 tile 0，全部 160 threads 经一次 `__syncthreads()` 发布。
迭代 t 中，consumer 读 `t%2`，producer 写 `(t+1)%2`，二者必不同。
双 barrier 版的 barrier 2 结束后，下一轮才允许回收当前 slot；single 的 barrier 1
同时完成 future 发布和 current 读者退出。最后一轮不加载越界 future，但所有角色
仍完成同样的同步。producer 退出后 consumer 只写自己独占的 C，无后续跨角色同步。

角色分支中使用非 `.aligned` 的 `barrier.sync id,160`，显式 160 个参与线程，
并带编译器 memory clobber。不能直接把要求 CTA 分支一致的 aligned barrier 放入
不同角色路径。[CUDA 12.4 PTX barrier 规范](https://docs.nvidia.com/cuda/archive/12.4.1/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-bar)
定义了参与计数、阶段复用和内存可见性。本实现没有自旋 flag 或需人工维护 phase 的 mbarrier。

CPU 模型对 1/2/3/4 个 tile、五个独立 warp 的全部可达事件交错检查：读取必须已就绪、
计算尚未结束的 slot 不能被覆盖、barrier 不可死锁。正式 GPU 验证范围更大，见后文。

## 4. 编译器先改变了什么

第一版分析器假设每个 compute backedge 的地址区间恰好只有 2,048 条 FFMA，遇到
serial 的 4,096 条便停止。初步猜测为外层展开；显式加入 `#pragma unroll 1` 后，
两版完整 SASS 哈希都没有变化，反证了这一猜测。实际存在由 `BRA.CONV` 选择的
两套计算路径，不是每轮把相同乘加重复做两遍。分析器已改为报告线性地址区间，
明确它不是动态执行路径或 latency。

后续动态计数再检查数学工作量，而不是从静态重复段推断性能原因。最终构建与
首轮共有的 K10/K30/serial/overlap 四个 kernel 的 SASS 哈希全部一致；新增 single
没有暗中改变对照。原始构建目录分别保留，没有覆盖旧文件：

- `benchmark_results/k32-20261013`：最初编译与 128-size 冒烟检查，无正式排名；
- `benchmark_results/k32-20261013-u1`：双版本五轮 pilot，结果保存至
  [pilot 目录](../results/k32-20261013-pilot)；
- `benchmark_results/k32-20261013-final`：增加 single 后冻结源码，三版完整重验与六轮比较。

### 寄存器容量必须按 SM 子分区计算

|模式|Threads|Registers/thread|Spill/stack|用户 shared B|CUDA 查询 blocks/SM|计算 warp 容量/SM|
|---|---:|---:|---|---:|---:|---:|
|K10|128|184|0|16640|2|8|
|K30|128|235|0|33280|2|8|
|serial|160|181|0|33280|1|4|
|overlap|160|167|0|33280|2|8|
|single|160|187|0|33280|1|4|

167 是实际编译结果，不是人为搜索/限制的上限。角色分离去掉了 consumer 的 global
fragment 搬运职责，寄存器数量大幅下降，但这些寄存器仍按整个 kernel 的分配规则
占用，不能按 producer 的实际变量数单独扣减。

CUDA 12.4 自带 `include/cuda_occupancy.h` 的
`cudaOccMaxBlocksPerMultiprocessorRegsLimit` 按四个寄存器子分区计算容量：
每子分区 16,384 个寄存器、每 warp 分配按 256 个寄存器取整，然后求每子分区可容纳
的完整 warp。对本机配置：

|模式|每 warp 实际分配|每子分区可容纳 warp|寄存器允许的 SM warp 总数|每 CTA warp|
|---|---:|---:|---:|---:|
|K30|7680|2|8|4|
|serial|5888|2|8|5|
|overlap|5376|3|12|5|
|single|6144|2|8|5|

所以 `floor(65536/(160*181))=2` 是错误的驻留预测；子分区最多容纳 8 个此类 warp，
不够两个 5-warp CTA。实际资源查询确认 serial/single 为 1 block、overlap 为 2。
这也说明：删除一条源码 barrier 不等于只删一条 SASS，编译器的重新分配可能跨过
资源阈值。本轮如实保留这个混杂因素，不把 serial→overlap 的全部加速归因于重叠。

## 5. 性能与 Nsight 结果

### 公平计时：重叠版两轮实验都与 K30 持平

统一 4096³、alpha=0.5、beta=3、seed=20261013；每次恢复同一 C0，copy 不计入
event。strict 为 `PEDANTIC_MATH + COMPUTE_32F_PEDANTIC + GEMM_DEFAULT`，清除
`NVIDIA_TF32_OVERRIDE`。各模式 warmup=10、repeat=50，GPU 测量串行执行。
最终六模式 × 六轮，完整轮换每个执行位置。范围是各轮均值范围，不是单 launch 极值。

|模式|最终平均 GFLOPS|各轮范围|标准差|平均 ms|相对 K30|strict 比例|
|---|---:|---:|---:|---:|---:|---:|
|strict|21606.37|21565.86–21620.78|20.32|6.361044|—|100.000%|
|K10|19664.40|19616.99–19742.39|45.48|6.989259|−5.694%|91.012%|
|K30|20851.74|20810.44–20870.31|22.07|6.591251|基线|96.507%|
|serial|13043.23|12889.01–13195.44|116.83|10.537891|−37.448%|60.368%|
|overlap|20842.38|20804.02–20988.68|72.82|6.594273|−0.045%|96.464%|
|single|18424.05|18390.05–18447.39|24.82|7.459770|−11.643%|85.271%|

数据：[最终六轮明细](../results/k32-20261013-final/fair.csv)、
[统计](../results/k32-20261013-final/performance.csv)。不能把 overlap 的最高一轮
20,988.68 当作新最佳：六轮中只有一轮快于同轮 K30，配对范围 −0.301%～+0.856%。

独立首轮五模式 × 五轮为 strict 21,611.08、K30 20,791.37、serial 13,107.06、
overlap 20,781.91 GFLOPS，overlap 对 K30 同样是 **−0.046%**。
这不是证明 overlap 确定慢了 0.05%，而是两轮都没有稳定提升，按噪声内持平处理。
本轮 K30 的 96.51% 是重新测量的同轮比例，不是对未改代码的 K30 宣布新优化。

### Nsight：一个等待指标改善，整体吞吐仍可能不变

下面为 cold kernel replay、cache=all、clock=base。eligible 是每 scheduler 平均
eligible warps；四种 stall 数值为平均 warp/active-cycle，不是时间百分比。
profile duration 不作为公平性能结果。

|模式|Issue %|FP32 active %|Occupancy %|Eligible|Long SB|Short SB|Barrier|Wait|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|strict|87.319|78.582|16.663|1.579|0.0236|0.0830|0.0955|0.0122|
|K10|79.136|73.538|16.535|1.356|0.0856|0.0295|0.1121|0.0892|
|K30|83.312|77.385|16.472|1.391|0.1762|0.0515|0.0718|0.0674|
|serial|46.968|44.043|10.387|0.470|0.0510|0.1233|0.4806|0.0638|
|overlap|82.595|77.604|20.289|1.686|0.0221|0.0678|0.2341|0.0898|
|single|70.492|66.259|10.381|0.829|0.0174|0.0360|0.1391|0.0569|

overlap 的 long-scoreboard 比 K30 降约 87.5%，eligible 与总 active warps 上升，
但 issue 反而下降，FP32 active 只略升，benchmark 持平。新增的 active warp 包含
不执行 FFMA 的 producer，不能把总 occupancy 从 16.47% 升到 20.29% 说成计算
并行度同比增加。两个版本的理论计算 warp 容量仍都是 8。

稳态 application replay（cache=none、clock=none）维持同一方向：

|模式|稳态 Issue %|FP32 active %|Long SB|Barrier|
|---|---:|---:|---:|---:|
|strict|87.244|78.461|0.0239|0.0959|
|K30|83.057|77.039|0.1968|0.0712|
|overlap|82.470|77.524|0.0269|0.2338|
|single|69.959|65.897|0.0230|0.1390|

两种 policy 各模式采集一次，不能给微小 counter 差异附加统计显著性。
完整 no-issue、dispatch、local memory 等在
[nsight.csv](../results/k32-20261013-final/nsight.csv)。

|模式|cold DRAM read GB|L2 hit %|整核 shared-load waves M|整核 load conflicts|
|---|---:|---:|---:|---:|
|strict|0.517|87.780|201.900|0|
|K30|0.811|80.726|201.347|20311|
|serial|1.179|73.390|201.327|0|
|overlap|1.382|68.779|201.420|93113|
|single|1.256|71.640|201.327|0|

布局虽然没变，整核 hardware counter 与 runtime/cache 行为仍会改变；不要因此
虚构新 bank 映射。逐 PC SourceCounters 中，K30/K32 读取仍为 201,326,592 waves、
LDS excessive=0。保留整核与逐 PC 两套口径，不把后者零值覆盖前者非零值。
DRAM 观测变差也是成本线索，但没有单独隔离缓存因素，不能把性能差额数值归因于它。

### SASS / 动态工作量：没有通过少算乘加换分

|模式|静态指令|静态 FFMA|静态 LDS|静态 BAR|IMAD|IADD3|MOV|
|---|---:|---:|---:|---:|---:|---:|---:|
|K30|3360|2304|96|2|58|120|21|
|serial|5688|4352|192|7|102|115|31|
|overlap|3544|2304|96|7|100|115|31|
|single|3536|2304|96|4|100|115|25|

K32 全部 HMMA/MMA/LDGSTS/DEPBAR/LDL/STL 为零。serial 两份计算路径中实际只执行
一份：全核实际执行 FFMA 位置为 2,176、LDS 位置为 96，与 K30/overlap/single 相同。
所有版本的有效 thread-FFMA 都是 68,736,253,952（包含 epilogue）。

|动态 warp 指令|K30|serial|overlap|single|
|---|---:|---:|---:|---:|
|FFMA|2148007936|2148007936|2148007936|2148007936|
|LDS|100663296|100663296|100663296|100663296|
|LDG|8519680|8519680|8519680|8519680|
|STS|20971520|20971520|20971520|20971520|
|BAR|1052672|2626560|2626560|1315840|
|IMAD|16826368|10813440|10813440|10813440|
|IADD3|10706944|2841600|2841600|2841600|

serial/overlap 的动态 BAR 确实相同；single 减少了约一半 BAR，却因编译资源与
调度改变而明显回退。single 即使每 tile 一次同步，总 BAR 仍比 K30 多，因为参与
同步的是 5 个 warp，而不是 4 个。更多静态 STS 也不等于更多动态写入：同一份工作
被集中到更少的 producer threads，循环展开后的代码体更大。

逐 PC 定位保留了三条可核验线索：

- K30 `0x9690 STS [R58.X4],R68` 仍有 13,145 个 long-scoreboard samples。
- overlap 的主要采样位置变为 consumer 尾部 `0x98d0 BAR.SYNC 0x2,0xa0`，
  有 23,304 个 barrier samples。
- single 的 barrier 后条件 CALL `0x9890` 有 94,072 个 barrier samples，
  与其较低驻留容量和较差 issue 一起观察。

这些是不同运行中的抽样数，不是时钟周期或损失百分比。barrier 阻塞也可能体现在
后继 PC，不能认为采样所在指令本身制造了全部等待；更不能将两种 stall 相减后
声称“等待时间等量转移”。数据与对应机器码见
[动态指令](../results/k32-20261013-final/source/dynamic-opcodes.csv)、
[逐 PC stall](../results/k32-20261013-final/source/pc-stalls.csv)、
[shared 访问](../results/k32-20261013-final/source/memory-instructions.csv)、
[静态 SASS](../results/k32-20261013-final/static-sass.csv)。

## 6. 验证、复现与证据边界

最终冻结构建：K10/K30/serial/overlap/single × 六尺寸
128/256/512/1024/2048/4096 × beta=0/1/3，共 **90/90 PASS**。
最大绝对误差 0.000610351562，阈值仍为 0.01，拒绝非有限值。
四个研究模式 K30/serial/overlap/single 的 memcheck（128/256/4096）为 **12/12 PASS**，
racecheck/synccheck（128/256）为 **16/16 PASS**，合计 **28 项**；所有 NCU local
load/store sectors 为零。pilot 的 72 项正确性、21 项 Sanitizer 单独保留，不拿来
代替最终构建验证。

3 组 CPU 覆盖/同步模型测试、4 组 Sanitizer 日志判据回归、Python/shell 语法检查
及 CMake Release `k32_audit` 构建通过。纯翻译检查仍固定 `aadcd8e`，35 个原文件
失败数为零。K30/K10 与主 dispatcher Git Diff 为零。

记录：[正确性](../results/k32-20261013-final/correctness.csv)、
[Sanitizer 摘要及日志哈希](../results/k32-20261013-final/sanitizer.csv)、
[编译器报告](../results/k32-20261013-final/ptxas.txt)、
[运行时资源](../results/k32-20261013-final/runtime-resources.csv)、
[binary/source SHA256](../results/k32-20261013-final/sha256.csv)。环境记录中的
Git HEAD `9cbe413` 是实验前基线；本轮代码当时在工作区，不冒充旧提交已经包含 K32。

在仓库根目录串行复现，选择新的输出目录：

```bash
OUT=benchmark_results/k32-reproduce-unique
REPORT=docs/results/k32-reproduce-unique
bash scripts/build_k32_audit.sh "$OUT"
python3 scripts/analyze_k32_binary.py "$OUT"
python3 scripts/test_k32_protocol.py
python3 scripts/run_k32_audit.py resources --directory "$OUT"
python3 scripts/run_k32_audit.py correctness --directory "$OUT"
python3 scripts/run_k32_audit.py memcheck --directory "$OUT"
python3 scripts/run_k32_audit.py sync --directory "$OUT"
python3 scripts/run_k32_audit.py fair --directory "$OUT" --rounds 6
python3 scripts/run_k32_audit.py profile --directory "$OUT"
python3 scripts/run_k32_audit.py source --directory "$OUT"
python3 scripts/run_k32_audit.py profile --directory "$OUT" --policies steady \
  --modes strict k30 overlap single
python3 scripts/summarize_k32_audit.py "$OUT" --output "$REPORT" --rounds 6 \
  --steady-modes strict k30 overlap single
python3 scripts/analyze_cutlass_reference_source.py "$OUT/source" \
  --modes strict k10 k30 serial overlap single --skip-mapping --output "$REPORT/source"
```

计时/采集复用 K30 的严格 harness 协议，不修改旧脚本行为。汇总前检查完整矩阵、
轮次数量、Sanitizer 摘要及指定稳态文件；已有结果只允许等值核对，不覆盖历史。
本地原始 binary、完整日志、`.ncu-rep` 保留在 benchmark_results，不提交大二进制。
独立 CMake 为 `-DCMAKE_BUILD_TYPE=Release -DSGEMM_BUILD_K32_AUDIT=ON`。

支持范围仅为 harness 声明的方阵、128 的整数倍、128–4096、非别名且正确对齐的
FP32 A/B/C。不声称覆盖任意 M/N/K 尾部、stride、别名或所有数值分布。

## 7. 本轮回答与停止条件

**本轮没有新的最佳实现，继续保留 K30。** 但它不是没有信息量的失败：

1. **角色分工确实可以改变等待结构。** overlap 的 long-scoreboard 明显下降，
   consumer 不再自己搬运 global tile；同时 barrier/wait 上升、issue 未改善，
   最终吞吐持平。降低一个等待指标不足以解释或消除剩余 3%–4%。
2. **“额外 producer 一定挤掉计算 warp”也不成立。** overlap 编译得到 167 regs，
   保住了两个 CTA 与 8 个计算 warp。反过来，181/187 regs 的 5-warp block 受
   子分区限制只能驻留一个。资源风险必须编译、查询，不能只按 65,536 总量估算。
3. **匹配源码变量不等于完全隔离硬件因果。** serial/overlap 的 barrier 数相同，
   但编译后的寄存器、驻留与控制流不同；不能把 13.04→20.84 TFLOPS 全部归因于
   实际重叠。single 的负收益也不能解释成“barrier 越多越好”。
4. **没有证据认定 cuBLAS 的关键就是 Warp Specialization。** 在本轮工作量下，
   cuBLAS 仍为约 21.61 TFLOPS，K30/K32-overlap 都约 20.85 TFLOPS。
   其更高 issue 与更成熟的搬运/消费组织依然需要解释，不能靠名称或类比补出私有架构。

因此停止本轮，不增加 producer 比例、不改 CTA、不搜寄存器、不加新 stage。
随后完成的静态追踪见第 8 节。它定位了 accumulator 重命名，但尚未完全解释
ptxas 的分配决策。当前按项目决定暂停优化，不继续实施寄存器限制或新 kernel。
若将来恢复，应先给出能保持原资源容量的具体代码假设，再决定是否值得新开实验。

这轮故事的核心是：**把加载等待从计算线程移开，只是微内核设计的一部分；交接协议、
编译器分配和资源粒度必须同时可用，才可能把指标改善兑现成性能。**

## 8. 收尾静态分析定位 accumulator 重命名

这部分使用已经测量过的 final 二进制，没有重新编译 kernel，也没有增加性能数据。
`nvdisasm -lrm count` 的全程序存活区间会涉及分支及编译器生成的 barrier helper；
为了限定推论，另对第一个 LDS 到尾部 `BRA.CONV` 之间无分支的主计算段做逐值追踪。
工具只接受 2,048 条 FFMA、96 条 LDS.128、128 条累加链各更新 16 次这一模式，
不符合时中止，不猜测任意 SASS 的语义。

|计算段指标|overlap|single|
|---|---:|---:|
|Accumulator 数量|128|128|
|Accumulator 加 fragment 同时存活峰值|155|159|
|Fragment 同时存活峰值|27|31|
|FFMA 的输出与累加输入寄存器不同的次数|73|103|
|LDS 向量到首次使用的指令距离中位数|35|50|
|同一 accumulator 相邻更新距离中位数|134|134|
|ptxas 分配 registers/thread|167|187|

这里的存活峰值按指令边界统计：最后一次读取后释放旧值，所有 128 个最终
accumulator 作为 live-out 保留。不包括地址、控制流、执行中的 scoreboard 约束，
不是整个 kernel 所需物理寄存器数。指令距离不是时钟周期，也不是动态延迟测量。

single 的新增高编号 `R168/R170/…/R184` 主要保存第 14/15 次更新后的 accumulator，
不是新增加的 global 预取 fragment。例如：

```text
0x8ed0  FFMA R184, R19.reuse, R4, R59
...
0x97c0  FFMA R60, R63.reuse, R44, R184
```

同一条累加链从 R59 换到 R184，再换到 R60。两次指令索引相差 143，不能说成
“有 143 cycles 延迟”。两版均未增加数学工作量；single 使用更多非原位更新，
尾部寄存器映射和 LDS/FFMA 顺序均发生变化。

因此要修正“少一次 barrier 导致多 20 个数据同时存活”这种过强解释：已追踪的
数据峰值只增加 4，而分配增加 20。也不能把差值 16 全算成浪费；不同时间的
存活区间、分配约束及控制流均可能影响分配，ptxas 的内部决策仍未完全确定。
目前可证实的是分配越过驻留阈值，不能量化全部吞吐损失的单一原因。

复跑已有分析，输出到新目录：

```bash
python3 scripts/analyze_k32_payload_liveness.py "$OUT" \
  --output docs/results/k32-static-reproduce-unique
```

`$OUT` 是第 6 节完成编译及反汇编的目录。无需 GPU；输入为其中的
`sass_overlap.txt`、`sass_single.txt`。保存结果含输入文件 SHA256：
[存活区间统计](../results/k32-20261013-static-analysis/payload-liveness.csv)、
[高编号寄存器定义与最后使用](../results/k32-20261013-static-analysis/high-register-values.csv)。
输入文件哈希含换行，与 `static-sass.csv` 中去掉末尾换行后的哈希口径不同。

阶段最终结论：K30 保留；K32 无可接受的新最佳；暂停继续调优，保留实验与反证供复现。
