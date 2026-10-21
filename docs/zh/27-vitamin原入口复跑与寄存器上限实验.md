# vitamin 原入口复跑与寄存器上限实验

> 归档可用性补记（2026-10-21，核对起点 `a54435d`）：本章引用的本地文件和目录均存在；本章、`20261021-vitamin-followup` 及部分配套脚本仍是未纳入 Git 的工作区文件，未随该提交发布。下方保留真实本地来源，历史测量值未改写。本轮 Tensor Core 多 seed 验证另见[第 33 章](33-TensorCore-BN64稳健性与尺寸适用范围.md)。

> 发布补记（2026-10-22）：本章、上述完整归档与配套脚本已纳入本次 RTX 4090 D 发布。上一条说明保留 2026-10-21 的核对状态；测量时的源码快照与原始结果没有改写。

本章承接[第26章](26-vitamin-SIMT复现与K30公平对照.md)，回答两个不同问题：作者公开的SIMT源码在本机是否重现了相对cuBLAS的领先；DBF的寄存器上限能否改善4096上的资源驻留与性能。第一项保留作者的Python测试函数和墙钟口径，第二项使用独立原生CUDA events实验，分别归档。第26章与前三阶段原始结果不改写。

## 1. 先把作者的历史数字与条件对齐

审计对象为 vitamin-cuda `zh-cn-beginner` 固定提交 `520f97093a2abc239977be83d244b327d0e2e2ef`。以下链接均固定到该提交和行号，不以当前网页后续变化替换历史内容。本次开始时上游工作区无修改。

### 1.1 原文有两组最终版本数字

| 记录 | GPU与尺寸 | torch均值 | 普通cuBLAS均值 | 最终DBF均值 | 来源 |
|---|---|---:|---:|---:|---|
| 文章开头叙述 | RTX 5060移动版，4096³ | 16.59 ms | 14.33 ms | 13.85 ms | [中文文章第13行](https://github.com/QiLHong/vitamin-cuda/blob/520f97093a2abc239977be83d244b327d0e2e2ef/posts/zh/sgemm_beating_cublas.md#L13) |
| 文章后部详细日志 | RTX 5060移动版，4096³ | 14.974799 ms | 14.523163 ms | 14.193397 ms | [设备说明及详细日志，第534～545行](https://github.com/QiLHong/vitamin-cuda/blob/520f97093a2abc239977be83d244b327d0e2e2ef/posts/zh/sgemm_beating_cublas.md#L534-L545) |

第二组日志也出现在 [`kernels/sgemm/readme.md` 第52～59行](https://github.com/QiLHong/vitamin-cuda/blob/520f97093a2abc239977be83d244b327d0e2e2ef/kernels/sgemm/readme.md#L52-L59)，是相同数字的重复记载，不应算成另一次独立实验。其五个SIMT版本完整时延为：

| 实现 | 作者详细日志均值 |
|---|---:|
| `sgemm_tiling` | 18.760985 ms |
| `sgemm_at_tiling` | 16.436968 ms |
| `sgemm_at_bcf_swizzling` | 15.706529 ms |
| `sgemm_at_bcf_swizzling_rw` | 15.522802 ms |
| `sgemm_at_bcf_swizzling_dbf_rw` | 14.193397 ms |

两组数字要分别保留。源码快照没有为它们提供同一批次标识或逐次样本，不能自行把13.85与14.523163拼成一组，也不能仅凭数字不同判断哪一组不真实。作者在第534行主动说明桌面程序与动态频率会让绝对时延波动。

原日志的`speedup:1.06`以torch作为比较基准：[`test.py` 第58～65行](https://github.com/QiLHong/vitamin-cuda/blob/520f97093a2abc239977be83d244b327d0e2e2ef/kernels/sgemm/test.py#L58-L65)把torch批次duration存入`baseline`，随后打印`baseline/duration`，即torch时延除以候选时延。因此详细日志的DBF相对torch吞吐约高5.51%，相对cuBLAS约高 **2.32%**，不是“比cuBLAS快6%”。后一个数对应约2.27%的时延下降。开头13.85/14.33一组对应的相对cuBLAS吞吐提升为约3.47%，仍属于另一组记录。

### 1.2 不能混入的其他日志

[`readme.md` 第144～149行](https://github.com/QiLHong/vitamin-cuda/blob/520f97093a2abc239977be83d244b327d0e2e2ef/kernels/sgemm/readme.md#L144-L149)还有一组4096³记录：torch16.741930、cuBLAS16.232075、`sgemm`17.583192、`sgemm_bcf`16.882191 ms。这里的旧函数名没有在该日志中映射到当前五个版本，不能将`sgemm_bcf`当作当前DBF。

[`readme.md` 第36～43行](https://github.com/QiLHong/vitamin-cuda/blob/520f97093a2abc239977be83d244b327d0e2e2ef/kernels/sgemm/readme.md#L36-L43)的8.275736 ms对应TF32 double-buffer实现；SIMT详细日志后的8.798057 ms则是cuBLAS TF32。两者均不进入本章普通FP32 SIMT比较。

### 1.3 能核实的条件与没有绑定到该次日志的条件

| 条件 | 固定提交提供的证据 | 本章如何使用 |
|---|---|---|
| GPU | 文章第13、534行均写RTX 5060移动版 | 明确属于另一款GPU |
| PyTorch | [README.en.md第52行](https://github.com/QiLHong/vitamin-cuda/blob/520f97093a2abc239977be83d244b327d0e2e2ef/README.en.md#L52)概括“PyTorch 2.9+” | 这是项目级说明，不是某条日志的精确补丁版本 |
| CUDA编译选项 | [`test.py` 第11～29行](https://github.com/QiLHong/vitamin-cuda/blob/520f97093a2abc239977be83d244b327d0e2e2ef/kernels/sgemm/test.py#L11-L29)：`-O3`、C++17、`--use_fast_math`、两个expt开关、半精度相关宏取消及ptxas报告 | 尽量保持原选项，Windows必要语法适配单列 |
| GPU架构选择 | [模块README第27～30行](https://github.com/QiLHong/vitamin-cuda/blob/520f97093a2abc239977be83d244b327d0e2e2ef/kernels/sgemm/readme.md#L27-L30)按nvidia-smi计算能力设置`TORCH_CUDA_ARCH_LIST` | 不能从该命令反推作者当时的完整实际编译命令 |
| 数学语义 | [`cublas.cu` 第19～22行](https://github.com/QiLHong/vitamin-cuda/blob/520f97093a2abc239977be83d244b327d0e2e2ef/kernels/sgemm/cublas.cu#L19-L22)：alpha1、beta0；自写核写C=AB | 本次保持同语义 |
| 普通cuBLAS | [同文件第43～62行](https://github.com/QiLHong/vitamin-cuda/blob/520f97093a2abc239977be83d244b327d0e2e2ef/kernels/sgemm/cublas.cu#L43-L62)：COMPUTE_32F、DEFAULT算法；借用PyTorch当前handle | 原路径没有显式设置PEDANTIC，不能把它写成严格B配置 |
| 输入与seed | [`test_4096` 第119～126行](https://github.com/QiLHong/vitamin-cuda/blob/520f97093a2abc239977be83d244b327d0e2e2ef/kernels/sgemm/test.py#L119-L126)：CPU randn转float32后.cuda，C为0；函数内没有seed | 本次seed在外部设置，属于新增可复现条件 |
| 计时 | [第35～55行](https://github.com/QiLHong/vitamin-cuda/blob/520f97093a2abc239977be83d244b327d0e2e2ef/kernels/sgemm/test.py#L35-L55)：默认10次预热、200次调用、perf_counter和前后同步 | 属于包含CPU派发的墙钟批均值，不是逐次CUDA event |
| 正确性 | [第69～74行](https://github.com/QiLHong/vitamin-cuda/blob/520f97093a2abc239977be83d244b327d0e2e2ef/kernels/sgemm/test.py#L69-L74)：非TF32 allclose atol=rtol=1e-3 | 原判据保留，额外指标另列，不改阈值 |
| 当时cuBLAS选核 | [文章第564行](https://github.com/QiLHong/vitamin-cuda/blob/520f97093a2abc239977be83d244b327d0e2e2ef/posts/zh/sgemm_beating_cublas.md#L564)：`cutlass_80_simt_sgemm_128x64_8x5_nn_align1` | 与本机第26章观察的256×128路径不同，不能假设库算法一致 |

在该提交的上述文章、模块README、顶层README和测试源码中，未找到与这些成绩绑定的精确驱动版本、CUDA Toolkit版本、cuBLAS版本、Python版本、操作系统、笔记本功耗档、锁定频率、每批时间戳、原始逐次时延或完整环境清单。顶层README的CUDA11.0+等通用依赖要求是安装概述，不是作者该次实测环境。文章引用的NCU截图外链本次未能读取，因此不从未读图片推断版本或硬件计数。

本章固定的是“要运行哪一份源码”，不是声称这份快照必然就是作者产生每条历史日志时的原始构建。历史输出没有绑定编译产物哈希；源码版本、库版本和运行状态这几个未知量须保留。

## 2. 本次“跑作者原入口”具体指什么

本轮已经直接调用原`test_4096()`函数体，保留它的torch→普通cuBLAS→五SIMT固定顺序、原`benchmark`和`diff_check`，连续做七批。Windows扩展编译沿用第26章已验证的本机pytorch环境和有记录的主机编译/链接适配；没有修改上游源码。

需要明确两处边界：

1. 默认`__main__`在[第221～224行](https://github.com/QiLHong/vitamin-cuda/blob/520f97093a2abc239977be83d244b327d0e2e2ef/kernels/sgemm/test.py#L221-L224)实际调用的是`test_tf32_4096()`。本轮以模块导入后调用SIMT函数，不能说默认主入口原封运行。
2. 原`test_4096()`在五SIMT之后还调用一次cuBLAS TF32，见[第147～155行](https://github.com/QiLHong/vitamin-cuda/blob/520f97093a2abc239977be83d244b327d0e2e2ef/kernels/sgemm/test.py#L147-L155)。本轮保持函数体不变，由库代理在首次TF32预热调用之前停止，所以执行的是该函数的完整普通FP32/SIMT部分，不是包括TF32尾段的完整函数。

原函数自身没有随机种子。本轮在每批调用函数前外部设置seed42，使批次输入可核对；这不是作者原来的隐含条件。七批仍是原固定顺序，没有RW/DBF先后交换。统计应报告七个墙钟批均值的中位数、最小/最大、逐批相对cuBLAS的比值，以及输入指纹和错误信息。它回答本机该入口下的表现，不提供独立机器或独立长期会话的置信区间。

## 3. 推荐的比较口径

首先只在每个实验自身条件内比较：作者详细日志的DBF/cuBLAS；本机七批原入口中的同批DBF/cuBLAS；本机原生寄存器实验中的同轮默认/上限128版本。不要将作者一组日志的cuBLAS与另一组DBF配对，不要把墙钟均值与CUDA event中位数配对，也不要用第26章另一会话或不同构建的数据替代本轮重新测得的基线。本轮与第26章均为alpha1、beta0。

跨GPU可以检验的问题是：**在RTX 4090 D、本次公开源码和明确工具链下，是否同样观察到DBF领先本机cuBLAS？** 这不是在复测RTX 5060移动版的14ms。即使在本机没有重现相对领先，也只能说明该优势没有直接迁移到本机条件；不能据此证伪作者在另一硬件/库版本上的记录。

使用同机cuBLAS比值能减少对绝对速度的依赖，但仍受不同架构、cuBLAS版本、选核、运行时状态影响，不是消除了全部硬件差异的“算法纯性能”。第26章已实测的资源变化说明，同一源码跨编译路径和GPU可能出现不同寄存器分配、驻留与调度。

## 4. 受控寄存器上限实验：验证假设而非追求预设结论

前提证据来自第26章严格B构建：DBF为143 registers/thread、256线程/CTA；两个CTA所需寄存器为`143*256*2=73216`，超过Ada每SM64K个32位寄存器，因此该寄存器分配最多允许一个CTA。这个限制只是值得验证的候选原因，不足以保证减少寄存器就提速。

本轮将DBF放入独立编译单元，对照默认编译与`--maxrregcount=128`。kernel源码、tile、线程输出映射、shared槽数、数学语义和测量方法保持一致，不对其他kernel施加该选项。真正改变的是编译器的寄存器上限约束；编译器可能因此产生spill或不同指令调度，需要测量这些结果。

在读时延之前，先回答：

- 默认与受限版本实际各用了多少寄存器，128上限是否真正生效。
- 是否新增stack/local/spill，SASS是否出现LDL/STL，NCU是否观察到local流量。
- 实际shared carveout、资源限制与occupancy是否允许更多CTA驻留；不能仅凭“目标128”宣称两个CTA。
- 正确性是否保持原阈值、NaN/Inf是否为0，再比较交替多轮CUDA events。

如果spill成本或调度变化抵消驻留收益，应报告无收益或退化，而不是继续修改精度、同时更改其他参数或删除慢样本。

## 5. 原入口七批实际结果

### 5.1 实际执行和计时条件

[`Run-VitaminAuthorSource.ps1`](../../scripts/windows/Run-VitaminAuthorSource.ps1)调用[`vitamin_author_source.py`](../../scripts/vitamin_author_source.py)，已完成七批、每批七个实现，共49条原始打印均值，退出码0。[完整控制台日志](../../results/windows/20261021-vitamin-followup/author-source/complete-console.log)保留原输出；[result.json](../../results/windows/20261021-vitamin-followup/author-source/result.json)记录函数体未变、未插入计时操作、未执行默认main与TF32。第26章单批Python04数据继续保留，未混入这里的七批。

本机环境为RTX 4090 D、Windows WDDM、驱动610.60、Windows CUDA Toolkit12.4.131、MSVC19.37；原入口使用已有的`D:\Anaconda\Anaconda\envs\pytorch\python.exe`，Python3.12.7、PyTorch2.5.1/CUDA12.4。`TORCH_CUDA_ARCH_LIST=8.9`；本进程禁用TF32、matmul精度设为`highest`。环境及文件SHA256见[environment.json](../../results/windows/20261021-vitamin-followup/author-source/environment.json)。

先实际尝试了原`load`参数，Windows构建失败；随后使用已记录的主机`/O2 /std:c++17`、`cublas.lib`和`-Xptxas=-v`等必要适配成功加载。两次记录见[build-attempts.json](../../results/windows/20261021-vitamin-followup/author-source/build-attempts.json)，最终加载的pyd与build.ninja及其哈希也已归档。CUDA `--use_fast_math`继续保留，原三个CU文件与三个测试函数体没有改写。因此它是**原SIMT入口的Windows适配复跑**，不是作者原机器环境的二进制复刻。

每个表项来自原函数10次预热后200次调用的墙钟批均值；统计七个批均值的中位数。计时包括Python/CPU派发和批次前后同步，未额外插入CUDA events。原先`beta=0`的语义使每次输出覆盖C，七批输入SHA256相同。未锁频，原固定顺序未交换，七批来自同一会话，不能视作七次独立环境复现。

### 5.2 七批均值的分布

数据来源：[wallclock-batches.csv](../../results/windows/20261021-vitamin-followup/author-source/wallclock-batches.csv)与[wallclock-summary.csv](../../results/windows/20261021-vitamin-followup/author-source/wallclock-summary.csv)。`波动=(最大批均值−最小批均值)/中位批均值`；最后一列是每批`cuBLAS时延/该实现时延`的中位数，百分比大于100%才表示该实现领先cuBLAS。它与“两列中位时延之比”不同。

| 实现，4096³ | 中位批均值/ms | 最小～最大/ms | 波动/% | 同批cuBLAS比值中位/% |
|---|---:|---:|---:|---:|
| torch | 2.880712 | 2.850184～2.940111 | 3.12 | 103.24 |
| 普通cuBLAS | 2.975850 | 2.931173～3.113700 | 6.13 | 100.00 |
| Tiling | 4.017488 | 3.961907～4.203855 | 6.02 | 73.98 |
| AT | 3.339826 | 3.328029～3.511154 | 5.48 | 88.63 |
| SW | 3.185724 | 3.178152～3.343273 | 5.18 | 92.86 |
| RW | 3.196809 | 3.187579～3.331775 | 4.51 | 92.48 |
| DBF | 3.229833 | 3.220918～3.384016 | 5.05 | 91.69 |

**本机这七批没有重现DBF领先普通cuBLAS。** DBF在7/7批均慢于cuBLAS；同批吞吐变化中位数为−8.31%，范围−10.98%～−5.07%，对应同批时延增加的中位数约9.07%。两列中位时延直接相除则为`2.975850/3.229833=92.14%`，即吞吐低7.86%；这是另一种统计量，不能与91.69%混用。另一次[独立CPU统计核验](../../results/windows/20261021-vitamin-followup/author-source/independent-statistical-audit.json)从49条原打印均值重新计算，结果一致。

同批数字全部保留如下，便于检查顺序和漂移：

| 批次 | cuBLAS/ms | DBF/ms | DBF相对cuBLAS吞吐变化/% |
|---|---:|---:|---:|
| 1 | 2.931173 | 3.221315 | −9.01 |
| 2 | 2.951185 | 3.220918 | −8.37 |
| 3 | 2.960243 | 3.228645 | −8.31 |
| 4 | 2.975850 | 3.229833 | −7.86 |
| 5 | 2.982208 | 3.252432 | −8.31 |
| 6 | 3.113700 | 3.280160 | −5.07 |
| 7 | 3.012363 | 3.384016 | −10.98 |

后几批多个实现的时延整体上升，且同批各实现依次执行，不是同时采样；仅凭这些墙钟数值不能区分温度、频率、桌面调度或其他运行状态的贡献。七批均值不包含200次调用的独立时延，无法据此计算逐次尾延迟。结果支持的是本机、本次顺序与版本下的相对表现，不能反推作者5060移动版的历史数字错误。

### 5.3 正确性证据覆盖到哪里

每批原控制流依次通过普通cuBLAS和五个SIMT的六次`diff_check`，才到达TF32截断点。因此七批共42次原allclose断言通过；计数由未改的顺序控制流到达截断点推定，没有替换`diff_check`或在计时内部加hook。

额外的批后检查只针对仍可读取的普通cuBLAS输出和最后DBF输出：七批最大绝对误差均为0、NaN/Inf均为0、`abs≤0.01`失败元素均为0，见[batch-evidence.json](../../results/windows/20261021-vitamin-followup/author-source/batch-evidence.json)。这是相对本批torch输出的比较，不是CPU-double全量证明；其他四个SIMT中间结果已被原函数覆盖，不能把这组“误差0”推广到它们。原allclose在各次覆盖之前已经执行，五个实现都保持了原正确性判据。

## 6. 寄存器实验实际结果与解释

### 6.1 实际编译与静态证据

独立子工程[`experiments/vitamin-dbf-register/CMakeLists.txt`](../../experiments/vitamin-dbf-register/CMakeLists.txt)只建立两个DBF可执行目标，共用[`windows_dbf_register_benchmark.cu`](../../scripts/windows_dbf_register_benchmark.cu)。实际编译命令确认两个目标均为`sm_89`、无`--use_fast_math`、无忽略编译器兼容性的参数；仅`maxr128`目标增加`--maxrregcount=128`，host宏仅用于记录配置名，不改变device body。

本轮实际结果见[verified-build.json](../../results/windows/20261021-vitamin-followup/register-limit/build01/verified-build.json)、[build.log](../../results/windows/20261021-vitamin-followup/register-limit/build01/build.log)与[静态比较JSON](../../results/windows/20261021-vitamin-followup/register-limit/static/comparison-summary.json)。

| 实际构建项 | default | maxr128 |
|---|---:|---:|
| registers/thread | 143 | 128 |
| 用户static shared/CTA | 32768 B | 32768 B |
| ptxas spill load/store | 0/0 | 0/0 |
| STACK / LOCAL | 0 / 0 | 0 / 0 |
| 整函数静态SASS指令数 | 2448 | 2432 |
| 整函数LDL / STL | 0 / 0 | 0 / 0 |
| 主循环FFMA / LDG / LDS / STS / BAR | 1024 / 4 / 64 / 10 / 1 | 1024 / 4 / 64 / 10 / 1 |
| 主循环各LDG之后尚余的FFMA数 | 192～210 | 206～221 |

新default与第26章严格B构建的DBF整函数SASS文本逐条一致，证明此次拆分编译单元保住了原基线。上限128确实生效，静态结果未发现spill；但整数指令、MOV、寄存器编号及调度发生变化，主循环的4条LDG位置也发生变化。因此测到的时差不能全部归因于occupancy，也不能只凭静态零LDL/STL宣称运行时local计数已测为0。另一次[源码一致性审计](../../results/windows/20261021-vitamin-followup/register-limit/profile-analysis/source-consistency.json)对输入、误差、参考、计时等十段host协议进行规范化哈希比较，确认与第26章协议一致。

### 6.2 正常CUDA events测量

[`Run-DbfRegister.ps1`](../../scripts/windows/Run-DbfRegister.ps1)已串行测完1024、2048、4096三个尺寸，每尺寸七对。奇数对先default后maxr128，偶数对反序；每一对成员是独立进程，各自建立严格FP32 cuBLAS参考和进行预检，再预热10次、测量30次。两个构建采用相同xorshift输入、seed20261020、alpha1、beta0；不启用fast math或TF32。每次调用前在同一stream恢复C，events只包围DBF调用，恢复C、进程启动、分配、参考和验证均不计时。详细协议见[environment.json](../../results/windows/20261021-vitamin-followup/register-limit/run01/environment.json)。

每轮取30个event时延的算术均值，再报告七轮均值的中位数。未删除任何慢样本。下表由[summary.csv](../../results/windows/20261021-vitamin-followup/register-limit/run01/analysis02/summary.csv)生成，TFLOPS=`2*N³/(中位轮均值ms*1e9)`；轮波动仍为`(最大−最小)/中位数`。

| N | default中位/ms | maxr128中位/ms | default TFLOPS | maxr128 TFLOPS | default轮波动/% | maxr128轮波动/% |
|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 0.081327 | 0.087834 | 26.406 | 24.449 | 1.05 | 0.54 |
| 2048 | 0.458683 | 0.470224 | 37.455 | 36.536 | 1.75 | 0.08 |
| 4096 | 3.137397 | 3.114325 | 43.807 | 44.131 | 3.97 | 5.44 |

逐对变化单独按`(default轮均值/maxr128轮均值−1)*100%`计算，再取七对中位数。数据为[paired.csv](../../results/windows/20261021-vitamin-followup/register-limit/run01/analysis02/paired.csv)和[paired-summary.csv](../../results/windows/20261021-vitamin-followup/register-limit/run01/analysis02/paired-summary.csv)：

| N | 配对吞吐变化中位/% | 最小～最大/% | maxr128更快的对数 | 两列中位时延之比对应变化/% |
|---:|---:|---:|---:|---:|
| 1024 | −7.40 | −7.94～−7.11 | 0/7 | −7.41 |
| 2048 | −2.50 | −2.52～−0.87 | 0/7 | −2.45 |
| 4096 | +2.61 | −0.73～+5.61 | 6/7 | +0.74 |

**128上限不是普遍加速：1024和2048七对全部退化；4096六对更快，但收益较小且随轮次波动。** 4096的“配对中位+2.61%”与“中位吞吐+0.74%”是不同的统计量，不能只挑较大的数而不说明口径。这里没有跨会话置信区间，也没有证明以后每次运行都能达到同一提升。

正式数据共有1260个events、42个轮均值，另有128尺寸的两个小测events。44次独立进程每次记录严格参考、候选预检、轮末输出三类检查，共132行，全部通过原allclose、`abs≤0.01`和有限性检查。相对严格参考最大绝对误差0.00341796875；CPU-double核验在128时覆盖全部16384个元素，其他尺寸为256个固定位置，最大误差0.0018326517129025888。检查每个预检和每轮最后一次输出，未对每个计时重复都读回；不能把抽样CPU核验写成大尺寸的CPU全量参考。见[correctness-all.csv](../../results/windows/20261021-vitamin-followup/register-limit/run01/analysis02/correctness-all.csv)和[audit.json](../../results/windows/20261021-vitamin-followup/register-limit/run01/analysis02/audit.json)。

另一次[独立CPU复算](../../results/windows/20261021-vitamin-followup/register-limit/run01/independent-statistical-audit.json)核对了44次调用、原始event字段、全部检查阶段、输入哈希和先后交替，并重新计算42个轮均值与三个配对中位数，结果一致。这里的“独立CPU复算”指核验保存的测量数据，不是再次在CPU上计算所有大矩阵。

记录的138个遥测点只覆盖42个“预热至验证完成”窗口中的17个，其余25个没有点。窗口比纯kernel计时更宽，仍不足以证明每轮频率稳定。WDDM、未锁频和独立进程边界是本次局限；例如4096 default最慢event为3.944448ms，已原样保留。

第一次CPU汇总因遥测数字格式解析失败，保留在[analysis/INCOMPLETE.txt](../../results/windows/20261021-vitamin-followup/register-limit/run01/analysis/INCOMPLETE.txt)。仅修复CPU解析后读取同一原始数据，成功结果存入`analysis02`；没有重新挑选或重新运行一组GPU时延。

### 6.3 profiler机制核对

NCU2024.1.1已完成两尺寸×两版本×两次，共八份采样。按DBF名称过滤，跳过一次preflight与十次预热，只捕获随后的一次DBF；两个重复交换版本先后。采用kernel replay、`--clock-control none`、`--cache-control none`。原始CSV和`.ncu-rep`在[profile01](../../results/windows/20261021-vitamin-followup/register-limit/profile01)，汇总在[profile-comparison.csv](../../results/windows/20261021-vitamin-followup/register-limit/profile-analysis/profile-comparison.csv)，两次样本范围在[profile-ranges.csv](../../results/windows/20261021-vitamin-followup/register-limit/profile-analysis/profile-ranges.csv)。NCU时间和profiler下的host events均未加入第6.2节性能统计。

**目标128这次确实提高了可驻留CTA数量，而且运行时shared carveout也变了。** 实际资源关系为：

| NCU资源项，两种尺寸一致 | default | maxr128 |
|---|---:|---:|
| registers/thread | 143 | 128 |
| static shared/CTA | 32768 B | 32768 B |
| 总shared/CTA，含系统保留 | 33792 B | 33792 B |
| 实际shared配置/SM | 65536 B，即64 KiB | 102400 B，即100 KiB |
| 寄存器允许的CTA/SM | 1 | 2 |
| shared允许的CTA/SM | 1 | 3 |
| local load/store sectors | 0 / 0 | 0 / 0 |

本次新采集了第26章未采集的carveout指标，不能把本轮数值追溯为第26章已知事实。本轮default不仅受到143寄存器的限制，两个CTA还需`33792*2=67584 B`，超过实际65536 B的shared配置；受限版本的100 KiB配置则足以容纳三个CTA，但128寄存器把上限约束为两个。用户shared数组仍是32768 B，没有改为更小数组，也没有主动设置carveout。寄存器限制这一编译选择连同生成代码、运行时配置发生了变化，不能把它解释成“只增加occupancy、其他硬件状态完全相同”。

下表取每格两份NCU样本的均值。occupancy和issue/FMA列为NCU百分比；active/eligible与barrier列是**每SMSP active cycle上的warp平均数**，不是时间占比或每warp等待时长。

| N / 版本 | achieved occupancy/% | active warps | eligible warps | issue active/% | FMA pipe active/% | barrier stalled warps |
|---|---:|---:|---:|---:|---:|---:|
| 2048 default | 16.65 | 2.00 | 1.37 | 76.58 | 67.805 | 0.07 |
| 2048 maxr128 | 29.025 | 3.49 | 2.32 | 74.805 | 67.355 | 0.25 |
| 4096 default | 16.66 | 2.00 | 1.28 | 71.93 | 65.655 | 0.09 |
| 4096 maxr128 | 31.97 | 3.84 | 2.56 | 75.91 | 68.76 | 0.30 |

4096的较多驻留warp伴随着更多eligible warp，以及较高的issue/FMA活动，方向与正常计时的部分收益相符；其long-scoreboard stalled warp数从0.18降到0.01。但2048虽同样提高驻留，issue/FMA活动反而略降，与正常计时七对退化构成反例：**更高occupancy不是更快kernel的充分条件。** 此外，barrier计数从0.07/0.09升到0.25/0.30时，resident warp总数也在增加；不能把0.30读成“30%的时间都在同步”，也不能直接据该数值大小比较每warp同步成本。

受限版本还观察到非零shared bank-conflict计数，default的这两项在四份采样中均为0：

| N | maxr128 shared load conflicts，两次范围 | maxr128 shared store conflicts，两次范围 |
|---:|---:|---:|
| 2048 | 10642～10911 | 61520～61711 |
| 4096 | 67600～69826 | 2946257～2956507 |

源码的shared地址布局没有变化，静态主循环LDG/LDS/STS数量也相同；此处仅记录动态计数现象，不据它编造新的索引冲突公式或断言退化全部由冲突导致。指令调度、同时驻留CTA、实际shared/L1划分均已不同，要区分贡献还需要进一步受控实验。

本轮能够确认的结论是：上限生效、没有观察到spill/local流量、实际驻留上升；性能收益依赖尺寸，在4096有限，在1024/2048为负。它支持继续研究资源与调度的权衡，不支持把128上限提升为全尺寸默认优化。两次profile只提供本会话的机制证据，replay和不锁频条件也不允许用NCU时间替换正常events。

## 7. 证据与复跑命令

先阅读本章第1～3节理解比较口径，再从[本轮归档目录](../../results/windows/20261021-vitamin-followup/README.md)进入两条路线。`author-source`保存原函数stdout、每批CSV、源文件快照、pyd/build.ninja和哈希；`register-limit/build01`保存两个独立目标及实际编译命令，`run01`保存正常events，`profile01`单独保存NCU采样。不要拿`profile01/events-*`计算正式性能。

以下命令可在普通PowerShell中执行，显式调用已有PowerShell7。脚本自己在本次子进程中加载MSVC/CUDA环境；无需全局修改PATH、重新安装工具或进入WSL。默认输出使用新时间戳目录，已存在目录会拒绝覆盖。

```powershell
Set-Location 'D:\CUDA-Lab\cuda-learning\SGEMM_CUDA'
$pwsh = 'C:\Users\Administrator\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\powershell\pwsh.exe'

# 原 test_4096 的 SIMT 部分，七批原墙钟协议。
& $pwsh -NoProfile -File .\scripts\windows\Run-VitaminAuthorSource.ps1 -Batches 7

# 独立构建 default 和 maxr128，自动核实架构、选项和来源。
& $pwsh -NoProfile -File .\scripts\windows\Build-DbfRegister.ps1 `
    -BuildDir build-windows-dbf-register-sm89

# 三尺寸，每尺寸七对，10次预热、每轮30个events，并自动汇总。
& $pwsh -NoProfile -File .\scripts\windows\Run-DbfRegister.ps1 `
    -BuildDir build-windows-dbf-register-sm89 `
    -Rounds 7 -Repeats 30 -Warmup 10

# 可选机制分析：2048/4096，各版本各采两次，与正常计时分开。
& $pwsh -NoProfile -File .\scripts\windows\Profile-DbfRegister.ps1 `
    -BuildDir build-windows-dbf-register-sm89 -ProfileRepeats 2
```

[`Build-DbfRegister.ps1`](../../scripts/windows/Build-DbfRegister.ps1)仅编译和导出静态证据，不启动GPU；run/profile会先校验编译产物及源码哈希。各命令应依次执行，避免两个GPU任务同时运行。如只想复算本次保存的数据，不需要再次运行GPU：

```powershell
Set-Location 'D:\CUDA-Lab\cuda-learning\SGEMM_CUDA'
$recheck = 'results/windows/dbf-register-recheck-' + (Get-Date -Format 'yyyyMMdd-HHmmss')
& 'D:\Anaconda\Anaconda\python.exe' .\scripts\summarize_dbf_register_results.py `
    --result-dir results/windows/20261021-vitamin-followup/register-limit/run01 `
    --output-dir $recheck

# 同样只用CPU：重新核对host协议、实际编译选项和已保存的NCU计数。
$evidenceRecheck = 'results/windows/dbf-evidence-recheck-' + (Get-Date -Format 'yyyyMMdd-HHmmss')
& 'D:\Anaconda\Anaconda\python.exe' -B .\scripts\analyze_dbf_register_evidence.py `
    --compile-commands results/windows/20261021-vitamin-followup/register-limit/build01/compile_commands.json `
    --profile results/windows/20261021-vitamin-followup/register-limit/profile01/ncu-summary.csv `
    --output-dir $evidenceRecheck
```

本章修改和新结果均留在本地工作分支，没有提交、推送或覆盖前三阶段记录。寄存器上限仅作用于两个独立实验目标中的maxr128目标，不改变K10、K30或主工程既有kernel的默认编译选项。

## 来源、署名与解释边界

vitamin-cuda版权为`Copyright (c) 2025 wingedge`，原MIT许可证与固定源码保持不变。原作者数字在本章只作带条件的历史引用；本机成绩必须来自本机实际产物。索引、shared布局与双缓冲生命周期的推导见第26章，本章重点是“原入口复现”和“单个编译约束的受控验证”。
