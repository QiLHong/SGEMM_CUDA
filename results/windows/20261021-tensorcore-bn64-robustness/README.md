# RTX 4090 D：TF32 BN64/BK32 稳健性与尺寸适用范围

本目录是 Windows 原生 RTX 4090 D 专属新归档。四尺寸 × 三 seed × 三独立进程，正式 **22680 events** 完整完成，独立复算 PASS；不是 RTX 3080、strict FP32、PyTorch 墙钟或历史批次成绩。

先读[第33章中文报告](../../../docs/zh/33-TensorCore-BN64稳健性与尺寸适用范围.md)；代码机制的前置教程为[第32章](../../../docs/zh/32-RTX4090D-TensorCore优化学习指南.md)。复跑所需 PowerShell 命令都在第33章末尾。

| N | BN64/作者：九组合配对中位数范围 | 正收益轮 | BN64/TF32 cuBLAS：九组合范围 | 正收益轮 |
|---:|---:|---:|---:|---:|
| 1024 | +3.573%～+4.546% | 58/63 | −8.426%～−5.946% | 1/63 |
| 2048 | +19.485%～+19.612% | 63/63 | +11.370%～+11.739% | 63/63 |
| 3072 | +7.455%～+9.330% | 63/63 | +0.458%～+2.247% | 51/63 |
| 4096 | −1.716%～+0.214% | 19/63 | +2.207%～+4.165% | 59/63 |

2048、3072各九组合均有正的配对中位数；3072对库仍有12个负收益轮。4096对作者只有1/9组合为正。完整36组合、轮均值及波动均保留，不以总体平均隐藏回退。

## 证据入口

| 文件/目录 | 内容 |
|---|---|
| [protocol.md](protocol.md)、[run/plan.csv](run/plan.csv) | GPU运行前固定的协议、756组执行顺序；plan哈希由脚本执行前复核 |
| [preflight](preflight/) | 两仓库Git状态/差异、所有列出文件实际字节哈希、真实工具链与可用NCU指标 |
| [build01](build01/) | 唯一正式构建、完整命令、exe、资源、PTX/SASS、源码快照及补充K30头文件 |
| [smoke](smoke/)、[smoke.log](smoke.log) | 128×1024×K32/64/96，CPU double每例131072输出全量；共9个辅助event |
| [sanitizer01](sanitizer01/) | 单个小案例memcheck附加超时；TIMEOUT_UNVERIFIED，未宣称任何模式通过 |
| [run](run/) | 三batch原始timings、五参考checks、bitwise、inputs、CPU位置、UTC窗口、遥测及进程记录 |
| [analysis/cases.csv](analysis/cases.csv) | 全部36个size×seed×batch组合 |
| [analysis/summary.csv](analysis/summary.csv) | 108实现结果；每轮30events均值，再取七轮中位数；含范围、标准差、CV |
| [analysis/paired-rounds.csv](analysis/paired-rounds.csv)、[analysis/paired.csv](analysis/paired.csv) | 504个逐轮配对与72个配对摘要，未删除负收益轮 |
| [analysis/errors.csv](analysis/errors.csv) | 原始FP32与各参考标准失败、最大误差；未放宽阈值 |
| [analysis/telemetry-summary.csv](analysis/telemetry-summary.csv) | 637个稀疏样本，包含CPU检查间隙；不是逐event恒频证据 |
| [verification/verification.json](verification/verification.json) | 独立读取raw，以fsum/独立median复算；计数、顺序、输入、检查、3PID/时段通过 |
| [profile01](profile01/) | 2048/3072/4096、seed20261020，两核共6份成功NCU报告；时间不进入公平表 |
| [evidence](evidence/) | 实际device字节、规范化SASS与编码字比较；新NCU整理、主循环片段 |
| [navigation-audit](navigation-audit/) | CPU全CTA/grid/输出/16B对齐审计、真实链接检查、旧NCU摘录与来源哈希 |
| [historical-statistics.json](historical-statistics.json) | 独立复算旧57.998成绩，明确不属于本轮数据 |
| [source-as-delivered.zip](source-as-delivered.zip)、[source-as-delivered-sha256.json](source-as-delivered-sha256.json) | 本轮实际交付文件字节快照，zip回读校验 |
| [delivery-check.json](delivery-check.json)、[archive-sha256.csv](archive-sha256.csv) | 旧文件保存性检查、源码/二进制核验、此归档除清单自身外逐文件SHA256 |

## 条件与边界

CUDA12.4.131、MSVC19.37.32826.1、Ninja1.12.1、Release/sm89，无fast-math；驱动610.60 WDDM，cuBLAS120405。两个自定义kernel源码与最终编码指令未变。新exe哈希为 `9440ffa182ba2906d4767f5c5bf323c6fedd0e006639786a630d08165a11efef`。

正式批次北京时间2026-10-21约22:16:28至22:18:50，进程39360、41760、39264，互不重叠。这是同日连续独立进程复测，没有验证跨天/重启。三seed是同一有界输入分布，不覆盖任意精度范围或形状。

计算C=AB；显式TF32准备在event外，三实现同一实际输入；每次同stream恢复4字节/元素C0，event外但可能影响缓存。cuBLAS用FAST_TF32、DISALLOW_REDUCED_PRECISION_REDUCTION、ATOMICS_NOT_ALLOWED、GEMM_DEFAULT。全量GPU检查加CPU大尺寸256固定点，不能描述成CPU全量大矩阵。

576次逐位比较无差异，4320检查无NaN/Inf；相对原始FP32的864条检查仍全部不满足abs0.01/SIMT/native标准。BN64/作者一致性不等于严格FP32精度，也不证明不存在数据竞争。Sanitizer附加失败已保存，没有绕过权限。当前脚本仅补记未来超时终止的返回码，`sanitizer01`中保留实际运行脚本。

NCU是单个launch的多pass采集，未锁频、未强制清缓存；计数器支持解释，不等于单因素因果证明。特别是4096单次profile的elapsed改善没有对应到公平测量的稳定领先；正反证据都在报告中。

本轮只补host入口、统计和文档。旧未提交修改保留，不提交推送、不替换默认kernel、不做自动分派；到此停止扩展实验。
