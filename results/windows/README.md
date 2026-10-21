# Windows 原生实验归档

## 先按 GPU、精度和实验批次找结果

本索引中的实测均单独归档为 Windows / RTX 4090 D。项目其他目录仍保留 RTX 3080、A6000、Linux 历史成绩，不能合并为同一排名。
原始 CSV、二进制、PTX/SASS、NCU 与测量时源码快照不随教程更新而重写。

发布补记（2026-10-22）：第一阶段 `run01` / `profile-script`、vitamin SIMT `stage3` / `followup`、第 24～27 章及本索引，在 2026-10-21 核对时均已存在于本机，但未随 `a54435d` 发布。本次按用户要求，将它们与最新 BN64 稳健性源码、脚本、第 33 章和完整原始归档一并纳入版本控制。冻结归档保持原始字节，其“未提交/未推送”说明仅记载实验结束时的状态；本次发布没有重跑或改写历史成绩。

| 实验 | 计算口径与范围 | 归档 | 中文报告 |
|---|---|---|---|
| Windows第一阶段 | strict FP32，alpha=0.5/beta=3 | [run01](20261020-rtx4090d-run01/) | [24](../../docs/zh/24-Windows原生构建与4090D实验.md) |
| K30同步 | K10/V1/V2/strict cuBLAS | [stage2](20261021-k30-stage2/) | [25](../../docs/zh/25-K30同步机制与4090D性能分析.md) |
| vitamin SIMT | C=AB，原数学配置与strict配置分开 | [stage3](20261021-vitamin-simt-stage3/) | [26](../../docs/zh/26-vitamin-SIMT复现与K30公平对照.md) |
| 原入口/寄存器实验 | Python墙钟与native events分开 | [followup](20261021-vitamin-followup/) | [27](../../docs/zh/27-vitamin原入口复跑与寄存器上限实验.md) |
| Tensor Core原生复现 | TF32、FP16与strict FP32分组，C=AB | [tensorcore](20261021-tensorcore/) | [28](../../docs/zh/28-TensorCore初体验与4090D实测.md) |
| Grid W实验 | TF32/FP16各自对照 | [grid](20261021-tensorcore-grid/) | [29](../../docs/zh/29-TensorCore网格分组优化与反例.md) |
| K分块实验 | 仅TF32，原版/动态BK16/BK32 | [ktile](20261021-tensorcore-ktile/README.md) | [30](../../docs/zh/30-TensorCore-K分块实验与资源代价.md) |
| BN64实验 | 仅TF32，原版/父BK32/BN64 | [bn64](20261021-tensorcore-bn64/README.md) | [31](../../docs/zh/31-TensorCore-BN64优化与尺寸效应.md)、[32](../../docs/zh/32-RTX4090D-TensorCore优化学习指南.md) |
| BN64稳健性验证 | 仅TF32，原版/BN64/cuBLAS；多seed、三批独立进程，含3072 | [robustness](20261021-tensorcore-bn64-robustness/README.md) | [33](../../docs/zh/33-TensorCore-BN64稳健性与尺寸适用范围.md) |

表格不是跨批排行榜。比如 BN64 与父 BK32 的四项比较必须使用 BN64 的 `run01`，不能从两份报告各取最大值相除。
TF32/FP16的输入转换在kernel计时之外，原始FP32误差失败照常保存；同精度库的局部胜出不是strict FP32或端到端结论。

### Tensor Core证据从哪里读

- `run*` / `confirm*` / `pair-confirm*`：原始event、输入指纹、逐项误差、UTC窗口和遥测；只在有完成标记时做完整批次统计。
- `analysis*`：由对应events计算的轮均值、中位数、波动和逐轮配对，不删除慢样本。
- `build*`：当时实际编译命令、二进制、源码快照、PTX/SASS和资源；不等于今天工作树的全部文件。
- `profile*` / `evidence*`：NCU及静态分析，与公平event表分开；中间失败/修正记录保留，以对应中文报告指定的最终分析为准。
- `source-as-delivered.zip` / 哈希清单：交付时快照；`--verify`文本比较与SHA256字节校验是不同检查。

前四组Tensor Core归档记录原生复现及优化过程；新增的稳健性归档单独保存本轮复测。没有BF16、3080 Tensor Core或Hopper TMA新结果。
整体阅读见[微教材第9章](../../docs/zh/微教材/09-TensorCore从SIMT到矩阵指令.md)与[第32章](../../docs/zh/32-RTX4090D-TensorCore优化学习指南.md)。

## 第一阶段的历史记录（2026-10-20）

- [`20261020-rtx4090d-run01`](20261020-rtx4090d-run01/)：正式正确性矩阵与七轮未插桩基准；原始 `timings-*.csv` 共 2520 个 event，汇总见 `summary.csv` 和 `paired-summary.csv`。
- [`20261020-rtx4090d-profile-script`](20261020-rtx4090d-profile-script/)：最终 PowerShell profiler 脚本的实际执行检查；有命令参数、退出码、NCU 版本、指标汇总和报告。其 event 时间不参与性能比较。
- 学习讲解、全部结果、测试条件与限制见[第 24 章](../../docs/zh/24-Windows原生构建与4090D实验.md)。

`run01/source-snapshot.zip` 与 `sha256.csv` 保存测量当时的源文件及可执行文件标识；
后来补齐的汇总、profiler 脚本和最终文档另存于 `delivery-source.zip`，不能把后者称为测量当时的快照。
第一阶段交付时的 benchmark 与该阶段实测版本只差一句中文注释，kernel 算法一致；这句话不描述后续 Tensor Core 子工程。

严格 FP32，固定 seed=20261020、alpha=0.5、beta=3，CUDA events 排除 C 复位拷贝；
每轮预热 10 次、计时 30 次，共 7 轮，先求轮均值，再取七轮中位数。
Windows/WDDM/动态时钟环境下的结果不直接等同于历史 Linux/A6000/RTX3080 成绩。
所有原始慢样本保留；`profile-*-events.csv` 与 `sanitizer-*` 不是正常基准。

额外 memcheck 附加超时，未完成检查；日志被保留，不视为 PASS。
未执行 racecheck/synccheck，未在 Linux 上重新构建。
原作者署名、许可证及旧成绩未修改。本段记载的是第一阶段当时尚未提交的状态；后续已发布的 Tensor Core 提交和教程以各章的版本说明为准，不能据此判断当前 Git 状态。
