# Windows 第二阶段：K30 同步与 RTX 4090 D

执行日期：2026-10-21，Asia/Shanghai。工作分支 `windows-sm89-learning-20261020`，
基点 `77567876cecc180a039d2855d748acd9d2ba427a`；包含尚未提交的第一阶段和本阶段修改。
没有自动提交、推送或改动旧成绩。

完整代码教程、线程坐标、同步时序、结论与复跑命令见
[第 25 章](../../../docs/zh/25-K30同步机制与4090D性能分析.md)。

## 证据目录

| 目录/文件 | 用途 |
|---|---|
| `preflight/` | 开始第二阶段前的未提交源码快照、Git 状态、tracked diff、历史结果 SHA256 |
| `mapping/` | CPU 完整 CTA 枚举：每种实现 16384 行；输出与搬运覆盖检查、旧脚本交叉验证 |
| `run01/` | 正常运行：环境、完整构建日志、实际编译命令、源快照、程序指纹、正确性、每个 CUDA event、遥测与汇总 |
| `profile01/` | 独立 Nsight Compute 2024.1.1 的 16 个 profile；命令、退出码、`.ncu-rep`、原始 CSV、含单位汇总 |
| `static/` | 实测二进制的 PTX、SASS、资源报告、主循环指令顺序及独立审阅笔记 |
| `verification/benchmark-audit.json` | 从原始 CSV 独立重算统计、输入一致性、慢样本、遥测覆盖和 69 项历史文件哈希 |

CPU 审计已经执行通过，第一阶段 69/69 个历史文件 SHA256 未变。
`static/evidence-reproduced` 由 `analyze_windows_k30_evidence.py` 重算，
19 行 PTX 位置与 224 行 profiler 指标范围表均与初次解析逐字节一致。

`run01/source-snapshot.zip` 对应当时实测源码，`run01/sha256.csv` 记录源码和可执行文件。
最终补齐的文档及 CPU 分析脚本另存交付源码包；不会回写实测快照。
交付文件为 [`delivery-source.zip`](delivery-source.zip) 和
[`delivery-sha256.csv`](delivery-sha256.csv)。
相对本阶段开始时的实际变更见 [`phase2-changed-files.csv`](phase2-changed-files.csv)：
6 个文件修改、8 个文件新增；K10/K30 kernel 源文件没有变化。
[`verification/source-preservation.json`](verification/source-preservation.json) 记录保护检查，
[`archive-sha256.csv`](archive-sha256.csv) 为本阶段各个归档文件的最终 SHA256 清单。

## 本轮测量

RTX 4090 D，114 SM，驱动 610.60，WDDM；Windows CUDA 12.4.131，
MSVC 19.37.32826.1，Ninja Release，全部七条 CUDA 编译命令确认 sm_89。

固定 `seed=20261020, alpha=0.5, beta=3`，PEDANTIC FP32，TF32 禁用。
每种实现、每轮预热 10 次、计时 30 次，每次 GEMM 前在同一流中 D2D 恢复 C0，
复位不计入 CUDA event。共 7 轮；V1/V2 相邻并交换先后。
轮内用全部 30 个 event 的均值，最终取七个轮均值的中位数换算吞吐，保留全部慢样本。

| N | K10 PAD4 TFLOPS | K30 V1 | K30 V2 | strict cuBLAS | V2/V1 配对提升中位数 |
|---:|---:|---:|---:|---:|---:|
| 1024 | 20.3276 | 19.8653 | 21.6727 | 29.2354 | +9.0586% |
| 2048 | 31.1242 | 29.3804 | 32.8450 | 47.3744 | +11.1082% |
| 4096 | 44.5683 | 41.5728 | 46.4742 | 48.4162 | +11.4267% |

统一入口 72 组正确性、84 个正式轮末输出与 2520 个计时 event 全部通过。
原独立 K30 audit 另跑 V1/V2 各 18 项，通过；这些单次计时不用于性能表。
原绝对误差阈值 0.01 未改变，统一入口最大绝对误差 0.001708984375，无 NaN/Inf。
CPU double 对照：N=128 全量，较大尺寸各 256 个固定输出点抽查。

V2/V1 的 21 个配对均更快；这支持本次会话内的稳定优势。
条件 bootstrap 区间只重采样已观察轮次，不覆盖跨会话或 WDDM 系统误差。
GPU 遥测共 42 点，温度 44–61°C、SM 时钟 2415–2715 MHz；
只有 30/84 个轮次窗口含遥测点，54 个窗口未知，不能据此声称各实现时钟恒定。

## 指令与硬件观察

V1/V2 分别 199/235 寄存器，ptxas 无 spill，profile 的 local load/store sectors 也为零。
每 CTA 用户 shared 为 16640/33280 字节；NCU total 另含每 block 的 1024 字节保留开销。
4096 时 occupancy 均约 16.6%，V2 的 eligible warps、issue 较高，short-scoreboard/barrier 较低。
V2 的 bank-conflict 计数并不为零，也没有比 V1 更低。

实际 PTX 在计算前加载 future；实际 sm_89 SASS 却把 V2 的八条 LDG 放在
2046 条 FFMA 之后，仅剩两条 FFMA。V1/V2 主循环分别有 2/1 条 BAR；
源码动态 CTA 屏障次数为 `2T`/`T+1`，`T=N/16`。
因此不能声称获得了充分的 global load/compute 重叠，也不能把全部提升归因于屏障次数。
Profiler 下的 events 和 NCU duration 严格留在 `profile01`，不纳入上表。

本轮只对照已有算法，没有实现第三阶段的优化候选。文档提出的 grouped-grid 实验尚未执行。
