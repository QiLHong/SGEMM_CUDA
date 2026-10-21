# RTX 4090 D / Windows：TF32 BK16 → BK32

本归档只包含2026-10-21本机4090D实验，未混用3080、其他卡或Linux历史成绩。
原始数据、低精度数值失败、慢样本均保留。没有自动提交或推送。

中文报告：[第30章](../../../docs/zh/30-TensorCore-K分块实验与资源代价.md)。
实验源码：[tensorcore-ktile](../../../experiments/tensorcore-ktile/README.md)。

## 已执行

| 目录 | 内容 |
|---|---|
| `preflight` | Git起点、已有未提交patch、408个既有源码和1364个历史文件哈希 |
| `build01` | Release/sm89实际命令、成功编译日志、可执行文件、PTX/SASS/资源、当时源码 |
| `mapping01` | CTA输出、BK16/32复制地址、K步序列完整CPU枚举 |
| `run01` | 原版/动态BK16/动态BK32/cuBLAS：2520正式events、数值、输入指纹、遥测 |
| `analysis01` | 初测七轮统计及逐轮配对，无删除事件 |
| `confirm01` | 原版/BK32相邻交换顺序的独立确认：1890正式events |
| `confirm-analysis01` | 独立确认统计 |
| `profile01` | NCU2024.1.1：3尺寸×4实现；12个rep、CSV、命令和日志 |
| `confirm-analysis02` | 与analysis01相同确认结果，另加NCU指标长表；不使用profile时间作性能表 |
| `evidence01` | NCU前的静态、保护和公平顺序核验 |
| `evidence02` | 第一次NCU表转换未完成：分类值CachePreferNone不能转换为float；保留失败说明 |
| `evidence03` | 修正分类值解析后的最终完整审计，旧数据未改变 |
| `source-as-delivered.zip` | 交付时项目源码快照，不包含其他历史results或build目录 |
| `source-as-delivered-sha256.csv` | UTF-8源码清单，核对zip每项内容 |
| `archive-sha256.csv` | UTF-8归档清单，覆盖生成时本目录除自身以外的所有文件 |

本轮正式共4410 events，小矩形另24 events；两批156次原版逐位比较零差异。
相对原始严格FP32的绝对0.01、SIMT allclose仍失败，不能因与TF32 cuBLAS相同就声称严格精度成功。
Sanitizer未运行；CPU double小矩形全量，大矩阵固定256点。

## 独立确认结果

| N | 原版 TFLOPS | BK32 TFLOPS | TF32 cuBLAS TFLOPS | BK32/原版逐轮配对中位提升 |
|---:|---:|---:|---:|---:|
| 1024 | 31.332 | 32.063 | 35.605 | +2.09%，7/7胜出 |
| 2048 | 48.461 | 46.155 | 51.904 | −4.75%，0/7胜出 |
| 4096 | 71.553 | 66.967 | 69.216 | −6.45%，0/7胜出 |

固定seed20261020，C=AB，同输入预转换，7轮×10预热×30测量，C0同stream D2D恢复在event外。
TFLOPS来自七个轮均值延迟的中位数；配对提升另按每轮baseline_ms/candidate_ms−1计算。
未锁频；温度43..53°C，SM2520..2715MHz；WDDM桌面环境。
BK32寄存器127→170，shared32→64KiB，CTA/SM2→1；不作为通用默认。

数据读取和派生脚本错误没有导致任何GPU重测或样本删除。
证据完整性以SHA256为准；发布前需保留CSV/日志原始字节，避免Git换行转换改变归档哈希。
