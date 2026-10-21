# Windows / RTX 4090 D：CUTLASS 独立对照

中文讲解、结果、局限和复跑命令见[第 34 章](../../../docs/zh/34-CUTLASS与BN64的4090D公平对照.md)。本目录只属于 RTX 4090 D，TF32 与 strict FP32 分表，不覆盖历史成绩。

## 正式证据链

| 目录 | 作用 |
|---|---|
| `preflight` | 实际 Windows 工具链、GPU、源提交和初始 Git 状态 |
| `build03` | 最终 Release/sm_89 编译日志、参数、exe、ptxas、PTX/SASS、原始源码和 CUTLASS 头文件快照 |
| `run04` | 最终 6 个不重叠进程、68040 个 events、小尺寸全量 CPU 检查、输入指纹、所有误差和遥测 |
| [analysis01](analysis01/README.md) | 两精度性能表、轮均值中位数、CV、逐轮配对、独立误差判定 |
| [verification02](verification02/verification.json) | 独立复算、原始文件和完整恢复链核验；包含 258 个复用批内文件字节检查 |
| `static01` | 8 个函数的资源与指令计数、SASS 回跳循环/加载顺序、sm_89 和精度证据 |
| `profile01` | 2048/4096 × 5 实现的 10 份 NCU 实测；不计入公平性能表 |
| `profile-analysis01` | NCU 原始单位保留的指标表及解析审计 |
| `source-as-delivered.zip` | 交付时脚本/教程快照，区别于 build03 的编译时源码 |
| `archive-sha256.csv` | 本目录交付文件的相对路径、大小和 SHA256；不包含清单自身 |

最终 binary SHA256 为 `F971E74DA275F5787583ABD9AA81535AC175001991AD6D77F2495131E93E1D8C`。
CUTLASS v3.5.1 commit 为 `f7b19de32c5d1f3cedfc735c2849f12b537522ee`。

## 保留的开发、失败和被排除记录

- `build01`：编译期间补充 host metadata，仅作开发记录，未跑 GPU。
- `build02` / `run01`：小尺寸通过，FP32 3072 的 SIMT allclose 失败使旧组合 gate 中止。没有降低阈值；build03 改为独立记录两套标准，FP32 按保留的 abs<=0.01 门限完成测量。
- `run02`：前三个完整批次被 run04 原样复用；TF32 批 2/3 因恢复时发生进程重叠而全部排除。
- `run03` 及 launcher 日志：包含一次未启动实际测量的后台启动尝试，以及随后被明确停止的重叠恢复进程；不进入正式性能表。
- `run04/recovery.json` / `reused-complete-batch-sha256.csv`：只复用截止 UTC 2026-10-21 17:30:25 前已完整结束的 TF32 批 1、FP32 批 1/2；重跑其余三批。最终六进程退出码 0、时间窗互不重叠。
- `verification01`：较早的独立复算，保留；verification02 增加了原始恢复来源和字节一致性检查。

`DATA_AUDIT_PASS` 是数据完整性结论，不是所有精度门限通过。FP32 3072 的 SIMT allclose 保持 FAIL；TF32 对同一量化输入的 abs<=0.01 在 2048/3072/4096 为 FAIL，对原始 FP32 则四尺寸均 FAIL。TF32 native allclose 通过，无 NaN/Inf。Sanitizer 本轮未成功完成，仍未验证。

本轮不自动提交、推送或合并。归档保留全部失败/异常慢样本；重跑时创建新目录，不改写这里的原始记录。
