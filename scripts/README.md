# 脚本怎么选

第一次来，不要挨个运行脚本。先读[代码与脚本使用索引](../docs/zh/23-代码与脚本使用索引.md)：
它解释此前整理的 122 个代码/配置文件并补充 Windows 入口，包含术语、调用关系、最小 K30 路线、结果读法和危险入口。

Windows 原生在仓库根目录使用 PowerShell 7：

```powershell
.\scripts\windows\Build.ps1 -WithExperiment
.\scripts\windows\Run-Benchmarks.ps1
# 可选，单独运行，避免与 benchmark 同时占用 GPU：
.\scripts\windows\Profile-Experiment.ps1
```

工具链选择、每个线程的数据流与本次实测见[Windows 教程](../docs/zh/24-Windows原生构建与4090D实验.md)。
第二阶段 K30 同步对照使用独立构建与结果目录：

```powershell
.\scripts\windows\Build.ps1 -BuildDir build-windows-k30-sync-sm89 -Architecture 89 -WithExperiment
.\scripts\windows\Run-K30Sync.ps1 -BuildDir build-windows-k30-sync-sm89
# benchmark 完成后再运行 profiler；不要并行占用 GPU：
.\scripts\windows\Profile-K30Sync.ps1 -BuildDir build-windows-k30-sync-sm89
```

不传 `OutputDir` 时脚本创建新的时间戳目录；指定已有目录会拒绝覆盖。
`Run-K30Sync.ps1` 自动调用 `Summarize-Results.ps1` 与 `Summarize-Telemetry.ps1`；
完整 CPU 映射、静态分析命令及指标单位见[第 25 章](../docs/zh/25-K30同步机制与4090D性能分析.md)。
以下 `.sh` 和历史复现说明继续保留为 Linux 入口。

## Tensor Core：四组独立实验

原理：[微教材第 9 章](../docs/zh/微教材/09-TensorCore从SIMT到矩阵指令.md)。
代码与结论：[第 32 章](../docs/zh/32-RTX4090D-TensorCore优化学习指南.md)。
四组均为 Windows 原生 CUDA，复用 CUDA 12.4 / MSVC 14.37 / Ninja，Release / sm_89；不依赖 PyTorch。

| 想复现什么 | Build / Run 后缀 | 运行选择 | 统计入口 | 文档 |
|---|---|---|---|---|
| 作者 TF32/FP16 和 K30/strict FP32 对照 | `TensorCore` | 默认八实现、三个尺寸；`-PairsOnly` 仅4096的四个Tensor实现/库对照 | `summarize_tensor_results.py --run … --build … --output …` | [28](../docs/zh/28-TensorCore初体验与4090D实测.md) |
| TF32/FP16 的 grid W=1/2/4 | `TensorGrid` | `-Selection all`扫描，`grid_w1`/`grid_w2`/`grid_w4`确认 | `summarize_tensor_grid.py --run … --output …` | [29](../docs/zh/29-TensorCore网格分组优化与反例.md) |
| TF32 K 分块 BK16/32 | `TensorKTile` | `all`包含动态BK16控制；`confirm`只有原版/BK32/cuBLAS | `summarize_tensor_ktile.py --run … --output …` | [30](../docs/zh/30-TensorCore-K分块实验与资源代价.md) |
| TF32 BN128/64，固定候选BK32 | `TensorBN64` | `all`比较四项；`confirm`只有原版/BN64/cuBLAS | `summarize_tensor_bn64.py --run … --output …` | [31](../docs/zh/31-TensorCore-BN64优化与尺寸效应.md)、[32](../docs/zh/32-RTX4090D-TensorCore优化学习指南.md) |

表中后缀对应 `scripts/windows/Build-后缀.ps1`、`Run-后缀.ps1` 和 `Profile-后缀.ps1`。
不要将 `TensorCore -PairsOnly` 写成 `-Selection confirm`，也不要给 BN64 统计器传原版专用的 `--build`。
四个 Build 脚本均只编译并检查 PTX/SASS；Run 才启动正确性检查和计时，Profile 单独执行 NCU。

### 复跑作者 TF32 / FP16 原生移植

下列默认运行包含 K30 与 strict FP32 对照，三种精度分组报告；它不是作者 Python 墙钟入口。
在普通 PowerShell 中调用本机 PowerShell 7，每个结果目录必须全新。构建目录和输出目录都放在旧证据树外：

```powershell
Set-Location 'D:\CUDA-Lab\cuda-learning\SGEMM_CUDA'
$ps7 = 'C:\Users\Administrator\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\powershell\pwsh.exe'
$py = 'D:\Anaconda\Anaconda\python.exe'
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$out = "results\windows\tensorcore-rerun-$stamp"
$build = "build-windows-tensorcore-$stamp"

& $ps7 -NoProfile -File scripts\windows\Build-TensorCore.ps1 -BuildDir $build -EvidenceDir "$out\build" -Python $py
if ($LASTEXITCODE) { throw 'build failed' }
& $ps7 -NoProfile -File scripts\windows\Run-TensorCore.ps1 -BuildDir $build -OutputDir "$out\run"
if ($LASTEXITCODE) { throw 'run failed; keep evidence' }
& $ps7 -NoProfile -File scripts\windows\Run-TensorCore.ps1 -BuildDir $build -OutputDir "$out\pairs" -PairsOnly
if ($LASTEXITCODE) { throw 'pair run failed' }
& $py scripts\summarize_tensor_results.py --run "$out\run" --build "$out\build" --output "$out\analysis"
if ($LASTEXITCODE) { throw 'analysis failed' }
& $py scripts\summarize_tensor_results.py --run "$out\pairs" --build "$out\build" --output "$out\pair-analysis"
if ($LASTEXITCODE) { throw 'pair analysis failed' }
```

需要原版 profiler 数据时，在正常计时结束后单独执行：

```powershell
& $ps7 -NoProfile -File scripts\windows\Profile-TensorCore.ps1 -BuildDir $build -OutputDir "$out\profile"
if ($LASTEXITCODE) { throw 'profile failed; normal event results remain valid separately' }
& $py scripts\summarize_tensor_results.py --run "$out\run" --build "$out\build" --profile "$out\profile" --output "$out\profile-analysis"
if ($LASTEXITCODE) { throw 'profile analysis failed' }
```

### 复跑 BK32 / BN64 优化与只读统计

BN64 的完整命令见[第 32 章](../docs/zh/32-RTX4090D-TensorCore优化学习指南.md#329-按目的选择复跑命令)。
它单独使用 `TensorBN64` 构建目录；不要把刚才 `TensorCore` 的 `$build` 传给 BN64。
所有 Run 默认至少七轮、每轮十次预热/三十次计时；原版 `-PairsOnly` 不是三个尺寸的完整确认。

只想复算已发布 BN64 成绩时，不需要 GPU，也不需要重新构建：

```powershell
Set-Location 'D:\CUDA-Lab\cuda-learning\SGEMM_CUDA'
$py = 'D:\Anaconda\Anaconda\python.exe'
$out = 'results\windows\tensorcore-statistics-' + (Get-Date -Format 'yyyyMMdd-HHmmss')
& $py scripts\summarize_tensor_bn64.py --run results\windows\20261021-tensorcore-bn64\confirm01 --output $out
if ($LASTEXITCODE) { throw 'saved event analysis failed' }
```

### 看到错误时先分清是哪一层

| 信息或现象 | 含义与下一步 |
|---|---|
| `Refusing overwrite` | 输出目录已有文件；改用新时间戳，不清理旧归档 |
| `Binary/source mismatch` / `Changed dependency` | 二进制或依赖与构建清单不同；使用该子工程 Build 生成新的证据后再运行 |
| 原始FP32 `FAIL`，同精度检查通过 | Tensor数值契约与严格FP32分开判定；看具体reference/threshold，不能放宽阈值消除失败 |
| `Shape rejection failed` | 主机形状保护没有按预期拒绝，必须停止；不能绕过入口直接launch |
| NCU权限/超时 | 保留日志；正常event与正确性结论单独看。可查看已有ptxas/PTX/SASS，不绕过权限 |
| 历史 `audit_tensor_*` 哈希失败 | 检查是合法文档/源码更新还是证据损坏；该脚本绑定旧快照，不等同当前代码的通用测试 |

`--verify` 用于提取/派生文本一致性；`archive-sha256.csv` 用于冻结文件字节校验。
`finalize_tensor_*` 是一次性归档器，不是复跑统计命令。脚本默认目录可能落在旧实验树下，故本页始终显式传 `-OutputDir` / `-EvidenceDir`。
本次文档维护保留 `experiments/tensorcore*` 原始字节，避免仅修改其中 README 就触发现有依赖清单不一致。

## 常用顺序

第三阶段 vitamin SIMT 原生复现（A 为原数学配置，B 为严格 FP32）：

```powershell
.\scripts\windows\Build-Vitamin.ps1
$result = "results\windows\$(Get-Date -Format 'yyyyMMdd-HHmmss')-vitamin-simt"
.\scripts\windows\Run-Vitamin.ps1 -OutputDir $result
.\scripts\windows\Summarize-Vitamin.ps1 -ResultDir $result
.\scripts\windows\Profile-Vitamin.ps1 -OutputDir "$result-profile"
```

以上在仓库根目录的 PowerShell 7 执行。全部结果为 `C=AB`，不与第二阶段 `alpha=0.5,beta=3` 直接拼表。
原生入口无 PyTorch 依赖。另有 `Run-VitaminPython.ps1` 使用现有 `envs/pytorch` 环境，经过 Windows 包装适配复用原 Python 计时/校验函数，墙钟数据单独归档；不执行原默认 TF32 测试。源码来源、线程映射、原始数据与静态分析命令见[第 26 章](../docs/zh/26-vitamin-SIMT复现与K30公平对照.md)。

```text
build     编译源码，得到可执行文件与资源日志
correctness  对同输入的严格 FP32 参考验算
memcheck / sync  检查访存、竞争与同步
fair      多轮交错计时，保存原始 CSV
profile / source  用 Nsight 解释硬件行为
analyze / summarize  读证据、统计并归档
```

RTX 3080 / strict FP32 阶段的最佳研究基线 K30 V2：从 [build_k30_audit.sh](build_k30_audit.sh) 和
[run_k30_audit.py](run_k30_audit.py) 开始。最小命令在上面的总索引中。
完整收尾复现使用包含 K10/K30/K32 的独立 harness，见[复现文档](../docs/zh/20-项目复现与简历讲述.md)。

每个文件顶部注明输入、输出、运行条件和覆盖范围，关键函数旁解释为什么这样做。
`.cu` 需要编译，`.sh` 用 Bash，`.py` 用 Python；`audit` 是编译出的程序，不是目录里的源文件。

## 先避开这些误用

- `kernel_9/10/11_autotuner.sh` 会原地改源码；先读，别直接跑。
- 旧绘图器会覆盖 README 和图片；不用它生成当前 strict FP32 研究结论。
- 输出目录已存在时换新名字，不删除旧结果。
- 只看到 GFLOPS 不等于实验成功；还要看 PASS、工具摘要、测试范围与日志是否齐全。
- 先串行验证，再计时；不要同时跑 profiler 与 benchmark。

## TF32 BN64 稳健性收尾（RTX 4090 D）

四尺寸 × 三 seed × 三个独立进程的验证入口见[第33章](../docs/zh/33-TensorCore-BN64稳健性与尺寸适用范围.md)，
本轮 22680 个正式 event 及失败记录见[独立归档](../results/windows/20261021-tensorcore-bn64-robustness/README.md)。
保持作者与 BN64 device 算法不变；与前面的单 seed 调优批次分开。

| 入口 | 用途 |
|---|---|
| [Build-TensorBN64Robustness.ps1](windows/Build-TensorBN64Robustness.ps1) | 独立 Release/sm89 构建，检查参数，保存二进制/源码/资源/PTX/SASS |
| [Run-TensorBN64Robustness.ps1](windows/Run-TensorBN64Robustness.ps1) | `-PrepareOnly` 冻结计划；`-ExecutePrepared` 按原计划串行三进程；默认包含小矩形检查 |
| [Test-TensorBN64RobustnessSanitizer.ps1](windows/Test-TensorBN64RobustnessSanitizer.ps1) | 单小案例、有时限memcheck；读取result.json，不能把退出或超时等同通过 |
| [Profile-TensorBN64Robustness.ps1](windows/Profile-TensorBN64Robustness.ps1) | 正常计时完成后，固定seed对2048/3072/4096两核定点NCU |
| [summarize_tensor_bn64_robustness.py](summarize_tensor_bn64_robustness.py) | 每例七轮均值中位数、逐轮配对、完整36组合及精度/遥测统计 |
| [verify_tensor_bn64_robustness.py](verify_tensor_bn64_robustness.py) | 从raw独立重算，检查事件/计划/输入/数值记录/三个进程 |
| [audit_tensor_bn64_robustness_evidence.py](audit_tensor_bn64_robustness_evidence.py) | 区分完整文件哈希、规范化SASS和编码字，整理NCU证据 |

输出目录必须全新。若需要重新统计旧CSV，使用新的analysis/verification目录，不重写原统计。
本轮Sanitizer附加超时，仍为未验证；六次NCU成功。完整命令与结论边界均在第33章。

## CUTLASS 与 BN64 / K30 公平对照（RTX 4090 D）

完整可复制命令见[第34章](../docs/zh/34-CUTLASS与BN64的4090D公平对照.md#348-下次复跑的准确-powershell-命令)，
证据见[独立归档](../results/windows/20261022-cutlass-rtx4090d/README.md)。官方 CUTLASS 固定为 v3.5.1，外部源码目录 `../deps/cutlass-v3.5.1`，不修改头文件。

| 入口 | 用途 |
|---|---|
| [Build-Cutlass4090D.ps1](windows/Build-Cutlass4090D.ps1) | 初始化已有 MSVC/CUDA，校验源码、编译 sm89、保存二进制/依赖/PTX/SASS；必传 `-EvidenceDir` |
| [Run-Cutlass4090D.ps1](windows/Run-Cutlass4090D.ps1) | 两精度分组，小矩形全量 CPU 检查、非法形状拒绝、三批×三 seed×四尺寸×7×30；必传全新 `-OutputDir` |
| [Resume-Cutlass4090D.ps1](windows/Resume-Cutlass4090D.ps1) | 原会话结束后复用完整进程；保留原计划/输入/binary，保存恢复文件哈希，不能用于拼接部分案例 |
| [summarize_cutlass_4090d.py](summarize_cutlass_4090d.py) | 从原始 events 生成中位数、CV、逐轮配对、误差与输入/遥测汇总 |
| [verify_cutlass_4090d.py](verify_cutlass_4090d.py) | 不导入 summarizer，独立复算并检查恢复链、原始文件和时间窗 |
| [Profile-Cutlass4090D.ps1](windows/Profile-Cutlass4090D.ps1) | 公平计时结束后采集 2048/4096 的五实现 NCU；超时有界，不混用 profiler 时间 |
| [analyze_cutlass_4090d_profiles.py](analyze_cutlass_4090d_profiles.py) | 读取 NCU CSV，保留指标单位并核验十次目标采集 |
| [analyze_cutlass_4090d_static.py](analyze_cutlass_4090d_static.py) | ptxas、PTX、SASS、加载与 MMA 顺序；静态交错不等于周期重叠证明 |

Run/Resume/Profile 共用独占文件锁，避免这些脚本并发测量。输出目录必须全新；abs、SIMT allclose、TF32 native allclose 分别判定，`DATA_AUDIT_PASS` 不覆盖数值 FAIL。
