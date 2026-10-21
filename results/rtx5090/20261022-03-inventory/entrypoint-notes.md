# RTX 5090 历史复现实验入口核对

核对日期：2026-10-22（Asia/Shanghai）。这里只做本地只读源码/历史证据盘点，没有运行 GPU、编译、提交或推送。主执行代理负责实际服务器实验及发布。

来源快照：SGEMM_CUDA `a2b23dabadc79440bf9cbf63f0b566f7eec429ba`，本地分支 `results/rtx4090d-k30-stage2-20261021`，检查时工作区干净；vitamin-cuda `520f97093a2abc239977be83d244b327d0e2e2ef`。本地仓库内未找到 AGENTS.md。CSV 的 source_snapshot_commit 表示本次读取对象，不把旧实验执行时的 Git HEAD 冒充已包含当时未提交的源码；例如第33章环境记录的 a54435d 与当前完整归档 a2b23da 应同时保留。源码 SHA256 以本次完整归档里的文件为准。

`reproduction-matrix.csv` 为244个配置/实验记录，BN64稳健性按36个尺寸/seed/独立进程格拆开；不是244次已完成测量。`historical-evidence-manifest.csv` 给出轻量历史源码、报告、CSV和环境元数据哈希。`audit-entrypoints.csv` 保留构建、GPU审计与静态分析入口，状态均独立于历史GPU成绩。未运行状态不能由旧机器PASS填充。

## 可直接复用但须适配协议的入口

| 系列 | 实际入口 | 关键约束 |
|---|---|---|
| K1–K12、K15–K23 | 根CMake target `sgemm`，`sgemm 编号` | 编译时变体由CMake宏选择；同一二进制不能运行多个编译期变体。K11默认BN256，128方阵非法。K7固定布局也先审计合法形状。 |
| K30 | `SGEMM_BUILD_K30_AUDIT=ON` → `k30_audit mode N beta repeat warmup` | mode=strict/k10/v1/v2/v3/v4。V1=Small stage1，V2=Small stage2，V3=Small stage4，V4=Large stage4；不是sgemm30。 |
| K31 | `SGEMM_BUILD_K31_AUDIT=ON` → `k31_audit mode N beta repeat warmup` | control/late/vectors/stage3/stage4/cta256/r220/r225/r230/r235/r240，加strict/k10/k30基线；不是sgemm31。 |
| K32 | `SGEMM_BUILD_K32_AUDIT=ON` → `k32_audit mode N beta repeat warmup` | serial/overlap/single，加strict/k10/k30；160线程，128计算线程；不是sgemm32。pilot缺single，final包含single，历史不能混表。 |
| K10 PAD4/PAD8、K30 V1/V2 | `SGEMM_BUILD_WINDOWS_BENCHMARK=ON` → `windows_benchmark N beta rounds repeats warmup new.csv selection` | `all`仅K10/PAD8/K30V2/strict，`sync`仅K10/K30V1/K30V2/strict；五项同批需局部扩展。alpha固定0.5，beta0不等于C=AB。 |
| vitamin 五SIMT | `SGEMM_BUILD_VITAMIN_BENCHMARK=ON` → `windows_vitamin_benchmark N rounds repeats warmup new.csv [all/名字]` | `SGEMM_VITAMIN_ORIGINAL_CONFIG` ON/OFF分开构建和表，ON保留fast_math及普通cuBLAS；C=AB。 |
| SIMT寄存器限制 | `experiments/vitamin-dbf-register` | targets=`windows_dbf_default_benchmark`和`windows_dbf_maxr128_benchmark`；宏及maxrregcount只作用于独立DBF target。 |
| Tensor Core原始跨库 | `experiments/tensorcore` → `windows_tensor_benchmark` | 命令 `M N K rounds repeats warmup new.csv [all/implementation/tensor_pairs]`。TF32输入先显式转换；FP16输出参考包含同样舍入。 |
| Grid W | `experiments/tensorcore-grid` → `windows_tensor_grid_benchmark` | W1/2/4名字为`tf32_w1/2/4`、`fp16_w1/2/4`。W8使用`vitamin_tf32_dbf`/`vitamin_fp16_dbf_rw`，没有`*_w8`。 |
| KTile | `experiments/tensorcore-ktile` → `windows_tensor_ktile_benchmark` | 准确名字`tf32_dynamic_bk16`、`tf32_dynamic_bk32`；默认all四组包括original/cuBLAS，confirm排除BK16。 |
| BN64 | `experiments/tensorcore-bn64` → `windows_tensor_bn64_benchmark` | `tf32_bn64_bk32`。Names虽含BK16，实际计时groups没有BK16，单选可能零events；BK16使用KTile入口。 |
| BN64稳健性 | `experiments/tensorcore-bn64-robustness` → `windows_tensor_bn64_robustness_benchmark` | `--smoke 新目录`；`--case M N K seed rounds repeats warmup new.csv`；`--batch 1..3 frozen-plan.csv 新目录`。完整36格×3实现×7轮×30=22680正式events。 |
| CUTLASS公开参考 | `scripts/build_cutlass_reference.sh 固定CUTLASS目录 新目录` → `reference cutlass N beta repeat warmup` | CUTLASS v3.5.1哈希`f7b19de32c5d1f3cedfc735c2849f12b537522ee`；OpClassSimt/Sm80模板，CTA256×128×8、4stage、align1；不是cuBLAS私有核。旧NVTX头/库可能需隔离适配。 |

根CMake和子工程均可传 `-DCMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_BUILD_TYPE=Release`；这只是目标参数，不是编译成功声明。CUDA12.4旧脚本中`-lnvToolsExt`可能失效，根CMake已经能选择CUDA::nvtx3。历史脚本有NCU `--clock-control base`，本轮不得不经核对照抄；实际使用SM120支持指标及不更改锁频的策略。

K30/K31/K32/CUTLASS原harness使用libc rand、seed20261013+矩阵序号偏移，alpha0.5、beta0/1/3，N128..4096且整除128。原计时只输出repeat均值，若本轮要求逐event证据，需独立包装或小改输出，保留原协议日志，不能称原CSV已有全部event。跨OS libc rand序列未必相同。4090D统一入口使用冻结xorshift输入，不能仅凭seed写同输入。

K31/32旧静态resource_gate要求registers≤240且无spill；这是历史筛选门槛。K31r220历史spill8B被淘汰，必须保留5090重新编译/验证状态。`analyze_k31_binary.py`主循环匹配依赖SM86的2048FFMA形态，SM120编译变化时应记录工具限制或适配解析，不删除断言伪造通过。

## SM120及上游入口

vitamin SM120确实只导出hgemm_bcf_dbf_rw、hgemm_k_stages、hgemm_tma_r_k_stages_64、hgemm_tma_r_k_stages_32和hgemm_cublas。源码另有空的`hgemm_tma_rw_k_stages_kernel`，没有绑定，不能列为已实现可测kernel。

TMA64/TMA32固定BM=BN128、256线程、3stage，动态shared分别98328/49176字节；普通cp.async binding BK32，multistage默认3stage。原Python host每次调用创建tensor maps并设置函数attribute，因此wallclock含这些host工作；统一可复用descriptor的kernel-only需另表。TMA descriptor按实际fast_dim×sizeof(T)字节stride编码，不能把TMEM术语套到它。

`kernels/hgemm_sm120/test.py`和普通HGEMM `test.py`默认BF16/4096，使用randn并没有在test_4096里显式设seed。`test_all()`里的seed42不自动作用到直接调用test_4096。复跑可以保存实际输入字节指纹，若加seed必须标包装适配。SGEMM `test.py`默认跑test_tf32_4096；重做历史SIMT原入口要明确调用test_4096并在进入TF32段前停止。

所有普通HGEMM四个真实变体（naive/bcf/dbf/dbf_rw）和cuBLAS均有FP16/BF16 binding。TF32五变体准确名字已写入CSV，不能只跑single/dbf后把其余默认通过。BF16/FP16是低精度输入输出、FP32累加，strict FP32的abs0.01标准和原作者低精度allclose须独立报告。

## 尚缺精确历史参数的条目

第04章明确提过六类选项（fast_math、额外向量化编译选项、maxrregcount、cache load、shared carveout、交换grid轴），但现有轻量归档未给全体参数/patch/原始结果。本表保留六条“工具受限”历史记录，不发明配置。逐tile launch-bounds表的threads256三行没有完整tile tuple，也单独标证据缺项。baseline和sub8x4n2可追溯冻结tile；不能仅从寄存器数猜threads256形状。

上游K9/K10/K11 autotuner保存的是宽泛搜索定义，包含原地sed改源码、清空旧输出、固定设备等行为；没有证据说明所有合法组合都在3080/4090D测过。已把三个入口留在清单，不能把它们当成新一轮无边界搜索授权。已实际归档的K10 24个tile与launch-bounds、padding、K15–23全部变体单独列出。

部分CSV一行包含多尺寸清单，完成其中一部分时应拆出按shape/beta/batch的子状态，或在result_path标完整已测范围，不把整行总状态直接置为通过。特别是单个seed20261020的四尺寸首批只占完整BN64稳健性矩阵的4/36格。
