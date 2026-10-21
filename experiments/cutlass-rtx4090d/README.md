# CUTLASS 3.5.1：Windows RTX 4090 D 独立对照

本子工程已经实际构建和测量，发布提交为 [9225bae](https://github.com/QiLHong/SGEMM_CUDA/commit/9225bae1d8b6ea1038f8be257067d841528e99b9)。完整报告及准确PowerShell命令见[第34章](../../docs/zh/34-CUTLASS与BN64的4090D公平对照.md)，原始证据见[4090 D独立归档](../../results/windows/20261022-cutlass-rtx4090d/README.md)。本README是发布后补齐的当前入口，不属于build03编译时快照。

## 先读哪些文件

| 文件 | 责任 |
|---|---|
| [CUTLASS-PROVENANCE.md](CUTLASS-PROVENANCE.md) | 固定上游commit、官方模板来源、适配边界、许可证与公开run()行为 |
| [CUTLASS-LICENSE.txt](CUTLASS-LICENSE.txt) | NVIDIA原始BSD-3-Clause声明 |
| [cutlass_adapter.cuh](cutlass_adapter.cuh) | 实例化两份TF32和一份SIMT模板，初始化及调用官方device::Gemm |
| [benchmark.cu](benchmark.cu) | 同输入C=AB、两个精度组、C0恢复、正确性、原始event、输入指纹 |
| [CMakeLists.txt](CMakeLists.txt) | 独立native target、可配置架构、cuBLAS链接及编译选项 |

上游检出目录为 `D:\CUDA-Lab\cuda-learning\deps\cutlass-v3.5.1`，固定commit `f7b19de32c5d1f3cedfc735c2849f12b537522ee`。未修改上游device算法；无需安装PyTorch来运行本target。构建脚本复用Windows CUDA12.4/MSVC19.37/Ninja，实际生成sm89。Sm80模板标签不是sm80二进制；本轮没有TMA、TF32x3或FP16/BF16结果。

## 运行与比较范围

- TF32组为作者版、BN64、CUTLASS128×128×32/3-stage、CUTLASS128×64×16/4-stage、TF32 cuBLAS；共用显式量化的设备输入。
- Strict FP32组为K10 PAD4、K30 V2、CUTLASS SIMT256×128×8/4-stage、PEDANTIC cuBLAS；共用原始FP32输入。
- 统一C=AB、alpha=1、beta=0，逐次在同stream、event之外恢复C0。初始化和workspace在计时外，官方run()每次的公开API行为保留。
- 主机入口限制M为128倍数、N为1024倍数、K为32倍数，并限制在第34章列明的范围；不得因为CUTLASS有predicated iterator，就绕过其他kernel的共同形状检查。
- 三批×三seed×四尺寸×七轮×30次，共68040正式events，另有小尺寸全量CPU double检查；大尺寸CPU只抽样256点。慢样本和不同误差标准的FAIL均保留。

## 从PowerShell进入

使用PowerShell7，在仓库根目录执行。完整、可复制的新目录构建/测试/统计/NCU命令在[34.8节](../../docs/zh/34-CUTLASS与BN64的4090D公平对照.md#348-下次复跑的准确-powershell-命令)。

| 操作 | 入口与必要参数 |
|---|---|
| 构建 | `scripts/windows/Build-Cutlass4090D.ps1 -BuildDir <新目录> -EvidenceDir <新归档/build>` |
| 测量 | `scripts/windows/Run-Cutlass4090D.ps1 -BuildDir <同一构建目录> -OutputDir <新归档/run>` |
| 统计 | `scripts/summarize_cutlass_4090d.py --run <run> --output <新analysis>` |
| 复核 | `scripts/verify_cutlass_4090d.py --run <run> --analysis <analysis> --output <新verification>` |
| NCU | `scripts/windows/Profile-Cutlass4090D.ps1 -BuildDir <同一构建目录> -OutputDir <新profile>` |

Build参数是`-EvidenceDir`，Run/Profile是`-OutputDir`。正常测量结束后才采集NCU；三个GPU脚本共享独占锁。中断恢复只接受完整进程，先确认原会话已结束，再用`Resume-Cutlass4090D.ps1`，不能同时开两个测量任务。

## 已验证结论与限制

固定的小TF32配置在本次四尺寸快于BN64；2048/4096为62.946/74.780 TFLOPS。严格FP32的CUTLASS SIMT未超过K30/cuBLAS。前者是同输入TF32对照，不是严格FP32获胜，也不是CUTLASS全库最优。

数值标准分别判定：TF32 native eps=0.1通过，严格abs<=0.01在大尺寸失败；FP32 abs门限通过，但3072的SIMT allclose失败。所有正式检查无NaN/Inf。Sanitizer未验证。10次NCU与静态指令分析支持继续研究shared operand读取工作量，尚未形成单因素因果证明。

原始归档、build03源码和交付zip保持原字节。当前文档可能比冻结快照更完整；不要修改快照或旧哈希来消除差异。
