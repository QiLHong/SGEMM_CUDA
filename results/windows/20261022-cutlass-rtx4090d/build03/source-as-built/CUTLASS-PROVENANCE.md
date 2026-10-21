# RTX 4090 D CUTLASS 对照的来源与包装边界

本目录的 `cutlass_adapter.cuh` 是本项目新增的主机包装与模板实例化文件。没有复制、改写 CUTLASS device kernel、iterator、shared layout 或同步算法。所需头文件来自独立的 NVIDIA CUTLASS 检出目录；此说明不代表已经完成所有平台的编译或性能验证，实际结果以本轮归档为准。

- 上游：[NVIDIA/CUTLASS](https://github.com/NVIDIA/cutlass)。
- 固定版本：v3.5.1，commit `f7b19de32c5d1f3cedfc735c2849f12b537522ee`。
- 本机路径：`D:\CUDA-Lab\cuda-learning\deps\cutlass-v3.5.1`。下载是源码检出，不修改 CUDA/MSVC/Python 安装。
- 许可证：BSD-3-Clause，版权属于 NVIDIA CORPORATION & AFFILIATES；完整声明原样保存在 [CUTLASS-LICENSE.txt](CUTLASS-LICENSE.txt)。
- [公开 Gemm 主机入口](https://github.com/NVIDIA/cutlass/blob/f7b19de32c5d1f3cedfc735c2849f12b537522ee/include/cutlass/gemm/device/gemm.h)。
- [官方 TF32 row-major 单元测试配置](https://github.com/NVIDIA/cutlass/blob/f7b19de32c5d1f3cedfc735c2849f12b537522ee/test/unit/gemm/device/gemm_tf32t_tf32t_f32t_tensor_op_f32_sm80.cu)。
- [单 TF32 MMA 指令实现](https://github.com/NVIDIA/cutlass/blob/f7b19de32c5d1f3cedfc735c2849f12b537522ee/include/cutlass/arch/mma_sm80.h)。

## 三个固定配置

选择在首次性能测量之前完成；不是运行后择优，也不是穷举 CUTLASS profiler 的全部实例。两个 TensorOp 配置来自上述官方文件的同名单元测试：`128x128x32_64x64x32` 和 `128x64x16_64x32x16`。较小配置保留官方 BK16/4-stage，原因是它的 A+B shared 原始容量为 49,152 B，而前者为 98,304 B，能够覆盖 Ada 上不同的 shared 容量需求。两者改变多个参数，不能当作单因素实验。

| 运行名 | A/B 类型 | CTA M×N×K | warp M×N×K | 指令 M×N×K | stages | threads | 对齐 A/B | epilogue 每次元素 |
|---|---|---|---|---|---:|---:|---:|---:|
| `cutlass_tf32_128x128` | `tfloat32_t` | 128×128×32 | 64×64×32 | 16×8×8 | 3 | 128 | 4 | 4 |
| `cutlass_tf32_128x64_bk16` | `tfloat32_t` | 128×64×16 | 64×32×16 | 16×8×8 | 4 | 128 | 4 | 4 |
| `cutlass_simt_256x128` | `float` | 256×128×8 | 64×64×8 | 1×1×1 | 4 | 256 | 1 | 1 |

SIMT 配置沿用本项目 [旧公开 CUTLASS 对照](../../scripts/cutlass_simt_reference.cu) 与[第 16 章](../../docs/zh/16-公开CUTLASS对照与主循环成本审计.md) 的模板参数；没有引用 RTX 3080 的历史吞吐来填本轮 RTX 4090 D 表格。共同条件是 row-major A/B/C、FP32 accumulator/output、identity swizzle 1、split-K=1。`Sm80` 是模板特性下限，本轮实际目标由构建命令指定 `sm_89`；不使用 Hopper TMA。

## 精度与调用语义

TensorOp 显式选择 `tfloat32_t` 和 `OpMultiplyAdd`，底层预期生成 `mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32`。没有使用 `OpMultiplyAddFastF32`、TF32x3、FP16/BF16。harness 在计时前生成量化后的 `float` 存储，并将同一份 A/B 字节同时交给作者版、BN64、CUTLASS 和 TF32 cuBLAS。包装仅将指针解释成等宽的 `tfloat32_t`，不做第二次输入转换。

SIMT 直接接收原始 FP32 A/B，使用 `OpClassSimt` 和普通 `OpMultiplyAdd`。它属于单独的 strict FP32 结果表；不能以 TF32 精度收益解释其相对性能。

`initialize(m,n,k,A,B,C,stream)` 创建 row-major tensor reference，stride 分别是 K、N、N，固定 `alpha=1,beta=0`，调用 `can_implement()`、查询 workspace、分配并初始化。这些发生在正确性、预热与计时之前。`run(stream)` 只调用已初始化的官方 `op.run(stream)`；没有额外 epilogue kernel，也没有将初始化/分配算入 GEMM event 时间。

C0 恢复由统一 harness 负责，适用形状由 harness 与 `can_implement()` 双重检查。不能仅凭 CUTLASS 的 tile grid 向上取整推断本项目其他无边界保护 kernel 支持尾块。

## 保留的上游行为与局限

`gemm.h::Gemm::run()` 在 shared storage 大小至少 48 KiB 时，每次发射都调用 `cudaFuncSetAttribute(...MaxDynamicSharedMemorySize...)`。本包装保留它，而非抽取私有 Params 自行发射。CUDA event 边界包围每次公开调用，可能包含 stream 暂时为空时的主机提交空隙；这是需要说明的计时局限，不能只看源码声称 event 完全排除了所有 host 影响。

本文给出的 shared 原始容量不是最终寄存器/occupancy 实测；具体 `sizeof(GemmKernel::SharedStorage)`、ptxas/SASS 和 NCU 证据由构建及运行记录确认。没有自动将两个固定官方配置的最好值称为 CUTLASS 库的最佳性能。
