// 【文件导读】K1–K23 设备实现的统一包含入口
// 给初学者：host 是 CPU 端控制代码，device 是 GPU 端计算代码；分配、拷贝、启动、等待、校验各有职责。
// 这是头文件集合，不是一个新 kernel；由 runner.cu 包含并在该编译单元实例化模板。
// 注意包含依赖：K15/K16 等复用 K10 的 wt 命名空间，K23 复用 K21 的 iterator/compute 辅助代码。
// 因此不要随意调整 include 顺序，或只拷贝一个实验头文件而遗漏其依赖。
// K30/K31/K32 不从主程序编号入口运行；它们的独立 harness 显式包含所需头文件。
// 中文总索引：docs/zh/23-代码与脚本使用索引.md；脚本命令默认在仓库根目录执行。

#pragma once

// 所有 GPU kernel 的统一入口头文件。学习顺序按编号 1→10，而非下面的排列。
// 保留上游 include 顺序：11/12 使用的 WARPSIZE 定义来自 10 号头文件。
#include "kernels/10_kernel_warptiling.cuh"
#include "kernels/15_kernel_grouped_swizzle.cuh"
#include "kernels/16_kernel_smem_xor_swizzle.cuh"
#include "kernels/17_kernel_b_async_pipeline.cuh"
#include "kernels/18_kernel_warp_fragment_layout.cuh"
#include "kernels/19_kernel_fragment_register_prefetch.cuh"
#include "kernels/20_kernel_simt_256x128.cuh"
#include "kernels/21_kernel_cutlass_style_skeleton.cuh"
#include "kernels/22_kernel_multistage_pipeline.cuh"
#include "kernels/23_kernel_shared_fragment_iterator.cuh"
#include "kernels/11_kernel_double_buffering.cuh"
#include "kernels/12_kernel_double_buffering.cuh"
#include "kernels/1_naive.cuh"
#include "kernels/2_kernel_global_mem_coalesce.cuh"
#include "kernels/3_kernel_shared_mem_blocking.cuh"
#include "kernels/4_kernel_1D_blocktiling.cuh"
#include "kernels/5_kernel_2D_blocktiling.cuh"
#include "kernels/6_kernel_vectorize.cuh"
#include "kernels/7_kernel_resolve_bank_conflicts.cuh"
#include "kernels/8_kernel_bank_extra_col.cuh"
#include "kernels/9_kernel_autotuned.cuh"
