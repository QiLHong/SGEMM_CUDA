#pragma once

// 所有 GPU kernel 的统一入口头文件。学习顺序按编号 1→10，而非下面的排列。
// 保留上游 include 顺序：11/12 使用的 WARPSIZE 定义来自 10 号头文件。
#include "kernels/10_kernel_warptiling.cuh"
#include "kernels/15_kernel_grouped_swizzle.cuh"
#include "kernels/16_kernel_smem_xor_swizzle.cuh"
#include "kernels/17_kernel_b_async_pipeline.cuh"
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
