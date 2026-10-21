// Original author: QiLHong / vitamin-cuda; see ../tensorcore/vendor/LICENSE.
// Source commit 520f97093a2abc239977be83d244b327d0e2e2ef, kernels/sgemm/sgemm_tf32.cu
// EXPERIMENT: BM128/BN64/BK32; 48KiB shared, 32 accumulators/thread; see device.diff.
#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <type_traits>
#include "../tensorcore/vendor/pack.cuh"
namespace bn64_tf32 {
#define SWIZZLE_A(row, col) ((col) ^ (((row >> 1) & 0x3) << 2))

#define SWIZZLE_B(row, col) ((col) ^ (((((((row) >> 1)) ^ ((row) >> 3))) & 0x3) << 2))

#define SWIZZLE_B_F2(row, col) ((col) ^ (((row) & 0x7) << 3))

// ---------------- Inline PTX Assembly Macros ----------------
// cp.async: async copy 16 bytes from gmem (src) to smem (dst_smem_32b)
#define CP_ASYNC_CG(dst_smem_32b, src_global_ptr)                                                                      \
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n" ::"r"(dst_smem_32b), "l"(src_global_ptr))

#define CP_ASYNC_COMMIT_GROUP() asm volatile("cp.async.commit_group;\n" ::)
#define CP_ASYNC_WAIT_GROUP_0() asm volatile("cp.async.wait_group 0;\n" ::)

// ldmatrix
#define LDMATRIX_X4(R0, R1, R2, R3, PTR)                                                                               \
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"                                    \
                 : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3)                                                              \
                 : "r"(PTR))

#define LDMATRIX_X2(R0, R1, PTR)                                                                                       \
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];" : "=r"(R0), "=r"(R1) : "r"(PTR))

// mma.sync
#define M16N8K8(C0, C1, C2, C3, A0, A1, A2, A3, B0, B1)                                                                \
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "                                                 \
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"                                         \
                 : "=f"(C0), "=f"(C1), "=f"(C2), "=f"(C3)                                                              \
                 : "r"(A0), "r"(A1), "r"(A2), "r"(A3), "r"(B0), "r"(B1), "f"(C0), "f"(C1), "f"(C2), "f"(C3))


// ------------------------------------------ b trans + ldmatrix  ----------------------------------------------------

// 一个CTA计算128x64；8个warp按2x4排列，每warp负责64x16。
template <const int BM = 128, const int BN = 64, const int BK = 32>
__global__ void sgemm_tf32_swizzle_bcf_dbf_kernel(float *a, float *b, float *c, int m, int n, int k) {
    static_assert(BM == 128 && BN == 64 && BK == 32);
    constexpr int Slices = BK / 16;
    // grid swizzling
    int linear_id = blockIdx.y * gridDim.x + blockIdx.x;
    const int SWIZZLE_W = 8; // grid swizzle tile width

    int bx = (linear_id % SWIZZLE_W) + (linear_id / (SWIZZLE_W * gridDim.y)) * SWIZZLE_W;
    int by = (linear_id / SWIZZLE_W) % gridDim.y;

    // int bx = blockIdx.x, by = blockIdx.y;
    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;

    int load_a_row = tid / 4;
    int load_a_col = (tid % 4) * 4;
    int load_b_row = tid / 16;
    int load_b_col = (tid % 16) * 4;

    // double buffer
    // BK32 requires opt-in dynamic shared memory. Each 16-wide slab keeps
    // the original A row stride and XOR mapping; do not use A[128][32].
    extern __shared__ __align__(16) unsigned char shared_bytes[];
    auto As = reinterpret_cast<float (*)[Slices][BM][16]>(shared_bytes);
    auto Bs = reinterpret_cast<float (*)[Slices][16][BN]>(
        shared_bytes + 2 * Slices * BM * 16 * sizeof(float));

    // 2x4 warp tiling
    int warp_id_m = warp_id / 4;
    int warp_id_n = warp_id % 4;

    float sum[4][2][4] = {0.f};
    // ------------------------------------------- Prefetch first tile
    // 每个16宽K slab：A仍两次16B拷贝，B改为一次；256线程覆盖B[16][64]。
    auto copy_tile = [&](int slot, int kk) {
#pragma unroll
        for (int slab = 0; slab < Slices; ++slab) {
    int a_swizzle_col_0 = SWIZZLE_A(load_a_row, load_a_col);
    int a_swizzle_col_1 = SWIZZLE_A(load_a_row + 64, load_a_col);

    uint32_t smem_a0 = static_cast<uint32_t>(__cvta_generic_to_shared(&As[slot][slab][load_a_row][a_swizzle_col_0]));
    uint32_t smem_a1 = static_cast<uint32_t>(__cvta_generic_to_shared(&As[slot][slab][load_a_row + 64][a_swizzle_col_1]));

    float *global_a0 = &a[(by * BM + load_a_row) * k + kk + slab * 16 + load_a_col];
    float *global_a1 = &a[(by * BM + load_a_row + 64) * k + kk + slab * 16 + load_a_col];
    CP_ASYNC_CG(smem_a0, global_a0);
    CP_ASYNC_CG(smem_a1, global_a1);

    uint32_t smem_b0 =
        static_cast<uint32_t>(__cvta_generic_to_shared(&Bs[slot][slab][load_b_row][SWIZZLE_B_F2(load_b_row, load_b_col)]));
    float *global_b0 = &b[(kk + slab * 16 + load_b_row) * n + bx * BN + load_b_col];

    CP_ASYNC_CG(smem_b0, global_b0);

        }
    };
    copy_tile(0, 0);

    CP_ASYNC_COMMIT_GROUP();
    CP_ASYNC_WAIT_GROUP_0();
    __syncthreads();

    int read_idx = 0;
    int write_idx = 1;

    // main loop
    for (int bk = BK; bk < k; bk += BK) {

        copy_tile(write_idx, bk);
        CP_ASYNC_COMMIT_GROUP();

        // 2. Tensor Core compute using current read_idx buffer
#pragma unroll
        for (int k_step = 0; k_step < BK / 8; ++k_step) {
            int slab = k_step / 2;
        int k_offset = (k_step % 2) * 8;
            uint32_t reg_a[4][4];
            uint32_t reg_b[2][2];

            // 4×16: 64 rows in M dimension
#pragma unroll
            for (int m_idx = 0; m_idx < 4; ++m_idx) {
                int a_row = warp_id_m * 64 + m_idx * 16 + (lane_id % 16);
                int a_col = k_offset + (lane_id / 16) * 4;
                uint32_t smem_addr =
                    static_cast<uint32_t>(__cvta_generic_to_shared(&As[read_idx][slab][a_row][SWIZZLE_A(a_row, a_col)]));
                LDMATRIX_X4(reg_a[m_idx][0], reg_a[m_idx][1], reg_a[m_idx][2], reg_a[m_idx][3], smem_addr);
            }

#pragma unroll
            for (int n_idx = 0; n_idx < 2; ++n_idx) {
                int n_base = warp_id_n * 16 + n_idx * 8;
                int b_col = n_base + (lane_id / 4);
                int b_row_0 = k_offset + (lane_id % 4);
                int b_row_1 = k_offset + (lane_id % 4) + 4;

                reg_b[n_idx][0] = __float_as_uint(Bs[read_idx][slab][b_row_0][SWIZZLE_B_F2(b_row_0, b_col)]);
                reg_b[n_idx][1] = __float_as_uint(Bs[read_idx][slab][b_row_1][SWIZZLE_B_F2(b_row_1, b_col)]);
            }

#pragma unroll
            for (int m_idx = 0; m_idx < 4; ++m_idx) {
#pragma unroll
                for (int n_idx = 0; n_idx < 2; ++n_idx) {
                    M16N8K8(sum[m_idx][n_idx][0],
                            sum[m_idx][n_idx][1],
                            sum[m_idx][n_idx][2],
                            sum[m_idx][n_idx][3],
                            reg_a[m_idx][0],
                            reg_a[m_idx][1],
                            reg_a[m_idx][2],
                            reg_a[m_idx][3],
                            reg_b[n_idx][0],
                            reg_b[n_idx][1]);
                }
            }
        }

        // 3. cp.async sync
        CP_ASYNC_WAIT_GROUP_0();
        __syncthreads();

        // Swap buffers
        read_idx ^= 1;
        write_idx ^= 1;
    }
    // Process last prefetched tile
#pragma unroll
    for (int k_step = 0; k_step < BK / 8; ++k_step) {
        int slab = k_step / 2;
        int k_offset = (k_step % 2) * 8;
        uint32_t reg_a[4][4];
        uint32_t reg_b[2][2];

#pragma unroll
        for (int m_idx = 0; m_idx < 4; ++m_idx) {
            int a_row = warp_id_m * 64 + m_idx * 16 + (lane_id % 16);
            int a_col = k_offset + (lane_id / 16) * 4;
            uint32_t smem_addr =
                static_cast<uint32_t>(__cvta_generic_to_shared(&As[read_idx][slab][a_row][SWIZZLE_A(a_row, a_col)]));
            LDMATRIX_X4(reg_a[m_idx][0], reg_a[m_idx][1], reg_a[m_idx][2], reg_a[m_idx][3], smem_addr);
        }

#pragma unroll
        for (int n_idx = 0; n_idx < 2; ++n_idx) {
            int n_base = warp_id_n * 16 + n_idx * 8;
            int b_col = n_base + (lane_id / 4);
            int b_row_0 = k_offset + (lane_id % 4);
            int b_row_1 = k_offset + (lane_id % 4) + 4;

            reg_b[n_idx][0] = __float_as_uint(Bs[read_idx][slab][b_row_0][SWIZZLE_B_F2(b_row_0, b_col)]);
            reg_b[n_idx][1] = __float_as_uint(Bs[read_idx][slab][b_row_1][SWIZZLE_B_F2(b_row_1, b_col)]);
        }

#pragma unroll
        for (int m_idx = 0; m_idx < 4; ++m_idx) {
#pragma unroll
            for (int n_idx = 0; n_idx < 2; ++n_idx) {
                M16N8K8(sum[m_idx][n_idx][0],
                        sum[m_idx][n_idx][1],
                        sum[m_idx][n_idx][2],
                        sum[m_idx][n_idx][3],
                        reg_a[m_idx][0],
                        reg_a[m_idx][1],
                        reg_a[m_idx][2],
                        reg_a[m_idx][3],
                        reg_b[n_idx][0],
                        reg_b[n_idx][1]);
            }
        }
    }

    // ---------------- Write back C ----------------
    int t_row = lane_id / 4;
    int t_col = (lane_id % 4) * 2;

#pragma unroll
    for (int m_idx = 0; m_idx < 4; ++m_idx) {
#pragma unroll
        for (int n_idx = 0; n_idx < 2; ++n_idx) {
            int c_base_row = by * BM + warp_id_m * 64 + m_idx * 16;
            int c_base_col = bx * BN + warp_id_n * 16 + n_idx * 8;

            FLOAT2(c[(c_base_row + t_row) * n + c_base_col + t_col]) = FLOAT2(sum[m_idx][n_idx][0]);
            FLOAT2(c[(c_base_row + t_row + 8) * n + c_base_col + t_col]) = FLOAT2(sum[m_idx][n_idx][2]);
        }
    }

}

}
#undef SWIZZLE_A
#undef SWIZZLE_B
#undef SWIZZLE_B_F2
#undef CP_ASYNC_CG
#undef CP_ASYNC_COMMIT_GROUP
#undef CP_ASYNC_WAIT_GROUP_0
#undef LDMATRIX_X4
#undef LDMATRIX_X2
#undef M16N8K8
