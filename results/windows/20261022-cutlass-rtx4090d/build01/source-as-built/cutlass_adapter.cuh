// 独立 CUTLASS 3.5.1 对照：只包装公开 device::Gemm，不复制或改写其 device 算法。
// 配置来源、许可证与差异见同目录 CUTLASS-PROVENANCE.md / CUTLASS-LICENSE.txt。
// TensorOp 的 tfloat32_t 接收已经量化的 float 存储；不是 OpMultiplyAddFastF32/TF32x3。
#pragma once

#include <cuda_runtime.h>
#include <cstddef>
#include <stdexcept>
#include <string>
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/version.h"

static_assert(CUTLASS_MAJOR == 3 && CUTLASS_MINOR == 5 && CUTLASS_PATCH == 1,
              "This comparison is pinned to CUTLASS 3.5.1");
static_assert(sizeof(cutlass::tfloat32_t) == sizeof(float), "TF32 storage width");

namespace cutlass_compare {

inline void check_cutlass(cutlass::Status status, const char* operation) {
  if (status != cutlass::Status::kSuccess)
    throw std::runtime_error(std::string(operation) + " failed, CUTLASS status=" +
                             std::to_string(static_cast<int>(status)));
}

inline void check_cuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}

// 生命周期由主机 harness 控制：initialize 在正确性/预热/计时之前完成，
// run 只执行已初始化的上游 run(stream)。禁止拷贝，避免 workspace 重复释放。
template<class Gemm>
class Instance {
 public:
  using Operator = Gemm;
  Instance() = default;
  Instance(const Instance&) = delete;
  Instance& operator=(const Instance&) = delete;
  ~Instance() { if (workspace_) cudaFree(workspace_); }

  void initialize(int m, int n, int k, const float* a, const float* b, float* c,
                  cudaStream_t stream = nullptr) {
    if (initialized_) throw std::runtime_error("CUTLASS instance already initialized");
    if (m <= 0 || n <= 0 || k <= 0 || !a || !b || !c)
      throw std::runtime_error("Invalid CUTLASS matrix dimensions or pointers");
    // Row-major strides: A[m,k] -> k, B[k,n] -> n, C/D[m,n] -> n.
    // TF32 case reinterprets identical quantized bytes; no conversion kernel or extra epilogue.
    typename Gemm::Arguments args(
        {m, n, k},
        {reinterpret_cast<const typename Gemm::ElementA*>(a), k},
        {reinterpret_cast<const typename Gemm::ElementB*>(b), n},
        {c, n}, {c, n}, {1.0f, 0.0f}, 1);
    check_cutlass(Gemm::can_implement(args), "can_implement");
    workspace_bytes_ = Gemm::get_workspace_size(args);
    if (workspace_bytes_)
      check_cuda(cudaMalloc(&workspace_, workspace_bytes_), "workspace cudaMalloc");
    check_cutlass(op_.initialize(args, workspace_, stream), "initialize");
    initialized_ = true;
  }

  void run(cudaStream_t stream = nullptr) {
    if (!initialized_) throw std::runtime_error("CUTLASS instance is not initialized");
    // 上游 run 对 >=48KiB shared 的实例会调用 cudaFuncSetAttribute；原样保留。
    check_cutlass(op_.run(stream), "run");
  }

  std::size_t workspace_size() const { return workspace_bytes_; }
  static constexpr int threads() { return Gemm::GemmKernel::kThreadCount; }
  static constexpr std::size_t shared_storage_bytes() {
    return sizeof(typename Gemm::GemmKernel::SharedStorage);
  }

 private:
  Gemm op_;
  void* workspace_ = nullptr;
  std::size_t workspace_bytes_ = 0;
  bool initialized_ = false;
};

// 官方 TF32 RRR unit-test 配置：FP32 累加、四元素向量化 epilogue，split-K=1。
template<int BN, int BK, int WN, int Stages>
using Tf32Gemm = cutlass::gemm::device::Gemm<
    cutlass::tfloat32_t, cutlass::layout::RowMajor,
    cutlass::tfloat32_t, cutlass::layout::RowMajor,
    float, cutlass::layout::RowMajor, float,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, BN, BK>,
    cutlass::gemm::GemmShape<64, WN, BK>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    cutlass::epilogue::thread::LinearCombination<float, 4, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,
    Stages, 4, 4, false, cutlass::arch::OpMultiplyAdd>;

class Tf32_128x128 : public Instance<Tf32Gemm<128, 32, 64, 3>> {
 public:
  static const char* description() {
    return "CUTLASS3.5.1;TF32x1;RRR;CTA=128x128x32;warp=64x64x32;inst=16x8x8;"
           "stages=3;alignmentAB=4;epilogue=4;swizzle=1;splitK=1;Sm80_compiled_sm89";
  }
};

class Tf32_128x64 : public Instance<Tf32Gemm<64, 16, 32, 4>> {
 public:
  static const char* description() {
    return "CUTLASS3.5.1;TF32x1;RRR;CTA=128x64x16;warp=64x32x16;inst=16x8x8;"
           "stages=4;alignmentAB=4;epilogue=4;swizzle=1;splitK=1;Sm80_compiled_sm89";
  }
};

// 沿用第16章的公开 SIMT 配置；float 原始输入，FFMA，完全独立于 TF32 结果表。
using SimtGemm = cutlass::gemm::device::Gemm<
    float, cutlass::layout::RowMajor, float, cutlass::layout::RowMajor,
    float, cutlass::layout::RowMajor, float,
    cutlass::arch::OpClassSimt, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<256, 128, 8>,
    cutlass::gemm::GemmShape<64, 64, 8>,
    cutlass::gemm::GemmShape<1, 1, 1>,
    cutlass::epilogue::thread::LinearCombination<float, 1, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,
    4, 1, 1, false, cutlass::arch::OpMultiplyAdd>;

class Simt_256x128 : public Instance<SimtGemm> {
 public:
  static const char* description() {
    return "CUTLASS3.5.1;SIMT_FP32;RRR;CTA=256x128x8;warp=64x64x8;inst=1x1x1;"
           "stages=4;alignmentAB=1;epilogue=1;swizzle=1;splitK=1;Sm80_compiled_sm89";
  }
};

} // namespace cutlass_compare
