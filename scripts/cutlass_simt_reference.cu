// Independent reference experiment; does not change the production dispatcher.
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <nvToolsExt.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include "cutlass/gemm/device/gemm.h"
#include "../src/kernels/10_kernel_warptiling.cuh"
#include "../src/kernels/21_kernel_cutlass_style_skeleton.cuh"

#define CUDA_OK(expr) do { auto e = (expr); if (e != cudaSuccess) { \
  fprintf(stderr, "%s: %s\n", #expr, cudaGetErrorString(e)); exit(1); } } while (0)
#define BLAS_OK(expr) do { if ((expr) != CUBLAS_STATUS_SUCCESS) { \
  fprintf(stderr, "cuBLAS failed: %s\n", #expr); exit(1); } } while (0)
#define CUT_OK(expr) do { auto e = (expr); if (e != cutlass::Status::kSuccess) { \
  fprintf(stderr, "CUTLASS failed: %s (%d)\n", #expr, int(e)); exit(1); } } while (0)

using Reference = cutlass::gemm::device::Gemm<
    float, cutlass::layout::RowMajor, float, cutlass::layout::RowMajor,
    float, cutlass::layout::RowMajor, float,
    cutlass::arch::OpClassSimt, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<256, 128, 8>,
    cutlass::gemm::GemmShape<64, 64, 8>,
    cutlass::gemm::GemmShape<1, 1, 1>,
    cutlass::epilogue::thread::LinearCombination<float, 1, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,
    4, 1, 1, false, cutlass::arch::OpMultiplyAdd>;

void initialize(std::vector<float>& a, unsigned index) {
  srand(20261013U + 0x9e3779b9U * index);
  for (auto &v : a) {
    float t = float(rand() % 5) + 0.01 * (rand() % 5);
    v = rand() % 2 == 0 ? t : -t;
  }
}

int main(int argc, char** argv) {
  if (argc != 6) {
    fprintf(stderr, "Usage: %s {strict|k10|k21|cutlass} size beta repeat warmup\n", argv[0]);
    return 2;
  }
  const std::string mode = argv[1];
  const int n = atoi(argv[2]), repeats = atoi(argv[4]), warmup = atoi(argv[5]);
  const float beta = atof(argv[3]), alpha = 0.5f;
  if (n < 128 || n > 4096 || n % 128 || repeats < 1 || warmup < 0 ||
      !(mode == "strict" || mode == "k10" || mode == "k21" || mode == "cutlass") ||
      !(beta == 0 || beta == 1 || beta == 3)) return 2;
  CUDA_OK(cudaSetDevice(0));
  const size_t count = size_t(n) * n, bytes = count * sizeof(float);
  std::vector<float> a(count), b(count), initial(count), expected(count), actual(count);
  initialize(a, 0); initialize(b, 1); initialize(initial, 2);
  float *da, *db, *dc, *di, *dr;
  CUDA_OK(cudaMalloc(&da, bytes)); CUDA_OK(cudaMalloc(&db, bytes));
  CUDA_OK(cudaMalloc(&dc, bytes)); CUDA_OK(cudaMalloc(&di, bytes));
  CUDA_OK(cudaMalloc(&dr, bytes));
  CUDA_OK(cudaMemcpy(da, a.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(db, b.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(di, initial.data(), bytes, cudaMemcpyHostToDevice));
  cublasHandle_t handle;
  BLAS_OK(cublasCreate(&handle));
  BLAS_OK(cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH));
  auto strict = [&](float *out) {
    BLAS_OK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n,
        &alpha, db, CUDA_R_32F, n, da, CUDA_R_32F, n, &beta,
        out, CUDA_R_32F, n, CUBLAS_COMPUTE_32F_PEDANTIC, CUBLAS_GEMM_DEFAULT));
  };
  Reference op;
  Reference::Arguments args({n,n,n}, {da,n}, {db,n}, {dc,n}, {dc,n}, {alpha,beta}, 1);
  CUT_OK(op.can_implement(args));
  CUT_OK(op.initialize(args));
  auto launch = [&]() {
    if (mode == "strict") strict(dc);
    else if (mode == "cutlass") { CUT_OK(op.run()); }
    else if (mode == "k10")
      sgemmWarptiling<128,128,16,64,64,4,8,4,128,4>
          <<<dim3(n/128,n/128),128>>>(n,n,n,alpha,da,db,beta,dc);
    else sgemmK21Iterator<<<dim3(n/128,n/128),128>>>(n,n,n,alpha,da,db,beta,dc);
    CUDA_OK(cudaGetLastError());
  };
  CUDA_OK(cudaMemcpy(dr, di, bytes, cudaMemcpyDeviceToDevice)); strict(dr);
  CUDA_OK(cudaMemcpy(dc, di, bytes, cudaMemcpyDeviceToDevice)); launch();
  CUDA_OK(cudaDeviceSynchronize());
  CUDA_OK(cudaMemcpy(expected.data(), dr, bytes, cudaMemcpyDeviceToHost));
  CUDA_OK(cudaMemcpy(actual.data(), dc, bytes, cudaMemcpyDeviceToHost));
  double max_error = 0;
  for (size_t i=0; i<count; ++i) {
    double e = std::abs(double(actual[i]) - expected[i]);
    if (!std::isfinite(e) || e > 0.01) {
      fprintf(stderr, "FAIL %s n=%d beta=%g index=%zu error=%g\n", mode.c_str(),n,beta,i,e);
      return 1;
    }
    max_error = std::max(max_error,e);
  }
  for (int i=0; i<warmup; ++i) {
    CUDA_OK(cudaMemcpy(dc, di, bytes, cudaMemcpyDeviceToDevice)); launch();
  }
  CUDA_OK(cudaDeviceSynchronize());
  cudaEvent_t start, stop;
  CUDA_OK(cudaEventCreate(&start)); CUDA_OK(cudaEventCreate(&stop));
  double total = 0;
  for (int i=0; i<repeats; ++i) {
    CUDA_OK(cudaMemcpy(dc, di, bytes, cudaMemcpyDeviceToDevice));
    // NCU filters only this launch: validation and warmup are never profiled.
    nvtxRangePushA("measure");
    CUDA_OK(cudaEventRecord(start)); launch(); CUDA_OK(cudaEventRecord(stop));
    nvtxRangePop();
    CUDA_OK(cudaEventSynchronize(stop));
    float ms; CUDA_OK(cudaEventElapsedTime(&ms,start,stop)); total += ms;
  }
  printf("RESULT,%s,%d,%g,%.9f,%.3f,%.9g,PASS\n", mode.c_str(), n, beta,
         total/repeats, 2.0*n*n*n/(total/repeats*1.e6),max_error);
  printf("REFERENCE,CTA=256x128x8,warp=64x64x8,threads=%d,stages=4,shared=%zu,split_k=1,layout=RRR\n",
         Reference::GemmKernel::kThreadCount,sizeof(Reference::GemmKernel::SharedStorage));
  CUDA_OK(cudaEventDestroy(start)); CUDA_OK(cudaEventDestroy(stop));
  BLAS_OK(cublasDestroy(handle));
  CUDA_OK(cudaFree(da)); CUDA_OK(cudaFree(db)); CUDA_OK(cudaFree(dc));
  CUDA_OK(cudaFree(di)); CUDA_OK(cudaFree(dr));
}
