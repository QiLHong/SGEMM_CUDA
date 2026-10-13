// K31 controlled refinement. Same strict FP32 input/validation/timing as K30.
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <nvToolsExt.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include "../src/kernels/31_kernel_mini_simt_refine.cuh"
#include "../src/kernels/10_kernel_warptiling.cuh"


#define CUDA_OK(expr) do { auto e = (expr); if (e != cudaSuccess) { \
  fprintf(stderr, "%s: %s\n", #expr, cudaGetErrorString(e)); exit(1); } } while (0)
#define BLAS_OK(expr) do { if ((expr) != CUBLAS_STATUS_SUCCESS) { \
  fprintf(stderr, "cuBLAS failed: %s\n", #expr); exit(1); } } while (0)
using Small = k30::Shape<128,128,16,64,64,4,4,128>;
using Large = k30::Shape<256,128,16,64,64,4,4,256>;
template<class S,int Stages> void runK30(int n,float alpha,const float *a,const float *b,float beta,float *c) {
  constexpr int shared=(k30::SharedLayoutA<S>::Size+k30::SharedLayoutB<S>::Size)*Stages*sizeof(float);
  sgemmK30<S,Stages><<<dim3(n/S::N,(n+S::M-1)/S::M),S::Threads,shared>>>(n,n,n,alpha,a,b,beta,c);
}
template<class S,int Stages> void configureK30() {
  constexpr int shared=(k30::SharedLayoutA<S>::Size+k30::SharedLayoutB<S>::Size)*Stages*sizeof(float);
  CUDA_OK(cudaFuncSetAttribute(sgemmK30<S,Stages>,cudaFuncAttributeMaxDynamicSharedMemorySize,shared));
}

template<int Variant> void configureK31() {
  using S=typename k31::Config<Variant>::S;
  constexpr int shared=(k30::SharedLayoutA<S>::Size+k30::SharedLayoutB<S>::Size)*k31::Config<Variant>::Stages*4;
  CUDA_OK(cudaFuncSetAttribute(sgemmK31<Variant>,cudaFuncAttributeMaxDynamicSharedMemorySize,shared));
}
template<int Variant> void runK31(int n,float alpha,const float *a,const float *b,float beta,float *c) {
  using S=typename k31::Config<Variant>::S;
  constexpr int shared=(k30::SharedLayoutA<S>::Size+k30::SharedLayoutB<S>::Size)*k31::Config<Variant>::Stages*4;
  sgemmK31<Variant><<<dim3(n/S::N,(n+S::M-1)/S::M),S::Threads,shared>>>(n,n,n,alpha,a,b,beta,c);
}
template<int Limit> void configureBudget() {
  CUDA_OK(cudaFuncSetAttribute(sgemmK31Budget<Limit>,cudaFuncAttributeMaxDynamicSharedMemorySize,33280));
}
template<int Limit> void runBudget(int n,float alpha,const float *a,const float *b,float beta,float *c) {
  sgemmK31Budget<Limit><<<dim3(n/128,n/128),128,33280>>>(n,n,n,alpha,a,b,beta,c);
}
template<class F> void resource(const char *mode,F kernel,int threads,int shared) {
  cudaFuncAttributes attrs;
  CUDA_OK(cudaFuncGetAttributes(&attrs,kernel));
  int blocks;
  CUDA_OK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,kernel,threads,shared));
  printf("RESOURCE,%s,%d,%d,%d,%zu,%d\n",mode,threads,attrs.numRegs,shared,attrs.localSizeBytes,blocks);
}
void resources() {
  CUDA_OK(cudaSetDevice(0));
  configureK30<Small,2>();
  configureK31<0>(); configureK31<1>(); configureK31<2>();
  configureK31<3>(); configureK31<4>(); configureK31<5>();
  resource("k10",sgemmWarptiling<128,128,16,64,64,4,8,4,128,4>,128,0);
  resource("k30",sgemmK30<Small,2>,128,33280);
  resource("control",sgemmK31<0>,128,33280);
  resource("late",sgemmK31<1>,128,33280);
  resource("vectors",sgemmK31<2>,128,33280);
  resource("stage3",sgemmK31<3>,128,49920);
  resource("stage4",sgemmK31<4>,128,66560);
  resource("cta256",sgemmK31<5>,256,49664);
  configureBudget<220>(); resource("r220",sgemmK31Budget<220>,128,33280);
  configureBudget<225>(); resource("r225",sgemmK31Budget<225>,128,33280);
  configureBudget<230>(); resource("r230",sgemmK31Budget<230>,128,33280);
  configureBudget<235>(); resource("r235",sgemmK31Budget<235>,128,33280);
  configureBudget<240>(); resource("r240",sgemmK31Budget<240>,128,33280);
}

void initialize(std::vector<float>& a, unsigned index) {
  srand(20261013U + 0x9e3779b9U * index);
  for (auto &v : a) {
    float t = float(rand() % 5) + 0.01 * (rand() % 5);
    v = rand() % 2 == 0 ? t : -t;
  }
}

int main(int argc, char** argv) {
  if(argc==2 && std::string(argv[1])=="resources") { resources(); return 0; }
  if (argc != 6) {
    fprintf(stderr, "Usage: %s {strict|k10|k30|control|late|vectors|stage3|stage4|cta256|r220|r225|r230|r235|r240} size beta repeat warmup\n", argv[0]);
    return 2;
  }
  const std::string mode = argv[1];
  const int n = atoi(argv[2]), repeats = atoi(argv[4]), warmup = atoi(argv[5]);
  const float beta = atof(argv[3]), alpha = 0.5f;
  if (n < 128 || n > 4096 || n % 128 || repeats < 1 || warmup < 0 ||
      !(mode == "strict" || mode == "k10" || mode=="k30" || mode=="control" || mode=="late" || mode=="vectors" || mode=="stage3" || mode=="stage4" || mode=="cta256" || mode=="r220" || mode=="r225" || mode=="r230" || mode=="r235" || mode=="r240") ||
      !(beta == 0 || beta == 1 || beta == 3)) return 2;
  CUDA_OK(cudaSetDevice(0));
  if(mode=="k30") configureK30<Small,2>();
  if(mode=="control") configureK31<0>();
  if(mode=="late") configureK31<1>();
  if(mode=="vectors") configureK31<2>();
  if(mode=="stage3") configureK31<3>();
  if(mode=="stage4") configureK31<4>();
  if(mode=="cta256") configureK31<5>();
  if(mode=="r220") configureBudget<220>();
  if(mode=="r225") configureBudget<225>();
  if(mode=="r230") configureBudget<230>();
  if(mode=="r235") configureBudget<235>();
  if(mode=="r240") configureBudget<240>();
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
  auto launch = [&]() {
    if (mode == "strict") strict(dc);
    else if (mode == "k10")
      sgemmWarptiling<128,128,16,64,64,4,8,4,128,4>
          <<<dim3(n/128,n/128),128>>>(n,n,n,alpha,da,db,beta,dc);
    else if(mode=="k30") runK30<Small,2>(n,alpha,da,db,beta,dc);
    else if(mode=="control") runK31<0>(n,alpha,da,db,beta,dc);
    else if(mode=="late") runK31<1>(n,alpha,da,db,beta,dc);
    else if(mode=="vectors") runK31<2>(n,alpha,da,db,beta,dc);
    else if(mode=="stage3") runK31<3>(n,alpha,da,db,beta,dc);
    else if(mode=="stage4") runK31<4>(n,alpha,da,db,beta,dc);
    else if(mode=="cta256") runK31<5>(n,alpha,da,db,beta,dc);
    else if(mode=="r220") runBudget<220>(n,alpha,da,db,beta,dc);
    else if(mode=="r225") runBudget<225>(n,alpha,da,db,beta,dc);
    else if(mode=="r230") runBudget<230>(n,alpha,da,db,beta,dc);
    else if(mode=="r235") runBudget<235>(n,alpha,da,db,beta,dc);
    else runBudget<240>(n,alpha,da,db,beta,dc);
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

  CUDA_OK(cudaEventDestroy(start)); CUDA_OK(cudaEventDestroy(stop));
  BLAS_OK(cublasDestroy(handle));
  CUDA_OK(cudaFree(da)); CUDA_OK(cudaFree(db)); CUDA_OK(cudaFree(dc));
  CUDA_OK(cudaFree(di)); CUDA_OK(cudaFree(dr));
}
