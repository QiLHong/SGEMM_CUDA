// 【文件导读】K30 独立测试程序：同输入验证与 event 计时
// 给初学者：audit 在这里指一套实验检查，不是另一个算法；构建、验算、计时、性能采样是不同步骤。
// 先由对应 build_k30_audit.sh 编译，再由 run_k30_audit.py 批量调用；.cu 源文件不能直接运行。
// 二进制参数：<mode> <方阵边长> <beta> <repeat次数> <warmup次数>；例如 audit v2 4096 3 50 10。
// 模式：strict/k10/v1/v2/v3/v4；v2 是常用 K30 双槽配置。
// 先生成固定种子的 A/B/初始 C，再算 strict FP32 参考，检查候选输出有限且绝对误差 ≤0.01。
// alpha 固定 0.5，beta 只接收 0/1/3；边长 128..4096 且为 128 的倍数，完整六尺寸由 Python 入口枚举。
// 每次预热和计时前都把初始 C 拷回去；复位在 event 外，不把 memcpy 时间算作 GEMM 时间。
// NVTX 的 measure 标签只圈计时调用，使 profiler 不误选前面的验证或预热；RESULT 的误差来自预先验证。
// 选择可见设备 0，不读取 DEVICE；如需选择物理 GPU，可在外部设 CUDA_VISIBLE_DEVICES。
// V1/V2/V3 是 Small 的 1/2/4 stage；V4 是 Large 的 4 stage，不能把 V4 与 V2 的差异只归因于线程数。
// 中文总索引：docs/zh/23-代码与脚本使用索引.md；脚本命令默认在仓库根目录执行。

// K30 独立验证：对照严格 FP32 cuBLAS、未修改的 K10 与精简 SIMT 实现，不依赖 CUTLASS。
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <nvToolsExt.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include "../src/kernels/30_kernel_mini_simt.cuh"
#include "../src/kernels/10_kernel_warptiling.cuh"


#define CUDA_OK(expr) do { auto e = (expr); if (e != cudaSuccess) { \
  fprintf(stderr, "%s: %s\n", #expr, cudaGetErrorString(e)); exit(1); } } while (0)
#define BLAS_OK(expr) do { if ((expr) != CUBLAS_STATUS_SUCCESS) { \
  fprintf(stderr, "cuBLAS failed: %s\n", #expr); exit(1); } } while (0)
// 这里把一长串模板参数起短名字：前 3 项是 block 的 M/N/K 分块，后面是 warp、线程小组和总线程数。
// Small/Large 只是尺寸类型，真正 shared 字节数在发射前按 stage 数算出。
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

// 准备可重复的输入：同一 index 用同一种子，A/B/C 用不同 index。
// 随机数来自 C 库 rand；同一环境可复现，不声称跨不同 C 库实现仍逐位相同。
void initialize(std::vector<float>& a, unsigned index) {
  srand(20261013U + 0x9e3779b9U * index);
  for (auto &v : a) {
    float t = float(rand() % 5) + 0.01 * (rand() % 5);
    v = rand() % 2 == 0 ? t : -t;
  }
}

int main(int argc, char** argv) {
  if (argc != 6) {
    fprintf(stderr, "Usage: %s {strict|k10|v1|v2|v3|v4} size beta repeat warmup\n", argv[0]);
    return 2;
  }
  const std::string mode = argv[1];
  const int n = atoi(argv[2]), repeats = atoi(argv[4]), warmup = atoi(argv[5]);
  const float beta = atof(argv[3]), alpha = 0.5f;
  if (n < 128 || n > 4096 || n % 128 || repeats < 1 || warmup < 0 ||
      !(mode == "strict" || mode == "k10" || mode == "v1" || mode == "v2" || mode == "v3" || mode == "v4") ||
      !(beta == 0 || beta == 1 || beta == 3)) return 2;
  CUDA_OK(cudaSetDevice(0));
  if(mode=="v1") configureK30<Small,1>();
  if(mode=="v2") configureK30<Small,2>();
  if(mode=="v3") configureK30<Small,4>();
  if(mode=="v4") configureK30<Large,4>();
  // count 是元素数、bytes 是字节数，cudaMalloc/cudaMemcpy 要字节，不能直接传 n*n。
  // a/b/initial/expected/actual 是 CPU 数组；da/db/di/dr/dc 是 GPU 地址，前缀 d 提醒所在位置。
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
  // strict 参考同时设置严格 math mode 和 COMPUTE_32F_PEDANTIC；float 输入本身不足以保证没有 TF32。
  // 下面交换 db/da 是行主序适配，不是把数学式改成 B*A：列主序观察到的是转置后的矩阵。
  BLAS_OK(cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH));
  auto strict = [&](float *out) {
    BLAS_OK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n,
        &alpha, db, CUDA_R_32F, n, da, CUDA_R_32F, n, &beta,
        out, CUDA_R_32F, n, CUBLAS_COMPUTE_32F_PEDANTIC, CUBLAS_GEMM_DEFAULT));
  };
  // lambda 像一个小的“发射按钮”：根据 mode 选核；这里不计时，主程序在外层决定何时按按钮。
  // cudaGetLastError 检查发射错误；设备端异步错误还要在同步/event 等待时检查。
  auto launch = [&]() {
    if (mode == "strict") strict(dc);
    else if (mode == "k10")
      sgemmWarptiling<128,128,16,64,64,4,8,4,128,4>
          <<<dim3(n/128,n/128),128>>>(n,n,n,alpha,da,db,beta,dc);
    else if(mode=="v1") runK30<Small,1>(n,alpha,da,db,beta,dc);
    else if(mode=="v2") runK30<Small,2>(n,alpha,da,db,beta,dc);
    else if(mode=="v3") runK30<Small,4>(n,alpha,da,db,beta,dc);
    else runK30<Large,4>(n,alpha,da,db,beta,dc);
    CUDA_OK(cudaGetLastError());
  };
  // 先让参考与候选都从完全相同的初始 C 开始，避免 beta*C 不同导致虚假误差。
  // 这是计时前的验算；后面报告的 max_error 来自这次，不是对每个计时 launch 都重新读回验证。
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
  // 预热不进入平均耗时：让 CUDA/cuBLAS 初始化与设备状态先稳定。
  // 每次仍要复位 C；否则 beta=3 时重复更新的是另一道数学题，还可能把数值放大到溢出。
  for (int i=0; i<warmup; ++i) {
    CUDA_OK(cudaMemcpy(dc, di, bytes, cudaMemcpyDeviceToDevice)); launch();
  }
  CUDA_OK(cudaDeviceSynchronize());
  cudaEvent_t start, stop;
  CUDA_OK(cudaEventCreate(&start)); CUDA_OK(cudaEventCreate(&stop));
  double total = 0;
  // CUDA Event 是 GPU 时间标记；CPU 调用 launch 很快返回，并不表示计算已经完成。
  // 先复位，再记录 start，启动核，记录 stop，等 stop 完成后读耗时；D2D 复位不计入 GEMM 时间。
  for (int i=0; i<repeats; ++i) {
    CUDA_OK(cudaMemcpy(dc, di, bytes, cudaMemcpyDeviceToDevice));
    // NCU 只筛选这次启动进行采样；正确性验证与预热启动不在采样范围内。
    nvtxRangePushA("measure");
    CUDA_OK(cudaEventRecord(start)); launch(); CUDA_OK(cudaEventRecord(stop));
    nvtxRangePop();
    CUDA_OK(cudaEventSynchronize(stop));
    float ms; CUDA_OK(cudaEventElapsedTime(&ms,start,stop)); total += ms;
  }
  // 固定 CSV 协议：RESULT,模式,边长,beta,平均毫秒,GFLOPS,最大绝对误差,PASS。
  // 一次矩阵乘加按约 2*n^3 个浮点操作计数；除以 ms*1e6 得 GFLOPS，不是内存带宽。
  // Python 脚本依赖这个字段顺序，不要为了改中文显示而改变 RESULT 行。
  printf("RESULT,%s,%d,%g,%.9f,%.3f,%.9g,PASS\n", mode.c_str(), n, beta,
         total/repeats, 2.0*n*n*n/(total/repeats*1.e6),max_error);

  CUDA_OK(cudaEventDestroy(start)); CUDA_OK(cudaEventDestroy(stop));
  BLAS_OK(cublasDestroy(handle));
  CUDA_OK(cudaFree(da)); CUDA_OK(cudaFree(db)); CUDA_OK(cudaFree(dc));
  CUDA_OK(cudaFree(di)); CUDA_OK(cudaFree(dr));
}
