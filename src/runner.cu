#include "kernels.cuh"
#include "runner.cuh"
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>

// 【学习补充】本文件是“主机调用”和“GPU kernel”之间的桥梁。
// 阅读顺序：末尾 run_kernel 的编号分发 → 对应 runSgemm... → 目标 kernel。
// 此处的模板参数决定分块形状；只读 kernel 而不读 launch 配置，会漏掉重要前提。
// 为兼容现有脚本，函数名、错误字符串、性能输出和参数声明均保持上游形式。

// 返回微秒量级的主机时间；不要因函数名含 sec 就直接按秒使用。
float get_current_sec() {
  struct timeval time;
  gettimeofday(&time, NULL);
  return (1e6 * time.tv_sec + time.tv_usec);
}

float cpu_elapsed_time(float &beg, float &end) { return 1.0e-6 * (end - beg); }

void cudaCheck(cudaError_t error, const char *file, int line) {
  if (error != cudaSuccess) {
    printf("[CUDA ERROR] at file %s:%d:\n%s\n", file, line,
           cudaGetErrorString(error));
    exit(EXIT_FAILURE);
  }
};

namespace {
void cublasCheck(cublasStatus_t status, const char *operation) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "[cuBLAS ERROR] %s failed with status %d\n", operation,
            static_cast<int>(status));
    exit(EXIT_FAILURE);
  }
}

void runCublasGemm(cublasHandle_t handle, cublasMath_t mathMode,
                   cublasComputeType_t computeType, cublasGemmAlgo_t algorithm,
                   int M, int N, int K, float alpha, float *A, float *B,
                   float beta, float *C) {
  cublasCheck(cublasSetMathMode(handle, mathMode), "cublasSetMathMode");
  cublasCheck(
      cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B,
                   CUDA_R_32F, N, A, CUDA_R_32F, K, &beta, C, CUDA_R_32F, N,
                   computeType, algorithm),
      "cublasGemmEx");
}
} // namespace

// 打印运行设备资源；资源上限用于理解配置，不等于实际达到的 occupancy。
void CudaDeviceInfo() {
  int deviceId;

  cudaGetDevice(&deviceId);

  cudaDeviceProp props{};
  cudaGetDeviceProperties(&props, deviceId);

  printf("Device ID: %d\n\
    Name: %s\n\
    Compute Capability: %d.%d\n\
    memoryBusWidth: %d\n\
    maxThreadsPerBlock: %d\n\
    maxThreadsPerMultiProcessor: %d\n\
    maxRegsPerBlock: %d\n\
    maxRegsPerMultiProcessor: %d\n\
    totalGlobalMem: %zuMB\n\
    sharedMemPerBlock: %zuKB\n\
    sharedMemPerMultiprocessor: %zuKB\n\
    totalConstMem: %zuKB\n\
    multiProcessorCount: %d\n\
    Warp Size: %d\n",
         deviceId, props.name, props.major, props.minor, props.memoryBusWidth,
         props.maxThreadsPerBlock, props.maxThreadsPerMultiProcessor,
         props.regsPerBlock, props.regsPerMultiprocessor,
         props.totalGlobalMem / 1024 / 1024, props.sharedMemPerBlock / 1024,
         props.sharedMemPerMultiprocessor / 1024, props.totalConstMem / 1024,
         props.multiProcessorCount, props.warpSize);
};

void randomize_matrix(float *mat, int N) {
  // 原作者提醒：time(NULL) 精度过低，连续调用可能生成相同随机序列。
  // 改用 gettimeofday 的微秒字段作种子；注意这也意味着每次运行输入未固定。
  if (const char *seedText = getenv("SGEMM_SEED")) {
    // 固定 seed 时，每次调用仍使用不同但可复现的序列，避免 A/B/C 完全相同。
    static unsigned callIndex = 0;
    char *end = nullptr;
    const unsigned long seed = strtoul(seedText, &end, 10);
    if (end == seedText || *end != '\0') {
      fprintf(stderr, "SGEMM_SEED must be an unsigned integer.\n");
      exit(EXIT_FAILURE);
    }
    srand(static_cast<unsigned>(seed) + 0x9e3779b9U * callIndex++);
  } else {
    struct timeval time {};
    gettimeofday(&time, nullptr);
    srand(time.tv_usec);
  }
  for (int i = 0; i < N; i++) {
    float tmp = (float)(rand() % 5) + 0.01 * (rand() % 5);
    tmp = (rand() % 2 == 0) ? tmp : tmp * (-1.);
    mat[i] = tmp;
  }
}

void range_init_matrix(float *mat, int N) {
  for (int i = 0; i < N; i++) {
    mat[i] = i;
  }
}

void zero_init_matrix(float *mat, int N) {
  for (int i = 0; i < N; i++) {
    mat[i] = 0.0;
  }
}

void copy_matrix(const float *src, float *dest, int N) {
  int i;
  for (i = 0; src + i && dest + i && i < N; i++)
    *(dest + i) = *(src + i);
  if (i != N)
    printf("copy failed at %d while there are %d elements in total.\n", i, N);
}

void print_matrix(const float *A, int M, int N, std::ofstream &fs) {
  int i;
  fs << std::setprecision(2)
     << std::fixed; // 设置两位小数和定点打印格式。
  fs << "[";
  for (i = 0; i < M * N; i++) {
    if ((i + 1) % N == 0)
      fs << std::setw(5) << A[i]; // 设置字段宽度并写入数值。
    else
      fs << std::setw(5) << A[i] << ", ";
    if ((i + 1) % N == 0) {
      if (i + 1 < M * N)
        fs << ";\n";
    }
  }
  fs << "]\n";
}

// 逐元素对照 cuBLAS：拒绝 NaN 差值以及绝对误差超过 0.01 的结果。
// 【学习补充】这是上游的固定校验策略，不是逐位相等或通用相对误差指标。
bool verify_matrix(float *matRef, float *matOut, int N) {
  double diff = 0.0;
  int i;
  for (i = 0; i < N; i++) {
    diff = std::fabs(matRef[i] - matOut[i]);
    if (isnan(diff) || diff > 0.01) {
      printf("Divergence! Should %5.2f, Is %5.2f (Diff %5.2f) at %d\n",
             matRef[i], matOut[i], diff, i);
      return false;
    }
  }
  return true;
}

int div_ceil(int numerator, int denominator) {
  std::div_t res = std::div(numerator, denominator);
  return res.rem ? (res.quot + 1) : res.quot;
}

void runCublasFP32(cublasHandle_t handle, int M, int N, int K, float alpha,
                   float *A, float *B, float beta, float *C) {
  // cuBLAS 按列主序解释数组。行主序 A/B 的同一内存可解释成其转置，
  // 因为 (B^T*A^T)^T=A*B，所以交换操作数并传入 N/M，无需实际转置拷贝。
  // 这里选择 CUBLAS_COMPUTE_32F，是主 benchmark 的 FP32 基线。
  runCublasGemm(handle, CUBLAS_DEFAULT_MATH, CUBLAS_COMPUTE_32F,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP, M, N, K, alpha, A, B, beta, C);
}

void runCublasStrictFP32(cublasHandle_t handle, int M, int N, int K,
                         float alpha, float *A, float *B, float beta,
                         float *C) {
  // 严格对照：禁止降精度路径，乘法与累加均遵守 FP32 pedantic 语义。
  runCublasGemm(handle, CUBLAS_PEDANTIC_MATH, CUBLAS_COMPUTE_32F_PEDANTIC,
                CUBLAS_GEMM_DEFAULT, M, N, K, alpha, A, B, beta, C);
}

void runCublasBF16(cublasHandle_t handle, int M, int N, int K, float alpha,
                   float *A, float *B, float beta, float *C) {
  // 混合精度路径：允许将 FP32 输入转换为 BF16 进行乘法。
  // 原作者注释给出约 4 倍加速，这是其观察值，不保证在每个尺寸/GPU 上成立。
  // 该函数不是 run_kernel 中编号 0 的路径，不能混用数值契约比较。
  runCublasGemm(handle, CUBLAS_DEFAULT_MATH, CUBLAS_COMPUTE_32F_FAST_16BF,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP, M, N, K, alpha, A, B, beta, C);
}

void runCublasTF32(cublasHandle_t handle, int M, int N, int K, float alpha,
                   float *A, float *B, float beta, float *C) {
  // 原注释复制了上一函数的 BF16/~4x 说明。
  // 【注释辨析】实际枚举为 FAST_TF32，表示允许 TF32 快速计算，而不是 BF16。
  // 保留代码；不要把这条混合精度路径与普通 FP32 kernel 当作同精度结果。
  runCublasGemm(handle, CUBLAS_TF32_TENSOR_OP_MATH,
                CUBLAS_COMPUTE_32F_FAST_TF32,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP, M, N, K, alpha, A, B, beta, C);
}

void run_sgemm_naive(int M, int N, int K, float alpha, float *A, float *B,
                     float beta, float *C) {
  // Kernel 1：二维 block，threadIdx.x 对应输出行；与 Kernel 2 的映射作对照。
  dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
  dim3 blockDim(32, 32);
  sgemm_naive<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

void run_sgemm_coalesce(int M, int N, int K, float alpha, float *A, float *B,
                        float beta, float *C) {
  // Kernel 2：同样 1024 个线程，但改用一维编号，再按除法/取余铺成输出块。
  dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
  dim3 blockDim(32 * 32);
  sgemm_global_mem_coalesce<32>
      <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

void run_sgemm_shared_mem_block(int M, int N, int K, float alpha, float *A,
                                float *B, float beta, float *C) {
  dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
  dim3 blockDim(32 * 32);
  // 原作者认为：此访问模式先经 shared memory 缓存，因此更偏好把片上容量给 SMEM。
  // 其测试中 occupancy 受寄存器/线程数限制，调整 carveout 暂未带来差异。
  // 【学习补充】这是针对本实现的偏好，不应推广成“使用 SMEM 后 L1 总是无用”。
  cudaFuncSetAttribute(sgemm_shared_mem_block<32>,
                       cudaFuncAttributePreferredSharedMemoryCarveout,
                       cudaSharedmemCarveoutMaxShared);
  sgemm_shared_mem_block<32>
      <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

void runSgemm1DBlocktiling(int M, int N, int K, float alpha, float *A, float *B,
                           float beta, float *C) {
  const uint BM = 64;
  const uint BN = 64;
  const uint BK = 8;
  const uint TM = 8;
  // 从 Kernel 4 起，grid.x 对应输出列块，grid.y 对应行块。
  dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
  dim3 blockDim((BM * BN) / TM);
  sgemm1DBlocktiling<BM, BN, BK, TM>
      <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

void runSgemm2DBlocktiling(int M, int N, int K, float alpha, float *A, float *B,
                           float beta, float *C) {
  const uint BK = 8;
  const uint TM = 8;
  const uint TN = 8;
  if (M >= 128 and N >= 128) {
    const uint BM = 128;
    const uint BN = 128;
    dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    dim3 blockDim((BM * BN) / (TM * TN));
    sgemm2DBlocktiling<BM, BN, BK, TM, TN>
        <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
  } else {
    // 原作者的临时小块方案：kernel 缺少完整边界检查。
    // 选 64 并不自动支持任意小尺寸；仍须检查整除和搬运覆盖。
    const uint BM = 64;
    const uint BN = 64;
    dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    dim3 blockDim((BM * BN) / (TM * TN));
    sgemm2DBlocktiling<BM, BN, BK, TM, TN>
        <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
  }
}

void runSgemmVectorize(int M, int N, int K, float alpha, float *A, float *B,
                       float beta, float *C) {
  const uint BK = 8;
  const uint TM = 8;
  const uint TN = 8;
  if (M >= 128 and N >= 128) {
    const uint BM = 128;
    const uint BN = 128;
    dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    dim3 blockDim((BM * BN) / (TM * TN));
    sgemmVectorize<BM, BN, BK, TM, TN>
        <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
  } else {
    // 原作者临时缩小分块以应对缺少边界检查的问题；不是通用 fallback。
    // 【学习补充】还必须核对每线程一个 float4 是否能覆盖完整 As/Bs。
    const uint BM = 64;
    const uint BN = 64;
    dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    dim3 blockDim((BM * BN) / (TM * TN));
    sgemmVectorize<BM, BN, BK, TM, TN>
        <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
  }
}

void runSgemmResolveBankConflicts(int M, int N, int K, float alpha, float *A,
                                  float *B, float beta, float *C) {
  const uint BK = 8;
  const uint TM = 8;
  const uint TN = 8;
  if (M >= 128 and N >= 128) {
    const uint BM = 128;
    const uint BN = 128;
    dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    dim3 blockDim((BM * BN) / (TM * TN));
    sgemmResolveBankConflicts<BM, BN, BK, TM, TN>
        <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
  } else {
    // 原作者的临时小块方案；kernel 内布局含 8/16 常量，不能仅缩小 BM/BN 就保证正确。
    // 本分支保留原逻辑，学习时应把此处与 7 号 kernel 的索引共同检查。
    const uint BM = 64;
    const uint BN = 64;
    dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    dim3 blockDim((BM * BN) / (TM * TN));
    sgemmResolveBankConflicts<BM, BN, BK, TM, TN>
        <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
  }
}

void runSgemmResolveBankExtraCol(int M, int N, int K, float alpha, float *A,
                                 float *B, float beta, float *C) {
  const uint BK = 8;
  const uint TM = 8;
  const uint TN = 8;
  if (M >= 128 and N >= 128) {
    const uint BM = 128;
    const uint BN = 128;
    dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    dim3 blockDim((BM * BN) / (TM * TN));
    sgemmResolveBankExtraCol<BM, BN, BK, TM, TN>
        <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
  } else {
    // 原作者的临时方案；缺少完善边界检查，仍须满足子块与加载覆盖前提。
    const uint BM = 64;
    const uint BN = 64;
    dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    dim3 blockDim((BM * BN) / (TM * TN));
    sgemmResolveBankExtraCol<BM, BN, BK, TM, TN>
        <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
  }
}

void runSgemmAutotuned(int M, int N, int K, float alpha, float *A, float *B,
                       float beta, float *C) {
  // 原作者为 A100 保留的配置（当前未启用）。
  // const uint K9_BK = 16;
  // const uint K9_TM = 4;
  // const uint K9_TN = 4;
  // const uint K9_BM = 64;
  // const uint K9_BN = 64;
  // 当前启用的 A6000 配置；不代表已为 RTX 3080 单独调优。
  const uint K9_BK = 16;
  const uint K9_TM = 8;
  const uint K9_TN = 8;
  const uint K9_BM = 128;
  const uint K9_BN = 128;
  dim3 blockDim(K9_NUM_THREADS);

  // 下面依次约束：一次搬运的行覆盖、线程输出区域整除以及 float4 搬运覆盖。
  // 错误字符串中的 quantization 指分块离散化/取整问题，不是低精度权重量化。
  // 原错误消息中的 As/Bs、参数编号等文字可能陈旧，以断言表达式为准。
  static_assert(
      (K9_NUM_THREADS * 4) % K9_BK == 0,
      "NUM_THREADS*4 must be multiple of K9_BK to avoid quantization issues "
      "during GMEM->SMEM tiling (loading only parts of the final row of Bs "
      "during each iteraion)");
  static_assert(
      (K9_NUM_THREADS * 4) % K9_BN == 0,
      "NUM_THREADS*4 must be multiple of K9_BN to avoid quantization issues "
      "during GMEM->SMEM tiling (loading only parts of the final row of As "
      "during each iteration)");
  static_assert(
      K9_BN % (16 * K9_TN) == 0,
      "K9_BN must be a multiple of 16*K9_TN to avoid quantization effects");
  static_assert(
      K9_BM % (16 * K9_TM) == 0,
      "K9_BM must be a multiple of 16*K9_TM to avoid quantization effects");
  static_assert((K9_BM * K9_BK) % (4 * K9_NUM_THREADS) == 0,
                "K9_BM*K9_BK must be a multiple of 4*256 to vectorize loads");
  static_assert((K9_BN * K9_BK) % (4 * K9_NUM_THREADS) == 0,
                "K9_BN*K9_BK must be a multiple of 4*256 to vectorize loads");

  dim3 gridDim(CEIL_DIV(N, K9_BN), CEIL_DIV(M, K9_BM));
  sgemmAutotuned<K9_BM, K9_BN, K9_BK, K9_TM, K9_TN>
      <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

void runSgemmWarptiling(int M, int N, int K, float alpha, float *A, float *B,
                        float beta, float *C) {
  // A100 配置（保留为注释）。
  // const uint K10_NUM_THREADS = 128;
  // const uint K10_BN = 128;
  // const uint K10_BM = 64;
  // const uint K10_BK = 16;
  // const uint K10_WN = 64;
  // const uint K10_WM = 32;
  // const uint K10_WNITER = 1;
  // const uint K10_TN = 4;
  // const uint K10_TM = 4;
  // block/warp tile 沿用 A6000 配置；RTX 3080 实测额外使用 AS_PAD=4。
  // SGEMM_K10_* 编译期定义让调优器覆盖参数，不再需要 sed 原地改写本文件。
#ifndef SGEMM_K10_NUM_THREADS
#define SGEMM_K10_NUM_THREADS 128
#endif
#ifndef SGEMM_K10_BN
#define SGEMM_K10_BN 128
#endif
#ifndef SGEMM_K10_BM
#define SGEMM_K10_BM 128
#endif
#ifndef SGEMM_K10_BK
#define SGEMM_K10_BK 16
#endif
#ifndef SGEMM_K10_WN
#define SGEMM_K10_WN 64
#endif
#ifndef SGEMM_K10_WM
#define SGEMM_K10_WM 64
#endif
#ifndef SGEMM_K10_WNITER
#define SGEMM_K10_WNITER 4
#endif
#ifndef SGEMM_K10_TN
#define SGEMM_K10_TN 4
#endif
#ifndef SGEMM_K10_TM
#define SGEMM_K10_TM 8
#endif
#ifndef SGEMM_K10_AS_PAD
#define SGEMM_K10_AS_PAD 4
#endif
  constexpr uint K10_NUM_THREADS = SGEMM_K10_NUM_THREADS;
  constexpr uint K10_BN = SGEMM_K10_BN;
  constexpr uint K10_BM = SGEMM_K10_BM;
  constexpr uint K10_BK = SGEMM_K10_BK;
  constexpr uint K10_WN = SGEMM_K10_WN;
  constexpr uint K10_WM = SGEMM_K10_WM;
  constexpr uint K10_WNITER = SGEMM_K10_WNITER;
  constexpr uint K10_TN = SGEMM_K10_TN;
  constexpr uint K10_TM = SGEMM_K10_TM;
  constexpr uint K10_AS_PAD = SGEMM_K10_AS_PAD;
  if (M <= 0 || N <= 0 || K <= 0 || M % K10_BM != 0 ||
      N % K10_BN != 0 || K % K10_BK != 0 || N % 4 != 0 || K % 4 != 0) {
    throw std::invalid_argument(
        "Kernel 10 requires positive dimensions divisible by BM/BN/BK and "
        "float4-aligned N/K");
  }
  dim3 blockDim(K10_NUM_THREADS);

  constexpr uint NUM_WARPS = K10_NUM_THREADS / 32;

  // warp tile 必须恰好铺满 block tile；这里 4 个 warp 覆盖 2×2 个区域。
  static_assert((K10_BN % K10_WN == 0) and (K10_BM % K10_WM == 0));
  static_assert((K10_BN / K10_WN) * (K10_BM / K10_WM) == NUM_WARPS);

  // warp 子块中的线程覆盖关系。
  static_assert((K10_WM * K10_WN) % (WARPSIZE * K10_TM * K10_TN * K10_WNITER) ==
                0);
  constexpr uint K10_WMITER =
      (K10_WM * K10_WN) / (32 * K10_TM * K10_TN * K10_WNITER);
  // 子块必须整除 warp tile；当前参数推导 WMITER=1。
  static_assert((K10_WM % K10_WMITER == 0) and (K10_WN % K10_WNITER == 0));

  // 搬运与向量化的整除约束；原字符串中的 K9/256 文案未更名。
  static_assert((K10_NUM_THREADS * 4) % K10_BK == 0,
                "NUM_THREADS*4 must be multiple of K9_BK to avoid quantization "
                "issues during GMEM->SMEM tiling (loading only parts of the "
                "final row of Bs during each iteraion)");
  static_assert((K10_NUM_THREADS * 4) % K10_BN == 0,
                "NUM_THREADS*4 must be multiple of K9_BN to avoid quantization "
                "issues during GMEM->SMEM tiling (loading only parts of the "
                "final row of As during each iteration)");
  static_assert(K10_BN % (16 * K10_TN) == 0,
                "BN must be a multiple of 16*TN to avoid quantization effects");
  static_assert(K10_BM % (16 * K10_TM) == 0,
                "BM must be a multiple of 16*TM to avoid quantization effects");
  static_assert((K10_BM * K10_BK) % (4 * K10_NUM_THREADS) == 0,
                "BM*BK must be a multiple of 4*NUM_THREADS");
  static_assert((K10_BN * K10_BK) % (4 * K10_NUM_THREADS) == 0,
                "BN*BK must be a multiple of 4*NUM_THREADS");

  dim3 gridDim(CEIL_DIV(N, K10_BN), CEIL_DIV(M, K10_BM));
  sgemmWarptiling<K10_BM, K10_BN, K10_BK, K10_WM, K10_WN, K10_WNITER, K10_TM,
                  K10_TN, K10_NUM_THREADS, K10_AS_PAD>
      <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

void runSgemmWarptilingGrouped(int M, int N, int K, float alpha, float *A,
                               float *B, float beta, float *C) {
  constexpr uint NUM_THREADS = SGEMM_K10_NUM_THREADS;
  constexpr uint BN = SGEMM_K10_BN;
  constexpr uint BM = SGEMM_K10_BM;
  constexpr uint BK = SGEMM_K10_BK;
  constexpr uint WN = SGEMM_K10_WN;
  constexpr uint WM = SGEMM_K10_WM;
  constexpr uint WNITER = SGEMM_K10_WNITER;
  constexpr uint TN = SGEMM_K10_TN;
  constexpr uint TM = SGEMM_K10_TM;
  constexpr uint AS_PAD = SGEMM_K10_AS_PAD;
  constexpr uint GROUP_M = SGEMM_K15_GROUP_M;

  if (M <= 0 || N <= 0 || K <= 0 || M % BM != 0 || N % BN != 0 ||
      K % BK != 0 || N % 4 != 0 || K % 4 != 0) {
    throw std::invalid_argument(
        "Kernel 15 requires positive dimensions divisible by BM/BN/BK and "
        "float4-aligned N/K");
  }
  static_assert(GROUP_M > 0);
  constexpr uint NUM_WARPS = NUM_THREADS / WARPSIZE;
  static_assert(NUM_THREADS % WARPSIZE == 0);
  static_assert((BN % WN == 0) and (BM % WM == 0));
  static_assert((BN / WN) * (BM / WM) == NUM_WARPS);
  static_assert((WM * WN) % (WARPSIZE * TM * TN * WNITER) == 0);
  constexpr uint WMITER =
      (WM * WN) / (WARPSIZE * TM * TN * WNITER);
  static_assert((WM % WMITER == 0) and (WN % WNITER == 0));
  static_assert((NUM_THREADS * 4) % BK == 0);
  static_assert((NUM_THREADS * 4) % BN == 0);
  static_assert(BN % (16 * TN) == 0);
  static_assert(BM % (16 * TM) == 0);
  static_assert((BM * BK) % (4 * NUM_THREADS) == 0);
  static_assert((BN * BK) % (4 * NUM_THREADS) == 0);

  const uint gridM = M / BM;
  const uint gridN = N / BN;
  dim3 gridDim(gridM * gridN);
  dim3 blockDim(NUM_THREADS);
  sgemmWarptilingGrouped<BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS,
                         AS_PAD, GROUP_M>
      <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

void runSgemmWarptilingASwizzle(int M, int N, int K, float alpha, float *A,
                                float *B, float beta, float *C) {
  constexpr uint NUM_THREADS = SGEMM_K10_NUM_THREADS;
  constexpr uint BN = SGEMM_K10_BN;
  constexpr uint BM = SGEMM_K10_BM;
  constexpr uint BK = SGEMM_K10_BK;
  constexpr uint WN = SGEMM_K10_WN;
  constexpr uint WM = SGEMM_K10_WM;
  constexpr uint WNITER = SGEMM_K10_WNITER;
  constexpr uint TN = SGEMM_K10_TN;
  constexpr uint TM = SGEMM_K10_TM;
  constexpr uint AS_PAD = SGEMM_K10_AS_PAD;
  constexpr uint XOR_MASK = SGEMM_K16_XOR_MASK;
  constexpr uint XOR_SHIFT = SGEMM_K16_XOR_SHIFT;

  if (M <= 0 || N <= 0 || K <= 0 || M % BM != 0 || N % BN != 0 ||
      K % BK != 0 || N % 4 != 0 || K % 4 != 0) {
    throw std::invalid_argument(
        "Kernel 16 requires positive dimensions divisible by BM/BN/BK and "
        "float4-aligned N/K");
  }
  static_assert(NUM_THREADS % WARPSIZE == 0);
  static_assert((BN % WN == 0) and (BM % WM == 0));
  static_assert((BN / WN) * (BM / WM) == NUM_THREADS / WARPSIZE);
  static_assert((WM * WN) % (WARPSIZE * TM * TN * WNITER) == 0);
  constexpr uint WMITER =
      (WM * WN) / (WARPSIZE * TM * TN * WNITER);
  static_assert((WM % WMITER == 0) and (WN % WNITER == 0));
  static_assert((NUM_THREADS * 4) % BK == 0);
  static_assert((NUM_THREADS * 4) % BN == 0);
  static_assert(BN % (16 * TN) == 0);
  static_assert(BM % (16 * TM) == 0);
  static_assert((BM * BK) % (4 * NUM_THREADS) == 0);
  static_assert((BN * BK) % (4 * NUM_THREADS) == 0);

  dim3 gridDim(N / BN, M / BM);
  dim3 blockDim(NUM_THREADS);
  sgemmWarptilingASwizzle<BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS,
                          AS_PAD, XOR_MASK, XOR_SHIFT>
      <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

void runSgemmWarptilingBAsync(int M, int N, int K, float alpha, float *A,
                              float *B, float beta, float *C) {
  constexpr uint NUM_THREADS = SGEMM_K10_NUM_THREADS;
  constexpr uint BN = SGEMM_K10_BN;
  constexpr uint BM = SGEMM_K10_BM;
  constexpr uint BK = SGEMM_K10_BK;
  constexpr uint WN = SGEMM_K10_WN;
  constexpr uint WM = SGEMM_K10_WM;
  constexpr uint WNITER = SGEMM_K10_WNITER;
  constexpr uint TN = SGEMM_K10_TN;
  constexpr uint TM = SGEMM_K10_TM;
  constexpr uint AS_PAD = SGEMM_K10_AS_PAD;

  if (M <= 0 || N <= 0 || K <= 0 || M % BM != 0 || N % BN != 0 ||
      K % BK != 0 || N % 4 != 0 || K % 4 != 0) {
    throw std::invalid_argument(
        "Kernel 17 requires positive dimensions divisible by BM/BN/BK and "
        "float4-aligned N/K");
  }
  static_assert(NUM_THREADS % WARPSIZE == 0);
  static_assert((BN % WN == 0) and (BM % WM == 0));
  static_assert((BN / WN) * (BM / WM) == NUM_THREADS / WARPSIZE);
  static_assert((WM * WN) % (WARPSIZE * TM * TN * WNITER) == 0);
  constexpr uint WMITER =
      (WM * WN) / (WARPSIZE * TM * TN * WNITER);
  static_assert((WM % WMITER == 0) and (WN % WNITER == 0));
  static_assert((NUM_THREADS * 4) % BK == 0);
  static_assert((NUM_THREADS * 4) % BN == 0);
  static_assert(BN % (16 * TN) == 0);
  static_assert(BM % (16 * TM) == 0);
  static_assert((BM * BK) % (4 * NUM_THREADS) == 0);
  static_assert((BN * BK) % (4 * NUM_THREADS) == 0);

  dim3 gridDim(N / BN, M / BM);
  dim3 blockDim(NUM_THREADS);
  sgemmWarptilingBAsync<BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS,
                        AS_PAD>
      <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

#ifndef SGEMM_K18_VARIANT
#define SGEMM_K18_VARIANT 0
#endif

void runSgemmWarptilingLayout(int M, int N, int K, float alpha, float *A,
                              float *B, float beta, float *C) {
  constexpr uint VARIANT = SGEMM_K18_VARIANT;
  constexpr uint BM = SGEMM_K10_BM;
  constexpr uint BN = SGEMM_K10_BN;
  constexpr uint BK = SGEMM_K10_BK;
  constexpr uint WM = SGEMM_K10_WM;
  constexpr uint WN = SGEMM_K10_WN;
  constexpr uint NUM_THREADS = SGEMM_K10_NUM_THREADS;
  constexpr uint AS_PAD = SGEMM_K10_AS_PAD;
  static_assert(VARIANT <= 5, "SGEMM_K18_VARIANT must be in [0,5]");
  static_assert(BM == 128 && BN == 128 && BK == 16);
  static_assert(WM == 64 && WN == 64 && NUM_THREADS == 128 && AS_PAD == 4);
  static_assert(SGEMM_K10_MIN_BLOCKS == 2,
                "K18 audit freezes launch_bounds(128,2)");
  if (M <= 0 || N <= 0 || K <= 0 || M % BM != 0 || N % BN != 0 ||
      K % BK != 0 || N % 4 != 0 || K % 4 != 0) {
    throw std::invalid_argument(
        "Kernel 18 requires positive dimensions divisible by 128/128/16 "
        "and float4-aligned N/K");
  }

  dim3 gridDim(N / BN, M / BM);
  dim3 blockDim(NUM_THREADS);
  if constexpr (VARIANT == 0 || VARIANT == 1 || VARIANT == 3) {
    sgemmWarptilingLayout<VARIANT, BM, BN, BK, WM, WN, 4, 8, 4,
                          NUM_THREADS, AS_PAD>
        <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
  } else if constexpr (VARIANT == 2) {
    sgemmWarptilingLayout<VARIANT, BM, BN, BK, WM, WN, 4, 4, 8,
                          NUM_THREADS, AS_PAD>
        <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
  } else if constexpr (VARIANT == 4) {
    sgemmWarptilingLayout<VARIANT, BM, BN, BK, WM, WN, 2, 8, 8,
                          NUM_THREADS, AS_PAD>
        <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
  } else {
    sgemmWarptilingLayout<VARIANT, BM, BN, BK, WM, WN, 1, 8, 16,
                          NUM_THREADS, AS_PAD>
        <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
  }
}

void runSgemmDoubleBuffering(int M, int N, int K, float alpha, float *A,
                             float *B, float beta, float *C) {
  // A100 配置（未启用）。
  // const uint K11_NUM_THREADS = 256;
  // const uint K11_BN = 128;
  // const uint K11_BM = 64;
  // const uint K11_BK = 16;
  // const uint K11_WN = 32;
  // const uint K11_WM = 32;
  // const uint K11_WNITER = 2;
  // const uint K11_TN = 4;
  // const uint K11_TM = 4;
  // A6000 配置。
  // 【学习补充】BN=256，但主程序从 N=128 开始；kernel 没有完整 N 边界屏蔽。
  // 这是应独立检查的上游尺寸前提，不要把分配了大缓冲区当成数学边界正确。
  const uint K11_NUM_THREADS = 256;
  const uint K11_BN = 256;
  const uint K11_BM = 128;
  const uint K11_BK = 16;
  const uint K11_WN = 32;
  const uint K11_WM = 128;
  const uint K11_WNITER = 1;
  const uint K11_TN = 8;
  const uint K11_TM = 8;
  dim3 blockDim(K11_NUM_THREADS);

  constexpr uint NUM_WARPS = K11_NUM_THREADS / 32;

  // warp tile 在 block tile 内的完整覆盖。
  static_assert((K11_BN % K11_WN == 0) and (K11_BM % K11_WM == 0));
  static_assert((K11_BN / K11_WN) * (K11_BM / K11_WM) == NUM_WARPS);

  // warp 子块的线程覆盖关系。
  static_assert((K11_WM * K11_WN) % (WARPSIZE * K11_TM * K11_TN * K11_WNITER) ==
                0);
  constexpr uint K11_WMITER =
      (K11_WM * K11_WN) / (32 * K11_TM * K11_TN * K11_WNITER);
  // warp 子块必须整除 warp tile。
  static_assert((K11_WM % K11_WMITER == 0) and (K11_WN % K11_WNITER == 0));

  // Kernel 11 用半个 block 合作加载，所以搬运约束中的线程数也除以 2。
  static_assert((K11_NUM_THREADS / 2 * 4) % K11_BK == 0,
                "NUM_THREADS*4 must be multiple of BK to avoid quantization "
                "issues during GMEM->SMEM tiling (loading only parts of the "
                "final row of Bs during each iteraion)");
  static_assert((K11_NUM_THREADS / 2 * 4) % K11_BN == 0,
                "NUM_THREADS*4 must be multiple of BN to avoid quantization "
                "issues during GMEM->SMEM tiling (loading only parts of the "
                "final row of As during each iteration)");
  static_assert(K11_BN % (16 * K11_TN) == 0,
                "BN must be a multiple of 16*TN to avoid quantization effects");
  static_assert(K11_BM % (16 * K11_TM) == 0,
                "BM must be a multiple of 16*TM to avoid quantization effects");
  static_assert((K11_BM * K11_BK) % (4 * K11_NUM_THREADS / 2) == 0,
                "BM*BK must be a multiple of 4*256 to vectorize loads");
  static_assert((K11_BN * K11_BK) % (4 * K11_NUM_THREADS / 2) == 0,
                "BN*BK must be a multiple of 4*256 to vectorize loads");

  dim3 gridDim(CEIL_DIV(N, K11_BN), CEIL_DIV(M, K11_BM));
  sgemmDoubleBuffering<K11_BM, K11_BN, K11_BK, K11_WM, K11_WN, K11_WNITER,
                       K11_TM, K11_TN, K11_NUM_THREADS>
      <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

void runSgemmDoubleBuffering2(int M, int N, int K, float alpha, float *A,
                              float *B, float beta, float *C) {
  // A6000 配置；中文分支不重调这些参数。
  const uint K12_NUM_THREADS = 128;
  const uint K12_BN = 128;
  const uint K12_BM = 128;
  const uint K12_BK = 16;
  const uint K12_WN = 64;
  const uint K12_WM = 64;
  const uint K12_WNITER = 4;
  const uint K12_TN = 4;
  const uint K12_TM = 8;
  dim3 blockDim(K12_NUM_THREADS);

  constexpr uint NUM_WARPS = K12_NUM_THREADS / 32;

  // warp tile 恰好铺满 block tile。
  static_assert((K12_BN % K12_WN == 0) and (K12_BM % K12_WM == 0));
  static_assert((K12_BN / K12_WN) * (K12_BM / K12_WM) == NUM_WARPS);

  // 线程、warp 子块、warp tile 三层覆盖关系。
  static_assert((K12_WM * K12_WN) % (WARPSIZE * K12_TM * K12_TN * K12_WNITER) ==
                0);
  constexpr uint K12_WMITER =
      (K12_WM * K12_WN) / (32 * K12_TM * K12_TN * K12_WNITER);
  // 子块必须整除整个 warp tile。
  static_assert((K12_WM % K12_WMITER == 0) and (K12_WN % K12_WNITER == 0));

  // 全 block 向量搬运的覆盖条件；这些编译期断言不替代运行时矩阵边界检查。
  static_assert((K12_NUM_THREADS * 4) % K12_BK == 0,
                "NUM_THREADS*4 must be multiple of K9_BK to avoid quantization "
                "issues during GMEM->SMEM tiling (loading only parts of the "
                "final row of Bs during each iteraion)");
  static_assert((K12_NUM_THREADS * 4) % K12_BN == 0,
                "NUM_THREADS*4 must be multiple of K9_BN to avoid quantization "
                "issues during GMEM->SMEM tiling (loading only parts of the "
                "final row of As during each iteration)");
  static_assert(K12_BN % (16 * K12_TN) == 0,
                "BN must be a multiple of 16*TN to avoid quantization effects");
  static_assert(K12_BM % (16 * K12_TM) == 0,
                "BM must be a multiple of 16*TM to avoid quantization effects");
  static_assert((K12_BM * K12_BK) % (4 * K12_NUM_THREADS) == 0,
                "BM*BK must be a multiple of 4*256 to vectorize loads");
  static_assert((K12_BN * K12_BK) % (4 * K12_NUM_THREADS) == 0,
                "BN*BK must be a multiple of 4*256 to vectorize loads");

  dim3 gridDim(CEIL_DIV(N, K12_BN), CEIL_DIV(M, K12_BM));
  runSgemmDoubleBuffering2<K12_BM, K12_BN, K12_BK, K12_WM, K12_WN, K12_WNITER,
                           K12_TM, K12_TN, K12_NUM_THREADS>
      <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

// kernel 编号的唯一分发表。0 保留历史 cuBLAS 口径；13/14 是显式精度对照。
void run_kernel(int kernel_num, int M, int N, int K, float alpha, float *A,
                float *B, float beta, float *C, cublasHandle_t handle) {
  switch (kernel_num) {
  case 0:
    runCublasFP32(handle, M, N, K, alpha, A, B, beta, C);
    break;
  case 1:
    run_sgemm_naive(M, N, K, alpha, A, B, beta, C);
    break;
  case 2:
    run_sgemm_coalesce(M, N, K, alpha, A, B, beta, C);
    break;
  case 3:
    run_sgemm_shared_mem_block(M, N, K, alpha, A, B, beta, C);
    break;
  case 4:
    runSgemm1DBlocktiling(M, N, K, alpha, A, B, beta, C);
    break;
  case 5:
    runSgemm2DBlocktiling(M, N, K, alpha, A, B, beta, C);
    break;
  case 6:
    runSgemmVectorize(M, N, K, alpha, A, B, beta, C);
    break;
  case 7:
    runSgemmResolveBankConflicts(M, N, K, alpha, A, B, beta, C);
    break;
  case 8:
    runSgemmResolveBankExtraCol(M, N, K, alpha, A, B, beta, C);
    break;
  case 9:
    runSgemmAutotuned(M, N, K, alpha, A, B, beta, C);
    break;
  case 10:
    runSgemmWarptiling(M, N, K, alpha, A, B, beta, C);
    break;
  case 11:
    runSgemmDoubleBuffering(M, N, K, alpha, A, B, beta, C);
    break;
  case 12:
    runSgemmDoubleBuffering2(M, N, K, alpha, A, B, beta, C);
    break;
  case 13:
    runCublasStrictFP32(handle, M, N, K, alpha, A, B, beta, C);
    break;
  case 14:
    runCublasTF32(handle, M, N, K, alpha, A, B, beta, C);
    break;
  case 15:
    runSgemmWarptilingGrouped(M, N, K, alpha, A, B, beta, C);
    break;
  case 16:
    runSgemmWarptilingASwizzle(M, N, K, alpha, A, B, beta, C);
    break;
  case 17:
    runSgemmWarptilingBAsync(M, N, K, alpha, A, B, beta, C);
    break;
  case 18:
    runSgemmWarptilingLayout(M, N, K, alpha, A, B, beta, C);
    break;
  default:
    throw std::invalid_argument("Unknown kernel number");
  }
}
