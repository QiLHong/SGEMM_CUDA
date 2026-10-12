#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <runner.cuh>
#include <vector>

// 【学习补充】主机端入口：选择 kernel → 分配 A/B/C → 对照 cuBLAS 校验 → 计时。
// 真正的网格/线程块参数在 src/runner.cu，GPU 实现在 src/kernels/。
// 英文输出字符串保持原样，因为绘图和自动调优脚本会解析它们。
#define cudaCheck(err) (cudaCheck(err, __FILE__, __LINE__))

const std::string errLogFile = "matrixValidationFailure.txt";

namespace {
bool read_bool_env(const char *name, bool defaultValue) {
  const char *text = getenv(name);
  if (text == nullptr)
    return defaultValue;
  if (std::string(text) == "0")
    return false;
  if (std::string(text) == "1")
    return true;
  std::cerr << name << " must be 0 or 1." << std::endl;
  exit(EXIT_FAILURE);
}

float read_float_env(const char *name, float defaultValue) {
  const char *text = getenv(name);
  if (text == nullptr)
    return defaultValue;
  char *end = nullptr;
  const float value = strtof(text, &end);
  if (end == text || *end != '\0' || !std::isfinite(value)) {
    std::cerr << name << " must be a finite floating-point value." << std::endl;
    exit(EXIT_FAILURE);
  }
  return value;
}

const char *kernel_mode_name(int kernel) {
  switch (kernel) {
  case 0:
    return "cuBLAS legacy (COMPUTE_32F + DEFAULT_TENSOR_OP)";
  case 13:
    return "cuBLAS strict FP32 (PEDANTIC)";
  case 14:
    return "cuBLAS explicit TF32 Tensor Core";
  case 15:
    return "K15 K10 + grouped grid swizzle";
  case 16:
    return "K16 K10 + A shared-memory XOR swizzle";
  case 17:
    return "K17 K10 + B-only cp.async pipeline";
  case 18:
    return "K18 K10 + warp fragment layout experiment";
  case 19:
    return "K19 K10 + B-fragment register prefetch";
  case 20:
    return "K20 256-thread 256x128 FP32 SIMT";
  case 21:
    return "K21 CUTLASS-style SIMT micro-kernel skeleton";
  case 22:
    return "K22 128-thread BK8 multi-stage pipeline";
  default:
    return "custom CUDA kernel";
  }
}
} // namespace

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "Please select a kernel (0-22; 13=strict FP32 cuBLAS, "
                 "14=TF32 cuBLAS, 15=grouped-grid K10, 16=A-smem swizzle, "
                 "17=B-only cp.async, 18=warp fragment layout, "
                 "19=B-fragment prefetch, 20=256-thread SIMT, "
                 "21=CUTLASS-style skeleton, 22=128-thread multi-stage)"
              << std::endl;
    exit(EXIT_FAILURE);
  }

  // 从命令行读取编号：0 为历史 cuBLAS，1–12 为教学实现，13/14
  // 分别为严格 FP32 与显式 TF32 对照，15–22 为独立受控实验。
  int kernel_num = std::stoi(argv[1]);
  if (kernel_num < 0 || kernel_num > 22) {
    std::cerr << "Please enter a valid kernel number (0-22)" << std::endl;
    exit(EXIT_FAILURE);
  }

  // DEVICE 是此程序自己读取的环境变量；未设置时选设备 0。
  int deviceIdx = 0;
  if (getenv("DEVICE") != NULL) {
    deviceIdx = atoi(getenv("DEVICE"));
  }
  cudaCheck(cudaSetDevice(deviceIdx));

  printf("Running kernel %d on device %d.\n", kernel_num, deviceIdx);
  std::cout << "Mode: " << kernel_mode_name(kernel_num) << std::endl;

  // 按需取消下一行注释，可打印更多设备信息；当前保持原样。
  // CudaDeviceInfo();

  // 创建 cuBLAS 句柄；返回的 cublasStatus_t 为 0 表示成功。
  // 这个句柄既用于基线计时，也用于自写 kernel 的参考计算。
  cublasHandle_t handle;
  if (cublasCreate(&handle)) {
    std::cerr << "Create cublas handle error." << std::endl;
    exit(EXIT_FAILURE);
  };

  // CUDA Event 被记录到 GPU stream 中，用于测量该 stream 上的执行时间。
  // 不能只用 CPU 上提交 kernel 的耗时冒充 GPU 计算耗时。
  float elapsed_time;
  cudaEvent_t beg, end;
  cudaCheck(cudaEventCreate(&beg));
  cudaCheck(cudaEventCreate(&end));

  // 原作者备注：cuBLAS 在 8192 尺寸达到 FLOPs 上限；当前实际只测到 4096。
  // 所有尺寸都保持上游值；特别注意较晚 kernel 的分块约束，不应盲加任意尺寸。
  std::vector<int> SIZE = {128, 256, 512, 1024, 2048, 4096};
  // 调优搜索可以只跑一个尺寸；未设置时仍保持上游六尺寸口径。
  if (const char *benchSize = getenv("SGEMM_BENCH_SIZE")) {
    const int requestedSize = atoi(benchSize);
    if (std::find(SIZE.begin(), SIZE.end(), requestedSize) == SIZE.end()) {
      std::cerr << "SGEMM_BENCH_SIZE must be one of: 128, 256, 512, 1024, "
                   "2048, 4096."
                << std::endl;
      exit(EXIT_FAILURE);
    }
    SIZE = {requestedSize};
  }

  long m, n, k, max_size;
  max_size = SIZE[SIZE.size() - 1];
  std::cout << "Max size: " << max_size << std::endl;

  const float alpha = read_float_env("SGEMM_ALPHA", 0.5f);
  const float beta = read_float_env("SGEMM_BETA", 3.0f);
  const bool resetCEachRepeat =
      read_bool_env("SGEMM_RESET_C_EACH_REPEAT", false);
  // GEMM 参数：C=α*AB+β*C，不只是 C=AB。公平模式每次从同一份 C 开始。
  std::cout << "Timing C policy: "
            << (resetCEachRepeat ? "reset before every measured call"
                                 : "legacy repeated beta*C updates")
            << std::endl;
  std::cout << "Input seed: "
            << (getenv("SGEMM_SEED") ? getenv("SGEMM_SEED") : "time-based")
            << std::endl;
  std::cout << "NVIDIA_TF32_OVERRIDE: "
            << (getenv("NVIDIA_TF32_OVERRIDE") ? getenv("NVIDIA_TF32_OVERRIDE")
                                                : "unset")
            << std::endl;

  float *A = nullptr, *B = nullptr, *C = nullptr,
        *C_ref = nullptr; // 主机（CPU）上的四份矩阵缓冲区。
  float *dA = nullptr, *dB = nullptr, *dC = nullptr, *dC_ref = nullptr,
        *dC_initial = nullptr; // initial 专供公平计时时复位 C。

  // 按最大尺寸一次分配，各个测试尺寸复用这些缓冲区。
  A = (float *)malloc(sizeof(float) * max_size * max_size);
  B = (float *)malloc(sizeof(float) * max_size * max_size);
  C = (float *)malloc(sizeof(float) * max_size * max_size);
  C_ref = (float *)malloc(sizeof(float) * max_size * max_size);
  if (!A || !B || !C || !C_ref) {
    std::cerr << "Host allocation failed." << std::endl;
    exit(EXIT_FAILURE);
  }

  randomize_matrix(A, max_size * max_size);
  randomize_matrix(B, max_size * max_size);
  randomize_matrix(C, max_size * max_size);

  cudaCheck(cudaMalloc((void **)&dA, sizeof(float) * max_size * max_size));
  cudaCheck(cudaMalloc((void **)&dB, sizeof(float) * max_size * max_size));
  cudaCheck(cudaMalloc((void **)&dC, sizeof(float) * max_size * max_size));
  cudaCheck(cudaMalloc((void **)&dC_ref, sizeof(float) * max_size * max_size));
  cudaCheck(
      cudaMalloc((void **)&dC_initial, sizeof(float) * max_size * max_size));

  // 两条计算路径必须从相同的旧 C 开始，因为 beta 非零。
  cudaCheck(cudaMemcpy(dA, A, sizeof(float) * max_size * max_size,
                       cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(dB, B, sizeof(float) * max_size * max_size,
                       cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(dC, C, sizeof(float) * max_size * max_size,
                       cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(dC_ref, C, sizeof(float) * max_size * max_size,
                       cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(dC_initial, C, sizeof(float) * max_size * max_size,
                       cudaMemcpyHostToDevice));

  int repeat_times = 50;
  // 搜索阶段可减少重复次数；候选胜出后应清除此变量并用默认 50 次复测。
  if (const char *benchRepeat = getenv("SGEMM_REPEAT")) {
    repeat_times = atoi(benchRepeat);
    if (repeat_times <= 0) {
      std::cerr << "SGEMM_REPEAT must be a positive integer." << std::endl;
      exit(EXIT_FAILURE);
    }
  }
  int warmup_times = 0;
  // 单尺寸调优需要显式预热 cuBLAS 和候选 kernel，避免把库初始化/JIT
  // 计入第一轮；默认 0 继续保持原 benchmark 的历史输出口径。
  if (const char *benchWarmup = getenv("SGEMM_WARMUP")) {
    warmup_times = atoi(benchWarmup);
    if (warmup_times < 0) {
      std::cerr << "SGEMM_WARMUP must be a non-negative integer." << std::endl;
      exit(EXIT_FAILURE);
    }
  }
  for (int size : SIZE) {
    m = n = k = size;

    std::cout << "dimensions(m=n=k) " << m << ", alpha: " << alpha
              << ", beta: " << beta << std::endl;
    const size_t matrixBytes = sizeof(float) * m * n;
    const bool isCustomKernel =
        (kernel_num >= 1 && kernel_num <= 12) || kernel_num == 15 ||
        kernel_num == 16 || kernel_num == 17 || kernel_num == 18 ||
        kernel_num == 19 || kernel_num == 20 || kernel_num == 21 ||
        kernel_num == 22;
    const bool compareCublasMode =
        resetCEachRepeat && (kernel_num == 0 || kernel_num == 14);

    // 自写 kernel 始终对照严格 FP32 cuBLAS；阈值仍为上游的绝对误差 0.01。
    // 公平模式还报告 legacy/TF32 与严格 FP32 的误差，但不把不同精度伪装成 PASS。
    if (isCustomKernel || compareCublasMode) {
      if (resetCEachRepeat) {
        cudaCheck(cudaMemcpy(dC, dC_initial, matrixBytes,
                             cudaMemcpyDeviceToDevice));
        cudaCheck(cudaMemcpy(dC_ref, dC_initial, matrixBytes,
                             cudaMemcpyDeviceToDevice));
      }
      run_kernel(13, m, n, k, alpha, dA, dB, beta, dC_ref,
                 handle); // 严格 FP32 参考输出。
      run_kernel(kernel_num, m, n, k, alpha, dA, dB, beta, dC,
                 handle);
      cudaCheck(cudaDeviceSynchronize());
      cudaCheck(cudaGetLastError()); // 检查异步 kernel 执行错误。
      cudaCheck(cudaMemcpy(C, dC, matrixBytes, cudaMemcpyDeviceToHost));
      cudaCheck(
          cudaMemcpy(C_ref, dC_ref, matrixBytes, cudaMemcpyDeviceToHost));

      if (isCustomKernel && !verify_matrix(C_ref, C, m * n)) {
        std::cout
            << "Failed to pass correctness verification against strict FP32 "
               "cuBLAS."
            << std::endl;
        if (m <= 128) {
          std::cout << " Logging faulty output into " << errLogFile << "\n";
          std::ofstream fs;
          fs.open(errLogFile);
          fs << "A:\n";
          print_matrix(A, m, n, fs);
          fs << "B:\n";
          print_matrix(B, m, n, fs);
          fs << "C:\n";
          print_matrix(C, m, n, fs);
          fs << "Should:\n";
          print_matrix(C_ref, m, n, fs);
        }
        exit(EXIT_FAILURE);
      }
      if (isCustomKernel) {
        std::cout << "Correctness vs strict FP32 cuBLAS: PASS" << std::endl;
      } else {
        double maxAbsDiff = 0.0;
        size_t overThreshold = 0;
        for (long i = 0; i < m * n; ++i) {
          const double diff = std::fabs(static_cast<double>(C_ref[i]) - C[i]);
          if (!std::isfinite(diff)) {
            maxAbsDiff = INFINITY;
            ++overThreshold;
          } else {
            maxAbsDiff = std::max(maxAbsDiff, diff);
            overThreshold += diff > 0.01;
          }
        }
        std::cout << "Difference vs strict FP32: max_abs=" << maxAbsDiff
                  << ", elements_over_0.01=" << overThreshold << "/" << m * n
                  << std::endl;
      }
    }

    for (int j = 0; j < warmup_times; j++) {
      if (resetCEachRepeat) {
        cudaCheck(cudaMemcpy(dC, dC_initial, matrixBytes,
                             cudaMemcpyDeviceToDevice));
      }
      run_kernel(kernel_num, m, n, k, alpha, dA, dB, beta, dC, handle);
    }
    if (warmup_times > 0) {
      cudaCheck(cudaDeviceSynchronize());
      cudaCheck(cudaGetLastError());
    }

    if (resetCEachRepeat) {
      float totalMilliseconds = 0.0f;
      for (int j = 0; j < repeat_times; j++) {
        // D2D 复位发生在 beg event 之前，不计入 GEMM 时间。
        cudaCheck(cudaMemcpy(dC, dC_initial, matrixBytes,
                             cudaMemcpyDeviceToDevice));
        cudaCheck(cudaEventRecord(beg));
        run_kernel(kernel_num, m, n, k, alpha, dA, dB, beta, dC, handle);
        cudaCheck(cudaEventRecord(end));
        cudaCheck(cudaEventSynchronize(end));
        float iterationMilliseconds = 0.0f;
        cudaCheck(cudaEventElapsedTime(&iterationMilliseconds, beg, end));
        totalMilliseconds += iterationMilliseconds;
      }
      elapsed_time = totalMilliseconds / 1000.0f;
    } else {
      cudaCheck(cudaEventRecord(beg));
      for (int j = 0; j < repeat_times; j++) {
        // 历史入口保留连续 beta*C 更新，便于复核旧数据；公平实验应启用复位。
        run_kernel(kernel_num, m, n, k, alpha, dA, dB, beta, dC, handle);
      }
      cudaCheck(cudaEventRecord(end));
      cudaCheck(cudaEventSynchronize(end));
      cudaCheck(cudaEventElapsedTime(&elapsed_time, beg, end));
      elapsed_time /= 1000.0f;
    }

    // 按 GEMM 主体的乘+加计数：约 2*M*N*K FLOPs；不是字节数。
    long flops = 2 * m * n * k;
    printf(
        "Average elapsed time: (%7.6f) s, performance: (%7.1f) GFLOPS. size: "
        "(%ld).\n",
        elapsed_time / repeat_times,
        (repeat_times * flops * 1e-9) / elapsed_time, m);
    fflush(stdout);
    if (resetCEachRepeat) {
      // 公平入口的下一尺寸也从原始 C 开始。
      cudaCheck(cudaMemcpy(dC, dC_initial, matrixBytes,
                           cudaMemcpyDeviceToDevice));
      cudaCheck(cudaMemcpy(dC_ref, dC_initial, matrixBytes,
                           cudaMemcpyDeviceToDevice));
    } else {
      // 历史入口保留原行为：把参考输出复制给自写路径，供下一尺寸继续使用。
      cudaCheck(
          cudaMemcpy(dC, dC_ref, matrixBytes, cudaMemcpyDeviceToDevice));
    }
  }

  // 释放 CPU/GPU 缓冲区和 cuBLAS 句柄。
  free(A);
  free(B);
  free(C);
  free(C_ref);
  cudaCheck(cudaFree(dA));
  cudaCheck(cudaFree(dB));
  cudaCheck(cudaFree(dC));
  cudaCheck(cudaFree(dC_ref));
  cudaCheck(cudaFree(dC_initial));
  cudaCheck(cudaEventDestroy(beg));
  cudaCheck(cudaEventDestroy(end));
  if (cublasDestroy(handle) != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "Destroy cublas handle error." << std::endl;
    return EXIT_FAILURE;
  }

  return 0;
};
