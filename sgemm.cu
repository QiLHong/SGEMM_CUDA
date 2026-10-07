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

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "Please select a kernel (range 0 - 12, 0 for NVIDIA cuBLAS)"
              << std::endl;
    exit(EXIT_FAILURE);
  }

  // 从命令行读取 kernel 编号；0 是 FP32 cuBLAS，1–12 是教学实现。
  int kernel_num = std::stoi(argv[1]);
  if (kernel_num < 0 || kernel_num > 12) {
    std::cerr << "Please enter a valid kernel number (0-12)" << std::endl;
    exit(EXIT_FAILURE);
  }

  // DEVICE 是此程序自己读取的环境变量；未设置时选设备 0。
  int deviceIdx = 0;
  if (getenv("DEVICE") != NULL) {
    deviceIdx = atoi(getenv("DEVICE"));
  }
  cudaCheck(cudaSetDevice(deviceIdx));

  printf("Running kernel %d on device %d.\n", kernel_num, deviceIdx);

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
  cudaEventCreate(&beg);
  cudaEventCreate(&end);

  // 原作者备注：cuBLAS 在 8192 尺寸达到 FLOPs 上限；当前实际只测到 4096。
  // 所有尺寸都保持上游值；特别注意较晚 kernel 的分块约束，不应盲加任意尺寸。
  std::vector<int> SIZE = {128, 256, 512, 1024, 2048, 4096};

  long m, n, k, max_size;
  max_size = SIZE[SIZE.size() - 1];
  std::cout << "Max size: " << max_size << std::endl;

  float alpha = 0.5, beta = 3.0; // GEMM 参数：C=α*AB+β*C，不只是 C=AB。

  float *A = nullptr, *B = nullptr, *C = nullptr,
        *C_ref = nullptr; // 主机（CPU）上的四份矩阵缓冲区。
  float *dA = nullptr, *dB = nullptr, *dC = nullptr,
        *dC_ref = nullptr; // 设备（GPU）上的四份矩阵缓冲区。

  // 按最大尺寸一次分配，各个测试尺寸复用这些缓冲区。
  A = (float *)malloc(sizeof(float) * max_size * max_size);
  B = (float *)malloc(sizeof(float) * max_size * max_size);
  C = (float *)malloc(sizeof(float) * max_size * max_size);
  C_ref = (float *)malloc(sizeof(float) * max_size * max_size);

  randomize_matrix(A, max_size * max_size);
  randomize_matrix(B, max_size * max_size);
  randomize_matrix(C, max_size * max_size);

  cudaCheck(cudaMalloc((void **)&dA, sizeof(float) * max_size * max_size));
  cudaCheck(cudaMalloc((void **)&dB, sizeof(float) * max_size * max_size));
  cudaCheck(cudaMalloc((void **)&dC, sizeof(float) * max_size * max_size));
  cudaCheck(cudaMalloc((void **)&dC_ref, sizeof(float) * max_size * max_size));

  // 两条计算路径必须从相同的旧 C 开始，因为 beta 非零。
  cudaCheck(cudaMemcpy(dA, A, sizeof(float) * max_size * max_size,
                       cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(dB, B, sizeof(float) * max_size * max_size,
                       cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(dC, C, sizeof(float) * max_size * max_size,
                       cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(dC_ref, C, sizeof(float) * max_size * max_size,
                       cudaMemcpyHostToDevice));

  int repeat_times = 50;
  for (int size : SIZE) {
    m = n = k = size;

    std::cout << "dimensions(m=n=k) " << m << ", alpha: " << alpha
              << ", beta: " << beta << std::endl;
    // 非 cuBLAS 路径：先各执行一次以校验，也让自写 kernel 在计时前预热。
    // 【学习补充】kernel 0 不进入此分支，原程序并未给所有路径做相同预热。
    if (kernel_num != 0) {
      run_kernel(0, m, n, k, alpha, dA, dB, beta, dC_ref,
                 handle); // cuBLAS 参考输出。
      run_kernel(kernel_num, m, n, k, alpha, dA, dB, beta, dC,
                 handle); // 自写 kernel 修改 dC。
      cudaCheck(cudaDeviceSynchronize());
      cudaCheck(cudaGetLastError()); // 检查异步 kernel 执行错误。
      cudaMemcpy(C, dC, sizeof(float) * m * n, cudaMemcpyDeviceToHost);
      cudaMemcpy(C_ref, dC_ref, sizeof(float) * m * n, cudaMemcpyDeviceToHost);

      // verify_matrix 使用上游固定的绝对误差门槛，不是逐位相等。
      if (!verify_matrix(C_ref, C, m * n)) {
        std::cout
            << "Failed to pass the correctness verification against NVIDIA "
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
    }

    cudaEventRecord(beg);
    for (int j = 0; j < repeat_times; j++) {
      // 为节省时间，重复计时之间不重置 dC；因此会反复执行 beta*C 更新。
      // 上面的正确性检查针对计时前那一次执行，并非逐次校验这 50 次输出。
      run_kernel(kernel_num, m, n, k, alpha, dA, dB, beta, dC, handle);
    }
    cudaEventRecord(end);
    cudaEventSynchronize(beg);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&elapsed_time, beg, end);
    elapsed_time /= 1000.; // CUDA Event 返回毫秒，这里转换为秒。

    // 按 GEMM 主体的乘+加计数：约 2*M*N*K FLOPs；不是字节数。
    long flops = 2 * m * n * k;
    printf(
        "Average elapsed time: (%7.6f) s, performance: (%7.1f) GFLOPS. size: "
        "(%ld).\n",
        elapsed_time / repeat_times,
        (repeat_times * flops * 1e-9) / elapsed_time, m);
    fflush(stdout);
    // 计时已改变 dC，恢复成 dC_ref，使下一尺寸的两条路径从相同状态出发。
    cudaCheck(cudaMemcpy(dC, dC_ref, sizeof(float) * m * n,
                         cudaMemcpyDeviceToDevice));
  }

  // 释放 CPU/GPU 缓冲区和 cuBLAS 句柄。
  free(A);
  free(B);
  free(C);
  free(C_ref);
  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);
  cudaFree(dC_ref);
  cublasDestroy(handle);

  return 0;
};
