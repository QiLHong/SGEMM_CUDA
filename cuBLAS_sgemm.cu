// 【文件导读】小矩阵 cuBLAS 行主序适配演示
// 给初学者：这是先理解基本概念的小例子；不要把示例输出当成正式 benchmark 结论。
// 用途是打印一个 2×3 乘 3×4 的示例，帮助理解 C=alpha*AB+beta*C，不是性能基准。
// 行主序数据传给列主序 cuBLAS 时交换 A/B 与 M/N，利用 (AB)^T=B^T A^T，未显式转置矩阵。
// alpha=1、beta=0.5，旧 C 参与结果；设备内存分配、拷贝、调用和释放均在 main 内。
// 注意已有注释指出的 A 打印索引问题：打印结果不能替代严格验证；本轮只补注释，不修算法或输出。
// 目标名见 CMakeLists.txt；不要把示例的默认 cuBLAS 配置当作 audit 中的 strict FP32 模式。
// 中文总索引：docs/zh/23-代码与脚本使用索引.md；脚本命令默认在仓库根目录执行。

#include <cstdio>
#include <cublas_v2.h>
#include <cuda_runtime.h>

/*
 * 原作者说明：独立调用并测试标准 cuBLAS SGEMM。
 * 【学习补充】当前文件是 2×3 乘 3×4 的打印演示，没有计时循环。
 * 正式比较吞吐率请使用 sgemm 0，而不要把这个演示当作 GFLOPS benchmark。
 */

int main(int argc, char *argv[]) {
  int m = 2;
  int k = 3;
  int n = 4;
  int print = 1;
  cudaError_t cudaStat;  // cudaMalloc 状态变量（保留原实现）。
  cublasStatus_t stat;   // cuBLAS 函数返回状态。
  cublasHandle_t handle; // cuBLAS 上下文句柄。

  int i, j;

  float *a, *b, *c;

  // 为 CPU 侧 A/B/C 分配内存。
  a = (float *)malloc(m * k * sizeof(float));
  b = (float *)malloc(k * n * sizeof(float));
  c = (float *)malloc(m * n * sizeof(float));

  int ind = 11;
  for (j = 0; j < m * k; j++) {
    a[j] = (float)ind++;
  }

  ind = 11;
  for (j = 0; j < k * n; j++) {
    b[j] = (float)ind++;
  }

  ind = 11;
  for (j = 0; j < m * n; j++) {
    c[j] = (float)ind++;
  }

  // GPU 侧指针。
  float *d_a, *d_b, *d_c;

  // 为设备端 A/B/C 分配空间。
  cudaMalloc((void **)&d_a, m * k * sizeof(float));
  cudaMalloc((void **)&d_b, k * n * sizeof(float));
  cudaMalloc((void **)&d_c, m * n * sizeof(float));

  stat = cublasCreate(&handle); // 初始化 cuBLAS 上下文。

  cudaMemcpy(d_a, a, m * k * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, k * n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_c, c, m * n * sizeof(float), cudaMemcpyHostToDevice);

  float alpha = 1.0f;
  float beta = 0.5f;

  if (print == 1) {
    printf("alpha = %4.0f, beta = %4.0f\n", alpha, beta);
    printf("A = (mxk: %d x %d)\n", m, k);
    for (i = 0; i < m; i++) {
      for (j = 0; j < k; j++) {
        // 【学习补充】原打印索引是 i*m+j；对行主序 [m,k] 应检查 i*k+j。
        // 本分支只注释，不修此上游演示问题；不要从这段打印反推正确布局。
        printf("%4.1f ", a[i * m + j]);
      }
      printf("\n");
    }
    printf("B = (kxn: %d x %d)\n", k, n);
    for (i = 0; i < k; i++) {
      for (j = 0; j < n; j++) {
        printf("%4.1f ", b[i * n + j]);
      }
      printf("\n");
    }
    printf("C = (mxn: %d x %d)\n", m, n);
    for (i = 0; i < m; i++) {
      for (j = 0; j < n; j++) {
        printf("%4.1f ", c[i * n + j]);
      }
      printf("\n");
    }
  }

  // cuBLAS 按列主序解释数据：交换 A/B 和 M/N，利用 (AB)^T=B^T*A^T。
  // 这里不实际分配转置矩阵，底层仍然使用相同的内存。
  stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, d_b, n,
                     d_a, k, &beta, d_c, n);

  cudaMemcpy(c, d_c, m * n * sizeof(float), cudaMemcpyDeviceToHost);

  if (print == 1) {
    printf("\nC after SGEMM = \n");
    for (i = 0; i < m; i++) {
      for (j = 0; j < n; j++) {
        printf("%4.1f ", c[i * n + j]);
      }
      printf("\n");
    }
  }

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  cublasDestroy(handle); // 销毁 cuBLAS 上下文。
  free(a);
  free(b);
  free(c);

  return EXIT_SUCCESS;
}
