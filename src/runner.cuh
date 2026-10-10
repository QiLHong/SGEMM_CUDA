#pragma once
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

// 主机端辅助接口：设备错误、矩阵初始化/校验以及 kernel 编号分发。
void cudaCheck(cudaError_t error, const char *file,
               int line); // 检查 CUDA 错误，并报告源码文件/行号。
void CudaDeviceInfo();    // 打印 CUDA 设备信息。

void range_init_matrix(float *mat, int N);
void randomize_matrix(float *mat, int N);
void zero_init_matrix(float *mat, int N);
void copy_matrix(const float *src, float *dest, int N);
void print_matrix(const float *A, int M, int N, std::ofstream &fs);
bool verify_matrix(float *mat1, float *mat2, int N);

float get_current_sec();                        // 获取当前时刻。
float cpu_elapsed_time(float &beg, float &end); // 计算 CPU 时间差。

// 入口编号见 runner.cu 的 switch；A[M,K]、B[K,N]、C[M,N] 均按行主序传入。
void run_kernel(int kernel_num, int m, int n, int k, float alpha, float *A,
                float *B, float beta, float *C, cublasHandle_t handle);
