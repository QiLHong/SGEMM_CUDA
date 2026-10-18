// 【文件导读】主程序与 runner 实现之间的声明接口
// 给初学者：host 是 CPU 端控制代码，device 是 GPU 端计算代码；分配、拷贝、启动、等待、校验各有职责。
// 供 sgemm.cu 引用；函数实现位于同目录 runner.cu，设备实现由 kernels.cuh 汇总。
// cudaCheck 接口接收 CUDA 错误码、源码文件名和行号；调用处的宏负责补充位置，添加注释会改变诊断行号。
// 矩阵函数处理 CPU 侧 float 缓冲；run_kernel 接收 GPU 指针和 cuBLAS handle，不能混用主机/设备地址。
// 各包装函数的编号、tile 和尺寸约束应结合 runner.cu 阅读，不要只凭声明推断支持任意尺寸。
// 中文总索引：docs/zh/23-代码与脚本使用索引.md；脚本命令默认在仓库根目录执行。

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
