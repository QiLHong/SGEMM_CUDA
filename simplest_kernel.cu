// 【文件导读】从一维线程号得到二维坐标的最小 CUDA 示例
// 给初学者：这是先理解基本概念的小例子；不要把示例输出当成正式 benchmark 结论。
// 一个 block 启动 16 个线程，用 threadIdx.x/4 与 threadIdx.x%4 映射到 4×4 坐标。
// 这里的除数 4 是示例写死的行宽，不是 warp 的大小；硬件 warp 仍是 32 个线程。
// 设备端 printf 的输出顺序不代表线程严格按编号顺序执行；同步后才能可靠看到全部输出。
// 本例不做矩阵乘法也不测吞吐，适合先学 kernel 发射、线程编号与主机/设备边界。
// 中文总索引：docs/zh/23-代码与脚本使用索引.md；脚本命令默认在仓库根目录执行。

#include <cuda_runtime.h>
#include <iostream>
#include <vector>

using uint = unsigned int;

// 【学习补充】阅读 GEMM 前的索引练习：16 个一维线程铺成 4×4 输出。
// x=tid/4 是行号，y=tid%4 是列号；row 参数在地址公式中实际充当行跨度。
// 这个例子硬编码了 4，不能只改 SIZE 就当成任意尺寸版本。
__global__ void kernel(uint *A, uint *B, int row) {
  auto x = threadIdx.x / 4;
  auto y = threadIdx.x % 4;
  A[x * row + y] = x;
  B[x * row + y] = y;
}

int main(int argc, char **argv) {
  uint *Xs, *Ys;
  uint *Xs_d, *Ys_d;

  uint SIZE = 4;

  // 主机内存与设备内存分别分配，指针不能混用。
  Xs = (uint *)malloc(SIZE * SIZE * sizeof(uint));
  Ys = (uint *)malloc(SIZE * SIZE * sizeof(uint));

  cudaMalloc((void **)&Xs_d, SIZE * SIZE * sizeof(uint));
  cudaMalloc((void **)&Ys_d, SIZE * SIZE * sizeof(uint));

  // 一个 block、16 个线程；不是 4×4 的二维 threadIdx。
  dim3 grid_size(1, 1, 1);
  dim3 block_size(4 * 4);

  kernel<<<grid_size, block_size>>>(Xs_d, Ys_d, 4);

  // 结果拷回 CPU 后，再由 CPU 打印。
  cudaMemcpy(Xs, Xs_d, SIZE * SIZE * sizeof(uint), cudaMemcpyDeviceToHost);
  cudaMemcpy(Ys, Ys_d, SIZE * SIZE * sizeof(uint), cudaMemcpyDeviceToHost);

  cudaDeviceSynchronize();

  // 预期每个位置打印 [行号|列号]；先手算 tid=0、3、4、15 对应的位置。
  for (int row = 0; row < SIZE; ++row) {
    for (int col = 0; col < SIZE; ++col) {
      std::cout << "[" << Xs[row * SIZE + col] << "|" << Ys[row * SIZE + col]
                << "] ";
    }
    std::cout << "\n";
  }

  cudaFree(Xs_d);
  cudaFree(Ys_d);
  free(Xs);
  free(Ys);
}
