// 最小原生 CUDA 自检：设备信息、分配/拷贝、kernel、同步、逐元素校验。
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#define CUDA_OK(expr) do { const auto e=(expr); if(e!=cudaSuccess) { \
  std::fprintf(stderr,"%s: %s\n",#expr,cudaGetErrorString(e)); return 1; } } while(0)
__global__ void add_one(int* data, int n) {
  const int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<n) data[i]+=1;
}
int main() {
  cudaDeviceProp prop{};
  CUDA_OK(cudaGetDeviceProperties(&prop,0));
  std::printf("GPU=%s compute=%d.%d VRAM_MiB=%zu\n",prop.name,
              prop.major,prop.minor,prop.totalGlobalMem/(1024*1024));
  constexpr int n=257;
  int host[n];
  for(int i=0;i<n;++i) host[i]=i;
  int* device=nullptr;
  CUDA_OK(cudaMalloc(&device,sizeof(host)));
  CUDA_OK(cudaMemcpy(device,host,sizeof(host),cudaMemcpyHostToDevice));
  add_one<<<(n+127)/128,128>>>(device,n);
  CUDA_OK(cudaGetLastError());
  CUDA_OK(cudaDeviceSynchronize());
  CUDA_OK(cudaMemcpy(host,device,sizeof(host),cudaMemcpyDeviceToHost));
  CUDA_OK(cudaFree(device));
  for(int i=0;i<n;++i) if(host[i]!=i+1) return 2;
  std::puts("CUDA_SMOKE PASS (257 integer outputs)");
}
