// 【文件导读】K32：实验：一个 warp 搬运、四个 warp 计算
// 给初学者：这是回答某个性能问题的实验，不是编号越大越快；先了解对照版本与唯一改动，再看复杂细节。
// 先知道：kernel 是在 GPU 上执行的函数；K32 是本工程实验编号，不代表性能排名。
// 这一版在做什么：总共 160 线程，前 32 个是 producer，后 128 个是 consumer；命名屏障让不同分支约定同一次交接。
// 谁调用它：scripts/k32_audit.cu 独立测试程序；主程序 ./build/sgemm 不支持编号 32。
// 输入/输出：行主序 FP32 矩阵 A[M,K]、B[K,N]、C[M,N]，计算 C=alpha*AB+beta*C。
// serial/overlap/single 比较交接协议；所有参与线程都须按顺序到达屏障，不能因自己不算某块就提前退出。
// 边界：这是特定分块/对齐条件下的实验核，不是任意尺寸 GEMM；先读调用方约束再改 M/N/K。
// 阅读顺序：看线程如何分工 → 数据怎么搬 → 何时同步 → 怎么乘加 → 如何写回 C。
// 评价方法：正确性 PASS 只说明该次测试满足误差条件；是否更快还要看同口径多轮计时和 profiler。
// 中文总索引：docs/zh/23-代码与脚本使用索引.md；脚本命令默认在仓库根目录执行。

#pragma once
#include "30_kernel_mini_simt.cuh"

// K32：一个生产者 warp 搬运数据，四个消费者 warp 沿用 K30 的计算分工。
// 用 serial、overlap、single 三种协议作对照；计算使用 SIMT 浮点乘加，不使用 Hopper 专用机制或 Tensor Core。
namespace k32 {
using S = k30::Shape<128,128,16,64,64,4,4,128>;
constexpr int Threads=160;
constexpr int ASize=k30::SharedLayoutA<S>::Size;
constexpr int StageSize=ASize+k30::SharedLayoutB<S>::Size;
constexpr int SharedBytes=2*StageSize*sizeof(float);

// rendezvous 可理解成“交接集合点”：Id 区分屏障，160 是必须到齐的线程数。
// producer 与 consumer 在不同分支但会到达同一个屏障；memory 告诉编译器这里有内存次序约束。
// 普通双屏障版本每轮先 1 后 2；single 只到 1。漏一方或提前 return 都可能死锁。
template<int Id> __device__ __forceinline__ void rendezvous() {
  // PTX 指令不带 .aligned：生产者与消费者从不同分支到达同一个命名屏障，每个 warp 内分支一致。
  // 每次屏障都要凑齐 160 个线程；serial/overlap 每轮先到 1 再到 2，single 每轮只到 1，不能提前退出。
  asm volatile("barrier.sync %0, 160;" :: "n"(Id) : "memory");
}

// 搬运量与 K30 相同，但从 128 人分摊变成 32 人分摊，所以每人要多搬几轮。
// A：lane/4 给出行组内起点，t*8 遍历 16 组行；B：32 lane 各搬 4 列，正好覆盖 128 列。
// 只让一组人搬运不一定快：其他 warp 虽可算数，但总资源、同步和等待关系也变了。
__device__ __forceinline__ void copyTile(const float *a,const float *b,
    int N,int K,int lane,float *as,float *bs) {
  // 32 个 lane 搬完 K30 原本分给 128 个 lane 的所有 float4 坐标。
  // 每次读入的向量随即写入共享内存，再复用寄存器；不会先用寄存器暂存整个 tile。
#pragma unroll
  for(int t=0;t<16;++t) {
    int m=lane/4+t*8, k=(lane%4)*4;
    float4 v=*reinterpret_cast<const float4*>(a+m*K+k);
    as[k30::SharedLayoutA<S>::offset(m,k+0)]=v.x;
    as[k30::SharedLayoutA<S>::offset(m,k+1)]=v.y;
    as[k30::SharedLayoutA<S>::offset(m,k+2)]=v.z;
    as[k30::SharedLayoutA<S>::offset(m,k+3)]=v.w;
  }
#pragma unroll
  for(int k=0;k<16;++k)
    *reinterpret_cast<float4*>(bs+k*128+lane*4)=
        *reinterpret_cast<const float4*>(b+k*N+lane*4);
}
} // namespace k32

// Overlap=false：搬运下一块 -> 屏障 1 -> 计算当前块 -> 屏障 2。
// Overlap=true 且 SingleBarrier=false：搬运下一块与计算当前块可重叠 -> 屏障 1 -> 屏障 2。
// Overlap=true 且 SingleBarrier=true：同样允许搬运与计算重叠，但每轮仅保留屏障 1。
// serial 是先搬 next、集合、再算 current、集合；overlap 是搬 next 与算 current 各自进行后集合。
// overlap 的第二道屏障刻意保留作等次数对照；single 再单独研究删掉这道重复屏障的影响。
// 这是一组归因实验，不是 Hopper 的专用硬件流水，也不把 producer 的寄存器免费返还给 consumer。
template<bool Overlap, bool SingleBarrier=false>
__global__ __launch_bounds__(k32::Threads)
void sgemmK32(int M,int N,int K,float alpha,const float *__restrict__ A,
    const float *__restrict__ B,float beta,float *__restrict__ C) {
  static_assert(!SingleBarrier || Overlap);
  using S=k32::S;
  extern __shared__ __align__(16) float storage[];
  int tid=threadIdx.x, blockM=blockIdx.y*128, blockN=blockIdx.x*128;
  bool producer=tid<32;
  const float *a=A+blockM*K, *b=B+blockN;
  if(producer) k32::copyTile(a,b,N,K,tid,storage,storage+k32::ASize);
  __syncthreads(); // 初始 tile 搬完后让整个线程块同步，确保所有消费者都能读取它。
  if(producer) {
#pragma unroll 1
    for(int tile=0;tile<K/16;++tile) {
      int future=tile+1;
      if(future<K/16) {
        float *dst=storage+(future%2)*k32::StageSize;
        k32::copyTile(a+future*16,b+future*16*N,N,K,tid,dst,dst+k32::ASize);
      }
      k32::rendezvous<1>();
      if constexpr(!SingleBarrier) k32::rendezvous<2>();
    }
  } else {
    // 消费者真实 tid=32..159，减 32 后恢复 K30 的计算编号 0..127。
    // 因此每个消费者算哪些 C 不变；前面的生产者 warp 不持有最终输出累加器。
    k30::FragmentIterator<S> map(tid-32);
    float acc[S::Rows*S::Cols]={};
#pragma unroll 1
    for(int tile=0;tile<K/16;++tile) {
      if constexpr(!Overlap) k32::rendezvous<1>();
      float *current=storage+(tile%2)*k32::StageSize;
      k30::compute<S>(current,current+k32::ASize,map,acc);
      if constexpr(Overlap) k32::rendezvous<1>();
      if constexpr(!SingleBarrier) k32::rendezvous<2>();
    }
    k30::epilogue<S>(M,N,blockM,blockN,alpha,beta,C,map,acc);
  }
}
