// 【文件导读】K31：K30 的受控精修对照，不是默认更快版本
// 给初学者：这是回答某个性能问题的实验，不是编号越大越快；先了解对照版本与唯一改动，再看复杂细节。
// 先知道：kernel 是在 GPU 上执行的函数；K31 是本工程实验编号，不代表性能排名。
// 这一版在做什么：分别研究搬运时机、逐向量搬运、stage 数、CTA 大小和寄存器上限；一次比较一个因素，失败方案仍保留作证据。
// 谁调用它：scripts/k31_audit.cu 独立测试程序；主程序 ./build/sgemm 不支持编号 31。
// 输入/输出：行主序 FP32 矩阵 A[M,K]、B[K,N]、C[M,N]，计算 C=alpha*AB+beta*C。
// 边界：这是特定分块/对齐条件下的实验核，不是任意尺寸 GEMM；先读调用方约束再改 M/N/K。
// 阅读顺序：看线程如何分工 → 数据怎么搬 → 何时同步 → 怎么乘加 → 如何写回 C。
// 评价方法：正确性 PASS 只说明该次测试满足误差条件；是否更快还要看同口径多轮计时和 profiler。
// 中文总索引：docs/zh/23-代码与脚本使用索引.md；脚本命令默认在仓库根目录执行。

#pragma once
#include "30_kernel_mini_simt.cuh"

// K31 复用 K30 的形状、线程分工、共享内存布局、运算和结果写回过程。
// 每个实验只改变临时变量存活期、流水阶段数、线程块大小或编译器
// 寄存器预算标注中的一项；保留 K30 作为不变的对照组。
namespace k31 {
using Small = k30::Shape<128,128,16,64,64,4,4,128>;
using Large = k30::Shape<256,128,16,64,64,4,4,256>;

// 实验菜单：0=原样 control，1=晚加载，2=逐 float4 加载后立刻存，3/4=三/四槽，5=256线程。
// 名字只表明改变了什么，不表示 5 比 0 更快；应先确认 control 与 K30 一致再比较。
template<int Variant> struct Config {
  static_assert(Variant>=0 && Variant<=5);
  using S = Small;
  static constexpr int Stages = Variant==3 ? 3 : Variant==4 ? 4 : 2;
  static constexpr int Policy = Variant==1 ? 1 : Variant==2 ? 2 : 0;
};
template<> struct Config<5> {
  using S = Large;
  static constexpr int Stages=2, Policy=0;
};

// 策略 2：每个 float4 读入后立即写入共享内存，再读下一个向量；
// 全局/共享内存的物理坐标和搬运次数均与 GlobalIterator 相同。
// ptxas 仍可能重新调度这些操作，源码中的变量作用域不保证最终指令顺序。
template<class S>
// 一个包裹取到就放下，尝试缩短临时值从产生到最后使用的时间。
// 坐标和搬运总量不变；编译器仍可能重排，所以 C++ 局部作用域不等于物理寄存器立即释放。
__device__ __forceinline__ void copyVectors(const k30::GlobalIterator<S> &g,
                                           float *as,float *bs) {
#pragma unroll
  for(int t=0;t<S::ALoads;++t) {
    int r=g.ar+t*(S::Threads*4/S::K);
    float4 value=r<g.mRemaining ? *reinterpret_cast<const float4*>(g.a+r*g.strideA)
                               : make_float4(0,0,0,0);
    as[k30::SharedLayoutA<S>::offset(r,g.ak+0)]=value.x;
    as[k30::SharedLayoutA<S>::offset(r,g.ak+1)]=value.y;
    as[k30::SharedLayoutA<S>::offset(r,g.ak+2)]=value.z;
    as[k30::SharedLayoutA<S>::offset(r,g.ak+3)]=value.w;
  }
#pragma unroll
  for(int t=0;t<S::BLoads;++t) {
    float4 value=*reinterpret_cast<const float4*>(g.b+t*(S::Threads*4/S::N)*g.strideB);
    *reinterpret_cast<float4*>(bs+k30::SharedLayoutB<S>::offset(
        g.bk+t*(S::Threads*4/S::N),g.bn))=value;
  }
}

// 复用 K30 的 ownership、shared 布局、乘加与写回，仅把搬运策略/槽数作为编译时选项。
// Policy=0 在 compute 前取 next；Policy=1 在 compute 后取；Policy=2 逐向量取了就存。
// 所有策略共用同一轮尾屏障协议，避免把同步变化误认为仅寄存器生命周期变化。
template<class S,int Stages,int Policy>
__device__ __forceinline__ void run(int M,int N,int K,float alpha,
    const float *__restrict__ A,const float *__restrict__ B,float beta,
    float *__restrict__ C,float *storage) {
  static_assert(Stages>=2 && Stages<=4 && Policy>=0 && Policy<=2);
  constexpr int ASize=k30::SharedLayoutA<S>::Size;
  constexpr int StageSize=ASize+k30::SharedLayoutB<S>::Size;
  int blockM=blockIdx.y*S::M, blockN=blockIdx.x*S::N;
  k30::GlobalIterator<S> global(A,B,M,N,K,blockM,blockN);
  k30::FragmentIterator<S> map(threadIdx.x);
  float acc[S::Rows*S::Cols]={};
#pragma unroll
  for(int stage=0;stage<Stages-1;++stage) {
    if(stage<K/S::K) {
      typename k30::GlobalIterator<S>::Fragment fragment;
      global.load(fragment); global.advance();
      float *dst=storage+stage*StageSize;
      global.store(fragment,dst,dst+ASize);
    }
  }
  __syncthreads();
  for(int tile=0;tile<K/S::K;++tile) {
    int future=tile+Stages-1;
    if constexpr(Policy==0) {
      typename k30::GlobalIterator<S>::Fragment next;
      if(future<K/S::K) { global.load(next); global.advance(); }
      float *current=storage+(tile%Stages)*StageSize;
      k30::compute<S>(current,current+ASize,map,acc);
      if(future<K/S::K) {
        float *dst=storage+(future%Stages)*StageSize;
        global.store(next,dst,dst+ASize);
      }
    } else {
      float *current=storage+(tile%Stages)*StageSize;
      k30::compute<S>(current,current+ASize,map,acc);
      if(future<K/S::K) {
        float *dst=storage+(future%Stages)*StageSize;
        if constexpr(Policy==1) {
          // 延后读取的对照版本：仅缩短 next 在源码层面的存活区间。
          typename k30::GlobalIterator<S>::Fragment next;
          global.load(next); global.advance();
          global.store(next,dst,dst+ASize);
        } else {
          copyVectors<S>(global,dst,dst+ASize);
          global.advance();
        }
      }
    }
    // 所有变体（包括三阶段版本）使用相同的旧阶段结束/新阶段发布协议。
    __syncthreads();
  }
  k30::epilogue<S>(M,N,blockM,blockN,alpha,beta,C,map,acc);
}
} // namespace k31

template<int Variant>
__global__ __launch_bounds__(k31::Config<Variant>::S::Threads,
                            (k31::Config<Variant>::S::Threads==128 ? 2 : 1))
void sgemmK31(int M,int N,int K,float alpha,const float *__restrict__ A,
    const float *__restrict__ B,float beta,float *__restrict__ C) {
  extern __shared__ __align__(16) float storage[];
  k31::run<typename k31::Config<Variant>::S,k31::Config<Variant>::Stages,
           k31::Config<Variant>::Policy>(M,N,K,alpha,A,B,beta,C,storage);
}

// __maxnreg__ 与 __launch_bounds__ 不能同时使用；这里分别研究两种
// 编译器标注，并不保证 ptxas 恰好分配 Limit 个寄存器。
// 这五种预算按寄存器资源估算，都至多允许同时驻留两个 128 线程的线程块。
// Limit 是编译器寄存器上限，不是实际使用量；限制过紧可能把数据溢出到 local memory。
// 这一路用 __maxnreg__，不能同时加 __launch_bounds__；比较时要保留这个编译约束差别。
template<int Limit>
__global__ __maxnreg__(Limit)
void sgemmK31Budget(int M,int N,int K,float alpha,const float *__restrict__ A,
    const float *__restrict__ B,float beta,float *__restrict__ C) {
  static_assert(Limit==220 || Limit==225 || Limit==230 || Limit==235 || Limit==240);
  extern __shared__ __align__(16) float storage[];
  k31::run<k31::Small,2,0>(M,N,K,alpha,A,B,beta,C,storage);
}
