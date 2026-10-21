// TF32 K-tile experiment; derived from the preserved Tensor Core event harness.
// Compare original static BK16, dynamic BK16 control, dynamic BK32 and TF32 cuBLAS.
// All: dynamic pair adjacent, order swapped; confirm: original/BK32 adjacent.
// Precision and event protocol retained; additionally require baseline bitwise equality.
// C=AB；TF32输入提前显式转换，FP16输入/输出 + FP32累加；转换不在GEMM事件内。
// CLI: M N K rounds repeats warmup new.csv [all|implementation|tensor_pairs]
//      --profile implementation N （只在cudaProfilerStart/Stop区间采一个GEMM）
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <cublas_v2.h>
#include <mma.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <sstream>
#include <string>
#include <vector>
#include "../../src/kernels/30_kernel_mini_simt.cuh"
#include "../tensorcore/vendor/vitamin_tf32.cuh"
#include "../tensorcore/vendor/vitamin_fp16.cuh"

#include "tf32_ktile.cuh"

void ck(cudaError_t s) { if(s != cudaSuccess) throw std::runtime_error(cudaGetErrorString(s)); }
void bk(cublasStatus_t s) { if(s != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cuBLAS "+std::to_string(s)); }
struct Buf {
  void* p=nullptr;
  explicit Buf(size_t bytes) { ck(cudaMalloc(&p,bytes)); }
  ~Buf(){ if(p) cudaFree(p); }
  float* f(){return static_cast<float*>(p);}
  half* h(){return static_cast<half*>(p);}
};
struct Event {
  cudaEvent_t e{};
  Event(){ck(cudaEventCreate(&e));}
  ~Event(){cudaEventDestroy(e);}
};
struct Handle {
  cublasHandle_t h{};
  explicit Handle(bool strict) {
    bk(cublasCreate(&h));
    bk(cublasSetStream(h,nullptr));
    bk(cublasSetAtomicsMode(h,CUBLAS_ATOMICS_NOT_ALLOWED));
    bk(cublasSetMathMode(h,strict ? CUBLAS_PEDANTIC_MATH :
        CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION));
  }
  ~Handle(){cublasDestroy(h);}
};
constexpr uint32_t Seed=20261020;
uint32_t rng(uint32_t& x){x^=x<<13;x^=x>>17;x^=x<<5;return x;}
std::vector<float> input(size_t count, unsigned stream){
  std::vector<float> out(count);
  uint32_t state=Seed^(0x9e3779b9U*(stream+1U));
  for(auto& x:out){
    auto i=rng(state)%5U, f=rng(state)%5U; bool negative=(rng(state)&1U)!=0;
    x=(negative?-1.f:1.f)*(float(i)+.01f*float(f));
  }
  return out;
}
std::string hash(const void* data,size_t bytes){
  uint64_t h=14695981039346656037ULL;
  const auto* p=static_cast<const unsigned char*>(data);
  for(size_t i=0;i<bytes;++i){h^=p[i];h*=1099511628211ULL;}
  std::ostringstream o;o<<std::hex<<h;return o.str();
}
long long utcMs(){
  return std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::system_clock::now().time_since_epoch()).count();
}
// 合法TF32位模式由CUDA转换内建函数生成，不能只把任意FP32寄存器声明成.tf32。
// 两条精度路线均从同一份原始FP32数据派生。此预处理不纳入GEMM主表。
__global__ void prepareInputs(const float* x,float* tf,half* fp16,float* fp16AsFloat,size_t n){
  size_t i=blockIdx.x*size_t(blockDim.x)+threadIdx.x;
  if(i<n){tf[i]=nvcuda::wmma::__float_to_tf32(x[i]);half h=__float2half_rn(x[i]);fp16[i]=h;fp16AsFloat[i]=__half2float(h);}
}
enum Impl {K30,Strict,TFSingle,TFDouble,TFBlas,HSingle,HDouble,HBlas,TFK16,TFK32};
const std::array<const char*,10> Names={{
    "k30_fp32","cublas_strict_fp32","vitamin_tf32_single","vitamin_tf32_dbf","cublas_tf32",
    "vitamin_fp16_single","vitamin_fp16_dbf_rw","cublas_fp16",
    "tf32_dynamic_bk16","tf32_dynamic_bk32"}};
int findImpl(const std::string& s){
  for(int i=0;i<10;++i) if(s==Names[i])return i;
  throw std::runtime_error("Unknown implementation: "+s);
}
int precision(int impl){return impl<8?(impl<2?0:(impl<5?1:2)):1;}
using Shape=k30::Shape<128,128,16,64,64,4,4,128>;
void blas(cublasHandle_t h,int m,int n,int k,void* a,void* b,void* c,
          cudaDataType_t type,cublasComputeType_t compute){
  const float alpha=1,beta=0;
  // Row-major C=AB -> column-major C^T=B^T*A^T；m/n和指针一同交换。
  bk(cublasGemmEx(h,CUBLAS_OP_N,CUBLAS_OP_N,n,m,k,&alpha,b,type,n,a,type,k,
                  &beta,c,type,n,compute,CUBLAS_GEMM_DEFAULT));
}
struct Data {
  int m,n,k;
  size_t ac,bc,cc;
  Buf a,b,ta,tb,ha,hb,haf,hbf,c,hc,c0,hc0;
  Handle strict{true},tensor{false};
  std::vector<float> hostA,hostB;
  Data(int mm,int nn,int kk):m(mm),n(nn),k(kk),ac(size_t(m)*k),bc(size_t(k)*n),cc(size_t(m)*n),
      a(ac*4),b(bc*4),ta(ac*4),tb(bc*4),ha(ac*2),hb(bc*2),haf(ac*4),hbf(bc*4),
      c(cc*4),hc(cc*2),c0(cc*4),hc0(cc*2),hostA(input(ac,0)),hostB(input(bc,1)){
    ck(cudaMemcpy(a.p,hostA.data(),ac*4,cudaMemcpyHostToDevice));
    ck(cudaMemcpy(b.p,hostB.data(),bc*4,cudaMemcpyHostToDevice));
    ck(cudaFuncSetAttribute(ktile_tf32::sgemm_tf32_swizzle_bcf_dbf_kernel<128,128,16>,
                           cudaFuncAttributeMaxDynamicSharedMemorySize,32768));
    ck(cudaFuncSetAttribute(ktile_tf32::sgemm_tf32_swizzle_bcf_dbf_kernel<128,128,32>,
                           cudaFuncAttributeMaxDynamicSharedMemorySize,65536));
    ck(cudaMemset(c0.p,0,cc*4));ck(cudaMemset(hc0.p,0,cc*2));
  }
  void reset(int impl){
    if(precision(impl)==2)ck(cudaMemcpyAsync(hc.p,hc0.p,cc*2,cudaMemcpyDeviceToDevice));
    else ck(cudaMemcpyAsync(c.p,c0.p,cc*4,cudaMemcpyDeviceToDevice));
  }
  void launch(int impl){
    dim3 grid(n/128,m/128);
    switch(impl){
      case TFK16:ktile_tf32::sgemm_tf32_swizzle_bcf_dbf_kernel<128,128,16>
          <<<grid,256,32768>>>(ta.f(),tb.f(),c.f(),m,n,k);break;
      case TFK32:ktile_tf32::sgemm_tf32_swizzle_bcf_dbf_kernel<128,128,32>
          <<<grid,256,65536>>>(ta.f(),tb.f(),c.f(),m,n,k);break;
      case K30:sgemmK30<Shape,2><<<grid,128,2*(k30::SharedLayoutA<Shape>::Size+k30::SharedLayoutB<Shape>::Size)*4>>>
                   (m,n,k,1.f,a.f(),b.f(),0.f,c.f());break;
      case Strict:blas(strict.h,m,n,k,a.p,b.p,c.p,CUDA_R_32F,CUBLAS_COMPUTE_32F_PEDANTIC);break;
      case TFSingle:vitamin_tf32::sgemm_tf32_swizzle_bcf_kernel<128,128,16>
                   <<<grid,256>>>(ta.f(),tb.f(),c.f(),m,n,k);break;
      case TFDouble:vitamin_tf32::sgemm_tf32_swizzle_bcf_dbf_kernel<128,128,16>
                   <<<grid,256>>>(ta.f(),tb.f(),c.f(),m,n,k);break;
      case TFBlas:blas(tensor.h,m,n,k,ta.p,tb.p,c.p,CUDA_R_32F,CUBLAS_COMPUTE_32F_FAST_TF32);break;
      case HSingle:vitamin_fp16::hgemm_bcf_kernel<128,128,32>
                   <<<grid,256>>>(ha.h(),hb.h(),hc.h(),m,n,k);break;
      case HDouble:vitamin_fp16::hgemm_bcf_dbf_rw_kernel<128,128,32>
                   <<<grid,256>>>(ha.h(),hb.h(),hc.h(),m,n,k);break;
      case HBlas:blas(tensor.h,m,n,k,ha.p,hb.p,hc.p,CUDA_R_16F,CUBLAS_COMPUTE_32F);break;
    }
    ck(cudaGetLastError());
  }
  std::vector<float> output(int impl){
    std::vector<float> v(cc);
    if(precision(impl)==2){
      std::vector<half> h(cc);ck(cudaMemcpy(h.data(),hc.p,cc*2,cudaMemcpyDeviceToHost));
      for(size_t i=0;i<cc;++i)v[i]=__half2float(h[i]);
    }else ck(cudaMemcpy(v.data(),c.p,cc*4,cudaMemcpyDeviceToHost));
    return v;
  }
};
std::vector<float> copyFloat(Buf& b,size_t n){
  std::vector<float> v(n);ck(cudaMemcpy(v.data(),b.p,n*4,cudaMemcpyDeviceToHost));return v;
}
struct Error {
  double maxAbs=0,maxRel=0,sumSq=0,refSq=0;
  size_t nonfinite=0,abs001=0,simt=0,native=0,count=0;
  void add(double got,double ref,double eps){
    ++count;
    if(!std::isfinite(got)||!std::isfinite(ref)){++nonfinite;++abs001;++simt;++native;return;}
    double e=std::abs(got-ref);
    maxAbs=std::max(maxAbs,e);maxRel=std::max(maxRel,e/std::max(std::abs(ref),1e-12));
    sumSq+=e*e;refSq+=ref*ref;
    abs001+=e>.01;simt+=e>.001+.001*std::abs(got);
    native+=e>eps+eps*std::abs(got);
  }
};
void emit(std::ofstream& out,int m,int n,int k,const char* name,int round,
          const char* reference,const Error& e,double eps){
  out<<m<<','<<n<<','<<k<<','<<name<<','<<round<<','<<reference<<','<<e.count<<','
     <<e.maxAbs<<','<<e.maxRel<<','<<std::sqrt(e.sumSq/std::max(e.refSq,1e-30))<<','
     <<e.nonfinite<<','<<e.abs001<<','<<e.simt<<','<<eps<<','<<e.native<<'\n';
}
void shapeCheck(int m,int n,int k){
  // 原核无尾块保护；宽8的grid swizzle必须覆盖完整的8列CTA。
  if(m<128||m>4096||m%128||n<1024||n>4096||n%1024||k<32||k>4096||k%32)
    throw std::runtime_error("Unsupported shape: require M=128..4096 multiple128, N=1024..4096 multiple1024, K=32..4096 multiple32");
}
int main(int argc,char** argv)try{
  bool profile=argc==4&&std::string(argv[1])=="--profile";
  if(!profile&&(argc<8||argc>9))throw std::runtime_error("Usage: M N K rounds repeats warmup new.csv [all|implementation]");
  int m,n,k,rounds=1,repeats=1,warm=0,selected=-1;
  std::string path;
  if(profile){m=n=k=std::stoi(argv[3]);selected=findImpl(argv[2]);}
  else{
    m=std::stoi(argv[1]);n=std::stoi(argv[2]);k=std::stoi(argv[3]);
    rounds=std::stoi(argv[4]);repeats=std::stoi(argv[5]);warm=std::stoi(argv[6]);path=argv[7];
    if(argc==9&&std::string(argv[8])!="all")
      selected=std::string(argv[8])=="confirm"?-2:findImpl(argv[8]);
  }
  shapeCheck(m,n,k);
  if(rounds<1||rounds>100||repeats<1||repeats>10000||warm<0||warm>1000)throw std::runtime_error("Invalid iteration count");
  if(!profile&&(std::ifstream(path).good()||std::ifstream(path+".checks.csv").good()))
    throw std::runtime_error("Refusing to overwrite output");
  cudaDeviceProp prop{};ck(cudaGetDeviceProperties(&prop,0));
  if(prop.major!=8||prop.minor<0)throw std::runtime_error("This experiment requires tested Ampere/Ada sm80+ hardware");
  const char* overrideValue=std::getenv("NVIDIA_TF32_OVERRIDE");
  if(overrideValue&&std::string(overrideValue)=="0")throw std::runtime_error("NVIDIA_TF32_OVERRIDE=0 would disable cuBLAS TF32; unset in this process");
  std::cout<<"GPU="<<prop.name<<" sm="<<prop.major<<prop.minor<<'\n';
  Data d(m,n,k);int version=0;bk(cublasGetVersion(d.tensor.h,&version));
  std::cout<<"cuBLAS="<<version<<" seed="<<Seed<<" alpha=1 beta=0\n";
  int occOriginal=0,occ16=0,occ32=0;
  ck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occOriginal,
     vitamin_tf32::sgemm_tf32_swizzle_bcf_dbf_kernel<128,128,16>,256,0));
  ck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ16,
     ktile_tf32::sgemm_tf32_swizzle_bcf_dbf_kernel<128,128,16>,256,32768));
  ck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ32,
     ktile_tf32::sgemm_tf32_swizzle_bcf_dbf_kernel<128,128,32>,256,65536));
  std::cout<<"maxSharedOptin="<<prop.sharedMemPerBlockOptin
           <<" CTA_per_SM(original/dynamic16/dynamic32)="<<occOriginal<<'/'<<occ16<<'/'<<occ32<<'\n';
  Event start,stop;
  ck(cudaEventRecord(start.e));
  prepareInputs<<<(d.ac+255)/256,256>>>(d.a.f(),d.ta.f(),d.ha.h(),d.haf.f(),d.ac);
  prepareInputs<<<(d.bc+255)/256,256>>>(d.b.f(),d.tb.f(),d.hb.h(),d.hbf.f(),d.bc);
  ck(cudaGetLastError());ck(cudaEventRecord(stop.e));ck(cudaEventSynchronize(stop.e));
  float prepMs;ck(cudaEventElapsedTime(&prepMs,start.e,stop.e));
  if(profile){
    for(int i=0;i<5;++i){d.reset(selected);d.launch(selected);}
    ck(cudaDeviceSynchronize());d.reset(selected);ck(cudaDeviceSynchronize());
    ck(cudaProfilerStart());d.launch(selected);ck(cudaDeviceSynchronize());ck(cudaProfilerStop());
    std::cout<<"PROFILE_COMPLETE "<<Names[selected]<<'\n';return 0;
  }
  std::ofstream bits(path+".bitwise.csv");
  std::ofstream events(path),checks(path+".checks.csv"),meta(path+".meta.txt"),windows(path+".windows.csv");
  if(!events||!checks||!meta||!windows)throw std::runtime_error("Cannot create output files");
  events<<std::setprecision(17);checks<<std::setprecision(17);meta<<std::setprecision(17);
  events<<"m,n,k,implementation,round,order,iteration,milliseconds,tflops\n";
  checks<<"m,n,k,implementation,round,reference,count,max_abs,max_rel,relative_l2,nonfinite,abs_001_fail,simt_allclose_fail,native_eps,native_allclose_fail\n";
  bits<<"implementation,round,count,bitwise_mismatches\n";
  windows<<"implementation,round,start_utc_ms,end_utc_ms\n";
  std::array<std::vector<float>,3> aa={d.hostA,copyFloat(d.ta,d.ac),copyFloat(d.haf,d.ac)};
  std::array<std::vector<float>,3> bb={d.hostB,copyFloat(d.tb,d.bc),copyFloat(d.hbf,d.bc)};
  meta<<"max_shared_optin_bytes="<<prop.sharedMemPerBlockOptin<<"\nctas_per_sm_original="<<occOriginal
      <<"\nctas_per_sm_dynamic16="<<occ16<<"\nctas_per_sm_dynamic32="<<occ32<<'\n';
  meta<<"seed="<<Seed<<"\ncombined_tf32_fp16_preparation_ms="<<prepMs<<"\ncuBLAS_version="<<version
      <<"\nreset=zero C0 D2D, same stream, before every warmup/timed invocation, outside events"
      <<"\ninput_conversion=GPU __float_to_tf32 and __float2half_rn, outside GEMM events"
      <<"\nfp16_reduction=FP32 (DISALLOW_REDUCED_PRECISION_REDUCTION)\n";
  for(int p=0;p<3;++p)meta<<"input_"<<p<<"_a_fnv1a64="<<hash(aa[p].data(),d.ac*4)
                         <<"\ninput_"<<p<<"_b_fnv1a64="<<hash(bb[p].data(),d.bc*4)<<'\n';
  // Reference1: 原始FP32数学；Reference2: 同一量化输入，用PEDANTIC FP32计算。
  std::array<std::vector<float>,3> references,blasReferences;
  for(int p=0;p<3;++p){
    void* ap=p==0?d.a.p:(p==1?d.ta.p:d.haf.p);
    void* bp=p==0?d.b.p:(p==1?d.tb.p:d.hbf.p);
    blas(d.strict.h,m,n,k,ap,bp,d.c.p,CUDA_R_32F,CUBLAS_COMPUTE_32F_PEDANTIC);
    references[p]=d.output(Strict);
    if(p==2)for(auto& v:references[p])v=__half2float(__float2half_rn(v));
    d.reset(p==0?Strict:(p==1?TFBlas:HBlas));d.launch(p==0?Strict:(p==1?TFBlas:HBlas));
    blasReferences[p]=d.output(p==0?Strict:(p==1?TFBlas:HBlas));
  }
  // 小矩形全量double；方阵256个确定性位置。每种精度使用实际表示的A/B。
  std::vector<size_t> indices;
  if(m==128&&n==1024){indices.resize(d.cc);std::iota(indices.begin(),indices.end(),size_t(0));}
  else{indices={0,size_t(n-1),size_t(m-1)*n,d.cc-1};
    for(size_t i=0;indices.size()<256;++i){size_t x=(Seed+i*104729U)%d.cc;
      if(std::find(indices.begin(),indices.end(),x)==indices.end())indices.push_back(x);}}
  meta<<"cpu_double_positions="<<indices.size()<<"/"<<d.cc<<"\n";
  std::array<std::vector<double>,3> cpu;
  for(int p=0;p<3;++p)for(size_t index:indices){
    double sum=0;size_t row=index/n,col=index%n;
    for(int t=0;t<k;++t)sum+=double(aa[p][row*k+t])*double(bb[p][size_t(t)*n+col]);
    if(p==2)sum=__half2float(__float2half_rn(float(sum)));
    cpu[p].push_back(sum);
  }
  bool failed=false;
  std::array<std::vector<float>,3> baseline;
  for(int p=1;p<3;++p){int impl=p==1?TFDouble:HDouble;
    d.reset(impl);d.launch(impl);baseline[p]=d.output(impl);}
  auto chosen=[&](int impl){
    if(selected>=0)return selected==impl;
    return impl==TFDouble||impl==TFBlas||impl==TFK32||(selected!=-2&&impl==TFK16);
  };
  auto validate=[&](int impl,int round){
    auto actual=d.output(impl);int p=precision(impl);
    double eps=p==0?.001:(p==1?.1:.016); // 上游各精度判据，另列严格标准，绝不覆盖。
    Error original,represented,same,cpuError;
    for(size_t i=0;i<d.cc;++i){
      original.add(actual[i],references[0][i],eps);
      represented.add(actual[i],references[p][i],eps);
      same.add(actual[i],blasReferences[p][i],eps);
    }
    for(size_t i=0;i<indices.size();++i)cpuError.add(actual[indices[i]],cpu[p][i],eps);
    emit(checks,m,n,k,Names[impl],round,"original_fp32",original,eps);
    emit(checks,m,n,k,Names[impl],round,"represented_input_strict_reference",represented,eps);
    emit(checks,m,n,k,Names[impl],round,"same_precision_cublas",same,eps);
    emit(checks,m,n,k,Names[impl],round,"represented_input_cpu_double",cpuError,eps);
    // 未量化FP32标准仍报告FAIL；它不能自动成为Tensor Core算法错误的判据。
    // 0.01严格FP32判据不改。Tensor精度另按原作者判据判断计算契约，
    // 即使量化输入参考的0.01失败也如实保留，不能把dtype_gate PASS写成FP32 PASS。
    bool bad=represented.nonfinite||same.nonfinite||cpuError.nonfinite||
             represented.native||same.native||cpuError.native;
    if(p==0)bad|=represented.abs001||cpuError.abs001;
    if(p>0&&impl!=TFBlas&&impl!=HBlas){
      size_t mismatch=0;
      for(size_t i=0;i<d.cc;++i)
        mismatch+=std::memcmp(&actual[i],&baseline[p][i],sizeof(float))!=0;
      bits<<Names[impl]<<','<<round<<','<<d.cc<<','<<mismatch<<'\n';bits.flush();
      bad|=mismatch!=0;
    }
    failed|=bad;
    std::cout<<Names[impl]<<" round="<<round<<" original_max_abs="<<original.maxAbs
             <<" represented_max_abs="<<represented.maxAbs<<" same_blas_max_abs="<<same.maxAbs
             <<" dtype_gate="<<(bad?"FAIL":"PASS")
             <<" original_fp32_abs_001="<<(original.abs001?"FAIL":"PASS")<<'\n';
    checks.flush();
  };
  // 每个候选先检查，验证失败不进入正式计时。
  for(int impl=0;impl<10;++impl)if(chosen(impl)){d.reset(impl);d.launch(impl);validate(impl,-1);}
  if(failed)throw std::runtime_error("Preflight numerical validation failed; evidence retained");
  for(int r=0;r<rounds;++r){
    std::vector<int> pair=r%2?std::vector<int>{TFK32,TFK16}:std::vector<int>{TFK16,TFK32};
    std::array<std::vector<int>,3> groups;
    if(selected==-2){
      groups={r%2?std::vector<int>{TFK32,TFDouble}:std::vector<int>{TFDouble,TFK32},
              std::vector<int>{TFBlas},std::vector<int>{}};
    }else{
      groups={pair,std::vector<int>{TFDouble},std::vector<int>{TFBlas}};
    }
    int order=0;
    for(int g=0;g<3;++g)for(int impl:groups[(r+g)%3]){
      if(!chosen(impl))continue;
      long long begin=utcMs();
      for(int w=0;w<warm;++w){d.reset(impl);d.launch(impl);}
      ck(cudaDeviceSynchronize());
      for(int i=0;i<repeats;++i){
        d.reset(impl);ck(cudaEventRecord(start.e));d.launch(impl);
        ck(cudaEventRecord(stop.e));ck(cudaEventSynchronize(stop.e));
        float ms=0;ck(cudaEventElapsedTime(&ms,start.e,stop.e));
        if(!(ms>0)||!std::isfinite(ms))throw std::runtime_error("Invalid event");
        events<<m<<','<<n<<','<<k<<','<<Names[impl]<<','<<r<<','<<order<<','<<i<<','<<ms
              <<','<<(2.*m*n*k/(ms*1e9))<<'\n';
      }
      validate(impl,r);windows<<Names[impl]<<','<<r<<','<<begin<<','<<utcMs()<<'\n';
      events.flush();windows.flush();++order;
    }
  }
  if(failed)throw std::runtime_error("Numerical validation failed");
  std::cout<<"COMPLETE raw events retained; original-FP32 failures are reported separately\n";
  return 0;
}catch(const std::exception& e){std::cerr<<"ERROR: "<<e.what()<<'\n';return 1;}
