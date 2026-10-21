// Host-only robustness extension. Device headers are included unchanged.
// Source harness: ../tensorcore-bn64/benchmark.cu; retained author/license in included headers.
// Three measured implementations: original TF32 DBF, BN64/BK32 and same-input cuBLAS TF32.
// C=AB; explicit TF32 input conversion, FP32 accumulation/output; preparation excluded from events.
// The original helper's unused FP16 preparation/data branches remain for build provenance only.
// CLI: --batch batch plan.csv outdir; --smoke outdir;
//      --case M N K seed rounds repeats warmup output.csv;
//      --profile implementation N [seed] (one GEMM inside cudaProfilerStart/Stop).
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
#include <filesystem>
#include <set>
#include <map>
#include <process.h>
#include "../../src/kernels/30_kernel_mini_simt.cuh"
#include "../tensorcore/vendor/vitamin_tf32.cuh"
#include "../tensorcore/vendor/vitamin_fp16.cuh"

#include "../tensorcore-ktile/tf32_ktile.cuh"
#include "../tensorcore-bn64/tf32_bn64.cuh"

uint64_t cudaChecks=0,blasChecks=0;
void ck(cudaError_t s) { ++cudaChecks; if(s != cudaSuccess) throw std::runtime_error("CUDA status="+std::to_string(s)+" "+cudaGetErrorString(s)); }
void bk(cublasStatus_t s) { ++blasChecks; if(s != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cuBLAS "+std::to_string(s)); }
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
constexpr uint32_t SampleSeed=20261020;
uint32_t ActiveSeed=20261020;
int ActiveBatch=0,ActiveCase=0;
uint32_t rng(uint32_t& x){x^=x<<13;x^=x>>17;x^=x<<5;return x;}
std::vector<float> input(size_t count, unsigned stream){
  std::vector<float> out(count);
  uint32_t state=ActiveSeed^(0x9e3779b9U*(stream+1U));
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
enum Impl {K30,Strict,TFSingle,TFDouble,TFBlas,HSingle,HDouble,HBlas,TFK16,TFK32,TFBN64};
const std::array<const char*,11> Names={{
    "k30_fp32","cublas_strict_fp32","vitamin_tf32_single","vitamin_tf32_dbf","cublas_tf32",
    "vitamin_fp16_single","vitamin_fp16_dbf_rw","cublas_fp16",
    "tf32_dynamic_bk16","tf32_dynamic_bk32","tf32_bn64_bk32"}};
int findImpl(const std::string& s){
  for(int i=0;i<11;++i) if(s==Names[i])return i;
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
    ck(cudaFuncSetAttribute(bn64_tf32::sgemm_tf32_swizzle_bcf_dbf_kernel<128,64,32>,
                           cudaFuncAttributeMaxDynamicSharedMemorySize,49152));
    ck(cudaMemset(c0.p,0,cc*4));ck(cudaMemset(hc0.p,0,cc*2));
  }
  void reset(int impl){
    if(precision(impl)==2)ck(cudaMemcpyAsync(hc.p,hc0.p,cc*2,cudaMemcpyDeviceToDevice));
    else ck(cudaMemcpyAsync(c.p,c0.p,cc*4,cudaMemcpyDeviceToDevice));
  }
  void launch(int impl){
    dim3 grid(n/128,m/128);
    switch(impl){
      case TFBN64:bn64_tf32::sgemm_tf32_swizzle_bcf_dbf_kernel<128,64,32>
          <<<dim3(n/64,m/128),256,49152>>>(ta.f(),tb.f(),c.f(),m,n,k);break;
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
     <<e.nonfinite<<','<<e.abs001<<','<<e.simt<<','<<eps<<','<<e.native<<','<<ActiveBatch<<','<<ActiveSeed<<','<<ActiveCase<<'\n';
}
void shapeCheck(int m,int n,int k){
  // 原核无尾块保护；宽8的grid swizzle必须覆盖完整的8列CTA。
  if(m<128||m>4096||m%128||n<1024||n>4096||n%1024||k<32||k>4096||k%32)
    throw std::runtime_error("Unsupported shape: require M=128..4096 multiple128, N=1024..4096 multiple1024, K=32..4096 multiple32");
}

using Order=std::array<int,3>;
const std::array<Order,6> Permutations={{
  {{TFDouble,TFBN64,TFBlas}},{{TFDouble,TFBlas,TFBN64}},
  {{TFBN64,TFDouble,TFBlas}},{{TFBN64,TFBlas,TFDouble}},
  {{TFBlas,TFDouble,TFBN64}},{{TFBlas,TFBN64,TFDouble}}}};
struct CasePlan {
  int batch=0,index=0,m=0,n=0,k=0;uint32_t seed=0;
  std::vector<Order> orders;
};
void refuseExisting(const std::string& path){
  for(const auto& suffix:{"",".checks.csv",".bitwise.csv",".meta.txt",".windows.csv",".inputs.csv",".cpu-indices.csv"})
    if(std::filesystem::exists(path+suffix))throw std::runtime_error("Refusing overwrite: "+path+suffix);
}
void runCase(const CasePlan& plan,int repeats,int warm,const std::string& path,int profile=-1){
  int m=plan.m,n=plan.n,k=plan.k;
  shapeCheck(m,n,k);
  if(profile<0)refuseExisting(path);
  ActiveBatch=plan.batch;ActiveCase=plan.index;ActiveSeed=plan.seed;
  cudaDeviceProp prop{};ck(cudaGetDeviceProperties(&prop,0));
  if(prop.major!=8||prop.minor!=9)throw std::runtime_error("This archive requires sm_89 hardware");
  const char* ov=std::getenv("NVIDIA_TF32_OVERRIDE");
  if(ov&&std::string(ov)=="0")throw std::runtime_error("NVIDIA_TF32_OVERRIDE=0 disables the required cuBLAS TF32 configuration");
  Data d(m,n,k);int version=0;bk(cublasGetVersion(d.tensor.h,&version));
  cublasMath_t tensorMath,strictMath;cublasAtomicsMode_t atomics;cudaStream_t stream;
  bk(cublasGetMathMode(d.tensor.h,&tensorMath));bk(cublasGetMathMode(d.strict.h,&strictMath));
  bk(cublasGetAtomicsMode(d.tensor.h,&atomics));bk(cublasGetStream(d.tensor.h,&stream));
  if(tensorMath!=CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION || strictMath!=CUBLAS_PEDANTIC_MATH ||
      atomics!=CUBLAS_ATOMICS_NOT_ALLOWED || stream!=nullptr)throw std::runtime_error("cuBLAS configuration differs from preserved harness");
  Event start,stop;
  ck(cudaEventRecord(start.e));
  // Preserved input preparation, including the unused FP16 side product. Conversion is outside GEMM timing.
  prepareInputs<<<(d.ac+255)/256,256>>>(d.a.f(),d.ta.f(),d.ha.h(),d.haf.f(),d.ac);
  prepareInputs<<<(d.bc+255)/256,256>>>(d.b.f(),d.tb.f(),d.hb.h(),d.hbf.f(),d.bc);
  ck(cudaGetLastError());ck(cudaEventRecord(stop.e));ck(cudaEventSynchronize(stop.e));
  float prepMs=0;ck(cudaEventElapsedTime(&prepMs,start.e,stop.e));
  if(profile>=0){
    for(int w=0;w<10;++w){d.reset(profile);d.launch(profile);}
    ck(cudaDeviceSynchronize());d.reset(profile);ck(cudaDeviceSynchronize());
    ck(cudaProfilerStart());d.launch(profile);ck(cudaDeviceSynchronize());ck(cudaProfilerStop());
    std::cout<<"PROFILE_COMPLETE "<<Names[profile]<<" seed="<<ActiveSeed<<'\n';return;
  }
  std::ofstream events(path),checks(path+".checks.csv"),bits(path+".bitwise.csv"),meta(path+".meta.txt"),
    windows(path+".windows.csv"),inputs(path+".inputs.csv"),sampleFile(path+".cpu-indices.csv");
  if(!events||!checks||!bits||!meta||!windows||!inputs||!sampleFile)throw std::runtime_error("Cannot open archive outputs");
  events<<std::setprecision(17);checks<<std::setprecision(17);meta<<std::setprecision(17);
  events<<"m,n,k,implementation,round,order,iteration,milliseconds,tflops,batch,seed,case_index,cuda_launch_status,cuda_event_sync_status\n";
  checks<<"m,n,k,implementation,round,reference,count,max_abs,max_rel,relative_l2,nonfinite,abs_001_fail,simt_allclose_fail,native_eps,native_allclose_fail,batch,seed,case_index\n";
  bits<<"implementation,round,count,bitwise_mismatches,batch,seed,case_index\n";
  windows<<"implementation,round,start_utc_ms,end_utc_ms,batch,seed,case_index\n";
  inputs<<"batch,seed,case_index,m,n,k,representation,matrix,element_count,bytes,fnv1a64,stage\n";
  sampleFile<<"sample_index,linear_index,row,col\n";
  std::array<std::vector<float>,2> aa={d.hostA,copyFloat(d.ta,d.ac)},bb={d.hostB,copyFloat(d.tb,d.bc)};
  auto inputHashes=[&](const char* stage){
    // Read actual device bytes for before/after fingerprints; this is outside all measured events.
    const std::array<std::vector<float>,4> copied={copyFloat(d.a,d.ac),copyFloat(d.b,d.bc),copyFloat(d.ta,d.ac),copyFloat(d.tb,d.bc)};
    for(int p=0;p<2;++p)for(int matrix=0;matrix<2;++matrix){
      const auto& v=copied[p*2+matrix];
      inputs<<ActiveBatch<<','<<ActiveSeed<<','<<ActiveCase<<','<<m<<','<<n<<','<<k<<','
        <<(p?"tf32":"original_fp32")<<','<<(matrix?"B":"A")<<','<<v.size()<<','<<v.size()*4<<','
        <<hash(v.data(),v.size()*4)<<','<<stage<<'\n';
      const auto& expected=matrix?bb[p]:aa[p];
      if(v.size()!=expected.size()||std::memcmp(v.data(),expected.data(),v.size()*4))
        throw std::runtime_error("Input bytes changed during experiment");
    }inputs.flush();
  };
  inputHashes("before");
  int occOriginal=0,occBn64=0;cudaFuncAttributes originalAttr{},bn64Attr{};
  ck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occOriginal,vitamin_tf32::sgemm_tf32_swizzle_bcf_dbf_kernel<128,128,16>,256,0));
  ck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occBn64,bn64_tf32::sgemm_tf32_swizzle_bcf_dbf_kernel<128,64,32>,256,49152));
  ck(cudaFuncGetAttributes(&originalAttr,vitamin_tf32::sgemm_tf32_swizzle_bcf_dbf_kernel<128,128,16>));
  ck(cudaFuncGetAttributes(&bn64Attr,bn64_tf32::sgemm_tf32_swizzle_bcf_dbf_kernel<128,64,32>));
  meta<<"batch="<<ActiveBatch<<"\nseed="<<ActiveSeed<<"\ncase_index="<<ActiveCase<<"\npid="<<_getpid()
      <<"\ncase_start_utc_ms="<<utcMs()<<"\ngpu="<<prop.name<<"\nsm_count="<<prop.multiProcessorCount
      <<"\ncuBLAS_version="<<version<<"\ntensor_compute=CUBLAS_COMPUTE_32F_FAST_TF32\ntensor_compute_enum="<<int(CUBLAS_COMPUTE_32F_FAST_TF32)
      <<"\nmath_mode=CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION\nmath_mode_enum="<<int(tensorMath)
      <<"\natomics=CUBLAS_ATOMICS_NOT_ALLOWED\natomics_enum="<<int(atomics)
      <<"\nalgorithm=CUBLAS_GEMM_DEFAULT\nalgorithm_enum="<<int(CUBLAS_GEMM_DEFAULT)
      <<"\nstrict_compute=CUBLAS_COMPUTE_32F_PEDANTIC\nstrict_math=CUBLAS_PEDANTIC_MATH"
      <<"\nalpha=1\nbeta=0\nstream=default\nshared_tf32_input_pointers=true"
      <<"\ncombined_tf32_fp16_preparation_ms="<<prepMs
      <<"\nreset=zero C0 D2D before every invocation on same stream outside events"
      <<"\nconversion=preserved GPU __float_to_tf32 and unused FP16 side product outside GEMM events"
      <<"\nhash=FNV1a64 over actual little-endian float32 bytes; non-cryptographic"
      <<"\nallclose=abs(got-ref)<=eps+eps*abs(got); TF32 eps=0.1; SIMT eps=0.001"
      <<"\nrelative_error=abs(got-ref)/max(abs(ref),1e-12)"
      <<"\nctas_per_sm_original="<<occOriginal<<"\nctas_per_sm_bn64="<<occBn64
      <<"\nregisters_original="<<originalAttr.numRegs<<"\nregisters_bn64="<<bn64Attr.numRegs
      <<"\nstatic_shared_original="<<originalAttr.sharedSizeBytes<<"\ndynamic_shared_bn64=49152\n";
  std::array<std::vector<float>,2> references;
  for(int p=0;p<2;++p){
    blas(d.strict.h,m,n,k,p?d.ta.p:d.a.p,p?d.tb.p:d.b.p,d.c.p,CUDA_R_32F,CUBLAS_COMPUTE_32F_PEDANTIC);
    references[p]=d.output(Strict);
  }
  d.reset(TFBlas);d.launch(TFBlas);auto blasReference=d.output(TFBlas);
  d.reset(TFDouble);d.launch(TFDouble);auto baseline=d.output(TFDouble);
  std::vector<size_t> indices;
  if(m==128&&n==1024){indices.resize(d.cc);std::iota(indices.begin(),indices.end(),size_t(0));}
  else{indices={0,size_t(n-1),size_t(m-1)*n,d.cc-1};
    for(size_t i=0;indices.size()<256;++i){size_t index=(SampleSeed+i*104729U)%d.cc;
      if(std::find(indices.begin(),indices.end(),index)==indices.end())indices.push_back(index);}}
  std::vector<double> cpu;
  for(size_t s=0;s<indices.size();++s){
    size_t index=indices[s],row=index/n,col=index%n;double sum=0;
    sampleFile<<s<<','<<index<<','<<row<<','<<col<<'\n';
    for(int t=0;t<k;++t)sum+=double(aa[1][row*k+t])*double(bb[1][size_t(t)*n+col]);
    cpu.push_back(sum);
  }sampleFile.flush();
  meta<<"cpu_double_positions="<<indices.size()<<'/'<<d.cc<<"\ncpu_index_seed="<<SampleSeed<<'\n';meta.flush();
  auto validate=[&](int impl,int round){
    auto actual=d.output(impl);Error original,represented,same,cpuError,author;constexpr double eps=.1;
    size_t mismatch=0;
    for(size_t i=0;i<d.cc;++i){
      original.add(actual[i],references[0][i],eps);represented.add(actual[i],references[1][i],eps);
      same.add(actual[i],blasReference[i],eps);author.add(actual[i],baseline[i],eps);
      if(impl!=TFBlas)mismatch+=std::memcmp(&actual[i],&baseline[i],4)!=0;
    }
    for(size_t i=0;i<indices.size();++i)cpuError.add(actual[indices[i]],cpu[i],eps);
    emit(checks,m,n,k,Names[impl],round,"original_fp32",original,eps);
    emit(checks,m,n,k,Names[impl],round,"represented_input_strict_reference",represented,eps);
    emit(checks,m,n,k,Names[impl],round,"same_precision_cublas",same,eps);
    emit(checks,m,n,k,Names[impl],round,"represented_input_cpu_double",cpuError,eps);
    emit(checks,m,n,k,Names[impl],round,"vitamin_tf32_baseline",author,eps);checks.flush();
    if(impl!=TFBlas){bits<<Names[impl]<<','<<round<<','<<d.cc<<','<<mismatch<<','<<ActiveBatch<<','<<ActiveSeed<<','<<ActiveCase<<'\n';bits.flush();}
    bool bad=represented.nonfinite||same.nonfinite||cpuError.nonfinite||author.nonfinite||
      represented.native||same.native||cpuError.native||author.native||mismatch;
    std::cout<<"batch="<<ActiveBatch<<" seed="<<ActiveSeed<<" n="<<n<<' '<<Names[impl]<<" round="<<round
      <<" dtype_gate="<<(bad?"FAIL":"PASS")<<" original_abs001_fail="<<original.abs001<<" bitwise="<<mismatch<<'\n';
    if(bad)throw std::runtime_error("Numerical/bitwise gate failed; no subsequent rounds or cases run");
  };
  for(int impl:{TFDouble,TFBN64,TFBlas}){d.reset(impl);d.launch(impl);validate(impl,-1);}
  for(size_t r=0;r<plan.orders.size();++r)for(int order=0;order<3;++order){
    int impl=plan.orders[r][order];long long begin=utcMs();
    for(int w=0;w<warm;++w){d.reset(impl);d.launch(impl);}ck(cudaDeviceSynchronize());
    for(int i=0;i<repeats;++i){
      d.reset(impl);ck(cudaEventRecord(start.e));d.launch(impl);
      ck(cudaEventRecord(stop.e));cudaError_t sync=cudaEventSynchronize(stop.e);ck(sync);
      float ms=0;ck(cudaEventElapsedTime(&ms,start.e,stop.e));
      if(!(ms>0)||!std::isfinite(ms))throw std::runtime_error("Invalid CUDA event time");
      events<<m<<','<<n<<','<<k<<','<<Names[impl]<<','<<r<<','<<order<<','<<i<<','<<ms<<','
        <<2.*m*n*k/(ms*1e9)<<','<<ActiveBatch<<','<<ActiveSeed<<','<<ActiveCase<<",0,"<<int(sync)<<'\n';
    }
    events.flush();validate(impl,int(r));
    windows<<Names[impl]<<','<<r<<','<<begin<<','<<utcMs()<<','<<ActiveBatch<<','<<ActiveSeed<<','<<ActiveCase<<'\n';windows.flush();
  }
  inputHashes("after");
  meta<<"case_end_utc_ms="<<utcMs()<<"\ncumulative_cuda_calls_checked_success="<<cudaChecks
      <<"\ncumulative_cublas_calls_checked_success="<<blasChecks<<"\nCASE_COMPLETE=true\n";
}
std::vector<std::string> split(const std::string& line){
  std::vector<std::string> r;std::istringstream in(line);std::string field;
  while(std::getline(in,field,',')){if(!field.empty()&&field.back()=='\r')field.pop_back();r.push_back(field);}return r;
}
std::vector<CasePlan> loadPlan(const std::string& file,int batch){
  std::ifstream input(file);if(!input)throw std::runtime_error("Cannot read plan");
  std::string line;std::getline(input,line);
  if(line!="batch,case_index,m,n,k,seed,round,order,implementation"&&line!="batch,case_index,m,n,k,seed,round,order,implementation\r")
    throw std::runtime_error("Unexpected plan schema");
  std::vector<CasePlan> cases;
  while(std::getline(input,line)){
    auto f=split(line);if(f.size()!=9)throw std::runtime_error("Malformed plan row");
    if(std::stoi(f[0])!=batch)continue;
    int index=std::stoi(f[1]),round=std::stoi(f[6]),order=std::stoi(f[7]);
    if(index<0||index>=12||round<0||round>=7||order<0||order>=3)throw std::runtime_error("Invalid plan indices");
    if(index==int(cases.size())){CasePlan c;c.batch=batch;c.index=index;c.m=std::stoi(f[2]);c.n=std::stoi(f[3]);c.k=std::stoi(f[4]);c.seed=uint32_t(std::stoul(f[5]));c.orders.resize(7,Order{{-1,-1,-1}});cases.push_back(c);}
    if(index!=int(cases.size())-1)throw std::runtime_error("Plan cases must be contiguous in execution order");
    auto& c=cases.back();
    if(c.m!=std::stoi(f[2])||c.n!=std::stoi(f[3])||c.k!=std::stoi(f[4])||c.seed!=uint32_t(std::stoul(f[5]))||c.orders[round][order]!=-1)
      throw std::runtime_error("Inconsistent or duplicate plan row");
    c.orders[round][order]=findImpl(f[8]);
  }
  if(cases.size()!=12)throw std::runtime_error("Expected 12 cases per batch");
  std::set<std::pair<int,uint32_t>> found;
  for(const auto& c:cases){
    if(c.m!=c.n||c.n!=c.k||std::set<int>{1024,2048,3072,4096}.count(c.n)!=1||std::set<uint32_t>{20261020,42,12345}.count(c.seed)!=1||!found.emplace(c.n,c.seed).second)
      throw std::runtime_error("Invalid/duplicate formal shape or seed");
    std::set<Order> unique;
    for(int r=0;r<7;++r){if(std::set<int>(c.orders[r].begin(),c.orders[r].end())!=std::set<int>{TFDouble,TFBN64,TFBlas})throw std::runtime_error("Invalid round permutation");if(r<6)unique.insert(c.orders[r]);}
    if(unique.size()!=6||c.orders[6]!=Permutations[batch-1])throw std::runtime_error("Required six permutations / extra batch rotation missing");
  }return cases;
}
int main(int argc,char** argv)try{
  if(argc<2)throw std::runtime_error("Use --batch batch plan.csv outdir; --smoke outdir; --case M N K seed rounds repeats warmup output.csv; --profile implementation N [seed]");
  std::string mode=argv[1];
  if(mode=="--profile"&&(argc==4||argc==5)){
    CasePlan c;c.m=c.n=c.k=std::stoi(argv[3]);c.seed=argc==5?uint32_t(std::stoul(argv[4])):20261020;
    int impl=findImpl(argv[2]);if(impl!=TFDouble&&impl!=TFBN64&&impl!=TFBlas)throw std::runtime_error("Profile implementation outside robustness scope");
    runCase(c,1,10,"",impl);return 0;
  }
  if(mode=="--case"&&argc==10){
    CasePlan c;c.m=std::stoi(argv[2]);c.n=std::stoi(argv[3]);c.k=std::stoi(argv[4]);c.seed=uint32_t(std::stoul(argv[5]));
    int rounds=std::stoi(argv[6]),repeats=std::stoi(argv[7]),warm=std::stoi(argv[8]);
    if(rounds<1||rounds>100||repeats<1||repeats>10000||warm<0||warm>1000)throw std::runtime_error("Invalid counts");
    for(int r=0;r<rounds;++r)c.orders.push_back(Permutations[r%6]);runCase(c,repeats,warm,argv[9]);return 0;
  }
  if(mode=="--smoke"&&argc==3){
    std::filesystem::path out=argv[2];if(std::filesystem::exists(out))throw std::runtime_error("Refusing existing smoke directory");std::filesystem::create_directories(out);
    for(int k:{32,64,96}){CasePlan c;c.m=128;c.n=1024;c.k=k;c.seed=20261020;c.orders={Permutations[0]};runCase(c,1,1,(out/("small-"+std::to_string(k)+".csv")).string());}
    std::ofstream(out/"SMOKE_COMPLETE.txt")<<"3 legal small rectangles; CPU double all 131072 outputs per case\n";return 0;
  }
  if(mode=="--batch"&&argc==5){
    int batch=std::stoi(argv[2]);if(batch<1||batch>3)throw std::runtime_error("Batch must be 1..3");
    auto cases=loadPlan(argv[3],batch);std::filesystem::path out=argv[4];
    if(std::filesystem::exists(out))throw std::runtime_error("Refusing existing batch directory");std::filesystem::create_directories(out);
    std::ofstream process(out/"process.csv");process<<"batch,pid,start_utc_ms,end_utc_ms,cases,formal_events,status\n";
    auto begin=utcMs();std::cout<<"BATCH_START batch="<<batch<<" pid="<<_getpid()<<" utc_ms="<<begin<<std::endl;
    for(const auto& c:cases){std::cout<<"CASE_START index="<<c.index<<" n="<<c.n<<" seed="<<c.seed<<std::endl;runCase(c,30,10,(out/("timings-"+std::to_string(c.n)+"-"+std::to_string(c.seed)+".csv")).string());}
    process<<batch<<','<<_getpid()<<','<<begin<<','<<utcMs()<<",12,7560,COMPLETE\n";process.flush();
    ck(cudaDeviceSynchronize());
    std::ofstream(out/"BATCH_COMPLETE.txt")<<"7560 formal single-GEMM CUDA events; original FP32 failures retained\n";return 0;
  }
  throw std::runtime_error("Invalid arguments");
}catch(const std::exception& e){std::cerr<<"ERROR: "<<e.what()<<'\n';return 1;}
