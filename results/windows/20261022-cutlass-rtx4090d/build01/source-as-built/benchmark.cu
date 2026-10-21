// Native Windows CUTLASS comparison. Existing device kernels are included unchanged.
// Host protocol derives from ../tensorcore-bn64-robustness/benchmark.cu.
// TF32 and strict FP32 groups use separate tables; all compute row-major C=AB.
// Input preparation, CUTLASS initialization/workspace and C0 restoration are outside events.
// Every event encloses the public launch API, including any GPU-visible enqueue gaps.
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
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef _WIN32
#include <process.h>
#else
#include <unistd.h>
#endif
#include "cutlass_adapter.cuh"
#include "../../src/kernels/10_kernel_warptiling.cuh"
#include "../../src/kernels/30_kernel_mini_simt.cuh"
#include "../tensorcore/vendor/vitamin_tf32.cuh"
#include "../tensorcore-bn64/tf32_bn64.cuh"

uint64_t cudaChecks=0,blasChecks=0;
void ck(cudaError_t s){++cudaChecks;if(s!=cudaSuccess)throw std::runtime_error("CUDA "+std::to_string(s)+" "+cudaGetErrorString(s));}
void bk(cublasStatus_t s){++blasChecks;if(s!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error("cuBLAS "+std::to_string(s));}
struct Buf{
  void* p=nullptr;
  explicit Buf(size_t bytes){ck(cudaMalloc(&p,bytes));}
  ~Buf(){if(p)cudaFree(p);}
  Buf(const Buf&)=delete;Buf& operator=(const Buf&)=delete;
  float* f(){return static_cast<float*>(p);}
};
struct Event{cudaEvent_t e{};Event(){ck(cudaEventCreate(&e));}~Event(){cudaEventDestroy(e);}};
struct Handle{
  cublasHandle_t h{};
  explicit Handle(bool strict){
    bk(cublasCreate(&h));bk(cublasSetStream(h,nullptr));bk(cublasSetAtomicsMode(h,CUBLAS_ATOMICS_NOT_ALLOWED));
    bk(cublasSetMathMode(h,strict?CUBLAS_PEDANTIC_MATH:CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION));
  }
  ~Handle(){cublasDestroy(h);}
};
enum Impl{Author,BN64,CutTF128,CutTF64,TFBlas,K10,K30,CutSimt,Strict,ImplCount};
const std::array<const char*,ImplCount> Names={{"vitamin_tf32_dbf","tf32_bn64_bk32","cutlass_tf32_128x128",
  "cutlass_tf32_128x64_bk16","cublas_tf32","k10_fp32","k30_fp32","cutlass_simt_256x128","cublas_strict_fp32"}};
const std::vector<int> TensorImpls={Author,BN64,CutTF128,CutTF64,TFBlas},StrictImpls={K10,K30,CutSimt,Strict};
const std::vector<int>& implementations(const std::string& group){
  if(group=="tf32")return TensorImpls;if(group=="fp32")return StrictImpls;
  throw std::runtime_error("Group must be tf32 or fp32");
}
int findImpl(const std::string& name){for(int i=0;i<ImplCount;++i)if(name==Names[i])return i;throw std::runtime_error("Unknown implementation "+name);}
bool isTensor(int impl){return impl<=TFBlas;}
constexpr uint32_t SampleSeed=20261020;
uint32_t ActiveSeed=20261020;int ActiveBatch=0,ActiveCase=0;
uint32_t rng(uint32_t& x){x^=x<<13;x^=x>>17;x^=x<<5;return x;}
std::vector<float> input(size_t count,unsigned stream){
  std::vector<float> v(count);uint32_t state=ActiveSeed^(0x9e3779b9U*(stream+1U));
  for(auto& x:v){auto i=rng(state)%5U,f=rng(state)%5U;bool negative=(rng(state)&1U)!=0;x=(negative?-1.f:1.f)*(float(i)+.01f*float(f));}return v;
}
std::string hash(const void* data,size_t bytes){
  uint64_t h=14695981039346656037ULL;const auto* p=static_cast<const unsigned char*>(data);
  for(size_t i=0;i<bytes;++i){h^=p[i];h*=1099511628211ULL;}std::ostringstream o;o<<std::hex<<h;return o.str();
}
long long utcMs(){return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();}
int processId(){
#ifdef _WIN32
  return _getpid();
#else
  return int(getpid());
#endif
}
// Same TF32 conversion as the preserved benchmark. No CPU approximation or TF32x3.
__global__ void prepareInputs(const float* x,float* tf,size_t n){size_t i=blockIdx.x*size_t(blockDim.x)+threadIdx.x;if(i<n)tf[i]=nvcuda::wmma::__float_to_tf32(x[i]);}
void blas(cublasHandle_t h,int m,int n,int k,const float* a,const float* b,float* c,cublasComputeType_t compute){
  const float alpha=1,beta=0;
  // Row-major C=AB -> column-major C^T=B^T*A^T; swap both dimensions and pointers.
  bk(cublasGemmEx(h,CUBLAS_OP_N,CUBLAS_OP_N,n,m,k,&alpha,b,CUDA_R_32F,n,a,CUDA_R_32F,k,&beta,c,CUDA_R_32F,n,compute,CUBLAS_GEMM_DEFAULT));
}
using K30Shape=k30::Shape<128,128,16,64,64,4,4,128>;
struct Data{
  int m,n,k;size_t ac,bc,cc;
  Buf a,b,ta,tb,c,c0;Handle strict{true},tensor{false};
  std::vector<float> hostA,hostB;
  cutlass_compare::Tf32_128x128 cut128;
  cutlass_compare::Tf32_128x64 cut64;
  cutlass_compare::Simt_256x128 cutSimt;
  Data(int mm,int nn,int kk,const std::string& group):m(mm),n(nn),k(kk),ac(size_t(m)*k),bc(size_t(k)*n),cc(size_t(m)*n),
    a(ac*4),b(bc*4),ta(ac*4),tb(bc*4),c(cc*4),c0(cc*4),hostA(input(ac,0)),hostB(input(bc,1)){
    ck(cudaMemcpy(a.p,hostA.data(),ac*4,cudaMemcpyHostToDevice));ck(cudaMemcpy(b.p,hostB.data(),bc*4,cudaMemcpyHostToDevice));
    ck(cudaMemset(c0.p,0,cc*4));
    ck(cudaFuncSetAttribute(bn64_tf32::sgemm_tf32_swizzle_bcf_dbf_kernel<128,64,32>,cudaFuncAttributeMaxDynamicSharedMemorySize,49152));
    // All initialization and workspace allocation occur before any timed event.
    if(group=="tf32"){cut128.initialize(m,n,k,ta.f(),tb.f(),c.f());cut64.initialize(m,n,k,ta.f(),tb.f(),c.f());}
    else cutSimt.initialize(m,n,k,a.f(),b.f(),c.f());
  }
  void reset(){ck(cudaMemcpyAsync(c.p,c0.p,cc*4,cudaMemcpyDeviceToDevice));}
  void launch(int impl){
    dim3 grid(n/128,m/128);
    switch(impl){
      case Author:vitamin_tf32::sgemm_tf32_swizzle_bcf_dbf_kernel<128,128,16><<<grid,256>>>(ta.f(),tb.f(),c.f(),m,n,k);break;
      case BN64:bn64_tf32::sgemm_tf32_swizzle_bcf_dbf_kernel<128,64,32><<<dim3(n/64,m/128),256,49152>>>(ta.f(),tb.f(),c.f(),m,n,k);break;
      case CutTF128:cut128.run();break;
      case CutTF64:cut64.run();break;
      case TFBlas:blas(tensor.h,m,n,k,ta.f(),tb.f(),c.f(),CUBLAS_COMPUTE_32F_FAST_TF32);break;
      case K10:sgemmWarptiling<128,128,16,64,64,4,8,4,128,4><<<grid,128>>>(m,n,k,1.f,a.f(),b.f(),0.f,c.f());break;
      case K30:sgemmK30<K30Shape,2><<<grid,128,2*(k30::SharedLayoutA<K30Shape>::Size+k30::SharedLayoutB<K30Shape>::Size)*4>>>(m,n,k,1.f,a.f(),b.f(),0.f,c.f());break;
      case CutSimt:cutSimt.run();break;
      case Strict:blas(strict.h,m,n,k,a.f(),b.f(),c.f(),CUBLAS_COMPUTE_32F_PEDANTIC);break;
      default:throw std::runtime_error("Invalid implementation enum");
    }ck(cudaGetLastError());
  }
  std::vector<float> output(){std::vector<float> v(cc);ck(cudaMemcpy(v.data(),c.p,cc*4,cudaMemcpyDeviceToHost));return v;}
};
std::vector<float> copyFloat(Buf& b,size_t count){std::vector<float> v(count);ck(cudaMemcpy(v.data(),b.p,count*4,cudaMemcpyDeviceToHost));return v;}
struct Error{
  double maxAbs=0,maxRel=0,sumSq=0,refSq=0;size_t nonfinite=0,abs001=0,simt=0,native=0,count=0;
  void add(double got,double ref,double eps){
    ++count;if(!std::isfinite(got)||!std::isfinite(ref)){++nonfinite;++abs001;++simt;++native;return;}
    double e=std::abs(got-ref);maxAbs=std::max(maxAbs,e);maxRel=std::max(maxRel,e/std::max(std::abs(ref),1e-12));sumSq+=e*e;refSq+=ref*ref;
    abs001+=e>.01;simt+=e>.001+.001*std::abs(got);native+=e>eps+eps*std::abs(got);
  }
};
void emit(std::ofstream& out,int m,int n,int k,const char* name,int round,const char* reference,const Error& e,double eps,const std::string& group){
  out<<m<<','<<n<<','<<k<<','<<name<<','<<round<<','<<reference<<','<<e.count<<','<<e.maxAbs<<','<<e.maxRel<<','<<std::sqrt(e.sumSq/std::max(e.refSq,1e-30))<<','
    <<e.nonfinite<<','<<e.abs001<<','<<e.simt<<','<<eps<<','<<e.native<<','<<ActiveBatch<<','<<ActiveSeed<<','<<ActiveCase<<','<<group<<'\n';
}
void shapeCheck(int m,int n,int k){
  // Unchanged author kernels have no tail predicates. Grid width 8 needs complete CTA groups.
  if(m<128||m>4096||m%128||n<1024||n>4096||n%1024||k<32||k>4096||k%32)
    throw std::runtime_error("Unsupported shape: M=128..4096 multiple128; N=1024..4096 multiple1024; K=32..4096 multiple32");
}
using Order=std::vector<int>;
struct CasePlan{int batch=0,index=0,m=0,n=0,k=0;uint32_t seed=0;std::string group;std::vector<Order> orders;};
void refuseExisting(const std::string& path){
  for(const auto& suffix:{"",".checks.csv",".bitwise.csv",".meta.txt",".windows.csv",".inputs.csv",".cpu-indices.csv"})
    if(std::filesystem::exists(path+suffix))throw std::runtime_error("Refusing overwrite: "+path+suffix);
}
void runCase(const CasePlan& plan,int repeats,int warm,const std::string& path,int profile=-1){
  int m=plan.m,n=plan.n,k=plan.k;shapeCheck(m,n,k);const auto& impls=implementations(plan.group);bool tf=plan.group=="tf32";
  if(profile<0)refuseExisting(path);ActiveBatch=plan.batch;ActiveCase=plan.index;ActiveSeed=plan.seed;
  cudaDeviceProp prop{};ck(cudaGetDeviceProperties(&prop,0));if(prop.major!=8||prop.minor!=9)throw std::runtime_error("Requires sm_89 hardware");
  const char* ov=std::getenv("NVIDIA_TF32_OVERRIDE");if(tf&&ov&&std::string(ov)=="0")throw std::runtime_error("NVIDIA_TF32_OVERRIDE=0 disables required cuBLAS TF32");
  Data d(m,n,k,plan.group);int version=0;bk(cublasGetVersion(d.tensor.h,&version));
  cublasMath_t tensorMath,strictMath;cublasAtomicsMode_t atomics;cudaStream_t stream;
  bk(cublasGetMathMode(d.tensor.h,&tensorMath));bk(cublasGetMathMode(d.strict.h,&strictMath));bk(cublasGetAtomicsMode(d.tensor.h,&atomics));bk(cublasGetStream(d.tensor.h,&stream));
  if(tensorMath!=CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION||strictMath!=CUBLAS_PEDANTIC_MATH||atomics!=CUBLAS_ATOMICS_NOT_ALLOWED||stream!=nullptr)
    throw std::runtime_error("Unexpected cuBLAS configuration");
  Event start,stop;ck(cudaEventRecord(start.e));
  prepareInputs<<<(d.ac+255)/256,256>>>(d.a.f(),d.ta.f(),d.ac);prepareInputs<<<(d.bc+255)/256,256>>>(d.b.f(),d.tb.f(),d.bc);
  ck(cudaGetLastError());ck(cudaEventRecord(stop.e));ck(cudaEventSynchronize(stop.e));float prepMs=0;ck(cudaEventElapsedTime(&prepMs,start.e,stop.e));
  if(profile>=0){
    for(int w=0;w<10;++w){d.reset();d.launch(profile);}ck(cudaDeviceSynchronize());d.reset();ck(cudaDeviceSynchronize());
    ck(cudaProfilerStart());d.launch(profile);ck(cudaDeviceSynchronize());ck(cudaProfilerStop());std::cout<<"PROFILE_COMPLETE "<<Names[profile]<<" seed="<<ActiveSeed<<'\n';return;
  }
  std::ofstream events(path),checks(path+".checks.csv"),bits(path+".bitwise.csv"),meta(path+".meta.txt"),windows(path+".windows.csv"),inputs(path+".inputs.csv"),sampleFile(path+".cpu-indices.csv");
  if(!events||!checks||!bits||!meta||!windows||!inputs||!sampleFile)throw std::runtime_error("Cannot open archive outputs");
  events<<std::setprecision(17);checks<<std::setprecision(17);meta<<std::setprecision(17);
  events<<"m,n,k,implementation,round,order,iteration,milliseconds,tflops,batch,seed,case_index,cuda_launch_status,cuda_event_sync_status,group\n";
  checks<<"m,n,k,implementation,round,reference,count,max_abs,max_rel,relative_l2,nonfinite,abs_001_fail,simt_allclose_fail,native_eps,native_allclose_fail,batch,seed,case_index,group\n";
  bits<<"implementation,round,count,bitwise_mismatches,batch,seed,case_index,group,reference,required\n";
  windows<<"implementation,round,start_utc_ms,end_utc_ms,batch,seed,case_index,group\n";
  inputs<<"batch,seed,case_index,m,n,k,representation,matrix,element_count,bytes,fnv1a64,stage,group\n";
  sampleFile<<"sample_index,linear_index,row,col\n";
  std::array<std::vector<float>,2> aa={d.hostA,copyFloat(d.ta,d.ac)},bb={d.hostB,copyFloat(d.tb,d.bc)};
  auto inputHashes=[&](const char* stage){
    const std::array<std::vector<float>,4> copied={copyFloat(d.a,d.ac),copyFloat(d.b,d.bc),copyFloat(d.ta,d.ac),copyFloat(d.tb,d.bc)};
    for(int p=0;p<2;++p)for(int matrix=0;matrix<2;++matrix){const auto& v=copied[p*2+matrix];
      inputs<<ActiveBatch<<','<<ActiveSeed<<','<<ActiveCase<<','<<m<<','<<n<<','<<k<<','<<(p?"tf32":"original_fp32")<<','<<(matrix?"B":"A")<<','<<v.size()<<','<<v.size()*4<<','<<hash(v.data(),v.size()*4)<<','<<stage<<','<<plan.group<<'\n';
      const auto& expected=matrix?bb[p]:aa[p];if(v.size()!=expected.size()||std::memcmp(v.data(),expected.data(),v.size()*4))throw std::runtime_error("Input bytes changed");
    }inputs.flush();
  };inputHashes("before");
  meta<<"group="<<plan.group<<"\nbatch="<<ActiveBatch<<"\nseed="<<ActiveSeed<<"\ncase_index="<<ActiveCase<<"\npid="<<processId()<<"\ncase_start_utc_ms="<<utcMs()
    <<"\ngpu="<<prop.name<<"\nsm_count="<<prop.multiProcessorCount<<"\ncuBLAS_version="<<version
    <<"\ntensor_compute=CUBLAS_COMPUTE_32F_FAST_TF32\nmath_mode=CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION\natomics=CUBLAS_ATOMICS_NOT_ALLOWED\nalgorithm=CUBLAS_GEMM_DEFAULT"
    <<"\nstrict_compute=CUBLAS_COMPUTE_32F_PEDANTIC\nstrict_math=CUBLAS_PEDANTIC_MATH\nalpha=1\nbeta=0\nstream=default\nshared_input_pointers_within_group=true"
    <<"\nrounds="<<plan.orders.size()<<"\nrepeats="<<repeats<<"\nwarmup="<<warm
    <<"\ninput_tf32_preparation_ms="<<prepMs<<"\nreset=zero C0 D2D before every invocation on same stream outside events"
    <<"\nconversion=GPU __float_to_tf32 outside GEMM events; no FP16 side product in this new harness"
    <<"\nhash=FNV1a64 over actual little-endian float32 bytes; non-cryptographic"
    <<"\nallclose=abs(got-ref)<=eps+eps*abs(got); TF32 eps=0.1; SIMT eps=0.001\nrelative_error=abs(got-ref)/max(abs(ref),1e-12)"
    <<"\ntf32_gate=represented-input native allclose; original FP32 failures are recorded\nfp32_gate=abs<=0.01 AND SIMT allclose against references and CPU samples"
    <<"\nbitwise_gate=author and BN64 against author only; other implementations recorded, not required"
    <<"\nevent_boundary=public launch API only; includes GPU-visible host enqueue gaps; excludes workspace and initialize"
    <<"\ncutlass_tf32_128x128="<<cutlass_compare::Tf32_128x128::description()
    <<"\ncutlass_tf32_128x64="<<cutlass_compare::Tf32_128x64::description()
    <<"\ncutlass_simt_256x128="<<cutlass_compare::Simt_256x128::description()
    <<"\ncutlass_threads_128="<<cutlass_compare::Tf32_128x128::threads()
    <<"\ncutlass_threads_64="<<cutlass_compare::Tf32_128x64::threads()
    <<"\ncutlass_threads_simt="<<cutlass_compare::Simt_256x128::threads()
    <<"\ncutlass_shared_128="<<cutlass_compare::Tf32_128x128::shared_storage_bytes()
    <<"\ncutlass_shared_64="<<cutlass_compare::Tf32_128x64::shared_storage_bytes()
    <<"\ncutlass_shared_simt="<<cutlass_compare::Simt_256x128::shared_storage_bytes()
    <<"\ncutlass_workspace_128="<<d.cut128.workspace_size()<<"\ncutlass_workspace_64="<<d.cut64.workspace_size()<<"\ncutlass_workspace_simt="<<d.cutSimt.workspace_size()<<'\n';
  std::array<std::vector<float>,2> references;
  for(int p=0;p<2;++p){d.reset();blas(d.strict.h,m,n,k,p?d.ta.f():d.a.f(),p?d.tb.f():d.b.f(),d.c.f(),CUBLAS_COMPUTE_32F_PEDANTIC);references[p]=d.output();}
  d.reset();d.launch(tf?TFBlas:Strict);auto blasReference=d.output();
  d.reset();d.launch(tf?Author:K30);auto baseline=d.output();
  std::vector<size_t> indices;
  if(m==128&&n==1024){indices.resize(d.cc);std::iota(indices.begin(),indices.end(),size_t(0));}
  else{indices={0,size_t(n-1),size_t(m-1)*n,d.cc-1};for(size_t i=0;indices.size()<256;++i){size_t index=(SampleSeed+i*104729U)%d.cc;if(std::find(indices.begin(),indices.end(),index)==indices.end())indices.push_back(index);}}
  std::vector<double> cpu;
  for(size_t s=0;s<indices.size();++s){size_t index=indices[s],row=index/n,col=index%n;double sum=0;sampleFile<<s<<','<<index<<','<<row<<','<<col<<'\n';
    for(int t=0;t<k;++t)sum+=double(aa[tf?1:0][row*k+t])*double(bb[tf?1:0][size_t(t)*n+col]);cpu.push_back(sum);
  }sampleFile.flush();meta<<"cpu_double_positions="<<indices.size()<<'/'<<d.cc<<"\ncpu_index_seed="<<SampleSeed<<'\n';meta.flush();
  auto validate=[&](int impl,int round){
    auto actual=d.output();Error original,represented,same,cpuError,base;double eps=tf?.1:.001;size_t mismatch=0;
    for(size_t i=0;i<d.cc;++i){original.add(actual[i],references[0][i],eps);represented.add(actual[i],references[tf?1:0][i],eps);same.add(actual[i],blasReference[i],eps);base.add(actual[i],baseline[i],eps);mismatch+=std::memcmp(&actual[i],&baseline[i],4)!=0;}
    for(size_t i=0;i<indices.size();++i)cpuError.add(actual[indices[i]],cpu[i],eps);
    emit(checks,m,n,k,Names[impl],round,"original_fp32",original,eps,plan.group);
    emit(checks,m,n,k,Names[impl],round,"represented_input_strict_reference",represented,eps,plan.group);
    emit(checks,m,n,k,Names[impl],round,"same_precision_cublas",same,eps,plan.group);
    emit(checks,m,n,k,Names[impl],round,"represented_input_cpu_double",cpuError,eps,plan.group);
    emit(checks,m,n,k,Names[impl],round,tf?"vitamin_tf32_baseline":"k30_fp32_baseline",base,eps,plan.group);checks.flush();
    bool requireBits=tf&&(impl==Author||impl==BN64);
    bits<<Names[impl]<<','<<round<<','<<d.cc<<','<<mismatch<<','<<ActiveBatch<<','<<ActiveSeed<<','<<ActiveCase<<','<<plan.group<<','<<(tf?"vitamin_tf32_baseline":"k30_fp32_baseline")<<','<<(requireBits?1:0)<<'\n';bits.flush();
    bool bad=original.nonfinite||represented.nonfinite||same.nonfinite||cpuError.nonfinite||base.nonfinite||represented.native||same.native||cpuError.native||base.native||(requireBits&&mismatch);
    if(!tf)bad=bad||original.abs001||represented.abs001||same.abs001||cpuError.abs001||base.abs001||original.simt||represented.simt||same.simt||cpuError.simt||base.simt;
    std::cout<<"group="<<plan.group<<" batch="<<ActiveBatch<<" seed="<<ActiveSeed<<" n="<<n<<' '<<Names[impl]<<" round="<<round<<" dtype_gate="<<(bad?"FAIL":"PASS")<<" original_abs001_fail="<<original.abs001<<" bitwise="<<mismatch<<'\n';
    if(bad)throw std::runtime_error("Correctness gate failed; retain partial outputs; no subsequent rounds");
  };
  for(int impl:impls){d.reset();d.launch(impl);validate(impl,-1);}
  for(size_t r=0;r<plan.orders.size();++r)for(size_t order=0;order<impls.size();++order){
    int impl=plan.orders[r][order];long long begin=utcMs();for(int w=0;w<warm;++w){d.reset();d.launch(impl);}ck(cudaDeviceSynchronize());
    for(int i=0;i<repeats;++i){d.reset();ck(cudaEventRecord(start.e));d.launch(impl);ck(cudaEventRecord(stop.e));cudaError_t sync=cudaEventSynchronize(stop.e);ck(sync);float ms=0;ck(cudaEventElapsedTime(&ms,start.e,stop.e));
      if(!(ms>0)||!std::isfinite(ms))throw std::runtime_error("Invalid CUDA event time");
      events<<m<<','<<n<<','<<k<<','<<Names[impl]<<','<<r<<','<<order<<','<<i<<','<<ms<<','<<2.*m*n*k/(ms*1e9)<<','<<ActiveBatch<<','<<ActiveSeed<<','<<ActiveCase<<",0,"<<int(sync)<<','<<plan.group<<'\n';
    }events.flush();validate(impl,int(r));windows<<Names[impl]<<','<<r<<','<<begin<<','<<utcMs()<<','<<ActiveBatch<<','<<ActiveSeed<<','<<ActiveCase<<','<<plan.group<<'\n';windows.flush();
  }inputHashes("after");meta<<"case_end_utc_ms="<<utcMs()<<"\ncumulative_cuda_calls_checked_success="<<cudaChecks<<"\ncumulative_cublas_calls_checked_success="<<blasChecks<<"\nCASE_COMPLETE=true\n";
}
std::vector<std::string> split(const std::string& line){std::vector<std::string> r;std::istringstream in(line);std::string f;while(std::getline(in,f,',')){if(!f.empty()&&f.back()=='\r')f.pop_back();r.push_back(f);}return r;}
std::vector<CasePlan> loadPlan(const std::string& file,int batch,const std::string& group){
  const auto& expected=implementations(group);std::ifstream input(file);if(!input)throw std::runtime_error("Cannot read plan");std::string line;std::getline(input,line);
  if(line!="batch,case_index,m,n,k,seed,round,order,implementation"&&line!="batch,case_index,m,n,k,seed,round,order,implementation\r")throw std::runtime_error("Unexpected plan schema");
  std::vector<CasePlan> cases;
  while(std::getline(input,line)){auto f=split(line);if(f.size()!=9)throw std::runtime_error("Malformed plan row");if(std::stoi(f[0])!=batch)continue;
    int index=std::stoi(f[1]),round=std::stoi(f[6]),order=std::stoi(f[7]);if(index<0||index>=12||round<0||round>=7||order<0||order>=int(expected.size()))throw std::runtime_error("Invalid plan indices");
    if(index==int(cases.size())){CasePlan c;c.group=group;c.batch=batch;c.index=index;c.m=std::stoi(f[2]);c.n=std::stoi(f[3]);c.k=std::stoi(f[4]);c.seed=uint32_t(std::stoul(f[5]));c.orders.resize(7,Order(expected.size(),-1));cases.push_back(c);}
    if(index!=int(cases.size())-1)throw std::runtime_error("Plan cases must be contiguous");auto& c=cases.back();
    if(c.m!=std::stoi(f[2])||c.n!=std::stoi(f[3])||c.k!=std::stoi(f[4])||c.seed!=uint32_t(std::stoul(f[5]))||c.orders[round][order]!=-1)throw std::runtime_error("Duplicate or inconsistent plan row");c.orders[round][order]=findImpl(f[8]);
  }
  if(cases.size()!=12)throw std::runtime_error("Expected 12 cases per batch");std::set<std::pair<int,uint32_t>> found;
  for(const auto& c:cases){
    if(c.m!=c.n||c.n!=c.k||std::set<int>{1024,2048,3072,4096}.count(c.n)!=1||std::set<uint32_t>{20261020,42,12345}.count(c.seed)!=1||!found.emplace(c.n,c.seed).second)throw std::runtime_error("Invalid/duplicate formal shape or seed");
    std::set<Order> unique;
    for(const auto& order:c.orders){if(std::set<int>(order.begin(),order.end())!=std::set<int>(expected.begin(),expected.end()))throw std::runtime_error("Invalid implementation set");unique.insert(order);}
    if(unique.size()!=7)throw std::runtime_error("Formal rounds must use seven distinct permutations");
  }return cases;
}
Order caseOrder(const std::string& group,int r){auto v=implementations(group);if(r%2)std::reverse(v.begin(),v.end());std::rotate(v.begin(),v.begin()+((r/2)%v.size()),v.end());return v;}
int main(int argc,char** argv)try{
  if(argc<2)throw std::runtime_error("Use --list; --batch group batch plan.csv outdir; --smoke outdir; --case group M N K seed rounds repeats warmup output.csv; --profile implementation N [seed]");
  std::string mode=argv[1];
  if(mode=="--list"&&argc==2){
    std::cout<<"implementation,group\n";for(int i=0;i<ImplCount;++i)std::cout<<Names[i]<<','<<(isTensor(i)?"tf32":"fp32")<<'\n';return 0;
  }
  if(mode=="--profile"&&(argc==4||argc==5)){
    int impl=findImpl(argv[2]);CasePlan c;c.group=isTensor(impl)?"tf32":"fp32";c.m=c.n=c.k=std::stoi(argv[3]);c.seed=argc==5?uint32_t(std::stoul(argv[4])):20261020;runCase(c,1,10,"",impl);return 0;
  }
  if(mode=="--case"&&argc==11){
    CasePlan c;c.group=argv[2];c.m=std::stoi(argv[3]);c.n=std::stoi(argv[4]);c.k=std::stoi(argv[5]);c.seed=uint32_t(std::stoul(argv[6]));
    int rounds=std::stoi(argv[7]),repeats=std::stoi(argv[8]),warm=std::stoi(argv[9]);if(rounds<1||rounds>100||repeats<1||repeats>10000||warm<0||warm>1000)throw std::runtime_error("Invalid counts");
    for(int r=0;r<rounds;++r)c.orders.push_back(caseOrder(c.group,r));runCase(c,repeats,warm,argv[10]);return 0;
  }
  if(mode=="--smoke"&&argc==3){
    std::filesystem::path out=argv[2];if(std::filesystem::exists(out))throw std::runtime_error("Refusing existing smoke directory");std::filesystem::create_directories(out);
    for(const std::string group:{"tf32","fp32"})for(int k:{32,64,96}){CasePlan c;c.group=group;c.m=128;c.n=1024;c.k=k;c.seed=20261020;c.orders={caseOrder(group,0)};runCase(c,1,1,(out/(group+"-small-"+std::to_string(k)+".csv")).string());}
    std::ofstream(out/"SMOKE_COMPLETE.txt")<<"2 groups x 3 legal small rectangles; CPU double all 131072 outputs per case\n";return 0;
  }
  if(mode=="--batch"&&argc==6){
    std::string group=argv[2];int batch=std::stoi(argv[3]);if(batch<1||batch>3)throw std::runtime_error("Batch must be 1..3");auto cases=loadPlan(argv[4],batch,group);std::filesystem::path out=argv[5];
    if(std::filesystem::exists(out))throw std::runtime_error("Refusing existing batch directory");std::filesystem::create_directories(out);
    size_t eventCount=12*7*30*implementations(group).size();std::ofstream process(out/"process.csv");process<<"batch,pid,start_utc_ms,end_utc_ms,cases,formal_events,status,group\n";auto begin=utcMs();
    std::cout<<"BATCH_START group="<<group<<" batch="<<batch<<" pid="<<processId()<<" utc_ms="<<begin<<std::endl;
    for(const auto& c:cases){std::cout<<"CASE_START index="<<c.index<<" n="<<c.n<<" seed="<<c.seed<<std::endl;runCase(c,30,10,(out/("timings-"+std::to_string(c.n)+"-"+std::to_string(c.seed)+".csv")).string());}
    process<<batch<<','<<processId()<<','<<begin<<','<<utcMs()<<",12,"<<eventCount<<",COMPLETE,"<<group<<'\n';process.flush();ck(cudaDeviceSynchronize());
    std::ofstream(out/"BATCH_COMPLETE.txt")<<eventCount<<" formal single-GEMM CUDA events; all precision/error standards retained\n";return 0;
  }
  throw std::runtime_error("Invalid arguments");
}catch(const std::exception& e){std::cerr<<"ERROR: "<<e.what()<<'\n';return 1;}
