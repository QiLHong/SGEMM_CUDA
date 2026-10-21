#pragma once
// Host-only instrumentation. Included before legacy main; device headers untouched.
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>

namespace stage3_capture {
struct State {
  std::vector<float> events;
  std::vector<std::vector<float>> inputs, outputs;
  std::string prefix, output_order;
  int n=0, original_exit=-999;
  bool finalized=false;
  double alpha=.5, beta=3;
};
inline State& state(){static State s;return s;}
inline uint64_t fnv(const void* data,size_t bytes){
  const auto* p=static_cast<const unsigned char*>(data);
  uint64_t h=14695981039346656037ull;
  for(size_t i=0;i<bytes;++i){h^=p[i];h*=1099511628211ull;}return h;
}
struct Error {
  double max_abs=0,max_rel=0,l2_num=0,l2_den=0;
  size_t tested=0,abs001_failures=0,nan_inf=0;
  void add(double got,double ref){
    ++tested;
    if(!std::isfinite(got)||!std::isfinite(ref)){++nan_inf;++abs001_failures;return;}
    double e=std::abs(got-ref);max_abs=std::max(max_abs,e);
    max_rel=std::max(max_rel,e/std::max(std::abs(ref),1e-30));
    l2_num+=e*e;l2_den+=ref*ref;if(e>.01)++abs001_failures;
  }
  void emit(std::ofstream& f,const char* name,const char* scope){
    f<<name<<','<<scope<<','<<tested<<','<<std::setprecision(17)<<max_abs<<','<<max_rel
     <<','<<std::sqrt(l2_num/std::max(l2_den,1e-300))<<','<<abs001_failures<<','<<nan_inf
     <<','<<((abs001_failures||nan_inf)?"FAIL":"PASS")<<'\n';
  }
};
inline void finish(){
  auto& s=state();if(s.finalized||s.prefix.empty())return;s.finalized=true;
  std::ofstream e(s.prefix+".events.partial.csv");e<<"event_index,ms\n";
  for(size_t i=0;i<s.events.size();++i)e<<i<<','<<std::setprecision(9)<<s.events[i]<<'\n';
  e.close();
  std::ofstream in(s.prefix+".inputs.csv");in<<"matrix,bytes,fnv1a64,algorithm,layout,stride_floats\n";
  const char* labels[]={"A","B","C0"};
  for(size_t i=0;i<s.inputs.size();++i)in<<labels[i]<<','<<s.inputs[i].size()*4<<','<<std::hex
    <<fnv(s.inputs[i].data(),s.inputs[i].size()*4)<<std::dec<<",FNV1a64_noncryptographic,row_major,"<<s.n<<'\n';
  in.close();
  std::ofstream checks(s.prefix+".checks.csv");
  checks<<"reference,scope,elements,max_abs,max_rel_denominator_ref,relative_l2,abs001_failures,nan_inf,status\n";
  if(s.outputs.size()>=2){
    const auto& actual=s.outputs[s.output_order=="actual_first"?0:1];
    const auto& reference=s.outputs[s.output_order=="actual_first"?1:0];
    Error gpu;
    for(size_t i=0;i<std::min(actual.size(),reference.size());++i)gpu.add(actual[i],reference[i]);
    gpu.emit(checks,"PEDANTIC_FP32","all_D2H_outputs_before_timing");
    if(s.inputs.size()==3 && s.inputs[0].size()==size_t(s.n)*s.n && actual.size()==size_t(s.n)*s.n){
      std::vector<size_t> indices;
      if(s.n<=256){for(size_t i=0;i<actual.size();++i)indices.push_back(i);}
      else{
        indices={0,size_t(s.n-1),size_t(s.n)*(s.n-1),actual.size()-1};
        uint32_t rng=20261020;
        while(indices.size()<256){rng^=rng<<13;rng^=rng>>17;rng^=rng<<5;size_t i=rng%actual.size();
          if(std::find(indices.begin(),indices.end(),i)==indices.end())indices.push_back(i);}
      }
      std::ofstream positions(s.prefix+".cpu-indices.csv");positions<<"flat_index,row,col\n";
      Error cpu;
      for(size_t i:indices){size_t row=i/s.n,col=i%s.n;double sum=0;
        for(int k=0;k<s.n;++k)sum+=double(s.inputs[0][row*s.n+k])*s.inputs[1][size_t(k)*s.n+col];
        cpu.add(actual[i],s.alpha*sum+s.beta*double(s.inputs[2][i]));positions<<i<<','<<row<<','<<col<<'\n';}
      cpu.emit(checks,"CPU_double_actual_FP32_bytes",s.n<=256?"full":"256_fixed_unique_positions");
    }
  }else checks<<"not_available,legacy_reference_only_or_failed_before_D2H,0,0,0,0,0,0,NOT_RUN\n";
  checks.close();
  std::ofstream meta(s.prefix+".capture.json");
  meta<<"{\"original_exit\":"<<s.original_exit<<",\"events\":"<<s.events.size()
      <<",\"h2d_inputs_captured\":"<<s.inputs.size()<<",\"d2h_outputs_captured\":"<<s.outputs.size()
      <<",\"source_modification\":\"host_api_instrumentation_only\",\"validation_position\":\"legacy_pre_timing_D2H\"}\n";
}
inline void begin(){
  auto& s=state();const char* p=std::getenv("STAGE3_CAPTURE_PREFIX");
  if(!p){std::fprintf(stderr,"STAGE3_CAPTURE_PREFIX missing\n");std::exit(91);}s.prefix=p;
  const char* n=std::getenv("STAGE3_N");const char* beta=std::getenv("STAGE3_BETA");
  if(!n||!beta||std::atoi(n)<=0){std::fprintf(stderr,"STAGE3_N/BETA missing or invalid\n");std::exit(91);}
  s.n=std::atoi(n);s.beta=std::atof(beta);
  const char* order=std::getenv("STAGE3_OUTPUT_ORDER");s.output_order=order?order:"reference_first";
  s.events.reserve(10000);std::atexit(finish);
}
inline cudaError_t elapsed(float* ms,cudaEvent_t start,cudaEvent_t stop){
  cudaError_t rc=cudaEventElapsedTime(ms,start,stop);if(rc==cudaSuccess)state().events.push_back(*ms);return rc;
}
inline cudaError_t copy(void* dst,const void* src,size_t bytes,cudaMemcpyKind kind){
  cudaError_t rc=cudaMemcpy(dst,src,bytes,kind);if(rc!=cudaSuccess)return rc;
  auto& s=state();
  if(kind==cudaMemcpyHostToDevice&&s.inputs.size()<3&&bytes%sizeof(float)==0){
    const auto* p=static_cast<const float*>(src);s.inputs.emplace_back(p,p+bytes/sizeof(float));
  }
  if(kind==cudaMemcpyDeviceToHost&&s.outputs.size()<2&&bytes%sizeof(float)==0){
    const auto* p=static_cast<const float*>(dst);s.outputs.emplace_back(p,p+bytes/sizeof(float));
  }
  return rc;
}
} // namespace stage3_capture
