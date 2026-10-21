// 独立 DBF 寄存器上限实验。仅实例化原 vitamin DBF；不改任何 device body。
// 复用 windows_vitamin_benchmark.cu 的输入、event、全量/CPU误差协议。
// 唯一 device 编译差异是 --maxrregcount=128；默认编译不设上限。
// DBF_MAX_REGISTERS 宏只为 host 日志/CSV命名；不参与 kernel 模板或函数体。
// CMake 子工程 experiments/vitamin-dbf-register 不包含其他旧 target。
// CLI: program n rounds repeats warmup new.csv [all|sgemm_at_bcf_swizzling_dbf_rw]
// Run-DbfRegister.ps1 每次调用1轮，按外层pair逐轮反转两个binary先后。
// 输入 alpha1 beta0 seed20261020；每次C0恢复在同流start event之前。
// 原allclose规则单独报告，abs<=.01/非有限/CPU错误导致非零退出。
// CPU128全量、其余256固定点；每轮只校验最后一次输出。
// 同源代码不意味着相同机器码；ptxas/cuobjdump/NCU需分别查实际资源与spill。
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>
#include "../src/kernels/vendor/vitamin_sgemm.cuh"
#ifndef DBF_MAX_REGISTERS
#error "Define DBF_MAX_REGISTERS as 0 (default) or 128."
#endif
#if DBF_MAX_REGISTERS != 0 && DBF_MAX_REGISTERS != 128
#error "DBF_MAX_REGISTERS must be 0 or 128."
#endif

namespace {
constexpr std::uint32_t Seed = 20261020U;
constexpr float Alpha = 1.0f, Beta = 0.0f;
constexpr double Tolerance = 0.01, UpstreamAtol = 0.001, UpstreamRtol = 0.001;
constexpr const char* Configuration = DBF_MAX_REGISTERS == 128 ? "dbf_maxr128" : "dbf_default";
void cudaCheck(cudaError_t status, const char* operation) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string(operation)+": "+cudaGetErrorString(status));
}
void blasCheck(cublasStatus_t status, const char* operation) {
  if (status != CUBLAS_STATUS_SUCCESS)
    throw std::runtime_error(std::string(operation)+": status="+std::to_string(int(status)));
}
#define CUDA_OK(x) cudaCheck((x), #x)
#define BLAS_OK(x) blasCheck((x), #x)
std::uint32_t nextRandom(std::uint32_t& state) {
  state ^= state << 13; state ^= state >> 17; state ^= state << 5;
  return state;
}
void initialize(std::vector<float>& values, std::uint32_t stream) {
  // 与前两阶段同一生成器，两个配置使用相同输入，不依赖 CRT rand()。
  std::uint32_t state = Seed ^ (0x9e3779b9U*(stream+1U));
  for (float& v : values) {
    const auto integral = nextRandom(state)%5U;
    const auto fractional = nextRandom(state)%5U;
    const bool negative = (nextRandom(state)&1U) != 0;
    const float magnitude = float(integral)+0.01f*float(fractional);
    v = negative ? -magnitude : magnitude;
  }
}
std::uint64_t hashInput(const std::vector<float>& values) {
  std::uint64_t hash = 14695981039346656037ULL;
  for (float value : values) {
    std::uint32_t bits = 0;
    std::memcpy(&bits,&value,sizeof(bits));
    for (unsigned shift = 0; shift < 32; shift += 8) {
      hash ^= (bits >> shift)&255U;
      hash *= 1099511628211ULL;
    }
  }
  return hash;
}
struct Errors {
  double maxAbs = 0, maxRel = 0;
  size_t failures = 0, nonfinite = 0;
  void observe(double actual, double expected) {
    if (!std::isfinite(actual) || !std::isfinite(expected)) {
      ++nonfinite; ++failures; return;
    }
    const double error = std::abs(actual-expected);
    maxAbs = std::max(maxAbs,error);
    // 相对误差只报告；近零分母下可能很大，不改变绝对误差通过标准。
    maxRel = std::max(maxRel,error/std::max(std::abs(expected),1.e-12));
    failures += error > Tolerance;
  }
};
struct Validation {
  Errors full, cpu;
  size_t upstreamFailures = 0;
  bool passed() const { return !full.failures && !cpu.failures; }
};
std::vector<size_t> cpuIndices(int n) {
  const size_t count = size_t(n)*n;
  std::vector<size_t> indices;
  if (n == 128) {
    indices.resize(count);
    std::iota(indices.begin(),indices.end(),size_t(0));
  } else {
    indices = {0,size_t(n-1),size_t(n-1)*n,count-1};
    for (size_t i = 0; indices.size() < 256; ++i) {
      const size_t index = (size_t(Seed)+i*104729U)%count;
      if (std::find(indices.begin(),indices.end(),index) == indices.end())
        indices.push_back(index);
    }
  }
  return indices;
}
std::vector<double> makeCpuReference(int n, const std::vector<float>& a,
    const std::vector<float>& b, const std::vector<float>& initial,
    const std::vector<size_t>& indices) {
  std::vector<double> values;
  values.reserve(indices.size());
  for (size_t index : indices) {
    const size_t row = index/n, col = index%n;
    double sum = 0;
    for (int k = 0; k < n; ++k)
      sum += double(a[row*n+k])*double(b[size_t(k)*n+col]);
    values.push_back(double(Alpha)*sum+double(Beta)*initial[index]);
  }
  return values;
}
Errors compareCpu(const std::vector<float>& actual, const std::vector<size_t>& indices,
                  const std::vector<double>& cpuExpected) {
  Errors errors;
  for (size_t i = 0; i < indices.size(); ++i)
    errors.observe(actual[indices[i]],cpuExpected[i]);
  return errors;
}
Validation compare(const std::vector<float>& actual, const std::vector<float>& expected,
                   const std::vector<size_t>& indices,
                   const std::vector<double>& cpuExpected) {
  Validation result;
  for (size_t i = 0; i < actual.size(); ++i) {
    const double candidate = actual[i], reference = expected[i];
    result.full.observe(candidate,reference);
    // 原调用 diff_check(c_reference,c_candidate)，rtol 项乘第二参数 candidate。
    // host double 评估原数学判据，不宣称 PyTorch 的位级复刻。
    if (!std::isfinite(candidate) || !std::isfinite(reference) ||
        std::abs(reference-candidate) > UpstreamAtol+UpstreamRtol*std::abs(candidate))
      ++result.upstreamFailures;
  }
  result.cpu = compareCpu(actual,indices,cpuExpected);
  return result;
}
enum class Impl { DbfRw };
const std::array<Impl,1> AllImpls = {{Impl::DbfRw}};
const char* name(Impl) { return "sgemm_at_bcf_swizzling_dbf_rw"; }
Impl parseImpl(const std::string& value) {
  if (value == name(Impl::DbfRw) || value == std::string(name(Impl::DbfRw))+"_kernel")
    return Impl::DbfRw;
  throw std::runtime_error("Unknown implementation: "+value);
}
struct DeviceBuffer {
  float* ptr = nullptr;
  explicit DeviceBuffer(size_t bytes) { CUDA_OK(cudaMalloc(&ptr,bytes)); }
  ~DeviceBuffer() { if (ptr) cudaFree(ptr); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};
struct Events {
  cudaEvent_t start = nullptr, stop = nullptr;
  Events() { CUDA_OK(cudaEventCreate(&start)); CUDA_OK(cudaEventCreate(&stop)); }
  ~Events() { if (start) cudaEventDestroy(start); if (stop) cudaEventDestroy(stop); }
};
struct BlasHandle {
  cublasHandle_t handle = nullptr;
  explicit BlasHandle(bool strict) {
    BLAS_OK(cublasCreate(&handle));
    BLAS_OK(cublasSetStream(handle,nullptr));
    if (strict) {
      BLAS_OK(cublasSetMathMode(handle,CUBLAS_PEDANTIC_MATH));
      BLAS_OK(cublasSetAtomicsMode(handle,CUBLAS_ATOMICS_NOT_ALLOWED));
    }
    // 普通配置保留新建 handle 的默认 math/atomics，运行时打印实值。
    // 上游借用 PyTorch 当前 handle，外部状态不能在独立入口中直接复刻。
  }
  ~BlasHandle() { if (handle) cublasDestroy(handle); }
};
void gemm(cublasHandle_t handle, bool strict, int n, const float* a,
          const float* b, float* c) {
  // 行主序 C=AB 等价于列主序 C^T=B^T A^T；交换 A/B 指针。
  BLAS_OK(cublasGemmEx(handle,CUBLAS_OP_N,CUBLAS_OP_N,n,n,n,
      &Alpha,b,CUDA_R_32F,n,a,CUDA_R_32F,n,&Beta,c,CUDA_R_32F,n,
      strict ? CUBLAS_COMPUTE_32F_PEDANTIC : CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT));
}
void launch(int n, float* a, float* b, float* c) {
  // Only this upstream template is instantiated in this TU; no algorithm edits.
  vitamin_cuda::sgemm_at_bcf_swizzling_dbf_rw_kernel<128,128,16,8,8>
      <<<dim3(n/128,n/128),256>>>(a,b,c,n,n,n);
  CUDA_OK(cudaGetLastError());
}
void printBlas(const char* role, cublasHandle_t handle, bool strict) {
  int version = 0;
  cublasMath_t math{};
  cublasAtomicsMode_t atomics{};
  BLAS_OK(cublasGetVersion(handle,&version));
  BLAS_OK(cublasGetMathMode(handle,&math));
  BLAS_OK(cublasGetAtomicsMode(handle,&atomics));
  std::printf("CUBLAS,role=%s,version=%d,math_enum=%d,compute=%s,atomics_enum=%d\n",
      role,version,int(math),strict ? "32F_PEDANTIC" : "32F",int(atomics));
}
double median(std::vector<double> values) {
  std::sort(values.begin(),values.end());
  const size_t mid = values.size()/2;
  return values.size()%2 ? values[mid] : (values[mid-1]+values[mid])/2;
}
long parseInteger(const char* input) {
  std::string value(input);
  size_t end = 0;
  const long result = std::stol(value,&end);
  if (end != value.size()) throw std::runtime_error("Invalid integer: "+value);
  return result;
}
long long unixMilliseconds() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::system_clock::now().time_since_epoch()).count();
}
const char* pass(size_t failures) { return failures ? "FAIL" : "PASS"; }
} // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 6 && argc != 7) {
      std::fprintf(stderr,"Usage: %s n rounds repeats warmup new.csv "
          "[all|sgemm_at_bcf_swizzling_dbf_rw]\n",argv[0]);
      return 2;
    }
    const long nInput = parseInteger(argv[1]), roundsInput = parseInteger(argv[2]);
    const long repeatsInput = parseInteger(argv[3]), warmupInput = parseInteger(argv[4]);
    if (nInput < 128 || nInput > 4096 || nInput%128 ||
        roundsInput < 1 || roundsInput > 1000 || repeatsInput < 1 || repeatsInput > 10000 ||
        warmupInput < 0 || warmupInput > 10000)
      throw std::runtime_error("Require n in128..4096 divisible by128, "
          "rounds1..1000, repeats1..10000, warmup0..10000.");
    const int n = int(nInput), rounds = int(roundsInput), repeats = int(repeatsInput);
    const int warmup = int(warmupInput);
    const std::string selection = argc == 7 ? argv[6] : "all";
    const bool all = selection == "all";
    const auto implementations = all ? std::vector<Impl>(AllImpls.begin(),AllImpls.end())
        : std::vector<Impl>{parseImpl(selection)};
    const std::string csvPath = argv[5], checkPath = csvPath+".correctness.csv";
    if (std::ifstream(csvPath).good() || std::ifstream(checkPath).good())
      throw std::runtime_error("Output or correctness sidecar already exists; choose a new path.");
    std::ofstream csv(csvPath), checks(checkPath);
    if (!csv || !checks) throw std::runtime_error("Cannot create output CSV and correctness sidecar.");
    csv << std::setprecision(17);
    checks << std::setprecision(17);
    csv << "configuration,n,alpha,beta,seed,implementation,round,order,iteration,"
           "event_ms,gflops,round_mean_ms,round_max_abs_error,round_max_rel_error,"
           "round_failure_count,round_nonfinite_count,upstream_allclose_failure_count,"
           "upstream_allclose_status,reference_nonfinite_count,cpu_reference_count,"
           "cpu_reference_max_abs_error,cpu_reference_max_rel_error,"
           "cpu_reference_failure_count,cpu_reference_nonfinite_count,"
           "cpu_implementation_max_abs_error,cpu_implementation_max_rel_error,"
           "cpu_implementation_failure_count,cpu_implementation_nonfinite_count,"
           "a_hash,b_hash,c0_hash,status,round_start_unix_ms,round_end_unix_ms\n";
    checks << "configuration,phase,n,alpha,beta,seed,implementation,round,order,"
              "full_element_count,max_abs_error,max_rel_error,failure_count,nonfinite_count,"
              "upstream_allclose_failure_count,upstream_allclose_status,"
              "reference_nonfinite_count,cpu_reference_count,"
              "cpu_reference_max_abs_error,cpu_reference_max_rel_error,"
              "cpu_reference_failure_count,cpu_reference_nonfinite_count,"
              "cpu_implementation_max_abs_error,cpu_implementation_max_rel_error,"
              "cpu_implementation_failure_count,cpu_implementation_nonfinite_count,"
              "a_hash,b_hash,c0_hash,status\n";

    CUDA_OK(cudaSetDevice(0));
    cudaDeviceProp device{};
    CUDA_OK(cudaGetDeviceProperties(&device,0));
    int runtimeVersion = 0, driverVersion = 0;
    CUDA_OK(cudaRuntimeGetVersion(&runtimeVersion));
    CUDA_OK(cudaDriverGetVersion(&driverVersion));
    std::printf("DEVICE,name=%s,compute=%d.%d,SMs=%d,global_bytes=%llu,runtime=%d,driver_api=%d\n",
        device.name,device.major,device.minor,device.multiProcessorCount,
        (unsigned long long)device.totalGlobalMem,runtimeVersion,driverVersion);
    if (device.major < 8) throw std::runtime_error("The existing teaching kernels require SM80+.");
    cudaFuncAttributes attributes{};
    CUDA_OK(cudaFuncGetAttributes(&attributes,
        vitamin_cuda::sgemm_at_bcf_swizzling_dbf_rw_kernel<128,128,16,8,8>));
    int occupancyBlocks = 0;
    CUDA_OK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancyBlocks,
        vitamin_cuda::sgemm_at_bcf_swizzling_dbf_rw_kernel<128,128,16,8,8>,256,0));
    std::printf("KERNEL_RESOURCE,register_limit=%d,registers=%d,static_shared_bytes=%zu,"
        "local_bytes=%zu,preferred_carveout=%d,occupancy_api_max_active_blocks=%d\n",
        DBF_MAX_REGISTERS,attributes.numRegs,attributes.sharedSizeBytes,
        attributes.localSizeBytes,attributes.preferredShmemCarveout,occupancyBlocks);
    std::printf("CONFIG,configuration=%s,n=%d,alpha=%g,beta=%g,seed=%u,rounds=%d,"
        "repeats=%d,warmup=%d,abs_tolerance=%.2f,selection=%s\n",
        Configuration,n,Alpha,Beta,Seed,rounds,repeats,warmup,Tolerance,selection.c_str());
    std::printf("UPSTREAM_RULE,abs(reference-candidate)<=0.001+0.001*abs(candidate),"
        "reference=independent_pedantic_cublas,evaluation=host_double,"
        "failure_recorded_separately_not_exit_failure\n");
    std::printf("TIMING,C0_D2D_reset_excluded,events_default_stream,"
        "allocation_H2D_D2H_validation_warmup_excluded,final_output_validated_per_round\n");
    std::printf("ROUND_WINDOW,utc_epoch_ms,boundaries=before_warmup/after_validation\n");
    std::printf("ORDER,single_kernel_process,external_script_alternates_binaries_per_pair\n");

    const size_t count = size_t(n)*n, bytes = count*sizeof(float);
    std::vector<float> a(count),b(count),initial(count),expected(count),actual(count);
    initialize(a,0); initialize(b,1); initialize(initial,2);
    const auto aHash = hashInput(a), bHash = hashInput(b), cHash = hashInput(initial);
    std::printf("INPUT_HASH,a=%llu,b=%llu,c0=%llu\n",
        (unsigned long long)aHash,(unsigned long long)bHash,(unsigned long long)cHash);
    DeviceBuffer da(bytes),db(bytes),dc(bytes),di(bytes),dr(bytes);
    CUDA_OK(cudaMemcpy(da.ptr,a.data(),bytes,cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(db.ptr,b.data(),bytes,cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(di.ptr,initial.data(),bytes,cudaMemcpyHostToDevice));
    BlasHandle referenceBlas(true);
    printBlas("independent_reference",referenceBlas.handle,true);
    CUDA_OK(cudaMemcpyAsync(dr.ptr,di.ptr,bytes,cudaMemcpyDeviceToDevice,nullptr));
    gemm(referenceBlas.handle,true,n,da.ptr,db.ptr,dr.ptr);
    CUDA_OK(cudaGetLastError());
    CUDA_OK(cudaDeviceSynchronize());
    CUDA_OK(cudaMemcpy(expected.data(),dr.ptr,bytes,cudaMemcpyDeviceToHost));
    size_t referenceNonfinite = 0;
    for (float value : expected) referenceNonfinite += !std::isfinite(value);
    const auto indices = cpuIndices(n);
    const auto cpuExpected = makeCpuReference(n,a,b,initial,indices);
    const Errors cpuErrors = compareCpu(expected,indices,cpuExpected);
    std::printf("CPU_REFERENCE,count=%zu,coverage=%s,max_abs=%.9g,max_rel=%.9g,"
        "failures=%zu,nonfinite=%zu,reference_nonfinite=%zu\n",indices.size(),
        n == 128 ? "all" : "fixed_256_points",cpuErrors.maxAbs,cpuErrors.maxRel,
        cpuErrors.failures,cpuErrors.nonfinite,referenceNonfinite);

    auto writeCheck = [&](const char* phase, const char* implName, int round,
                          size_t slot, const Validation& result) {
      checks << Configuration << ',' << phase << ',' << n << ',' << Alpha << ',' << Beta
          << ',' << Seed << ',' << implName << ',' << round << ',' << slot << ',' << count
          << ',' << result.full.maxAbs << ',' << result.full.maxRel << ',' << result.full.failures
          << ',' << result.full.nonfinite << ',' << result.upstreamFailures
          << ',' << pass(result.upstreamFailures) << ',' << referenceNonfinite << ',' << indices.size()
          << ',' << cpuErrors.maxAbs << ',' << cpuErrors.maxRel << ',' << cpuErrors.failures
          << ',' << cpuErrors.nonfinite << ',' << result.cpu.maxAbs << ',' << result.cpu.maxRel
          << ',' << result.cpu.failures << ',' << result.cpu.nonfinite
          << ',' << aHash << ',' << bHash << ',' << cHash
          << ',' << (result.passed() ? "PASS" : "FAIL") << '\n';
      checks.flush();
      if (!checks) throw std::runtime_error("Failed writing correctness sidecar.");
    };
    const Validation referenceResult = compare(expected,expected,indices,cpuExpected);
    writeCheck("reference","independent_strict_reference",0,0,referenceResult);
    if (referenceNonfinite || cpuErrors.failures)
      throw std::runtime_error("Independent strict reference failed finite/CPU-double check.");
    auto reset = [&]() {
      CUDA_OK(cudaMemcpyAsync(dc.ptr,di.ptr,bytes,cudaMemcpyDeviceToDevice,nullptr));
    };
    auto run = [&](Impl impl) {
      launch(n,da.ptr,db.ptr,dc.ptr);
    };
    auto validate = [&]() {
      CUDA_OK(cudaMemcpy(actual.data(),dc.ptr,bytes,cudaMemcpyDeviceToHost));
      return compare(actual,expected,indices,cpuExpected);
    };
    std::array<bool,1> upstreamPassed;
    upstreamPassed.fill(true);
    for (Impl impl : implementations) {
      reset(); run(impl);
      CUDA_OK(cudaDeviceSynchronize());
      const Validation result = validate();
      writeCheck("preflight",name(impl),0,0,result);
      upstreamPassed[size_t(impl)] = result.upstreamFailures == 0;
      std::printf("CHECK,%s,n=%d,max_abs=%.9g,max_rel=%.9g,failures=%zu,nonfinite=%zu,"
          "upstream_allclose_failures=%zu,upstream_allclose=%s,"
          "cpu_count=%zu,cpu_max_abs=%.9g,cpu_failures=%zu,abs_status=%s\n",
          name(impl),n,result.full.maxAbs,result.full.maxRel,result.full.failures,
          result.full.nonfinite,result.upstreamFailures,pass(result.upstreamFailures),
          indices.size(),result.cpu.maxAbs,result.cpu.failures,result.passed() ? "PASS" : "FAIL");
      if (!result.passed()) throw std::runtime_error(std::string(name(impl))+" preflight abs/CPU failed.");
    }

    Events events;
    std::array<std::vector<double>,1> roundMeans;
    for (int round = 0; round < rounds; ++round) {
      const auto& order = implementations;
      for (size_t slot = 0; slot < order.size(); ++slot) {
        const Impl impl = order[slot];
        const long long roundStart = unixMilliseconds();
        for (int i = 0; i < warmup; ++i) { reset(); run(impl); }
        CUDA_OK(cudaDeviceSynchronize());
        std::vector<double> timings;
        timings.reserve(repeats);
        for (int iteration = 0; iteration < repeats; ++iteration) {
          reset();
          CUDA_OK(cudaEventRecord(events.start,nullptr));
          run(impl);
          CUDA_OK(cudaEventRecord(events.stop,nullptr));
          CUDA_OK(cudaEventSynchronize(events.stop));
          float ms = 0;
          CUDA_OK(cudaEventElapsedTime(&ms,events.start,events.stop));
          if (!std::isfinite(ms) || ms <= 0) throw std::runtime_error("Invalid CUDA event duration.");
          timings.push_back(ms);
        }
        const Validation result = validate();
        const long long roundEnd = unixMilliseconds();
        writeCheck("timed_round",name(impl),round+1,slot+1,result);
        upstreamPassed[size_t(impl)] = upstreamPassed[size_t(impl)] && !result.upstreamFailures;
        const double mean = std::accumulate(timings.begin(),timings.end(),0.0)/timings.size();
        roundMeans[size_t(impl)].push_back(mean);
        for (int iteration = 0; iteration < repeats; ++iteration) {
          const double ms = timings[iteration];
          csv << Configuration << ',' << n << ',' << Alpha << ',' << Beta << ',' << Seed
              << ',' << name(impl) << ',' << round+1 << ',' << slot+1 << ',' << iteration+1
              << ',' << ms << ',' << 2.0*n*n*n/(ms*1.e6) << ',' << mean
              << ',' << result.full.maxAbs << ',' << result.full.maxRel << ',' << result.full.failures
              << ',' << result.full.nonfinite << ',' << result.upstreamFailures
              << ',' << pass(result.upstreamFailures) << ',' << referenceNonfinite << ',' << indices.size()
              << ',' << cpuErrors.maxAbs << ',' << cpuErrors.maxRel << ',' << cpuErrors.failures
              << ',' << cpuErrors.nonfinite << ',' << result.cpu.maxAbs << ',' << result.cpu.maxRel
              << ',' << result.cpu.failures << ',' << result.cpu.nonfinite
              << ',' << aHash << ',' << bHash << ',' << cHash
              << ',' << (result.passed() ? "PASS" : "FAIL") << ',' << roundStart << ',' << roundEnd << '\n';
        }
        csv.flush();
        if (!csv) throw std::runtime_error("Failed writing timing CSV.");
        std::printf("ROUND,%s,n=%d,round=%d,order=%zu,mean_ms=%.9f,median_sample_ms=%.9f,"
            "gflops=%.3f,max_abs=%.9g,failures=%zu,nonfinite=%zu,"
            "upstream_allclose_failures=%zu,upstream_allclose=%s,"
            "cpu_max_abs=%.9g,cpu_failures=%zu,abs_status=%s\n",
            name(impl),n,round+1,slot+1,mean,median(timings),2.0*n*n*n/(mean*1.e6),
            result.full.maxAbs,result.full.failures,result.full.nonfinite,
            result.upstreamFailures,pass(result.upstreamFailures),
            result.cpu.maxAbs,result.cpu.failures,result.passed() ? "PASS" : "FAIL");
        std::fflush(stdout);
        if (!result.passed()) throw std::runtime_error(std::string(name(impl))+" timed output abs/CPU failed.");
      }
    }
    for (Impl impl : implementations) {
      const auto& means = roundMeans[size_t(impl)];
      const double med = median(means);
      const auto range = std::minmax_element(means.begin(),means.end());
      std::printf("SUMMARY,%s,configuration=%s,n=%d,rounds=%d,median_round_ms=%.9f,"
          "min_round_ms=%.9f,max_round_ms=%.9f,range_over_median_pct=%.4f,"
          "median_gflops=%.3f,abs_status=PASS,upstream_allclose=%s\n",
          name(impl),Configuration,n,rounds,med,*range.first,*range.second,
          100.0*(*range.second-*range.first)/med,2.0*n*n*n/(med*1.e6),
          upstreamPassed[size_t(impl)] ? "PASS" : "FAIL");
    }
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr,"FAIL: %s\n",error.what());
    return 1;
  }
}
