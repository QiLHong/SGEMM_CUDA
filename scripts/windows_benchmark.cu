// 【文件导读】Windows / Linux 共用的独立 FP32 实验入口，不修改原 K10 或 K30。
// 来源：本项目 Simon Boehm K10，以及 zh-cn-learning 的 K30 / k30_audit.cu。
// 教学方法参考 vitamin-cuda zh-cn-beginner：先写数据流和瓶颈假设，再单因素验证。
// 本次假设：A shared 行跨度由 132 改成 136 个 float，可能改变转置写入的 bank 分布。
// 唯一实验参数 AS_PAD: 4 -> 8；tiling、线程分工、B 布局、同步、FP32 运算均保留。
// 不预设性能改善。原 K10 的算法、默认参数不改，也不新增或占用原项目的算法编号。
//
// Usage: windows_benchmark n beta rounds repeats warmup new.csv
//        [all|sync|k10|k10_pad8|k30_v1|k30_v2|strict]
// all 保留第一阶段 PAD8 实验；sync 对照 K10/K30 V1/K30 V2/strict，V1/V2 相邻交替。
// 附加 UTC epoch 毫秒窗口：每项预热之前到该项完整验证之后，用于关联外部周期遥测。
// n=128..4096 且为128倍数，beta=0/1/3；正式测量 rounds>=5，检查可用1。
// 固定 alpha=0.5、seed=20261020；同进程所有实现使用同一份 A/B/C0。
// 每次 launch 前恢复 C0，拷贝在 events 外；CSV 保留每个 event 样本，供按轮汇总。
// 每轮预热后计时；每轮最后的输出做完整验算，验算和 D2H 不计入性能时间。
// 所有 stream 操作都在默认流；严格 cuBLAS handle 显式绑定同一默认流。
// 与历史 audit 的不同：自定义确定性 PRNG，跨 Windows/Linux 可生成同样的输入。

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>
#include "../src/kernels/30_kernel_mini_simt.cuh"
#include "../src/kernels/10_kernel_warptiling.cuh"

namespace {
constexpr std::uint32_t Seed = 20261020U;
constexpr float Alpha = 0.5f;
constexpr double Tolerance = 0.01; // 沿用项目绝对误差标准，不以相对误差替代它。
using K30Shape = k30::Shape<128,128,16,64,64,4,4,128>;
constexpr int K30SharedV1 = (k30::SharedLayoutA<K30Shape>::Size +
                            k30::SharedLayoutB<K30Shape>::Size) * sizeof(float);
constexpr int K30Shared = K30SharedV1 * 2;

void cudaCheck(cudaError_t status, const char* operation) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}
void blasCheck(cublasStatus_t status, const char* operation) {
  if (status != CUBLAS_STATUS_SUCCESS)
    throw std::runtime_error(std::string(operation) + ": status=" + std::to_string(int(status)));
}
#define CUDA_OK(x) cudaCheck((x), #x)
#define BLAS_OK(x) blasCheck((x), #x)

// uint32 无符号移位/溢出规则固定；避免不同 CRT 的 rand() 实现产生不同输入。
std::uint32_t nextRandom(std::uint32_t& state) {
  state ^= state << 13;
  state ^= state >> 17;
  state ^= state << 5;
  return state;
}
void initialize(std::vector<float>& values, std::uint32_t stream) {
  std::uint32_t state = Seed ^ (0x9e3779b9U * (stream + 1U));
  for (float& v : values) {
    const auto integral = nextRandom(state) % 5U;
    const auto fractional = nextRandom(state) % 5U;
    const bool negative = (nextRandom(state) & 1U) != 0;
    const float magnitude = float(integral) + 0.01f * float(fractional);
    v = negative ? -magnitude : magnitude;
  }
}
std::uint64_t hashInput(const std::vector<float>& values) {
  // 对 float 的32位表示计算 FNV-1a；打印便于复跑核对，不是安全哈希。
  std::uint64_t hash = 14695981039346656037ULL;
  for (float value : values) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    for (unsigned shift = 0; shift < 32; shift += 8) {
      hash ^= (bits >> shift) & 255U;
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
      ++nonfinite;
      ++failures;
      return;
    }
    const double error = std::abs(actual - expected);
    maxAbs = std::max(maxAbs, error);
    // 相对误差只报告；参考接近0时可能很大，不用于改变通过标准。
    maxRel = std::max(maxRel, error / std::max(std::abs(expected), 1.e-12));
    failures += error > Tolerance;
  }
};
Errors compare(const std::vector<float>& actual, const std::vector<float>& expected) {
  Errors errors;
  for (size_t i = 0; i < actual.size(); ++i) errors.observe(actual[i], expected[i]);
  return errors;
}
std::vector<size_t> cpuIndices(int n) {
  const size_t count = size_t(n) * n;
  std::vector<size_t> indices;
  if (n == 128) {
    indices.resize(count);
    std::iota(indices.begin(), indices.end(), size_t(0));
  } else {
    // 四角加252个固定、互异的样本，防止只用 cuBLAS 自己对照自己的循环论证。
    indices = {0, size_t(n - 1), size_t(n - 1) * n, count - 1};
    for (size_t i = 0; indices.size() < 256; ++i) {
      const size_t index = (size_t(Seed) + i * 104729U) % count;
      if (std::find(indices.begin(), indices.end(), index) == indices.end())
        indices.push_back(index);
    }
  }
  return indices;
}
Errors checkCpuReference(int n, float beta, const std::vector<float>& a,
                         const std::vector<float>& b, const std::vector<float>& initial,
                         const std::vector<float>& reference,
                         const std::vector<size_t>& indices) {
  Errors errors;
  for (size_t index : indices) {
    const size_t row = index / n, col = index % n;
    double sum = 0;
    for (int k = 0; k < n; ++k) sum += double(a[row * n + k]) * double(b[size_t(k) * n + col]);
    const double expected = double(Alpha) * sum + double(beta) * initial[index];
    errors.observe(reference[index], expected);
  }
  return errors;
}

enum class Impl { K10, Pad8, K30, Strict, K30V1 };
const char* name(Impl impl) {
  switch (impl) {
    case Impl::K10: return "k10";
    case Impl::Pad8: return "k10_pad8";
    case Impl::K30: return "k30_v2";
    case Impl::K30V1: return "k30_v1";
    default: return "strict";
  }
}
Impl parseImpl(const std::string& value) {
  for (Impl impl : {Impl::K10, Impl::Pad8, Impl::K30, Impl::Strict, Impl::K30V1})
    if (value == name(impl)) return impl;
  throw std::runtime_error("Unknown implementation: " + value);
}
struct DeviceBuffer {
  float* ptr = nullptr;
  explicit DeviceBuffer(size_t bytes) { CUDA_OK(cudaMalloc(&ptr, bytes)); }
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
  BlasHandle() {
    BLAS_OK(cublasCreate(&handle));
    BLAS_OK(cublasSetStream(handle, nullptr));
    BLAS_OK(cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH));
    BLAS_OK(cublasSetAtomicsMode(handle, CUBLAS_ATOMICS_NOT_ALLOWED));
  }
  ~BlasHandle() { if (handle) cublasDestroy(handle); }
};
void launch(Impl impl, int n, float beta, const float* a, const float* b,
            float* c, cublasHandle_t handle) {
  switch (impl) {
    case Impl::K10:
      sgemmWarptiling<128,128,16,64,64,4,8,4,128,4>
          <<<dim3(n/128,n/128),128>>>(n,n,n,Alpha,a,b,beta,c);
      break;
    case Impl::Pad8:
      // 实验版本唯一源码差异是末尾 AS_PAD 参数4 -> 8。
      sgemmWarptiling<128,128,16,64,64,4,8,4,128,8>
          <<<dim3(n/128,n/128),128>>>(n,n,n,Alpha,a,b,beta,c);
      break;
    case Impl::K30:
      sgemmK30<K30Shape,2><<<dim3(n/128,n/128),128,K30Shared>>>
          (n,n,n,Alpha,a,b,beta,c);
      break;
    case Impl::K30V1:
      // 与 V2 共用原有 Shape/fragment，只选择现有的一槽同步协议。
      sgemmK30<K30Shape,1><<<dim3(n/128,n/128),128,K30SharedV1>>>
          (n,n,n,Alpha,a,b,beta,c);
      break;
    case Impl::Strict:
      // 行主序 C=AB 等价于列主序 C^T=B^T A^T，因此交换输入指针。
      // 两个 PEDANTIC 开关共同明确严格 FP32；没有 TF32 / Tensor Core 比较混入。
      BLAS_OK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, n,n,n,
          &Alpha,b,CUDA_R_32F,n,a,CUDA_R_32F,n,&beta,c,CUDA_R_32F,n,
          CUBLAS_COMPUTE_32F_PEDANTIC,CUBLAS_GEMM_DEFAULT));
      break;
  }
  CUDA_OK(cudaGetLastError());
}
double median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  const size_t mid = values.size()/2;
  return values.size()%2 ? values[mid] : (values[mid-1]+values[mid])/2;
}
long parseInteger(const char* input) {
  std::string value(input);
  size_t end = 0;
  const long result = std::stol(value, &end);
  if (end != value.size()) throw std::runtime_error("Invalid integer: " + value);
  return result;
}

long long unixMilliseconds() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::system_clock::now().time_since_epoch()).count();
}
} // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 7 && argc != 8) {
      std::fprintf(stderr, "Usage: %s n beta rounds repeats warmup new.csv "
                          "[all|sync|k10|k10_pad8|k30_v1|k30_v2|strict]\n", argv[0]);
      return 2;
    }
    const long nInput = parseInteger(argv[1]), betaInput = parseInteger(argv[2]);
    const long roundsInput = parseInteger(argv[3]), repeatsInput = parseInteger(argv[4]);
    const long warmupInput = parseInteger(argv[5]);
    if (nInput < 128 || nInput > 4096 || nInput % 128 ||
        !(betaInput == 0 || betaInput == 1 || betaInput == 3) ||
        roundsInput < 1 || roundsInput > 1000 || repeatsInput < 1 ||
        repeatsInput > 10000 || warmupInput < 0 || warmupInput > 10000)
      throw std::runtime_error("Require n in128..4096 divisible by128, beta0/1/3, "
                               "rounds1..1000, repeats1..10000, warmup0..10000.");
    const int n = int(nInput), rounds = int(roundsInput), repeats = int(repeatsInput);
    const int warmup = int(warmupInput);
    const float beta = float(betaInput);
    const std::string selection = argc >= 8 ? argv[7] : "all";
    const bool all = selection == "all";
    const bool sync = selection == "sync";
    const Impl selected = (all || sync) ? Impl::K10 : parseImpl(selection);
    const std::vector<Impl> implementations = all
        ? std::vector<Impl>{Impl::K10,Impl::Pad8,Impl::K30,Impl::Strict}
        : sync ? std::vector<Impl>{Impl::K10,Impl::K30V1,Impl::K30,Impl::Strict}
               : std::vector<Impl>{selected};
    // 不覆盖旧结果；脚本应为每次测量创建新目录。
    if (std::ifstream(argv[6]).good())
      throw std::runtime_error("Output already exists; choose a new CSV path.");
    std::ofstream csv(argv[6], std::ios::out);
    if (!csv) throw std::runtime_error("Cannot create output CSV.");
    csv << std::setprecision(17);
    csv << "n,alpha,beta,seed,implementation,round,order,iteration,event_ms,gflops,"
           "round_mean_ms,round_max_abs_error,round_max_rel_error,round_failure_count,"
           "round_nonfinite_count,reference_nonfinite_count,cpu_reference_count,"
           "cpu_reference_max_abs_error,cpu_reference_max_rel_error,"
           "cpu_reference_failure_count,cpu_reference_nonfinite_count,"
           "a_hash,b_hash,c0_hash,status,round_start_unix_ms,round_end_unix_ms\n";

    CUDA_OK(cudaSetDevice(0));
    cudaDeviceProp device{};
    CUDA_OK(cudaGetDeviceProperties(&device,0));
    int runtimeVersion = 0, driverVersion = 0;
    CUDA_OK(cudaRuntimeGetVersion(&runtimeVersion));
    CUDA_OK(cudaDriverGetVersion(&driverVersion));
    std::printf("DEVICE,name=%s,compute=%d.%d,SMs=%d,global_bytes=%llu,"
                "runtime=%d,driver_api=%d\n", device.name,device.major,device.minor,
                device.multiProcessorCount,(unsigned long long)device.totalGlobalMem,
                runtimeVersion,driverVersion);
    if (device.major < 8)
      throw std::runtime_error("This benchmark targets the existing SM80+ teaching kernels.");
    CUDA_OK(cudaFuncSetAttribute(sgemmK30<K30Shape,2>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,K30Shared));
    if (sync || selected == Impl::K30V1)
      CUDA_OK(cudaFuncSetAttribute(sgemmK30<K30Shape,1>,
          cudaFuncAttributeMaxDynamicSharedMemorySize,K30SharedV1));
    std::printf("CONFIG,n=%d,alpha=%g,beta=%g,seed=%u,rounds=%d,repeats=%d,warmup=%d,"
                "abs_tolerance=%.2f,precision=strict_fp32,selection=%s\n",
                n,Alpha,beta,Seed,rounds,repeats,warmup,Tolerance,selection.c_str());
    std::printf("TIMING,C0_D2D_reset_excluded,events_default_stream,"
                "allocation_H2D_D2H_validation_warmup_excluded,"
                "final_output_validated_per_round\n");
    std::printf("ROUND_WINDOW,utc_epoch_ms,boundaries=before_warmup/after_validation\n");

    const size_t count = size_t(n) * n, bytes = count * sizeof(float);
    std::vector<float> a(count),b(count),initial(count),expected(count),actual(count);
    initialize(a,0); initialize(b,1); initialize(initial,2);
    const auto aHash = hashInput(a), bHash = hashInput(b), cHash = hashInput(initial);
    std::printf("INPUT_HASH,a=%llu,b=%llu,c0=%llu\n",(unsigned long long)aHash,
                (unsigned long long)bHash,(unsigned long long)cHash);
    DeviceBuffer da(bytes),db(bytes),dc(bytes),di(bytes),dr(bytes);
    CUDA_OK(cudaMemcpy(da.ptr,a.data(),bytes,cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(db.ptr,b.data(),bytes,cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(di.ptr,initial.data(),bytes,cudaMemcpyHostToDevice));
    BlasHandle blas;
    int blasVersion = 0;
    BLAS_OK(cublasGetVersion(blas.handle,&blasVersion));
    std::printf("CUBLAS,version=%d,math=PEDANTIC,compute=32F_PEDANTIC,atomics=NOT_ALLOWED\n",
                blasVersion);
    CUDA_OK(cudaMemcpy(dr.ptr,di.ptr,bytes,cudaMemcpyDeviceToDevice));
    launch(Impl::Strict,n,beta,da.ptr,db.ptr,dr.ptr,blas.handle);
    CUDA_OK(cudaDeviceSynchronize());
    CUDA_OK(cudaMemcpy(expected.data(),dr.ptr,bytes,cudaMemcpyDeviceToHost));
    size_t referenceNonfinite = 0;
    for (float value : expected) referenceNonfinite += !std::isfinite(value);
    if (referenceNonfinite)
      throw std::runtime_error("Strict reference contains NaN/Inf: " +
                               std::to_string(referenceNonfinite));
    const auto indices = cpuIndices(n);
    const Errors cpuErrors = checkCpuReference(n,beta,a,b,initial,expected,indices);
    std::printf("CPU_REFERENCE,count=%zu,max_abs=%.9g,max_rel=%.9g,"
                "failures=%zu,nonfinite=%zu\n",indices.size(),cpuErrors.maxAbs,
                cpuErrors.maxRel,cpuErrors.failures,cpuErrors.nonfinite);
    if (cpuErrors.failures)
      throw std::runtime_error("Strict FP32 reference failed CPU-double check at abs0.01.");

    auto reset = [&]() {
      // D2D、events 和 kernels 都按同一默认流排序；复位结束后才会记录 start。
      CUDA_OK(cudaMemcpyAsync(dc.ptr,di.ptr,bytes,cudaMemcpyDeviceToDevice,nullptr));
    };
    auto run = [&](Impl impl) {
      launch(impl,n,beta,da.ptr,db.ptr,dc.ptr,blas.handle);
    };
    auto validate = [&]() {
      CUDA_OK(cudaMemcpy(actual.data(),dc.ptr,bytes,cudaMemcpyDeviceToHost));
      return compare(actual,expected);
    };
    for (Impl impl : implementations) {
      reset(); run(impl);
      CUDA_OK(cudaDeviceSynchronize());
      const Errors errors = validate();
      std::printf("CHECK,%s,n=%d,beta=%g,max_abs=%.9g,max_rel=%.9g,"
                  "failures=%zu,nonfinite=%zu,%s\n",name(impl),n,beta,errors.maxAbs,
                  errors.maxRel,errors.failures,errors.nonfinite,
                  errors.failures ? "FAIL" : "PASS");
      if (errors.failures) throw std::runtime_error(std::string(name(impl)) + " preflight failed.");
    }

    // 每4轮覆盖 AB/CD、BA/DC、CD/AB、DC/BA：K10与pad8总是相邻，先后也互换。
    const std::array<std::array<Impl,4>,4> orders = {{
      {{Impl::K10,Impl::Pad8,Impl::K30,Impl::Strict}},
      {{Impl::Pad8,Impl::K10,Impl::Strict,Impl::K30}},
      {{Impl::K30,Impl::Strict,Impl::K10,Impl::Pad8}},
      {{Impl::Strict,Impl::K30,Impl::Pad8,Impl::K10}}
    }};
    const std::array<std::array<Impl,4>,4> syncOrders = {{
      {{Impl::K10,Impl::K30V1,Impl::K30,Impl::Strict}},
      {{Impl::Strict,Impl::K30,Impl::K30V1,Impl::K10}},
      {{Impl::K30V1,Impl::K30,Impl::Strict,Impl::K10}},
      {{Impl::K10,Impl::Strict,Impl::K30,Impl::K30V1}}
    }};
    Events events;
    std::array<std::vector<double>,5> roundMeans;
    for (int round = 0; round < rounds; ++round) {
      std::vector<Impl> order = all
          ? std::vector<Impl>(orders[round%4].begin(),orders[round%4].end())
          : sync ? std::vector<Impl>(syncOrders[round%4].begin(),syncOrders[round%4].end())
                 : implementations;
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
          if (!std::isfinite(ms) || ms <= 0)
            throw std::runtime_error("Invalid CUDA event duration.");
          timings.push_back(ms);
        }
        // 回读计时轮的最后一个结果。其余样本仅计时；不把它们误称为逐次验算。
        const Errors errors = validate();
        const long long roundEnd = unixMilliseconds();
        const double mean = std::accumulate(timings.begin(),timings.end(),0.0)/timings.size();
        const double gflops = 2.0*n*n*n/(mean*1.e6);
        roundMeans[size_t(impl)].push_back(mean);
        for (int iteration = 0; iteration < repeats; ++iteration) {
          const double ms = timings[iteration];
          csv << n << ',' << Alpha << ',' << beta << ',' << Seed << ',' << name(impl)
              << ',' << round+1 << ',' << slot+1 << ',' << iteration+1 << ',' << ms
              << ',' << 2.0*n*n*n/(ms*1.e6) << ',' << mean
              << ',' << errors.maxAbs << ',' << errors.maxRel << ',' << errors.failures
              << ',' << errors.nonfinite << ',' << referenceNonfinite << ',' << indices.size()
              << ',' << cpuErrors.maxAbs << ',' << cpuErrors.maxRel << ',' << cpuErrors.failures
              << ',' << cpuErrors.nonfinite << ',' << aHash << ',' << bHash << ',' << cHash
              << ',' << (errors.failures ? "FAIL" : "PASS")
              << ',' << roundStart << ',' << roundEnd << '\n';
        }
        csv.flush();
        if (!csv) throw std::runtime_error("Failed writing CSV.");
        std::printf("ROUND,%s,n=%d,beta=%g,round=%d,order=%zu,mean_ms=%.9f,"
                    "median_sample_ms=%.9f,gflops=%.3f,max_abs=%.9g,"
                    "failures=%zu,nonfinite=%zu,%s\n",
                    name(impl),n,beta,round+1,slot+1,mean,median(timings),gflops,
                    errors.maxAbs,errors.failures,errors.nonfinite,
                    errors.failures ? "FAIL" : "PASS");
        std::fflush(stdout);
        if (errors.failures)
          throw std::runtime_error(std::string(name(impl)) + " timed-round output failed.");
      }
    }
    for (Impl impl : implementations) {
      const auto& means = roundMeans[size_t(impl)];
      const double med = median(means);
      const auto range = std::minmax_element(means.begin(),means.end());
      std::printf("SUMMARY,%s,n=%d,beta=%g,rounds=%d,median_round_ms=%.9f,"
                  "min_round_ms=%.9f,max_round_ms=%.9f,range_over_median_pct=%.4f,"
                  "median_gflops=%.3f,PASS\n",name(impl),n,beta,rounds,med,
                  *range.first,*range.second,100.0*(*range.second-*range.first)/med,
                  2.0*n*n*n/(med*1.e6));
    }
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "FAIL: %s\n", error.what());
    return 1;
  }
}
