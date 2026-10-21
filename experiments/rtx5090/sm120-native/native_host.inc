// Host-only audit addition. The upstream device source remains byte-identical.
#include <chrono>
#include <string>
#include <vector>
#include <cublas_v2.h>
#include <ATen/cuda/CUDABlas.h>
#include <pybind11/stl.h>

namespace py = pybind11;
#define AUDIT_CUDA(x) do { cudaError_t audit_e = (x); TORCH_CHECK(audit_e == cudaSuccess, #x, ": ", cudaGetErrorString(audit_e)); } while(0)
#define AUDIT_BLAS(x) do { cublasStatus_t audit_e = (x); TORCH_CHECK(audit_e == CUBLAS_STATUS_SUCCESS, #x, ": cublas status ", int(audit_e)); } while(0)

static void audit_contract(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda() && c.is_cuda(), "CUDA tensors required");
    TORCH_CHECK(a.dim()==2 && b.dim()==2 && c.dim()==2, "rank2 required");
    TORCH_CHECK(a.is_contiguous() && b.is_contiguous() && c.is_contiguous(), "contiguous required");
    TORCH_CHECK(a.device()==b.device() && a.device()==c.device(), "same device required");
    TORCH_CHECK(a.scalar_type()==b.scalar_type() && a.scalar_type()==c.scalar_type(), "same dtype required");
    TORCH_CHECK(a.size(1)==b.size(0) && c.size(0)==a.size(0) && c.size(1)==b.size(1), "shape mismatch");
    int device=0; AUDIT_CUDA(cudaGetDevice(&device));
    TORCH_CHECK(device==a.get_device(), "caller must set current CUDA device to tensor device");
}

struct AuditEvents {
    cudaEvent_t start{}, stop{};
    AuditEvents() { AUDIT_CUDA(cudaEventCreate(&start)); AUDIT_CUDA(cudaEventCreate(&stop)); }
    ~AuditEvents() { if(start) cudaEventDestroy(start); if(stop) cudaEventDestroy(stop); }
};

template<class T>
static py::dict audit_measure_t(torch::Tensor a, torch::Tensor b, torch::Tensor c,
                                torch::Tensor c0, int variant, int warmup, int repeat) {
    const int M=a.size(0), K=a.size(1), N=b.size(1);
    const dim3 grid(N/128,M/128);
    const int block=256;
    cudaStream_t stream=at::cuda::getCurrentCUDAStream();
    cublasHandle_t handle=at::cuda::getCurrentCUDABlasHandle();
    cudaStream_t handle_stream{}; AUDIT_BLAS(cublasGetStream(handle,&handle_stream));
    TORCH_CHECK(handle_stream==stream,"cuBLAS and kernels must use the same stream");
    cublasPointerMode_t pointer_mode; AUDIT_BLAS(cublasGetPointerMode(handle,&pointer_mode));
    TORCH_CHECK(pointer_mode==CUBLAS_POINTER_MODE_HOST,"cuBLAS host pointer mode required");
    cublasMath_t math_mode; AUDIT_BLAS(cublasGetMathMode(handle,&math_mode));
    int blas_version=0; AUDIT_BLAS(cublasGetVersion(handle,&blas_version));
    T *ap=reinterpret_cast<T*>(a.data_ptr()),*bp=reinterpret_cast<T*>(b.data_ptr()),*cp=reinterpret_cast<T*>(c.data_ptr());
    cudaDataType_t dtype = (std::is_same<T, __half>::value) ? CUDA_R_16F : CUDA_R_16BF;
    const float alpha=1.f,beta=0.f;
    CUtensorMap ta{},tb{};
    int dynamic_shared=0;
    const void *fn=nullptr;
    double descriptor_us=0.0;
    if(variant==0) fn=reinterpret_cast<const void*>(hgemm_bcf_dbf_rw_kernel<128,128,32,T>);
    if(variant==1) fn=reinterpret_cast<const void*>(hgemm_k_stages_kernel<128,128,32,3,T>);
    if(variant==2 || variant==3) {
        auto t0=std::chrono::steady_clock::now();
        if(variant==2) {
            ta=create_tensor_map<T,128>(ap,K,M,64,128);
            tb=create_tensor_map<T,128>(bp,N,K,64,64);
            fn=reinterpret_cast<const void*>(hgemm_tma_r_k_stages_kernel<128,128,64,3,T>);
            dynamic_shared=98328;
        } else {
            ta=create_tensor_map<T,64>(ap,K,M,32,128);
            tb=create_tensor_map<T,128>(bp,N,K,64,32);
            fn=reinterpret_cast<const void*>(hgemm_tma_r_k_stages_kernel<128,128,32,3,T>);
            dynamic_shared=49176;
        }
        descriptor_us=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-t0).count();
        AUDIT_CUDA(cudaFuncSetAttribute(fn,cudaFuncAttributeMaxDynamicSharedMemorySize,dynamic_shared));
    }
    py::dict resource;
    if(fn) {
        cudaFuncAttributes attr{}; AUDIT_CUDA(cudaFuncGetAttributes(&attr,fn));
        int active=0; AUDIT_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,fn,block,dynamic_shared));
        resource["registers_per_thread"]=attr.numRegs;
        resource["static_shared_bytes"]=attr.sharedSizeBytes;
        resource["dynamic_shared_bytes"]=dynamic_shared;
        resource["local_bytes_per_thread"]=attr.localSizeBytes;
        resource["max_threads_per_block"]=attr.maxThreadsPerBlock;
        resource["ptx_version"]=attr.ptxVersion;
        resource["binary_version"]=attr.binaryVersion;
        resource["occupancy_api_active_blocks_per_sm"]=active;
    } else resource["note"]="cuBLAS runtime implementation; kernel resources not exposed by this wrapper";
    int device=0; AUDIT_CUDA(cudaGetDevice(&device));
    cudaDeviceProp prop{}; AUDIT_CUDA(cudaGetDeviceProperties(&prop,device));
    resource["SM_count"]=prop.multiProcessorCount;
    resource["device_shared_per_SM"]=prop.sharedMemPerMultiprocessor;
    resource["device_regs_per_SM"]=prop.regsPerMultiprocessor;
    resource["device_max_threads_per_SM"]=prop.maxThreadsPerMultiProcessor;
    auto launch=[&]() {
        switch(variant) {
        case 0: hgemm_bcf_dbf_rw_kernel<128,128,32,T><<<grid,block,0,stream>>>(ap,bp,cp,M,N,K); break;
        case 1: hgemm_k_stages_kernel<128,128,32,3,T><<<grid,block,0,stream>>>(ap,bp,cp,M,N,K); break;
        case 2: hgemm_tma_r_k_stages_kernel<128,128,64,3,T><<<grid,block,98328,stream>>>(ta,tb,cp,M,N,K); break;
        case 3: hgemm_tma_r_k_stages_kernel<128,128,32,3,T><<<grid,block,49176,stream>>>(ta,tb,cp,M,N,K); break;
        case 4: AUDIT_BLAS(cublasGemmEx(handle,CUBLAS_OP_N,CUBLAS_OP_N,N,M,K,&alpha,bp,dtype,N,ap,dtype,K,&beta,cp,dtype,N,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT)); break;
        }
        AUDIT_CUDA(cudaGetLastError());
    };
    const size_t bytes=c.numel()*c.element_size();
    for(int i=0;i<warmup;++i) {
        AUDIT_CUDA(cudaMemcpyAsync(c.data_ptr(),c0.data_ptr(),bytes,cudaMemcpyDeviceToDevice,stream));
        launch();
    }
    AUDIT_CUDA(cudaStreamSynchronize(stream));
    AuditEvents events;
    std::vector<float> milliseconds; milliseconds.reserve(repeat);
    for(int i=0;i<repeat;++i) {
        AUDIT_CUDA(cudaMemcpyAsync(c.data_ptr(),c0.data_ptr(),bytes,cudaMemcpyDeviceToDevice,stream));
        AUDIT_CUDA(cudaEventRecord(events.start,stream));
        launch();
        AUDIT_CUDA(cudaEventRecord(events.stop,stream));
        AUDIT_CUDA(cudaEventSynchronize(events.stop));
        float ms=0.f; AUDIT_CUDA(cudaEventElapsedTime(&ms,events.start,events.stop));
        milliseconds.push_back(ms);
    }
    py::dict result;
    result["milliseconds"]=milliseconds;
    result["descriptor_encode_cpu_us"]=descriptor_us;
    result["descriptor_encodes_per_measurement_call"]=(variant==2||variant==3)?2:0;
    result["resources"]=resource;
    result["cublas_runtime_version"]=blas_version;
    result["cublas_math_mode"]=int(math_mode);
    result["cublas_compute_type"]="CUBLAS_COMPUTE_32F (same low-precision contract as upstream)";
    return result;
}

static py::dict audit_measure(torch::Tensor a,torch::Tensor b,torch::Tensor c,torch::Tensor c0,int variant,int warmup,int repeat) {
    audit_contract(a,b,c);
    TORCH_CHECK(c0.is_cuda() && c0.is_contiguous() && c0.sizes()==c.sizes() && c0.device()==c.device() && c0.scalar_type()==c.scalar_type(),"C0 contract mismatch");
    TORCH_CHECK(c.data_ptr()!=c0.data_ptr(),"C and C0 must have distinct storage");
    TORCH_CHECK(a.size(0)%128==0 && b.size(1)%1024==0 && a.size(1)>=128 && a.size(1)%64==0,"common legal shape required");
    TORCH_CHECK(variant>=0 && variant<=4 && warmup>=0 && repeat>0,"invalid measurement args");
    TORCH_CHECK(a.scalar_type()==torch::kHalf || a.scalar_type()==torch::kBFloat16,"FP16/BF16 only");
    if(a.scalar_type()==torch::kHalf) return audit_measure_t<__half>(a,b,c,c0,variant,warmup,repeat);
    return audit_measure_t<__nv_bfloat16>(a,b,c,c0,variant,warmup,repeat);
}

static void audit_pedantic(torch::Tensor a,torch::Tensor b,torch::Tensor c) {
    audit_contract(a,b,c);
    TORCH_CHECK(a.scalar_type()==torch::kFloat32,"PEDANTIC input/output must be float32");
    cublasHandle_t handle=at::cuda::getCurrentCUDABlasHandle();
    cublasPointerMode_t pointer_mode; AUDIT_BLAS(cublasGetPointerMode(handle,&pointer_mode));
    TORCH_CHECK(pointer_mode==CUBLAS_POINTER_MODE_HOST,"host pointer mode required");
    const float alpha=1.f,beta=0.f;
    const int M=a.size(0),K=a.size(1),N=b.size(1);
    AUDIT_BLAS(cublasGemmEx(handle,CUBLAS_OP_N,CUBLAS_OP_N,N,M,K,&alpha,b.data_ptr(),CUDA_R_32F,N,a.data_ptr(),CUDA_R_32F,K,&beta,c.data_ptr(),CUDA_R_32F,N,CUBLAS_COMPUTE_32F_PEDANTIC,CUBLAS_GEMM_DEFAULT));
    AUDIT_CUDA(cudaGetLastError());
}
