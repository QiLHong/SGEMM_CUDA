#include <cuda_runtime.h>
#include <cuda.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cmath>
#include <fstream>
#include <string>
#define CK(x) do { auto e=(x); if(e!=cudaSuccess){std::fprintf(stderr,"%s: %s\n",#x,cudaGetErrorString(e));return 1;} } while(0)
__global__ void gemm(const float* a,const float* b,float* c){int i=threadIdx.x/16,j=threadIdx.x%16;float s=0;for(int k=0;k<16;k++)s+=a[i*16+k]*b[k*16+j];c[i*16+j]=s;}
int main(){
 cudaDeviceProp p{}; CK(cudaGetDeviceProperties(&p,0));int rt=0,dr=0;CK(cudaRuntimeGetVersion(&rt));CK(cudaDriverGetVersion(&dr));
 std::printf("GPU=%s capability=%d.%d SM=%d runtime=%d driver=%d shared_per_sm=%zu shared_optin=%zu regs_per_sm=%d\n",p.name,p.major,p.minor,p.multiProcessorCount,rt,dr,p.sharedMemPerMultiprocessor,p.sharedMemPerBlockOptin,p.regsPerMultiprocessor);
 cublasHandle_t h; if(cublasCreate(&h)!=CUBLAS_STATUS_SUCCESS)return 2;int cv=0;cublasGetVersion(h,&cv);std::printf("cublas_version=%d\n",cv);
 float a[256],b[256],c[256];for(int i=0;i<256;i++){a[i]=float((i*7)%31-15)/16;b[i]=float((i*11)%29-14)/16;}
 float *da,*db,*dc;CK(cudaMalloc(&da,sizeof(a)));CK(cudaMalloc(&db,sizeof(b)));CK(cudaMalloc(&dc,sizeof(c)));CK(cudaMemcpy(da,a,sizeof(a),cudaMemcpyHostToDevice));CK(cudaMemcpy(db,b,sizeof(b),cudaMemcpyHostToDevice));
 gemm<<<1,256>>>(da,db,dc);CK(cudaGetLastError());CK(cudaDeviceSynchronize());CK(cudaMemcpy(c,dc,sizeof(c),cudaMemcpyDeviceToHost));double err=0;int bad=0;for(int i=0;i<16;i++)for(int j=0;j<16;j++){double ref=0;for(int k=0;k<16;k++)ref+=double(a[i*16+k])*b[k*16+j];err=fmax(err,fabs(c[i*16+j]-ref));bad+=(!std::isfinite(c[i*16+j])||fabs(c[i*16+j]-ref)>1e-6);}
 std::printf("GEMM_SMOKE %s full_CPU_double=256 max_abs=%.17g\n",bad?"FAIL":"PASS",err);
 int blocks=0;CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,gemm,256,0));std::printf("smoke_active_blocks_per_sm=%d\n",blocks);
 std::ifstream maps("/proc/self/maps");std::string l;while(std::getline(maps,l))if(l.find("libcuda")!=std::string::npos||l.find("libcublas")!=std::string::npos){auto k=l.find('/');if(k!=std::string::npos)std::printf("loaded=%s\n",l.substr(k).c_str());}
 cublasDestroy(h);CK(cudaFree(da));CK(cudaFree(db));CK(cudaFree(dc));return bad?3:0;
}

