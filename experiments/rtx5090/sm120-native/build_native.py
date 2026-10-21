"""Generate host-extended source without changing upstream device bytes."""
import hashlib
import json
import os
from pathlib import Path

def digest(x):
    return hashlib.sha256(x).hexdigest()

def build(upstream, build_root):
    from torch.utils.cpp_extension import load
    upstream, build_root = Path(upstream).resolve(), Path(build_root).resolve()
    build_root.mkdir(parents=True, exist_ok=True)
    raw = (upstream / "hgemm_sm120.cu").read_bytes()
    source = raw.decode("utf-8")
    common = upstream.parent / "common" / "pack.cuh"
    assert common.is_file(), common
    include = '#include "../common/pack.cuh"'
    assert source.count(include)==1
    marker = 'PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {'
    assert source.count(marker)==1
    host = (Path(__file__).parent / "native_host.inc").read_text(encoding="utf-8")
    # The source's device region is unchanged even before line-ending normalization.
    device_start = source.index('#define SWIZZLE_A')
    device_end = source.index('#define CHECK_T')
    device_original = source[device_start:device_end]
    generated = source.replace(include, '#include "'+common.as_posix()+'"')
    generated = generated.replace(marker, host+'\n'+marker+'\n    m.def("audit_measure", &audit_measure);\n    m.def("audit_pedantic", &audit_pedantic);')
    assert generated[generated.index('#define SWIZZLE_A'):generated.index('#define CHECK_T')] == device_original
    path = build_root / "hgemm_sm120_native.cu"
    if not path.exists() or path.read_bytes() != generated.encode("utf-8"):
        path.write_text(generated, encoding="utf-8", newline="")
    common_flags = ["-O3","-std=c++17"]
    flags = common_flags + ["-U__CUDA_NO_HALF_OPERATORS__","-U__CUDA_NO_HALF_CONVERSIONS__","-U__CUDA_NO_HALF2_OPERATORS__","-U__CUDA_NO_BFLOAT16_CONVERSIONS__","--expt-relaxed-constexpr","--expt-extended-lambda","--use_fast_math","-Xptxas","-v"]
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")
    manifest = {"upstream_source_sha256":digest(raw),"upstream_cublas_sha256":digest((upstream/'cublas.cu').read_bytes()),
                "common_header_sha256":digest(common.read_bytes()),"host_addition_sha256":digest(host.encode()),
                "generated_source_sha256":digest(path.read_bytes()),"device_region_sha256":digest(device_original.encode()),
                "device_region_byte_equal":True,"TORCH_CUDA_ARCH_LIST":os.environ['TORCH_CUDA_ARCH_LIST'],"extra_cuda_cflags":flags,
                "extra_ldflags":["-L/usr/local/cuda/lib64/stubs","-lcuda"],"timing_boundary":"native CUDA event around one launch; descriptor encoding/attribute setup/alloc/conversion/C0 restore outside events"}
    (build_root/'source-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    lib=load(name="sm120_native_audit_lib", sources=[str(path),str(upstream/'cublas.cu')], extra_cuda_cflags=flags,
             extra_cflags=common_flags,extra_ldflags=["-L/usr/local/cuda/lib64/stubs","-lcuda"],build_directory=str(build_root),verbose=True)
    return lib,manifest

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument('--upstream',required=True)
    p.add_argument('--build-root',required=True)
    a=p.parse_args()
    lib,manifest=build(a.upstream,a.build_root)
    print(json.dumps(manifest,indent=2))
