"""Reuse stage-4 numerical/event protocol, add BK16/BK32 dynamic-shared cases."""
from pathlib import Path
import difflib
import sys

HERE = Path(__file__).resolve().parent
old = (HERE.parent / "tensorcore-grid/benchmark.cu").read_text(encoding="utf-8")
s = old.replace('#include "tf32_grid.cuh"\n#include "fp16_grid.cuh"', '#include "tf32_ktile.cuh"')
s = s.replace("// Tensor Core grid-width experiment; derived from ../tensorcore/benchmark.cu.",
              "// TF32 K-tile experiment; derived from the preserved Tensor Core event harness.")
s = s.replace("// Original kernels stay unchanged; experimental kernels only parameterize GRID_W.",
              "// Compare original static BK16, dynamic BK16 control, dynamic BK32 and TF32 cuBLAS.")
s = s.replace("// Default scan: W8/W1/W2/W4/cuBLAS per precision; confirm grid_w1/2/4 separately.",
              "// All: dynamic pair adjacent, order swapped; confirm: original/BK32 adjacent.")
s = s.replace("TFW1,TFW2,TFW4,HW1,HW2,HW4", "TFK16,TFK32")
s = s.replace("std::array<const char*,14>", "std::array<const char*,10>")
s = s.replace('"tf32_w1","tf32_w2","tf32_w4","fp16_w1","fp16_w2","fp16_w4"',
              '"tf32_dynamic_bk16","tf32_dynamic_bk32"')
s = s.replace("i<14", "i<10").replace("impl<14", "impl<10")
s = s.replace("(impl<11?1:2)", "1")
a = s.index("      case TFW1:")
b = s.index("      case K30:", a)
s = s[:a] + """      case TFK16:ktile_tf32::sgemm_tf32_swizzle_bcf_dbf_kernel<128,128,16>
          <<<grid,256,32768>>>(ta.f(),tb.f(),c.f(),m,n,k);break;
      case TFK32:ktile_tf32::sgemm_tf32_swizzle_bcf_dbf_kernel<128,128,32>
          <<<grid,256,65536>>>(ta.f(),tb.f(),c.f(),m,n,k);break;
""" + s[b:]
s = s.replace("    ck(cudaMemset(c0.p,0,cc*4));", """    ck(cudaFuncSetAttribute(ktile_tf32::sgemm_tf32_swizzle_bcf_dbf_kernel<128,128,16>,
                           cudaFuncAttributeMaxDynamicSharedMemorySize,32768));
    ck(cudaFuncSetAttribute(ktile_tf32::sgemm_tf32_swizzle_bcf_dbf_kernel<128,128,32>,
                           cudaFuncAttributeMaxDynamicSharedMemorySize,65536));
    ck(cudaMemset(c0.p,0,cc*4));""")
a = s.index('selected=std::string(argv[8])=="grid_w1"')
b = s.index(';', a)+1
s = s[:a] + 'selected=std::string(argv[8])=="confirm"?-2:findImpl(argv[8]);' + s[b:]
a = s.index("  auto chosen=")
b = s.index("  auto validate=", a)
s = s[:a] + """  auto chosen=[&](int impl){
    if(selected>=0)return selected==impl;
    return impl==TFDouble||impl==TFBlas||impl==TFK32||(selected!=-2&&impl==TFK16);
  };
""" + s[b:]
a = s.index("    std::array<std::vector<int>,3> groups")
b = s.index("    int order=0;", a)
s = s[:a] + """    std::vector<int> pair=r%2?std::vector<int>{TFK32,TFK16}:std::vector<int>{TFK16,TFK32};
    std::array<std::vector<int>,3> groups;
    if(selected==-2){
      groups={r%2?std::vector<int>{TFK32,TFDouble}:std::vector<int>{TFDouble,TFK32},
              std::vector<int>{TFBlas},std::vector<int>{}};
    }else{
      groups={pair,std::vector<int>{TFDouble},std::vector<int>{TFBlas}};
    }
""" + s[b:]
s = s.replace("  Event start,stop;", """  int occOriginal=0,occ16=0,occ32=0;
  ck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occOriginal,
     vitamin_tf32::sgemm_tf32_swizzle_bcf_dbf_kernel<128,128,16>,256,0));
  ck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ16,
     ktile_tf32::sgemm_tf32_swizzle_bcf_dbf_kernel<128,128,16>,256,32768));
  ck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ32,
     ktile_tf32::sgemm_tf32_swizzle_bcf_dbf_kernel<128,128,32>,256,65536));
  std::cout<<"maxSharedOptin="<<prop.sharedMemPerBlockOptin
           <<" CTA_per_SM(original/dynamic16/dynamic32)="<<occOriginal<<'/'<<occ16<<'/'<<occ32<<'\\n';
  Event start,stop;""")
s = s.replace('  meta<<"seed="', '  meta<<"max_shared_optin_bytes="<<prop.sharedMemPerBlockOptin<<"\\nctas_per_sm_original="<<occOriginal\n'
              '      <<"\\nctas_per_sm_dynamic16="<<occ16<<"\\nctas_per_sm_dynamic32="<<occ32<<\'\\n\';\n  meta<<"seed="')
s = s.replace("all|grid_w1|grid_w2|grid_w4", "all|confirm")
outputs = {"benchmark.cu":s, "host-harness.diff":"".join(difflib.unified_diff(
    old.splitlines(True),s.splitlines(True),fromfile="tensorcore-grid/benchmark.cu",tofile="tensorcore-ktile/benchmark.cu"))}
for name,text in outputs.items():
    path=HERE/name
    if "--verify" in sys.argv:
        assert path.read_text(encoding="utf-8")==text,path
    else:
        if path.exists():raise FileExistsError(path)
        path.write_text(text,encoding="utf-8",newline="\n")
print("Verified host derivation" if "--verify" in sys.argv else "Derived host harness")
