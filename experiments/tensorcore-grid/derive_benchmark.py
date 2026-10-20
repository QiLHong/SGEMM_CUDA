"""Derive the host harness from stage 4; preserve numerical/event protocol."""
from pathlib import Path
import difflib
import sys

HERE = Path(__file__).resolve().parent
old = (HERE.parent / "tensorcore/benchmark.cu").read_text(encoding="utf-8")
s = old.replace('#include "vendor/', '#include "../tensorcore/vendor/')
s = s.replace("// 原生 Tensor Core 教学实验。来源见 vendor/provenance.json；原 device region 不改。",
"""// Tensor Core grid-width experiment; derived from ../tensorcore/benchmark.cu.
// Original kernels stay unchanged; experimental kernels only parameterize GRID_W.
// Default scan: W8/W1/W2/W4/cuBLAS per precision; confirm grid_w1/2/4 separately.
// Precision and event protocol retained; additionally require baseline bitwise equality.""")
s = s.replace('void ck(cudaError_t', '#include "tf32_grid.cuh"\n#include "fp16_grid.cuh"\n\nvoid ck(cudaError_t', 1)
s = s.replace("HSingle,HDouble,HBlas};", "HSingle,HDouble,HBlas,TFW1,TFW2,TFW4,HW1,HW2,HW4};")
s = s.replace("std::array<const char*,8>", "std::array<const char*,14>")
s = s.replace('"vitamin_fp16_single","vitamin_fp16_dbf_rw","cublas_fp16"}};',
              '"vitamin_fp16_single","vitamin_fp16_dbf_rw","cublas_fp16",\n'
              '    "tf32_w1","tf32_w2","tf32_w4","fp16_w1","fp16_w2","fp16_w4"}};')
s = s.replace("for(int i=0;i<8;++i)", "for(int i=0;i<14;++i)")
s = s.replace("return impl<2?0:(impl<5?1:2);", "return impl<8?(impl<2?0:(impl<5?1:2)):(impl<11?1:2);")
extra = ""
for kind, ids, name, bk in [
    ("tf32", ["TFW1", "TFW2", "TFW4"], "sgemm_tf32_swizzle_bcf_dbf_kernel", 16),
    ("fp16", ["HW1", "HW2", "HW4"], "hgemm_bcf_dbf_rw_kernel", 32),
]:
    pointers = "ta.f(),tb.f(),c.f()" if kind == "tf32" else "ha.h(),hb.h(),hc.h()"
    for width, ident in zip((1, 2, 4), ids):
        extra += f"      case {ident}:grid_{kind}::{name}<{width},128,128,{bk}>" \
                 f"<<<grid,256>>>({pointers},m,n,k);break;\n"
s = s.replace("    switch(impl){", "    switch(impl){\n" + extra, 1)
s = s.replace('selected=std::string(argv[8])=="tensor_pairs"?-2:findImpl(argv[8]);',
"""selected=std::string(argv[8])=="grid_w1"?-4:
               std::string(argv[8])=="grid_w2"?-5:
               std::string(argv[8])=="grid_w4"?-6:findImpl(argv[8]);""")
s = s.replace("std::ofstream events(path),checks", "std::ofstream bits(path+\".bitwise.csv\");\n  std::ofstream events(path),checks")
s = s.replace("  windows<<\"implementation", '  bits<<"implementation,round,count,bitwise_mismatches\\n";\n  windows<<"implementation')
start = s.index("  auto chosen=")
end = s.index("  auto validate=", start)
s = s[:start] + """  std::array<std::vector<float>,3> baseline;
  for(int p=1;p<3;++p){int impl=p==1?TFDouble:HDouble;
    d.reset(impl);d.launch(impl);baseline[p]=d.output(impl);}
  auto chosen=[&](int impl){
    if(selected>=0)return selected==impl;
    if(selected<=-4){int offset=-selected-4;
      return impl==TFDouble||impl==TFBlas||impl==HDouble||impl==HBlas||
             impl==TFW1+offset||impl==HW1+offset;}
    return impl==TFDouble||impl==TFBlas||impl==HDouble||impl==HBlas||impl>=TFW1;
  };
""" + s[end:]
s = s.replace("    failed|=bad;", """    if(p>0&&impl!=TFBlas&&impl!=HBlas){
      size_t mismatch=0;
      for(size_t i=0;i<d.cc;++i)
        mismatch+=std::memcmp(&actual[i],&baseline[p][i],sizeof(float))!=0;
      bits<<Names[impl]<<','<<round<<','<<d.cc<<','<<mismatch<<'\\n';bits.flush();
      bad|=mismatch!=0;
    }
    failed|=bad;""")
s = s.replace("for(int impl=0;impl<8;++impl)", "for(int impl=0;impl<14;++impl)")
start = s.index("    std::array<std::vector<int>,3> groups")
end = s.index("    int order=0;", start)
s = s[:start] + """    std::array<std::vector<int>,3> groups={std::vector<int>{},
        std::vector<int>{TFDouble,TFW1,TFW2,TFW4,TFBlas},
        std::vector<int>{HDouble,HW1,HW2,HW4,HBlas}};
    for(int p=1;p<3;++p){
      if(selected<=-4){
        int base=p==1?TFDouble:HDouble, lib=p==1?TFBlas:HBlas;
        int candidate=(p==1?TFW1:HW1)+(-selected-4);
        groups[p]=r%2?std::vector<int>{candidate,base}:std::vector<int>{base,candidate};
        if(r%3==0)groups[p].insert(groups[p].begin(),lib);else groups[p].push_back(lib);
      }else{
        std::rotate(groups[p].begin(),groups[p].begin()+r%5,groups[p].end());
        if(r%2)std::reverse(groups[p].begin(),groups[p].end());
      }
    }
""" + s[end:]
outputs = {"benchmark.cu": s,
           "host-harness.diff": "".join(difflib.unified_diff(old.splitlines(True), s.splitlines(True),
                                  fromfile="tensorcore/benchmark.cu", tofile="tensorcore-grid/benchmark.cu"))}
for name, text in outputs.items():
    path = HERE / name
    if "--verify" in sys.argv:
        assert path.read_text(encoding="utf-8") == text, path
    else:
        if path.exists():
            raise FileExistsError(path)
        path.write_text(text, encoding="utf-8", newline="\n")
print("Verified host derivation" if "--verify" in sys.argv else "Derived host harness")
