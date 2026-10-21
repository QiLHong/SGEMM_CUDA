"""Derive BM128/BN64/BK32 plus its isolated harness; never edit old experiments."""
import argparse
import difflib
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "experiments/tensorcore-bn64"


def products():
    old = (ROOT / "experiments/tensorcore-ktile/tf32_ktile.cuh").read_text(encoding="utf-8")
    s = old.replace("namespace ktile_tf32", "namespace bn64_tf32")
    s = s.replace("// EXPERIMENT: dynamic shared and 16-wide slabs allow BK=16/32; see device.diff.",
                  "// EXPERIMENT: BM128/BN64/BK32; 48KiB shared, 32 accumulators/thread; see device.diff.")
    s = s.replace("const int BN = 128, const int BK = 16", "const int BN = 64, const int BK = 32")
    s = s.replace("BM == 128 && BN == 128 && (BK == 16 || BK == 32)", "BM == 128 && BN == 64 && BK == 32")
    s = s.replace("// a block calculate c[128][128]", "// 一个CTA计算128x64；8个warp按2x4排列，每warp负责64x16。")
    s = s.replace("int load_b_row = tid / WARP_SIZE;", "int load_b_row = tid / 16;")
    s = s.replace("int load_b_col = (tid % WARP_SIZE) * 4;", "int load_b_col = (tid % 16) * 4;")
    s = s.replace("float sum[4][4][4]", "float sum[4][2][4]")
    s = s.replace("uint32_t reg_b[4][2]", "uint32_t reg_b[2][2]")
    s = s.replace("n_idx < 4", "n_idx < 2").replace("warp_id_n * 32", "warp_id_n * 16")
    a = s.index("    uint32_t smem_b1 =")
    b = s.index("    float *global_b0", a)
    s = s[:a]+s[b:]
    s = s.replace("    float *global_b1 = &b[(kk + slab * 16 + load_b_row + 8) * n + bx * BN + load_b_col];\n", "")
    s = s.replace("    CP_ASYNC_CG(smem_b1, global_b1);\n", "")
    s = s.replace("    auto copy_tile =", "    // 每个16宽K slab：A仍两次16B拷贝，B改为一次；256线程覆盖B[16][64]。\n    auto copy_tile =")
    # The upstream disabled C-staging alternative still describes BN128. Remove comment only.
    a = s.index("    // Reuse As/Bs for coalesced C write-back")
    b = s.index("    }*/", a)+len("    }*/")
    s = s[:a]+s[b:]
    assert s.count("CP_ASYNC_CG(smem_") == 3
    assert "smem_b1" not in s and "n_idx < 4" not in s
    outputs = {DEST / "tf32_bn64.cuh": s,
               DEST / "device.diff": "".join(difflib.unified_diff(old.splitlines(True), s.splitlines(True),
                   fromfile="tensorcore-ktile/tf32_ktile.cuh", tofile="tensorcore-bn64/tf32_bn64.cuh"))}
    provenance = dict(author="QiLHong / vitamin-cuda", upstream_commit="520f97093a2abc239977be83d244b327d0e2e2ef",
                      upstream_path="kernels/sgemm/sgemm_tf32.cu", function="sgemm_tf32_swizzle_bcf_dbf_kernel",
                      parent_path="experiments/tensorcore-ktile/tf32_ktile.cuh",
                      parent_sha256=hashlib.sha256(old.encode()).hexdigest(),
                      candidate_sha256=hashlib.sha256(s.encode()).hexdigest(),
                      changes=["BN128 -> BN64", "B rows tid/16, columns (tid%16)*4, one copy/slab",
                               "warp N extent32 -> 16, N fragments4 -> 2", "48KiB dynamic shared",
                               "remove disabled BN128 C-staging comment"],
                      unchanged=["BM128 BK32", "256 threads, 2x4 warps", "W8 grid formula",
                                 "A layout/copy, slab swizzles", "MMA K order", "commit/wait/barrier protocol", "TF32 precision"])
    outputs[DEST / "provenance.json"] = json.dumps(provenance, indent=2)+"\n"

    old = (ROOT / "experiments/tensorcore-ktile/benchmark.cu").read_text(encoding="utf-8")
    s = old.replace('#include "tf32_ktile.cuh"', '#include "../tensorcore-ktile/tf32_ktile.cuh"\n#include "tf32_bn64.cuh"')
    s = s.replace("// TF32 K-tile experiment; derived from the preserved Tensor Core event harness.",
                  "// TF32 BN64 experiment; derived from the preserved K-tile event harness.")
    s = s.replace("// Compare original static BK16, dynamic BK16 control, dynamic BK32 and TF32 cuBLAS.",
                  "// Compare original BK16, BN128/BK32, BN64/BK32 and same-input TF32 cuBLAS.")
    s = s.replace("// All: dynamic pair adjacent, order swapped; confirm: original/BK32 adjacent.",
                  "// All: BK32 BN128/BN64 adjacent and swapped; confirm: original/BN64 adjacent.")
    s = s.replace("[all|implementation|tensor_pairs]", "[all|confirm|implementation]")
    s = s.replace("[all|implementation]", "[all|confirm|implementation]")
    s = s.replace("TFK16,TFK32", "TFK16,TFK32,TFBN64").replace("std::array<const char*,10>", "std::array<const char*,11>")
    s = s.replace('"tf32_dynamic_bk16","tf32_dynamic_bk32"', '"tf32_dynamic_bk16","tf32_dynamic_bk32","tf32_bn64_bk32"')
    s = s.replace("i<10", "i<11").replace("impl<10", "impl<11")
    s = s.replace("    ck(cudaMemset(c0.p,0,cc*4));", """    ck(cudaFuncSetAttribute(bn64_tf32::sgemm_tf32_swizzle_bcf_dbf_kernel<128,64,32>,
                           cudaFuncAttributeMaxDynamicSharedMemorySize,49152));
    ck(cudaMemset(c0.p,0,cc*4));""")
    s = s.replace("    switch(impl){", """    switch(impl){
      case TFBN64:bn64_tf32::sgemm_tf32_swizzle_bcf_dbf_kernel<128,64,32>
          <<<dim3(n/64,m/128),256,49152>>>(ta.f(),tb.f(),c.f(),m,n,k);break;""")
    s = s.replace("occOriginal=0,occ16=0,occ32=0;", "occOriginal=0,occ16=0,occ32=0,occBn64=0;")
    s = s.replace('  Event start,stop;', """  ck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occBn64,
     bn64_tf32::sgemm_tf32_swizzle_bcf_dbf_kernel<128,64,32>,256,49152));
  std::cout<<"BN64_CTA_per_SM="<<occBn64<<'\\n';
  Event start,stop;""")
    s = s.replace('  meta<<"seed="', '  meta<<"ctas_per_sm_bn64="<<occBn64<<\'\\n\';\n  meta<<"seed="')
    s = s.replace("return impl==TFDouble||impl==TFBlas||impl==TFK32||(selected!=-2&&impl==TFK16);",
                  "return impl==TFDouble||impl==TFBlas||impl==TFBN64||(selected!=-2&&impl==TFK32);")
    s = s.replace("std::vector<int>{TFK32,TFK16}:std::vector<int>{TFK16,TFK32}",
                  "std::vector<int>{TFBN64,TFK32}:std::vector<int>{TFK32,TFBN64}")
    s = s.replace("std::vector<int>{TFK32,TFDouble}:std::vector<int>{TFDouble,TFK32}",
                  "std::vector<int>{TFBN64,TFDouble}:std::vector<int>{TFDouble,TFBN64}")
    outputs[DEST / "benchmark.cu"] = s
    outputs[DEST / "host-harness.diff"] = "".join(difflib.unified_diff(old.splitlines(True), s.splitlines(True),
        fromfile="tensorcore-ktile/benchmark.cu", tofile="tensorcore-bn64/benchmark.cu"))
    s = (ROOT / "experiments/tensorcore-ktile/CMakeLists.txt").read_text(encoding="utf-8")
    outputs[DEST / "CMakeLists.txt"] = s.replace("KTileLearning", "BN64Learning").replace("tensor_ktile", "tensor_bn64")
    for action in ("Build", "Run", "Profile"):
        s = (ROOT / f"scripts/windows/{action}-TensorKTile.ps1").read_text(encoding="utf-8-sig")
        s = s.replace("tensorcore-ktile", "tensorcore-bn64").replace("tensor-ktile", "tensor-bn64")
        s = s.replace("tensor_ktile", "tensor_bn64").replace("TensorKTile", "TensorBN64")
        if action == "Build":
            s = s.replace('if ($LASTEXITCODE) { throw "Grid device derivation changed" }', 'if ($LASTEXITCODE) { throw "BN64 derivation changed" }')
            s = s.replace('& $Python "$source\\derive_benchmark.py" --verify\nif ($LASTEXITCODE) { throw "Host derivation changed" }\n', "")
            s = s.replace('"$repo\\experiments\\tensorcore-grid" -File', '"$repo\\experiments\\tensorcore-grid","$repo\\experiments\\tensorcore-ktile" -File')
            s = s.replace('$manifest=[ordered]@{', 'Copy-Item "$repo\\experiments\\tensorcore-ktile" "$EvidenceDir\\ktile-source" -Recurse\n$manifest=[ordered]@{')
        if action == "Run":
            s = s.replace("all: dynamic BK16/BK32 adjacent and swap; rotate pair/original/cuBLAS. confirm: original/BK32 adjacent and swap; rotate cuBLAS.",
                          "all: BN128/BK32 and BN64/BK32 adjacent and swap; rotate original/cuBLAS. confirm: original/BN64 adjacent and swap.")
        if action == "Profile":
            s = s.replace("'tf32_dynamic_bk16','tf32_dynamic_bk32'", "'tf32_dynamic_bk32','tf32_bn64_bk32'")
        outputs[ROOT / f"scripts/windows/{action}-TensorBN64.ps1"] = s
    s = (ROOT / "scripts/summarize_tensor_ktile.py").read_text(encoding="utf-8")
    s = s.replace('baseline, "tf32_dynamic_bk16", "cublas_tf32"', 'baseline, "tf32_dynamic_bk32", "cublas_tf32"')
    s = s.replace('if "dynamic" in row["candidate"]', 'if "bn64" in row["candidate"]')
    outputs[ROOT / "scripts/summarize_tensor_bn64.py"] = s
    return outputs


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--verify", action="store_true")
    args = p.parse_args()
    for path, source in products().items():
        if args.verify:
            assert path.read_text(encoding="utf-8-sig") == source, path
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                raise FileExistsError(path)
            path.write_text(source, encoding="utf-8", newline="\n")
    print("Verified BN64 derivation" if args.verify else "Generated BN64 experiment")
