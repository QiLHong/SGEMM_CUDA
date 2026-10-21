"""Derive the TF32 K-tile experiment without editing the upstream baseline."""
import argparse
import difflib
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "experiments/tensorcore-ktile"


def products():
    original = (ROOT / "experiments/tensorcore/vendor/vitamin_tf32.cuh").read_text(encoding="utf-8")
    name = "sgemm_tf32_swizzle_bcf_dbf_kernel"
    start = original.rfind("template <", 0, original.index("void " + name))
    end = original.index("\n}\n", start)+3
    body = original[start:end]
    changed = body.replace("    // grid swizzling", "    static_assert(BM == 128 && BN == 128 && (BK == 16 || BK == 32));\n    constexpr int Slices = BK / 16;\n    // grid swizzling", 1)
    changed = changed.replace("    __shared__ float As[2][BM][BK];\n    __shared__ float Bs[2][BK][BM];", """    // BK32 requires opt-in dynamic shared memory. Each 16-wide slab keeps
    // the original A row stride and XOR mapping; do not use A[128][32].
    extern __shared__ __align__(16) unsigned char shared_bytes[];
    auto As = reinterpret_cast<float (*)[Slices][BM][16]>(shared_bytes);
    auto Bs = reinterpret_cast<float (*)[Slices][16][BN]>(
        shared_bytes + 2 * Slices * BM * 16 * sizeof(float));""")
    # Reuse the exact four upstream copies for every 16-wide K slab.
    a = changed.index("    int a_swizzle_col_0")
    b = changed.index("    CP_ASYNC_COMMIT_GROUP();", a)
    copies = changed[a:b]
    copies = copies.replace("As[0][", "As[slot][slab][").replace("Bs[0][", "Bs[slot][slab][")
    copies = copies.replace("* k + 0 +", "* k + kk + slab * 16 +")
    copies = copies.replace("[(0 + load_b_row", "[(kk + slab * 16 + load_b_row")
    changed = changed[:a] + """    auto copy_tile = [&](int slot, int kk) {
#pragma unroll
        for (int slab = 0; slab < Slices; ++slab) {
""" + copies + """        }
    };
    copy_tile(0, 0);

""" + changed[b:]
    a = changed.index("        // 1. Issue async prefetch")
    b = changed.index("        CP_ASYNC_COMMIT_GROUP();", a)
    changed = changed[:a] + "        copy_tile(write_idx, bk);\n" + changed[b:]
    assert changed.count("k_step < 2") == 2
    changed = changed.replace("k_step < 2", "k_step < BK / 8")
    changed = changed.replace("int k_offset = k_step * 8;", "int slab = k_step / 2;\n        int k_offset = (k_step % 2) * 8;")
    changed = changed.replace("As[read_idx][", "As[read_idx][slab][").replace("Bs[read_idx][", "Bs[read_idx][slab][")
    # Record exact changes. Main compute is the original fragment/MMA body;
    # only slab selection and K-step count differ.
    assert changed.count("const int SWIZZLE_W = 8;") == 1
    prefix = original[:original.index("template <")]
    prefix = prefix.replace("see LICENSE in this directory.", "see ../tensorcore/vendor/LICENSE.")
    prefix = prefix.replace("// Device region below is unchanged. Only headers, namespace and macro cleanup are adapted.",
                            "// EXPERIMENT: dynamic shared and 16-wide slabs allow BK=16/32; see device.diff.")
    prefix = prefix.replace('"pack.cuh"', '"../tensorcore/vendor/pack.cuh"').replace("namespace vitamin_tf32", "namespace ktile_tf32")
    code = prefix + changed + "\n}\n" + original[original.index("#undef SWIZZLE_A"):]
    manifest = dict(author="QiLHong / vitamin-cuda", commit="520f97093a2abc239977be83d244b327d0e2e2ef",
                    source="kernels/sgemm/sgemm_tf32.cu", function=name,
                    original_function_sha256=hashlib.sha256(body.encode()).hexdigest(),
                    experiment_sha256=hashlib.sha256(code.encode()).hexdigest(),
                    changes=["dynamic shared memory for both controls", "BK/16 slabs preserving 16-wide layout",
                             "four original copies per slab", "BK/8 ordered MMA steps", "static shape constraints"],
                    unchanged=["W8 grid mapping", "CTA/warp/output mapping", "TF32/FP32 contract",
                               "MMA fragment body", "double-buffer commit/wait/barrier protocol"])
    return {"tf32_ktile.cuh": code,
            "device.diff": "".join(difflib.unified_diff(body.splitlines(True), changed.splitlines(True),
                                    fromfile="upstream/BK16", tofile="experimental/BK16-or-32")),
            "provenance.json": json.dumps(manifest, indent=2)+"\n"}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--verify", action="store_true")
    args = p.parse_args()
    for name, text in products().items():
        target = DEST/name
        if args.verify:
            assert target.read_text(encoding="utf-8") == text, target
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                raise FileExistsError(target)
            target.write_text(text, encoding="utf-8", newline="\n")
    print("Verified K-tile derivation" if args.verify else "Generated K-tile derivation")
