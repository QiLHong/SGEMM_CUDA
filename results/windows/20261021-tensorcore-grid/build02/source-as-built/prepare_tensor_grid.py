"""Create/verify an isolated, single-factor grid-swizzle experiment.

The original vendor files and benchmark are never edited. Kernel names, license,
tile sizes, MMA, shared layouts and synchronization remain as upstream.
"""
import argparse
import difflib
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "experiments/tensorcore-grid"


def products():
    records, outputs = [], {}
    for kind, name in [
        ("tf32", "sgemm_tf32_swizzle_bcf_dbf_kernel"),
        ("fp16", "hgemm_bcf_dbf_rw_kernel"),
    ]:
        path = ROOT / f"experiments/tensorcore/vendor/vitamin_{kind}.cuh"
        original = path.read_text(encoding="utf-8")
        start = original.rfind("template <", 0, original.index(f"void {name}"))
        end = original.index("\n}\n", start) + 3
        body = original[start:end]
        assert body.count("const int SWIZZLE_W = 8;") == 1
        changed = body.replace("template <", "template <const int GRID_W, ", 1)
        changed = changed.replace("const int SWIZZLE_W = 8;", "const int SWIZZLE_W = GRID_W;")
        assert changed.replace("template <const int GRID_W, ", "template <", 1).replace(
            "const int SWIZZLE_W = GRID_W;", "const int SWIZZLE_W = 8;") == body
        prefix = original[:original.index("template <")]
        prefix = prefix.replace('see LICENSE in this directory.', 'see ../tensorcore/vendor/LICENSE.')
        prefix = prefix.replace(
            "// Device region below is unchanged. Only headers, namespace and macro cleanup are adapted.",
            "// EXPERIMENT: only grid width is parameterized; not an unchanged upstream kernel.")
        prefix = prefix.replace('"pack.cuh"', '"../tensorcore/vendor/pack.cuh"')
        prefix = prefix.replace(f"namespace vitamin_{kind}", f"namespace grid_{kind}")
        suffix = original[original.index("#undef SWIZZLE_A"):]
        text = prefix + changed + "\n}\n" + suffix
        outputs[f"{kind}_grid.cuh"] = text
        diff = "".join(difflib.unified_diff(body.splitlines(True), changed.splitlines(True),
                      fromfile=f"upstream/{name}", tofile=f"grid/{name}"))
        outputs[f"{kind}-device.diff"] = diff
        records.append(dict(kind=kind, original_header=str(path.relative_to(ROOT)),
                            function=name, source_body_sha256=hashlib.sha256(body.encode()).hexdigest(),
                            experiment_sha256=hashlib.sha256(text.encode()).hexdigest(),
                            changes=["add GRID_W template parameter", "SWIZZLE_W=GRID_W instead of 8"]))
    outputs["provenance.json"] = json.dumps(dict(
        author="QiLHong / vitamin-cuda",
        commit="520f97093a2abc239977be83d244b327d0e2e2ef",
        source_paths=["kernels/sgemm/sgemm_tf32.cu", "kernels/hgemm/hgemm.cu"],
        adaptation=records), indent=2) + "\n"
    return outputs


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--verify", action="store_true")
    args = p.parse_args()
    for name, text in products().items():
        dest = DEST / name
        if args.verify:
            assert dest.read_text(encoding="utf-8") == text, dest
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists():
                raise FileExistsError(dest)
            dest.write_text(text, encoding="utf-8", newline="\n")
    print("VERIFIED: kernel changes restricted to grid width" if args.verify else "GENERATED grid experiment")
