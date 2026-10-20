"""Extract upstream Tensor Core device regions verbatim; never modify upstream."""
import argparse
import hashlib
import json
import pathlib
import re
import subprocess

COMMIT = "520f97093a2abc239977be83d244b327d0e2e2ef"
ROOT = pathlib.Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--verify", action="store_true")
    args = p.parse_args()
    upstream = ROOT.parent / "vitamin-cuda"
    assert subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip() == COMMIT
    dest = ROOT / "experiments/tensorcore/vendor"
    records = []
    for kind, relative in (("tf32", "kernels/sgemm/sgemm_tf32.cu"), ("fp16", "kernels/hgemm/hgemm.cu")):
        raw = (upstream / relative).read_bytes()
        source = raw.decode("utf-8").replace("\r\n", "\n")
        start = source.index("#define SWIZZLE_A")
        end = source.index("#define CHECK_T")
        device = source[start:end]
        macros = re.findall(r"^#define\s+(\w+)", device, re.M)
        header = (
            "// Original author: QiLHong / vitamin-cuda; see LICENSE in this directory.\n"
            f"// Source commit {COMMIT}, {relative}\n"
            "// Device region below is unchanged. Only headers, namespace and macro cleanup are adapted.\n"
            "#pragma once\n#include <cuda_runtime.h>\n#include <cuda_fp16.h>\n"
            "#include <cuda_bf16.h>\n#include <cstdint>\n#include <type_traits>\n"
            '#include "pack.cuh"\n'
            f"namespace vitamin_{kind} {{\n"
        )
        text = header + device + "\n}\n" + "\n".join(f"#undef {x}" for x in dict.fromkeys(macros)) + "\n"
        target = dest / f"vitamin_{kind}.cuh"
        if args.verify:
            assert target.read_text(encoding="utf-8") == text, target
        else:
            dest.mkdir(parents=True, exist_ok=True)
            if target.exists():
                raise FileExistsError(target)
            target.write_text(text, encoding="utf-8", newline="\n")
        records.append(dict(source=relative, commit=COMMIT,
                            source_sha256=hashlib.sha256(raw).hexdigest(),
                            device_sha256=hashlib.sha256(device.encode()).hexdigest(),
                            extracted_sha256=hashlib.sha256(text.encode()).hexdigest(),
                            first_source_line=source[:start].count("\n")+1,
                            last_source_line=source[:end].count("\n"),
                            changes="headers, namespace, macro cleanup only"))
    for relative, filename in (("kernels/common/pack.cuh", "pack.cuh"), ("LICENSE", "LICENSE")):
        raw = (upstream / relative).read_bytes()
        target = dest / filename
        if args.verify:
            assert target.read_bytes() == raw
        else:
            if target.exists():
                raise FileExistsError(target)
            target.write_bytes(raw)
    manifest = dict(commit=COMMIT, regions=records,
                    contract="M%128=0,N%1024=0,K%32=0; positive; TF32 input explicitly preconverted outside benchmark")
    if args.verify:
        assert json.loads((dest / "provenance.json").read_text()) == manifest
    else:
        (dest / "provenance.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print("VERIFIED unchanged device regions" if args.verify else "EXTRACTED unchanged device regions")


if __name__ == "__main__":
    main()
