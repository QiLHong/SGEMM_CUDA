"""Reuse upstream benchmark/diff_check in an isolated Windows Python experiment.

Never executes test.py's __main__ (which selects TF32 kernels). First try its
original extension load. Only after failure, retry with MSVC host flags and an
explicit cuBLAS import library; original source files and CUDA flags are retained.
The output is the upstream single-batch wallclock protocol, not CUDA events.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
from functools import partial
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import sys
import traceback


MODES = (
    "sgemm_cublas", "sgemm_tiling", "sgemm_at_tiling",
    "sgemm_at_bcf_swizzling", "sgemm_at_bcf_swizzling_rw",
    "sgemm_at_bcf_swizzling_dbf_rw",
)


def save_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


class Tee:
    def __init__(self, first, second):
        self.first, self.second = first, second

    def write(self, text):
        self.first.write(text)
        self.first.flush()
        return self.second.write(text)

    def flush(self):
        self.first.flush()
        self.second.flush()


def load_upstream(path, output, torch_extension):
    attempts = []
    original_load = torch_extension.load
    for adapted in (False, True):
        record = {"attempt": "windows_host_link_adapter" if adapted else "original_load",
                  "source": str(path), "default_main_executed": False}

        def observed_load(*args, **kwargs):
            record["upstream_load_arguments"] = dict(kwargs)
            if adapted:
                if kwargs.get("extra_cflags") != ["-O3", "-std=c++17"]:
                    raise ValueError("Upstream host flags changed; refusing an unreviewed adaptation.")
                kwargs = dict(kwargs)
                kwargs["name"] = "vitamin_windows_probe"
                kwargs["extra_cflags"] = ["/O2", "/std:c++17"]
                # Windows subprocess quoting treats "-Xptxas -v" as one option.
                # Preserve its diagnostics meaning with nvcc's equals spelling.
                kwargs["extra_cuda_cflags"] = [
                    "-Xptxas=-v" if flag == "-Xptxas -v" else flag
                    for flag in kwargs.get("extra_cuda_cflags", [])
                ]
                cuda_path = Path(os.environ["CUDA_PATH"])
                cublas = cuda_path / "lib" / "x64" / "cublas.lib"
                if not cublas.is_file():
                    raise FileNotFoundError(cublas)
                kwargs["extra_ldflags"] = [*kwargs.get("extra_ldflags", []), str(cublas)]
                record["adaptations"] = [
                    "New extension cache name preserves failed original build artifacts.",
                    "Host -O3/-std=c++17 replaced by /O2 and /std:c++17.",
                    "Explicit existing Windows CUDA cublas.lib link dependency.",
                    "Equivalent -Xptxas=-v spelling avoids Windows quoting the upstream two-word string as one unknown option.",
                    "Original CUDA optimization/math flags (including --use_fast_math) and all three CU sources retained.",
                ]
            record["effective_load_arguments"] = dict(kwargs)
            return original_load(*args, **kwargs)

        print("BUILD_ATTEMPT_BEGIN", record["attempt"], flush=True)
        torch_extension.load = observed_load
        try:
            spec = importlib.util.spec_from_file_location("vitamin_upstream_wallclock_probe", path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Cannot load {path}")
            module = importlib.util.module_from_spec(spec)
            # Name differs from __main__: no upstream default TF32 test executes.
            spec.loader.exec_module(module)
            record["status"] = "PASS"
            attempts.append(record)
            save_json(output / "build-attempts.json", attempts)
            print("BUILD_ATTEMPT_PASS", record["attempt"], flush=True)
            return module, record["attempt"]
        except Exception:
            record["status"] = "FAIL"
            record["traceback"] = traceback.format_exc()
            attempts.append(record)
            save_json(output / "build-attempts.json", attempts)
            print(record["traceback"], flush=True)
            print("BUILD_ATTEMPT_FAIL", record["attempt"], flush=True)
        finally:
            torch_extension.load = original_load
    raise RuntimeError("Original load and bounded Windows host/link adapter both failed; no installation attempted.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vitamin", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Existing fresh directory created by the PowerShell wrapper")
    parser.add_argument("--sizes", type=int, nargs="+", default=[1024, 2048, 4096])
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if not output.is_dir() or (output / "python-environment.json").exists():
        raise ValueError("Output must be a fresh wrapper-created directory.")
    if any(size not in (1024, 2048, 4096) for size in args.sizes):
        raise ValueError("Only reviewed aligned sizes 1024, 2048, 4096 are supported.")

    import numpy as np
    import torch
    import torch.utils.cpp_extension as extension

    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    if not torch.cuda.is_available():
        raise RuntimeError("Existing PyTorch CUDA runtime is unavailable.")
    source_dir = args.vitamin.resolve() / "kernels" / "sgemm"
    source_files = [source_dir / name for name in ("test.py", "sgemm.cu", "sgemm_tf32.cu", "cublas.cu")]
    save_json(output / "python-environment.json", {
        "python": sys.executable, "python_version": sys.version,
        "torch_version": torch.__version__, "torch_cuda_build": torch.version.cuda,
        "device": torch.cuda.get_device_name(0), "capability": torch.cuda.get_device_capability(0),
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "environment": {name: os.environ.get(name) for name in (
            "CUDA_PATH", "TORCH_CUDA_ARCH_LIST", "TORCH_EXTENSIONS_DIR",
            "NVIDIA_TF32_OVERRIDE", "MAX_JOBS", "PYTHONDONTWRITEBYTECODE",
            "PYTHONUTF8", "PYTHONIOENCODING", "VSLANG", "DISTUTILS_USE_SDK")},
        "source_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files},
        "protocol": "Unmodified upstream benchmark/diff_check function bodies; new Windows invocation wrapper.",
        "timer": "Upstream time.perf_counter around 200 CPU-dispatched operations plus final synchronize.",
        "warmup": 10, "repetitions": 200, "independent_batches_per_implementation": 1,
        "seed": 42, "seed_policy": "Reset torch.manual_seed(42) before each size.",
        "input": "Upstream-style CPU torch.randn float32 transferred to CUDA; zero C; differs from native xorshift input.",
        "alpha": 1, "beta": 0,
        "correctness_direction": "Original diff_check(reference, candidate); torch.allclose uses candidate magnitude for rtol.",
        "precision": "Original custom-kernel --use_fast_math retained; TF32 disabled in process; ordinary CUBLAS_COMPUTE_32F.",
        "limitation": "Single wallclock batch and rounded printed means; not comparable to native event median tables.",
    })
    module, build_method = load_upstream(source_dir / "test.py", output, extension)
    inputs, results = [], []
    fields = (
        "size", "implementation", "seed", "warmup", "repetitions", "printed_mean_ms",
        "printed_tflops", "upstream_allclose", "upstream_failure_count",
        "max_abs_error", "max_rel_error_reference", "abs_001_failure_count",
        "candidate_nonfinite_count", "reference_nonfinite_count", "a_sha256", "b_sha256",
        "initial_c_sha256", "build_method",
    )
    failed = False
    for n in args.sizes:
        torch.manual_seed(42)
        a = torch.randn(n, n).float().cuda()
        b = torch.randn(n, n).float().cuda()
        reference = torch.zeros(n, n).float().cuda()
        hashes = {
            "a_sha256": hashlib.sha256(a.cpu().numpy().tobytes(order="C")).hexdigest(),
            "b_sha256": hashlib.sha256(b.cpu().numpy().tobytes(order="C")).hexdigest(),
            "initial_c_sha256": hashlib.sha256(reference.cpu().numpy().tobytes(order="C")).hexdigest(),
        }
        inputs.append({"size": n, "dtype": "float32", "storage": "C-order row major",
                       "seed": 42, **hashes})
        save_json(output / "input-sha256.json", inputs)
        print(f"SIZE_BEGIN n={n} alpha=1 beta=0 seed=42 warmup=10 rep=200", flush=True)
        reference_cpu = None
        for mode in ("torch", *MODES):
            capture = io.StringIO()
            if mode == "torch":
                with contextlib.redirect_stdout(Tee(sys.stdout, capture)):
                    module.benchmark(partial(torch.matmul, out=reference), a, b,
                                     warmup=10, rep=200, prefix="torch")
                reference_cpu = reference.cpu().numpy().astype(np.float64)
                candidate_cpu = reference_cpu
                upstream_status = "REFERENCE"
            else:
                candidate = torch.zeros_like(reference)
                with contextlib.redirect_stdout(Tee(sys.stdout, capture)):
                    module.benchmark(getattr(module.lib, mode), a, b, candidate,
                                     warmup=10, rep=200, prefix=mode)
                    try:
                        module.diff_check(reference, candidate, prefix=mode)
                        upstream_status = "PASS"
                    except AssertionError:
                        upstream_status = "FAIL"
                        failed = True
                        print("UPSTREAM_DIFF_CHECK_FAIL", mode, flush=True)
                candidate_cpu = candidate.cpu().numpy().astype(np.float64)
                del candidate
            printed = capture.getvalue()
            mean = re.search(r"mean time:\s*([0-9.]+)\s*ms", printed)
            tflops = (re.search(r"tflops:\s*([0-9.]+)", printed) or
                      re.search(r"([0-9.]+)\s*tflops", printed))
            if mean is None or tflops is None:
                raise ValueError(f"Cannot parse upstream benchmark output: {printed}")
            delta = np.abs(candidate_cpu - reference_cpu)
            finite = np.isfinite(candidate_cpu) & np.isfinite(reference_cpu)
            reference_nonfinite = int(np.count_nonzero(~np.isfinite(reference_cpu)))
            candidate_nonfinite = int(np.count_nonzero(~np.isfinite(candidate_cpu)))
            upstream_failures = int(np.count_nonzero(~finite | (delta > 0.001 + 0.001*np.abs(candidate_cpu))))
            absolute_failures = int(np.count_nonzero(~finite | (delta > 0.01)))
            failed |= bool(reference_nonfinite or candidate_nonfinite or absolute_failures)
            record = {
                "size": n, "implementation": mode, "seed": 42, "warmup": 10, "repetitions": 200,
                "printed_mean_ms": mean[1], "printed_tflops": tflops[1],
                "upstream_allclose": upstream_status, "upstream_failure_count": upstream_failures,
                "max_abs_error": float(np.max(delta)) if finite.all() else None,
                "max_rel_error_reference": float(np.max(delta / np.maximum(np.abs(reference_cpu), 1e-12))) if finite.all() else None,
                "abs_001_failure_count": absolute_failures,
                "candidate_nonfinite_count": candidate_nonfinite, "reference_nonfinite_count": reference_nonfinite,
                **hashes, "build_method": build_method,
            }
            results.append(record)
            with (output / "wallclock-results.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(results)
            print("CORRECTNESS", json.dumps(record), flush=True)
        del a, b, reference, reference_cpu, candidate_cpu
    save_json(output / "python-result.json", {
        "status": "FAIL" if failed else "PASS", "rows": len(results),
        "sizes": args.sizes, "build_method": build_method,
        "original_test_default_main_executed": False, "tf32_kernels_executed": False,
        "description": "Upstream wallclock/validation functions reused through a Windows wrapper; not an unchanged full test.py reproduction.",
    })
    return 3 if failed else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
