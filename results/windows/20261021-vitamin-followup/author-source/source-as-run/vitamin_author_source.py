"""Call the author's unchanged test_4096 seven times, stopping before TF32.

Only the extension loader receives the existing Windows build adaptation.
The original test_4096, benchmark and diff_check function bodies are called
unchanged. The library proxy returns the original SIMT functions directly, so
there is no per-launch wrapper in their timed loops. The first TF32 warmup call
raises a scope-ending signal before executing any TF32 kernel.
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import csv
import gc
import hashlib
import inspect
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys
import traceback

from vitamin_python_probe import Tee, load_upstream


COMMIT = "520f97093a2abc239977be83d244b327d0e2e2ef"
MODES = ("torch", "sgemm_cublas", "sgemm_tiling", "sgemm_at_tiling",
         "sgemm_at_bcf_swizzling", "sgemm_at_bcf_swizzling_rw",
         "sgemm_at_bcf_swizzling_dbf_rw")


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def save_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class BeforeTf32(RuntimeError):
    """Expected end of the requested SIMT portion; not a completed default main."""


class SimtOnlyLibrary:
    def __init__(self, library):
        self.library = library
        self.stop_calls = 0

    def __getattr__(self, name):
        if name == "sgemm_cublas_tf32":
            return self.stop_before_tf32
        if "tf32" in name:
            raise RuntimeError(f"Unexpected TF32 method requested: {name}")
        # Returning the pybind function itself avoids wrappers in timed loops.
        return getattr(self.library, name)

    def stop_before_tf32(self, *unused_args, **unused_kwargs):
        self.stop_calls += 1
        raise BeforeTf32("Stopped on first sgemm_cublas_tf32 warmup call; TF32 was not executed.")


def read_test_locals(error, code):
    current = error.__traceback__
    found, stack = None, []
    while current is not None:
        frame = current.tb_frame
        stack.append({"file": frame.f_code.co_filename, "function": frame.f_code.co_name,
                      "line": current.tb_lineno})
        if frame.f_code is code:
            found = dict(frame.f_locals)
        current = current.tb_next
    if found is None or not {"a", "b", "c", "c_cublas", "c_my"} <= set(found):
        raise RuntimeError("Expected test_4096 tensor locals were not present in stop traceback.")
    traceback.clear_frames(error.__traceback__)
    return found, stack


def parse_stdout(text, batch):
    rows = []
    for line in text.splitlines():
        match = re.match(r"^(\w+)\s+mean time:\s*([0-9.]+)\s*ms", line)
        if not match:
            continue
        tflops = (re.search(r"tflops:\s*([0-9.]+)", line) or
                  re.search(r"([0-9.]+)\s*tflops", line))
        if not tflops:
            raise ValueError(f"Missing printed TFLOPS: {line}")
        rows.append({"batch": batch, "size": 4096, "implementation": match[1],
                     "printed_mean_ms": float(match[2]), "printed_tflops": float(tflops[1]),
                     "warmup": 10, "repetitions": 200, "seed": 42,
                     "original_diff_check": "REFERENCE" if match[1] == "torch" else
                     "PASS_inferred_from_reaching_stop_after_all_six_checks"})
    if [row["implementation"] for row in rows] != list(MODES):
        raise ValueError("Original stdout did not contain exactly the seven expected entries in original order.")
    if any(not math.isfinite(row["printed_mean_ms"]) or row["printed_mean_ms"] <= 0 for row in rows):
        raise ValueError("Invalid printed wallclock mean.")
    return rows


def summarize(rows):
    grouped = {mode: [row for row in rows if row["implementation"] == mode] for mode in MODES}
    baselines = {row["batch"]: row["printed_mean_ms"] for row in grouped["sgemm_cublas"]}
    baseline_median = statistics.median(baselines.values())
    summaries = []
    for mode in MODES:
        samples = grouped[mode]
        times = [row["printed_mean_ms"] for row in samples]
        median = statistics.median(times)
        summaries.append({
            "implementation": mode, "batches": len(times), "median_batch_mean_ms": median,
            "minimum_batch_mean_ms": min(times), "maximum_batch_mean_ms": max(times),
            "sample_stddev_ms": statistics.stdev(times) if len(times) > 1 else 0,
            "range_over_median_pct": 100 * (max(times) - min(times)) / median,
            "tflops_from_median_printed_ms": 2 * 4096**3 / (median * 1e9),
            "cublas_speedup_ratio_of_medians": baseline_median / median,
            "median_of_paired_cublas_speedups": statistics.median(
                baselines[row["batch"]] / row["printed_mean_ms"] for row in samples),
        })
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vitamin", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batches", type=int, default=7)
    args = parser.parse_args()
    if not 1 <= args.batches <= 100:
        raise ValueError("Require 1..100 batches; the reviewed experiment uses seven.")
    output, vitamin = args.output_dir.resolve(), args.vitamin.resolve()
    if not output.is_dir() or (output / "environment.json").exists():
        raise ValueError("Require a fresh wrapper-created output directory.")
    commit = subprocess.check_output(["git", "-C", str(vitamin), "rev-parse", "HEAD"], text=True).strip()
    if commit != COMMIT:
        raise ValueError(f"Expected reviewed original commit {COMMIT}, got {commit}.")
    source_paths = [
        "kernels/sgemm/test.py", "kernels/sgemm/sgemm.cu", "kernels/sgemm/cublas.cu",
        "kernels/sgemm/sgemm_tf32.cu", "kernels/common/pack.cuh",
    ]
    sources = []
    for relative in source_paths:
        path = vitamin / relative
        original = subprocess.check_output(["git", "-C", str(vitamin), "show", f"HEAD:{relative}"])
        raw = path.read_bytes()
        if raw.replace(b"\r\n", b"\n") != original.replace(b"\r\n", b"\n"):
            raise ValueError(f"Working source differs from the reviewed original: {relative}")
        destination = output / "source-as-run" / "upstream" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(raw)
        sources.append({"path": str(path), "snapshot": str(destination),
                        "sha256": hashlib.sha256(raw).hexdigest()})
    test_path = vitamin / "kernels/sgemm/test.py"
    test_source = test_path.read_text(encoding="utf-8")
    tree = ast.parse(test_source)
    test_node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "test_4096")
    # Validate the unmodified sequential protocol used to infer completed checks.
    checks = []
    stopped_at_tf32 = False
    for statement in test_node.body:
        if not isinstance(statement, ast.Expr) or not isinstance(statement.value, ast.Call):
            continue
        call = statement.value
        name = call.func.id if isinstance(call.func, ast.Name) else None
        prefix = next((keyword.value.value for keyword in call.keywords if keyword.arg == "prefix"
                       and isinstance(keyword.value, ast.Constant)), None)
        if name == "benchmark" and prefix == "sgemm_cublas_tf32":
            stopped_at_tf32 = True
            break
        if name == "diff_check":
            checks.append({"prefix": prefix, "source_line": statement.lineno})
    if not stopped_at_tf32 or [row["prefix"] for row in checks] != list(MODES[1:]):
        raise ValueError("Original sequential SIMT validation protocol changed.")

    import numpy as np
    import torch
    import torch.utils.cpp_extension as extension
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.set_grad_enabled(False)
    if not torch.cuda.is_available():
        raise RuntimeError("Existing PyTorch CUDA runtime is unavailable.")
    save_json(output / "environment.json", {
        "python": sys.executable, "python_version": sys.version, "torch_version": torch.__version__,
        "torch_cuda_build": torch.version.cuda, "device": torch.cuda.get_device_name(0),
        "capability": torch.cuda.get_device_capability(0), "upstream_commit": commit,
        "source_files": sources, "batches": args.batches, "seed": 42,
        "seed_policy": "torch.manual_seed(42) immediately before each unchanged test_4096 call.",
        "timer": "Unmodified upstream benchmark time.perf_counter single-batch mean.",
        "warmup": 10, "repetitions": 200, "order": MODES,
        "original_diff_checks_per_completed_batch": checks,
        "check_count_method": "Inferred from unchanged sequential control flow reaching stop after all six original assertions; no diff_check hook.",
        "scope": "test_4096 SIMT portion only; stopped on first TF32 warmup call. Not a default-main reproduction.",
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "environment": {name: os.environ.get(name) for name in (
            "CUDA_PATH", "TORCH_CUDA_ARCH_LIST", "TORCH_EXTENSIONS_DIR", "NVIDIA_TF32_OVERRIDE",
            "MAX_JOBS", "PYTHONUTF8", "PYTHONIOENCODING", "PYTHONDONTWRITEBYTECODE", "DISTUTILS_USE_SDK")},
        "limits": [
            "Seven repetitions retain the original fixed implementation order; order/thermal bias remains.",
            "Printed means have six decimal places in milliseconds; no new timing calls are inserted.",
            "Post-batch hashes/validation occur only after the original SIMT test has finished.",
            "Earlier intermediate outputs are overwritten by the original function; additional error metrics cover only the final DBF output and cuBLAS output.",
        ],
    })
    module, build_method = load_upstream(test_path, output, extension)
    original_codes = {name: getattr(module, name).__code__ for name in ("test_4096", "benchmark", "diff_check")}
    save_json(output / "original-functions.json", {
        name: {"source_sha256": hashlib.sha256(inspect.getsource(getattr(module, name)).encode()).hexdigest(),
               "file": original_codes[name].co_filename, "first_line": original_codes[name].co_firstlineno}
        for name in original_codes
    })
    loaded = Path(module.lib.__file__).resolve()
    artifacts = []
    for path in (loaded, loaded.parent / "build.ninja"):
        destination = output / ("extension-" + path.name)
        shutil.copy2(path, destination)
        artifacts.append({"loaded_or_build_source": str(path), "snapshot": str(destination),
                          "sha256": hashlib.sha256(destination.read_bytes()).hexdigest()})
    save_json(output / "extension-artifacts.json", artifacts)
    proxy = SimtOnlyLibrary(module.lib)
    module.lib = proxy
    all_rows, batch_records = [], []
    expected_inputs = None
    zero_hash = hashlib.sha256(bytes(4096 * 4096 * 4)).hexdigest()
    for batch in range(1, args.batches + 1):
        for name, code in original_codes.items():
            if getattr(module, name).__code__ is not code:
                raise RuntimeError(f"Original function object was replaced: {name}")
        torch.manual_seed(42)
        capture = io.StringIO()
        print(f"AUTHOR_BATCH_BEGIN {batch}/{args.batches}", flush=True)
        with contextlib.redirect_stdout(Tee(sys.stdout, capture)):
            try:
                module.test_4096()
            except BeforeTf32 as stop:
                captured, stack = read_test_locals(stop, original_codes["test_4096"])
            else:
                raise RuntimeError("Expected TF32 boundary signal was not reached.")
        stdout = capture.getvalue()
        (output / f"batch-{batch:02d}-stdout.txt").write_text(stdout, encoding="utf-8")
        if proxy.stop_calls != batch:
            raise RuntimeError("Unexpected number of intercepted TF32 calls.")
        rows = parse_stdout(stdout, batch)
        # No copies, hashes or extra validation took place inside original timing.
        arrays = {name: captured[name].detach().cpu().numpy() for name in ("a", "b", "c", "c_cublas", "c_my")}
        hashes = {name + "_sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest()
                  for name, array in arrays.items()}
        input_hashes = {"a_sha256": hashes["a_sha256"], "b_sha256": hashes["b_sha256"],
                        "initial_c_sha256": zero_hash}
        if expected_inputs is None:
            expected_inputs = input_hashes
        elif input_hashes != expected_inputs:
            raise RuntimeError("Seeded input hashes differ across batches.")
        reference = arrays["c"].astype(np.float64)
        validations = {}
        for name in ("c_cublas", "c_my"):
            candidate = arrays[name].astype(np.float64)
            finite = np.isfinite(reference) & np.isfinite(candidate)
            delta = np.abs(reference - candidate)
            validations[name] = {
                "implementation": "sgemm_cublas" if name == "c_cublas" else "sgemm_at_bcf_swizzling_dbf_rw",
                "max_abs_error": float(delta.max()) if finite.all() else None,
                "abs_0p01_failure_count": int(np.count_nonzero(~finite | (delta > 0.01))),
                "candidate_nonfinite": int(np.count_nonzero(~np.isfinite(candidate))),
                "reference_nonfinite": int(np.count_nonzero(~np.isfinite(reference))),
            }
            if validations[name]["abs_0p01_failure_count"]:
                raise RuntimeError(f"Post-batch finite/absolute-error validation failed: {name}")
        record = {
            "batch": batch, "seed": 42, "size": 4096, "input_hashes": input_hashes,
            "initial_c_hash_method": "Derived from unchanged test_4096 torch.zeros allocation; initial C has been overwritten in recovered locals.",
            "observed_output_hashes": {key: value for key, value in hashes.items() if key.startswith("c")},
            "completed_original_diff_checks_inferred": len(checks), "stopped_before_tf32": True,
            "traceback_boundary": stack, "post_batch_validation": validations,
        }
        if batch == args.batches:
            path = output / "last-batch-dbf-output.npy"
            np.save(path, arrays["c_my"], allow_pickle=False)
            record["last_output_artifact"] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        save_csv(output / f"batch-{batch:02d}.csv", rows)
        all_rows.extend(rows)
        batch_records.append(record)
        save_csv(output / "wallclock-batches.csv", all_rows)
        save_csv(output / "wallclock-summary.csv", summarize(all_rows))
        save_json(output / "batch-evidence.json", batch_records)
        print(f"AUTHOR_BATCH_PASS batch={batch} original_diff_checks=6 TF32_executed=false", flush=True)
        captured.clear()
        del captured, arrays, reference, candidate, finite, delta
        gc.collect()
    save_json(output / "result.json", {
        "status": "PASS", "batches": args.batches, "rows": len(all_rows), "build_method": build_method,
        "completed_original_diff_checks_inferred": len(checks) * args.batches,
        "original_test_4096_function_body_changed": False,
        "original_benchmark_function_body_changed": False, "original_diff_check_function_body_changed": False,
        "timing_instrumentation_added": False, "original_default_main_executed": False,
        "tf32_kernels_executed": False, "input_hashes_identical_between_batches": True,
    })
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
