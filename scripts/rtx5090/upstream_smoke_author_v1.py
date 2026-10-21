#!/usr/bin/env python3
"""Audit the unmodified vitamin SM120 binding; no performance claims from smoke.

Modes smoke and author preserve device source. The latter calls the actual
test_4096 function and wraps only logging around its original benchmark/check.
Use a fresh output directory; exceptions leave PARTIAL.json and nonzero exit.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import pathlib
import statistics
import sys
import time
import traceback


def sha(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--upstream", required=True, help="kernels/hgemm_sm120 directory")
    p.add_argument("--output", required=True)
    p.add_argument("--mode", choices=("smoke", "author", "events"), default="smoke")
    p.add_argument("--dtype", choices=("bf16", "fp16", "both"), default="bf16")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--k", type=int, default=128, help="smoke K: >=128 and divisible by64")
    p.add_argument("--size", type=int, default=1024, help="one complete square event batch, multiple1024")
    p.add_argument("--cpu-threads", type=int, default=4)
    args = p.parse_args()
    out = pathlib.Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    upstream = pathlib.Path(args.upstream).resolve()
    import torch
    torch.set_grad_enabled(False)
    torch.set_num_threads(args.cpu_threads)
    meta = {"mode": args.mode, "seed": args.seed, "torch": torch.__version__,
            "cuda": torch.version.cuda, "source_files": {n: sha(upstream / n) for n in ("test.py", "hgemm_sm120.cu", "cublas.cu")},
            "wrapper_sha256": sha(__file__), "complete": False,
            "input_generator": "torch CPU randn FP32, then CPU cast to BF16/FP16, then CUDA copy",
            "alpha": 1.0, "beta": 0.0, "input_layout": "row-major contiguous",
            "author_criterion": "abs(ref-candidate)<=0.016+0.016*abs(candidate), torch.allclose(ref,candidate)",
            "strict_diagnostic": "abs(ref-candidate)<=0.01 (separate, never relabeled as low-precision acceptance)"}
    checks = []
    inputs = []

    def save(name, data):
        (out / name).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")

    def fingerprint(t):
        x = t.detach().cpu().contiguous()
        return {"shape": list(x.shape), "stride": list(t.stride()), "dtype": str(x.dtype),
                "sha256": hashlib.sha256(x.view(torch.uint8).numpy().tobytes()).hexdigest()}

    def metrics(reference, candidate, kernel, reference_name):
        ref = reference.detach().cpu().double()
        got = candidate.detach().cpu().double()
        diff = (ref-got).abs()
        finite = torch.isfinite(ref) & torch.isfinite(got)
        safe_diff = diff[finite]
        author_bad = ~finite | (diff > (0.016 + 0.016*got.abs()))
        strict_bad = ~finite | (diff > 0.01)
        rel = safe_diff / ref[finite].abs().clamp_min(1e-30)
        result = {"kernel": kernel, "reference": reference_name, "elements": ref.numel(),
                  "max_abs": float(safe_diff.max()) if safe_diff.numel() else None,
                  "max_rel_ref_denominator_floor_1e_30": float(rel.max()) if rel.numel() else None,
                  "relative_l2": float(torch.linalg.vector_norm(safe_diff) / torch.linalg.vector_norm(ref[finite]).clamp_min(1e-30)),
                  "nonfinite_elements": int((~finite).sum()), "candidate_nan": int(torch.isnan(got).sum()),
                  "candidate_inf": int(torch.isinf(got).sum()), "author_failed_elements": int(author_bad.sum()),
                  "strict_abs_001_failed_elements": int(strict_bad.sum()),
                  "author_allclose": bool(torch.allclose(reference.cpu(), candidate.cpu(), atol=.016, rtol=.016)),
                  "strict_abs_001_pass": bool(not strict_bad.any()),
                  "exact_equal_elements": int((ref == got).sum())}
        checks.append(result)
        save("correctness.json", checks)
        print(json.dumps(result), flush=True)
        return result

    save("PARTIAL.json", meta)
    try:
        spec = importlib.util.spec_from_file_location("upstream_sm120_test", upstream / "test.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        if args.mode == "smoke":
            if args.k < 128 or args.k % 64:
                raise ValueError("common smoke requires K>=128 divisible by64")
            names = ("hgemm_bcf_dbf_rw", "hgemm_k_stages", "hgemm_tma_r_k_stages_64", "hgemm_tma_r_k_stages_32", "hgemm_cublas")
            types = [torch.bfloat16, torch.float16] if args.dtype == "both" else [torch.bfloat16 if args.dtype == "bf16" else torch.float16]
            all_pass = True
            for dtype in types:
                torch.manual_seed(args.seed)
                a_cpu = torch.randn(128, args.k).to(dtype)
                b_cpu = torch.randn(args.k, 1024).to(dtype)
                c0 = torch.zeros(128, 1024, dtype=dtype)
                a, b, c = a_cpu.cuda(), b_cpu.cuda(), c0.cuda()
                inputs.append({"A": fingerprint(a), "B": fingerprint(b), "C0": fingerprint(c0)})
                save("inputs.json", inputs)
                # Full, actual represented inputs, final low-precision output rounding.
                reference = (a_cpu.double() @ b_cpu.double()).to(dtype)
                libref = torch.zeros_like(c)
                mod.lib.hgemm_cublas(a, b, libref)
                torch.cuda.synchronize()
                for name in names:
                    c.copy_(c0)
                    getattr(mod.lib, name)(a, b, c)
                    torch.cuda.synchronize()
                    r = metrics(reference, c, f"{name}/{dtype}", "full_CPU_double_actual_inputs_then_same_output_rounding")
                    metrics(libref, c, f"{name}/{dtype}", "upstream_same_dtype_cublas_COMPUTE_32F")
                    all_pass &= r["author_allclose"] and r["nonfinite_elements"] == 0
            meta["acceptance_pass"] = bool(all_pass)
        elif args.mode == "author":
            if args.dtype != "bf16":
                raise ValueError("unmodified test_4096 is BF16-only; use --dtype bf16")
            torch.manual_seed(args.seed)
            original_benchmark = mod.benchmark
            original_check = mod.diff_check
            seen = False

            def logged_benchmark(op, a, b, c=None, warmup=10, rep=200, prefix="torch"):
                nonlocal seen
                if not seen:
                    zero_c0 = torch.zeros(a.shape[0], b.shape[1], dtype=a.dtype)
                    inputs.append({"A": fingerprint(a), "B": fingerprint(b), "C0": fingerprint(zero_c0)})
                    save("inputs.json", inputs)
                    seen = True
                return original_benchmark(op, a, b, c, warmup=warmup, rep=rep, prefix=prefix)

            def logged_check(a, b, prefix="torch", eps=.016):
                metrics(a, b, prefix, "upstream_torch_matmul_same_dtype")
                return original_check(a, b, prefix=prefix, eps=eps)

            mod.benchmark, mod.diff_check = logged_benchmark, logged_check
            meta["protocol"] = "actual unmodified test_4096; fixed original order; 10 warmup,200 repeated host calls timed by perf_counter + synchronize; includes host wrapper/descriptor overhead; external manual_seed recorded"
            mod.test_4096()
            meta["acceptance_pass"] = True
        else:
            if args.size < 1024 or args.size % 1024 or args.dtype == "both":
                raise ValueError("one events batch requires size multiple1024 and one dtype")
            dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
            n = args.size
            torch.manual_seed(args.seed)
            # Same upstream input generator, actual bytes shared across all implementations.
            a_cpu = torch.randn(n, n).to(dtype)
            b_cpu = torch.randn(n, n).to(dtype)
            c0_cpu = torch.zeros(n, n, dtype=dtype)
            a, b, c0 = a_cpu.cuda(), b_cpu.cuda(), c0_cpu.cuda()
            c = torch.empty_like(c0)
            libref = torch.empty_like(c0)
            mod.lib.hgemm_cublas(a, b, libref)
            torch.cuda.synchronize()
            inputs.append({"A": fingerprint(a), "B": fingerprint(b), "C0": fingerprint(c0)})
            save("inputs.json", inputs)
            # Fixed sample is shared by every implementation, full K reduction in CPU FP64.
            sample_gen = torch.Generator().manual_seed(20261022)
            sample_flat = torch.randperm(n*n, generator=sample_gen)[:256]
            sr, sc = sample_flat // n, sample_flat % n
            sample_ref = (a_cpu[sr, :].double() * b_cpu[:, sc].T.double()).sum(1).to(dtype)
            save("cpu_sample_coordinates.json", list(zip(sr.tolist(), sc.tolist())))
            names = ["hgemm_bcf_dbf_rw", "hgemm_k_stages", "hgemm_tma_r_k_stages_64", "hgemm_tma_r_k_stages_32", "hgemm_cublas"]
            orders = []
            for r in range(7):
                order = names[r % len(names):] + names[:r % len(names)]
                if r % 2:
                    order = list(reversed(order))
                orders.append(order)
            save("order.json", orders)
            meta["protocol"] = "API-dispatch CUDA-event;7 rounds;10 warmups/implementation/round;30 individually timed calls;one stream;C0 copy before every start event;host TMA encoding/dispatch may contribute GPU idle;NOT cached-descriptor kernel-only"
            meta["shape"] = [n,n,n]
            meta["rounds"] = 7
            meta["warmups"] = 10
            meta["repeats"] = 30
            meta["expected_events"] = 7*30*len(names)
            save("PARTIAL.json", meta)
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            records = []
            all_pass = True
            with (out / "events.csv").open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["round","order_position","implementation","sample","milliseconds","M","N","K","seed","dtype"])
                writer.writeheader()
                for r, order in enumerate(orders):
                    for pos, name in enumerate(order):
                        op = getattr(mod.lib, name)
                        for _ in range(10):
                            c.copy_(c0)
                            op(a,b,c)
                        torch.cuda.synchronize()
                        for sample in range(30):
                            c.copy_(c0)
                            start.record()
                            op(a,b,c)
                            end.record()
                            end.synchronize()
                            record = dict(round=r,order_position=pos,implementation=name,sample=sample,milliseconds=start.elapsed_time(end),M=n,N=n,K=n,seed=args.seed,dtype=args.dtype)
                            writer.writerow(record)
                            records.append(record)
                        f.flush()
                        rlib = metrics(libref, c, f"round{r}/{name}/{dtype}", "full_upstream_same_dtype_cublas_COMPUTE_32F")
                        rsample = metrics(sample_ref, c[sr.cuda(),sc.cuda()], f"round{r}/{name}/{dtype}", "fixed256_CPU_double_actual_inputs_then_same_output_rounding")
                        all_pass &= rlib["author_allclose"] and rsample["author_allclose"]
            # Independent readback from the closed CSV, retaining every sample.
            with (out / "events.csv").open(newline="",encoding="utf-8") as f:
                readback = list(csv.DictReader(f))
            if len(readback) != meta["expected_events"]:
                raise RuntimeError("event count mismatch")
            means = {name:[statistics.mean(float(x["milliseconds"]) for x in readback if x["implementation"]==name and int(x["round"])==r) for r in range(7)] for name in names}
            summary = []
            for name in names:
                median = statistics.median(means[name])
                summary.append({"implementation":name,"round_mean_ms":means[name],"median_round_mean_ms":median,
                                "tflops_from_median_round_mean":2*n*n*n/(median*1e9),
                                "median_paired_speedup_vs_cublas":statistics.median(means["hgemm_cublas"][r]/means[name][r]-1 for r in range(7))})
            save("summary.json", summary)
            meta["actual_events"] = len(readback)
            meta["events_sha256"] = sha(out / "events.csv")
            meta["acceptance_pass"] = bool(all_pass)
        meta["complete"] = True
        save("COMPLETED.json", meta)
        # PARTIAL remains an intentional start record; COMPLETED is authoritative.
        if not meta["acceptance_pass"]:
            return 2
        return 0
    except Exception:
        meta["exception"] = traceback.format_exc()
        save("PARTIAL.json", meta)
        raise


if __name__ == "__main__":
    sys.exit(main())

