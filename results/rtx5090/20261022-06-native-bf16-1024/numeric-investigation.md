# BF16 1024 numerical failure retained

All 1050 CUDA event measurements completed and were independently recomputed. This is a closed measurement batch with failed numerical acceptance, not a valid accuracy-qualified cuBLAS performance comparison.

All four custom kernels pass full represented-input FP32 PEDANTIC and sampled CPU double checks. The original default cuBLAS output fails these checks. The four custom outputs differ from default cuBLAS at 7408 elements (maximum absolute difference 0.25). All 35 comparisons with the original unchanged bindings are bitwise identical. Host extraction therefore reproduced original behavior.

Hypothesis to test next: default cuBLAS permits reduced precision intermediate reduction. Add a separate comparator with CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION and restore the handle math mode. Preserve the default comparator, all errors and unchanged tolerances. No cause is considered confirmed by this batch alone.
