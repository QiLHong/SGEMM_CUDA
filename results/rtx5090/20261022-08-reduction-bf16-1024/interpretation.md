# Controlled BF16 1024 result

1260 events complete. All four custom kernels and the cuBLAS comparator with DISALLOW_REDUCED_PRECISION_REDUCTION pass unchanged independent accuracy criteria. Original default cuBLAS fails and remains present, so aggregate exit status is 2. The controlled flag removes the observed accuracy failure in this case; no threshold or device kernel was changed.

Use reduction-analysis/summary.json for accuracy-qualified comparisons against the reduction-controlled comparator. independent-summary.json is the generic raw timing closure audit and labels its original default-cuBLAS denominator explicitly; it is not the accepted precision baseline.
