# Native Windows Tensor Core experiment

Start with [the Chinese tutorial and measured RTX 4090 D results](../../docs/zh/28-TensorCore初体验与4090D实测.md).

This isolated CMake project reproduces four unmodified vitamin-cuda Tensor Core device implementations, with explicit native input contracts and same-precision cuBLAS comparisons. TF32 and FP16 results are not strict-FP32 replacements.

Build/run/profile through `scripts/windows/*-TensorCore.ps1`. Kernels are attributed in `vendor/provenance.json`, with the original license retained. Only RTX 4090 D was measured; no RTX 3080 results are claimed.
