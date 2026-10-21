# Stage 3 historical GEMM execution preparation

This directory contains execution preparation, not RTX 5090 results. Source snapshot: `a2b23dabadc79440bf9cbf63f0b566f7eec429ba`. CUDA compilation and GPU execution were not performed by the preparation agent. Frozen source checks normalize only CRLF to LF because Windows checkout bytes differ from Linux Git checkout bytes; actual build-worktree raw SHA256 hashes are recorded separately in every prepare receipt.

`execution-specs.json` contains 80 frozen build configurations in 54 series: K1–12; every archived K15–23 variant; K30 V1–4; all 11 K31 candidates including r220; all three K32 modes; 24 archived K10 tile configurations; padding and known launch-bounds configurations; the CUTLASS SIMT reference; and separately labeled cuBLAS math modes. This is a finite replay menu, not an autotuning search. Historical records without exact parameter evidence remain excluded and listed in the spec. The full 244-row reproduction inventory remains the coverage authority.

## Linux commands

Copy only the seven package files listed below to a fresh preparation directory on the GPU server. Set these paths to actual directories; do not reuse the CPU validation build directory because its generated paths intentionally refer to Windows and a fake CUTLASS fixture.

```bash
PREP=/absolute/path/to/stage3-prep
REPO=/absolute/path/to/SGEMM_CUDA
BUILD=/absolute/path/to/fresh-stage3-build
OUTPUT=/absolute/path/to/fresh-stage3-results
LOCK=/absolute/path/to/shared-gpu.lock

# Read-only menu and portable source preparation:
python3 "$PREP/stage3_runner.py" list --repo "$REPO" --build-root "$BUILD"
python3 "$PREP/stage3_runner.py" prepare --repo "$REPO" --build-root "$BUILD" --series k31

# Compilation only. Records nvcc, full command, ptxas registers/spills and binary SHA256:
python3 "$PREP/stage3_runner.py" build --repo "$REPO" --build-root "$BUILD" --series k31

# One closed series / size / beta batch. Coordinate this lock with all other GPU work:
python3 "$PREP/stage3_runner.py" run --repo "$REPO" --build-root "$BUILD" \
  --series k31 --size 256 --beta 3 --protocol events7 \
  --output "$OUTPUT/k31-n256-b3" --gpu-lock "$LOCK" \
  --timeout 180 --batch-budget 1800
```

Repeat only after the current batch closes and is published by the parent task. The runner does not commit, push, create background jobs, or launch concurrent GPU processes. It never calls the historical autotuners. A batch at 4096 is a separate invocation and output directory. Beta 0, 1 and 3 are separate batches; alpha remains 0.5, so beta 0 is not `C=AB`.

Replace `k31` with `teaching`, `k15` … `k23`, `k30`, `k32`, `cutlass`, `k10-tile-bk8`, `k10-pad8`, etc.; `list` is authoritative. K11 requires N≥256. Frozen K10 tiles require at least max(BM,BN,128). Illegal shapes are explicitly skipped, causing the batch to remain partial. No unsupported shape is launched. K1–3 may be slow; timeout and total batch budget are enforced without discarding the partial batch.

For CUTLASS pass `--cutlass /absolute/path/to/cutlass` to prepare/build. Build requires a clean checkout at v3.5.1 commit `f7b19de32c5d1f3cedfc735c2849f12b537522ee`. The dependency is not downloaded automatically. Its original SIMT Sm80 template is preserved and compiled for sm_120. CUDA NVTX3 is used by a local host header shim; `--nvtx-library` is only for an older toolkit needing legacy NVTX. An exact `nvcc -std=c++17 -O3 -DNDEBUG -arch=sm_120 -lineinfo -Xptxas=-v ... -lcublas` command is written to every `prepare.json`, including all selected compile definitions.

## Protocol and output interpretation

`events7` means 7 rotating rounds, 10 warmups and 30 individually timed calls per implementation. It reuses the original program's event boundaries, C reset outside events, and launch code. Each round/implementation is a separate process, with a separate CUDA context; it is not the same-context unified benchmark required for a new main performance table. Parent task must use its unified host harness for that claim.

`historical-counts` selects archived round/repeat counts. It is not an exact replay of an entire archived script: this runner closes one size/beta at a time, may change the comparison group, adds small-shape smoke, and omits the archived K30/31/32 fair script's K10 4096 thermal prelude (100 repeats / 20 warmups). The original fair script measures N4096/beta3 with 50 repeats/10 warmups, seven rounds for K30/K31 and six for K32. Original correctness is N128/256/512/1024/2048/4096 × beta0/1/3, one repeat/no warmup. K23 V4's seven-round addendum is separate from its three-round earlier variants. Preserve the archived original commands/logs alongside these explicitly derived results.

Host-only instrumentation intercepts successful `cudaEventElapsedTime` calls and copies of A/B/C0 and pre-timing validation outputs. Raw timing samples are buffered in memory and written after execution. Device headers and selected launch function bodies are unchanged and hash checked. For frozen K10 macros, unrelated host launchers are disabled in a generated runner copy to avoid unrelated K18/K19 static assertions. Independent audit copies only relocate quoted includes and append an explicit success return: C++'s implicit return from `main` does not apply after the wrapper renames that function.

Every invocation retains command, stdout/stderr, exit code, time limit status, binary hash, actual A/B/C0 byte fingerprints, raw events and error metrics. Full GPU-reference output checks and CPU-double checks (all elements for N≤256; 256 fixed positions for larger shapes) use the legacy pre-timing output. The original stock strict-reference-only mode has no D2H validation output and is explicitly `NOT_RUN` for that check. This is not post-timing validation, and no sanitizer pass is claimed. CPU validation samples and indexing are written to CSV. NaN/Inf or abs error >0.01 is failure; TF32 is kept separate from strict FP32 claims.

Completed invocations get `capture.events.csv`; incomplete or failed invocation samples retain `.partial.csv`. Batch closure independently rereads the raw event CSVs, checks sample indices/counts and recomputes round means. `summary.csv` reports median round mean, variability, TFLOPS and paired gains only if actual A/B/C0 fingerprints agree across every passed implementation/round. A numerical failure, timeout, spill or unsupported shape is preserved rather than silently omitted. Spills are reported without applying the old architecture's resource filter automatically.

`BATCH_CLOSED.json` and the SHA256 manifest indicate that files are closed and ready for a parent checkpoint. `BATCH_COMPLETE.txt` means every planned group passed with matching inputs; `BATCH_PARTIAL.txt` preserves failures or missing work. A partial batch may be published as partial evidence and must not be reported as completed coverage. The advisory flock only coordinates processes using the same lock path; the parent must also avoid other GPU workloads. Before/after nvidia-smi snapshots do not prove frequency or temperature was stable throughout the run.

## Preparation validation and package

The CPU validation prepares all 80 configurations, verifies 34 source hashes, checks generated-file tamper detection, parses the archived K31 ptxas output including its spill record, and exercises synthetic complete/timeout/numerical-failure/input-mismatch batch closure. It independently checks the resulting manifests and confirms summaries are recomputed from raw CSV rather than supplied means. It never invokes a compiler, nvidia-smi or GPU workload.

```bash
python3 "$PREP/validate_cpu.py" --repo "$REPO" --output /absolute/path/to/fresh-cpu-validation
```

Publish/copy only these seven package files: `README.md`, `event_capture.hpp`, `execution-specs.json`, `make_specs.py`, `stage3_runner.py`, `validate_cpu.py`, `validation-report.json`. Omit `__pycache__`, `prepared-review`, `runtime-validation-*` and their synthetic fixture data. The report labels CUDA compilation and GPU execution `NOT_RUN`; a first real server build and small-shape smoke remains required.
