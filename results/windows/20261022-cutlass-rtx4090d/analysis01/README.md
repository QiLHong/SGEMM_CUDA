# RTX 4090 D CUTLASS 对照统计

正式数据：68,040 个 events，两个精度组，6 个独立进程。

每组合先计算每轮全部 30 events 的平均延迟，再取七轮均值的中位数。未删除慢样本；不混入 profiler、smoke 或旧成绩。

下表为三个 batch × 三个 seed 的九个组合：TFLOPS 中位数（最小～最大）。这只是描述性汇总，组合嵌套在同一进程中，不是九次完全独立试验。

## TF32 输入 / FP32 累加

| N | vitamin_tf32_dbf | tf32_bn64_bk32 | cutlass_tf32_128x128 | cutlass_tf32_128x64_bk16 | cublas_tf32 |
|---:|---:|---:|---:|---:|---:|
| 1024 | 31.408 (31.348～33.569) | 32.647 (32.413～34.972) | 35.525 (35.283～37.989) | 35.851 (35.644～38.388) | 35.487 (35.051～37.478) |
| 2048 | 48.480 (48.447～48.535) | 57.939 (57.886～58.032) | 51.932 (51.819～51.975) | 62.946 (62.710～63.075) | 51.776 (51.707～51.862) |
| 3072 | 54.823 (54.257～55.323) | 59.447 (58.077～59.744) | 58.000 (57.693～59.017) | 64.203 (63.080～64.281) | 58.799 (58.231～59.217) |
| 4096 | 69.723 (68.757～70.174) | 69.300 (68.734～69.789) | 73.978 (73.342～74.513) | 74.780 (74.257～75.073) | 67.287 (66.558～67.554) |

| N | CUTLASS 配置 | 参考 | 九组合配对提升中位数范围 | 正收益组合 | 正收益轮 |
|---:|---|---|---:|---:|---:|
| 1024 | cutlass_tf32_128x128 | vitamin_tf32_dbf | +12.477%～+13.465% | 9/9 | 62/63 |
| 1024 | cutlass_tf32_128x128 | tf32_bn64_bk32 | +8.289%～+8.844% | 9/9 | 62/63 |
| 1024 | cutlass_tf32_128x128 | cublas_tf32 | -0.679%～+3.006% | 7/9 | 39/63 |
| 1024 | cutlass_tf32_128x64_bk16 | vitamin_tf32_dbf | +13.613%～+14.861% | 9/9 | 63/63 |
| 1024 | cutlass_tf32_128x64_bk16 | tf32_bn64_bk32 | +9.142%～+10.534% | 9/9 | 63/63 |
| 1024 | cutlass_tf32_128x64_bk16 | cublas_tf32 | +0.120%～+2.508% | 9/9 | 58/63 |
| 2048 | cutlass_tf32_128x128 | vitamin_tf32_dbf | +6.969%～+7.267% | 9/9 | 63/63 |
| 2048 | cutlass_tf32_128x128 | tf32_bn64_bk32 | -10.565%～-10.224% | 0/9 | 0/63 |
| 2048 | cutlass_tf32_128x128 | cublas_tf32 | +0.048%～+0.362% | 9/9 | 50/63 |
| 2048 | cutlass_tf32_128x64_bk16 | vitamin_tf32_dbf | +29.649%～+30.070% | 9/9 | 63/63 |
| 2048 | cutlass_tf32_128x64_bk16 | tf32_bn64_bk32 | +8.095%～+8.803% | 9/9 | 63/63 |
| 2048 | cutlass_tf32_128x64_bk16 | cublas_tf32 | +21.307%～+21.762% | 9/9 | 63/63 |
| 3072 | cutlass_tf32_128x128 | vitamin_tf32_dbf | +5.267%～+8.175% | 9/9 | 63/63 |
| 3072 | cutlass_tf32_128x128 | tf32_bn64_bk32 | -2.533%～+0.750% | 1/9 | 9/63 |
| 3072 | cutlass_tf32_128x128 | cublas_tf32 | -2.205%～+0.316% | 1/9 | 18/63 |
| 3072 | cutlass_tf32_128x64_bk16 | vitamin_tf32_dbf | +15.616%～+17.694% | 9/9 | 63/63 |
| 3072 | cutlass_tf32_128x64_bk16 | tf32_bn64_bk32 | +7.327%～+9.157% | 9/9 | 62/63 |
| 3072 | cutlass_tf32_128x64_bk16 | cublas_tf32 | +7.963%～+9.413% | 9/9 | 63/63 |
| 4096 | cutlass_tf32_128x128 | vitamin_tf32_dbf | +5.339%～+7.438% | 9/9 | 61/63 |
| 4096 | cutlass_tf32_128x128 | tf32_bn64_bk32 | +6.371%～+8.333% | 9/9 | 63/63 |
| 4096 | cutlass_tf32_128x128 | cublas_tf32 | +8.811%～+11.530% | 9/9 | 63/63 |
| 4096 | cutlass_tf32_128x64_bk16 | vitamin_tf32_dbf | +6.615%～+8.585% | 9/9 | 62/63 |
| 4096 | cutlass_tf32_128x64_bk16 | tf32_bn64_bk32 | +6.810%～+8.725% | 9/9 | 63/63 |
| 4096 | cutlass_tf32_128x64_bk16 | cublas_tf32 | +10.274%～+11.714% | 9/9 | 63/63 |

各误差标准独立判定（任一 seed/batch/check 失败即该格为 FAIL）：TF32 这里采用同一量化输入的 GPU 参考及 CPU double；strict FP32 采用原始输入。CPU 仅抽样 256 点，GPU 参考为全量输出。

| N | 实现 | 最大绝对误差 | abs≤0.01 | SIMT allclose | 本精度 native allclose | NaN/Inf 检查 |
|---:|---|---:|---|---|---|---|
| 1024 | vitamin_tf32_dbf | 0.003845215 | PASS | FAIL | PASS | PASS |
| 1024 | tf32_bn64_bk32 | 0.003845215 | PASS | FAIL | PASS | PASS |
| 1024 | cutlass_tf32_128x128 | 0.003845215 | PASS | FAIL | PASS | PASS |
| 1024 | cutlass_tf32_128x64_bk16 | 0.003845215 | PASS | FAIL | PASS | PASS |
| 1024 | cublas_tf32 | 0.003845215 | PASS | FAIL | PASS | PASS |
| 2048 | vitamin_tf32_dbf | 0.01293945 | FAIL | FAIL | PASS | PASS |
| 2048 | tf32_bn64_bk32 | 0.01293945 | FAIL | FAIL | PASS | PASS |
| 2048 | cutlass_tf32_128x128 | 0.01293945 | FAIL | FAIL | PASS | PASS |
| 2048 | cutlass_tf32_128x64_bk16 | 0.01293945 | FAIL | FAIL | PASS | PASS |
| 2048 | cublas_tf32 | 0.01293945 | FAIL | FAIL | PASS | PASS |
| 3072 | vitamin_tf32_dbf | 0.0213623 | FAIL | FAIL | PASS | PASS |
| 3072 | tf32_bn64_bk32 | 0.0213623 | FAIL | FAIL | PASS | PASS |
| 3072 | cutlass_tf32_128x128 | 0.0213623 | FAIL | FAIL | PASS | PASS |
| 3072 | cutlass_tf32_128x64_bk16 | 0.0213623 | FAIL | FAIL | PASS | PASS |
| 3072 | cublas_tf32 | 0.0213623 | FAIL | FAIL | PASS | PASS |
| 4096 | vitamin_tf32_dbf | 0.03076172 | FAIL | FAIL | PASS | PASS |
| 4096 | tf32_bn64_bk32 | 0.03076172 | FAIL | FAIL | PASS | PASS |
| 4096 | cutlass_tf32_128x128 | 0.03076172 | FAIL | FAIL | PASS | PASS |
| 4096 | cutlass_tf32_128x64_bk16 | 0.03076172 | FAIL | FAIL | PASS | PASS |
| 4096 | cublas_tf32 | 0.03076172 | FAIL | FAIL | PASS | PASS |

TF32 对原始 FP32 的精度损失另表，不与上面的同输入正确性混淆：

| N | 实现 | 最大绝对误差 | abs≤0.01 | SIMT allclose |
|---:|---|---:|---|---|
| 1024 | vitamin_tf32_dbf | 0.3443909 | FAIL | FAIL |
| 1024 | tf32_bn64_bk32 | 0.3443909 | FAIL | FAIL |
| 1024 | cutlass_tf32_128x128 | 0.3443909 | FAIL | FAIL |
| 1024 | cutlass_tf32_128x64_bk16 | 0.3443909 | FAIL | FAIL |
| 1024 | cublas_tf32 | 0.3443909 | FAIL | FAIL |
| 2048 | vitamin_tf32_dbf | 0.5679932 | FAIL | FAIL |
| 2048 | tf32_bn64_bk32 | 0.5679932 | FAIL | FAIL |
| 2048 | cutlass_tf32_128x128 | 0.5679932 | FAIL | FAIL |
| 2048 | cutlass_tf32_128x64_bk16 | 0.5679932 | FAIL | FAIL |
| 2048 | cublas_tf32 | 0.5679932 | FAIL | FAIL |
| 3072 | vitamin_tf32_dbf | 0.6939697 | FAIL | FAIL |
| 3072 | tf32_bn64_bk32 | 0.6939697 | FAIL | FAIL |
| 3072 | cutlass_tf32_128x128 | 0.6939697 | FAIL | FAIL |
| 3072 | cutlass_tf32_128x64_bk16 | 0.6939697 | FAIL | FAIL |
| 3072 | cublas_tf32 | 0.6939697 | FAIL | FAIL |
| 4096 | vitamin_tf32_dbf | 0.7731018 | FAIL | FAIL |
| 4096 | tf32_bn64_bk32 | 0.7731018 | FAIL | FAIL |
| 4096 | cutlass_tf32_128x128 | 0.7731018 | FAIL | FAIL |
| 4096 | cutlass_tf32_128x64_bk16 | 0.7731018 | FAIL | FAIL |
| 4096 | cublas_tf32 | 0.7731018 | FAIL | FAIL |

## Strict FP32 SIMT

| N | k10_fp32 | k30_fp32 | cutlass_simt_256x128 | cublas_strict_fp32 |
|---:|---:|---:|---:|---:|
| 1024 | 21.428 (20.204～21.711) | 23.189 (21.965～23.598) | 14.093 (13.600～14.614) | 30.675 (29.775～32.299) |
| 2048 | 28.725 (28.453～29.456) | 30.801 (30.157～31.353) | 27.350 (27.246～27.599) | 44.506 (44.369～44.668) |
| 3072 | 41.454 (41.274～41.974) | 41.687 (41.228～42.269) | 37.241 (36.800～37.697) | 45.730 (45.357～46.344) |
| 4096 | 44.277 (43.610～44.818) | 45.446 (45.180～45.851) | 43.897 (43.448～44.191) | 48.128 (47.884～48.472) |

| N | CUTLASS 配置 | 参考 | 九组合配对提升中位数范围 | 正收益组合 | 正收益轮 |
|---:|---|---|---:|---:|---:|
| 1024 | cutlass_simt_256x128 | k10_fp32 | -32.873%～-32.271% | 0/9 | 0/63 |
| 1024 | cutlass_simt_256x128 | k30_fp32 | -38.163%～-37.505% | 0/9 | 0/63 |
| 1024 | cutlass_simt_256x128 | cublas_strict_fp32 | -55.100%～-53.517% | 0/9 | 0/63 |
| 2048 | cutlass_simt_256x128 | k10_fp32 | -5.937%～-4.078% | 0/9 | 2/63 |
| 2048 | cutlass_simt_256x128 | k30_fp32 | -12.055%～-9.258% | 0/9 | 0/63 |
| 2048 | cutlass_simt_256x128 | cublas_strict_fp32 | -38.695%～-37.285% | 0/9 | 0/63 |
| 3072 | cutlass_simt_256x128 | k10_fp32 | -12.482%～-8.796% | 0/9 | 0/63 |
| 3072 | cutlass_simt_256x128 | k30_fp32 | -11.212%～-9.591% | 0/9 | 0/63 |
| 3072 | cutlass_simt_256x128 | cublas_strict_fp32 | -19.063%～-18.092% | 0/9 | 0/63 |
| 4096 | cutlass_simt_256x128 | k10_fp32 | -1.769%～+1.520% | 3/9 | 22/63 |
| 4096 | cutlass_simt_256x128 | k30_fp32 | -4.814%～-2.726% | 0/9 | 1/63 |
| 4096 | cutlass_simt_256x128 | cublas_strict_fp32 | -9.732%～-7.914% | 0/9 | 0/63 |

各误差标准独立判定（任一 seed/batch/check 失败即该格为 FAIL）：TF32 这里采用同一量化输入的 GPU 参考及 CPU double；strict FP32 采用原始输入。CPU 仅抽样 256 点，GPU 参考为全量输出。

| N | 实现 | 最大绝对误差 | abs≤0.01 | SIMT allclose | 本精度 native allclose | NaN/Inf 检查 |
|---:|---|---:|---|---|---|---|
| 1024 | k10_fp32 | 0.001312256 | PASS | PASS | PASS | PASS |
| 1024 | k30_fp32 | 0.001312256 | PASS | PASS | PASS | PASS |
| 1024 | cutlass_simt_256x128 | 0.001312256 | PASS | PASS | PASS | PASS |
| 1024 | cublas_strict_fp32 | 0.001312256 | PASS | PASS | PASS | PASS |
| 2048 | k10_fp32 | 0.003417969 | PASS | PASS | PASS | PASS |
| 2048 | k30_fp32 | 0.003417969 | PASS | PASS | PASS | PASS |
| 2048 | cutlass_simt_256x128 | 0.003417969 | PASS | PASS | PASS | PASS |
| 2048 | cublas_strict_fp32 | 0.003417969 | PASS | PASS | PASS | PASS |
| 3072 | k10_fp32 | 0.005615234 | PASS | FAIL | FAIL | PASS |
| 3072 | k30_fp32 | 0.005615234 | PASS | FAIL | FAIL | PASS |
| 3072 | cutlass_simt_256x128 | 0.005615234 | PASS | FAIL | FAIL | PASS |
| 3072 | cublas_strict_fp32 | 0.005615234 | PASS | FAIL | FAIL | PASS |
| 4096 | k10_fp32 | 0.002008449 | PASS | PASS | PASS | PASS |
| 4096 | k30_fp32 | 0.002008449 | PASS | PASS | PASS | PASS |
| 4096 | cutlass_simt_256x128 | 0.002008449 | PASS | PASS | PASS | PASS |
| 4096 | cublas_strict_fp32 | 0.002008449 | PASS | PASS | PASS | PASS |

配对提升按每轮 `reference_ms / candidate_ms - 1` 计算，和两个中位数之比不同；两种数值均保留在 CSV。

TF32 相对原始 FP32 的误差失败保留在 errors.csv，不能解释为严格 FP32 成绩。CUTLASS 仅测预先选定模板，未搜索整个库。

DATA_AUDIT_PASS 表示原始记录、计划和统计完整性通过，不表示所有数值阈值都通过。SIMT allclose 的 FAIL 保持原阈值和失败数量，未删除失败检查，也未将其改成 PASS。
