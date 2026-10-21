# 阶段 2：严格 FP32 核心对照

RTX 5090，五实现同一进程、同一默认 stream、相同实际输入。每个尺寸和数学问题独立测量；每实现 7 轮，每轮 10 次预热和 30 次正式 CUDA event。八批合计 8400 个正式 event。

实现为 K10 PAD4、K10 PAD8、K30 V1、K30 V2、PEDANTIC FP32 cuBLAS。历史表使用 alpha=0.5、beta=3；跨库表使用 alpha=1、beta=0（C=AB），两表不混排。输入使用 seed=20261020 的 xorshift32，实际 A/B/C0 字节指纹随批次保存。

| 协议 | N | 实现 | TFLOPS | 相对 strict（中位延迟比） | GPU参考最大绝对误差 | CPU double最大绝对误差 |
|---|---:|---|---:|---:|---:|---:|
| historical_alpha0.5_beta3 | 1024 | k10 | 19.32 | 0.420 | 0.000671387 | 0.000170363 |
| historical_alpha0.5_beta3 | 1024 | k10_pad8 | 19.05 | 0.414 | 0.000671387 | 0.000170363 |
| historical_alpha0.5_beta3 | 1024 | k30_v1 | 21.38 | 0.465 | 0.000671387 | 0.000170363 |
| historical_alpha0.5_beta3 | 1024 | k30_v2 | 22.24 | 0.484 | 0.000671387 | 0.000170363 |
| historical_alpha0.5_beta3 | 1024 | strict | 45.99 | 1.000 | 0 | 7.48575e-05 |
| historical_alpha0.5_beta3 | 2048 | k10 | 52.42 | 0.760 | 0.00152588 | 0.000388946 |
| historical_alpha0.5_beta3 | 2048 | k10_pad8 | 49.94 | 0.724 | 0.00152588 | 0.000388946 |
| historical_alpha0.5_beta3 | 2048 | k30_v1 | 57.62 | 0.835 | 0.00152588 | 0.000388946 |
| historical_alpha0.5_beta3 | 2048 | k30_v2 | 56.89 | 0.825 | 0.00152588 | 0.000388946 |
| historical_alpha0.5_beta3 | 2048 | strict | 68.99 | 1.000 | 0 | 0.000162357 |
| historical_alpha0.5_beta3 | 3072 | k10 | 56.33 | 0.782 | 0.00244141 | 0.000792168 |
| historical_alpha0.5_beta3 | 3072 | k10_pad8 | 54.92 | 0.762 | 0.00244141 | 0.000792168 |
| historical_alpha0.5_beta3 | 3072 | k30_v1 | 61.62 | 0.855 | 0.00244141 | 0.000792168 |
| historical_alpha0.5_beta3 | 3072 | k30_v2 | 64.29 | 0.892 | 0.00244141 | 0.000792168 |
| historical_alpha0.5_beta3 | 3072 | strict | 72.06 | 1.000 | 0 | 0.000311433 |
| historical_alpha0.5_beta3 | 4096 | k10 | 53.73 | 0.777 | 0.00305176 | 0.000887716 |
| historical_alpha0.5_beta3 | 4096 | k10_pad8 | 52.69 | 0.762 | 0.00305176 | 0.000887716 |
| historical_alpha0.5_beta3 | 4096 | k30_v1 | 57.85 | 0.837 | 0.00305176 | 0.000887716 |
| historical_alpha0.5_beta3 | 4096 | k30_v2 | 59.82 | 0.865 | 0.00305176 | 0.000887716 |
| historical_alpha0.5_beta3 | 4096 | strict | 69.12 | 1.000 | 0 | 0.000420665 |
| C_equals_AB_alpha1_beta0 | 1024 | k10 | 19.31 | 0.418 | 0.00134277 | 0.000333573 |
| C_equals_AB_alpha1_beta0 | 1024 | k10_pad8 | 19.06 | 0.413 | 0.00134277 | 0.000333573 |
| C_equals_AB_alpha1_beta0 | 1024 | k30_v1 | 21.84 | 0.473 | 0.00134277 | 0.000333573 |
| C_equals_AB_alpha1_beta0 | 1024 | k30_v2 | 22.77 | 0.493 | 0.00134277 | 0.000333573 |
| C_equals_AB_alpha1_beta0 | 1024 | strict | 46.18 | 1.000 | 0 | 0.000147275 |
| C_equals_AB_alpha1_beta0 | 2048 | k10 | 52.44 | 0.755 | 0.00305176 | 0.000768355 |
| C_equals_AB_alpha1_beta0 | 2048 | k10_pad8 | 49.95 | 0.720 | 0.00305176 | 0.000768355 |
| C_equals_AB_alpha1_beta0 | 2048 | k30_v1 | 58.16 | 0.838 | 0.00305176 | 0.000768355 |
| C_equals_AB_alpha1_beta0 | 2048 | k30_v2 | 57.42 | 0.827 | 0.00305176 | 0.000768355 |
| C_equals_AB_alpha1_beta0 | 2048 | strict | 69.41 | 1.000 | 0 | 0.000345695 |
| C_equals_AB_alpha1_beta0 | 3072 | k10 | 56.30 | 0.775 | 0.00488281 | 0.00157702 |
| C_equals_AB_alpha1_beta0 | 3072 | k10_pad8 | 54.85 | 0.755 | 0.00488281 | 0.00157702 |
| C_equals_AB_alpha1_beta0 | 3072 | k30_v1 | 61.89 | 0.852 | 0.00488281 | 0.00157702 |
| C_equals_AB_alpha1_beta0 | 3072 | k30_v2 | 64.47 | 0.888 | 0.00488281 | 0.00157702 |
| C_equals_AB_alpha1_beta0 | 3072 | strict | 72.62 | 1.000 | 0 | 0.000618098 |
| C_equals_AB_alpha1_beta0 | 4096 | k10 | 53.95 | 0.775 | 0.00610352 | 0.00183265 |
| C_equals_AB_alpha1_beta0 | 4096 | k10_pad8 | 52.70 | 0.757 | 0.00610352 | 0.00183265 |
| C_equals_AB_alpha1_beta0 | 4096 | k30_v1 | 58.52 | 0.841 | 0.00610352 | 0.00183265 |
| C_equals_AB_alpha1_beta0 | 4096 | k30_v2 | 59.89 | 0.861 | 0.00610352 | 0.00183265 |
| C_equals_AB_alpha1_beta0 | 4096 | strict | 69.58 | 1.000 | 0 | 0.000841329 |

统计为每轮 event 均值的跨轮中位数；逐轮配对提升另存于每批 analysis/paired-summary.csv。C0 的 D2D 恢复在 start event 前。每轮末输出做全量 PEDANTIC GPU 参考检查，大尺寸另用固定 256 点 CPU double 直接检查候选输出。所有八批数值验收通过。

## 已核验提交

- historical_alpha0.5_beta3 / N=1024: `results/rtx5090/20261022-17-strict-historical-1024`，`9c382a88e41b49f656db5ccedbbbcd1b998511a8`。
- historical_alpha0.5_beta3 / N=2048: `results/rtx5090/20261022-19-strict-historical-2048`，`f39aa2b0a31fbcd84f7b0d938a111fbccafcd1be`。
- historical_alpha0.5_beta3 / N=3072: `results/rtx5090/20261022-21-strict-historical-3072`，`4eff92a470842f541a02e6b81e2a5a16ff1bac51`。
- historical_alpha0.5_beta3 / N=4096: `results/rtx5090/20261022-23-strict-historical-4096`，`b824a7a01c1a1cf5eed8a5666c50dc419a41eaea`。
- C_equals_AB_alpha1_beta0 / N=1024: `results/rtx5090/20261022-18-strict-ab-1024`，`911615b97fa51e7769012fa1036dc756e09a6bff`。
- C_equals_AB_alpha1_beta0 / N=2048: `results/rtx5090/20261022-20-strict-ab-2048`，`e72009415723bf716e3d276d297768ec6f004def`。
- C_equals_AB_alpha1_beta0 / N=3072: `results/rtx5090/20261022-22-strict-ab-3072`，`1808d6c8f7512641480f175284977ccdc2a7c937`。
- C_equals_AB_alpha1_beta0 / N=4096: `results/rtx5090/20261022-24-strict-ab-4096`，`6f5a8a4a66ceeb10a379ec68cfea079dbe070699`。
