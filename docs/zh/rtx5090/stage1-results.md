# 阶段 1：SM120 TMA 实测结果

RTX 5090，CUDA 12.8.93，cuBLAS 12.8.4.1。上游源码冻结于 `520f97093a2abc239977be83d244b327d0e2e2ef`，device 代码区域保持原样；派生 host 增加原生单调用 CUDA event 和可复用 descriptor。

主表为 CPU FP32 uniform(-1,1)、seed=42，再转换为对应输入类型；输出同类型，FP32 累加，alpha=1、beta=0。每尺寸独立进程，六实现共享实际输入，每实现 7 轮、每轮 10 预热和 30 正式 event，轮间改变顺序。总计 7560 正式 event。C0 恢复、转换、分配、descriptor 准备、校验位于 event 外。

作者原始 BF16 4096 正态输入、10 预热/200 次墙钟结果单独保存在 `20261022-04-author-bf16-4096`，不可与此表视作相同输入和计时协议。

基线为另增的 cuBLAS `CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION` 对照；每次调用验证数学模式恢复。原默认 cuBLAS 始终保留：BF16 1024 未通过独立精度检查，原始五路失败批次也保存在 `20261022-06-native-bf16-1024`。标记 FAIL 的吞吐不用于精度合格的性能结论。

| 类型 | N | 实现 | TFLOPS | 逐轮配对增益中位数 | 独立精度 |
|---|---:|---|---:|---:|---|
| bf16 | 1024 | hgemm_bcf_dbf_rw | 71.39 | -12.98% | PASS |
| bf16 | 1024 | hgemm_k_stages | 71.39 | -13.42% | PASS |
| bf16 | 1024 | hgemm_tma_r_k_stages_64 | 71.69 | -12.78% | PASS |
| bf16 | 1024 | hgemm_tma_r_k_stages_32 | 67.70 | -17.72% | PASS |
| bf16 | 1024 | hgemm_cublas | 83.62 | +1.46% | FAIL |
| bf16 | 1024 | hgemm_cublas_no_reduced_reduction | 82.24 | +0.00% | PASS |
| bf16 | 2048 | hgemm_bcf_dbf_rw | 167.45 | +1.34% | PASS |
| bf16 | 2048 | hgemm_k_stages | 167.35 | +1.17% | PASS |
| bf16 | 2048 | hgemm_tma_r_k_stages_64 | 160.13 | -3.18% | PASS |
| bf16 | 2048 | hgemm_tma_r_k_stages_32 | 163.46 | -1.07% | PASS |
| bf16 | 2048 | hgemm_cublas | 165.48 | +0.05% | PASS |
| bf16 | 2048 | hgemm_cublas_no_reduced_reduction | 165.19 | +0.00% | PASS |
| bf16 | 4096 | hgemm_bcf_dbf_rw | 203.20 | +11.47% | PASS |
| bf16 | 4096 | hgemm_k_stages | 202.16 | +10.99% | PASS |
| bf16 | 4096 | hgemm_tma_r_k_stages_64 | 192.53 | +5.76% | PASS |
| bf16 | 4096 | hgemm_tma_r_k_stages_32 | 201.09 | +10.46% | PASS |
| bf16 | 4096 | hgemm_cublas | 182.00 | -0.01% | PASS |
| bf16 | 4096 | hgemm_cublas_no_reduced_reduction | 182.06 | +0.00% | PASS |
| fp16 | 1024 | hgemm_bcf_dbf_rw | 71.10 | -31.57% | PASS |
| fp16 | 1024 | hgemm_k_stages | 71.15 | -31.37% | PASS |
| fp16 | 1024 | hgemm_tma_r_k_stages_64 | 71.80 | -30.72% | PASS |
| fp16 | 1024 | hgemm_tma_r_k_stages_32 | 66.98 | -34.93% | PASS |
| fp16 | 1024 | hgemm_cublas | 104.02 | +0.27% | PASS |
| fp16 | 1024 | hgemm_cublas_no_reduced_reduction | 103.90 | +0.00% | PASS |
| fp16 | 2048 | hgemm_bcf_dbf_rw | 167.40 | -0.51% | PASS |
| fp16 | 2048 | hgemm_k_stages | 167.57 | -0.53% | PASS |
| fp16 | 2048 | hgemm_tma_r_k_stages_64 | 160.60 | -4.43% | PASS |
| fp16 | 2048 | hgemm_tma_r_k_stages_32 | 164.18 | -2.42% | PASS |
| fp16 | 2048 | hgemm_cublas | 167.84 | -0.05% | PASS |
| fp16 | 2048 | hgemm_cublas_no_reduced_reduction | 168.33 | +0.00% | PASS |
| fp16 | 4096 | hgemm_bcf_dbf_rw | 202.36 | -6.40% | PASS |
| fp16 | 4096 | hgemm_k_stages | 201.35 | -6.82% | PASS |
| fp16 | 4096 | hgemm_tma_r_k_stages_64 | 192.07 | -11.14% | PASS |
| fp16 | 4096 | hgemm_tma_r_k_stages_32 | 200.15 | -7.39% | PASS |
| fp16 | 4096 | hgemm_cublas | 216.09 | +0.01% | PASS |
| fp16 | 4096 | hgemm_cublas_no_reduced_reduction | 216.09 | +0.00% | PASS |

延迟统计为各轮均值的中位数，再按 2N³/time 换算吞吐；配对增益为各轮基线时间/实现时间−1 的中位数。两种统计不可混用。

正确性同时保留同类型 cuBLAS、实际表示输入的全输出 PEDANTIC、原始 FP32 输入的 PEDANTIC、固定 256 点 CPU double 和原作者路径逐位对照。小形状 128×1024×384 为全输出 CPU double。作者 atol=.016、rtol=.016（相对项为 candidate）与 abs≤.01 各自报告，作者判据通过不代表严格 FP32 通过。详见 stage1-errors.csv。

资源来自本机 occupancy API：实际寄存器、shared 和每 SM 驻留数在 stage1-resources.csv。TMA32/TMA64、DBF、多 stage 在 BK、stage、寄存器和写回上同时存在差异，因此只能称整体实现比较。低频遥测不能证明逐 event 频率恒定。

原生构建 SASS 实际包含 UTMALDG.2D、HMMA.16816、LDGSTS 和等待指令；没有将 TMA 当作 TMEM 或 tcgen05。NCU 计数器受宿主权限限制，静态证据不能证明流水充分重叠。原BF16 K384 三项 Sanitizer 通过，其他检查状态以各自日志为准。

8192 方阵为可选扩展，本阶段未运行；先覆盖已要求的三尺寸与 FP16 对照。

## 已提交的原始数据

- bf16/1024: `results/rtx5090/20261022-08-reduction-bf16-1024`，提交 `6b6086566ec85fd15c711425bfdac4f8cfb58fd7`；events.csv、输入指纹、顺序、正确性、资源、独立分析齐全。
- bf16/2048: `results/rtx5090/20261022-09-reduction-bf16-2048`，提交 `9ae0f7fff847021f6dc21efec4063af1d1b1b971`；events.csv、输入指纹、顺序、正确性、资源、独立分析齐全。
- bf16/4096: `results/rtx5090/20261022-10-reduction-bf16-4096`，提交 `0ebc1549db593d6a7e02bfa91b1bbd7ba589e7d6`；events.csv、输入指纹、顺序、正确性、资源、独立分析齐全。
- fp16/1024: `results/rtx5090/20261022-11-reduction-fp16-1024`，提交 `7b952c77cd9b1e3852a4cab3826f6591f6cfaa00`；events.csv、输入指纹、顺序、正确性、资源、独立分析齐全。
- fp16/2048: `results/rtx5090/20261022-12-reduction-fp16-2048`，提交 `c6fb28f8edd4762f3df18f53dc9ffc444e7097a2`；events.csv、输入指纹、顺序、正确性、资源、独立分析齐全。
- fp16/4096: `results/rtx5090/20261022-13-reduction-fp16-4096`，提交 `f65a4b728b79aa34bd60dc972ff304afc4f70826`；events.csv、输入指纹、顺序、正确性、资源、独立分析齐全。
